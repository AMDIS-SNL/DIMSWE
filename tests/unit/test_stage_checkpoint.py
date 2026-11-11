"""Tests for the trajectory stage checkpoint (optimizer checkpoint 'stages', plan items A1/A2).

With checkpoint 'stages', jac stores every step's stage values in its forward pass, restores them
in the adjoint sweep instead of recomputing the step, and stores the stage adjoints and lambdas.
An HVP at the point of the last jac then restores all first-order work and only solves the
tangent and second-order adjoint stages; at any other point it recomputes as with 'states'.

The stored values are exactly what the recomputation produces, so gradients and HVPs must agree
with checkpoint 'states' to round-off, for coefficient and initial-state directions together,
for the LieSplit timestepper of tests/tswe_small.cfg and for a single RK timestepper. The stage
passes per substep must follow COST_MODEL_STAGE_PASSES_CHECKPOINT.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from tests.gradient_helpers import SOLVER_PARAMETERS, flat, setup_problem  # noqa: E402
from workflows.profile_hvp import (  # noqa: E402
    COST_MODEL_STAGE_PASSES_CHECKPOINT,
    profiling_module,
    stage_passes,
    timestepper_layout,
)

NSTEPS = 3
OBS_STEPS = [1, 3]
RTOL = 1e-10


@pytest.fixture(scope="module")
def setup():
    return setup_problem(NSTEPS, obs_steps=OBS_STEPS)


@pytest.fixture(scope="module")
def rk_optimizer(setup):
    """An optimizer on the same objective with a single RK4 timestepper (the model's dynamics terms)."""
    from dimswe.optimize import Lagrangian_ODEConstrainedOptimization
    from dimswe.timestepping import RK4

    terms = setup.parameters["timestepping"]["termlist"][0]
    timestepper = RK4(setup.model, setup.logger, SOLVER_PARAMETERS, terms=terms)
    timestepper.set_coeff(setup.coeff)
    optimizer = Lagrangian_ODEConstrainedOptimization(setup.model, timestepper, setup.objective, setup.dt)
    optimizer.verbose = False
    return optimizer


def point_and_direction(setup, seed=50, shift=1.0):
    """Return a coefficient + initial-state point and direction away from the truth (seeded)."""
    rng = np.random.default_rng(seed)
    coeffs = flat(setup.coeff) * (1.0 + 0.02 * shift)
    ics = (flat(setup.x0[0]) * (1.0 + 0.01 * shift))[None, :]
    coeffs_dir = 0.1 * flat(setup.coeff) * rng.standard_normal(coeffs.size)
    ics_dir = 0.01 * (np.abs(ics) + 1.0) * rng.standard_normal(ics.shape)
    return coeffs, ics, coeffs_dir, ics_dir


def set_mode(optimizer, mode):
    """Switch an optimizer's checkpoint mode, dropping any stored checkpoint."""
    optimizer.checkpoint = mode
    optimizer.clear_checkpoint()


def relative_difference(a, b):
    """Return max|a - b| / max|b|."""
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


def evaluate(optimizer, mode, point, directions):
    """Return (gradient, [HVP per direction], HVP-sweep gradient) at point in one checkpoint mode."""
    coeffs, ics = point
    set_mode(optimizer, mode)
    try:
        gradient = np.asarray(optimizer.jac(coeffs, ics))
        hvps = [np.asarray(optimizer.hessp(coeffs, ics, coeffs_dir_arr=cd, ics_dir_arr=icd)) for cd, icd in directions]
        grad_coeffs, grad_ics, _, _ = optimizer._hvp_sweep(coeffs, ics, *directions[0])
        sweep_gradient = np.hstack([grad_coeffs, np.ravel(grad_ics)])
    finally:
        set_mode(optimizer, "states")
    return gradient, hvps, sweep_gradient


def test_unknown_checkpoint_modes_are_rejected(setup):
    from dimswe.optimize import Lagrangian_ODEConstrainedOptimization

    with pytest.raises(ValueError, match="checkpoint"):
        Lagrangian_ODEConstrainedOptimization(setup.model, setup.timestepper, setup.objective, setup.dt,
                                              checkpoint="everything")


def test_stages_is_the_default_when_the_timestepper_supports_it(setup):
    from dimswe.optimize import Lagrangian_ODEConstrainedOptimization

    assert setup.timestepper.supports_stage_checkpoint is True
    optimizer = Lagrangian_ODEConstrainedOptimization(setup.model, setup.timestepper, setup.objective, setup.dt)
    assert optimizer.checkpoint == "stages"
    explicit = Lagrangian_ODEConstrainedOptimization(setup.model, setup.timestepper, setup.objective, setup.dt,
                                                     checkpoint="states")
    assert explicit.checkpoint == "states"


def test_resolve_checkpoint_falls_back_to_states_without_support():
    from types import SimpleNamespace

    from dimswe.optimize import resolve_checkpoint

    supported = SimpleNamespace(supports_stage_checkpoint=True)
    unsupported = SimpleNamespace(supports_stage_checkpoint=False)
    assert resolve_checkpoint(None, supported) == "stages"
    assert resolve_checkpoint(None, unsupported) == "states"
    # a timestepper that predates the attribute (for example an implicit one) counts as unsupported
    assert resolve_checkpoint(None, SimpleNamespace()) == "states"
    assert resolve_checkpoint("states", supported) == "states"
    with pytest.raises(ValueError, match="explicit RK"):
        resolve_checkpoint("stages", unsupported)
    with pytest.raises(ValueError, match="checkpoint must be one of"):
        resolve_checkpoint("everything", supported)


@pytest.mark.parametrize("which", ["lie_split", "rk4"])
def test_checkpointed_gradient_and_hvps_equal_the_recomputed_ones(setup, rk_optimizer, which):
    optimizer = setup.optimizer if which == "lie_split" else rk_optimizer
    coeffs, ics, coeffs_dir, ics_dir = point_and_direction(setup)
    rng = np.random.default_rng(51)
    second = (coeffs_dir * rng.standard_normal(coeffs_dir.size), ics_dir * rng.standard_normal(ics_dir.shape))
    directions = [(coeffs_dir, ics_dir), second]
    gradient_s, hvps_s, sweep_s = evaluate(optimizer, "stages", (coeffs, ics), directions)
    gradient_r, hvps_r, sweep_r = evaluate(optimizer, "states", (coeffs, ics), directions)
    print(which, "gradient diff", relative_difference(gradient_s, gradient_r))
    assert relative_difference(gradient_s, gradient_r) <= RTOL
    for hvp_s, hvp_r in zip(hvps_s, hvps_r):
        print(which, "hvp diff", relative_difference(hvp_s, hvp_r))
        assert relative_difference(hvp_s, hvp_r) <= RTOL
    # the gradient returned by a checkpointed HVP sweep is jac's stored one
    assert relative_difference(sweep_s, gradient_s) <= RTOL
    assert relative_difference(sweep_r, gradient_r) <= RTOL


def test_hvps_away_from_the_last_gradient_point_recompute(setup):
    optimizer = setup.optimizer
    coeffs, ics, coeffs_dir, ics_dir = point_and_direction(setup)
    other_coeffs, other_ics, _, _ = point_and_direction(setup, shift=2.0)
    set_mode(optimizer, "states")
    expected = np.asarray(optimizer.hessp(other_coeffs, other_ics, coeffs_dir, ics_dir))
    set_mode(optimizer, "stages")
    try:
        optimizer.jac(coeffs, ics)
        assert optimizer._checkpoint_matches(optimizer._point_key(coeffs, ics, None))
        assert not optimizer._checkpoint_matches(optimizer._point_key(other_coeffs, other_ics, None))
        # an objective evaluation elsewhere (a line search) does not disturb the checkpoint
        optimizer.obj(other_coeffs, other_ics)
        at_other = np.asarray(optimizer.hessp(other_coeffs, other_ics, coeffs_dir, ics_dir))
        assert relative_difference(at_other, expected) <= RTOL
        # the checkpoint of the jac point is still valid and used afterwards
        set_mode(optimizer, "states")
        reference = np.asarray(optimizer.hessp(coeffs, ics, coeffs_dir, ics_dir))
        set_mode(optimizer, "stages")
        optimizer.jac(coeffs, ics)
        optimizer.obj(other_coeffs, other_ics)
        at_point = np.asarray(optimizer.hessp(coeffs, ics, coeffs_dir, ics_dir))
        assert relative_difference(at_point, reference) <= RTOL
    finally:
        set_mode(optimizer, "states")


def test_ic_params_problem_with_stage_checkpoint_matches_states(setup):
    from dimswe.optimize import ParametricInitialCondition

    optimizer = setup.optimizer
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ["ox", "oy"])
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
    problem = optimizer.reduced_problem(params, opt_type="ic-params", coeffs0=setup.coeff, ic_map=ic_map,
                                        scaling_factors=np.array([0.1, 0.1]))
    x = problem.to_full(problem.z0)
    directions = [np.array([1.0, 0.5]), np.array([-0.3, 2.0])]
    results = {}
    for mode in ("stages", "states"):
        set_mode(optimizer, mode)
        try:
            gradient = problem.full_jac(x)
            results[mode] = (gradient, [problem.full_hessp(x, d) for d in directions])
        finally:
            set_mode(optimizer, "states")
    assert relative_difference(results["stages"][0], results["states"][0]) <= RTOL
    for hvp_s, hvp_r in zip(results["stages"][1], results["states"][1]):
        assert relative_difference(hvp_s, hvp_r) <= RTOL


def test_stage_passes_per_substep_follow_the_checkpoint_cost_model(setup):
    profiling = profiling_module()
    optimizer = setup.optimizer
    nsubsteps, _ = timestepper_layout(setup.timestepper)
    coeffs, ics, coeffs_dir, ics_dir = point_and_direction(setup)
    was_enabled = profiling.is_enabled()
    set_mode(optimizer, "stages")
    profiling.enable()
    try:
        calls = (("jac", lambda: optimizer.jac(coeffs, ics)),
                 ("hessp", lambda: optimizer.hessp(coeffs, ics, coeffs_dir, ics_dir)))
        for kind, call in calls:
            profiling.reset()
            with profiling.region(kind + "-1"):
                call()
            record = {"label": kind + "-1", "kind": kind, "regions": profiling.snapshot()}
            passes = stage_passes(record, NSTEPS, nsubsteps, checkpoint_stages=True)
            for stage_kind, entry in passes.items():
                assert entry["measured"] == COST_MODEL_STAGE_PASSES_CHECKPOINT[kind].get(stage_kind, 0), (kind, stage_kind, entry)
    finally:
        profiling.enable(was_enabled)
        profiling.reset()
        set_mode(optimizer, "states")
