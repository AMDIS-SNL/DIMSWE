"""Tests for stage reuse within a LieSplit step (plan item A3, docs/hvp_performance_plan.md).

The LieSplit adjoint and second-order adjoint steps recompute the substep states, which solves
every substep's stages. With reuse_stages (the default) those stage values, and for HVPs the
tangent stages, are stored per substep and restored in the substep adjoint steps instead of being
solved a second time. The stored values are exactly what the re-solve would produce, so gradients
and Hessian-vector products must agree with the recompute path (reuse_stages false) to
round-off, for coefficient and initial-state directions together. The instrumented stage passes
must drop from 3 to 2 per substep. All of this is on the optimizer checkpoint 'states', since
the default 'stages' does not recompute steps at all.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from tests.gradient_helpers import flat, setup_problem  # noqa: E402
from workflows.profile_hvp import (  # noqa: E402
    COST_MODEL_STAGE_PASSES,
    COST_MODEL_STAGE_PASSES_RECOMPUTE,
    profiling_module,
    stage_passes,
    timestepper_layout,
)

NSTEPS = 3
OBS_STEPS = [1, 3]
RTOL = 1e-10


@pytest.fixture(scope="module")
def setup():
    """The tswe_small problem on the 'states' checkpoint, the path whose steps are recomputed.

    The default checkpoint 'stages' never recomputes a step, so stage reuse within a step only
    applies to 'states' (tests/unit/test_stage_checkpoint.py covers 'stages').
    """
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    setup.optimizer.checkpoint = "states"
    setup.optimizer.clear_checkpoint()
    return setup


def point_and_direction(setup):
    """Return a coefficient + initial-state point and direction away from the truth (seeded)."""
    rng = np.random.default_rng(40)
    coeffs = flat(setup.coeff) * 1.02
    ics = (flat(setup.x0[0]) * 1.01)[None, :]
    coeffs_dir = 0.1 * flat(setup.coeff) * rng.standard_normal(coeffs.size)
    ics_dir = 0.01 * (np.abs(ics) + 1.0) * rng.standard_normal(ics.shape)
    return coeffs, ics, coeffs_dir, ics_dir


def evaluate(setup, reuse):
    """Return (gradient, HVP) with stage reuse switched on or off."""
    setup.timestepper.reuse_stages = reuse
    coeffs, ics, coeffs_dir, ics_dir = point_and_direction(setup)
    try:
        gradient = setup.optimizer.jac(coeffs, ics)
        hvp = setup.optimizer.hessp(coeffs, ics, coeffs_dir_arr=coeffs_dir, ics_dir_arr=ics_dir)
    finally:
        setup.timestepper.reuse_stages = True
    return np.asarray(gradient), np.asarray(hvp)


def test_lie_split_reuses_stages_by_default(setup):
    assert setup.parameters["timestepping"]["method"] == "LieSplit"
    assert setup.timestepper.reuse_stages is True


def test_reused_stages_give_the_recomputed_gradient_and_hvp(setup):
    gradient_reuse, hvp_reuse = evaluate(setup, True)
    gradient_recompute, hvp_recompute = evaluate(setup, False)
    scale = lambda arr: np.max(np.abs(arr))
    print("gradient diff", scale(gradient_reuse - gradient_recompute) / scale(gradient_recompute))
    print("hvp diff", scale(hvp_reuse - hvp_recompute) / scale(hvp_recompute))
    assert scale(gradient_reuse - gradient_recompute) <= RTOL * scale(gradient_recompute)
    assert scale(hvp_reuse - hvp_recompute) <= RTOL * scale(hvp_recompute)
    # the gradient returned alongside the HVP (by _hvp_sweep) also matches jac with reuse
    coeffs, ics, coeffs_dir, ics_dir = point_and_direction(setup)
    grad_coeffs, grad_ics, _, _ = setup.optimizer._hvp_sweep(coeffs, ics, coeffs_dir, ics_dir)
    combined = np.hstack([grad_coeffs, np.ravel(grad_ics)])
    assert scale(combined - gradient_reuse) <= RTOL * scale(gradient_reuse)


@pytest.mark.parametrize("reuse", [True, False])
def test_stage_passes_per_substep_follow_the_cost_model(setup, reuse):
    profiling = profiling_module()
    setup.timestepper.reuse_stages = reuse
    nsubsteps, recompute = timestepper_layout(setup.timestepper)
    assert recompute is (not reuse)
    coeffs, ics, coeffs_dir, ics_dir = point_and_direction(setup)
    was_enabled = profiling.is_enabled()
    profiling.enable()
    try:
        for kind, call in (("jac", lambda: setup.optimizer.jac(coeffs, ics)),
                           ("hessp", lambda: setup.optimizer.hessp(coeffs, ics, coeffs_dir, ics_dir))):
            profiling.reset()
            with profiling.region(kind + "-1"):
                call()
            record = {"label": kind + "-1", "kind": kind, "regions": profiling.snapshot()}
            passes = stage_passes(record, NSTEPS, nsubsteps, recompute_stages=recompute)
            model = (COST_MODEL_STAGE_PASSES_RECOMPUTE if recompute else COST_MODEL_STAGE_PASSES)[kind]
            for stage_kind, entry in passes.items():
                assert entry["measured"] == model.get(stage_kind, 0), (reuse, kind, stage_kind, entry)
    finally:
        profiling.enable(was_enabled)
        profiling.reset()
        setup.timestepper.reuse_stages = True
