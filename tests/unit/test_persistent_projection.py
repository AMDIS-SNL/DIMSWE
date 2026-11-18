"""Tests for the persistent projections of the initial-condition map (dimswe/assembly.py PersistentProjector).

ParametricInitialCondition used to call Function.project for every component of the initial
state, its tangents and second tangents on every evaluation, which built a new solver and LU
factorization each time. It now keeps one PersistentProjector per projection: a mass matrix
factored once per space (shared through the model's mass-solver cache) and a persistently
assembled right-hand side. These tests check, on tests/tswe_small.cfg, that the projections equal
Firedrake's project at several parameter points (so the persistent right-hand side sees the new
parameter values), that the projectors are built once and use the shared mass path, and that
iterative solver parameters keep Firedrake's project.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from tests.gradient_helpers import flat, setup_problem  # noqa: E402

PARAM_NAMES = ["ox", "oy"]
# both paths solve the same mass system with a direct solver, so they agree to round-off
RTOL = 1e-10


@pytest.fixture(scope="module")
def setup():
    return setup_problem(3, obs_steps=[1, 3])


def projected_with_firedrake(ic_map):
    """Return (x0, tangents, second tangents) at the map's current point, with Function.project."""
    def project(func, sub, exprs):
        func.assign(0)
        for var, expr in exprs.items():
            if expr is not None:
                sub[var].project(expr, solver_parameters=ic_map.projection_parameters)
        return flat(func).copy()

    x0 = project(ic_map.x0, ic_map.x0_sub, ic_map.exprs)
    tangents = [project(func, sub, exprs) for (func, sub, _), exprs in zip(ic_map.tangent_vars, ic_map.tangent_exprs)]
    second, second_sub, _ = ic_map._second_tangent_var
    nparams = len(ic_map.param_names)
    seconds = {}
    for j in range(nparams):
        for k in range(j, nparams):
            exprs = ic_map._second_tangent_exprs[j][k]
            if any(expr is not None for expr in exprs.values()):
                seconds[j, k] = project(second, second_sub, exprs)
    return x0, tangents, seconds


def assert_close(actual, expected, label):
    """Assert agreement to RTOL relative to the largest entry of expected."""
    scale = np.max(np.abs(expected), initial=0.0)
    assert np.max(np.abs(actual - expected), initial=0.0) <= RTOL * scale + 1e-300, label


def test_ic_map_projections_match_firedrake_project_at_several_points(setup):
    from dimswe.optimize import ParametricInitialCondition

    ic_map = ParametricInitialCondition(setup.model, setup.parameters, PARAM_NAMES, cache=False)
    rng = np.random.default_rng(31)
    base = ic_map.get_initial_values(setup.parameters)
    nprojectors = None
    # four points: the first two validate the persistent assemblers, the later ones use them alone
    for point in range(4):
        params = base + 0.01 * rng.standard_normal(base.size)
        x0 = ic_map.forward(params)
        tangents = ic_map.tangents()
        second = ic_map.second_tangents()
        expected_x0, expected_tangents, expected_second = projected_with_firedrake(ic_map)
        assert_close(x0, expected_x0, ("x0", point))
        for j, (tangent, expected) in enumerate(zip(tangents, expected_tangents)):
            assert_close(tangent, expected, ("tangent", j, point))
        for (j, k), expected in expected_second.items():
            assert_close(second[j][k], expected, ("second tangent", j, k, point))
        # one projector per projection, built on the first evaluation and reused afterwards
        if nprojectors is None:
            nprojectors = len(ic_map._projectors)
        assert len(ic_map._projectors) == nprojectors
    fallbacks = {key: projector.fallback_reason for key, projector in ic_map._projectors.items()
                 if projector.strategy != 'mass'}
    assert not fallbacks, fallbacks
    # the factored mass matrices are shared: at most one per component space of x
    mass_solvers = {id(projector.mass_solver) for projector in ic_map._projectors.values()}
    assert len(mass_solvers) <= len(ic_map.x_var_list)
    print("projectors", nprojectors, "mass solvers", len(mass_solvers),
          "methods", sorted({projector.mass_solver.method for projector in ic_map._projectors.values()}))


def test_iterative_projection_parameters_keep_firedrake_project(setup):
    from dimswe.assembly import PersistentProjector
    from dimswe.optimize import ParametricInitialCondition

    ic_map = ParametricInitialCondition(setup.model, setup.parameters, PARAM_NAMES, cache=False)
    ic_map.set_param_values(ic_map.get_initial_values(setup.parameters))
    var, expr = next(iter(ic_map.exprs.items()))
    target = ic_map.x0_sub[var]
    projector = PersistentProjector(target, expr, {}, {'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_rtol': 1e-13})
    assert projector.strategy == 'project' and projector.mass_solver is None
    iterative = flat(projector.project()).copy()
    target.project(expr, solver_parameters=ic_map.projection_parameters)
    # a converged CG solve, not the direct one: close, not equal to round-off
    scale = np.max(np.abs(flat(target)))
    assert np.max(np.abs(iterative - flat(target))) <= 1e-8 * scale
