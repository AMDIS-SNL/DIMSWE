"""Taylor tests of the Hessian-vector products of Lagrangian_ODEConstrainedOptimization.

Each supported optimization type ('coeffs', 'ics', 'coeffs+ics', 'ic-params') is checked with
- a full-space Taylor test of H v against gradient differences, g(x + eps v) = g(x) + eps H v +
  O(eps^2), run twice to check repeatability,
- a symmetry test u . (H v) = v . (H u),
and the gradient returned by the combined HVP sweep is checked against jac. These use a loss that
compares at two observation steps, so intermediate second-order observation sources are covered.

The remaining tests mirror the gradient tests (test_timestepping_*_gradients.py,
test_optimize_gradients.py) in reduced, scaled coordinates through ReducedProblem.hessp: the
final-state L2 loss for coefficients and initial states over 1 and 5 steps, a weighted
trajectory loss, analytic initial-condition parameters with scaling and a fixed parameter, and
positive curvature at the truth.
"""

import numpy as np
import pytest

from dimswe.optimize import ParametricInitialCondition

from tests.gradient_helpers import (
    assert_repeatable,
    assert_taylor_convergence,
    flat,
    hessian_taylor_test,
    perturbed_coeffs,
    perturbed_initial_state,
    reduced_hessian_taylor_test,
    setup_problem,
)

NSTEPS = 3
OBS_STEPS = [1, 3]


def state_direction(x0, seed):
    """Return a seeded random initial-state direction scaled like the state."""
    generator = np.random.default_rng(seed)
    return 0.01 * (np.abs(x0) + 1e-3 * np.max(np.abs(x0))) * generator.standard_normal(x0.size)


def build_case(setup, opt_type):
    """Return (full_jac, full_hessp, x, (u, v), eps_list) for an optimization type.

    x is a point away from the truth (where the gradient would vanish), and u, v are two
    independent seeded directions in the full parameter space of opt_type.
    """
    optimizer = setup.optimizer
    coeffs = flat(setup.coeff)
    x0 = flat(setup.x0[0])
    generator = np.random.default_rng(10)
    eps_list = None
    if opt_type == 'coeffs':
        x = coeffs * 1.1
        # the default eps list (starting at 1e-2) is used: the loss is strongly nonlinear in the
        # hyperviscosity exponent, so from eps = 1e-1 the gradient-difference remainders are still
        # pre-asymptotic (rates 1.5-1.9) even though they converge to 2
        directions = [0.1 * coeffs * generator.standard_normal(coeffs.size) for _ in range(2)]
        problem = optimizer.reduced_problem(x, opt_type='coeffs')
        hessp = optimizer.full_hessp('coeffs')
    elif opt_type == 'ics':
        x = x0 * 1.01
        directions = [state_direction(x0, 11), state_direction(x0, 12)]
        problem = optimizer.reduced_problem(x, opt_type='ics', coeffs0=setup.coeff)
        hessp = optimizer.full_hessp('ics', coeffs0=setup.coeff)
    elif opt_type == 'coeffs+ics':
        x = np.hstack([coeffs * 1.1, x0 * 1.01])
        directions = [np.hstack([0.01 * coeffs * generator.standard_normal(coeffs.size), state_direction(x0, seed)]) for seed in (13, 14)]
        problem = optimizer.reduced_problem(x, opt_type='coeffs+ics')
        hessp = optimizer.full_hessp('coeffs+ics')
    elif opt_type == 'ic-params':
        ic_map = ParametricInitialCondition(setup.model, setup.parameters, ['ox', 'oy'])
        x = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
        directions = [np.array([0.1, 0.05]), np.array([-0.03, 0.1])]
        problem = optimizer.reduced_problem(x, opt_type='ic-params', coeffs0=setup.coeff, ic_map=ic_map)
        hessp = optimizer.full_hessp('ic-params', coeffs0=setup.coeff, ic_map=ic_map)
    else:
        raise ValueError(opt_type)
    return problem.full_jac, hessp, x, directions, eps_list


@pytest.mark.parametrize('opt_type', ['coeffs', 'ics', 'coeffs+ics', 'ic-params'])
def test_hessp_taylor(opt_type):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    full_jac, full_hessp, x, directions, eps_list = build_case(setup, opt_type)
    # run twice: identical remainders and rates show no state leaks between HVP evaluations
    _, remainders, rates, change = assert_repeatable(
        lambda: hessian_taylor_test(full_jac, full_hessp, x, directions[0], eps_list=eps_list))
    print(opt_type, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


@pytest.mark.parametrize('opt_type', ['coeffs', 'ics', 'coeffs+ics', 'ic-params'])
def test_hessp_symmetric(opt_type):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    _, full_hessp, x, (u, v), _ = build_case(setup, opt_type)
    u_hv = float(np.dot(u, full_hessp(x, v)))
    v_hu = float(np.dot(v, full_hessp(x, u)))
    print(opt_type, u_hv, v_hu)
    assert u_hv != 0
    assert np.isclose(u_hv, v_hu, rtol=1e-6, atol=0.0)


def test_hvp_sweep_gradient_matches_jac():
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    optimizer = setup.optimizer
    coeffs = flat(setup.coeff) * 1.1
    ics = (flat(setup.x0[0]) * 1.01)[None, :]
    expected = optimizer.jac(coeffs, ics)
    grad_coeffs, grad_ics, _, _ = optimizer._hvp_sweep(coeffs, ics, 0.1 * coeffs, state_direction(ics[0], 15)[None, :])
    result = np.hstack([grad_coeffs, np.ravel(grad_ics)])
    assert np.allclose(result, expected, rtol=1e-10, atol=1e-10 * np.max(np.abs(expected)))


def test_hessp_is_linear_in_direction():
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    full_jac, full_hessp, x, (u, v), _ = build_case(setup, 'coeffs+ics')
    combined = full_hessp(x, 2.0 * u - 0.5 * v)
    separate = 2.0 * full_hessp(x, u) - 0.5 * full_hessp(x, v)
    assert np.allclose(combined, separate, rtol=1e-8, atol=1e-10 * np.max(np.abs(separate)))


def test_reduced_hessp_taylor_with_scaling_and_free_mask():
    """Reduced-space HVP of a real model problem, with scaling and a partially fixed parameter set."""
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    coeffs = flat(setup.coeff)
    x0 = flat(setup.x0[0])
    x = np.hstack([coeffs * 1.1, x0 * 1.01])
    scaling = np.hstack([np.abs(coeffs), np.abs(x0) + 1.0])
    free_mask = np.ones(x.size, dtype=bool)
    free_mask[0] = False  # hold the first coefficient fixed
    problem = setup.optimizer.reduced_problem(x, opt_type='coeffs+ics', scaling_factors=scaling, free_mask=free_mask)
    assert problem.has_hessp()

    full_direction = np.hstack([0.01 * coeffs, state_direction(x0, 16)])
    p = problem.to_reduced(full_direction)
    # hessian_taylor_test perturbs its point linearly, which in reduced coordinates is z0 + eps*p
    _, remainders, rates, change = hessian_taylor_test(problem.jac, problem.hessp, problem.z0, p)
    print(remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)
    assert problem.nhvp == 1


# perturbations away from the truth in tests/tswe_small.cfg (c0=0.07, s=3.2, ox=oy=0.1), as in the
# gradient tests
PERTURBED_COEFFS = {'hyperviscosity': {'c0': 0.05, 's': 2.8}}
PERTURBED_IC = {'ox': 0.11, 'oy': 0.11}


@pytest.mark.parametrize('nsteps', [1, 5])
def test_final_state_coeff_hessp_taylor(nsteps):
    setup = setup_problem(nsteps)
    coeffs = perturbed_coeffs(setup, PERTURBED_COEFFS)
    # reduced direction of size 0.1 in variables scaled by |coeffs|; with eps <= 1e-2 this keeps
    # the coefficient perturbation in the asymptotic regime (see test_hessp_taylor[coeffs])
    direction = 0.1 * np.random.default_rng(20).standard_normal(coeffs.size)
    make_problem = lambda: setup.optimizer.reduced_problem(coeffs, opt_type='coeffs', scaling_factors=np.abs(coeffs))
    _, remainders, rates, change = assert_repeatable(lambda: reduced_hessian_taylor_test(make_problem, direction))
    print(nsteps, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


@pytest.mark.parametrize('nsteps', [1, 5])
def test_final_state_ic_hessp_taylor(nsteps):
    setup = setup_problem(nsteps)
    x0_new = perturbed_initial_state(setup, PERTURBED_IC)
    make_problem = lambda: setup.optimizer.reduced_problem(x0_new, opt_type='ics', coeffs0=setup.coeff)
    _, remainders, rates, change = assert_repeatable(lambda: reduced_hessian_taylor_test(make_problem, 0.05 * x0_new))
    print(nsteps, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


def test_weighted_trajectory_ic_hessp_taylor():
    setup = setup_problem(3, obs_steps=[1, 3], var_weights={'v': 1.0, 'h': 1.0e-4, 'S': 1.0e-6})
    x0 = flat(setup.x0[0])
    direction = np.random.default_rng(21).standard_normal(x0.size)
    make_problem = lambda: setup.optimizer.reduced_problem(x0 * 1.01, opt_type='ics', coeffs0=setup.coeff,
        scaling_factors=np.abs(x0) + 1.0)
    _, remainders, rates, change = assert_repeatable(lambda: reduced_hessian_taylor_test(make_problem, direction))
    print(remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


@pytest.mark.parametrize('free_mask, direction', [
    (None, [1.0, 0.5]),
    ([True, False], [1.0]),   # oy fixed: only the ox row and column of the Hessian are used
])
def test_ic_params_reduced_hessp_taylor(free_mask, direction):
    setup = setup_problem(3, obs_steps=[3])
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ['ox', 'oy'])
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
    make_problem = lambda: setup.optimizer.reduced_problem(params, opt_type='ic-params', coeffs0=setup.coeff,
        ic_map=ic_map, scaling_factors=np.array([0.1, 0.1]), free_mask=free_mask)
    _, remainders, rates, change = assert_repeatable(lambda: reduced_hessian_taylor_test(make_problem, direction))
    print(free_mask, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


@pytest.mark.parametrize('opt_type', ['coeffs', 'ics'])
def test_hessp_at_truth_has_positive_curvature(opt_type):
    """At the truth the misfit vanishes, so H = J^T W J (Gauss-Newton) and v.Hv = ||J v||_W^2 > 0."""
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    if opt_type == 'coeffs':
        x = flat(setup.coeff)
        problem = setup.optimizer.reduced_problem(x, opt_type='coeffs')
        direction = 0.1 * x
    else:
        x = flat(setup.x0[0])
        problem = setup.optimizer.reduced_problem(x, opt_type='ics', coeffs0=setup.coeff)
        direction = state_direction(x, 22)
    assert np.all(problem.jac(problem.z0) == 0)
    curvature = float(np.dot(direction, problem.hessp(problem.z0, direction)))
    print(opt_type, curvature)
    assert curvature > 0
