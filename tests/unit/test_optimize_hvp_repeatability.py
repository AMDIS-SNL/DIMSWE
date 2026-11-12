"""Repeatability tests for Hessian-vector-product evaluation.

The first-order path already depends on timestepper.reset_internal_vars() for repeatable results,
and the HVP adds more internal state: tangent and second-order stage variables, stored tangent
trajectories, and the coefficient direction theta_dot set on the timestepper. These tests check
that none of it leaks between evaluations:
- the same HVP is identical before and after unrelated objective, gradient and HVP evaluations,
- an HVP does not change later objective or gradient values,
- the coefficient direction of one HVP does not leak into a later HVP without one,
- the coefficient direction is cleared even when an HVP fails part way,
- repeated ReducedProblem.hessp calls agree and are each counted.
(The Taylor tests in test_optimize_hvp.py additionally run every HVP Taylor test twice and require
identical remainders and rates.)
"""

import numpy as np
import pytest

from tests.gradient_helpers import flat, setup_problem

NSTEPS = 3
OBS_STEPS = [1, 3]
# with the direct solvers of tests/gradient_helpers.SOLVER_PARAMETERS, repeated evaluations should
# agree to round-off; this allows for summation-order differences only
RTOL = 1e-12


def assert_same(first, second):
    """Assert two arrays agree to round-off, relative to the larger entries."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))), 1e-300)
    assert np.allclose(first, second, rtol=RTOL, atol=RTOL * scale), f'max difference {np.max(np.abs(first - second))} (scale {scale})'


def coefficient_directions(integrator):
    """Return the flattened coefficient directions of a timestepper and its sub-integrators."""
    integrators = list(getattr(integrator, 'time_integrators', [integrator]))
    return [flat(sub.coeff_dir) for sub in integrators if sub.coeff_dir is not None]


def coeffs_plus_ics_case(setup, seed):
    """Return (x, v) for opt_type 'coeffs+ics' away from the truth, with a seeded direction."""
    generator = np.random.default_rng(seed)
    coeffs = flat(setup.coeff)
    x0 = flat(setup.x0[0])
    x = np.hstack([coeffs * 1.1, x0 * 1.01])
    v = np.hstack([
        0.01 * coeffs * generator.standard_normal(coeffs.size),
        0.01 * (np.abs(x0) + 1e-3 * np.max(np.abs(x0))) * generator.standard_normal(x0.size),
    ])
    return x, v


def test_hessp_unaffected_by_interleaved_evaluations():
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    optimizer = setup.optimizer
    ncoeff = setup.model.get_coeff_size()
    hessp = optimizer.full_hessp('coeffs+ics')
    x, v = coeffs_plus_ics_case(setup, seed=0)
    x_other, u = coeffs_plus_ics_case(setup, seed=1)
    x_other = x_other * 1.02

    first = hessp(x, v)
    # unrelated evaluations at another point and in another direction
    optimizer.obj(x_other[:ncoeff], x_other[None, ncoeff:])
    optimizer.jac(x_other[:ncoeff], x_other[None, ncoeff:])
    hessp(x_other, u)
    second = hessp(x, v)
    assert_same(first, second)


def test_first_order_results_unchanged_by_hessp():
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    optimizer = setup.optimizer
    ncoeff = setup.model.get_coeff_size()
    x, v = coeffs_plus_ics_case(setup, seed=2)
    coeffs, ics = x[:ncoeff], x[None, ncoeff:]

    f_before = optimizer.obj(coeffs, ics)
    g_before = optimizer.jac(coeffs, ics)
    optimizer.full_hessp('coeffs+ics')(x, v)
    f_after = optimizer.obj(coeffs, ics)
    g_after = optimizer.jac(coeffs, ics)
    assert f_before == pytest.approx(f_after, rel=RTOL)
    assert_same(g_before, g_after)
    # the coefficient direction is cleared after the HVP
    assert all(np.all(direction == 0) for direction in coefficient_directions(optimizer.timestepper))


def test_coefficient_direction_does_not_leak_into_later_hessp():
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    optimizer = setup.optimizer
    ncoeff = setup.model.get_coeff_size()
    x, v = coeffs_plus_ics_case(setup, seed=3)
    ic_hessp = optimizer.full_hessp('ics', coeffs0=setup.coeff)
    ics, ics_direction = x[ncoeff:], v[ncoeff:]

    before = ic_hessp(ics, ics_direction)
    # an HVP with a nonzero coefficient direction in between
    optimizer.full_hessp('coeffs+ics')(x, v)
    after = ic_hessp(ics, ics_direction)
    assert_same(before, after)


def test_coefficient_direction_is_cleared_after_a_failed_hessp(monkeypatch):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    optimizer = setup.optimizer
    timestepper = optimizer.timestepper
    coeffs = flat(setup.coeff) * 1.1
    direction = 0.01 * flat(setup.coeff)
    expected = optimizer.hessp(coeffs, None, coeffs_dir_arr=direction)

    calls = []

    def failing_step(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('simulated failure in the reverse sweep')

    monkeypatch.setattr(timestepper, 'take_second_order_adjoint_step', failing_step)
    with pytest.raises(RuntimeError, match='simulated failure'):
        optimizer.hessp(coeffs, None, coeffs_dir_arr=direction)
    monkeypatch.undo()
    assert calls, 'the failure was not triggered'
    assert all(np.all(value == 0) for value in coefficient_directions(timestepper))
    # and the next evaluation is unaffected by the aborted one
    assert_same(expected, optimizer.hessp(coeffs, None, coeffs_dir_arr=direction))


def test_reduced_hessp_repeated_calls_agree_and_are_counted():
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS)
    x, v = coeffs_plus_ics_case(setup, seed=4)
    problem = setup.optimizer.reduced_problem(x, opt_type='coeffs+ics', scaling_factors=np.abs(x) + 1.0)
    p = problem.to_reduced(v)
    first = problem.hessp(problem.z0, p)
    problem.jac(problem.z0 + 0.01 * p)
    second = problem.hessp(problem.z0, p)
    assert_same(first, second)
    assert problem.nhvp == 2
