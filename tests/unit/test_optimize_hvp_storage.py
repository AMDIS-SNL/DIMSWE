"""Tests of the HVP storage and forward passes in Lagrangian_ODEConstrainedOptimization.

The Hessian-vector product needs, per block, the full state trajectory, the full tangent-linear
trajectory, and the first- and second-order observation sources (docs/hessian_vector_products.md,
section 4). These tests check that
- compute_states_and_tangents reproduces compute_states for the states,
- the stored tangent trajectory is the derivative of the state trajectory, for initial-state and
  coefficient directions,
- the second-order observation sources are the derivative of the first-order ones.
"""

import numpy as np
import pytest
from firedrake import assemble

from dimswe.numpy_helpers import set_mixed_function_from_flattened_array
from dimswe.optimize import compute_states, compute_states_and_tangents, create_states, create_tangent_states

from tests.gradient_helpers import (
    assert_taylor_convergence,
    build_model,
    convergence_rates,
    flat,
    setup_problem,
)

EPS_LIST = list(1e-2 * 0.5 ** np.arange(0, 6))
NSTEPS = 3


def taylor_remainders(evaluate, base, derivative):
    """Return (remainders, rates, change) for |evaluate(eps) - base - eps*derivative| over EPS_LIST."""
    remainders = []
    change = 0.0
    for eps in EPS_LIST:
        value = evaluate(eps)
        remainders.append(float(np.linalg.norm(value - base - eps * derivative)))
        change = max(change, float(np.linalg.norm(value - base)))
    return np.array(remainders), convergence_rates(EPS_LIST, remainders), change


def seeded_directions(x0, coeffs, case, seed=0):
    """Return (ic_direction, coeff_direction) arrays for a direction case, zero where unused."""
    generator = np.random.default_rng(seed)
    ic_dir = np.zeros_like(x0)
    coeff_dir = np.zeros_like(coeffs)
    if case in ('ics', 'all'):
        ic_dir = 0.01 * (np.abs(x0) + 1e-3 * np.max(np.abs(x0))) * generator.standard_normal(x0.size)
    if case in ('coeffs', 'all'):
        coeff_dir = 0.1 * coeffs
    return ic_dir, coeff_dir


def test_states_match_compute_states():
    setup = build_model()
    model = setup.model
    ts = setup.timestepper

    ref, ref_subs, ref_tns = create_states(model, NSTEPS)
    ts.reset_internal_vars()
    compute_states(ref, ref_subs, ref_tns, model, ts, NSTEPS, setup.dt, setup.x0, setup.t0)

    xns, _, tns = create_states(model, NSTEPS)
    wns = create_tangent_states(model, NSTEPS)
    ts.set_numpy_coeff_direction(None)
    ts.reset_internal_vars()
    compute_states_and_tangents(xns, tns, wns, ts, NSTEPS, setup.dt, setup.x0, None, setup.t0)

    for n in range(NSTEPS + 1):
        expected = flat(ref[n][0])
        assert np.allclose(flat(xns[n][0]), expected, rtol=1e-10, atol=1e-12 * np.max(np.abs(expected)))
        assert float(tns[n]) == float(ref_tns[n])
        # zero start tangent and zero coefficient direction give a zero tangent trajectory
        assert np.all(flat(wns[n][0]) == 0)


@pytest.mark.parametrize('case', ['ics', 'coeffs', 'all'])
def test_tangent_trajectory_taylor(case):
    setup = build_model()
    model = setup.model
    ts = setup.timestepper
    x0 = flat(setup.x0[0])
    coeffs0 = flat(setup.coeff)
    ic_dir, coeff_dir = seeded_directions(x0, coeffs0, case)

    xns, xn_subs, tns = create_states(model, NSTEPS)
    x_start, _, _ = model.get_full_var('x_start', split_x_and_aux=True)
    w_start, _, _ = model.get_x_var('w_start')

    def final_state(eps):
        """x_N for start state x0 + eps*ic_dir and coefficients coeffs0 + eps*coeff_dir."""
        ts.set_numpy_coeff(coeffs0 + eps * coeff_dir)
        set_mixed_function_from_flattened_array(x_start[0], x0 + eps * ic_dir)
        ts.reset_internal_vars()
        compute_states(xns, xn_subs, tns, model, ts, NSTEPS, setup.dt, x_start, setup.t0)
        return flat(xns[NSTEPS][0])

    base = final_state(0.0)

    wns = create_tangent_states(model, NSTEPS)
    set_mixed_function_from_flattened_array(w_start, ic_dir)
    ts.set_numpy_coeff_direction(coeff_dir)
    ts.reset_internal_vars()
    compute_states_and_tangents(xns, tns, wns, ts, NSTEPS, setup.dt, x_start, w_start, setup.t0)
    tangent = flat(wns[NSTEPS][0])
    assert np.any(tangent != 0)

    remainders, rates, change = taylor_remainders(final_state, base, tangent)
    ts.set_numpy_coeff(coeffs0)
    print(case, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


@pytest.mark.parametrize('var_weights', [None, {'v': 1.0, 'h': 1.0e-4, 'S': 1.0e-6}])
def test_observation_tangent_sources_taylor(var_weights):
    setup = setup_problem(NSTEPS, obs_steps=[1, NSTEPS], var_weights=var_weights)
    optimizer = setup.optimizer
    model = setup.model
    x0 = flat(setup.x0[0]) * 1.01  # away from the truth, where the sources vanish
    coeffs0 = flat(setup.coeff)
    ic_dir, coeff_dir = seeded_directions(x0, coeffs0, 'all', seed=2)

    optimizer._setup_hvp()
    optimizer._set_coeffs(coeffs0, None)

    def observation_dual_sum(eps):
        """sum_n s_n (dual) for start state x0 + eps*ic_dir and coefficients coeffs0 + eps*coeff_dir."""
        optimizer._set_coeffs(coeffs0 + eps * coeff_dir, None)
        optimizer.timestepper.reset_internal_vars()
        optimizer._compute_block_states(0, (x0 + eps * ic_dir)[None, :])
        total = np.zeros_like(x0)
        for step in optimizer.obs_steps:
            optimizer._assign_mismatch(0, step)
            total += flat(assemble(optimizer.source_form))
        return total

    base = observation_dual_sum(0.0)

    optimizer._set_coeffs(coeffs0, None)
    optimizer._set_coeff_direction(coeff_dir)
    optimizer.timestepper.reset_internal_vars()
    optimizer._compute_block_states_and_tangents(0, x0[None, :], ic_dir[None, :])
    grad_sum, hess_sum = optimizer._compute_observation_sources(0)
    optimizer._set_coeff_direction(None)

    # the first-order sums are the ones jac adds to grad_ics
    assert np.allclose(grad_sum, base, rtol=1e-10, atol=1e-10 * np.max(np.abs(base)))
    assert np.any(hess_sum != 0)
    # the primal second-order sources M^{-1} sdot_n used by the reverse sweep are populated
    assert all(np.any(flat(source) != 0) for source in optimizer.obs_tangent_sources)

    remainders, rates, change = taylor_remainders(observation_dual_sum, base, hess_sum)
    print(var_weights, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)
