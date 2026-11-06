"""Taylor tests of the tangent-linear and second-order adjoint timestepper steps.

These are the per-step building blocks of the forward-over-adjoint Hessian-vector product
(petsc-opt.pdf, eq. 2.11; see docs/hessian_vector_products.md):

- take_tangent_step must be the directional derivative of take_forward_step with respect to the
  start state and the coefficients,
- the second-order outputs of take_second_order_adjoint_step must be the directional derivative of
  the outputs of take_adjoint_step with respect to the start state, the coefficients and lambda,
- and the first-order outputs of both steps must agree with the existing first-order steps.

Each test runs for the timestepper of tests/tswe.cfg (a LieSplit of RK4 and Euler) and for a
plain RK4 GeneralRK, and for directions perturbing the state, the coefficients, or both.
"""

import numpy as np
import pytest

from dimswe.timestepping import RK4
from tests.gradient_helpers import (
    SOLVER_PARAMETERS,
    assert_taylor_convergence,
    build_model,
    convergence_rates,
    flat,
    random_direction,
)

EPS_LIST = list(1e-2 * 0.5 ** np.arange(0, 6))
INTEGRATORS = ['cfg', 'rk4']
DIRECTION_CASES = ['state', 'coeff', 'all']


def make_integrator(setup, kind):
    """Return the cfg timestepper ('cfg') or a freshly built RK4 integrator ('rk4')."""
    if kind == 'cfg':
        return setup.timestepper
    integrator = RK4(setup.model, setup.logger, SOLVER_PARAMETERS)
    integrator.set_coeff(setup.coeff)
    return integrator


def make_directions(setup, case):
    """Return seeded directions (w, coeff_dir, lam, lam_dot) for a direction case.

    w is an [x, aux] list holding the state direction, coeff_dir a flattened coefficient
    direction, lam a base first-order adjoint and lam_dot its direction. The 'state' case
    perturbs the state and lambda only, 'coeff' the coefficients only, and 'all' everything.
    """
    model = setup.model
    if case in ('coeff', 'all') and not model.has_coeff():
        pytest.skip('model has no coefficients')
    w, _, _ = model.get_full_var('w_direction', split_x_and_aux=True)
    lam, _, _ = model.get_x_var('lam_base')
    lam_dot, _, _ = model.get_x_var('lam_direction')
    random_direction(lam, base=setup.x0[0], seed=3)
    coeff_dir = np.zeros_like(flat(setup.coeff)) if model.has_coeff() else None
    if case in ('state', 'all'):
        random_direction(w[0], base=setup.x0[0], seed=1)
        random_direction(lam_dot, base=lam, seed=4)
    if case in ('coeff', 'all'):
        coeff_dir = 0.1 * flat(setup.coeff)
    return w, coeff_dir, lam, lam_dot


def taylor_remainders(evaluate, base, derivative):
    """Return (remainders, rates, change) for |evaluate(eps) - base - eps*derivative| over EPS_LIST."""
    remainders = []
    change = 0.0
    for eps in EPS_LIST:
        value = evaluate(eps)
        remainders.append(float(np.linalg.norm(value - base - eps * derivative)))
        change = max(change, float(np.linalg.norm(value - base)))
    return np.array(remainders), convergence_rates(EPS_LIST, remainders), change


def shifted_coeffs(coeffs0, coeff_dir, eps):
    """Return coeffs0 + eps*coeff_dir, or None when the model has no coefficients."""
    if coeffs0 is None:
        return None
    return coeffs0 + eps * coeff_dir


@pytest.mark.parametrize('case', DIRECTION_CASES)
@pytest.mark.parametrize('kind', INTEGRATORS)
def test_tangent_step_taylor(kind, case):
    setup = build_model()
    model = setup.model
    integrator = make_integrator(setup, kind)
    w, coeff_dir, _, _ = make_directions(setup, case)
    coeffs0 = flat(setup.coeff) if model.has_coeff() else None

    x_pert, _, _ = model.get_full_var('x_pert', split_x_and_aux=True)
    xnp1, xnp1_sub, _ = model.get_full_var('xnp1', split_x_and_aux=True)
    xnp1_tlm, _, _ = model.get_full_var('xnp1_tlm', split_x_and_aux=True)
    wnp1, _, _ = model.get_full_var('wnp1', split_x_and_aux=True)

    def forward(eps):
        """x-part of N(x0 + eps*w; coeffs0 + eps*coeff_dir)."""
        if coeffs0 is not None:
            integrator.set_numpy_coeff(shifted_coeffs(coeffs0, coeff_dir, eps))
        x_pert[0].assign(setup.x0[0] + float(eps) * w[0])
        if x_pert[1] is not None:
            x_pert[1].assign(setup.x0[1])
        integrator.reset_internal_vars()
        integrator.take_forward_step(xnp1, xnp1_sub, x_pert, setup.t0, setup.dt)
        return flat(xnp1[0])

    x1 = forward(0.0)
    integrator.set_numpy_coeff_direction(coeff_dir)
    integrator.reset_internal_vars()
    integrator.take_tangent_step(wnp1, w, x_pert, setup.t0, setup.dt, xnp1=xnp1_tlm)
    tangent = flat(wnp1[0])

    # the forward update written by take_tangent_step matches take_forward_step
    assert np.allclose(flat(xnp1_tlm[0]), x1, rtol=1e-12, atol=1e-12 * np.max(np.abs(x1)))
    assert np.any(tangent != 0)

    remainders, rates, change = taylor_remainders(forward, x1, tangent)
    if coeffs0 is not None:
        integrator.set_numpy_coeff(coeffs0)
    print(kind, case, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


@pytest.mark.parametrize('case', DIRECTION_CASES)
@pytest.mark.parametrize('kind', INTEGRATORS)
def test_second_order_adjoint_step_taylor(kind, case):
    setup = build_model()
    model = setup.model
    integrator = make_integrator(setup, kind)
    w, coeff_dir, lam, lam_dot = make_directions(setup, case)
    coeffs0 = flat(setup.coeff) if model.has_coeff() else None

    t1 = model.get_t_var()
    t1.assign(setup.t0 + setup.dt)
    x_pert, _, _ = model.get_full_var('x_pert', split_x_and_aux=True)
    lam_pert, _, _ = model.get_x_var('lam_pert')
    delta_lambda, _, _ = model.get_x_var('delta_lambda_out')
    delta_lambda_dot, _, _ = model.get_x_var('delta_lambda_dot_out')
    delta_grad_coeff, _, _ = model.get_coeff_var('delta_grad_coeff_out')

    def stack(lambda_part, coeff_part):
        """Concatenate the state and (if present) coefficient parts of a step output."""
        return np.hstack([lambda_part] + ([coeff_part] if coeff_part is not None else []))

    def first_order(eps):
        """take_adjoint_step outputs at (x0 + eps*w, coeffs0 + eps*coeff_dir, lam + eps*lam_dot)."""
        if coeffs0 is not None:
            integrator.set_numpy_coeff(shifted_coeffs(coeffs0, coeff_dir, eps))
        x_pert[0].assign(setup.x0[0] + float(eps) * w[0])
        lam_pert.assign(lam + float(eps) * lam_dot)
        integrator.reset_internal_vars()
        delta_lambda_rhs, delta_grad_rhs = integrator.take_adjoint_step(delta_grad_coeff, delta_lambda, lam_pert, x_pert, t1, setup.dt)
        return stack(delta_lambda_rhs, delta_grad_rhs)

    base = first_order(0.0)
    base_delta_lambda = flat(delta_lambda)

    integrator.set_numpy_coeff_direction(coeff_dir)
    integrator.reset_internal_vars()
    outputs = integrator.take_second_order_adjoint_step(delta_lambda, delta_lambda_dot, lam_pert, lam_dot, x_pert, w, t1, setup.dt)
    delta_lambda_rhs, delta_grad_rhs, delta_lambda_dot_rhs, delta_hess_coeff_rhs = outputs

    # the first-order part matches take_adjoint_step
    first = stack(delta_lambda_rhs, delta_grad_rhs)
    assert np.allclose(first, base, rtol=1e-10, atol=1e-10 * np.max(np.abs(base)))
    assert np.allclose(flat(delta_lambda), base_delta_lambda, rtol=1e-10, atol=1e-10 * np.max(np.abs(base_delta_lambda)))

    second = stack(delta_lambda_dot_rhs, delta_hess_coeff_rhs)
    assert np.any(second != 0)

    remainders, rates, change = taylor_remainders(first_order, base, second)
    if coeffs0 is not None:
        integrator.set_numpy_coeff(coeffs0)
    print(kind, case, remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)
