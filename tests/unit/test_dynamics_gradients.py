"""Taylor tests of the UFL derivatives of the timestepper residual forms.

The adjoint steps are built from derivative() of the right-hand-side forms with respect to the
state x, the auxiliary variables W, and the coefficients. These tests check each derivative used
there against the forms themselves. For a LieSplit timestepper, each sub-integrator is tested.
"""

from firedrake import derivative

from tests.gradient_helpers import assert_taylor_convergence, build_model, form_taylor_test, random_direction


def explicit_integrators(timestepper):
    """Return the explicit RK integrators of a timestepper (the sub-integrators of a LieSplit)."""
    return list(getattr(timestepper, 'time_integrators', [timestepper]))


def derivative_cases(setup, integrator):
    """Return (name, form, jacobian, variable, direction) for every derivative used by the adjoint.

    The integrator is first stepped forward from the initial state, so its stage and auxiliary
    variables hold realistic values, and the derivatives are evaluated at that state. Directions
    are seeded random perturbations rather than multiples of the current values, because some
    values are legitimately zero: a sub-integrator of a LieSplit only computes the auxiliary
    variables of its own terms, and leaves the others at zero.
    """
    model = setup.model
    x1, x1_sub, _ = model.get_full_var('x1', split_x_and_aux=True)
    integrator.reset_internal_vars()
    integrator.take_forward_step(x1, x1_sub, setup.x0, setup.t0, setup.dt)
    integrator.xk[0].assign(setup.x0[0])

    xtrial, _ = model.get_full_trial_vars(split_x_and_aux=True)
    rhs_F, rhs_W, _, _ = integrator.get_rhs_expr()

    x_direction, _, _ = model.get_x_var('x_direction')
    random_direction(x_direction, base=setup.x0[0], seed=1)
    cases = [('dF/dx', rhs_F, derivative(rhs_F, integrator.xk[0], xtrial[0]), integrator.xk[0], x_direction)]

    if model.has_coeff():
        coeff_direction, _, _ = model.get_coeff_var('coeff_direction')
        random_direction(coeff_direction, base=setup.coeff, seed=2)
        cases.append(('dF/dcoeff', rhs_F, derivative(rhs_F, integrator.coeff, integrator.grad_trial),
                      integrator.coeff, coeff_direction))

    if model.has_aux():
        integrator.xk[1].assign(integrator.Fi[0][0][1])
        aux_direction, _, _ = model.get_aux_var('aux_direction')
        random_direction(aux_direction, base=integrator.Fi[0][0][1], seed=3)
        cases.append(('dF/dW', rhs_F, derivative(rhs_F, integrator.xk[1], xtrial[1]), integrator.xk[1], aux_direction))
        if not rhs_W == 0:
            cases.append(('dW/dx', rhs_W, derivative(rhs_W, integrator.xk[0], xtrial[0]), integrator.xk[0], x_direction))
        # dW/dW and dW/dcoeff are identically zero for the current forcing terms, but UFL does not
        # simplify them to empty forms, so they are not tested here

    return [case for case in cases if not case[2].empty()]


def test_residual_form_derivatives():
    setup = build_model()
    checked = []
    for index, integrator in enumerate(explicit_integrators(setup.timestepper)):
        for name, form, jacobian, variable, direction in derivative_cases(setup, integrator):
            _, remainders, rates, change = form_taylor_test(form, jacobian, variable, direction)
            print(f'integrator {index} {name}', remainders, rates)
            assert_taylor_convergence(rates, remainders=remainders, change=change)
            checked.append(name)
    assert 'dF/dx' in checked
