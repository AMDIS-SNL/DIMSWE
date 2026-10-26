"""Taylor tests for the observation-based losses and the ic-params gradient.

These check that the gradients computed by Lagrangian_ODEConstrainedOptimization converge at
second order when the loss compares the model with data at several observation steps, and when
the parameters are analytic initial-condition parameters differentiated with UFL.
"""

import numpy as np

from dimswe.optimize import ParametricInitialCondition

from tests.gradient_helpers import (
    assert_repeatable,
    assert_taylor_convergence,
    flat,
    reduced_taylor_test,
    setup_problem,
)


def test_trajectory_loss_coefficient_gradient():
    setup = setup_problem(4, obs_steps=[1, 2, 3, 4])
    coeffs = flat(setup.coeff)
    direction = np.random.default_rng(0).standard_normal(coeffs.size)

    def run_test():
        # move away from the optimum, where the gradient vanishes
        make_problem = lambda: setup.optimizer.reduced_problem(coeffs * 1.1, opt_type='coeffs', scaling_factors=np.abs(coeffs))
        return reduced_taylor_test(make_problem, direction)

    _, remainders, rates = assert_repeatable(run_test)
    print(remainders, rates)
    assert_taylor_convergence(rates)


def test_trajectory_loss_ic_gradient_with_weights():
    setup = setup_problem(3, obs_steps=[1, 3], var_weights={'v': 1.0, 'h': 1.0e-4, 'S': 1.0e-6})
    x0 = flat(setup.x0[0])
    direction = np.random.default_rng(1).standard_normal(x0.size)

    def run_test():
        make_problem = lambda: setup.optimizer.reduced_problem(x0 * 1.01, opt_type='ics', coeffs0=setup.coeff,
            scaling_factors=np.abs(x0) + 1.0)
        return reduced_taylor_test(make_problem, direction)

    _, remainders, rates = assert_repeatable(run_test)
    print(remainders, rates)
    assert_taylor_convergence(rates)


def test_ic_params_gradient_uses_ufl_tangents():
    setup = setup_problem(3, obs_steps=[3])
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ['ox', 'oy'])
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])

    def run_test():
        make_problem = lambda: setup.optimizer.reduced_problem(params, opt_type='ic-params', coeffs0=setup.coeff,
            ic_map=ic_map, scaling_factors=np.array([0.1, 0.1]))
        return reduced_taylor_test(make_problem, [1.0, 0.5])

    _, remainders, rates = assert_repeatable(run_test)
    print(remainders, rates)
    assert_taylor_convergence(rates)


def test_ic_params_forward_matches_model_initialize():
    setup = setup_problem(1, obs_steps=[1])
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ['ox', 'oy'])
    x0_map = ic_map.forward(ic_map.get_initial_values(setup.parameters))
    x0_model = flat(setup.x0[0])
    assert np.allclose(x0_map, x0_model, rtol=1e-6, atol=1e-8 * np.max(np.abs(x0_model)))


def test_workflow_model_reproduces_truth_trajectory():
    """The workflow's optimization model must reproduce the forward run exactly at the truth.

    Regression test: optimize_params.py built models without calling model.initialize, so the
    Coriolis parameter and forcing-term topography stayed zero and the optimized model was not
    the model that made the truth. At the true coefficients the loss must then be exactly zero.
    """
    from dimswe.optimize import compute_states, create_states
    from dimswe.timestepping import get_timestepper
    from workflows.opt_helpers import build_model, copy_function_data

    from tests.gradient_helpers import SOLVER_PARAMETERS

    truth = setup_problem(3)
    model = build_model(truth.parameters, truth.model.mesh)
    assert float(np.max(np.abs(model.dynamics.coriolis.dat.data_ro))) > 0, 'coriolis was not initialized'

    timestepper = get_timestepper(truth.parameters, model, truth.logger, SOLVER_PARAMETERS)
    coeff, coeff_sub, _ = model.get_coeff_var('coeff')
    model.set_coeffs(truth.parameters, coeff_sub)
    timestepper.set_coeff(coeff)
    x0, _, _ = model.get_full_var('x0_workflow', split_x_and_aux=True)
    # x0 is a mixed Function, so copy component by component
    copy_function_data(x0[0], truth.x0[0])
    states, state_subs, tns = create_states(model, 3)
    compute_states(states, state_subs, tns, model, timestepper, 3, truth.dt, x0, truth.t0)
    truth_final = flat(truth.objective.data_blocks[0][1][0])
    assert np.allclose(flat(states[3][0]), truth_final, rtol=1e-10, atol=1e-12 * np.max(np.abs(truth_final)))
