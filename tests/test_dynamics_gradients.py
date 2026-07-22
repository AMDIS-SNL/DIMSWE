import numpy as np
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import EmptyLogger
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.optimize import compute_state_block, compute_states, create_states
from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function

import pytest

from firedrake import assemble, inner, norm, derivative, action, replace

solver_parameters=overall_solver_parameters
basic_linear_system = { 'ksp_type': 'preonly', 'pc_type' : 'lu'} #'ksp_monitor_true_residual': None}
overall_solver_parameters['erkstage-f'] = basic_linear_system
overall_solver_parameters['erkstage-aux'] = basic_linear_system
overall_solver_parameters['erkstage-mu'] = basic_linear_system
overall_solver_parameters['erkstage-muaux'] = basic_linear_system
overall_solver_parameters['erk-dlambda'] = basic_linear_system
overall_solver_parameters['erk-grad'] = basic_linear_system



def taylor_remainder_check(func, jac, x, x0, xperturb):
    jac = create_flattened_numpy_arr_from_mixed_function(assemble(replace(action(jac, xperturb), {x : x0})))
    diffs = []
    remainders = []
    factors = 0.5 ** np.arange(0, 8)
    eps_list = list(1e-2 * factors)
    for eps in eps_list:
        fp = create_flattened_numpy_arr_from_mixed_function(assemble(replace(func, {x : x0+float(eps)*xperturb})))
        fm = create_flattened_numpy_arr_from_mixed_function(assemble(replace(func, {x : x0-float(eps)*xperturb})))
        remainder = np.linalg.norm(fp - fm - 2.*eps*jac)
        remainders.append([eps, remainder])
        print(eps, remainder, np.linalg.norm(jac), np.linalg.norm(fp - fm), np.linalg.norm(2.*eps*jac))
    rates = []
    for i in range(len(remainders)-1):
        r1 = remainders[i][1]
        r2 = remainders[i + 1][1]
        h1 = remainders[i][0]
        h2 = remainders[i + 1][0]
        rate = np.log(r1 / r2) / np.log(h1 / h2)
        rates.append(rate)
    print(rates)

def test_dynamics_gradients():
    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeff, model_coeff_sub, model_coeff_split = model.get_coeff_var('coeff')
    timestepper = get_timestepper(parameters, model, logger, solver_parameters=solver_parameters)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    x1, x1_sub, x1_split = model.get_full_var('x1', split_x_and_aux=True)
    xtrial, xtrial_subs = model.get_full_trial_vars(split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)
    timestepper.set_coeff(model_coeff)

    rhs_F, rhs_W, rhs_Fis, rhs_Wis = timestepper.get_rhs_expr()


    coeff_pertubation, _, _ = model.get_coeff_var('coeff_perturbation')
    x_pertubation, _, _ = model.get_x_var('x_perturbation')
    aux_pertubation, _, _ = model.get_aux_var('aux_perturbation')

    timestepper.take_forward_step(x1, x1_sub, x0, t0, dt)

    coeff_pertubation.assign(model_coeff * 0.1)
    x_pertubation.assign(x0[0] * 0.1)
    aux_pertubation.assign(timestepper.Fi[0][0][1] * 0.1)
    timestepper.xk[1].assign(timestepper.Fi[0][0][1])



    if model.has_coeff():
        derivF_coeff = derivative(rhs_F, timestepper.coeff, timestepper.grad_trial)
        #derivF_coeff_rhs = assemble(action(derivF_coeff, coeff_pertubation))
        print('derivF_coeff')
        if not derivF_coeff.empty():
            taylor_remainder_check(rhs_F, derivF_coeff, timestepper.coeff, model_coeff, coeff_pertubation)

    derivF_F = derivative(rhs_F, timestepper.xk[0], xtrial[0])
    #derivF_F_rhs = assemble(action(derivF_F, x_pertubation))
    print('derivF_F')
    if not derivF_F.empty():
        taylor_remainder_check(rhs_F, derivF_F, timestepper.xk[0], timestepper.xk[0], x_pertubation)


    if model.has_aux():
        derivF_W = derivative(rhs_F, timestepper.xk[1], xtrial[1])
        #derivF_W_rhs = assemble(action(derivF_W, aux_pertubation))
        print('derivF_W')
        if not derivF_W.empty():
            taylor_remainder_check(rhs_F, derivF_W, timestepper.xk[1], timestepper.xk[1], aux_pertubation)

        if model.has_coeff():

            derivW_coeff = derivative(rhs_W, timestepper.coeff, timestepper.grad_trial)
    #THIS IS FAILING TO RECOGNIZE THAT IT IS ACTUALLY ZERO...
            #derivW_coeff_rhs = assemble(action(derivW_coeff, coeff_pertubation))
            #print('derivW_coeff')
            #if not derivW_coeff.empty():
            #    print('rhsW', rhs_W)
            #    print('deriv', derivW_coeff)
            #    taylor_remainder_check(rhs_W, derivW_coeff, model_coeff, coeff_pertubation, set_coeff_vals)

        derivW_F = derivative(rhs_W, timestepper.xk[0], xtrial[0])
        #derivW_F_rhs = assemble(action(derivW_F, x_pertubation))
        print('derivW_F')
        if not derivW_F.empty():
            taylor_remainder_check(rhs_W, derivW_F, timestepper.xk[0], timestepper.xk[0], x_pertubation)

        derivW_W = derivative(rhs_W, timestepper.xk[1], xtrial[1])
        #derivW_W_rhs = assemble(action(derivW_W, aux_pertubation))
    #THIS IS FAILING TO RECOGNIZE THAT IT IS ACTUALLY ZERO...
    #    print('derivW_W')
    #    if not derivW_W.empty():
    #        taylor_remainder_check(rhs_W, derivW_W,  timestepper.xk[1], aux_pertubation, set_aux_vals)
















# for i in range(timestepper.nstages):
#     timestepper.Fi[i][0][0].assign()
#     if model.has_coeff():
#
#         derivF_coeff = derivative(rhs_Fis[i], timestepper.coeff, timestepper.grad_trial)
#         derivF_coeff_rhs = assemble(action(derivF_coeff, coeff_pertubation))
#
#     derivF_F = derivative(rhs_Fis[i], timestepper.xk[0], timestepper.xk_trial[0])
#     derivF_F_rhs = assemble(action(derivF_F, x_pertubation))
#
#     if model.has_aux():
#
#         derivF_W = derivative(rhs_Fis[i], timestepper.xk[1], timestepper.xk_trial[1])
#         derivF_W_rhs = assemble(action(derivF_W, aux_pertubation))
#         if model.has_coeff():
#             derivW_coeff = derivative(rhs_Wis[i], timestepper.coeff, timestepper.grad_trial)
#             derivW_coeff_rhs = assemble(action(derivW_coeff, coeff_pertubation))
#
#         derivW_F = derivative(rhs_Wis[i], timestepper.xk[0], timestepper.xk_trial[0])
#         derivW_F_rhs = assemble(action(derivW_F, x_pertubation))
#
#         derivW_W = derivative(rhs_Wis[i], timestepper.xk[1], timestepper.xk_trial[1])
#         derivW_W_rhs = assemble(action(derivW_W, aux_pertubation))
