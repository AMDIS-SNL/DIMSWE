#import matplotlib.pyplot as plt
import numpy as np
from dimswe.optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import EmptyLogger
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.optimize import compute_state_block, compute_states, create_states
from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function

import pytest

from firedrake import assemble, inner, norm


solver_parameters=overall_solver_parameters
basic_linear_system = { 'ksp_type': 'preonly', 'pc_type' : 'lu'} #'ksp_monitor_true_residual': None}
overall_solver_parameters['erkstage-f'] = basic_linear_system
overall_solver_parameters['erkstage-aux'] = basic_linear_system
overall_solver_parameters['erkstage-mu'] = basic_linear_system
overall_solver_parameters['erkstage-muaux'] = basic_linear_system
overall_solver_parameters['erk-dlambda'] = basic_linear_system
overall_solver_parameters['erk-grad'] = basic_linear_system

def taylor_remainder_check(jac_func, fd_func, x_func, x0, x0_perturb, x0_perturbed):
    remainders = []
    factors = 0.5 ** np.arange(0, 8)
    eps_list = list(1e-2 * factors)
    for eps in eps_list:
        fdp = fd_func(x_func(eps, x0, x0_perturb, x0_perturbed))
        fdm = fd_func(x_func(-eps, x0, x0_perturb, x0_perturbed))
        jac = jac_func(x0, x0_perturb)
        taylor_remainder =  fdp - fdm - 2.*eps*jac
        print(eps, taylor_remainder, fdp, fdm, 2.*eps*jac, fdp - fdm)
        remainders.append([eps,taylor_remainder])
    rates = []
    for i in range(len(remainders)-1):
        r1 = remainders[i][1]
        r2 = remainders[i + 1][1]
        h1 = remainders[i][0]
        h2 = remainders[i + 1][0]
        rate = np.log(r1 / r2) / np.log(h1 / h2)
        rates.append(rate)
    print(rates)

@pytest.mark.skip
def test_multiple_timestep_gradient_ic():

    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeff, model_coeff_sub, model_coeff_split = model.get_coeff_var('coeff')
    timestepper = get_timestepper(parameters, model, logger, solver_parameters=solver_parameters)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)
    timestepper.set_coeff(model_coeff)

    nsteps = 5
    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, nsteps, dt, x0, t0)
    #xns2, xn_subs2, steps2, tns2 = compute_state_block(model, timestepper, 2, nsteps//2, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, model_coeff, nsteps, model.spaces.dx)
    #objective_2 = L2Objective(xns2, tns2, model_coeff, nsteps//2, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)
    #optimizer_2 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_2, dt)
    x0_perturbed, x0_perturbed_sub, x0_perturbed_split = model.get_x_var('x0_perturbed')
    x0_perturb, x0_perturb_sub, x0_perturb_split = model.get_x_var('x0_perturb')

#SHOULD EAT A SET OF ICs!
#AND COMPUTE BOTH JAC1 AND JAC2
    def jac_func(x0, x0perturb):
        x0_arr = np.expand_dims(create_flattened_numpy_arr_from_mixed_function(x0), 0)
        x0_perturb_arr = np.expand_dims(create_flattened_numpy_arr_from_mixed_function(x0_perturb), 0)
        jac_ic_1 = optimizer_1.jac(None, x0_arr, coeffs0=model_coeff)
        return jac_ic_1.dot(np.ravel(x0_perturb_arr))

#SHOULD EAT A SET OF ICs!
#AND COMPUTE BOTH OBJ1 and OBJ2
    def fd_func(xarr):
        return optimizer_1.obj(None, xarr, coeffs0=model_coeff)

#SHOULD EAT A SET OF ICs!
    def x_func(eps, x0, x0_perturb, x0_perturbed):
        x0_perturbed.assign(x0 + float(eps)*x0_perturb)
        return np.expand_dims(create_flattened_numpy_arr_from_mixed_function(x0_perturbed), 0)

    #zero gradients at optimality
#CREATE SET OF ICs!
    x0_perturb.assign(x0[0] * 0.05)
#SHOULD EAT A SET OF ICs!
    taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
    taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)

    x0_new, x0_new_sub, x0_new_split = model.get_full_var('x0_new', split_x_and_aux=True)
    parameters['initial-conditions']['ox'] = 0.11
    parameters['initial-conditions']['oy'] = 0.11
    model.initialize(x0_new_sub, t0, new_params=parameters)
    x0_perturb.assign(x0_new[0] * 0.05)
#CREATE SET OF ICs!

#SHOULD EAT A SET OF ICs!
    taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)
    taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)

@pytest.mark.skip
def test_single_timestep_gradient_ic():

    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeff, model_coeff_sub, model_coeff_split = model.get_coeff_var('coeff')
    timestepper = get_timestepper(parameters, model, logger, solver_parameters=solver_parameters)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)
    timestepper.set_coeff(model_coeff)

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 1, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, model_coeff, 1, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)
    x0_perturbed, x0_perturbed_sub, x0_perturbed_split = model.get_x_var('x0_perturbed')
    x0_perturb, x0_perturb_sub, x0_perturb_split = model.get_x_var('x0_perturb')

    def jac_func(x0, x0perturb):
        x0_arr = np.expand_dims(create_flattened_numpy_arr_from_mixed_function(x0), 0)
        x0_perturb_arr = np.expand_dims(create_flattened_numpy_arr_from_mixed_function(x0_perturb), 0)
        jac_ic_1 = optimizer_1.jac(None, x0_arr, coeffs0=model_coeff)
        return jac_ic_1.dot(np.ravel(x0_perturb_arr))

    def fd_func(xarr):
        return optimizer_1.obj(None, xarr, coeffs0=model_coeff)

    def x_func(eps, x0, x0_perturb, x0_perturbed):
        x0_perturbed.assign(x0 + float(eps)*x0_perturb)
        return np.expand_dims(create_flattened_numpy_arr_from_mixed_function(x0_perturbed), 0)

    #zero gradients at optimality
    x0_perturb.assign(x0[0] * 0.05)

    taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
    taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
#ADD A CHECK THAT THIS GRADIENT IS FUNCTIONALLY ZERO?

    #check gradients

    x0_new, x0_new_sub, x0_new_split = model.get_full_var('x0_new', split_x_and_aux=True)
    parameters['initial-conditions']['ox'] = 0.11
    parameters['initial-conditions']['oy'] = 0.11
    model.initialize(x0_new_sub, t0, new_params=parameters)
    x0_perturb.assign(x0_new[0] * 0.05)

    taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)
    taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)


def test_single_timestep_gradient_params():

    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeffs = model.get_coeff_var('coeff')
    model_coeff, model_coeff_sub, model_coeff_split = model_coeffs
    timestepper = get_timestepper(parameters, model, logger, solver_parameters=solver_parameters)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)
    timestepper.set_coeff(model_coeff)

    coeff, coeff_sub, coeff_split = model.get_coeff_var('coeff')
    delta_coeff, _, _ = model.get_coeff_var('delta_coeff')
    perturbed_coeff, _, _ = model.get_coeff_var('perturbed_coeff')

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 1, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, coeff, 1, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)



    def jac_func(param, param_perturb):
        param_arr = create_flattened_numpy_arr_from_mixed_function(param)
        param_perturb_arr = create_flattened_numpy_arr_from_mixed_function(param_perturb)
        jac_param_1 = optimizer_1.jac(param_arr, None)
        return jac_param_1.dot(param_perturb_arr)

    def fd_func(param_arr):
        return optimizer_1.obj(param_arr, None)

    def param_func(eps, param, param_perturb, param_perturbed):
        param_perturbed.assign(param + float(eps)*param_perturb)
        return create_flattened_numpy_arr_from_mixed_function(param_perturbed)

    #zero gradients at optimality
    model.set_coeffs(parameters, coeff_sub)
    delta_coeff.assign(coeff * 0.1)
    #print('coeff norm', model.norm(coeff))
#ADD A CHECK THAT THIS GRADIENT IS FUNCTIONALLY ZERO?
    taylor_remainder_check(jac_func, fd_func, param_func, coeff, delta_coeff, perturbed_coeff)
    taylor_remainder_check(jac_func, fd_func, param_func, coeff, delta_coeff, perturbed_coeff)


    #check gradients
    parameters['hyperviscosity']['c0'] = 0.05
    parameters['hyperviscosity']['s'] = 2.8
    model.set_coeffs(parameters, coeff_sub)
    delta_coeff.assign(coeff * 0.1)
    #print('coeff norm', model.norm(coeff))

#THIS IS STILL FAILING :(
    taylor_remainder_check(jac_func, fd_func, param_func, coeff, delta_coeff, perturbed_coeff)
    taylor_remainder_check(jac_func, fd_func, param_func, coeff, delta_coeff, perturbed_coeff)
