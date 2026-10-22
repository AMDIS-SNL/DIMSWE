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
#basic_linear_system = { 'ksp_type': 'cg', 'pc_type' : 'jacobi'} #'ksp_monitor_true_residual': None}
solver_parameters['erkstage-f'] = basic_linear_system
solver_parameters['erkstage-aux'] = basic_linear_system
solver_parameters['erkstage-mu'] = basic_linear_system
solver_parameters['erkstage-muaux'] = basic_linear_system
solver_parameters['erk-dlambda'] = basic_linear_system
solver_parameters['erk-grad'] = basic_linear_system

def taylor_remainder_check(jac_func, fd_func, x_func, x0, x0_perturb, x0_perturbed):
    remainders = []
    factors = 0.5 ** np.arange(0, 8)
    eps_list = list(1e-2 * factors)
    jac = jac_func(x0, x0_perturb)
    for eps in eps_list:
        fdp = fd_func(x_func(eps, x0, x0_perturb, x0_perturbed))
        fdm = fd_func(x_func(-eps, x0, x0_perturb, x0_perturbed))
        taylor_remainder =  fdp - fdm - 2.*eps*jac
        print(eps, taylor_remainder, fdp, fdm, 2.*eps*jac, fdp - fdm, jac)
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
    return jac, np.array(remainders), np.array(rates)


def test_multiple_timestep_gradient_ic():

    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeff, model_coeff_sub, model_coeff_split = model.get_coeff_var('coeff')
    timestepper = get_timestepper(parameters, model, logger, solver_parameters)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)
    timestepper.set_coeff(model_coeff)

    nsteps = 5
    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, nsteps, dt, x0, t0)
    #xns2, xn_subs2, steps2, tns2 = compute_state_block(model, timestepper, 2, nsteps//2, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, nsteps, model.spaces.dx)
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
    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
    assert(jac1==0)
    assert(jac2==0)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1, np.ones(rates1.shape)*3.0, rtol=1e-1, atol=1e-5))

    x0_new, x0_new_sub, x0_new_split = model.get_full_var('x0_new', split_x_and_aux=True)
    parameters['initial-conditions']['ox'] = 0.11
    parameters['initial-conditions']['oy'] = 0.11
    model.initialize(x0_new_sub, t0, new_params=parameters)
    x0_perturb.assign(x0_new[0] * 0.05)
#CREATE SET OF ICs!

#SHOULD EAT A SET OF ICs!
    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1[:5], np.ones(rates1[:5].shape)*3.0, rtol=1e-1, atol=1e-5))

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
    objective_1 = L2Objective(xns1, tns1, 1, model.spaces.dx)
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

    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, x_func, x0[0], x0_perturb, x0_perturbed)
    assert(jac1==0)
    assert(jac2==0)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1, np.ones(rates1.shape)*3.0, rtol=1e-1, atol=1e-5))

    x0_new, x0_new_sub, x0_new_split = model.get_full_var('x0_new', split_x_and_aux=True)
    parameters['initial-conditions']['ox'] = 0.11
    parameters['initial-conditions']['oy'] = 0.11
    model.initialize(x0_new_sub, t0, new_params=parameters)
    x0_perturb.assign(x0_new[0] * 0.05)

    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, x_func, x0_new[0], x0_perturb, x0_perturbed)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1[:4], np.ones(rates1[:4].shape)*3.0, rtol=1e-1, atol=1e-5))
