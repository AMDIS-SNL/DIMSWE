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


solver_parameters=overall_solver_parameters.copy()
basic_linear_system = { 'ksp_type': 'preonly', 'pc_type' : 'lu'} #'ksp_monitor_true_residual': None}
solver_parameters['erkstage-f'] = basic_linear_system
solver_parameters['erkstage-aux'] = basic_linear_system
solver_parameters['erkstage-mu'] = basic_linear_system
solver_parameters['erkstage-muaux'] = basic_linear_system
solver_parameters['erk-dlambda'] = basic_linear_system
solver_parameters['erk-grad'] = basic_linear_system

def taylor_remainder_check(jac_func, fd_func, x_func, x0, x0_perturb, x0_perturbed):
    remainders = []
    factors = 0.5 ** np.arange(0, 8)
    eps_list = list(1e-1 * factors)
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

def test_multiple_timestep_gradient_coeffs():

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

    nsteps = 5
    coeff, coeff_sub, coeff_split = model.get_coeff_var('coeff')
    new_coeff, new_coeff_sub, new_coeff_split = model.get_coeff_var('new_coeff')
    delta_coeff, _, _ = model.get_coeff_var('delta_coeff')
    perturbed_coeff, _, _ = model.get_coeff_var('perturbed_coeff')

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, nsteps, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, nsteps, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)



    def jac_func(coeff, coeff_perturb):
        coeff_arr = create_flattened_numpy_arr_from_mixed_function(coeff)
        coeff_perturb_arr = create_flattened_numpy_arr_from_mixed_function(coeff_perturb)
        jac_coeff_1 = optimizer_1.jac(coeff_arr, None)
        return jac_coeff_1.dot(coeff_perturb_arr)

    def fd_func(coeff_arr):
        return optimizer_1.obj(coeff_arr, None)

    def coeff_func(eps, coeff, coeff_perturb, coeff_perturbed):
        coeff_perturbed.assign(coeff + float(eps)*coeff_perturb)
        return create_flattened_numpy_arr_from_mixed_function(coeff_perturbed)

    #zero gradients at optimality
    model.set_coeffs(parameters, coeff_sub)
    delta_coeff.assign(coeff * 0.1)
    #print('coeff norm', model.norm(coeff))
    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, coeff_func, coeff, delta_coeff, perturbed_coeff)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, coeff_func, coeff, delta_coeff, perturbed_coeff)
    assert(jac1==0)
    assert(jac2==0)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1, np.ones(rates1.shape)*3.0, rtol=1e-1, atol=1e-5))

    #check gradients
    parameters['hyperviscosity']['c0'] = 0.05
    parameters['hyperviscosity']['s'] = 2.8
    model.set_coeffs(parameters, new_coeff_sub)
    delta_coeff.assign(new_coeff * 0.1)

    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, coeff_func, new_coeff, delta_coeff, perturbed_coeff)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, coeff_func, new_coeff, delta_coeff, perturbed_coeff)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1, np.ones(rates1.shape)*3.0, rtol=1e-1, atol=1e-5))

def test_single_timestep_gradient_coeffs():

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
    new_coeff, new_coeff_sub, new_coeff_split = model.get_coeff_var('new_coeff')
    delta_coeff, _, _ = model.get_coeff_var('delta_coeff')
    perturbed_coeff, _, _ = model.get_coeff_var('perturbed_coeff')

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 1, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, 1, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)



    def jac_func(coeff, coeff_perturb):
        coeff_arr = create_flattened_numpy_arr_from_mixed_function(coeff)
        coeff_perturb_arr = create_flattened_numpy_arr_from_mixed_function(coeff_perturb)
        jac_coeff_1 = optimizer_1.jac(coeff_arr, None)
        return jac_coeff_1.dot(coeff_perturb_arr)

    def fd_func(coeff_arr):
        return optimizer_1.obj(coeff_arr, None)

    def coeff_func(eps, coeff, coeff_perturb, coeff_perturbed):
        coeff_perturbed.assign(coeff + float(eps)*coeff_perturb)
        return create_flattened_numpy_arr_from_mixed_function(coeff_perturbed)

    #zero gradients at optimality
    model.set_coeffs(parameters, coeff_sub)
    delta_coeff.assign(coeff * 0.1)
    #print('coeff norm', model.norm(coeff))
    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, coeff_func, coeff, delta_coeff, perturbed_coeff)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, coeff_func, coeff, delta_coeff, perturbed_coeff)
    assert(jac1==0)
    assert(jac2==0)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1, np.ones(rates1.shape)*3.0, rtol=1e-1, atol=1e-5))

    #check gradients
    parameters['hyperviscosity']['c0'] = 0.05
    parameters['hyperviscosity']['s'] = 2.8
    model.set_coeffs(parameters, new_coeff_sub)
    delta_coeff.assign(new_coeff * 0.1)

    jac1, remainders1, rates1 = taylor_remainder_check(jac_func, fd_func, coeff_func, new_coeff, delta_coeff, perturbed_coeff)
    jac2, remainders2, rates2 = taylor_remainder_check(jac_func, fd_func, coeff_func, new_coeff, delta_coeff, perturbed_coeff)
    assert(np.allclose(rates1, rates2, equal_nan=True))
    assert(np.allclose(remainders1, remainders2))
    assert(np.allclose(rates1, np.ones(rates1.shape)*3.0, rtol=1e-1, atol=1e-5))
