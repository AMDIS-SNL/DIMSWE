#import matplotlib.pyplot as plt
import numpy as np
from dimswe.optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import EmptyLogger
from dimswe.parameters import get_parameters
from dimswe.optimize import compute_state_block, compute_states, create_states

def test_single_timestep_gradient_ic():

    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeffs = model.get_coeff_var('coeff')
    model_coeff, model_coeff_sub, model_coeff_split = model_coeffs
    timestepper = get_timestepper(parameters, model, logger, model_coeffs)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)
#UNCLEAR HOW TO FEED THIS IN CORRECTLY?
#BASICALLY WHEN MODEL.INITIALIZE IS CALLED WE NEED TO DO THE "RIGHT" THING...
#AND IN OBJ/JAC WE NEED TO CORRECTLY SET THE IC
#DO THIS VIA A LAMBDA FUNCTION ie _obj(params, ic, ...)

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 1, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, model_coeff, 1, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)

    eps = 0.000001
    x0_perturb, x0_perturb_sub, x0_perturb_split = model.get_full_var('x0_perturb', split_x_and_aux=True)
    x0_perturbed, x0_perturbed_sub, x0_perturbed_split = model.get_full_var('x0_perturbed', split_x_and_aux=True)

    x0_perturb.assign(x0 * 0.01)
    x0_perturbed.assign(x0 + eps*x0_perturb)
#SHOULD PROBABLY RAVEL THIS INTO A LONG FLAT ARRAY
    x0_perturb_arr = x0_perturb.dat.data

    fd_jac_ic_1 = (optimizer_1.obj(coeff, ic=x0_perturbed) - optimizer_1.obj(coeff, ic=x0))/eps
    jac_ic_1 = optimizer_1.jac(coeff, ic=x0)
    print(jac_ic_1)
    print(jac_ic_1.dot(x0_perturb_arr))
    print(fd_jac_ic_1)

    assert(np.count_nonzero(jac_ic_1.dot(x0_perturb_arr)) == 0)
#THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    #assert(np.allclose(jac_ic_1.dot(x0_perturb_arr), fd_jac_ic_1))

    #check gradients
#SET A NEW IC HERE!
    x0_new, x0_new_sub, x0_new_split = model.get_full_var('x0_new', split_x_and_aux=True)
    parameters['initial-conditions']['ox'] = 0.08
    parameters['initial-conditions']['oy'] = 0.12
    model.initialize(x0_new_sub, t0, new_params=parameters)
    x0_perturb.assign(x0_new * 0.01)
    x0_perturbed.assign(x0_new + eps*x0_perturb)

    fd_jac_ic_1 = (optimizer_1.obj(coeff, ic=x0_perturbed) - optimizer_1.obj(coeff, ic=x0_new))/eps
    jac_ic_1 = optimizer_1.jac(coeff, ic=x0_new)
    print(jac_ic_1)
    print(jac_ic_1.dot(x0_perturb_arr))
    print(fd_jac_ic_1)

#THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_ic_1.dot(delta_ic_arr), fd_jac_ic_1))

def test_single_timestep_gradient_params():

    parameters = get_parameters('tests/tswe.cfg')
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    model_coeffs = model.get_coeff_var('coeff')
    model_coeff, model_coeff_sub, model_coeff_split = model_coeffs
    timestepper = get_timestepper(parameters, model, logger, model_coeffs)
    dt = parameters['timestepping']['dt']
    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, model_coeff_sub)

    coeff, coeff_sub, coeff_split = model.get_coeff_var('coeff')
    model.set_coeffs(parameters, coeff_sub)

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 1, dt, x0, t0)
    objective_1 = L2Objective(xns1, tns1, coeff, 1, model.spaces.dx)
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)

    eps = 0.000001

    delta_coeff, _, _ = model.get_coeff_var('delta_coeff')
#THIS IS VERY HACKY, NEED A BETTER WAY TO HANDLE THIS!
    delta_coeff.assign(coeff * 0.01)
    delta_coeff_arr = delta_coeff.dat.data[0]

    newcoeff, _, _ = model.get_coeff_var('new_coeff')
    newcoeff.assign(coeff + eps*delta_coeff)
    #print(old.data.data[0])

    #print(coeff.data.data[0])
    #print(newcoeff.data.data[0])

#SET DELTA SOMEHOW!

    #check zero gradients at optimality
    #\+eps*delta_coeff
#THESE ALL GIVE DIFFERENT ANSWERS!
    #print(optimizer_1.obj(coeff))
    #print(optimizer_1.obj(coeff))
    #print('coeff', optimizer_1.obj(coeff))
#SOMETHING IS WRONG IN OBJ CALCULATIONS, CLEARLY...
    #print('coeff+delta', optimizer_1.obj(newcoeff))
    #print('coeff+delta', optimizer_1.obj(newcoeff))
    #print('coeff', optimizer_1.obj(coeff))
    #print('coeff', optimizer_1.obj(coeff))
    fd_jac_params_1 = (optimizer_1.obj(newcoeff) - optimizer_1.obj(coeff))/eps
    jac_params_1 = optimizer_1.jac(coeff)
    print(jac_params_1)
    print(jac_params_1.dot(delta_coeff_arr))
    print(fd_jac_params_1)

    assert(np.count_nonzero(jac_params_1.dot(delta_coeff_arr)) == 0)
#THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    #assert(np.allclose(jac_params_1.dot(delta_coeff_arr), fd_jac_params_1))


    #check gradients
    parameters['hyperviscosity']['c0'] = 0.05
    parameters['hyperviscosity']['s'] = 3.0
    model.set_coeffs(parameters, coeff_sub)
    newcoeff.assign(coeff + eps*delta_coeff)

    fd_jac_params_1 = (optimizer_1.obj(newcoeff) - optimizer_1.obj(coeff))/eps
    jac_params_1 = optimizer_1.jac(coeff)
    print(jac_params_1)
    print(jac_params_1.dot(delta_coeff_arr))
    print(fd_jac_params_1)

#THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params_1.dot(delta_coeff_arr), fd_jac_params_1))

# def test_multiple_timestep_gradient():
#
#
#     parameters = get_parameters('tests/tswe.cfg')
#     logger = EmptyLogger()
#     model = get_model(parameters, logger, has_dynamics_statistics=False)
#     coeffs = model.get_coeff_var('coeff')
#     coeff, coeff_sub, coeff_split = coeffs
#     timestepper = get_timestepper(parameters, model, logger, coeffs)
#     dt = parameters['timestepping']['dt']
#
#     x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
#     t0 = model.get_t_var()
#     model.initialize(x0_sub, t0)
#     model.set_coeffs(parameters, coeff_sub)
#
# #ADD COEFF SETTING HERE SOMEHOW?
#     #xns, xn_subs, tns = create_states(model, 10)
#     #compute_states(xns, xn_subs, tns, model, timestepper, 10, dt, x0, t0)
#
#     xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 10, dt, x0, t0)
#     xns2, xn_subs2, steps2, tns2 = compute_state_block(model, timestepper, 2, 5, dt, x0, t0)
#     xns5, xn_subs5, steps5, tns5 = compute_state_block(model, timestepper, 5, 2, dt, x0, t0)
#
#
#     objective_1 = L2Objective(xns1, tns1, coeff, 1, model.spaces.dx)
#     objective_2 = L2Objective(xns2, tns2, coeff, 2, model.spaces.dx)
#     objective_5 = L2Objective(xns5, tns5, coeff, 5, model.spaces.dx)
#
#     optimizer_1 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_1, dt)
#     optimizer_2 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_2, dt)
#     optimizer_5 = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective_5, dt)
#
#
#     eps = 0.00001
#     delta_coeff, _, _ = model.get_coeff_var('coeff')
# #SET DELTA SOMEHOW!
#
#     #check zero gradients at optimality
# #BROKEN HERE DOWN- NEED TO CREATE L2OBJECTIVES AND OPTIMIZATION PROBLEMS SOMEHOW...
#     fd_jac_params_1 = (optimizer_1.obj(coeff+eps*delta_coeff) - optimizer_1.obj(coeff))/eps
#     fd_jac_params_2 = (optimizer_2.obj(coeff+eps*delta_coeff) - optimizer_2.obj(coeff))/eps
#     fd_jac_params_5 = (optimizer_5.obj(coeff+eps*delta_coeff) - optimizer_5.obj(coeff))/eps
#     jac_params_1 = optimizer_1.jac(coeff)
#     jac_params_2 = optimizer_2.jac(coeff)
#     jac_params_5 = optimizer_5.jac(coeff)
#
#     assert(np.count_nonzero(jac_params_1.dot(delta_coeff_arr)) == 0)
#     assert(np.count_nonzero(jac_params_2.dot(delta_coeff_arr)) == 0)
#     assert(np.count_nonzero(jac_params_5.dot(delta_coeff_arr)) == 0)
#
#     #THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
#     assert(np.allclose(jac_params_1.dot(delta_coeff_arr), fd_jac_params_1))
#     assert(np.allclose(jac_params_2.dot(delta_coeff_arr), fd_jac_params_2))
#     assert(np.allclose(jac_params_5.dot(delta_coeff_arr), fd_jac_params_5))
#
#
#     #check gradients
#     parameters['hyperviscosity']['c0'] = 0.05
#     parameters['hyperviscosity']['s'] = 3.0
#     model.set_coeffs(parameters, coeff_sub)
#
#     jac_params_1 = optimizer_1.jac(coeff)
#     fd_jac_params_1 = (optimizer_1.obj(coeff+eps*delta_coeff) - optimizer_1.obj(coeff))/eps
#     jac_params_2 = optimizer_2.jac(coeff)
#     fd_jac_params_2 = (optimizer_2.obj(coeff+eps*delta_coeff) - optimizer_2.obj(coeff))/eps
#     jac_params_5 = optimizer_5.jac(coeff)
#     fd_jac_params_5 = (optimizer_5.obj(coeff+eps*delta_coeff) - optimizer_5.obj(coeff))/eps
#
#     #THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
#     assert(np.allclose(jac_params_1.dot(delta_coeff_arr), fd_jac_params_1))
#     assert(np.allclose(jac_params_2.dot(delta_coeff_arr), fd_jac_params_2))
#     assert(np.allclose(jac_params_5.dot(delta_coeff_arr), fd_jac_params_5))
#
#     #taylor_remainder_check(params, delta_params, lambda p: compute_state(timestepper, dynamics, nsteps, p, t0, x0, dt))
