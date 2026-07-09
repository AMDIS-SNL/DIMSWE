#import matplotlib.pyplot as plt
import numpy as np
from dimswe.optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import Logger
from dimswe.parameters import get_parameters

def create_states(model, nsteps):
    xns = []
    xn_subs = []
    tns = []
    for i in range(nsteps+1):
        t = model.get_t_var()
        xn, xn_sub, x_split = model.get_full_var('x'+str(i), split_x_and_aux=True)
        xns.append(xn)
        xn_subs.append(xn_sub)
        tns.append(t)
    return xns, xn_subs, tns

def compute_states(model, timestepper, nsteps, dt, x0, t0):
    xns, xn_subs, tns = create_states(model, nsteps)
    model.restart(xns[0], x0, tns[0], t0)
    for n in range(nsteps):
        timestepper.take_forward_step(xns[n+1], xn_subs[n], xns[n], tns[n], dt)
        tns[n+1].assign(tns[n] + dt)
    return xns, xn_subs, tns


def compute_state_block(model, timestepper, nblocks, nsteps, dt, x0, t0):
    xns = []
    xn_subs = []
    tns = []
    steps = []
    for k in range(nblocks):
        xn, xn_sub, tn = create_states(model, nsteps)
        if (k==0):
            xn, xn_sub, t = compute_states(model, timestepper, nsteps, dt, x0, t0)
        else:
            xn, xn_sub, t = compute_states(model, timestepper, nsteps, dt, xns[k-1][-1], tns[k-1][-1])
        xns.append(xn[::nsteps])
        xn_subs.append(xn_sub[::nsteps])
        tns.append(t[::nsteps])
        steps.append(nsteps)
    return xns, xn_subs, steps, tns

def test_timestep_gradient():


    parameters = get_parameters('mtswe.cfg')
    logger = Logger(parameters)
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    coeffs = model.get_coeff_var('coeff')
    coeff, coeff_sub, coeff_split = coeffs
    timestepper = get_timestepper(parameters, model, logger, coeffs)


    dt = parameters['timestepping']['dt']

    x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_default_coeffs(coeff_sub)

#ADD COEFF SETTING HERE SOMEHOW?

    xns, xn_subs, tns = compute_states(model, timestepper, 10, dt, x0, t0)

    xns1, xn_subs1, steps1, tns1 = compute_state_block(model, timestepper, 1, 10, dt, x0, t0)
    xns2, xn_subs2, steps2, tns2 = compute_state_block(model, timestepper, 2, 5, dt, x0, t0)
    xns5, xn_subs5, steps5, tns5 = compute_state_block(model, timestepper, 5, 2, dt, x0, t0)


    objective_1 = L2Objective(xns1, tns1, coeff)
    objective_2 = L2Objective(xns2, tns2, coeff)
    objective_5 = L2Objective(xns5, tns5, coeff)

    optimizer_1 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_1, dt)
    optimizer_2 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_2, dt)
    optimizer_5 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_5, dt)

#BROKEN HERE DOWN- NEED TO CREATE L2OBJECTIVES AND OPTIMIZATION PROBLEMS SOMEHOW...

    eps = 0.00001
    delta_coeff, _, _ = model.get_coeff_var('coeff')

    #check zero gradients at optimality

    jac_params_1 = optimizer_1.jac(params)
    fd_jac_params_1 = (optimizer_1.obj(params+eps*delta_params) - optimizer_1.obj(params))/eps
    jac_params_2 = optimizer_2.jac(params)
    fd_jac_params_2 = (optimizer_2.obj(params+eps*delta_params) - optimizer_2.obj(params))/eps
    jac_params_5 = optimizer_5.jac(params)
    fd_jac_params_5 = (optimizer_5.obj(params+eps*delta_params) - optimizer_5.obj(params))/eps

    assert(np.count_nonzero(jac_params_1.dot(delta_params)) == 0)
    assert(np.count_nonzero(jac_params_2.dot(delta_params)) == 0)
    assert(np.count_nonzero(jac_params_5.dot(delta_params)) == 0)

    #THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params_1.dot(delta_params), fd_jac_params_1))
    assert(np.allclose(jac_params_2.dot(delta_params), fd_jac_params_2))
    assert(np.allclose(jac_params_5.dot(delta_params), fd_jac_params_5))


    #check gradients

    jac_params_1 = optimizer_1.jac(params0)
    fd_jac_params_1 = (optimizer_1.obj(params0+eps*delta_params) - optimizer_1.obj(params0))/eps
    jac_params_2 = optimizer_2.jac(params0)
    fd_jac_params_2 = (optimizer_2.obj(params0+eps*delta_params) - optimizer_2.obj(params0))/eps
    jac_params_5 = optimizer_5.jac(params0)
    fd_jac_params_5 = (optimizer_5.obj(params0+eps*delta_params) - optimizer_5.obj(params0))/eps

    #THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params_1.dot(delta_params), fd_jac_params_1))
    assert(np.allclose(jac_params_2.dot(delta_params), fd_jac_params_2))
    assert(np.allclose(jac_params_5.dot(delta_params), fd_jac_params_5))

    #taylor_remainder_check(params, delta_params, lambda p: compute_state(timestepper, dynamics, nsteps, p, t0, x0, dt))
