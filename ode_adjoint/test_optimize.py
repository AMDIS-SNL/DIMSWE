from adjoint_timesteppers import RK4, Euler
from ode_optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dynamics import LotkaVolterra, LogisticEquation
import numpy as np
import matplotlib.pyplot as plt

import pytest



def test_optimize_params():
    nsteps = 500
    dt = 0.01
    t0 = 0
    x0 = np.array([2.,1.]) #np.array([5.,])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics) #Euler RK4
    nblocks = 10
    params =  np.array([1.5,1.,0.5,2.]) #np.array([0.8,50.])

    #Generate FOM data
    xns, steps, tns = timestepper.compute_state_block(nblocks, nsteps//nblocks, t0, x0, dt, params)

    #optimize
    objective = L2Objective(xns, steps, tns, dynamics.get_x_size(), dynamics.get_param_size())
    params0 = np.array([2.,1.5,1.,1.])
    #params0 = np.array([0.3,30.])
    optimizer = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)
    new_params = optimizer.optimize(params0, opt_type='params')
    print(params - new_params)
    assert(np.allclose(params, new_params))


def test_optimize_ic():
    nsteps = 500
    dt = 0.01
    t0 = 0
    x0 = np.array([2.,1.]) #np.array([5.,])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics) #Euler RK4
    nblocks = 2



    #Generate FOM data
    params =  np.array([1.5,1.,0.5,2.]) #np.array([0.8,50.])
    xns, steps, tns = timestepper.compute_state_block(nblocks, nsteps//nblocks, t0, x0, dt, params)

    ics = []
    delta_ics = []
    for i in range(nblocks):
        ics.append(xns[i][0,:])
        delta_ics.append(xns[i][0,:] * 0.05)
    ics = np.array(ics)
    delta_ics = np.array(delta_ics)

    objective = L2Objective(xns, steps, tns, dynamics.get_x_size(), dynamics.get_param_size())
    params0 = np.array([2.,1.5,1.,1.])
    #params0 = np.array([0.3,30.])
    optimizer = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)
    new_ics = optimizer.optimize(np.ravel(ics+delta_ics), opt_type='ics', params0=params)
    #print(np.ravel(ics))
    #print(new_ics)
    print(np.ravel(ics)- new_ics)
    assert(np.allclose(np.ravel(ics), new_ics))

@pytest.mark.xfail
def test_optimize_params_plus_ic():
    nsteps = 500
    dt = 0.01
    t0 = 0
    x0 = np.array([2.,1.]) #np.array([5.,])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics) #Euler RK4
    nblocks = 5
    params0 =  np.array([1.5,1.,0.5,2.]) #np.array([0.8,50.])

    #Generate FOM data
    xns, steps, tns = timestepper.compute_state_block(nblocks, nsteps//nblocks, t0, x0, dt, params0)

    ics = []
    delta_ics = []
    for i in range(nblocks):
        ics.append(xns[i][0,:])
        delta_ics.append(xns[i][0,:] * 0.1)
    ics = np.array(ics)
    delta_ics = np.array(delta_ics)

    objective = L2Objective(xns, steps, tns, dynamics.get_x_size(), dynamics.get_param_size())
    params = np.array([2.,1.5,1.,1.])
    optimizer = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)
    #new_params_ics = optimizer.optimize(np.hstack([params0, np.ravel(ics)]), opt_type='params+ics')
    new_params_ics = optimizer.optimize(np.hstack([params, np.ravel(ics+delta_ics)]), opt_type='params+ics')
    new_ics = new_params_ics[params.shape[0]:]
    new_params = new_params_ics[:params.shape[0]]


    # xn,t = timestepper.compute_state(nsteps, params0, t0, x0, dt)
    # plt.figure()
    # for k in range(dynamics.get_x_size()):
    #     plt.plot(t,xn[:,k])
    # plt.savefig('exact.png')
    #
    # xn,t = timestepper.compute_state(nsteps, new_params, t0, new_ics[:dynamics.get_x_size()], dt)
    # plt.figure()
    # for k in range(dynamics.get_x_size()):
    #     plt.plot(t,xn[:,k])
    # plt.savefig('learned.png')

    print(x0, new_ics[:dynamics.get_x_size()], x0 - new_ics[:dynamics.get_x_size()])
    print(params0, new_params, params0 - new_params)

    #print(new_params_ics)
    #print(new_params_ics - np.hstack([params0, np.ravel(ics)]))
    assert(np.allclose(np.ravel(ics), new_ics))
    assert(np.allclose(params0, new_params))
