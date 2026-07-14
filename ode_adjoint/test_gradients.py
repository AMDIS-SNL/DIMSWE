from adjoint_timesteppers import RK4, Euler
from ode_optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dynamics import LotkaVolterra, LogisticEquation
import matplotlib.pyplot as plt
import numpy as np

def _test_gradients(dynamics,x,params,delta_params,delta_x,eps,t):

    jac_x = dynamics.jac_x(x,t,params)
    assert(np.array_equal(jac_x.T,dynamics.jacT_x(x,t,params)))
    for epsilon in [eps, eps/2., eps/4., eps/8., eps/16., eps/32.]:
        fd_jac_x = (dynamics.rhs(x+eps*delta_x,t,params) - dynamics.rhs(x,t,params))/eps

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
        assert(np.allclose(jac_x.dot(delta_x), fd_jac_x ))

    jac_params = dynamics.jac_params(x,t,params)
    assert(np.array_equal(jac_params.T,dynamics.jacT_params(x,t,params)))
    fd_jac_params = (dynamics.rhs(x,t,params+eps*delta_params) - dynamics.rhs(x,t,params))/eps

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params.dot(delta_params), fd_jac_params ))


def test_dynamics_gradients():

    dynamics = LogisticEquation() #LogisticEquation LotkaVolterra
    params = np.array([1.5,1.2])
    x = np.array([1.,])
    t = None
    delta_params = np.array([0.1,0.2])
    delta_x = np.array([0.1,])
    eps = 0.00001
    _test_gradients(dynamics,x,params,delta_params,delta_x,eps,t)

    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    params = np.array([1.5,1.2,0.8,1.3])
    x = np.array([2.,1.])
    t = None
    delta_params = np.array([0.1,0.2,0.15,0.12])
    delta_x = np.array([0.1,0.15])
    eps = 0.00001
    _test_gradients(dynamics,x,params,delta_params,delta_x,eps,t)





def test_multi_timestep_gradient_params():

    dt = 0.01
    t0 = 0
    params = np.array([1.5,1.2,0.8,1.3])
    x0 = np.array([2.,1.])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics)

    xns1, steps1, tns1  = timestepper.compute_state_block(10, 1, t0, x0, dt, params)
    xns2, steps2, tns2 = timestepper.compute_state_block(5, 2, t0, x0, dt, params)
    xns5, steps5, tns5 = timestepper.compute_state_block(2, 5, t0, x0, dt, params)
    xns10, steps10, tns10 = timestepper.compute_state_block(1, 10, t0, x0, dt, params)

    params0 = np.array([2.,1.5,1.,1.])

    objective_1 = L2Objective(xns1, steps1, tns1, dynamics.get_x_size(), dynamics.get_param_size())
    objective_2 = L2Objective(xns2, steps2, tns2, dynamics.get_x_size(), dynamics.get_param_size())
    objective_5 = L2Objective(xns5, steps5, tns5, dynamics.get_x_size(), dynamics.get_param_size())
    objective_10 = L2Objective(xns10, steps10, tns10, dynamics.get_x_size(), dynamics.get_param_size())
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_1, dt)
    optimizer_2 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_2, dt)
    optimizer_5 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_5, dt)
    optimizer_10 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_10, dt)

    eps = 0.00001
    delta_params = np.array([0.1,0.2,0.15,0.12])

    #check zero gradients at optimality

    jac_params_1 = optimizer_1.jac(params, None)
    fd_jac_params_1 = (optimizer_1.obj(params+eps*delta_params, None) - optimizer_1.obj(params, None))/eps
    jac_params_2 = optimizer_2.jac(params, None)
    fd_jac_params_2 = (optimizer_2.obj(params+eps*delta_params, None) - optimizer_2.obj(params, None))/eps
    jac_params_5 = optimizer_5.jac(params, None)
    fd_jac_params_5 = (optimizer_5.obj(params+eps*delta_params, None) - optimizer_5.obj(params, None))/eps
    jac_params_10 = optimizer_10.jac(params, None)
    fd_jac_params_10 = (optimizer_10.obj(params+eps*delta_params, None) - optimizer_10.obj(params, None))/eps

    assert(np.count_nonzero(jac_params_1.dot(delta_params)) == 0)
    assert(np.count_nonzero(jac_params_2.dot(delta_params)) == 0)
    assert(np.count_nonzero(jac_params_5.dot(delta_params)) == 0)
    assert(np.count_nonzero(jac_params_10.dot(delta_params)) == 0)

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params_1.dot(delta_params), fd_jac_params_1))
    assert(np.allclose(jac_params_2.dot(delta_params), fd_jac_params_2))
    assert(np.allclose(jac_params_5.dot(delta_params), fd_jac_params_5))
    assert(np.allclose(jac_params_10.dot(delta_params), fd_jac_params_10))


    #check gradients

    jac_params_1 = optimizer_1.jac(params0, None)
    fd_jac_params_1 = (optimizer_1.obj(params0+eps*delta_params, None) - optimizer_1.obj(params0, None))/eps
    jac_params_2 = optimizer_2.jac(params0, None)
    fd_jac_params_2 = (optimizer_2.obj(params0+eps*delta_params, None) - optimizer_2.obj(params0, None))/eps
    jac_params_5 = optimizer_5.jac(params0, None)
    fd_jac_params_5 = (optimizer_5.obj(params0+eps*delta_params, None) - optimizer_5.obj(params0, None))/eps
    jac_params_10 = optimizer_10.jac(params0, None)
    fd_jac_params_10 = (optimizer_10.obj(params0+eps*delta_params, None) - optimizer_10.obj(params0, None))/eps

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params_1.dot(delta_params), fd_jac_params_1))
    assert(np.allclose(jac_params_2.dot(delta_params), fd_jac_params_2))
    assert(np.allclose(jac_params_5.dot(delta_params), fd_jac_params_5))
    assert(np.allclose(jac_params_10.dot(delta_params), fd_jac_params_10))

    #taylor_remainder_check(params, delta_params, lambda p: compute_state(timestepper, dynamics, nsteps, p, t0, x0, dt))

def test_single_timestep_gradient_ic():
    dt = 0.01
    t0 = 0
    params = np.array([1.5,1.2,0.8,1.3])
    x0 = np.array([2.,1.])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics)


    xn, t = timestepper.compute_state(1, params, t0, x0, dt)

    eps = 0.0000001
    delta_ic = np.array([0.1,0.2])

    objective = L2Objective([xn,], [1,], [t,], dynamics.get_x_size(), dynamics.get_param_size())

    optimizer_single = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)


    #check zero gradients at optimality
    ics = np.reshape(x0, (1,2))
    jac_ic = optimizer_single.jac(None, ics, params0=params)
    fd_jac_ic = (optimizer_single.obj(None, ics+eps*delta_ic, params0=params) - optimizer_single.obj(None, ics, params0=params))/eps

    assert(np.count_nonzero(jac_ic.dot(delta_ic)) == 0)
    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    #print(jac_ic.dot(delta_ic))
    #print(fd_jac_ic)
    assert(np.allclose(jac_ic.dot(delta_ic), fd_jac_ic))

    #check gradients
    x0 = np.array([2.3,1.4])
    ics = np.reshape(x0, (1,2))
    jac_ic = optimizer_single.jac(None, ics, params0=params)
    fd_jac_ic = (optimizer_single.obj(None, ics+eps*delta_ic, params0=params) - optimizer_single.obj(None, ics, params0=params))/eps
    #print(jac_ic.dot(delta_ic))
    #print(fd_jac_ic)

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_ic.dot(delta_ic), fd_jac_ic))


def test_single_timestep_gradient_params_and_ic():
    dt = 0.01
    t0 = 0
    params = np.array([1.5,1.2,0.8,1.3])
    x0 = np.array([2.,1.])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics)


    xn, t = timestepper.compute_state(1, params, t0, x0, dt)

    eps = 0.0000001
    delta_ic = np.array([0.1,0.2])
    delta_params = np.array([0.1,0.2,0.15,0.12])
    delta = np.hstack([delta_params, delta_ic])
    objective = L2Objective([xn,], [1,], [t,], dynamics.get_x_size(), dynamics.get_param_size())

    optimizer_single = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)


    #check zero gradients at optimality
    ics = np.reshape(x0, (1,2))
    jac = optimizer_single.jac(params, ics)
    fd_jac = (optimizer_single.obj(params+eps*delta_params, ics+eps*delta_ic) - optimizer_single.obj(params, ics))/eps

    assert(np.count_nonzero(jac.dot(delta)) == 0)
    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    #print(jac_ic.dot(delta_ic))
    #print(fd_jac_ic)
    assert(np.allclose(jac.dot(delta), fd_jac))

    #check gradients
    x0 = np.array([2.3,1.4])
    params = np.array([2.,1.5,1.,1.])
    ics = np.reshape(x0, (1,2))
    jac = optimizer_single.jac(params, ics)
    fd_jac = (optimizer_single.obj(params+eps*delta_params, ics+eps*delta_ic) - optimizer_single.obj(params, ics))/eps
    #print(jac_ic.dot(delta_ic))
    #print(fd_jac_ic)

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac.dot(delta), fd_jac))

def test_multi_timestep_gradient_ic_params():
    dt = 0.01
    t0 = 0
    params = np.array([1.5,1.2,0.8,1.3])
    x0 = np.array([2.,1.])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics)

    xns1, steps1, tns1  = timestepper.compute_state_block(10, 1, t0, x0, dt, params)
    xns2, steps2, tns2 = timestepper.compute_state_block(5, 2, t0, x0, dt, params)
    xns5, steps5, tns5 = timestepper.compute_state_block(2, 5, t0, x0, dt, params)
    xns10, steps10, tns10 = timestepper.compute_state_block(1, 10, t0, x0, dt, params)

    objective_1 = L2Objective(xns1, steps1, tns1, dynamics.get_x_size(), dynamics.get_param_size())
    objective_2 = L2Objective(xns2, steps2, tns2, dynamics.get_x_size(), dynamics.get_param_size())
    objective_5 = L2Objective(xns5, steps5, tns5, dynamics.get_x_size(), dynamics.get_param_size())
    objective_10 = L2Objective(xns10, steps10, tns10, dynamics.get_x_size(), dynamics.get_param_size())
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_1, dt)
    optimizer_2 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_2, dt)
    optimizer_5 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_5, dt)
    optimizer_10 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_10, dt)

    eps = 0.00000001
    ics10 = np.array([xns1[0][0,:],])
    ics5 = np.array([xns5[0][0,:],xns5[1][0,:]])
    ics2 = np.array([xns2[0][0,:],xns2[1][0,:],xns2[2][0,:],xns2[3][0,:],xns2[4][0,:]])
    ics1 = np.array([xns1[0][0,:],xns1[1][0,:],xns1[2][0,:],xns1[3][0,:],xns1[4][0,:],xns1[5][0,:],xns1[6][0,:],xns1[7][0,:],xns1[8][0,:],xns1[9][0,:]])
    delta_ic10 = np.array([[0.1,0.2],])
    delta_ic5 = np.array([[0.1,0.2],[0.1,0.2]])
    delta_ic2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
    delta_ic1 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
    delta_params = np.array([0.1,0.2,0.15,0.12])
    delta_10 = np.hstack([delta_params, np.ravel(delta_ic10)])
    delta_5 = np.hstack([delta_params, np.ravel(delta_ic5)])
    delta_2 = np.hstack([delta_params, np.ravel(delta_ic2)])
    delta_1 = np.hstack([delta_params, np.ravel(delta_ic1)])
    #print(ics1.shape)
    #print(ics2.shape)
    #print(ics5.shape)
    #check zero gradients at optimality

    jac_1 = optimizer_1.jac(params, ics1)
    fd_jac_1 = (optimizer_1.obj(params+eps*delta_params, ics1+eps*delta_ic1) - optimizer_1.obj(params, ics1))/eps
    jac_2 = optimizer_2.jac(params, ics2)
    fd_jac_2 = (optimizer_2.obj(params+eps*delta_params, ics2+eps*delta_ic2) - optimizer_2.obj(params, ics2))/eps
    jac_5 = optimizer_5.jac(params, ics5)
    fd_jac_5 = (optimizer_5.obj(params+eps*delta_params, ics5+eps*delta_ic5) - optimizer_5.obj(params, ics5))/eps
    jac_10 = optimizer_10.jac(params, ics10)
    fd_jac_10 = (optimizer_10.obj(params+eps*delta_params, ics10+eps*delta_ic10) - optimizer_10.obj(params, ics10))/eps

    #print(jac_ic_1.dot(np.ravel(delta_ic1)))
    #print(fd_jac_ic_1)
    assert(np.count_nonzero(jac_1.dot(np.ravel(delta_1))) == 0)
    assert(np.count_nonzero(jac_2.dot(np.ravel(delta_2))) == 0)
    assert(np.count_nonzero(jac_5.dot(np.ravel(delta_5))) == 0)
    assert(np.count_nonzero(jac_10.dot(np.ravel(delta_10))) == 0)

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_1.dot(np.ravel(delta_1)), fd_jac_1))
    assert(np.allclose(jac_2.dot(np.ravel(delta_2)), fd_jac_2))
    assert(np.allclose(jac_5.dot(np.ravel(delta_5)), fd_jac_5))
    assert(np.allclose(jac_10.dot(np.ravel(delta_10)), fd_jac_10))


    #check gradients
    initial_delta_ic10 = np.array([[0.1,0.2],])
    initial_delta_ic5 = np.array([[0.1,0.2],[0.1,0.2]])
    initial_delta_ic2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
    initial_delta_ic1 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])

    ics10 = initial_delta_ic10 + np.array([xns1[0][0,:],])
    ics5 = initial_delta_ic5 + np.array([xns5[0][0,:],xns5[1][0,:]])
    ics2 = initial_delta_ic2 + np.array([xns2[0][0,:],xns2[1][0,:],xns2[2][0,:],xns2[3][0,:],xns2[4][0,:]])
    ics1 = initial_delta_ic1 + np.array([xns1[0][0,:],xns1[1][0,:],xns1[2][0,:],xns1[3][0,:],xns1[4][0,:],xns1[5][0,:],xns1[6][0,:],xns1[7][0,:],xns1[8][0,:],xns1[9][0,:]])
    params = np.array([2.,1.5,1.,1.])

    jac_1 = optimizer_1.jac(params, ics1)
    fd_jac_1 = (optimizer_1.obj(params+eps*delta_params, ics1+eps*delta_ic1) - optimizer_1.obj(params, ics1))/eps
    jac_2 = optimizer_2.jac(params, ics2)
    fd_jac_2 = (optimizer_2.obj(params+eps*delta_params, ics2+eps*delta_ic2) - optimizer_2.obj(params, ics2))/eps
    jac_5 = optimizer_5.jac(params, ics5)
    fd_jac_5 = (optimizer_5.obj(params+eps*delta_params, ics5+eps*delta_ic5) - optimizer_5.obj(params, ics5))/eps
    jac_10 = optimizer_10.jac(params, ics10)
    fd_jac_10 = (optimizer_10.obj(params+eps*delta_params, ics10+eps*delta_ic10) - optimizer_10.obj(params, ics10))/eps

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_1.dot(np.ravel(delta_1)), fd_jac_1))
    assert(np.allclose(jac_2.dot(np.ravel(delta_2)), fd_jac_2))
    assert(np.allclose(jac_5.dot(np.ravel(delta_5)), fd_jac_5))
    assert(np.allclose(jac_10.dot(np.ravel(delta_10)), fd_jac_10))
    #print(jac_1.dot(np.ravel(delta_1)), fd_jac_1)
    #print(jac_2.dot(np.ravel(delta_2)), fd_jac_2)
    #print(jac_5.dot(np.ravel(delta_5)), fd_jac_5)
    #print(jac_10.dot(np.ravel(delta_10)), fd_jac_10)


def test_multi_timestep_gradient_ic():
    dt = 0.01
    t0 = 0
    params = np.array([1.5,1.2,0.8,1.3])
    x0 = np.array([2.,1.])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics)

    xns1, steps1, tns1  = timestepper.compute_state_block(10, 1, t0, x0, dt, params)
    xns2, steps2, tns2 = timestepper.compute_state_block(5, 2, t0, x0, dt, params)
    xns5, steps5, tns5 = timestepper.compute_state_block(2, 5, t0, x0, dt, params)
    xns10, steps10, tns10 = timestepper.compute_state_block(1, 10, t0, x0, dt, params)

    objective_1 = L2Objective(xns1, steps1, tns1, dynamics.get_x_size(), dynamics.get_param_size())
    objective_2 = L2Objective(xns2, steps2, tns2, dynamics.get_x_size(), dynamics.get_param_size())
    objective_5 = L2Objective(xns5, steps5, tns5, dynamics.get_x_size(), dynamics.get_param_size())
    objective_10 = L2Objective(xns10, steps10, tns10, dynamics.get_x_size(), dynamics.get_param_size())
    optimizer_1 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_1, dt)
    optimizer_2 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_2, dt)
    optimizer_5 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_5, dt)
    optimizer_10 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_10, dt)

    eps = 0.00000001
    ics10 = np.array([xns1[0][0,:],])
    ics5 = np.array([xns5[0][0,:],xns5[1][0,:]])
    ics2 = np.array([xns2[0][0,:],xns2[1][0,:],xns2[2][0,:],xns2[3][0,:],xns2[4][0,:]])
    ics1 = np.array([xns1[0][0,:],xns1[1][0,:],xns1[2][0,:],xns1[3][0,:],xns1[4][0,:],xns1[5][0,:],xns1[6][0,:],xns1[7][0,:],xns1[8][0,:],xns1[9][0,:]])
    delta_ic10 = np.array([[0.1,0.2],])
    delta_ic5 = np.array([[0.1,0.2],[0.1,0.2]])
    delta_ic2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
    delta_ic1 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])

    #print(ics1.shape)
    #print(ics2.shape)
    #print(ics5.shape)
    #check zero gradients at optimality

    jac_ic_1 = optimizer_1.jac(None, ics1, params0=params)
    fd_jac_ic_1 = (optimizer_1.obj(None, ics1+eps*delta_ic1, params0=params) - optimizer_1.obj(None, ics1, params0=params))/eps
    jac_ic_2 = optimizer_2.jac(None, ics2, params0=params)
    fd_jac_ic_2 = (optimizer_2.obj(None, ics2+eps*delta_ic2, params0=params) - optimizer_2.obj(None, ics2, params0=params))/eps
    jac_ic_5 = optimizer_5.jac(None, ics5, params0=params)
    fd_jac_ic_5 = (optimizer_5.obj(None, ics5+eps*delta_ic5, params0=params) - optimizer_5.obj(None, ics5, params0=params))/eps
    jac_ic_10 = optimizer_10.jac(None, ics10, params0=params)
    fd_jac_ic_10 = (optimizer_10.obj(None, ics10+eps*delta_ic10, params0=params) - optimizer_10.obj(None, ics10, params0=params))/eps

    #print(jac_ic_1.dot(np.ravel(delta_ic1)))
    #print(fd_jac_ic_1)
    assert(np.count_nonzero(jac_ic_1.dot(np.ravel(delta_ic1))) == 0)
    assert(np.count_nonzero(jac_ic_2.dot(np.ravel(delta_ic2))) == 0)
    assert(np.count_nonzero(jac_ic_5.dot(np.ravel(delta_ic5))) == 0)
    assert(np.count_nonzero(jac_ic_10.dot(np.ravel(delta_ic10))) == 0)

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_ic_1.dot(np.ravel(delta_ic1)), fd_jac_ic_1))
    assert(np.allclose(jac_ic_2.dot(np.ravel(delta_ic2)), fd_jac_ic_2))
    assert(np.allclose(jac_ic_5.dot(np.ravel(delta_ic5)), fd_jac_ic_5))
    assert(np.allclose(jac_ic_10.dot(np.ravel(delta_ic10)), fd_jac_ic_10))


    #check gradients
    initial_delta_ic10 = np.array([[0.1,0.2],])
    initial_delta_ic5 = np.array([[0.1,0.2],[0.1,0.2]])
    initial_delta_ic2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
    initial_delta_ic1 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])

    ics10 = initial_delta_ic10 + np.array([xns1[0][0,:],])
    ics5 = initial_delta_ic5 + np.array([xns5[0][0,:],xns5[1][0,:]])
    ics2 = initial_delta_ic2 + np.array([xns2[0][0,:],xns2[1][0,:],xns2[2][0,:],xns2[3][0,:],xns2[4][0,:]])
    ics1 = initial_delta_ic1 + np.array([xns1[0][0,:],xns1[1][0,:],xns1[2][0,:],xns1[3][0,:],xns1[4][0,:],xns1[5][0,:],xns1[6][0,:],xns1[7][0,:],xns1[8][0,:],xns1[9][0,:]])

    jac_ic_1 = optimizer_1.jac(None, ics1, params0=params)
    fd_jac_ic_1 = (optimizer_1.obj(None, ics1+eps*delta_ic1, params0=params) - optimizer_1.obj(None, ics1, params0=params))/eps
    jac_ic_2 = optimizer_2.jac(None, ics2, params0=params)
    fd_jac_ic_2 = (optimizer_2.obj(None, ics2+eps*delta_ic2, params0=params) - optimizer_2.obj(None, ics2, params0=params))/eps
    jac_ic_5 = optimizer_5.jac(None, ics5, params0=params)
    fd_jac_ic_5 = (optimizer_5.obj(None, ics5+eps*delta_ic5, params0=params) - optimizer_5.obj(None, ics5, params0=params))/eps
    jac_ic_10 = optimizer_10.jac(None, ics10, params0=params)
    fd_jac_ic_10 = (optimizer_10.obj(None, ics10+eps*delta_ic10, params0=params) - optimizer_10.obj(None, ics10, params0=params))/eps

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_ic_1.dot(np.ravel(delta_ic1)), fd_jac_ic_1))
    assert(np.allclose(jac_ic_2.dot(np.ravel(delta_ic2)), fd_jac_ic_2))
    assert(np.allclose(jac_ic_5.dot(np.ravel(delta_ic5)), fd_jac_ic_5))
    assert(np.allclose(jac_ic_10.dot(np.ravel(delta_ic10)), fd_jac_ic_10))



def test_single_timestep_gradient_params():

    dt = 0.01
    t0 = 0
    params = np.array([1.5,1.2,0.8,1.3])
    x0 = np.array([2.,1.])
    dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
    timestepper = RK4(dynamics)


    xn, t = timestepper.compute_state(1, params, t0, x0, dt)

    eps = 0.00001
    delta_params = np.array([0.1,0.2,0.15,0.12])

    objective = L2Objective([xn,], [1,], [t,], dynamics.get_x_size(), dynamics.get_param_size())

    optimizer_single = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)


    #check zero gradients at optimality

    jac_params = optimizer_single.jac(params, None)
    fd_jac_params = (optimizer_single.obj(params+eps*delta_params, None) - optimizer_single.obj(params, None))/eps

    assert(np.count_nonzero(jac_params.dot(delta_params)) == 0)
    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params.dot(delta_params), fd_jac_params ))

    #check gradients
    params0 = np.array([2.,1.5,1.,1.])

    jac_params = optimizer_single.jac(params0, None)
    fd_jac_params = (optimizer_single.obj(params0+eps*delta_params, None) - optimizer_single.obj(params0, None))/eps
    #print(jac_params.dot(delta_params))
    #print(fd_jac_params)

    #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
    assert(np.allclose(jac_params.dot(delta_params), fd_jac_params ))

    #taylor_remainder_check(params, delta_params, lambda p: compute_state(timestepper, dynamics, nsteps, p, t0, x0, dt))












def taylor_remainder_check(
    p0,
    direction,
    solve_state,
    objective,
    adjoint_gradient,
    step_sizes=None,
    verbose=True,
):

    dp = direction / np.linalg.norm(direction)

    if step_sizes is None:
        step_sizes = np.logspace(-1, -8, 8)

    u0, _ = solve_state(p0)
    j0 = objective(u0, p0)
    g = adjoint_gradient(p0)
    gdir = np.dot(g.ravel(), dp.ravel())

    results = []

    for eps in step_sizes:
        p = p0 + eps * dp
        u = solve_state(p)
        j = objective(u, p)

        remainder = abs(j - j0 - eps * gdir)
        results.append((eps, remainder))

    if verbose:
        print(f"{'h':>12} {'Taylor remainder':>20}")
        print("-" * 35)
        for h, remainder in results:
            print(f"{h:12.3e} {remainder:20.12e}")

    return results


import numpy as np


def taylor_remainder_check_rates(
    m,
    direction,
    solve_state,
    objective,
    adjoint_gradient,
    step_sizes=None,
    verbose=True,
):
    """
    Taylor remainder check for an adjoint gradient, including observed rates.

    Parameters
    ----------
    m : ndarray
        Parameter vector.
    direction : ndarray
        Perturbation direction.
    solve_state : callable
        u = solve_state(m)
    objective : callable
        J = objective(u, m)
    adjoint_gradient : callable
        g = adjoint_gradient(m)
    step_sizes : array-like, optional
        Sequence of step sizes, ideally decreasing geometrically.
    verbose : bool
        If True, print a results table.

    Returns
    -------
    results : list of dict
        Each entry contains:
            h
            J(m+h p)
            remainder
            rate
    """
    m = np.asarray(m, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if m.shape != direction.shape:
        raise ValueError("m and direction must have the same shape")

    norm_p = np.linalg.norm(direction)
    if norm_p == 0:
        raise ValueError("direction must be nonzero")

    p = direction / norm_p

    if step_sizes is None:
        step_sizes = 0.5 ** np.arange(1, 11)  # 1/2, 1/4, ..., 1/1024

    step_sizes = np.asarray(step_sizes, dtype=float)

    if np.any(step_sizes <= 0):
        raise ValueError("step_sizes must be positive")

    # Sort from large to small for cleaner rate computation
    step_sizes = np.sort(step_sizes)[::-1]

    u0 = solve_state(m)
    j0 = objective(u0, m)

    g = np.asarray(adjoint_gradient(m), dtype=float)
    gdir = np.dot(g.ravel(), p.ravel())

    results = []

    for h in step_sizes:
        mh = m + h * p
        uh = solve_state(mh)
        jh = objective(uh, mh)

        remainder = abs(jh - j0 - h * gdir)

        results.append({
            "h": h,
            "Jmh": jh,
            "remainder": remainder,
            "rate": np.nan,
        })

    # Compute observed convergence rates
    for i in range(len(results) - 1):
        r1 = results[i]["remainder"]
        r2 = results[i + 1]["remainder"]
        h1 = results[i]["h"]
        h2 = results[i + 1]["h"]

        if r1 > 0 and r2 > 0:
            rate = np.log(r1 / r2) / np.log(h1 / h2)
        else:
            rate = np.nan

        results[i + 1]["rate"] = rate

    if verbose:
        print(f"{'h':>12} {'remainder':>20} {'observed rate':>16}")
        print("-" * 52)
        for r in results:
            rate_str = f"{r['rate']:.6f}" if np.isfinite(r["rate"]) else "   ---"
            print(f"{r['h']:12.3e} {r['remainder']:20.12e} {rate_str:>16}")

    return results
