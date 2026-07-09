# import numpy as np
#
#
# def test_multi_timestep_gradient():
#
#     dt = SOMETHING
#     t0 = SOMETHING
#     params = SOMETHING
#     x0 = SOMETHING
#     dynamics = SOMETHING()
#     timestepper = SOMETHING(dynamics)
#
#     timestepper.set_coeff(params)
#
#     xns1, steps1, tns1  = timestepper.compute_state_block(10, 1, t0, x0, dt)
#     xns2, steps2, tns2 = timestepper.compute_state_block(5, 2, t0, x0, dt)
#     xns5, steps5, tns5 = timestepper.compute_state_block(2, 5, t0, x0, dt)
#
#     params0 = SOMETHING
#
# #FIX THESE UP
#     objective_1 = L2Objective(xns1, steps1, tns1, dynamics.get_x_size(), dynamics.get_param_size())
#     objective_2 = L2Objective(xns2, steps2, tns2, dynamics.get_x_size(), dynamics.get_param_size())
#     objective_5 = L2Objective(xns5, steps5, tns5, dynamics.get_x_size(), dynamics.get_param_size())
#     optimizer_1 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_1, dt)
#     optimizer_2 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_2, dt)
#     optimizer_5 = Lagrangian_ODEConstrainedOptimization(timestepper, objective_5, dt)
#
#     eps = 0.00001
#     delta_params = SOMETHING
#
#     #check zero gradients at optimality
#
#     jac_params_1 = optimizer_1.jac(params)
#     fd_jac_params_1 = (optimizer_1.obj(params+eps*delta_params) - optimizer_1.obj(params))/eps
#     jac_params_2 = optimizer_2.jac(params)
#     fd_jac_params_2 = (optimizer_2.obj(params+eps*delta_params) - optimizer_2.obj(params))/eps
#     jac_params_5 = optimizer_5.jac(params)
#     fd_jac_params_5 = (optimizer_5.obj(params+eps*delta_params) - optimizer_5.obj(params))/eps
#
#     assert(np.count_nonzero(jac_params_1.dot(delta_params)) == 0)
#     assert(np.count_nonzero(jac_params_2.dot(delta_params)) == 0)
#     assert(np.count_nonzero(jac_params_5.dot(delta_params)) == 0)
#
#     #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
#     assert(np.allclose(jac_params_1.dot(delta_params), fd_jac_params_1))
#     assert(np.allclose(jac_params_2.dot(delta_params), fd_jac_params_2))
#     assert(np.allclose(jac_params_5.dot(delta_params), fd_jac_params_5))
#
#
#     #check gradients
#
#     jac_params_1 = optimizer_1.jac(params0)
#     fd_jac_params_1 = (optimizer_1.obj(params0+eps*delta_params) - optimizer_1.obj(params0))/eps
#     jac_params_2 = optimizer_2.jac(params0)
#     fd_jac_params_2 = (optimizer_2.obj(params0+eps*delta_params) - optimizer_2.obj(params0))/eps
#     jac_params_5 = optimizer_5.jac(params0)
#     fd_jac_params_5 = (optimizer_5.obj(params0+eps*delta_params) - optimizer_5.obj(params0))/eps
#
#     #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
#     assert(np.allclose(jac_params_1.dot(delta_params), fd_jac_params_1))
#     assert(np.allclose(jac_params_2.dot(delta_params), fd_jac_params_2))
#     assert(np.allclose(jac_params_5.dot(delta_params), fd_jac_params_5))
#
#     #taylor_remainder_check(params, delta_params, lambda p: compute_state(timestepper, dynamics, nsteps, p, t0, x0, dt))
#
#
# def test_single_timestep_gradient():
#
#     dt = SOMETHING
#     t0 = SOMETHING
#     params = SOMETHING
#     x0 = SOMETHING
#     dynamics = SOMETHING()
#     timestepper = SOMETHING(dynamics)
#
#     timestepper.set_coeff(params)
#
# #FIX
#     xn, t = timestepper.compute_state(1, t0, x0, dt)
#
#     eps = 0.00001
#     delta_params = SOMETHING
#
# #FIX
#     objective = L2Objective([xn,], [1,], [t,], dynamics.get_x_size(), dynamics.get_param_size())
#     optimizer_single = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)
#
#
#     #check zero gradients at optimality
#
#     jac_params = optimizer_single.jac(params)
#     fd_jac_params = (optimizer_single.obj(params+eps*delta_params) - optimizer_single.obj(params))/eps
#
#     assert(np.count_nonzero(jac_params.dot(delta_params)) == 0)
#     #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
#     assert(np.allclose(jac_params.dot(delta_params), fd_jac_params ))
#
#     #check gradients
#     params0 = SOMETHING
#
#     jac_params = optimizer_single.jac(params0)
#     fd_jac_params = (optimizer_single.obj(params0+eps*delta_params) - optimizer_single.obj(params0))/eps
#     #print(jac_params.dot(delta_params))
#     #print(fd_jac_params)
#
#     #EVENTUALLY THIS SHOULD BE A CONVERGENCE TEST WITH EPS DECREASING
#     assert(np.allclose(jac_params.dot(delta_params), fd_jac_params ))
#
#     #taylor_remainder_check(params, delta_params, lambda p: compute_state(timestepper, dynamics, nsteps, p, t0, x0, dt))
