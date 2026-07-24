#import matplotlib.pyplot as plt
import numpy as np
from dimswe.optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import EmptyLogger
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.optimize import compute_state_block, compute_states, create_states
from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function, set_mixed_function_from_flattened_array
from firedrake import norm
from dimswe.plotting import plot_variable

solver_parameters=overall_solver_parameters.copy()
basic_linear_system = { 'ksp_type': 'preonly', 'pc_type' : 'lu'} #'ksp_monitor_true_residual': None}
#basic_linear_system = { 'ksp_type': 'cg', 'pc_type' : 'jacobi'} #'ksp_monitor_true_residual': None}
solver_parameters['erkstage-f'] = basic_linear_system
solver_parameters['erkstage-aux'] = basic_linear_system
solver_parameters['erkstage-mu'] = basic_linear_system
solver_parameters['erkstage-muaux'] = basic_linear_system
solver_parameters['erk-dlambda'] = basic_linear_system
solver_parameters['erk-grad'] = basic_linear_system

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

nsteps = 1

xns, xn_subs, steps, tns = compute_state_block(model, timestepper, 1, nsteps, dt, x0, t0)
objective = L2Objective(xns, tns, nsteps, model.spaces.dx)
optimizer = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective, dt)


x0_new, x0_new_sub, x0_new_split = model.get_full_var('x0_new', split_x_and_aux=True)
x0_diff, _, _ = model.get_x_var('x0_diff')
x0_opt, _, _ = model.get_x_var('x0_opt')
opt_diff, _, _ = model.get_x_var('opt_diff')

parameters['initial-conditions']['ox'] = 0.11
parameters['initial-conditions']['oy'] = 0.11
model.initialize(x0_new_sub, t0, new_params=parameters)
plot_variable(x0_new[0].sub(0), 'v-start',  2, True)
plot_variable(x0_new[0].sub(1), 'h-start',  2, False)
plot_variable(x0_new[0].sub(2), 'S-start',  2, False)
plot_variable(x0[0].sub(0), 'v-orig',  2, True)
plot_variable(x0[0].sub(1), 'h-orig',  2, False)
plot_variable(x0[0].sub(2), 'S-orig',  2, False)

x0_diff.assign(x0_new[0] - x0[0])
plot_variable(x0_diff.sub(0), 'v-diff',  2, True)
plot_variable(x0_diff.sub(1), 'h-diff',  2, False)
plot_variable(x0_diff.sub(2), 'S-diff',  2, False)

x0_new_arr = create_flattened_numpy_arr_from_mixed_function(x0_new[0]).copy()
opt_ic_arr = optimizer.optimize(x0_new_arr, opt_type='ics', method='TNC', coeffs0=model_coeff)
set_mixed_function_from_flattened_array(x0_opt, opt_ic_arr)
opt_diff.assign(x0_opt - x0[0])

print(model.norm(opt_diff))
plot_variable(x0_opt.sub(0), 'v-opt',  2, True)
plot_variable(x0_opt.sub(1), 'h-opt',  2, False)
plot_variable(x0_opt.sub(2), 'S-opt',  2, False)
plot_variable(opt_diff.sub(0), 'v-opt-diff',  2, True)
plot_variable(opt_diff.sub(1), 'h-opt-diff',  2, False)
plot_variable(opt_diff.sub(2), 'S-opt-diff',  2, False)


#all in some big multiplot...
#PROBABLY PLOT THE DIFFERENCES?
#PROBABLY PLOT X0 and X0_NEW ALSO!
