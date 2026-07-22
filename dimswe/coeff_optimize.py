#import matplotlib.pyplot as plt
import numpy as np
from dimswe.optimize import Lagrangian_ODEConstrainedOptimization, L2Objective
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import EmptyLogger
from dimswe.parameters import get_parameters
from dimswe.optimize import compute_state_block, compute_states, create_states
from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function, set_mixed_function_from_flattened_array
from firedrake import norm
from dimswe.plotting import plot_variable

parameters = get_parameters('tests/tswe.cfg')
logger = EmptyLogger()
model = get_model(parameters, logger, has_dynamics_statistics=False)
model_coeff, model_coeff_sub, model_coeff_split = model.get_coeff_var('coeff')
timestepper = get_timestepper(parameters, model, logger)
dt = parameters['timestepping']['dt']
x0, x0_sub, x0_split = model.get_full_var('x0', split_x_and_aux=True)
t0 = model.get_t_var()
model.initialize(x0_sub, t0)
model.set_coeffs(parameters, model_coeff_sub)
timestepper.set_coeff(model_coeff)

nsteps = 3

xns, xn_subs, steps, tns = compute_state_block(model, timestepper, 1, nsteps, dt, x0, t0)
objective = L2Objective(xns, tns, nsteps, model.spaces.dx)
optimizer = Lagrangian_ODEConstrainedOptimization(model, timestepper, objective, dt)

coeff_new, coeff_new_sub, _ = model.get_coeff_var('coeff_new')
coeff_opt, _, _ = model.get_coeff_var('coeff_opt')
coeff_diff, _, _ = model.get_coeff_var('coeff_diff')
coeff_opt_diff, _, _ = model.get_coeff_var('coeff_opt_diff')

parameters['hyperviscosity']['c0'] = 0.05
parameters['hyperviscosity']['s'] = 2.8
model.set_coeffs(parameters, coeff_new_sub)
coeff_diff.assign(coeff_new - model_coeff)

plot_variable(model_coeff, 'mu-orig',  2, False)
plot_variable(coeff_new, 'mu-start',  2, False)
plot_variable(coeff_diff, 'mu-diff',  2, False)

coeff_new_arr = create_flattened_numpy_arr_from_mixed_function(coeff_new)
coeff_opt_arr = optimizer.optimize(coeff_new_arr, opt_type='coeffs', use_jacobian=False)
set_mixed_function_from_flattened_array(coeff_opt, coeff_opt_arr)
coeff_opt_diff.assign(coeff_opt - model_coeff)

print(model.norm(coeff_opt_diff))
plot_variable(coeff_opt, 'mu-opt',  2, False)
plot_variable(coeff_opt_diff, 'mu-opt-diff',  2, False)


#all in some big multiplot...
#PROBABLY PLOT THE DIFFERENCES?
#PROBABLY PLOT X0 and X0_NEW ALSO!
