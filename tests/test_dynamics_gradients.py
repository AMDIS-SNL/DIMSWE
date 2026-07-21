import numpy as np
from dimswe.models import get_model
from dimswe.timestepping import get_timestepper
from dimswe.logger import EmptyLogger
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.optimize import compute_state_block, compute_states, create_states
from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function

import pytest

from firedrake import assemble, inner, norm

#compare FD dynamics gradient with the UFL gradient calc
#for Fi/wi, and also coeffs
#hopefully the latter one illuminates the issue...

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


rhs_F, rhs_Fis, rhs_W, rhs_Wis = timestepper.get_rhs_expr()

#COEFFVAR
#XVAR
#WVAR


def taylor_remainder_check():
    pass

#IS THERE A VERSION OF THIS THAT IS CONSISTENT WITH VARIATIONAL JACOBIAN?
#PROBABLY SOME SORT OF MASS MATRICIZED VERSION?
#UNCLEAR...
def compute_func():
    pass

def compute_jacobian():
    pass

timestepper.xk[0].assign(SOMETHING)
timestepper.xk[1].assign(SOMETHING)


if model.has_coeff():
    derivF_coeff = derivative(rhs_F, timestepper.coeff, timestepper.coeff_trial)
#THIS IS THE COORDINATE VERSION
    derivF_coeff_rhs = assemble(action(derivF_coeff, COEFFVAR))
#THE VARIATIONAL VERSION WOULD SOLVE A MASS MATRIX EQUATION

derivF_F = derivative(rhs_F, timestepper.xk[0], timestepper.xk_trial[0])
derivF_F_rhs = assemble(action(derivF_F, XVAR))

if model.has_aux():
    derivF_W = derivative(rhs_F, timestepper.xk[1], timestepper.xk_trial[1])
    derivF_W_rhs = assemble(action(derivF_W, WVAR))

    if model.has_coeff():

        derivW_coeff = derivative(rhs_W, timestepper.coeff, timestepper.coeff_trial)
        derivW_coeff_rhs = assemble(action(derivW_coeff, COEFFVAR))

    derivW_F = derivative(rhs_W, timestepper.xk[0], timestepper.xk_trial[0])
    derivW_F_rhs = assemble(action(derivW_F, XVAR))

    derivW_W = derivative(rhs_W, timestepper.xk[1], timestepper.xk_trial[1])
    derivW_W_rhs = assemble(action(derivW_W, WVAR))

#MAYBE DONT NEED TO DO THIS? UNCLEAR...
for i in range(timestepper.nstages):
#ASSIGN VALUES FOR FIJ!
    if model.has_coeff():

        derivF_coeff = derivative(rhs_Fis[i], timestepper.coeff, timestepper.coeff_trial)
        derivF_coeff_rhs = assemble(action(derivF_coeff, COEFFVAR))

    derivF_F = derivative(rhs_Fis[i], timestepper.xk[0], timestepper.xk_trial[0])
    derivF_F_rhs = assemble(action(derivF_F, XVAR))

    if model.has_aux():

        derivF_W = derivative(rhs_Fis[i], timestepper.xk[1], timestepper.xk_trial[1])
        derivF_W_rhs = assemble(action(derivF_W, WVAR))
        if model.has_coeff():
            derivW_coeff = derivative(rhs_Wis[i], timestepper.coeff, timestepper.coeff_trial)
            derivW_coeff_rhs = assemble(action(derivW_coeff, COEFFVAR))

        derivW_F = derivative(rhs_Wis[i], timestepper.xk[0], timestepper.xk_trial[0])
        derivW_F_rhs = assemble(action(derivW_F, XVAR))

        derivW_W = derivative(rhs_Wis[i], timestepper.xk[1], timestepper.xk_trial[1])
        derivW_W_rhs = assemble(action(derivW_W, WVAR))
