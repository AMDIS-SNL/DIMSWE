from .models import get_model
from .output import Output
from .timestepping import get_timestepper
from .parameters import get_parameters, overall_solver_parameters
from .logger import Logger
import sys
from firedrake.petsc import PETSc

def calculate_timestep(cfl_const, mindx, wavespeed):
    return cfl_const * mindx / wavespeed


def run_model(parameters):

    logger = Logger(parameters)

    logger.output('Setting up simulation', 0)
    model = get_model(parameters, logger)

#SUPER HACKY RIGHT NOW
#THIS IS A HORRIBLE HACK FOR MAXWELL IN A BOX USING LOWEST ORDER SPACES
    if parameters['timestepping']['dt_type'] == 'cfl':
        min_dx = min(min(mesh.dx, mesh.dy), mesh.dz)
        wavespeed = model.get_max_wavespeed()
        parameters['timestepping']['dt'] = calculate_timestep(parameters['timestepping']['cfl_const'], min_dx, wavespeed)
        logger.output('calculated cfl-based dt as ' + str(parameters['timestepping']['dt']), 0)

    coeff, coeff_sub, coeff_split = model.get_coeff_var('coeff')
    t = model.get_t_var()

    timestepper = get_timestepper(parameters, model, logger, solver_parameters=overall_solver_parameters)

#HOW DO WE HANDLE THIS IN THE GENERAL CASE FOR LIE SPLIT INTEGRATOR?
#THE ISSUE IS A MIX OF EXPLICIT AND IMPLICIT
#SIMILAR CONCERNS FOR IMEX SCHEMES...maybe not actually here- we are actually just doing a bunch of nonlinear solves, at least for dirk
#REALLY THE PROBLEM IS- GIVEN X FULL, CAN I EXTRACT SUBFUNCTIONS AND ASSIGN INTO XPARTIAL?
#THIS CAN BE EASILY DONE WITH SUBFUCTIONS, BUT IT WOULD BE NICE TO DO IT "ALL AT ONCE"

    #xn, xn_sub, x_split = model.get_full_var('xn', split_x_and_aux=timestepper.split_x_and_aux())
    xn, xn_sub, x_split = model.get_full_var('xn', split_x_and_aux=True)

    logger.output('Starting simulation', 0)

    output = Output(xn, xn_sub, coeff_sub, parameters, model, logger)

    model.initialize(xn_sub, t)
    model.set_coeffs(parameters, coeff_sub)
    timestepper.set_coeff(coeff)

    model.create_diagnostics(xn_sub, t, coeff_sub)
    model.create_statistics(xn_sub, t, coeff_sub)
    model.compute_diagnostics()
    model.compute_statistics(0, 0)

    dt = parameters['timestepping']['dt']
    output.output(t, 0, 0, 0)
    for n in range(1, parameters['timestepping']['num_steps']+1):
        logger.output('taking time step n=' + str(n), 1)
        with PETSc.Log.Event("time step"):
            timestepper.take_forward_step(xn, xn_sub, xn, t, dt)
        t.assign(t + dt)
        if ((n % parameters['output']['stat_freq']) == 0):
            model.compute_statistics(n, n // parameters['output']['stat_freq'])
        if ((n % parameters['output']['output_freq']) == 0):
            model.compute_diagnostics()
            output.output(t, n, n // parameters['output']['output_freq'], n // parameters['output']['stat_freq'])

    logger.output('Ended simulation', 0)

if __name__ == "__main__":
    cfgfile = sys.argv[1]
    parameters = get_parameters(cfgfile)
    run_model(parameters)
