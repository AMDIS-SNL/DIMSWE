from .meshes import get_mesh_and_spaces
from .dynamics import get_dynamics
from .initial_conditions import get_initial_condition
from .output import Output
from .timestepping import get_timestepper
from .parameters import get_parameters
from .logger import Logger
import sys

def calculate_timestep(cfl_const, mindx, wavespeed):
    return cfl_const * mindx / wavespeed


def run_model(parameters):

    logger = Logger(parameters)

    logger.output('Setting up simulation', 0)
    initcond = get_initial_condition(parameters)
    mesh, spaces = get_mesh_and_spaces(parameters, initcond)
    dynamics = get_dynamics(parameters, mesh, spaces, logger, initcond)
    xn, xn_sub, x_split = dynamics.get_x_var('xn')
    t = dynamics.get_t_var()
    coeffs = dynamics.get_coeff_var()
    coeff, coeff_sub, coeff_split, coeff_trial = coeffs

#SWAP FROM DYNAMICS TO MODEL AS THE MAIN OBJECT!


#SUPER HACKY RIGHT NOW
#THIS IS A HORRIBLE HACK FOR MAXWELL IN A BOX USING LOWEST ORDER SPACES
    if parameters['timestepping']['dt_type'] == 'cfl':
        min_dx = min(min(mesh.dx, mesh.dy), mesh.dz)
        wavespeed = dynamics.get_max_wavespeed()
        parameters['timestepping']['dt'] = calculate_timestep(parameters['timestepping']['cfl_const'], min_dx, wavespeed)
        logger.output('calculated cfl-based dt as ' + str(parameters['timestepping']['dt']), 0)

    timestepper = get_timestepper(parameters, dynamics, initcond, logger, coeffs)

    logger.output('Starting simulation', 0)

    output = Output(xn, coeff, parameters, dynamics, timestepper, logger)

    dynamics.initialize(xn, t)
    dynamics.set_default_coeffs(coeff_sub)

    dynamics.create_diagnostics(xn_sub, t, coeff)
    dynamics.create_statistics(xn_sub, t, coeff)
    dynamics.compute_diagnostics()
    dynamics.compute_statistics(0, 0)

    dt = parameters['timestepping']['dt']
    output.output(t, 0, 0, 0)
    for n in range(1, parameters['timestepping']['num_steps']+1):
        logger.output('taking time step n=' + str(n), 1)
        timestepper.take_forward_step(xn, xn_sub, xn, t, dt)
        t.assign(t + dt)
        if ((n % parameters['output']['stat_freq']) == 0):
            dynamics.compute_statistics(n, n // parameters['output']['stat_freq']) #THIS SHOULD EAT THE CURRENT T
        if ((n % parameters['output']['output_freq']) == 0):
            dynamics.compute_diagnostics()
            output.output(t, n, n // parameters['output']['output_freq'], n // parameters['output']['stat_freq'])

    logger.output('Ended simulation', 0)

if __name__ == "__main__":
    cfgfile = sys.argv[1]
    parameters = get_parameters(cfgfile)
    run_model(parameters)
