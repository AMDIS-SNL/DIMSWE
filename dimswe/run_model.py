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

#SUPER HACKY RIGHT NOW
#THIS IS A HORRIBLE HACK FOR MAXWELL IN A BOX USING LOWEST ORDER SPACES
    if parameters['timestepping']['dt_type'] == 'cfl':
        min_dx = min(min(mesh.dx, mesh.dy), mesh.dz)
        wavespeed = dynamics.get_max_wavespeed()
        parameters['timestepping']['dt'] = calculate_timestep(parameters['timestepping']['cfl_const'], min_dx, wavespeed)
        logger.output('calculated cfl-based dt as ' + str(parameters['timestepping']['dt']), 0)
    timestepper = get_timestepper(parameters, dynamics, initcond, logger)
    output = Output(parameters, dynamics, timestepper, logger)

    logger.output('Starting simulation', 0)
    xn = dynamics.get_x_var('xn')
    dynamics.initialize(xn)
    #MIGHT NEED XN_SUB HERE?
    #SHOULD THIS BE MORE STATELESS?
    self.dynamics.create_diagnostics(xn)
    self.dynamics.create_statistics(xn)
    dynamics.compute_diagnostics(xn, t)
    dynamics.compute_statistics(xn, t, 0, 0)

    t = dynamics.get_t0()
    dt = parameters['timestepping']['dt']
    output.output(xn, t, 0, 0, 0)
    for n in range(1, parameters['timestepping']['num_steps']+1):
        logger.output('taking time step n=' + str(n), 1)\
        timestepper.take_forward_step(xn, xn, t, dt)
        t = t + dt
        if ((n % parameters['output']['stat_freq']) == 0):
            dynamics.compute_statistics(xn, t, n, n // parameters['output']['stat_freq']) #THIS SHOULD EAT THE CURRENT T
        if ((n % parameters['output']['output_freq']) == 0):
            dynamics.compute_diagnostics(xn, t)
            output.output(xn, t, n, n // parameters['output']['output_freq'], n // parameters['output']['stat_freq'])

    logger.output('Ended simulation', 0)

if __name__ == "__main__":
    cfgfile = sys.argv[1]
    parameters = get_parameters(cfgfile)
    run_model(parameters)
