from .meshes import get_mesh_and_spaces
from .dynamics import get_dynamics
from .initial_conditions import get_initial_condition
from .output import Output
from .timestepping import get_timestepper
from .parameters import get_parameters
from .logger import Logger

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
    if parameters['dt_type'] == 'cfl':
        min_dx = min(min(mesh.dx, mesh.dy), mesh.dz)
        wavespeed = dynamics.get_max_wavespeed()
        parameters['dt'] = calculate_timestep(parameters['cfl_const'], min_dx, wavespeed)
        logger.output('calculated cfl-based dt as ' + str(parameters['dt']), 0)
    timestepper = get_timestepper(parameters, dynamics, initcond, logger)
    output = Output(parameters, dynamics, timestepper, logger)

    logger.output('Starting simulation', 0)
    timestepper.initialize()
    timestepper.create_diagnostics_statistics()
    timestepper.compute_diagnostics()
    timestepper.compute_statistics(0, 0)
#HORRIBLE HACK, SHOULD COME FROM INITCOND
    t0 = 0.0

    t = t0
    dt = parameters['dt']
    output.output(t, 0, 0, 0)
    for n in range(1, parameters['num_steps']+1):
        logger.output('taking time step n=' + str(n), 1)\
        #THIS SHOULD ALSO REALLY EAT THE CURRENT T
        timestepper.take_step(dt)
        t = t + dt
        if ((n % parameters['stat_freq']) == 0):
            timestepper.compute_statistics(n, n // parameters['stat_freq']) #THIS SHOULD EAT THE CURRENT T
        if ((n % parameters['output_freq']) == 0):
            timestepper.compute_diagnostics() #THIS SHOULD EAT THE CURRENT T
            output.output(t, n, n // parameters['output_freq'], n // parameters['stat_freq'])
    logger.output('Ended simulation', 0)

if __name__ == "__main__":
    parameters = get_parameters()
    run_model(parameters)
