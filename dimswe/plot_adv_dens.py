from .dynamics import get_dynamics
from .logger import Logger

from .initial_conditions import get_initial_condition
from .meshes import set_dimension
from .plotting import plot_mesh, plot_scalar2D, plot_scalar1D, plot_vector2D_mag, plot_vector2D_quiver, plot_variable, plot_statistic, animate_scalar2D
from firedrake import CheckpointFile
import numpy as np


def plot_adv_dens(parameters):
    #load mesh
    logger = Logger(parameters)
    set_dimension(parameters)
    initcond = get_initial_condition(parameters)
    dynamics = get_dynamics(parameters, None, None, logger, initcond)

    #EVETUALLY TIE THIS TO A VARIABLE TYPE THAT DYNAMICS KNOWS
    #ie scalar or vector
    vector_list = ['m', 'v', 'u', 'F']

    with CheckpointFile(parameters['output']['outfile_name'] + '.h5', 'r') as chkpoint_file:

        mesh = chkpoint_file.load_mesh()
        #if parameters['dim'] >= 2:
        #    plot_mesh(mesh)

        h5file = chkpoint_file.h5pyfile
        for stat in dynamics.statistics.statistic_names:
            statdata = np.array(h5file[stat])
            plot_statistic(statdata, stat)

        noutput = (parameters['timestepping']['num_steps'] // parameters['output']['output_freq']) + 1
        for var in dynamics.variableset.varlist:
            animate_scalar2D(chkpoint_file, mesh, var, noutput, var + '-mov')
        for var in dynamics.coefflist:
            animate_scalar2D(chkpoint_file, mesh, var, noutput, var + '-mov')
        for var in dynamics.diagnostics.var_list:
              animate_scalar2D(chkpoint_file, mesh, var, noutput, var + '-mov')

        if parameters['plot']['static_plots']:
            for n in range(0, parameters['timestepping']['num_steps']+1):
                if ((n % parameters['output']['output_freq']) == 0):
                    output_step = n // parameters['output']['output_freq']
                    print('plotting output at step ' + str(n) + ' output step ' + str(output_step))

                    for var in dynamics.variableset.varlist:
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(vardat, var + '.' + str(n),  parameters['mesh']['dim'], var in vector_list)

                    for var in dynamics.coefflist:
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(vardat, var + '.' + str(n),  parameters['mesh']['dim'], var in vector_list)

                    if parameters['output']['output_aux_vars']:

                        for var in dynamics.q_aux_var_list:
                            vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                            plot_variable(vardat, var + '.'+  str(n), parameters['mesh']['dim'], var in vector_list)

                        for var in dynamics.dfdx_aux_var_list:
                            vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                            plot_variable(vardat, var + '.' + str(n), parameters['mesh']['dim'], var in vector_list)

                    for var in dynamics.diagnostics.var_list:
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(vardat, var + '.' + str(n),  parameters['mesh']['dim'], var in vector_list)
