from .models import get_model
from .logger import Logger

from .meshes import set_dimension
from .plotting import plot_mesh, plot_scalar2D, plot_scalar1D, plot_vector2D_mag, plot_vector2D_quiver, plot_variable, plot_statistic, animate_scalar2D
from firedrake import CheckpointFile
import numpy as np


def plot_adv_dens(parameters):
    #load mesh
    logger = Logger(parameters)
    set_dimension(parameters)

    model = get_model(parameters, logger)

    #EVETUALLY TIE THIS TO A VARIABLE TYPE THAT DYNAMICS KNOWS
    #ie scalar or vector
    vector_list = ['m', 'v', 'u', 'F']

    with CheckpointFile(parameters['output']['outfile_name'] + '.h5', 'r') as chkpoint_file:

        mesh = chkpoint_file.load_mesh()
        #if parameters['dim'] >= 2:
        #    plot_mesh(mesh)

        h5file = chkpoint_file.h5pyfile
        for stat in model.get_statistics_list():
            statdata = np.array(h5file[stat])
            plot_statistic(statdata, stat)

        noutput = (parameters['timestepping']['num_steps'] // parameters['output']['output_freq']) + 1
        for var in model.get_x_var_list():
            animate_scalar2D(chkpoint_file, mesh, var, noutput, var + '-mov')
        for var in model.get_coeff_list():
            animate_scalar2D(chkpoint_file, mesh, var, noutput, var + '-mov')
        for var in model.get_diagnostics_list():
            animate_scalar2D(chkpoint_file, mesh, var, noutput, var + '-mov')

        if parameters['plot']['static_plots']:
            for n in range(0, parameters['timestepping']['num_steps']+1):
                if ((n % parameters['output']['output_freq']) == 0):
                    output_step = n // parameters['output']['output_freq']
                    print('plotting output at step ' + str(n) + ' output step ' + str(output_step))

                    for var in model.get_x_var_list():
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(vardat, var + '.' + str(n),  parameters['mesh']['dim'], var in vector_list)

                    for var in model.get_coeff_list():
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(vardat, var + '.' + str(n),  parameters['mesh']['dim'], var in vector_list)

                    if parameters['output']['output_aux_vars']:

                        for var in model.get_aux_var_list():
                            vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                            plot_variable(vardat, var + '.'+  str(n), parameters['mesh']['dim'], var in vector_list)

                    for var in model.get_diagnostics_list():
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(vardat, var + '.' + str(n),  parameters['mesh']['dim'], var in vector_list)
