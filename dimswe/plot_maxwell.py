import pyvista as pv
from .models import get_model
from .logger import Logger
from .parameters import get_parameters
from .meshes import set_dimension
from .plotting import plot_statistic
from firedrake import CheckpointFile
import numpy as np



def plot_maxwell(parameters):

    logger = Logger(parameters)
    set_dimension(parameters)
    initcond = get_initial_condition(parameters)
    dynamics = get_dynamics(parameters, None, None, logger, initcond)

    mesh_pv = pv.read("sim.pvd")
    print(dir(mesh_pv))

    with CheckpointFile(parameters['output']['outfile_name'] + '.h5', 'r') as chkpoint_file:

        h5file = chkpoint_file.h5pyfile
        for stat in dynamics.statistics.statistic_names:
            statdata = np.array(h5file[stat])
            plot_statistic(statdata, stat)
