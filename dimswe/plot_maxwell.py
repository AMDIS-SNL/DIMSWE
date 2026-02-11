import pyvista as pv
from .dynamics import get_dynamics
from .logger import Logger
from .parameters import get_parameters
from .initial_conditions import get_initial_condition
from .meshes import set_dimension
from .plotting import plot_statistic
from firedrake import CheckpointFile
import numpy as np



if __name__ == "__main__":

    #load mesh
    parameters = get_parameters()
    logger = Logger(parameters)
    set_dimension(parameters)
    initcond = get_initial_condition(parameters)
    dynamics = get_dynamics(parameters, None, None, logger, initcond)

    mesh_pv = pv.read("sim.pvd")
    print(dir(mesh_pv))

    with CheckpointFile(parameters['outfile_name'] + '.h5', 'r') as chkpoint_file:

        h5file = chkpoint_file.h5pyfile
        for stat in dynamics.statistics.statistic_names:
            statdata = np.array(h5file[stat])
            plot_statistic(statdata, stat)
