from .model import run_model
from .plot_adv_dens import plot_variable, plot_statistic
from .parameters import get_parameters
from .initial_conditions import get_initial_condition
from .logger import Logger
from .meshes import set_dimension
from .dynamics import get_dynamics
import numpy as np
from firedrake import CheckpointFile

parameters = {}

parameters["mesh"] = "rectangle-periodic"  # line rectangle

parameters["nz"] = 100
parameters["diagonal"] = "crossed"  # WHAT ARE THE OPTIONS HERE?
parameters["simplicial_cells"] = False

# THESE ARE FE SPECIFIC
parameters["family"] = "Q"
# MISSING LOTS OF STUFF FOR UPWINDING, ETC.

parameters["loglevel"] = 100

# OTHER STUFF WILL BE DECLIB SPECIFIC
# FOR EXAMPLE
# parameters['num_form_quad'] = 3

parameters["num_steps"] = 1000
parameters["dt"] = 200
parameters["num_avf_quad"] = 2
parameters["output_freq"] = 50
parameters["stat_freq"] = 1
parameters["output_aux_vars"] = False  # super useful for debugging


parameters["model"] = (
    "tswe-cf"  # tswe-cf tswe-lp tswe-cf-h1 ce-cf ce-lp mhd maxwell eulermaxwell scalarwave
)
parameters["tracer_names"] = []  # ['T1', 'T2']
parameters["thermo"] = "idealgas-entropy"
parameters["tracer_init_conds"] = []  # ['gaussian', 'block']

parameters["initialcondition"] = "doublevortex"


# EVETUALLY TIE THIS TO A VARIABLE TYPE THAT DYNAMICS KNOWS
# ie scalar or vector
vector_list = ["m", "v", "u", "F"]


def plot_run(outfile):
    parameters = get_parameters()
    logger = Logger(parameters)
    set_dimension(parameters)
    initcond = get_initial_condition(parameters)
    dynamics = get_dynamics(parameters, None, None, logger, initcond)

    with CheckpointFile(outfile + ".h5", "r") as chkpoint_file:

        mesh = chkpoint_file.load_mesh()
        # if parameters['dim'] >= 2:
        #    plot_mesh(mesh)

        h5file = chkpoint_file.h5pyfile
        for stat in dynamics.statistics.statistic_names:
            statdata = np.array(h5file[stat])
            plot_statistic(statdata, stat)

        for n in range(0, parameters["num_steps"] + 1):
            if (n % parameters["output_freq"]) == 0:
                output_step = n // parameters["output_freq"]
                print(
                    "plotting output at step "
                    + str(n)
                    + " output step "
                    + str(output_step)
                )
                for var in dynamics.variableset.varlist:

                    vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                    plot_variable(
                        vardat, var + str(n), parameters["dim"], var in vector_list
                    )

                if parameters["output_aux_vars"]:

                    for var in dynamics.q_aux_var_list:
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(
                            vardat, var + str(n), parameters["dim"], var in vector_list
                        )

                    for var in dynamics.dfdx_aux_var_list:
                        vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                        plot_variable(
                            vardat, var + str(n), parameters["dim"], var in vector_list
                        )

                for var in dynamics.diagnostics.var_list:
                    vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                    plot_variable(
                        vardat, var + str(n), parameters["dim"], var in vector_list
                    )


# ADD THIS
# CREATE A DIRECTORY
# MOVE FILES TO APPROPRIATE DIRECTORY


base_nx = 100
solvers = {}
solvers["AVF2"] = [
    "qn",
]  # fixedpoint
# ADD RK and KGRK STUFF HERE
outfiles = []
for order in [1, 2]:
    parameters["order"] = order
    parameters["nx"] = base_nx / order
    parameters["ny"] = base_nx / order
    for method in [
        "AVF2",
    ]:
        parameters["timestep_method"] = method
        for solver in solvers[method]:
            parameters["avf_solver"] = solver
            for alpha_s in [0, 1]:
                parameters["alpha_s"] = alpha_s
                outname = (
                    method
                    + "-order"
                    + str(order)
                    + "-"
                    + solver
                    + "-alpha"
                    + str(alpha_s)
                )
                parameters["outfile_name"] = outname
                run_model(parameters)
                plot_run(outname)
