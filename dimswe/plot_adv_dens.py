from .dynamics import get_dynamics
from .logger import Logger
from .parameters import get_parameters
from .initial_conditions import get_initial_condition
from .meshes import set_dimension

import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf, tricontour, tripcolor, quiver, triplot, plot
from firedrake import CheckpointFile
import numpy as np


#load mesh
parameters = get_parameters()
logger = Logger(parameters)
set_dimension(parameters)
initcond = get_initial_condition(parameters)
dynamics = get_dynamics(parameters, None, None, logger, initcond)


def plot_mesh(mesh):
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()
    fig.savefig('mesh.png')

def plot_scalar2D(func, name):
    fig, axes = plt.subplots()
    #contours = tricontour(func, axes=axes)
    #contours = tricontourf(func, axes=axes, cmap="inferno")
    contours = tripcolor(func, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '.png')
    plt.close()

def plot_vector2D_quiver(func, name):
    fig, axes = plt.subplots()
    contours = quiver(func, axes=axes)
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '.png')
    plt.close()

def plot_vector2D_mag(func, name):
    fig, axes = plt.subplots()
    contours = tripcolor(func, axes=axes)
    # contours = tricontourf(func, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '-mag.png')
    plt.close()

def plot_scalar1D(func, name):

    fig, axes = plt.subplots()
    plot(func, axes=axes)
    fig.savefig(name + '.png')
    plt.close()

def plot_variable(data, name, dim, is_vector):
    if dim == 1:
        plot_scalar1D(data, name)
    if dim == 2:
        if is_vector:
            plot_vector2D_quiver(data, name)
            # from firedrake import Function, VectorFunctionSpace, FunctionSpace, TestFunction, TrialFunction, dx, inner, LinearVariationalProblem, LinearVariationalSolver
            # from ufl_helpers import skewgrad
            # from parameters import overall_solver_parameters
            # space = VectorFunctionSpace(mesh, "CG", 2)
            # vectordat = Function(space)
            # vectordat.project(data)
            # plot_vector2D_mag(vectordat, name)
            # space = FunctionSpace(mesh, "CG", 2)
            # qhat = TestFunction(space)
            # qtrial = TrialFunction(space)
            # zeta = Function(space)
            # zeta_expr = [inner(qhat, qtrial)*dx, inner(-skewgrad(qhat), data)*dx]
            # zeta_problem = LinearVariationalProblem(zeta_expr[0], zeta_expr[1], zeta)
            # zeta_solver = LinearVariationalSolver(zeta_problem, solver_parameters=overall_solver_parameters['zetadiag'], options_prefix='zetadiag')
            # zeta_solver.solve()
            # plot_scalar2D(zeta, 'zeta-computed')
        else:
            plot_scalar2D(data, name)

def plot_statistic(data, name):
    plt.figure()
    plt.plot(data)
    plt.savefig(name + '.png')
    plt.close()

    plt.figure()
    plt.plot((data - data[0])/data[0]*100)
    plt.savefig(name + '-fractional-change.png')
    plt.close()

#EVETUALLY TIE THIS TO A VARIABLE TYPE THAT DYNAMICS KNOWS
#ie scalar or vector
vector_list = ['m', 'v', 'u', 'F']

with CheckpointFile(parameters['outfile_name'] + '.h5', 'r') as chkpoint_file:

    mesh = chkpoint_file.load_mesh()
    #if parameters['dim'] >= 2:
    #    plot_mesh(mesh)

    h5file = chkpoint_file.h5pyfile
    for stat in dynamics.statistics.statistic_names:
        statdata = np.array(h5file[stat])
        plot_statistic(statdata, stat)

    for n in range(0, parameters['num_steps']+1):
        if ((n % parameters['output_freq']) == 0):
            output_step = n // parameters['output_freq']
            print('plotting output at step ' + str(n) + ' output step ' + str(output_step))
            for var in dynamics.variableset.varlist:

                vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                plot_variable(vardat, var + '.' + str(n),  parameters['dim'], var in vector_list)

            if parameters['output_aux_vars']:

                for var in dynamics.q_aux_var_list:
                    vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                    plot_variable(vardat, var + '.'+  str(n), parameters['dim'], var in vector_list)

                for var in dynamics.dfdx_aux_var_list:
                    vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                    plot_variable(vardat, var + '.' + str(n), parameters['dim'], var in vector_list)

            for var in dynamics.diagnostics.var_list:
                vardat = chkpoint_file.load_function(mesh, var, idx=output_step)
                plot_variable(vardat, var + '.' + str(n),  parameters['dim'], var in vector_list)
