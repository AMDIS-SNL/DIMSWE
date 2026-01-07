import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
from dynamics import get_dynamics
from time_integrators import get_timestepper
from initialconditions import get_initcond

from parameters import parameters

#load results
ofile = h5.File(parameters['outputname'] + '.hdf5', 'r')

initcond = get_initcond(parameters)
model = get_dynamics(parameters, initcond)
timestepper = get_timestepper(parameters, model, initcond)

nsteps = np.arange(parameters['nsteps'] + 1)

for statistic_name in model.statistics.get_statistics_names():
    statistic_data = np.array(ofile[statistic_name])
    plt.figure()
    plt.plot(nsteps, statistic_data)
    plt.savefig(statistic_name + '.png')
    plt.close()

    plt.figure()
    plt.plot(nsteps, (statistic_data - statistic_data[0])/statistic_data[0]*100.)
    plt.savefig(statistic_name + '-change.png')
    plt.close()

varnames = model.vars.variable_names()
var_data = np.array(ofile['x'])

for i in range(model.vars.dim):
    plt.figure()
    plt.plot(nsteps, var_data[:,i])
    plt.savefig(varnames[i] + '.png')

for varname in timestepper.get_convergence_data_names():
    var_data = np.array(ofile[varname])
    plt.figure()
    plt.plot(np.arange(var_data.shape[0]), var_data)
    plt.savefig(varname + '.png')
