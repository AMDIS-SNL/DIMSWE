from model import run_model
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

parameters = {}
parameters['nsteps'] = 1000000
parameters['dt'] = 0.01
parameters['bracket'] = 'canonical'
parameters['hamiltonian'] = 'doublewell-oscillator'
parameters['omega'] = 1.0
parameters['num_variable_pairs'] = 1
parameters['initcond'] = 'harmonic-oscillator'

parameters['timestepper'] = 'AVF2'
avf_quad_pts_list = [1,2]
for avf_quad_pt in avf_quad_pts_list:

    parameters['outputname'] = 'avf_nquad' + str(avf_quad_pt)
    parameters['avf_quad_pts'] = avf_quad_pt
    run_model(parameters)

parameters['timestepper'] = 'RK4'
parameters['outputname'] = 'rk4'
run_model(parameters)

energies = []
vardats = []
for avf_quad_pt in avf_quad_pts_list:
    ofile = h5.File('avf_nquad' + str(avf_quad_pt) + '.hdf5', 'r')
    energies.append(np.array(ofile['energy']))
    vardats.append(np.array(ofile['x']))
ofile = h5.File('rk4.hdf5', 'r')
energies.append(np.array(ofile['energy']))
vardats.append(np.array(ofile['x']))

nsteps = np.arange(parameters['nsteps'] + 1)

plt.figure(figsize=(12,12))
for i,avf_quad_pt in enumerate(avf_quad_pts_list):
    plt.plot(nsteps, energies[i], label=str(avf_quad_pt))
plt.plot(nsteps, energies[-1], label='rk4')
plt.legend()
plt.savefig('energy.png')
plt.close()

plt.figure(figsize=(12,12))
for i,avf_quad_pt in enumerate(avf_quad_pts_list):
    plt.semilogy(nsteps, (energies[i]-energies[i][0])/energies[i][0]*100., label=str(avf_quad_pt))
plt.semilogy(nsteps, (energies[-1]-energies[-1][0])/energies[-1][0]*100., label='rk4')
plt.legend()
plt.savefig('energy-fractional-change.png')
plt.close()

plt.figure(figsize=(12,12))
for i,avf_quad_pt in enumerate(avf_quad_pts_list):
    plt.plot(nsteps, vardats[i][:,0], label=str(avf_quad_pt))
plt.plot(nsteps, vardats[-1][:,0], label='rk4')
plt.legend()
plt.savefig('p.png')
plt.close()

plt.figure(figsize=(12,12))
for i,avf_quad_pt in enumerate(avf_quad_pts_list):
    plt.plot(nsteps, vardats[i][:,1], label=str(avf_quad_pt))
plt.plot(nsteps, vardats[-1][:,1], label='rk4')
plt.legend()
plt.savefig('q.png')
plt.close()

last1000 = np.arange(1000)

plt.figure(figsize=(12,12))
for i,avf_quad_pt in enumerate(avf_quad_pts_list):
    plt.plot(last1000, vardats[i][-1000:,0], label=str(avf_quad_pt))
plt.plot(last1000, vardats[-1][-1000:,0], label='rk4')
plt.legend()
plt.savefig('p-last1000.png')
plt.close()

plt.figure(figsize=(12,12))
for i,avf_quad_pt in enumerate(avf_quad_pts_list):
    plt.plot(last1000, vardats[i][-1000:,1], label=str(avf_quad_pt))
plt.plot(last1000, vardats[-1][-1000::,1], label='rk4')
plt.legend()
plt.savefig('q-last1000.png')
plt.close()
