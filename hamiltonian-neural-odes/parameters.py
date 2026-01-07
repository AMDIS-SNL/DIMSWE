import numpy as np

parameters = {}
parameters['timestepper'] = 'AVF2'
parameters['avf_quad_pts'] = 4
parameters['nsteps'] = 10000
parameters['dt'] = 0.01
parameters['outputname'] = 'sim'

parameters['bracket'] = 'canonical'
#parameters['hamiltonian'] = 'harmonic-oscillator'
#parameters['omega'] = 1.0
#parameters['num_variable_pairs'] = 1

parameters['hamiltonian'] = 'nonlinear-oscillator'
parameters['omega'] = 1.0
parameters['num_variable_pairs'] = 1

#parameters['hamiltonian'] = 'multiple-harmonic-oscillator'
#parameters['omegas'] = np.array([0.5,0.75,1.0])
#parameters['num_variable_pairs'] = 3

parameters['initcond'] = 'harmonic-oscillator'
