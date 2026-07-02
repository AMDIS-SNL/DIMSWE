
from adjoint_timesteppers import LotkaVolterra, RK4, Lagrangian_ODEConstrainedOptimization, L2Objective, Euler, LogisticEquation
import matplotlib.pyplot as plt
import numpy as np

nsteps = 500
dt = 0.01
t0 = 0
x0 = np.array([2.,1.]) #np.array([5.,])
dynamics = LotkaVolterra() #LogisticEquation LotkaVolterra
timestepper = RK4(dynamics) #Euler RK4
nblocks = 500

#Generate FOM data
params =  np.array([1.5,1.,0.5,2.]) #np.array([0.8,50.])
xn,t = timestepper.compute_state(nsteps, params, t0, x0, dt)

plt.figure()
for k in range(dynamics.get_x_size()):
    plt.plot(t,xn[:,k])
plt.savefig('exact.png')


#Do ODE constrained optimization to recover parameters

xns, steps, tns = timestepper.compute_state_block(nblocks, nsteps//nblocks, t0, x0, dt, params)

objective = L2Objective(xns, steps, tns, dynamics.get_x_size(), dynamics.get_param_size())
params0 = np.array([2.,1.5,1.,1.])
#params0 = np.array([0.3,30.])
optimizer = Lagrangian_ODEConstrainedOptimization(timestepper, objective, dt)
new_params = optimizer.optimize(params0)
print('new params', new_params)

#Regenerate FOM data with new params

xn,t = timestepper.compute_state(nsteps, new_params, t0, x0, dt)

plt.figure()
for k in range(dynamics.get_x_size()):
    plt.plot(t,xn[:,k])
plt.savefig('learned.png')
