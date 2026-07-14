import numpy as np
import scipy as sp

class _TimeStepper():

    def compute_state(self, nsteps, params, t0, x0, dt):

        t = np.zeros(nsteps+1)
        xn = np.zeros((nsteps+1,self.dynamics.get_x_size()))
        t[0] = t0
        xn[0,:] = x0

        for n in range(nsteps):
            self.take_forward_step(xn[n+1,:], dt, t[n], xn[n,:], params)
            t[n+1] = t[n] + dt
        return xn, t

    def compute_state_block(self, nblocks, nsteps, t0, x0, dt, params):
        xns = []
        tns = []
        steps = []
        for k in range(nblocks):
            if (k==0):
                xn, t = self.compute_state(nsteps, params, t0, x0, dt)
            else:
                xn, t = self.compute_state(nsteps, params, t[-1], xn[-1,:], dt)
            xns.append(xn[::nsteps,:])
            tns.append(t[::nsteps])
            steps.append(nsteps)
        return xns, steps, tns

class _GeneralRK(_TimeStepper):
    def __init__(self, dynamics, A, c, b, nstages):
        self.A = A
        self.c = c
        self.b = b
        self.nstages = nstages
        self.dynamics = dynamics
        self.Fi = []
        self.ti = []
        self.mui = []
        self.Yi = []
        self.li = []
        for i in range(self.nstages):
            self.Fi.append(np.zeros(dynamics.get_x_size()))
            self.Yi.append(np.zeros(dynamics.get_x_size()))
            self.li.append(np.zeros(dynamics.get_x_size()))
            self.ti.append(0.)
            self.mui.append(np.zeros(dynamics.get_x_size()))

#THIS IS EXPLICIT ONLY...
    def take_forward_step(self, xnp1, dt, tn, xn, params):
        for i in range(self.nstages):
            self.Yi[i][:] = xn[:]
            for j in range(i):
                self.Yi[i] = self.Yi[i] + dt * self.A[i,j] * self.Fi[j]
            self.Fi[i][:] = self.dynamics.rhs(self.Yi[i], tn + self.c[i] * dt, params)
        xnp1[:] = xn[:]
        for i in range(self.nstages):
            xnp1[:] = xnp1[:] + self.b[i] * dt * self.Fi[i]

#THIS IS EXPLICIT ONLY...
    def take_adjoint_step(self, ts_grad, delta_lambda, dt, tnp1, lambda_np1, params):
        tn = tnp1 - dt

        #compute mui
        for i in range(self.nstages-1,-1,-1):
            self.mui[i][:] = dt * self.b[i] * lambda_np1
            self.li[i][:] = 0.0
            for j in range(self.nstages):
                self.li[i][:] = self.li[i][:] + dt * self.A[j,i] * self.mui[j]
            jacT = self.dynamics.jacT_x(self.Yi[i], tn + self.c[i] * dt, params)
            self.mui[i][:] = self.mui[i][:] + jacT.dot(self.li[i])

        #compute grad
        ts_grad[:] = 0.0
        for i in range(self.nstages):
            jacT_params = self.dynamics.jacT_params(self.Yi[i], tn + self.c[i] * dt, params)
            ts_grad[:] = ts_grad[:] - jacT_params.dot(self.mui[i])


        #compute lambda_n
        delta_lambda[:] = 0.0
        for i in range(self.nstages):
            jacT = self.dynamics.jacT_x(self.Yi[i], tn + self.c[i] * dt, params)
            delta_lambda[:] = delta_lambda[:] + jacT.dot(self.mui[i][:])

class Euler(_GeneralRK):
    def __init__(self, dynamics):
        A = np.array([[0.0,],])
        b = np.array([1.0,])
        c = np.array([0.0,])
        _GeneralRK.__init__(self, dynamics, A, c, b, 1)

class RK4(_GeneralRK):
    def __init__(self, dynamics):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        b = np.array([1./6., 1./3., 1./3., 1./6.])
        c = np.array([0.0, 0.5, 0.5, 1.0])
        _GeneralRK.__init__(self, dynamics, A, c, b, 4)

class SSPRK3(_GeneralRK):
    def __init__(self, dynamics, nstages):
        A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        b = np.array([1./6., 1./6., 2./3.])
        c = np.array([0.0, 1.0, 0.5])
        _GeneralRK.__init__(self, dynamics, A, c, b, 3)

class SSPRK43(_GeneralRK):
    def __init__(self, dynamics, nstages):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0/6.0, 1.0/6.0, 1.0/6.0, 0.0]])
        b = np.array([1./6., 1./6., 1./6., 3./6.])
        c = np.array([0.0, 0.5, 1.0, 0.5])
        _GeneralRK.__init__(self, dynamics, A, c, b, 4)
