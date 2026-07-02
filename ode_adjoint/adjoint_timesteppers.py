import numpy as np
import scipy as sp

class _TimeStepper():

    def compute_state(self, nsteps, params, t0, x0, dt):

        t = np.zeros(nsteps+1)
        xn = np.zeros((nsteps+1,self.dynamics.get_x_size()))
        t[0] = t0
        xn[0,:] = x0

        for n in range(nsteps):
            xn[n+1,:] = self.take_forward_step(dt, t[n], xn[n,:], params)
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

class _ExplicitRK(_TimeStepper):
    def __init__(self, dynamics, A, c, b, nstages):
        self.A = A
        self.c = c
        self.b = b
        self.nstages = nstages
        self.dynamics = dynamics
        self.Yi = []
        self.fi = []
        self.ti = []
        self.mui = []
        self.li = []
        for i in range(self.nstages):
            self.Yi.append(np.zeros(dynamics.get_x_size()))
            self.fi.append(np.zeros(dynamics.get_x_size()))
            self.ti.append(0.)
            self.mui.append(np.zeros(dynamics.get_x_size()))
            self.li.append(np.zeros(dynamics.get_x_size()))

    def take_forward_step(self, dt, t0, xn, params):
        self.Yi[0][:] = xn[:]
        self.ti[0] = t0
        self.fi[0][:] = self.dynamics.rhs(self.Yi[0], self.ti[0], params)
        for i in range(1,self.nstages):
            self.Yi[i][:] = xn[:]
            for j in range(i):
                self.Yi[i][:] = self.Yi[i] + dt * self.A[i-1,j] * self.fi[j]
            self.ti[i] = t0 + self.c[i-1] * dt
            self.fi[i] = self.dynamics.rhs(self.Yi[i], self.ti[i], params)
        xnp1 = xn.copy()
        for i in range(self.nstages):
            xnp1 = xnp1 + self.b[i] * dt * self.fi[i]
        return xnp1

    def take_adjoint_step(self, dt, lambda_np1, params):

        #compute mui
        for i in range(self.nstages-1,-1,-1):
            self.li[i][:] = self.b[i] * lambda_np1
            #print(i,self.b[i])
            for j in range(i,self.nstages-1):
                self.li[i][:] = self.li[i][:] + self.A[i,j] * self.mui[j+1]
                #print(i,j,self.A[i,j])
            jacT = self.dynamics.jacT_x(self.Yi[i], self.ti[i], params)
            self.mui[i][:] = dt * jacT.dot(self.li[i])

        #mu4 = self.dt * self.dynamics.jacT_x(self.Yi[3], self.ti[3], params).dot(1./6. * lambda_np1)
        #mu3 = self.dt * self.dynamics.jacT_x(self.Yi[2], self.ti[2], params).dot(1./3. * lambda_np1 + mu4)
        #mu2 = self.dt * self.dynamics.jacT_x(self.Yi[1], self.ti[1], params).dot(1./3. * lambda_np1 + 1./2. * mu3)
        #mu1 = self.dt * self.dynamics.jacT_x(self.Yi[0], self.ti[0], params).dot(1./6. * lambda_np1 + 1./2. * mu2)
        #print(mu4 - self.mui[3], self.li[3] - (1./6. * lambda_np1))
        #print(mu3 -self.mui[2], self.li[2] - (1./3. * lambda_np1 + mu4))
        #print(mu2 -self.mui[1], self.li[1] - (1./3. * lambda_np1 + 1./2. * mu3))
        #print(mu1 - self.mui[0], self.li[0] - (1./6. * lambda_np1 + 1./2. * mu2))


        #compute grad
        ts_grad = np.zeros(self.dynamics.get_param_size())
        for i in range(self.nstages):
            jac_params = self.dynamics.jac_params(self.Yi[i], self.ti[i], params)
            ts_grad[:] = ts_grad[:] - dt * (self.li[i].T).dot(jac_params)


#RK4 SPECIFIC



        #ts_grad1 = np.zeros(self.dynamics.get_param_size())
        #ts_grad1 = ts_grad1 - self.dt * ((1./2.*mu2 + 1./6.*lambda_np1).T).dot(self.dynamics.jac_params(self.Yi[0], self.ti[0], params))
        #ts_grad1 = ts_grad1 - self.dt * ((1./2.*mu3 + 1./3.*lambda_np1).T).dot(self.dynamics.jac_params(self.Yi[1], self.ti[1], params))
        #ts_grad1 = ts_grad1 - self.dt * ((mu4 + 1./3.*lambda_np1).T).dot(self.dynamics.jac_params(self.Yi[2], self.ti[2], params))
        #ts_grad1 = ts_grad1 - self.dt * ((1./6.*lambda_np1).T).dot(self.dynamics.jac_params(self.Yi[3], self.ti[3], params))

        #print(ts_grad - ts_grad1)

        #lambda_n = lambda_np1 + mu1 + mu2 + mu3 + mu4
        #compute lambda_n
        delta_lambda = np.zeros(self.dynamics.get_x_size())
        for i in range(self.nstages):
            #lambda_np1 = lambda_np1 + self.mui[i][:]
            delta_lambda = delta_lambda + self.mui[i][:]
#Euler specific
        #print(lambda_np1.shape)
        #print(self.dynamics.jac_params(self.Yi[0], self.ti[0], params).shape)
        #ts_grad = -self.dt * (lambda_np1.T).dot(self.dynamics.jac_params(self.Yi[0], self.ti[0], params))


        #print(self.Yi)
        #print(self.ti)
        #for i in range(self.nstages):
        #    print(self.dynamics.jacT_x(self.Yi[i], self.ti[i], params))
        #print(self.mui)
        #print(ts_grad, lambda_np1, params)
        #return ts_grad, lambda_np1
        return ts_grad, delta_lambda

class Euler(_ExplicitRK):
    def __init__(self, dynamics):
        A = None
        c = None
        b = np.array([1.,])
        _ExplicitRK.__init__(self, dynamics, A, c, b, 1)

class RK4(_ExplicitRK):
    def __init__(self, dynamics):
        A = np.array([[1./2.,0,0],[0,1./2.,0],[0,0,1.]])
        c = np.array([0.5,0.5,1])
        b = np.array([1./6.,1./3.,1./3.,1./6.])
        _ExplicitRK.__init__(self, dynamics, A, c, b, 4)

class KGRK2(_ExplicitRK):
    def __init__(self, dynamics, nstages):
        A = SOMETHING
        c = SOMETHING
        b = SOMETHING
        _ExplicitRK.__init__(self, dynamics, A, c, b, nstages)

class KGRK3(_ExplicitRK):
    def __init__(self, dynamics, nstages):
        A = SOMETHING
        c = SOMETHING
        b = SOMETHING
        _ExplicitRK.__init__(self, dynamics ,A, c, b, nstages)


class _Dynamics():
    def __init__(self):
        pass

class LotkaVolterra(_Dynamics):

    def rhs(self, x, t, params):
        x1 = params[0]*x[0] - params[1]*x[0]*x[1]
        x2 = params[2]*x[0]*x[1] - params[3]*x[1]
        return np.array([x1,x2])

    def jac_x(self, x, t, params):
        return np.array([[params[0]-params[1]*x[1],-params[1]*x[0]],[params[2]*x[1],params[2]*x[0]-params[3]]])

    def jacT_x(self, x, t, params):
        return np.array([[params[0]-params[1]*x[1],params[2]*x[1]],[-params[1]*x[0],params[2]*x[0]-params[3]]])
        #self.jac_x(x,t,params).T

    def jac_params(self, x, t, params):
        return np.array([[x[0],-x[0]*x[1],0,0],[0,0,x[0]*x[1],-x[1]]])

    def jacT_params(self, x, t, params):
        return self.jac_params(x,t,params).T
        #np.array([[x[0],0],[-x[0]*x[1], 0],[0,x[0]*x[1]],[0,-x[1]]])

    def get_x_size(self):
        return 2

    def get_param_size(self):
        return 4

    def get_param_bounds(self):
        return (1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)

class LogisticEquation(_Dynamics):

    def rhs(self, x, t, params):
        x1 = params[0]*x[0] * (1. - x[0]/params[1])
        return np.array([x1,])

    def jac_x(self, x, t, params):
        return np.array([[params[0]*(1.-2.*x[0]/params[1]),]])

    def jacT_x(self, x, t, params):
        return np.array([[params[0]*(1.-2.*x[0]/params[1]),]])
        #self.jac_x(x,t,params).T

    def jac_params(self, x, t, params):
        return np.array([[x[0] - x[0]*x[0]/params[1],params[0]*x[0]*x[0]/params[1]/params[1]]])

    def jacT_params(self, x, t, params):
        return self.jac_params(x,t,params).T
        #np.array([[x[0],0],[-x[0]*x[1], 0],[0,x[0]*x[1]],[0,-x[1]]])

    def get_x_size(self):
        return 1

    def get_param_size(self):
        return 2

    def get_param_bounds(self):
        return (1e-6, None), (1e-6, None)

class _Objective():
    pass


class ODEConstraint():
    def __init__(self, timestepper):
        self.timestepper = timestepper

    def multistep(self):
        pass

#EVENTUALLY ADD REGULARIZED OBJECTIVE ALSO
class L2Objective(_Objective):
    def __init__(self, data_blocks, data_blocks_nsteps, t_blocks, xsize, nparams):
        self.data_blocks = data_blocks
        self.data_blocks_nsteps = data_blocks_nsteps
        self.nparams = nparams
        self.xsize = xsize
        self.t_blocks = t_blocks
        self.num_data_blocks = len(self.data_blocks)

    def jac_x(self, x, params):
        jac = np.zeros((self.xsize,1))
        jac[:,0] = x
        return jac

    def jac_params(self, x, params):
        return np.zeros(self.nparams)

    def jacT_x(self, x, params):
        jacT = np.zeros((1,self.xsize))
        jacT[0,:] = x
        return jacT

    def jacT_params(self, x, params):
        return np.zeros((1,self.nparams))

    def evaluate(self, soln_blocks, params):
        l2loss = 0
        for i in range(self.num_data_blocks):
            residual = soln_blocks[i][-1,:] - self.data_blocks[i][1,:]
            l2loss = l2loss +  0.5 * np.sum(residual**2)
        return l2loss


class _ODEConstrainedOptimization():
    def __init__(self, timestepper, objective, dt):
        self.timestepper = timestepper
        self.objective = objective
        self.states = []
        self.ts = []
        for i in range(self.objective.num_data_blocks):
            self.states.append(np.zeros((self.objective.data_blocks_nsteps[i]+1,self.timestepper.dynamics.get_x_size())))
            self.ts.append(np.zeros(self.objective.data_blocks_nsteps[i]+1))
        self.dt = dt

    def optimize(self, params, method='L-BFGS-B',): #cg, bfgs
        res = sp.optimize.minimize(self.obj, params,
            method=method, bounds=self.timestepper.dynamics.get_param_bounds(), options={'disp': True, 'maxiter': 200}, jac=self.jac, hessp=self.hessp)
        print('optimizer success', res.success, res.status, res.message)
        print('optimizer nits', res.nit)
        print('optimizer num func evals', res.nfev)
        print('optimizer num jac evals', res.njev)
        return res.x

    def obj(self, params):
        for i in range(self.objective.num_data_blocks):
            self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params, self.objective.t_blocks[i][0], self.objective.data_blocks[i][0,:], self.dt)
        return self.objective.evaluate(self.states, params)


#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, params):
        grad = np.zeros(self.objective.nparams)

        for i in range(self.objective.num_data_blocks):
            #forward pass with params to populate state
            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params, self.objective.t_blocks[i][0], self.objective.data_blocks[i][0,:], self.dt)
            #print(self.states[i])

            #adjoint pass with params
            lambda_n = self.objective.data_blocks[i][1,:] - self.states[i][-1,:]
            #print(k,lambda_n)
            for n in range(self.objective.data_blocks_nsteps[i],0,-1):
                #take a forward step to populate stage values within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
                _ = self.timestepper.take_forward_step(self.dt, self.ts[i][n-1], self.states[i][n-1,:], params)

                #take an adjoint step
                #ts_grad, lambda_n = self.timestepper.take_adjoint_step(lambda_n, params)
                ts_grad, delta_lambda = self.timestepper.take_adjoint_step(self.dt, lambda_n, params)

                #sum gradient
                grad[:] = grad[:] + ts_grad
                lambda_n[:] = lambda_n[:] + delta_lambda
                #print(k,n,lambda_n)

            #ADD DELTA CALCULATION AT THE END
            #grad[:] = grad[:] + self.timesteppher

#HOW SHOULD X DEPENDENCE ACTUALLY BE HANDLED HERE?
#REALLY IT IS X AT THE END OF EACH SIMULATION BLOCK!
        grad[:] = grad[:] + self.objective.jac_params(None, params)
        return grad

    def hessp(self, x, params):
        pass
