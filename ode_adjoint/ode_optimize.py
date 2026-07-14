import numpy as np
import scipy as sp

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
        self.delta_lambda = np.zeros(self.timestepper.dynamics.get_x_size())
        self.ts_grad = np.zeros(self.objective.nparams)

    def optimize(self, initial_guess, method='L-BFGS-B', opt_type='params', params0=None): #cg, bfgs
        if opt_type == 'params':
            obj = lambda params: self.obj(params, None)
            jac = lambda params: self.jac(params, None)
            bounds = self.timestepper.dynamics.get_param_bounds()
        elif opt_type == 'ics':
            obj = lambda ics: self.obj(None, np.reshape(ics, (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())), params0=params0)
            jac = lambda ics: self.jac(None, np.reshape(ics, (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())), params0=params0)
            bounds = self.timestepper.dynamics.get_ic_bounds() * self.objective.num_data_blocks
        elif opt_type == 'params+ics':
            obj = lambda params_plus_ic: self.obj(params_plus_ic[:self.objective.nparams], np.reshape(params_plus_ic[self.objective.nparams:], (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())))
            jac = lambda params_plus_ic: self.jac(params_plus_ic[:self.objective.nparams], np.reshape(params_plus_ic[self.objective.nparams:], (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())))
            bounds = self.timestepper.dynamics.get_param_bounds() + self.timestepper.dynamics.get_ic_bounds() * self.objective.num_data_blocks

        res = sp.optimize.minimize(obj, initial_guess,
        method=method, bounds=bounds,
        options={'disp': True, 'maxiter': 200},
        jac=jac, hessp=self.hessp)
        #'gtol':1e-08, 'ftol':1000.* np.finfo(float).eps
        #'gtol':1e-05, 'ftol':10000000.* np.finfo(float).eps

        #res = sp.optimize.minimize(obj, initial_params,
        #method=method, options={'disp': True, 'maxiter': 200},)
        #print(res)
        print('optimizer success', res.success, res.status, res.message)
        print('optimizer nits', res.nit)
        #print('optimizer num func evals', res.nfev)
        #print('optimizer num jac evals', res.njev)
        return res.x

    def obj(self, params, ics, params0=None):
        for i in range(self.objective.num_data_blocks):
            if ics is None:
                self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params, self.objective.t_blocks[i][0], self.objective.data_blocks[i][0,:], self.dt)
            elif params is None:
                self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params0, self.objective.t_blocks[i][0], ics[i,:], self.dt)
            else:
                self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params, self.objective.t_blocks[i][0], ics[i,:], self.dt)

        return self.objective.evaluate(self.states, params)


#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, params, ics, params0=None):
        grad_params = np.zeros(self.objective.nparams)
        grad_ics = np.zeros((self.objective.num_data_blocks,self.timestepper.dynamics.get_x_size()))

        for i in range(self.objective.num_data_blocks):
            #forward pass to populate state
            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            if ics is None:
                self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params, self.objective.t_blocks[i][0], self.objective.data_blocks[i][0,:], self.dt)
                model_params = params
            elif params is None:
                self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params0, self.objective.t_blocks[i][0], ics[i,:], self.dt)
                model_params = params0
            else:
                self.states[i][:], self.ts[i][:] = self.timestepper.compute_state(self.objective.data_blocks_nsteps[i], params, self.objective.t_blocks[i][0], ics[i,:], self.dt)            #print(self.states[i])
                model_params = params

            #adjoint pass
            lambda_n = self.objective.data_blocks[i][1,:] - self.states[i][-1,:]
            #print(k,lambda_n)
            for n in range(self.objective.data_blocks_nsteps[i],0,-1):
                #take a forward step to populate stage values within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
                self.timestepper.take_forward_step(self.states[i][n,:], self.dt, self.ts[i][n-1], self.states[i][n-1,:], model_params)

                #take an adjoint step
                self.timestepper.take_adjoint_step(self.ts_grad, self.delta_lambda, self.dt, self.ts[i][n], lambda_n, model_params)

                #sum gradient
                grad_params[:] = grad_params[:] + self.ts_grad
                lambda_n[:] = lambda_n[:] + self.delta_lambda
                #print(k,n,lambda_n)

        #SIGN?
            grad_ics[i,:] = -lambda_n

#HOW SHOULD X DEPENDENCE ACTUALLY BE HANDLED HERE?
#REALLY IT IS X AT THE END OF EACH SIMULATION BLOCK!
        #grad[:] = grad[:] + self.objective.jac_params(None, params)
        #print(params)
        #print(ics)
        if ics is None:
            return grad_params
        elif params is None:
            return np.ravel(grad_ics)
        else:
            return np.hstack([grad_params, np.ravel(grad_ics)])

    def hessp(self, x, params):
        pass
