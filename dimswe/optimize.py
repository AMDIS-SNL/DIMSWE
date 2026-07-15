from firedrake import assemble, inner
import numpy as np
from .numpy_helpers import set_mixed_function_from_flattened_array, create_flattened_numpy_arr_from_mixed_function

def create_states(model, nsteps):
    xns = []
    xn_subs = []
    tns = []
    for i in range(nsteps+1):
        t = model.get_t_var()
        xn, xn_sub, x_split = model.get_full_var('x'+str(i), split_x_and_aux=True)
        xns.append(xn)
        xn_subs.append(xn_sub)
        tns.append(t)
    return xns, xn_subs, tns

def compute_states(xns, xn_subs, tns, model, timestepper, nsteps, dt, x0, t0):
    model.restart(xns[0], x0, tns[0], t0)
    for n in range(nsteps):
        timestepper.take_forward_step(xns[n+1], xn_subs[n], xns[n], tns[n], dt)
        tns[n+1].assign(tns[n] + dt)
    #print(xns[-1][0].dat.data[0])

def compute_state_block(model, timestepper, nblocks, nsteps, dt, x0, t0):
    xns = []
    xn_subs = []
    tns = []
    steps = []
    for k in range(nblocks):
        xn, xn_sub, tn = create_states(model, nsteps)
        if (k==0):
            compute_states(xn, xn_sub, tn, model, timestepper, nsteps, dt, x0, t0)
        else:
            compute_states(xn, xn_sub, tn, model, timestepper, nsteps, dt, xns[k-1][-1], tns[k-1][-1])
        xns.append(xn[::nsteps])
        xn_subs.append(xn_sub[::nsteps])
        tns.append(tn[::nsteps])
        steps.append(nsteps)
    return xns, xn_subs, steps, tns


class _Objective:
    pass

#EVENTUALLY ADD REGULARIZED OBJECTIVE ALSO
class L2Objective(_Objective):
    def __init__(self, data_blocks, t_blocks, coeff, nsteps, dx):
        self.data_blocks = data_blocks
        self.coeff = coeff
        self.t_blocks = t_blocks
        self.num_data_blocks = len(self.data_blocks)
        self.nsteps = nsteps
        self.dx = dx

#THIS IS A LITTLE WRONG, AND SHOULD REALLY BE PART OF ConstrainedOptimizer...
####
    def jac_x(self, x, params):
        return x

    def jacT_x(self, x, params):
        return x
###

    def jac_params(self, x, params):
        return 0


    def jacT_params(self, x, params):
        return 0

#THIS IS A MASS MATRIX WEIGHTED INNER PRODUCT!
#SO BE VERY CAREFUL WHEN WE SET LAMBDA I THINK?
    def evaluate(self, block_id, soln, tns):
#THIS INCLUDES AUX VARIABLES FOR DIRK AND IRK CASES- PROBABLY TRY TO AVOID THOSE IF POSSIBLE?
        residual = soln[0] - self.data_blocks[block_id][1][0]
        res0 = soln[0].dat.data[0] - self.data_blocks[block_id][1][0].dat.data[0]
        res1 = soln[0].dat.data[1] - self.data_blocks[block_id][1][0].dat.data[1]
        res2 = soln[0].dat.data[2] - self.data_blocks[block_id][1][0].dat.data[2]
        full_res = np.sum(res0**2) + np.sum(res1**2) + np.sum(res2**2)
        print(0.5 * full_res)
        #print(soln[0].dat.data[0] - self.data_blocks[block_id][1][0].dat.data[0])
        #print(soln[0].dat.data[1] - self.data_blocks[block_id][1][0].dat.data[1])
        #print(soln[0].dat.data[2] - self.data_blocks[block_id][1][0].dat.data[2])
        print(assemble(0.5*inner(residual,residual)*self.dx))

        #return assemble(0.5*inner(residual,residual)*self.dx)
        return 0.5*full_res

class _ODEConstrainedOptimization():
    def __init__(self, model, timestepper, objective, dt):
        self.model = model
        self.timestepper = timestepper
        self.objective = objective
#THIS IS A CHECKPOINTING/STORAGE CHOICE
        self.states, self.states_sub, self.tns = create_states(model, self.objective.nsteps)
        self.dt = dt
        self.lambda_n, _, _ = self.model.get_x_var('lambda')
        self.grad_coeff, _, _ = self.model.get_coeff_var('grad_coeff')
        self.grad_ic, _, _ = self.model.get_x_var('grad_ic')
        self.ic, _, _ = self.model.get_x_var('ic')

        self.grad_params = np.zeros(self.model.get_coeff_size())
        self.grad_ics = np.zeros((self.objective.num_data_blocks, self.model.get_x_size()))



    def optimize(self, initial_guess, method='L-BFGS-B', opt_type='params', params0=None): #cg, bfgs

        if opt_type == 'params':
            obj = lambda params: self.obj(params, None)
            jac = lambda params: self.jac(params, None)
            bounds = self.timestepper.dynamics.get_param_bounds()
        elif opt_type == 'ics':
            obj = lambda ics: self.obj(None, np.reshape(ics, (self.objective.num_data_blocks, self.model.get_x_size())), params0=params0)
            jac = lambda ics: self.jac(None, np.reshape(ics, (self.objective.num_data_blocks, self.model.get_x_size())), params0=params0)
            bounds = self.timestepper.dynamics.get_ic_bounds() * self.objective.num_data_blocks
        elif opt_type == 'params+ics':
            obj = lambda params_plus_ic: self.obj(params_plus_ic[:self.objective.nparams], np.reshape(params_plus_ic[self.objective.nparams:], (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())))
            jac = lambda params_plus_ic: self.jac(params_plus_ic[:self.objective.nparams], np.reshape(params_plus_ic[self.objective.nparams:], (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())))
            bounds = self.timestepper.dynamics.get_param_bounds() + self.timestepper.dynamics.get_ic_bounds() * self.objective.num_data_blocks

#HOW DO PARAMETER BOUNDS WORK FOR LARGE SCALE PROBLEMS LIKE THIS?
#CLEARLY THE LIST/ARRAY IS A PROBABLY A BAD IDEA?
        res = sp.optimize.minimize(obj, initial_guess, method=method, bounds=None, options={'disp': True, 'maxiter': 200}, jac=jac, hessp=self.hessp)
        print('optimizer success', res.success, res.status, res.message)
        print('optimizer nits', res.nit)
        print('optimizer num func evals', res.nfev)
        print('optimizer num jac evals', res.njev)
        return res.x

    def obj(self, params_arr, ics_arr, params0=None):
        if params_arr is None:
            self.timestepper.set_coeff(params0)
        else:
            self.timestepper.set_numpy_coeff(params_arr)

        l2loss = 0.0
        for i in range(self.objective.num_data_blocks):
            #self.timestepper.reset_internal_vars()
            #self.states[0][0].assign(0)
            #self.states[0][1].assign(0)
            #self.states[1][0].assign(0)
            #self.states[1][1].assign(0)
            #print(len(self.states))
            if ics_arr is None:
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[i][0], self.objective.t_blocks[i][0])
            else:
                set_mixed_function_from_flattened_array(self.ic, ics_arr[i,:])
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.ic, self.objective.t_blocks[i][0])
            #print(self.objective.data_blocks[i][1][0].dat.data[0] - self.states[-1][0].dat.data[0])
            #print(self.objective.data_blocks[i][1][0].dat.data[1] - self.states[-1][0].dat.data[1])
            #print(self.objective.data_blocks[i][1][0].dat.data[2] - self.states[-1][0].dat.data[2])
            l2loss += self.objective.evaluate(i, self.states[-1], self.tns[-1])
        return l2loss

#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, params_arr, ics_arr, params0=None):
        self.grad_coeff.assign(0)
        self.grad_params[:] = 0.0
        self.grad_ics[:] = 0.0

        if params_arr is None:
            self.timestepper.set_coeff(params0)
        else:
            self.timestepper.set_numpy_coeff(paramsarr)

        for i in range(self.objective.num_data_blocks):
            #forward pass with params to populate states
#ADD CORRECT IC BEHAVIOUR HERE!
            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            if ics_arr is None:
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[i][0], self.objective.t_blocks[i][0])
            else:
                set_mixed_function_from_flattened_array(self.ic, ics_arr[i,:])
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.ic, self.objective.t_blocks[i][0])


            #adjoint pass with params
#THIS REALLY SHOULD COME FROM DERIVATIVE OF OBJECTIVE!
#NOT ACTUALLY SURE THIS IS CORRECT FOR CHOSEN OBJECTIVE?
            self.lambda_n.assign(self.objective.data_blocks[i][1][0] - self.states[-1][0])
            #self.lambda_n.project(self.objective.data_blocks[i][1][0] - self.states[-1][0])
            #print('lambda_n before', self.lambda_n.dat.data[0])
            for n in range(self.objective.nsteps,0,-1):
                #take a forward step to populate stage values within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
                self.timestepper.take_forward_step(self.states[n], self.states_sub[n], self.states[n-1], self.tns[n-1], self.dt)

                #take an adjoint step
                self.timestepper.take_adjoint_step(self.grad_coeff, self.lambda_n, self.lambda_n, self.tns[n], self.dt)
            #print('lambda_n after', self.lambda_n.dat.data[0])
            #print('grad', self.grad.dat.data[0])

            self.grad_ics[i,:] = -create_flattened_numpy_arr_from_mixed_function(self.lambda_n)

        self.grad_params[:] = self.grad_params[:] + create_flattened_numpy_arr_from_mixed_function(self.grad_coeff)

#SET GRAD IC!!!

            #grad[:] = grad[:] + self.timesteppher

#MAYBE THIS FUNCTION RETURNS A JACOBIAN WRT PARAMS, AND A JACOBIAN WRT ICS?
#YES THIS IS THE WAY

#THEN SOME OTHER FUNCTION CAN COMBINE THEM AS NEEDED, DEPENDING ON WHAT WE ARE OPTIMIZING WRT TO!

#ADD THIS ALSO!
        #self.grad = self.grad + self.objective.jac_params(self.states[-1], params)
        if ics_arr is None:
            return self.grad_params
        elif params_arr is None:
            return np.ravel(self.grad_ics)
        else:
            return np.hstack([self.grad_params, np.ravel(self.grad_ics)])

    def hessp(self, x, params):
        pass
