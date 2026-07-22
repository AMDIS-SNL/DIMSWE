from firedrake import assemble, inner, norm, LinearVariationalProblem, LinearVariationalSolver
import numpy as np
from .numpy_helpers import set_mixed_function_from_flattened_array, create_flattened_numpy_arr_from_mixed_function
import scipy as sp

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
#WHY DO WE NEED TO DO?
#THERE IS SOME SORT OF DATA BEING CARRIED BETWEEN STEPS...
        timestepper.reset_internal_vars()
        timestepper.take_forward_step(xns[n+1], xn_subs[n+1], xns[n], tns[n], dt)
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
    def __init__(self, data_blocks, t_blocks, nsteps, dx):
        self.data_blocks = data_blocks
        self.t_blocks = t_blocks
        self.num_data_blocks = len(self.data_blocks)
        self.nsteps = nsteps
        self.dx = dx

#THIS IS A LITTLE WRONG, AND SHOULD REALLY BE PART OF ConstrainedOptimizer...
####
    def jac_x(self, x, coeffs):
        return x

    def jacT_x(self, x, coeffs):
        return x
###

    def jac_coeffs(self, x, coeffs):
        return 0


    def jacT_coeffs(self, x, coeffs):
        return 0

#THIS IS A MASS MATRIX WEIGHTED INNER PRODUCT!
#SO BE VERY CAREFUL WHEN WE SET LAMBDA I THINK?
    def evaluate(self, block_id, soln, tns):
#THIS INCLUDES AUX VARIABLES FOR DIRK AND IRK CASES- PROBABLY TRY TO AVOID THOSE IF POSSIBLE?
        residual = soln[0] - self.data_blocks[block_id][1][0]
        #res0 = soln[0].dat.data[0] - self.data_blocks[block_id][1][0].dat.data[0]
        #res1 = soln[0].dat.data[1] - self.data_blocks[block_id][1][0].dat.data[1]
        #res2 = soln[0].dat.data[2] - self.data_blocks[block_id][1][0].dat.data[2]
        #full_res = np.sum(res0**2) + np.sum(res1**2) + np.sum(res2**2)
        #print(0.5 * full_res)
        #print(soln[0].dat.data[0] - self.data_blocks[block_id][1][0].dat.data[0])
        #print(soln[0].dat.data[1] - self.data_blocks[block_id][1][0].dat.data[1])
        #print(soln[0].dat.data[2] - self.data_blocks[block_id][1][0].dat.data[2])
        #print(assemble(0.5*inner(residual,residual)*self.dx))

        return assemble(0.5*inner(residual,residual)*self.dx)
        #return 0.5*full_res

class _ODEConstrainedOptimization():
    def __init__(self, model, timestepper, objective, dt):
        self.model = model
        self.timestepper = timestepper
        self.objective = objective
#THIS IS A CHECKPOINTING/STORAGE CHOICE
        self.states, self.states_sub, self.tns = create_states(model, self.objective.nsteps)
        self.dt = dt
        self.lambda_np1, _, _ = self.model.get_x_var('lambda_np1')
        self.delta_lambda, _, _ = self.model.get_x_var('delta_lambda')
        #self.grad_coeff, _, _ = self.model.get_coeff_var('grad_coeff')
        self.delta_grad_coeff, _, _ = self.model.get_coeff_var('delta_grad_coeff')
        self.grad_ic, _, _ = self.model.get_x_var('grad_ic')
        self.ic, _, _ = self.model.get_x_var('ic')

        self.old_state, self.old_state_sub, _ = self.model.get_x_var('old_state')


        self.xtest, _ = self.model.get_x_test_vars()

        self.grad_coeffs = np.zeros(self.model.get_coeff_size())
        self.grad_ics = np.zeros((self.objective.num_data_blocks, self.model.get_x_size()))

        self.initial_mismatch, _, _ = self.model.get_x_var('initial_mismatch')
        xtest, _ = self.model.get_x_test_vars()
        xtrial, _ = self.model.get_x_trial_vars()
        a = inner(xtest, xtrial)* self.model.spaces.dx
        L = inner(xtest, self.initial_mismatch)*self.model.spaces.dx
        lambda_np1_projector_problem = LinearVariationalProblem(a, L, self.lambda_np1, constant_jacobian=True)
        self.lambda_np1_projector = LinearVariationalSolver(lambda_np1_projector_problem,
            solver_parameters={ 'ksp_type': 'preonly', 'pc_type' : 'lu'}, #{ 'ksp_type': 'cg', 'pc_type' : 'jacobi',},
                options_prefix='opt-proj')


    def optimize(self, initial_guess, method='L-BFGS-B', opt_type='coeffs', use_jacobian=True, coeffs0=None): #cg, bfgs

#HOW DO WE CORRECTLY HANDLE BOUNDS HERE?
        if opt_type == 'coeffs':
            obj = lambda coeffs: self.obj(coeffs, None)
            jac = lambda coeffs: self.jac(coeffs, None)
            #bounds = self.model.get_param_bounds()
        elif opt_type == 'ics':
            obj = lambda ics: self.obj(None, np.reshape(ics, (self.objective.num_data_blocks, self.model.get_x_size())), coeffs0=coeffs0)
            jac = lambda ics: self.jac(None, np.reshape(ics, (self.objective.num_data_blocks, self.model.get_x_size())), coeffs0=coeffs0)
            #bounds = self.model.get_ic_bounds() * self.objective.num_data_blocks
        elif opt_type == 'coeffs+ics':
            obj = lambda coeffs_plus_ic: self.obj(coeffs_plus_ic[:self.model.get_coeff_size()], np.reshape(coeffs_plus_ic[self.model.get_coeff_size():], (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())))
            jac = lambda coeffs_plus_ic: self.jac(coeffs_plus_ic[:self.model.get_coeff_size()], np.reshape(coeffs_plus_ic[self.model.get_coeff_size():], (self.objective.num_data_blocks, self.timestepper.dynamics.get_x_size())))
            #bounds = self.model.get_param_bounds() + self.model.get_ic_bounds() * self.objective.num_data_blocks

#HOW DO PARAMETER BOUNDS WORK FOR LARGE SCALE PROBLEMS LIKE THIS?
#CLEARLY THE LIST/ARRAY IS A PROBABLY A BAD IDEA?
        if use_jacobian:
            res = sp.optimize.minimize(obj, initial_guess,
                method=method, bounds=None,
                options={'disp': True, 'maxiter': 50000, 'maxfun': 50000},
                jac=jac, hessp=self.hessp)
        else:
            res = sp.optimize.minimize(obj, initial_guess,
                method=method, bounds=None,
                options={'disp': True, 'maxiter': 50000, 'maxfun': 50000},)
        print('optimizer success', res.success, res.status, res.message)
        print('optimizer nits', res.nit)
        print('optimizer num func evals', res.nfev)
        if use_jacobian:
            print('optimizer num jac evals', res.njev)
        return res.x

    def obj(self, coeffs_arr, ics_arr, coeffs0=None):
        if coeffs_arr is None:
            self.timestepper.set_coeff(coeffs0)
        else:
            self.timestepper.set_numpy_coeff(coeffs_arr)

        l2loss = 0.0
        #print('calling obj with coeff norm', self.model.norm(self.timestepper.coeff))
        for i in range(self.objective.num_data_blocks):
#THIS IS REQUIRED TO GET REPEATABILITY FOR OPT
#WHY?????
            self.timestepper.reset_internal_vars()

            if ics_arr is None:
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[i][0], self.objective.t_blocks[i][0])
            else:
                set_mixed_function_from_flattened_array(self.ic, ics_arr[i,:])
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.ic, self.objective.t_blocks[i][0])
            l2loss += self.objective.evaluate(i, self.states[-1], self.tns[-1])
        return l2loss

#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, coeffs_arr, ics_arr, coeffs0=None):
        if self.model.has_coeff():
            self.grad_coeffs[:] = 0.0
        self.grad_ics[:] = 0.0

        if coeffs_arr is None:
            self.timestepper.set_coeff(coeffs0)
        else:
            self.timestepper.set_numpy_coeff(coeffs_arr)

        #print('calling jac with coeff norm', self.model.norm(self.timestepper.coeff))

        #print('ts coeff norm', self.model.norm(self.timestepper.coeff))
        for i in range(self.objective.num_data_blocks):

#THIS IS REQUIRED TO GET REPEATABILITY FOR JAC
#WHY?????
            self.timestepper.reset_internal_vars()

            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            if ics_arr is None:
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[i][0], self.objective.t_blocks[i][0])
            else:
                set_mixed_function_from_flattened_array(self.ic, ics_arr[i,:])
                compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.ic, self.objective.t_blocks[i][0])

            self.initial_mismatch.assign(self.states[-1][0] - self.objective.data_blocks[i][1][0])
            self.lambda_np1_projector.solve()
            #print('initial mismatch norm', self.model.norm(self.initial_mismatch))
            #print('lambda_np1 norm', self.model.norm(self.lambda_np1))
            #print('state data end diff', self.model.norm(self.states[-1][0] - self.objective.data_blocks[i][1][0]))

            #self.timestepper.reset_internal_vars()

            self.grad_ics[i, :] = create_flattened_numpy_arr_from_mixed_function(assemble(inner(self.xtest, self.states[-1][0] - self.objective.data_blocks[i][1][0])*self.model.spaces.dx))
            for n in range(self.objective.nsteps,0,-1):

                #IF THIS IS NOT DONE THEN THE STATES[N] CHANGES!
                #WHY??
                #BUT THIS ONLY FIXES THE LAST ONE, THE OTHERS ARE STILL BROKEN
                self.timestepper.reset_internal_vars()

                #self.old_state.assign(self.states[n][0])

                #take a forward step to populate internal variables within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE\
                self.timestepper.take_forward_step(self.states[n], self.states_sub[n], self.states[n-1], self.tns[n-1], self.dt)
                #self.timestepper.take_forward_step([self.old_state,], self.old_state_sub, self.states[n-1], self.tns[n-1], self.dt)

                delta_lambda_rhs, delta_grad_rhs = self.timestepper.take_adjoint_step(self.delta_grad_coeff, self.delta_lambda, self.lambda_np1, self.tns[n], self.dt)
                self.lambda_np1.assign(self.lambda_np1 + self.delta_lambda)

                #print('state diff ' + str(n), self.model.norm(self.old_state - self.states[n][0]))

                if self.model.has_coeff():
                    #print(delta_grad_rhs.dat.data)
                    self.grad_coeffs[:] = self.grad_coeffs[:] + create_flattened_numpy_arr_from_mixed_function(delta_grad_rhs)
            #print('state data end diff', self.model.norm(self.states[-1][0] - self.objective.data_blocks[i][1][0]))

                self.grad_ics[i, :] = self.grad_ics[i, :] + create_flattened_numpy_arr_from_mixed_function(delta_lambda_rhs)

        if ics_arr is None:
            return self.grad_coeffs
        elif coeffs_arr is None:
            return np.ravel(self.grad_ics)
        else:
            return np.hstack([self.grad_coeffs, np.ravel(self.grad_ics)])

    def hessp(self, x, params):
        pass
