from firedrake import assemble
import numpy as np

class _Objective:
    pass

#EVENTUALLY ADD REGULARIZED OBJECTIVE ALSO
class L2Objective(_Objective):
    def __init__(self, data_blocks, t_blocks, coeff):
        self.data_blocks = data_blocks
        self.coeff = coeff
        self.t_blocks = t_blocks
        self.num_data_blocks = len(self.data_blocks)

#THIS IS A LITTLE WRONG, AND SHOULD REALLY BE PART OF ContrainedOptimizer...
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

    def evaluate(self, block_id, soln, params):
        residual = soln - self.data_blocks[block_id][1]
        return assemble(0.5*residual*residual*self.dx)


class _ODEConstrainedOptimization():
    def __init__(self, timestepper, objective, dt):
        self.timestepper = timestepper
        self.objective = objective
        self.states = []
        self.ts = []
#THIS IS A CHECKPOINTING/STORAGE CHOICE
        for i in range(self.objective.num_steps+1):
            self.states.append(self.timestepper.dynamics.get_full_var('x'+str(i))[0])
        self.ts = np.zeros(self.objective.num_steps+1)
        self.dt = dt
        self.lambda_n = self.timestepper.dynamics.get_x_var('lambda')
        self.grad = np.zeros(self.objective.nparams)

#HOW DO PARAMETER BOUNDS WORK FOR LARGE SCALE PROBLEMS LIKE THIS?
#CLEARLY THE LIST/ARRAY IS A PROBABLY A BAD IDEA?
    def optimize(self, params, method='L-BFGS-B',): #cg, bfgs
        res = sp.optimize.minimize(self.obj, params,
            method=method, bounds=self.timestepper.dynamics.get_param_bounds(), options={'disp': True, 'maxiter': 200}, jac=self.jac, hessp=self.hessp)
        print('optimizer success', res.success, res.status, res.message)
        print('optimizer nits', res.nit)
        print('optimizer num func evals', res.nfev)
        print('optimizer num jac evals', res.njev)
        return res.x

    def obj(self, params):
        self.dynamics.set_coeff(params)
        l2loss = 0.0
        for i in range(self.objective.num_data_blocks):
            self.timestepper.compute_states(self.states, self.objective.num_steps, params, self.objective.t_blocks[i][0], self.objective.data_blocks[i][0], self.dt)
            l2loss += self.objective.evaluate(i, self.states[-1], params)
        return l2loss

#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, params):
        grad.zero()
        self.timestepper.set_coeff(params)

        for i in range(self.objective.num_data_blocks):
            #forward pass with params to populate states
            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            self.timestepper.compute_states(self.states, self.objective.num_steps, params, self.objective.t_blocks[i][0], self.objective.data_blocks[i][0], self.dt)


            #adjoint pass with params
#THIS REALLY SHOULD COME FROM DERIVATIVE OF OBJECTIVE!
#NOT ACTUALLY SURE THIS IS CORRECT FOR CHOSEN OBJECTIVE?
            self.lambda_n.assign(self.objective.data_blocks[i][1] - self.states[-1])

            for n in range(self.objective.num_steps,0,-1):
                #take a forward step to populate stage values within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
                self.timestepper.take_forward_step(self.states[n], self.states_sub[n], self.states[n-1], self.ts[n-1], self.dt)

                #take an adjoint step
                self.timestepper.take_adjoint_step(self.grad, self.lambda_n, self.lambda_n, self.ts[n], self.dt)

#ADD DELTA CALCULATION AT THE END
#THIS IS ACTUALLY NEEDED FOR IC OPTIMIZATION
            #grad[:] = grad[:] + self.timesteppher

        self.grad[:] = self.grad[:] + self.objective.jac_params(self.states[-1], params)
        return self.grad

    def hessp(self, x, params):
        pass
