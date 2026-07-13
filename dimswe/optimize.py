from firedrake import assemble, inner
import numpy as np

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

    def evaluate(self, block_id, soln, tns, params):
        residual = soln[0] - self.data_blocks[block_id][1][0]
        #print(assemble(0.5*inner(residual,residual)*self.dx))
        return assemble(0.5*inner(residual,residual)*self.dx)


class _ODEConstrainedOptimization():
    def __init__(self, model, timestepper, objective, dt):
        self.model = model
        self.timestepper = timestepper
        self.objective = objective
#THIS IS A CHECKPOINTING/STORAGE CHOICE
        self.states, self.states_sub, self.tns = create_states(model, self.objective.nsteps)
        self.dt = dt
        self.lambda_n, _, _ = self.model.get_x_var('lambda')
        self.grad, _, _ = self.model.get_coeff_var('grad')

#HOW DO PARAMETER BOUNDS WORK FOR LARGE SCALE PROBLEMS LIKE THIS?
#CLEARLY THE LIST/ARRAY IS A PROBABLY A BAD IDEA?
    def optimize(self, params, method='L-BFGS-B',): #cg, bfgs
        res = sp.optimize.minimize(self.obj, params,
            method=method, bounds=self.model.get_param_bounds(), options={'disp': True, 'maxiter': 200}, jac=self.jac, hessp=self.hessp)
        print('optimizer success', res.success, res.status, res.message)
        print('optimizer nits', res.nit)
        print('optimizer num func evals', res.nfev)
        print('optimizer num jac evals', res.njev)
        return res.x

    def obj(self, params):
        self.timestepper.set_coeff(params)
        l2loss = 0.0
        for i in range(self.objective.num_data_blocks):
            self.timestepper.reset_internal_vars()
            #self.states[0][0].assign(0)
            #self.states[0][1].assign(0)
            #self.states[1][0].assign(0)
            #self.states[1][1].assign(0)
            #print(len(self.states))
            compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[i][0], self.objective.t_blocks[i][0])
            #print(self.objective.data_blocks[i][1][0].dat.data[0] - self.states[-1][0].dat.data[0])
            #print(self.objective.data_blocks[i][1][0].dat.data[1] - self.states[-1][0].dat.data[1])
            #print(self.objective.data_blocks[i][1][0].dat.data[2] - self.states[-1][0].dat.data[2])
            l2loss += self.objective.evaluate(i, self.states[-1], self.tns[-1], params)
        return l2loss

#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, params):
        self.grad.assign(0)
        self.timestepper.set_coeff(params)

        for i in range(self.objective.num_data_blocks):
            #forward pass with params to populate states
            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[i][0], self.objective.t_blocks[i][0])


            #adjoint pass with params
#THIS REALLY SHOULD COME FROM DERIVATIVE OF OBJECTIVE!
#NOT ACTUALLY SURE THIS IS CORRECT FOR CHOSEN OBJECTIVE?
            self.lambda_n.assign(self.objective.data_blocks[i][1][0] - self.states[-1][0])

            for n in range(self.objective.nsteps,0,-1):
                #take a forward step to populate stage values within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
                self.timestepper.take_forward_step(self.states[n], self.states_sub[n], self.states[n-1], self.tns[n-1], self.dt)

                #take an adjoint step
                self.timestepper.take_adjoint_step(self.grad, self.lambda_n, self.lambda_n, self.tns[n], self.dt)

#ADD DELTA CALCULATION AT THE END
#THIS IS ACTUALLY NEEDED FOR IC OPTIMIZATION
            #grad[:] = grad[:] + self.timesteppher

#ADD THIS ALSO!
        #self.grad = self.grad + self.objective.jac_params(self.states[-1], params)
        return self.grad.dat.data[0]

    def hessp(self, x, params):
        pass
