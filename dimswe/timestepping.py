from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, LinearVariationalProblem, LinearVariationalSolver
from .parameters import overall_solver_parameters
import numpy as np
from firedrake import Constant, inner, grad, TestFunction, derivative, norm, assemble, Function, TestFunctions, split, TrialFunction, adjoint, action, replace
from .numpy_helpers import set_mixed_function_from_flattened_array
from ufl.algorithms import extract_coefficients

class DummySolver():
    def solve(self):
        pass

def get_time_integrator(name, parameters, solver_parameters):
    if name == 'RK4':
        return RK4
    elif name == 'Euler':
        return Euler
    elif name == 'SSPRK3':
        return SSPRK3
    elif name == 'SSPRK43':
        return SSPRK43
    elif name == 'LieSplit':
        timestepper_list = parameters['timestepping']['timestepper_list']
        termlist = parameters['timestepping']['termlist']
        subcycle_list = parameters['timestepping']['subcycle_list']

        return lambda model, logger, coeffs: LieSplittingIntegrator(model, logger, timestepper_list, termlist, subcycle_list, solver_parameters)
    else:
        raise ValueError("time step method " + name + " is unknown")

def get_timestepper(parameters, model, logger, solver_parameters):
    return get_time_integrator(parameters['timestepping']['method'], parameters, solver_parameters)(model, logger, solver_parameters)


def create_linear_solver_from_residual(residual, var, trialvar, constant_jacobian=False, solver_parameters={}, options_prefix=''):
    a = derivative(residual, var, trialvar)
    L = action(a, var) - residual
    problem = LinearVariationalProblem(a, L, var, constant_jacobian=constant_jacobian)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters, options_prefix=options_prefix)
    return solver, a, L

def extract_a_L_from_residual(residual, var, trialvar):
    a = derivative(residual, var, trialvar)
    L = action(a, var) - residual
    return a, L


class TimeStepper():


    def __init__(self, model, logger, solver_parameters, terms='all'):
        self.model = model
        self.logger = logger
        self.terms = terms
        self.dx = model.spaces.dx
        self.solver_parameters = solver_parameters

        self.delta_lambda_var, self.delta_lambda_sub, self.delta_lambda_split = self.model.get_x_var('delta_lambda')
        self.lambda_var, self.lambda_sub, self.lambda_split = self.model.get_x_var('lambda')

        self.coeff, self.coeff_sub, self.coeff_split = self.model.get_coeff_var('coeff')
        self.delta_grad, self.delta_grad_sub, self.delta_grad_split = self.model.get_coeff_var('delta_grad')
        self.grad_test, self.grad_test_subs, self.grad_trial, self.grad_trial_subs = self.model.get_coeff_test_trial_vars()

        self.t = Constant(1.)
        self.dt = Constant(1.)

    def set_coeff(self, coeff_val):
        if self.model.has_coeff():
            self.coeff.assign(coeff_val)

    def set_numpy_coeff(self, coeff_val_arr):
        set_mixed_function_from_flattened_array(self.coeff, coeff_val_arr)

class AdditiveGeneralRK(TimeStepper):
    def __init__(self, model, logger, A1, A2, b1, b2, c1, c2, nstages, solver_parameters, terms='all'):
        TimeStepper.__init__(self, model, logger, solver_parameters, terms=terms)
        self.A1 = A1
        self.b1 = b1
        self.c1 = c1
        self.A2 = A2
        self.b2 = b2
        self.c2 = c2
        self.nstages = nstages
        self.solver_parameters = solver_parameters

        zero_diag1 = not np.any(np.diagonal(self.A1))
        triangular1 = np.allclose(self.A1, np.tril(self.A1))
        self.is_explicit1 = triangular1 and zero_diag1
        self.is_dirk1 = triangular1 and (not zero_diag1)

        zero_diag2 = not np.any(np.diagonal(self.A2))
        triangular2 = np.allclose(self.A2, np.tril(self.A2))
        self.is_explicit2 = triangular2 and zero_diag2
        self.is_dirk2 = triangular2 and (not zero_diag2)

class GeneralRK(TimeStepper):
    def __init__(self, model, logger, A, b, c, nstages, solver_parameters, terms='all'):
        TimeStepper.__init__(self, model, logger, solver_parameters, terms=terms)
        self.A = A
        self.b = b
        self.c = c
        self.nstages = nstages

        zero_diag = not np.any(np.diagonal(self.A))
        triangular = np.allclose(self.A, np.tril(self.A))
        self.is_explicit = triangular and zero_diag
        self.is_dirk = triangular and (not zero_diag)

        self.aux_var_list = self.model.get_aux_var_list(terms=terms)

        self.xk, self.xk_sub, self.xk_split = self.model.get_full_var('xk', split_x_and_aux=self.is_explicit)

        self.Fi = []
        self.mui = []
        for i in range(nstages):
            self.Fi.append(self.model.get_full_var('F'+str(i), split_x_and_aux=self.is_explicit))
            self.mui.append(self.model.get_full_var('mu'+str(i), split_x_and_aux=self.is_explicit))

        xhat, xhat_subs = self.model.get_full_test_vars(split_x_and_aux=self.is_explicit)
        xtrial, xtrial_subs = self.model.get_full_trial_vars(split_x_and_aux=self.is_explicit)

#FOR IRK, we need to build the FULL space of s copies as a giant mixed system
#then F[i][0][0], and xhat, etc. is actually a SPLIT of the full function (or really a sum of the appropriate splits...)
#NOT SURE THIS WORKS, BUT SOMETHING LIKE THIS!

#what get_full should return is xfull,auxfull indexed by [i=num copies], etc.
#then we can write general code below!

#but otherwise the code below is fine
#WE SHOULD MODIFY XHAT AND XTRIAL TO HAVE AN I INDEX
#easiest solution- modify get full test/trial vars and get full var to take an index for how many copies,
#and also an arugment about whether to split the copies as well
#then the code below (with xhat_subs[i] replacement, etc.) should seamlessly generalize!


        #construct xi variables
        xi_splits = []
        for i in range(nstages):
            xi_split = {}
            for var in self.model.get_x_var_list():
                xi_split[self.xk_split[var]] = self.xk_split[var]
                for j in range(nstages):
                    if not (self.A[i,j] == 0): xi_split[self.xk_split[var]] = xi_split[self.xk_split[var]] + self.dt*float(self.A[i,j]) * self.Fi[j][2][var]
            for var in self.model.get_aux_var_list():
                xi_split[self.xk_split[var]] = self.Fi[i][2][var]
            xi_split[self.t] = self.t + float(self.c[i])*self.dt
            xi_splits.append(xi_split)

        #construct residuals for F and aux
        residuals_F = []
        residuals_aux = []
        for i in range(nstages):
        #this sign is due to writing things as dxdt + F(x) = 0
            rhs_Fi = -model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
            rhs_Fi = replace(rhs_Fi, xi_splits[i])
            residual_F = inner(xhat[0], self.Fi[i][0][0])*self.dx - rhs_Fi
#MIGHT HAVE TO MODIFY THIS INNER PRODUCT A LITTLE FOR IRK GENERALITY

            residual_aux = 0
            if self.model.has_aux():
                aux_expressions = self.model.compute_aux_expressions(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
                rhs_W = 0
                lhs_W = 0
                for var in self.model.get_aux_var_list(terms=terms):
                    rhs_W = rhs_W + aux_expressions[var][1]
                    lhs_W = lhs_W + aux_expressions[var][0]
                if not (lhs_W == 0):
                    residual_aux = replace(lhs_W - rhs_W, xi_splits[i])


            residuals_aux.append(residual_aux)
            residuals_F.append(residual_F)

        #construct residuals for mu and muaux
        residuals_mu = []
        residuals_muaux = []

        residual_xk = inner(self.delta_lambda_var, xhat[0])*self.dx
#THIS MIGHT NEED TO MODIFIED FOR DIRK/IRK?
        if model.has_coeff():
            residual_grad = inner(self.delta_grad, self.grad_test)*self.dx
        for i in range(nstages):
            residual_mu = inner(self.mui[i][0][0], xhat[0])*self.dx
            residual_mu = residual_mu - self.dt * float(self.b[i])*inner(self.lambda_var, xhat[0])*self.dx
            residual_muaux = 0

            for j in range(nstages):
                rhs_Fj = -model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
                rhs_Fj = replace(rhs_Fj, xi_splits[j])
                derivT_Fj_Fi = adjoint(derivative(rhs_Fj, self.Fi[i][0][0], xtrial[0]))
                if not derivT_Fj_Fi.empty():
                    residual_mu = residual_mu - action(derivT_Fj_Fi, self.mui[j][0][0])
                if model.has_aux():
                    aux_expressions = self.model.compute_aux_expressions(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
                    rhs_Wj = 0
                    for var in self.model.get_aux_var_list(terms=terms):
                        rhs_Wj = rhs_Wj + aux_expressions[var][1]
                    if not (rhs_Wj == 0):
                        rhs_Wj = replace(rhs_Wj, xi_splits[j])
                        derivT_Wj_Fi = adjoint(derivative(rhs_Wj, self.Fi[i][0][0], xtrial[0]))
                        if not derivT_Wj_Fi.empty():
                            residual_mu = residual_mu - action(derivT_Wj_Fi, self.mui[j][0][1])

            rhs_Fi = -model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
            rhs_Fi = replace(rhs_Fi, xi_splits[i])
            derivT_Fi_xk = adjoint(derivative(rhs_Fi, self.xk[0], xtrial[0]))
            if not derivT_Fi_xk.empty():
                residual_xk = residual_xk -  action(derivT_Fi_xk, self.mui[i][0][0])
            if model.has_aux():
                aux_expressions = self.model.compute_aux_expressions(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
                rhs_Wi = 0
                for var in self.model.get_aux_var_list(terms=terms):
                    rhs_Wi = rhs_Wi + aux_expressions[var][1]
                if not (rhs_Wi == 0):
                    rhs_Wi = replace(rhs_Wi, xi_splits[i])
                    derivT_Wi_xk = adjoint(derivative(rhs_Wi, self.xk[0], xtrial[0]))
                    if not derivT_Wi_xk.empty():
                        residual_xk = residual_xk -  action(derivT_Wi_xk, self.mui[i][0][1])
    #THIS IS ASSUMING THAT LHS W IS LINEAR IN W
#IF THERE IS STATE DEPENDENCE THEN IT MUST BE CHANGED!!!
                    residual_muaux = inner(self.mui[i][0][1], xhat[1])*self.dx
                    derivT_Fi_wi = adjoint(derivative(rhs_Fi, self.Fi[i][0][1], xtrial[1]))
                    if not derivT_Fi_wi.empty():
                        residual_muaux = residual_muaux - action(derivT_Fi_wi, self.mui[i][0][0])
                    if model.has_coeff():
                        derivT_Wi_theta = adjoint(derivative(rhs_Wi, self.coeff, self.grad_trial))
                        if not derivT_Wi_theta.empty():
                            residual_grad = residual_grad - action(derivT_Wi_theta, self.mui[i][0][1])
            if model.has_coeff():
                derivT_Fi_theta = adjoint(derivative(rhs_Fi, self.coeff, self.grad_trial))
                if not derivT_Fi_theta.empty():
                    residual_grad = residual_grad - action(derivT_Fi_theta, self.mui[i][0][0])

            residuals_mu.append(residual_mu)
            residuals_muaux.append(residual_muaux)


#EVENTUALLY MAKE THESE SPECIFIC TO A GIVEN TYPE OF RK IE SUBCLASS STUFF
        #construct solvers
        if self.is_explicit:
            self.Fsolvers = []
            self.auxsolvers = []
            self.musolvers = []
            self.muauxsolvers = []
            for i in range(self.nstages):
                #print('creating solvers', i)
                if (not self.model.has_aux()) or residuals_aux[i] == 0:
                    self.auxsolvers.append(DummySolver())
                    self.muauxsolvers.append(DummySolver())
                else:
#THIS IS GOING TO FAIL WHEN AUXILIARY VARIABLES HAVE NON-CONSTANT JACOBIANS
#CAN/SHOULD MAYBE SPLIT THESE INTO SEPARATE SOLVERS FOR EACH AUX VAR?
#THIS WILL BE A MESS FOR COMPUTING ADJOINT STUFF, BUT MANAGEABLE!
                    self.auxsolvers.append(create_linear_solver_from_residual(residuals_aux[i], self.Fi[i][0][1], xtrial[1], constant_jacobian=True, solver_parameters=self.solver_parameters['erkstage-aux'], options_prefix = 'erk-aux')[0])
                    self.muauxsolvers.append(create_linear_solver_from_residual(residuals_muaux[i], self.mui[i][0][1], xtrial[1], constant_jacobian=True, solver_parameters=self.solver_parameters['erkstage-muaux'], options_prefix = 'erk-muaux')[0])

                self.Fsolvers.append(create_linear_solver_from_residual(residuals_F[i], self.Fi[i][0][0], xtrial[0], constant_jacobian=True, solver_parameters=self.solver_parameters['erkstage-f'], options_prefix = 'erk-f')[0])
                self.musolvers.append(create_linear_solver_from_residual(residuals_mu[i], self.mui[i][0][0], xtrial[0], constant_jacobian=True, solver_parameters=self.solver_parameters['erkstage-mu'], options_prefix = 'erk-mu')[0])
            self.deltalambdasolver, _, self.L_delta_lambda = create_linear_solver_from_residual(residual_xk, self.delta_lambda_var, xtrial[0], constant_jacobian=True, solver_parameters=self.solver_parameters['erk-dlambda'], options_prefix = 'erk-lambda')
            if model.has_coeff():
                _, self.L_delta_grad = extract_a_L_from_residual(residual_grad, self.delta_grad, self.grad_trial)
                #self.deltagradsolver, _, self.L_delta_grad = create_linear_solver_from_residual(residual_grad, self.delta_grad, self.grad_trial, constant_jacobian=True, solver_parameters=self.solver_parameters['erk-grad'], options_prefix = 'erk-grad')
        elif self.is_dirk:
            raise NotImplementedError('dirk not done yet')
            #self.Fauxsolvers = []
            #self.mufullsolvers = []
            #for i in range(self.nstages):
            #    full_residual = self.residual_F[i] + self.residual_aux[i]
            #    fullproblem = NonlinearVariationalProblem(full_residual, self.Fi[i][0][0])
            #    self.Fauxsolvers.append(NonlinearVariationalSolver(fullproblem, solver_parameters=overall_solver_parameters['dirkstage'], options_prefix = 'dirk'))
            #    full_mu_residual = self.residual_mu[i] + self.residual_muaux[i]
            #    self.mufullsolvers.append(create_linear_solver_from_residual(full_mu_residual, self.Fi[i][0][0], xtrial[0], solver_parameters=overall_solver_parameters['dirkstage-mu'], options_prefix = 'dirk-mu'))

        else:
            raise NotImplementedError('implicit RK not done yet')
            #total_residual = 0
            #total_mu_residual = 0
            #for i in range(self.nstages):
            #    total_residual = total_residual + self.residual_F[i] + self.residual_aux[i]
            #    total_mu_residual = total_mu_residual + self.residual_mu[i] + self.residual_muaux[i]
            #totalproblem = NonlinearVariationalProblem(total_residual, self.Fi[0]) #SOMETHIGN LIKE THIS
            #NEED TO RETURN THIS FULL VARIABLE SOMEHOW! This is pretty straightforward I think, just another thing that get_full_var (and get_full_test_vars, etc.) does!
            #self.Fauxsolver = NonlinearVariationalSolver(totalproblem, solver_parameters=overall_solver_parameters['irk'], options_prefix = 'irk')
            #self.mufullsolver = create_linear_solver_from_residual(total_mu_residual, self.mui[0], xtrial[0], solver_parameters=overall_solver_parameters['irk-mu'], options_prefix = 'irk-mu')


    def get_rhs_expr(self):

        xhat, xhat_subs = self.model.get_full_test_vars(split_x_and_aux=self.is_explicit)

        xi_splits = []
        for i in range(self.nstages):
            xi_split = {}
            for var in self.model.get_x_var_list():
                xi_split[self.xk_split[var]] = self.xk_split[var]
                for j in range(self.nstages):
                    if not (self.A[i,j] == 0): xi_split[self.xk_split[var]] = xi_split[self.xk_split[var]] + self.dt*float(self.A[i,j]) * self.Fi[j][2][var]
            for var in self.model.get_aux_var_list():
                xi_split[self.xk_split[var]] = self.Fi[i][2][var]
            xi_split[self.t] = self.t + float(self.c[i])*self.dt
            xi_splits.append(xi_split)

        rhs_F = -self.model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs)
        rhs_W = 0
        if self.model.has_aux():
            aux_expressions = self.model.compute_aux_expressions(self.xk_split, self.t, self.coeff_split, xhat_subs)
            for var in self.model.get_aux_var_list():
                rhs_W = rhs_W + aux_expressions[var][1]

        rhs_Fis = []
        rhs_Ws = []
        for i in range(self.nstages):
            rhs_Fi = -self.model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs)
            rhs_Fi = replace(rhs_Fi, xi_splits[i])
            rhs_Fis.append(rhs_Fi)
            rhs_Wi = 0
            if self.model.has_aux():
                aux_expressions = self.model.compute_aux_expressions(self.xk_split, self.t, self.coeff_split, xhat_subs)
                for var in self.model.get_aux_var_list():
                    rhs_Wi = rhs_Wi + aux_expressions[var][1]
                if not (rhs_Wi == 0):
                    rhs_Wi = replace(rhs_Wi, xi_splits[i])
                rhs_Ws.append(rhs_Wi)

        return rhs_F, rhs_W, rhs_Fis, rhs_Ws

    def split_x_and_aux(self):
        return self.is_explicit

    def reset_internal_vars(self):
        self.xk[0].assign(0)
        self.delta_lambda_var.assign(0)
        self.lambda_var.assign(0)
        if self.model.has_coeff():
            self.delta_grad.assign(0)
        if self.model.has_aux():
            self.xk[1].assign(0)
        for i in range(self.nstages):
            self.Fi[i][0][0].assign(0)
            self.mui[i][0][0].assign(0)
            if self.model.has_aux():
                self.mui[i][0][1].assign(0)
                self.Fi[i][0][1].assign(0)

#EVENTUALLY MAKE THESE SPECIFIC TO A GIVEN TYPE OF RK IE SUBCLASS STUFF
    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):
        self.t.assign(tn)
        self.dt.assign(dt)
        self.xk[0].assign(xn[0])
#NEED SOME WAY OF DOING THIS FOR FULL SPACE, SINCE AUX INITIAL GUESS SHOULD GO IN FI...
        #if len(self.xk) > 1:
        #    self.Fi[0][0][1].assign(xn[1])

        if self.is_explicit:
            for i in range(self.nstages):
                self.auxsolvers[i].solve()
                self.Fsolvers[i].solve()
        elif self.is_dirk:
            for i in range(self.nstages):
                self.Fauxsolvers[i].solve()
        else:
            self.Fauxsolver.solve()
        #print('dt', self.dt)
        #print('xn', self.model.norm(xn[0]))

#THIS BREAKS A LITTLE FOR FULL SPACE VERSION
#IE WE SHOULD JUST BE ASSIGNING THE X VARIABLES HERE
#AND THEN DOING SOMETHING FOR THE AUX VARS
        xnp1[0].assign(xn[0] + self.dt * sum(float(self.b[i]) * self.Fi[i][0][0] for i in range(self.nstages)))
        #print('xnp1', self.model.norm(xnp1[0]))
        #if len(xnp1) > 1:
#IDEALLY THIS IS AUX VARS EVALUATED AT XNP1- provides a good initial guess at least?
        #    xnp1[1].assign(self.Fi[-1][0][1])

    def take_adjoint_step(self, delta_grad, delta_lambda, lambda_np1, xn, tnp1, dt):

        self.dt.assign(dt)
        self.t.assign(tnp1 - dt)
        self.lambda_var.assign(lambda_np1)

        #take a forward step from xn to populate Fi
        self.xk[0].assign(xn[0])
        if self.is_explicit:
            for i in range(self.nstages):
                self.auxsolvers[i].solve()
                self.Fsolvers[i].solve()
        elif self.is_dirk:
            for i in range(self.nstages):
                self.Fauxsolvers[i].solve()
        else:
            self.Fauxsolver.solve()
        #print("lambda var norm", norm(self.lambda_var))

#EVENTUALLY MAKE THESE SPECIFIC TO A GIVEN TYPE OF RK IE SUBCLASS STUFF?
        #compute mu's
        if self.is_explicit:
            for i in range(self.nstages-1,-1,-1):
                self.musolvers[i].solve()
                self.muauxsolvers[i].solve()
        elif self.is_dirk:
            for i in range(self.nstages-1,-1,-1):
                self.mufullsolvers.solve()
        else:
            self.mufullsolver.solve()

        #compute grad
        if self.model.has_coeff():
            #self.deltagradsolver.solve()
            #delta_grad.assign(self.delta_grad)
#THIS IS VERY SLOW, LIKELY DUE TO SET UP COSTS
#CAN WE MAKE IT FASTER SOMEHOW?\
            delta_grad_rhs = assemble(self.L_delta_grad)

        #compute lambda_n
        self.deltalambdasolver.solve()
        delta_lambda.assign(self.delta_lambda_var)
#THIS IS VERY SLOW, LIKELY DUE TO SET UP COSTS
#CAN WE MAKE IT FASTER SOMEHOW?
        delta_lambda_rhs = assemble(self.L_delta_lambda)
        if self.model.has_coeff():
            return delta_lambda_rhs, delta_grad_rhs
        else:
            return delta_lambda_rhs, None


class Euler(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = np.array([[0.0,],])
        b = np.array([1.0,])
        c = np.array([0.0,])
        GeneralRK.__init__(self, model, logger, A, b, c, 1, solver_parameters, terms=terms)

class RK4(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        b = np.array([1./6., 1./3., 1./3., 1./6.])
        c = np.array([0.0, 0.5, 0.5, 1.0])
        GeneralRK.__init__(self, model, logger, A, b, c, 4, solver_parameters, terms=terms)

class SSPRK3(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        b = np.array([1./6., 1./6., 2./3.])
        c = np.array([0.0, 1.0, 0.5])
        GeneralRK.__init__(self, model, logger, A, b, c, 3, solver_parameters, terms=terms)


class SSPRK43(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0/6.0, 1.0/6.0, 1.0/6.0, 0.0]])
        b = np.array([1./6., 1./6., 1./6., 3./6.])
        c = np.array([0.0, 0.5, 1.0, 0.5])
        GeneralRK.__init__(self, model, logger, A, b, c, 4, solver_parameters, terms=terms)


#ADD THESE!!!
class KGRK2(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        GeneralRK.__init__(self, model, logger, A, b, c, nstages, solver_parameters, terms=terms)


class KGRK3(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        GeneralRK.__init__(self, model, logger, A, b, c, nstages, solver_parameters, terms=terms)

class SOMEDIRK(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        GeneralRK.__init__(self, model, logger, A, b, c, nstages, solver_parameters, terms=terms)


class SOMEIMPLICITRK(GeneralRK):
    def __init__(self, model, logger, solver_parameters, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        GeneralRK.__init__(self, model, logger,  A, b, c, nstages, solver_parameters, terms=terms)




class LieSplittingIntegrator():
    def __init__(self, model, logger, timestepper_list, termlist, subcycle_list, solver_parameters):

        self.subcycle_list = subcycle_list
        self.timestepper_list = timestepper_list
        self.termlist = termlist
        self.solver_parameters = solver_parameters

        self.time_integrators = []
        for i,time_integrator_name in enumerate(timestepper_list):
            time_integrator = get_time_integrator(time_integrator_name, None, solver_parameters)
            self.time_integrators.append(time_integrator(model, logger, solver_parameters, terms=termlist[i]))

        self.lambda_k, self.lambda_sub, self.lambda_split = model.get_x_var('lambda_k')
        self.delta_lambda, _, _ = model.get_x_var('delta_lambda')

#HOW DO WE HANDLE SPLITTING X AND AUX IN THE GENERAL RK CASE?
        self.xks = []
        self.xks.append(model.get_full_var('x0', split_x_and_aux=True))
        for i in range(len(self.time_integrators)):
            for k in range(self.subcycle_list[i]):
                self.xks.append(model.get_full_var('x'+str(i)+'_'+str(k), split_x_and_aux=True))

    def reset_internal_vars(self):
        self.lambda_k.assign(0)
        self.delta_lambda.assign(0)
        for xk in self.xks:
            self.xk[0].assign(0)
        for time_integrator in self.time_integrator:
            time_integrator.reset_internal_vars()

#DOES THIS NEED TO RESET INTERNAL VARS TO ENSURE REPEATABILITY?
    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):
        self.xks[0][0][0].assign(xn[0])
        l = 1
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
                time_integrator.take_forward_step(self.xks[l][0], self.xks[l][1], self.xks[l-1][0], tn + k * sub_dt, sub_dt)
                l = l +1
        xnp1[0].assign(self.xks[-1][0][0])

#RESET INTERNAL VARS
#needs to reset self vars and also timestepper vars!


#HOW DO WE HANDLE DELTA GRAD AND DELTA LAMBDA MORE GENERALLY?
#THEY ARENT NEEDED FOR SCIPY VARIANT
#UNCLEAR IF THEY WILL BE NEEDED FOR ROL VARIANT?
#PROBABLY COMMENT OUT THEIR MACHINERY FOR NOW...
    def take_adjoint_step(delta_grad, delta_lambda, lambda_np1, xn, tnp1, dt):
        self.lambda_k.assign(lambda_np1)
        delta_lambda_rhs = 0
        delta_grad_rhs = 0

        self.xks[0][0][0].assign(xn[0])
        l = 1
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
                time_integrator.take_forward_step(self.xks[l][0], self.xks[l][1], self.xks[l-1][0], tn + k * sub_dt, sub_dt)
                l = l +1

        l = 0
        lt = len(self.xks)-1
        for i,time_integrator in enumerate(reversed(self.time_integrators)):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
#DO WE NEED TO RESET INTERNAL VARS HERE?
#ALSO UNCLEAR ABOUT DELTA_LAMBDA VS _DELTA_LAMBDA_RHS, ETC.
                _delta_lambda_rhs, _delta_grad_rhs = time_integrator.take_adjoint_step(self.delta_lambda, self.lambda_k, self.xks[lt-l-1][0], tnp1 - k * sub_dt, sub_dt)
                self.lambda_k.assign(self.lambda_k + self.delta_lambda)
                #self.grad.assign(self.grad + delta_grad)
                delta_grad_rhs = delta_grad_rhs + _delta_grad_rhs
                delta_lambda_rhs = delta_lambda_rhs + _delta_lambda_rhs
                l = l + 1
        return delta_lambda_rhs, delta_grad_rhs

    def set_coeff(self, coeff_val):
        for time_integrator in self.time_integrators:
            time_integrator.set_coeff(coeff_val)


    def set_numpy_coeff(self, coeff_val_arr):
        for time_integrator in self.time_integrators:
            time_integrator.set_numpy_coeff(coeff_val_arr)








#
# class FixedPointSolver():
#     def __init__(self, fexpr, xnp1, varspace, pre_function_callback, post_function_callback, dx):
#         self.fexpr = fexpr
#         self.xnp1 = xnp1 #this is xk
#         #self.xkp1 = xnp1.copy(deepcopy=True)
#         self.xkp1 = Function(varspace)
#         self.pre_function_callback = pre_function_callback
#         self.post_function_callback = post_function_callback
#         self.dx = dx
#
# #MOVE TO PARAMETERS AT SOME POINT
#         self.eps = 1e-12
#         self.max_iters = 50
#
#         xtest = TestFunction(varspace)
#         a = derivative(inner(xtest,self.xkp1)*self.dx, self.xkp1)
#         linearproblem = LinearVariationalProblem(a, fexpr, self.xkp1)
#         self.linearsolver = LinearVariationalSolver(linearproblem, solver_parameters=overall_solver_parameters['fixedpoint'], options_prefix = 'fixedpoint')
#
#     def solve(self):
#
#         niters = 0
#         rel_tol = 100.0
#
# #EVENTUALLY ADD ANDERSON ACCELERATION?
#         while (rel_tol > self.eps and niters<self.max_iters):
#             with self.xnp1.dat.vec_ro as vardat:
#                 self.pre_function_callback(vardat)
#
#             self.linearsolver.solve()
#             rel_tol = norm(self.xkp1 - self.xnp1) / max(norm(self.xkp1), 1.0)
#
# #SUPER UNCLEAR IF A LIMITER HERE WILL ACTUALLY WORK?
#             with self.xkp1.dat.vec_wo as xkp1:
#                     self.post_function_callback(xkp1)
#
#             with self.xkp1.dat.vec_ro as xkp1:
#                 with self.xnp1.dat.vec_wo as xnp1:
#                     xkp1.copy(xnp1)
#             niters = niters + 1
# #probably store these somewhere eventually
# #ALSO DO THIS FOR NEWTON SOLVER AS WELL!!!
#         #print(rel_tol, niters)
#
#
#
#
#
# class AVF2_Integrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#         self.dx = model.spaces.dx
#
#
#         self.q_aux_vars = self.model.get_q_aux_vars(terms=terms)
#         self.dfdx_aux_vars = self.model.get_dfdx_aux_vars(terms=terms)
#
#         if xn is None:
#             self.xn = self.model.get_x_var('xn')
#         else:
#             self.xn = xn
#         self.xk = self.model.get_x_var('xk')
#         self.xnp1 = self.model.get_x_var('xnp1')
#
#         self.points, self.weights = np.polynomial.legendre.leggauss(parameters['timestepping']['num_avf_quad'])
#         self.npoints = parameters['timestepping']['num_avf_quad']
#         #renormalize to [0,1] interval
#         self.points = (self.points + 1.) / 2.0
#         self.weights = self.weights / 2.
#
# #CAN TRIM OR REMOVE A LOT OF THIS?
#         self.xn_sub = {}
#         self.xk_sub = {}
#         self.xnp1_sub = {}
#         self.xnp1_split = {}
#         xnp1_split_temp = split(self.xnp1)
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.xn_sub[var] = self.xn.sub(i)
#             self.xnp1_sub[var] = self.xnp1.sub(i)
#             self.xnp1_split[var]  = xnp1_split_temp[i]
#             self.xk_sub[var] = self.xk.sub(i)
#
#
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.q_aux_vars[var] = 0.5 * self.xnp1_split[var] + 0.5 * self.xn.sub(i)
#             #REPLACE WITH self.xn_sub here!
#
#         q_expressions = self.model.compute_q_expressions(self.q_aux_vars, terms=terms)
#         self.q_aux_solvers = []
#         for var in self.model.get_q_aux_var_list(terms=terms):
#             a, L = q_expressions[var]
#             qproblem = LinearVariationalProblem(a, L, self.q_aux_vars[var])
#             qsolver = LinearVariationalSolver(qproblem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.q_aux_solvers.append(qsolver)
#
#
#         self.dfdx_aux_solvers = []
#         xq = {}
#         dfdx_expressions = {}
#         for var in self.model.get_dfdx_aux_var_list(terms=terms):
#             dfdx_expressions[var] = [0,0]
#         for i in range(self.npoints):
#             point, weight = self.points[i], self.weights[i]
#             for j,var in enumerate(self.model.variableset.varlist):
#                 xq[var] = (1. - float(point)) * self.xn.sub(j) + float(point) * self.xk.sub(j)
#             xq_dfdx_expressions = self.model.compute_dfdx_expressions(xq, terms=terms)
#             for var in self.model.get_dfdx_aux_var_list(terms=terms):
#                 a, L = make_a_L(*xq_dfdx_expressions[var], self.dx)
#                 dfdx_expressions[var][0] = dfdx_expressions[var][0] + float(weight) * a
#                 dfdx_expressions[var][1] = dfdx_expressions[var][1] + float(weight) * L
#         for var in self.model.get_dfdx_aux_var_list(terms=terms):
#             a, L = dfdx_expressions[var]
#             dfdx_problem = LinearVariationalProblem(a, L, self.dfdx_aux_vars[var])
#             dfdx_solver = LinearVariationalSolver(dfdx_problem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.dfdx_aux_solvers.append(dfdx_solver)
#
#
#         xhat = self.model.variableset.get_test_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         rhs_expr = model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#
#         self.dt = Constant(1.)
#         self.t = Constant(1.)
#         self.tn = 0.
#
# #CANNOT REALLY DO FIXED POINT USING PETSC, OR AT LEAST I CAN'T FIGURE IT OUT
# #probably possible with some careful combination of NL solver options
#         if parameters['timestepping']['avf_solver'] == 'fixedpoint':
#             nl_expr = inner(xhat, self.xn)*self.dx - self.dt*rhs_expr
#             self.rhs_nl_solver = FixedPointSolver(nl_expr, self.xnp1, self.model.variableset.mixedspace,
#                 lambda state: self.pre_callback(state), lambda state: self.post_callback(state), self.dx)
# #SUPER UNCLEAR IF A LIMITER HERE FOR POST FUNCTION CALLBACK WILL ACTUALLY WORK?
#
#         elif parameters['timestepping']['avf_solver'] == 'qn':
#             nl_expr = inner(xhat, self.xnp1 - self.xn)*self.dx + self.dt*rhs_expr
#             rhs_linear_expr = model.linear_rhs(self.initcond.const_state, self.q_aux_vars, xhat_subs, terms=terms)
#             linear_expr = inner(xhat, self.xnp1 - self.xn)*self.dx + self.dt*rhs_linear_expr
#             J_expr = derivative(linear_expr, self.xnp1)
#             rhs_nl_problem = NonlinearVariationalProblem(nl_expr, self.xnp1, J=J_expr)
#             self.rhs_nl_solver = NonlinearVariationalSolver(rhs_nl_problem, solver_parameters=overall_solver_parameters['qn'],
#                 options_prefix = 'avf2qn', pre_function_callback=lambda state: self.pre_callback(state), post_function_callback=lambda state, jacobian: self.post_callback(state, jacobian))
#
#
#
#
#     def pre_callback(self, state):
#         with self.xk.dat.vec_wo as xk:
#             state.copy(xk)
#         for solver in self.q_aux_solvers:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers:
#             solver.solve()
#
#     def initialize(self, init_xn=True):
#         if init_xn:
#             self.xn.zero()
#             varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#             self.model.initialize(varexpr, self.xn)
#
#         #self.xnp1.zero()
#         #self.xk.zero()
#         self.xnp1.assign(self.xn)
# #        self.xn.dat.copy(self.xk.dat) #DONT THINK WE NEED THIS COPY
#         #self.xn.dat.copy(self.xnp1.dat)
#
# #THIS IS ACTUALLY BROKEN- UNCLEAR EXACTLY HOW TO MODIFY CURRENT STATE AFTER A STEP IS COMPUTED?
#     def post_callback(self, state, jacobian):
#         pass
#         #with self.xk.dat.vec_wo as xk:
#         #    state.copy(xk)
#         #self.model.post_step(self.xk_sub)
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
# #HOW EXACTLY SHOULD self.t be handled here?
# #IT needs to be fed into rhs, J, etc. at the appropriate points...
# #especially BCs might be time dependent...
#         self.rhs_nl_solver.solve()
#         self.xn.assign(self.xnp1)
#         self.tn = self.tn + dt
#
#
#
# #FIX THIS EVENTUALLY
# #Here we need to somehow specify who is staggered, and also split various dHdx calcs, etc.
# class TimeStaggered_Integrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#         self.dx = model.spaces.dx
#
# #FIX THIS
#     def initialize(self, init_xn=True):
#         if init_xn:
#             self.xn.zero()
#             varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#             self.model.initialize(varexpr, self.xn)
#
#     def pre_stepA_solvers(self):
#         for solver in self.q_aux_solvers_A:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers_A:
#             solver.solve()
#
#     def pre_stepB_solvers(self):
#         for solver in self.q_aux_solvers_B:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers_B:
#             solver.solve()
#
# #FIX THIS
#     def take_step(self, dt):
#         self.dt.assign(dt)
# #DO WE NEED XK REALLY HERE? Probably not, can just update xn...
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn)
#
# #SPLITTING IS TRICKY- PROBABLY WANT TWO SEPARATE XN VARIABLES? Kind of unclear
# #or do manual updates, etc.?
#         self.pre_stepA_solvers()
#         self.stepA_solver.solve()
#         self.xn.assign(self.xn + self.dt * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn + dt/2.0)
#         self.pre_stepB_solvers()
#         self.stepB_solver.solve()
#         self.xn.assign(self.xn + self.dt * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
#
# class RK_Integrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger, xn=None, use_xn_as_xk=False, terms='all'):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#         self.terms = terms
#
#         self.dx = model.spaces.dx
#
#         self.q_aux_vars = self.model.get_q_aux_vars(terms=terms)
#         self.dfdx_aux_vars = self.model.get_dfdx_aux_vars(terms=terms)
#
#         if xn is None:
#             self.xn = self.model.get_x_var('xn')
#         else:
#             self.xn = xn
#
#         if not use_xn_as_xk:
#             self.xk = self.model.get_x_var('xk')
#         else:
#             self.xk = self.xn
#
#         self.xn_sub = {}
#         self.xk_sub = {}
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.xn_sub[var] = self.xn.sub(i)
#             self.xk_sub[var] = self.xk.sub(i)
#             self.q_aux_vars[var] = self.xk.sub(i)
#
#
#         q_expressions = self.model.compute_q_expressions(self.q_aux_vars, terms=terms)
#         self.q_aux_solvers = []
#         for var in self.model.get_q_aux_var_list(terms=terms):
#             a, L = q_expressions[var]
#             qproblem = LinearVariationalProblem(a, L, self.q_aux_vars[var])
#             qsolver = LinearVariationalSolver(qproblem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.q_aux_solvers.append(qsolver)
#
#         self.dfdx_aux_solvers = []
#         dfdx_expressions = self.model.compute_dfdx_expressions(self.q_aux_vars, terms=terms)
#         for var in self.model.get_dfdx_aux_var_list(terms=terms):
#             a, L = make_a_L(*dfdx_expressions[var], self.dx)
#             dfdx_problem = LinearVariationalProblem(a, L, self.dfdx_aux_vars[var])
#             dfdx_solver = LinearVariationalSolver(dfdx_problem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.dfdx_aux_solvers.append(dfdx_solver)
#
#         self.dt = Constant(1.)
#         self.t = Constant(1.)
#         self.tn = 0.
#
#     def pre_step_solvers(self):
#         for solver in self.q_aux_solvers:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers:
#             solver.solve()
#
#     def initialize(self, init_xn=True):
#         if init_xn:
#             self.xn.zero()
#             varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#             self.model.initialize(varexpr, self.xn)
#
# class RK4_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F1 = self.model.get_x_var('F1')
#         self.F2 = self.model.get_x_var('F2')
#         self.F3 = self.model.get_x_var('F3')
#         self.F4 = self.model.get_x_var('F4')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         F1problem = LinearVariationalProblem(A, rhsproblem, self.F1)
#         self.F1solver = LinearVariationalSolver(F1problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f1')
#
#         F2problem = LinearVariationalProblem(A, rhsproblem, self.F2)
#         self.F2solver = LinearVariationalSolver(F2problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f2')
#
#         F3problem = LinearVariationalProblem(A, rhsproblem, self.F3)
#         self.F3solver = LinearVariationalSolver(F3problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f3')
#
#         F4problem = LinearVariationalProblem(A, rhsproblem, self.F4)
#         self.F4solver = LinearVariationalSolver(F4problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f4')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.xk.assign(self.xn)
#         self.pre_step_solvers()
#         self.F1solver.solve()
#
#         self.xk.assign(self.xn + self.dt/2.0*self.F1)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#         self.t.assign(self.tn + self.dt/2.)
#         self.pre_step_solvers()
#         self.F2solver.solve()
#
#         self.xk.assign(self.xn + self.dt/2.0*self.F2)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#         self.t.assign(self.tn + self.dt/2.)
#         self.pre_step_solvers()
#         self.F3solver.solve()
#
#         self.xk.assign(self.xn + self.dt*self.F3)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#         self.t.assign(self.tn + self.dt)
#         self.pre_step_solvers()
#         self.F4solver.solve()
#
#         self.xn.assign(self.xn + self.dt/6. * (self.F1 + 2.*self.F2 + 2.*self.F3 + self.F4))
#         self.model.post_step(self.xn_sub, terms=self.terms)
#         self.tn = self.tn + dt
#
# #three register kinnmark + grey time integrators
# #DO THESE OFFER ANYTHING OVER THE 2 STAGE INTEGRATORS?
# #TALK TO MARK AND ANDREW STEYER
# #ie either better accuracy or more efficiency? My guess is 53 is hard to beat..
# class KGRK3_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F1 = self.model.get_x_var('F1')
#         self.F2 = self.model.get_x_var('F2')
#
# #FIX THIS- THERE MIGHT BE TYPOS IN THE IMEX KG PAPER!
#         #set values for alpha, beta, etc.!
#         if parameters['timestepping']['nkg_stages'] == 3:
#             self.alpha = np.array([1./3., 1./3., 3./4.])
#             self.beta = np.array([0., ])
#             self.c = np.array([1./2., 1./2., 1.])
#         elif parameters['timestepping']['nkg_stages'] == 3:
#             self.alpha = np.array([1./4., 1./3., 1./2., 1.])
#             self.beta = np.array([0., ])
#             self.c = np.array([1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['nkg_stages'] == 3:
#             self.alpha = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#             self.beta = np.array([0., ])
#             self.c = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#
#         self.num_stages = self.alpha.shape[0]
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         F1problem = LinearVariationalProblem(A, rhsproblem, self.F1)
#         self.F1solver = LinearVariationalSolver(F1problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'kgrk3-f')
#
#         F2problem = LinearVariationalProblem(A, rhsproblem, self.F2)
#         self.F1solver = LinearVariationalSolver(F2problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'kgrk3-f')
#
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn)
#         self.pre_step_solvers()
#         self.F1solver.solve()
#
#         for i in range(self.num_stages-1):
#             self.xk.assign(self.xn + self.dt * (self.beta[i] * self.F1 + self.alpha[i]* self.F2))
#             self.model.post_step(self.xk_sub, terms=self.terms)
#             self.t.assign(self.tn + self.c[i] * self.dt)
#             self.pre_step_solvers()
#             self.F2solver.solve()
#
#         self.xn.assign(self.xn + self.dt * (self.beta[-1] * self.F1 + self.alpha[-1] * self.F2))
#         self.model.post_step(self.xn_sub, terms=self.terms)
#         self.tn = self.tn + dt
#
# #two register kinnemark + grey time integrators
# class KGRK2_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F = self.model.get_x_var('F')
#
#         #set values for alpha, etc.!
#         if parameters['timestepping']['kgrk2_name'] == '32': #NOT SURE THIS IS 2ND OR 1ST ORDER?
#             self.alpha = np.array([1./2., 1./2., 1.])
#             self.c = np.array([1./2., 1./2., 1.])
#         if parameters['timestepping']['kgrk2_name'] == '32a': #NOT SURE THIS IS 2ND OR 1ST ORDER?
#             self.alpha = np.array([1./3., 1./2., 1.])
#             self.c = np.array([1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '42': #NOT SURE THIS IS 2ND OR 1ST ORDER?
#             self.alpha = np.array([1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '52':
#             self.alpha = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#             self.c = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '53':
#             self.alpha = np.array([1./5., 1./5., 1./3., 1./2., 1.])
#             self.c = np.array([1./5., 1./5, 1./3., 1./2., 1.])
#         #SUPER UNCLEAR WHAT THE ORDER OF THESE METHODS ARE?
#         elif parameters['timestepping']['kgrk2_name'] == '62':
#             self.alpha = np.array([1./6., 2./15., 1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./6., 2./15., 1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '72':
#             self.alpha = np.array([1./7., 2./21., 1./5., 8./35., 1./3., 1./2., 1.])
#             self.c = np.array([1./7., 2./21., 1./5., 8./35., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '82':
#             self.alpha = np.array([1./8., 1./14., 1./6., 1./6., 1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./8., 1./14., 1./6., 1./6., 1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '92':
#             self.alpha = np.array([1./9., 1./18., 1./7., 8./63., 1./5., 5./21., 1./3., 1./2., 1.])
#             self.c = np.array([1./9., 1./18., 1./7., 8./63., 1./5., 5./21., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '102':
#             self.alpha = np.array([1./10., 2./45., 1./8., 1./10., 1./6., 9./50., 1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./10., 2./45., 1./8., 1./10., 1./6., 9./50., 1./4., 1./3., 1./2., 1.])
#
#
#
#
#
#         self.num_stages = self.alpha.shape[0]
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         Fproblem = LinearVariationalProblem(A, rhsproblem, self.F)
#         self.Fsolver = LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'kgrk2-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         for i in range(self.num_stages):
#             self.xk.assign(self.xn + self.dt * self.alpha[i] * self.F)
#             self.model.post_step(self.xk_sub, terms=self.terms)
#             self.t.assign(self.tn + self.c[i] * self.dt)
#             self.pre_step_solvers()
#             self.Fsolver.solve()
#
#         self.xn.assign(self.xn + self.dt * self.alpha[-1] * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#         self.tn = self.tn + dt
#
# class TimeSplitIntegrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#
#         self.num_subcycles = parameters['timestepping']['timestepper_substeps']
#
#         termlist = parameters['timestepping']['timestepper_split_terms']
#
#         self.xn = model.get_x_var('xn')
#         self.xn_sub = {}
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.xn_sub[var] = self.xn.sub(i)
#
#         self.time_integrators = []
#         for i,time_integrator_name in enumerate(parameters['timestepping']['timestepper_list']):
#             time_integrator = get_time_integrator(time_integrator_name)
#             self.time_integrators.append(time_integrator(parameters, model, initcond, logger, xn=self.xn, terms=termlist[i]))
#
#     def initialize(self):
#         self.xn.zero()
#         varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#         self.model.initialize(varexpr, self.xn)
#         for time_integrator in self.time_integrators:
#             time_integrator.initialize(init_xn=False)
#
# #ADD ABILITY TO INTERLEAVE THIS ORDERING IE model, HYPER, DYNANICS, HPER, PHYSICS
# #ADD ABILITY TO SWITCH BETWEEN LIE AND STRANG SPLITTING
#     def take_step(self, dt):
#         for i,time_integrator in enumerate(self.time_integrators):
#             for k in range(self.num_subcycles[i]):
#                 time_integrator.take_step(dt/self.num_subcycles[i])
#
# class Euler_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, use_xn_as_xk=True, terms=terms)
#
#         self.F1 = self.model.get_x_var('F1')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         F1problem = LinearVariationalProblem(A, rhsproblem, self.F1)
#         self.F1solver = LinearVariationalSolver(F1problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'euler-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.t.assign(self.tn)
#         self.pre_step_solvers()
#         self.F1solver.solve()
#
#         self.xn.assign(self.xn + self.dt * self.F1)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
#
# class SSPRK3_Integrator(RK_Integrator):
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F = self.model.get_x_var('F')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         Fproblem = LinearVariationalProblem(A, rhsproblem, self.F)
#         self.Fsolver = LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'ssprk3-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#         self.t.assign(self.tn)
#
#         self.xk.assign(self.xn)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
# #ALL WRONG
#         self.xk.assign(self.xn + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(self.xk + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(2./3. * self.xn + 1./3. * self.xk + 1./6. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xn.assign(self.xk + self.dt * 1./2. * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
#
# class SSPRK43_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F = self.model.get_x_var('F')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         Fproblem = LinearVariationalProblem(A, rhsproblem, self.F)
#         self.Fsolver = LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'ssprk43-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#         self.t.assign(self.tn)
#
#         self.xk.assign(self.xn)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(self.xn + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(self.xk + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(2./3. * self.xn + 1./3. * self.xk + 1./6. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xn.assign(self.xk + self.dt * 1./2. * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
