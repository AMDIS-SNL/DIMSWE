from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, LinearVariationalProblem, LinearVariationalSolver
from .parameters import overall_solver_parameters
import numpy as np
from firedrake import Constant, inner, grad, TestFunction, derivative, norm, assemble, Function, TestFunctions, split, TrialFunction, adjoint, action, replace
from .numpy_helpers import set_mixed_function_from_flattened_array, create_flattened_numpy_arr_from_mixed_function
from ufl.algorithms import extract_coefficients, expand_derivatives
from .profiling import region
from .assembly import PersistentAssembler

class DummySolver():
    def solve(self):
        pass


def is_zero_form(form):
    """Return True if form is the integer/float placeholder 0 used for absent residuals."""
    return isinstance(form, (int, float)) and form == 0


def depends_on(form, func):
    """Return True if the UFL form contains the Function func as a coefficient."""
    return any(coefficient is func for coefficient in extract_coefficients(form))


def directional_derivative(form, pairs):
    """Return the Gateaux derivative of form in the direction given by (Function, direction) pairs.

    The result is sum_k derivative(form, u_k, du_k) over the pairs whose Function u_k actually
    appears in form, with derivatives expanded so that the directions du_k appear explicitly
    (later derivatives with respect to a direction, as done when extracting a linear solve for
    it, then work). Returns None if form depends on none of the Functions.
    """
    result = None
    for func, direction in pairs:
        if func is None or not depends_on(form, func):
            continue
        term = derivative(form, func, direction)
        result = term if result is None else result + term
    if result is None:
        return None
    return expand_derivatives(result)


def accumulate(total, increment):
    """Add increment to total, where either may be None (meaning no contribution)."""
    if increment is None:
        return total
    if total is None:
        return increment
    return total + increment

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

        # coefficient direction theta_dot for tangent-linear and second-order adjoint steps; it is
        # an input like coeff, so reset_internal_vars does not touch it
        self.coeff_dir, _, _ = self.model.get_coeff_var('coeff_dir')

        self.t = Constant(1.)
        self.dt = Constant(1.)

    def set_coeff(self, coeff_val):
        if self.model.has_coeff():
            self.coeff.assign(coeff_val)

    def set_numpy_coeff(self, coeff_val_arr):
        set_mixed_function_from_flattened_array(self.coeff, coeff_val_arr)

    def set_coeff_direction(self, coeff_dir_val):
        """Set the coefficient direction theta_dot from a Function, or zero it if coeff_dir_val is None."""
        if self.model.has_coeff():
            if coeff_dir_val is None:
                self.coeff_dir.assign(0)
            else:
                self.coeff_dir.assign(coeff_dir_val)

    def set_numpy_coeff_direction(self, coeff_dir_arr):
        """Set the coefficient direction theta_dot from a flattened array (None zeroes it)."""
        if self.model.has_coeff():
            if coeff_dir_arr is None:
                self.coeff_dir.assign(0)
            else:
                set_mixed_function_from_flattened_array(self.coeff_dir, coeff_dir_arr)

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
#THIS IS A HACKY THING TO MAKE CF WORK DUE TO FPERP...
            if 'F' in self.model.get_aux_var_list():
                xi_split[self.xk_split['F'][0]] = self.Fi[i][2]['F'][0]
                xi_split[self.xk_split['F'][1]] = self.Fi[i][2]['F'][1]
            xi_splits.append(xi_split)

        #construct residuals for F and aux
        residuals_F = []
        residuals_aux = []
        for i in range(nstages):
        #this sign is due to writing things as dxdt + F(x) = 0
            rhs_Fi = -model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
            rhs_Fi = replace(rhs_Fi, xi_splits[i])
            residual_F = inner(xhat[0], self.Fi[i][0][0])*self.dx - rhs_Fi
            #print('resF', i, residual_F)
            #print('xisplits', xi_splits[i])
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

        # keep the stage and first-order adjoint residuals: the tangent-linear and second-order
        # adjoint forms are their directional derivatives, built lazily by _setup_second_order
        self.residuals_F = residuals_F
        self.residuals_aux = residuals_aux
        self.residuals_mu = residuals_mu
        self.residuals_muaux = residuals_muaux
        self.residual_xk = residual_xk
        self.residual_grad = residual_grad if model.has_coeff() else None
        self.xtrial = xtrial
        self._create_second_order_vars()
        self._second_order_ready = False
        # persistent assemblers of the increment forms L_delta_lambda, L_delta_grad, ...,
        # created on first use (see _assemble_increment)
        self._increment_assemblers = {}


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
        self._reset_second_order_vars()

    def _create_second_order_vars(self):
        """Create the Functions used by take_tangent_step and take_second_order_adjoint_step.

        Following the notation of docs/hessian_vector_products.md:
        - wk: tangent w_n of the step's start state xk
        - Fdoti[i]: tangent stage values (Fdot_i, Wdot_i), same layout as Fi[i]
        - mudoti[i]: second-order stage adjoints (mu_dot_i, nu_dot_i), same layout as mui[i]
        - lambda_dot_var: second-order adjoint lambda_dot_{n+1} at the end of the step
        - delta_lambda_dot_var: increment lambda_dot_n - lambda_dot_{n+1} (primal)
        - delta_grad_dot: coefficient-space unknown whose residual yields the coefficient
          Hessian-vector-product increment
        """
        self.wk, _, _ = self.model.get_x_var('wk')
        self.lambda_dot_var, _, _ = self.model.get_x_var('lambda_dot')
        self.delta_lambda_dot_var, _, _ = self.model.get_x_var('delta_lambda_dot')
        self.Fdoti = []
        self.mudoti = []
        for i in range(self.nstages):
            self.Fdoti.append(self.model.get_full_var('Fdot'+str(i), split_x_and_aux=self.is_explicit))
            self.mudoti.append(self.model.get_full_var('mudot'+str(i), split_x_and_aux=self.is_explicit))
        self.delta_grad_dot, _, _ = self.model.get_coeff_var('delta_grad_dot')

    def _reset_second_order_vars(self):
        """Zero the tangent and second-order adjoint internal variables (not coeff_dir)."""
        self.wk.assign(0)
        self.lambda_dot_var.assign(0)
        self.delta_lambda_dot_var.assign(0)
        if self.model.has_coeff():
            self.delta_grad_dot.assign(0)
        for i in range(self.nstages):
            for func in list(self.Fdoti[i][0]) + list(self.mudoti[i][0]):
                if func is not None:
                    func.assign(0)

    def _forward_pairs(self):
        """Return (Function, direction) pairs for all forward quantities a stage form depends on.

        These are the step start state xk, every stage value F_j and aux value W_j, and the
        coefficients, paired with their tangents.
        """
        pairs = [(self.xk[0], self.wk)]
        for i in range(self.nstages):
            pairs.append((self.Fi[i][0][0], self.Fdoti[i][0][0]))
            if self.model.has_aux():
                pairs.append((self.Fi[i][0][1], self.Fdoti[i][0][1]))
        if self.model.has_coeff():
            pairs.append((self.coeff, self.coeff_dir))
        return pairs

    def _adjoint_pairs(self):
        """Return (Function, direction) pairs for the first-order adjoint quantities.

        These are lambda_{n+1}, the stage adjoints (mu_i, nu_i) and the unknowns of the
        delta_lambda and delta_grad residuals, paired with their second-order counterparts.
        """
        pairs = [(self.lambda_var, self.lambda_dot_var), (self.delta_lambda_var, self.delta_lambda_dot_var)]
        for i in range(self.nstages):
            pairs.append((self.mui[i][0][0], self.mudoti[i][0][0]))
            if self.model.has_aux():
                pairs.append((self.mui[i][0][1], self.mudoti[i][0][1]))
        if self.model.has_coeff():
            pairs.append((self.delta_grad, self.delta_grad_dot))
        return pairs

    def _setup_second_order(self):
        """Build the tangent-linear and second-order adjoint forms and solvers on first use.

        Tangent-linear stage residuals are directional derivatives of the forward stage residuals
        (residuals_F, residuals_aux) in the direction of all forward tangents. Second-order adjoint
        residuals are directional derivatives of the first-order adjoint residuals (residuals_mu,
        residuals_muaux, residual_xk, residual_grad) in the direction of all forward tangents and
        all first-order adjoint tangents, so the resulting Hessian-vector product is the exact
        derivative of the gradient computed by take_adjoint_step. Each derivative is linear in
        its own unknown (Fdot_i, Wdot_i, mu_dot_i, ...), with the same mass-matrix operator as
        the corresponding first-order solve. The construction is lazy because compiling these
        second-derivative forms is costly and only needed for Hessian-vector products.
        """
        if self._second_order_ready:
            return
        if not self.is_explicit:
            raise NotImplementedError('tangent-linear and second-order adjoint steps are only implemented for explicit RK')
        # only the UFL form and solver construction is timed here; TSFC compilation of the new
        # forms happens on their first assembly, inside the first tangent/SOA solves
        with region("setup second order"):
            self._build_second_order_solvers()
        self._second_order_ready = True

    def _build_second_order_solvers(self):
        """Build the tangent-linear and second-order adjoint solvers (see _setup_second_order)."""
        sp = self.solver_parameters
        has_aux = self.model.has_aux()
        forward_pairs = self._forward_pairs()
        all_pairs = forward_pairs + self._adjoint_pairs()

        # tangent-linear stage solvers, M_W Wdot_i = ... and M Fdot_i = ...
        self.tlm_Fsolvers = []
        self.tlm_auxsolvers = []
        for i in range(self.nstages):
            if (not has_aux) or is_zero_form(self.residuals_aux[i]):
                self.tlm_auxsolvers.append(DummySolver())
            else:
                residual = directional_derivative(self.residuals_aux[i], forward_pairs)
                self.tlm_auxsolvers.append(create_linear_solver_from_residual(residual, self.Fdoti[i][0][1], self.xtrial[1], constant_jacobian=True, solver_parameters=sp['erkstage-aux'], options_prefix='erk-tlm-aux')[0])
            residual = directional_derivative(self.residuals_F[i], forward_pairs)
            self.tlm_Fsolvers.append(create_linear_solver_from_residual(residual, self.Fdoti[i][0][0], self.xtrial[0], constant_jacobian=True, solver_parameters=sp['erkstage-f'], options_prefix='erk-tlm-f')[0])

        # second-order adjoint stage solvers, M mu_dot_i = ... and M_W nu_dot_i = ...
        self.soa_musolvers = []
        self.soa_muauxsolvers = []
        for i in range(self.nstages):
            residual = directional_derivative(self.residuals_mu[i], all_pairs)
            self.soa_musolvers.append(create_linear_solver_from_residual(residual, self.mudoti[i][0][0], self.xtrial[0], constant_jacobian=True, solver_parameters=sp['erkstage-mu'], options_prefix='erk-soa-mu')[0])
            # mirror the first-order muauxsolvers, which exist exactly when the aux residual does
            if (not has_aux) or is_zero_form(self.residuals_aux[i]) or is_zero_form(self.residuals_muaux[i]):
                self.soa_muauxsolvers.append(DummySolver())
            else:
                residual = directional_derivative(self.residuals_muaux[i], all_pairs)
                self.soa_muauxsolvers.append(create_linear_solver_from_residual(residual, self.mudoti[i][0][1], self.xtrial[1], constant_jacobian=True, solver_parameters=sp['erkstage-muaux'], options_prefix='erk-soa-muaux')[0])

        # second-order lambda increment: M delta_lambda_dot = rdot^lam, with rdot^lam = L_delta_lambda_dot
        residual = directional_derivative(self.residual_xk, all_pairs)
        self.soa_deltalambdasolver, _, self.L_delta_lambda_dot = create_linear_solver_from_residual(residual, self.delta_lambda_dot_var, self.xtrial[0], constant_jacobian=True, solver_parameters=sp['erk-dlambda'], options_prefix='erk-soa-lambda')

        # coefficient Hessian-vector-product increment rdot^theta = L_delta_grad_dot (dual)
        if self.model.has_coeff():
            residual = directional_derivative(self.residual_grad, all_pairs)
            _, self.L_delta_grad_dot = extract_a_L_from_residual(residual, self.delta_grad_dot, self.grad_trial)

    def _assemble_increment(self, name):
        """Assemble the increment form self.<name> and return it as a rank-local flattened array.

        name is one of 'L_delta_lambda', 'L_delta_grad', 'L_delta_lambda_dot' and
        'L_delta_grad_dot'. A bare assemble() of these large forms repeats its per-call setup and
        cost 0.1-0.5 s per substep, so each form gets one PersistentAssembler that reuses its
        assembler and output (plan item B0, docs/hvp_performance_plan.md). The result is a new
        array, as create_flattened_numpy_arr_from_mixed_function(assemble(form)) returned before.
        """
        assembler = self._increment_assemblers.get(name)
        if assembler is None:
            assembler = PersistentAssembler(getattr(self, name))
            self._increment_assemblers[name] = assembler
        return assembler.assemble_flat()

    def _solve_forward_stages(self):
        """Solve the forward stage (and aux) equations from the current xk, t, dt and coeff."""
        with region("forward stages"):
            if self.is_explicit:
                for i in range(self.nstages):
                    self.auxsolvers[i].solve()
                    self.Fsolvers[i].solve()
            elif self.is_dirk:
                for i in range(self.nstages):
                    self.Fauxsolvers[i].solve()
            else:
                self.Fauxsolver.solve()

    def _solve_tangent_stages(self):
        """Solve the tangent-linear stage equations; requires current forward stages and wk."""
        with region("tangent stages"):
            for i in range(self.nstages):
                self.tlm_auxsolvers[i].solve()
                self.tlm_Fsolvers[i].solve()

    def _solve_first_order_adjoint_stages(self):
        """Solve the first-order stage adjoints mu_i, nu_i backward over the stages."""
        with region("adjoint stages"):
            if self.is_explicit:
                for i in range(self.nstages-1,-1,-1):
                    self.musolvers[i].solve()
                    self.muauxsolvers[i].solve()
            elif self.is_dirk:
                for i in range(self.nstages-1,-1,-1):
                    self.mufullsolvers.solve()
            else:
                self.mufullsolver.solve()

#EVENTUALLY MAKE THESE SPECIFIC TO A GIVEN TYPE OF RK IE SUBCLASS STUFF
    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):
        self.t.assign(tn)
        self.dt.assign(dt)
        self.xk[0].assign(xn[0])
#NEED SOME WAY OF DOING THIS FOR FULL SPACE, SINCE AUX INITIAL GUESS SHOULD GO IN FI...
        if len(self.xk) > 1:
            self.Fi[0][0][1].assign(xn[1])
        # the same stage solves, in the same order, as every recompute of the stages
        self._solve_forward_stages()
        #print('dt', self.dt)
        #print('xn', self.model.norm(xn[0]))

#THIS BREAKS A LITTLE FOR FULL SPACE VERSION
#IE WE SHOULD JUST BE ASSIGNING THE X VARIABLES HERE
#AND THEN DOING SOMETHING FOR THE AUX VARS

#the general solution is to assign each variable separately using sub
#this is maybe slower? probably not actually though...
        with region("update"):
            xnp1[0].assign(xn[0] + self.dt * sum(float(self.b[i]) * self.Fi[i][0][0] for i in range(self.nstages)))

#this gives a good initial guess at least?
            if len(xnp1) > 1:
                xnp1[1].assign(self.Fi[-1][0][1])

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
        self._solve_forward_stages()
        #print("lambda var norm", norm(self.lambda_var))

#EVENTUALLY MAKE THESE SPECIFIC TO A GIVEN TYPE OF RK IE SUBCLASS STUFF?
        #compute mu's
        self._solve_first_order_adjoint_stages()

        #compute grad
        if self.model.has_coeff():
            #self.deltagradsolver.solve()
            #delta_grad.assign(self.delta_grad)
            # increments use persistent assemblers (_assemble_increment): a bare assemble() of
            # these forms was the "very slow" set-up cost noted here
            with region("gradient increments"):
                delta_grad_rhs = self._assemble_increment('L_delta_grad')

        #compute lambda_n
        with region("lambda increment solve"):
            self.deltalambdasolver.solve()
            delta_lambda.assign(self.delta_lambda_var)
        with region("gradient increments"):
            delta_lambda_rhs = self._assemble_increment('L_delta_lambda')
        if self.model.has_coeff():
            return delta_lambda_rhs, delta_grad_rhs
        else:
            return delta_lambda_rhs, None

    def take_tangent_step(self, wnp1, wn, xn, tn, dt, xnp1=None):
        """Propagate a tangent-linear perturbation through one step (the boxed terms of eq. 2.11).

        Computes w_{n+1} = dN/dx(x_n) w_n + dN/dtheta(x_n) theta_dot, where theta_dot is the
        coefficient direction set with set_coeff_direction / set_numpy_coeff_direction. wn, wnp1,
        xn and (optionally) xnp1 are [x, aux] lists as used by take_forward_step; only their x
        entries are used for the tangents. The forward stages are recomputed from xn. If xnp1 is
        given, the forward update x_{n+1} is also written to it, so the forward and tangent
        trajectories can be advanced together without a second forward solve.
        """
        self._setup_second_order()
        self.t.assign(tn)
        self.dt.assign(dt)
        self.xk[0].assign(xn[0])
        self.wk.assign(wn[0])
        self._solve_forward_stages()
        self._solve_tangent_stages()
        with region("update"):
            wnp1[0].assign(wn[0] + self.dt * sum(float(self.b[i]) * self.Fdoti[i][0][0] for i in range(self.nstages)))
            if xnp1 is not None:
                xnp1[0].assign(xn[0] + self.dt * sum(float(self.b[i]) * self.Fi[i][0][0] for i in range(self.nstages)))
                # same aux initial guess as take_forward_step
                if len(xnp1) > 1 and xnp1[1] is not None:
                    xnp1[1].assign(self.Fi[-1][0][1])

    def take_second_order_adjoint_step(self, delta_lambda, delta_lambda_dot, lambda_np1, lambda_dot_np1, xn, wn, tnp1, dt):
        """Take one combined first-order and second-order adjoint step (eq. 2.11, forward-over-adjoint).

        Inputs are lambda_{n+1} (lambda_np1), lambda_dot_{n+1} (lambda_dot_np1), the step start
        state x_n (xn) and its tangent w_n (wn), as [x, aux] lists for xn and wn, plus the
        coefficient direction set with set_coeff_direction. The forward stages, tangent stages and
        first-order stage adjoints are recomputed internally; the second-order stage adjoints are
        then solved backward over the stages.

        delta_lambda and delta_lambda_dot are set to the primal increments lambda_n - lambda_{n+1}
        and lambda_dot_n - lambda_dot_{n+1}. Returns the flattened dual increments
        (delta_lambda_rhs, delta_grad_rhs, delta_lambda_dot_rhs, delta_hess_coeff_rhs): the first
        two equal the outputs of take_adjoint_step, the last two are their directional
        derivatives, i.e. this step's contributions to H_x0 v and H_theta v. The coefficient
        entries are None when the model has no coefficients.
        """
        self._setup_second_order()
        self.dt.assign(dt)
        self.t.assign(tnp1 - dt)
        self.lambda_var.assign(lambda_np1)
        self.lambda_dot_var.assign(lambda_dot_np1)
        self.xk[0].assign(xn[0])
        self.wk.assign(wn[0])

        # forward stages, tangent stages and first-order stage adjoints are all inputs to the
        # second-order stage equations
        self._solve_forward_stages()
        self._solve_tangent_stages()
        self._solve_first_order_adjoint_stages()

        # second-order stage adjoints, backward over the stages like the first-order ones
        with region("soa stages"):
            for i in range(self.nstages-1,-1,-1):
                self.soa_musolvers[i].solve()
                self.soa_muauxsolvers[i].solve()

        # first-order increments, identical to take_adjoint_step
        delta_grad_rhs = None
        delta_hess_coeff_rhs = None
        if self.model.has_coeff():
            with region("gradient increments"):
                delta_grad_rhs = self._assemble_increment('L_delta_grad')
            with region("hessian increments"):
                delta_hess_coeff_rhs = self._assemble_increment('L_delta_grad_dot')
        with region("lambda increment solve"):
            self.deltalambdasolver.solve()
            delta_lambda.assign(self.delta_lambda_var)
        with region("gradient increments"):
            delta_lambda_rhs = self._assemble_increment('L_delta_lambda')

        # second-order increments
        with region("lambda_dot increment solve"):
            self.soa_deltalambdasolver.solve()
            delta_lambda_dot.assign(self.delta_lambda_dot_var)
        with region("hessian increments"):
            delta_lambda_dot_rhs = self._assemble_increment('L_delta_lambda_dot')
        return delta_lambda_rhs, delta_grad_rhs, delta_lambda_dot_rhs, delta_hess_coeff_rhs


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

        # kept so the coefficient-gradient bookkeeping can be skipped for models without
        # coefficients, for which get_coeff_var returns None (as GeneralRK does)
        self.model = model
        self.subcycle_list = subcycle_list
        self.timestepper_list = timestepper_list
        self.termlist = termlist
        self.solver_parameters = solver_parameters

        self.time_integrators = []
        # profiling region names of the sub-integrators, e.g. "0:RK4(model)", so the cost of each
        # dynamics/forcing substep can be told apart (dimswe/profiling.py)
        self.profile_labels = []
        for i,time_integrator_name in enumerate(timestepper_list):
            time_integrator = get_time_integrator(time_integrator_name, None, solver_parameters)
            self.time_integrators.append(time_integrator(model, logger, solver_parameters, terms=termlist[i]))
            terms = termlist[i] if isinstance(termlist[i], (list, tuple)) else [termlist[i]]
            self.profile_labels.append(str(i) + ':' + str(time_integrator_name) + '(' + ','.join(str(term) for term in terms) + ')')

        self.lambda_k, self.lambda_sub, self.lambda_split = model.get_x_var('lambda_k')
        self.delta_lambda, _, _ = model.get_x_var('delta_lambda')
        self.delta_grad_coeff, _, _ = model.get_coeff_var('delta_grad_coeff')

#HOW DO WE HANDLE SPLITTING X AND AUX IN THE GENERAL RK CASE?
        self.xks = []
        self.xks.append(model.get_full_var('x0', split_x_and_aux=True))
        for i in range(len(self.time_integrators)):
            for k in range(self.subcycle_list[i]):
                self.xks.append(model.get_full_var('x'+str(i)+'_'+str(k), split_x_and_aux=True))

        # second-order variables: substep tangents w (parallel to xks), and the second-order
        # adjoint lambda_dot carried backward over the substeps
        self.lambda_dot_k, _, _ = model.get_x_var('lambda_dot_k')
        self.delta_lambda_dot, _, _ = model.get_x_var('delta_lambda_dot')
        self.wks = []
        self.wks.append(model.get_full_var('w0', split_x_and_aux=True))
        for i in range(len(self.time_integrators)):
            for k in range(self.subcycle_list[i]):
                self.wks.append(model.get_full_var('w'+str(i)+'_'+str(k), split_x_and_aux=True))

    def reset_internal_vars(self):
        self.lambda_k.assign(0)
        self.delta_lambda.assign(0)
        if self.model.has_coeff():
            self.delta_grad_coeff.assign(0)
        for xk, _, _ in self.xks:
            xk[0].assign(0)
        self.lambda_dot_k.assign(0)
        self.delta_lambda_dot.assign(0)
        for wk, _, _ in self.wks:
            wk[0].assign(0)
        for time_integrator in self.time_integrators:
            time_integrator.reset_internal_vars()

#DOES THIS NEED TO RESET INTERNAL VARS TO ENSURE REPEATABILITY?
    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):
        self.xks[0][0][0].assign(xn[0])
        l = 1
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
                with region(self.profile_labels[i]):
                    time_integrator.reset_internal_vars()
                    time_integrator.take_forward_step(self.xks[l][0], self.xks[l][1], self.xks[l-1][0], tn + k * sub_dt, sub_dt)
                l = l +1
        xnp1[0].assign(self.xks[-1][0][0])

    def take_adjoint_step(self, delta_grad_coeff, delta_lambda, lambda_np1, xn, tnp1, dt):
        """Take one first-order adjoint split step, carrying lambda backward through the substeps.

        The substep states are recomputed forward from xn, then the sub-integrators' adjoint steps
        are taken in reverse order. Returns (delta_lambda_rhs, delta_grad_rhs), the summed dual
        increments; delta_grad_rhs is None for a model without coefficients, in which case
        delta_grad_coeff (also None then) is not touched.
        """
        has_coeff = self.model.has_coeff()
        self.lambda_k.assign(lambda_np1)
        if has_coeff:
            delta_grad_coeff.assign(0)
        delta_lambda.assign(0)

        delta_lambda_rhs = 0
        delta_grad_rhs = None

        # recompute the substep states from xn (checkpoint + recompute within the step)
        with region("substep recompute"):
            self.xks[0][0][0].assign(xn[0])
            l = 1
            for i,time_integrator in enumerate(self.time_integrators):
                sub_dt = dt / self.subcycle_list[i]
                for k in range(self.subcycle_list[i]):
                    with region(self.profile_labels[i]):
                        time_integrator.reset_internal_vars()
                        time_integrator.take_forward_step(self.xks[l][0], self.xks[l][1], self.xks[l-1][0], tnp1 - dt + k * sub_dt, sub_dt)
                    l = l +1

        nintegrators = len(self.time_integrators)
        l = 0
        lt = len(self.xks)-1
        with region("substep adjoints"):
            for i,time_integrator in enumerate(reversed(self.time_integrators)):
                label = self.profile_labels[nintegrators-i-1]
                sub_dt = dt / self.subcycle_list[nintegrators-i-1]
                for k in reversed(range(self.subcycle_list[nintegrators-i-1])):
                    #print(i,k,l,lt-l-1,lt)
                    with region(label):
                        time_integrator.reset_internal_vars()
                        self.delta_lambda.assign(0)
                        if has_coeff:
                            self.delta_grad_coeff.assign(0)
                        # take_adjoint_step expects the substep end time: substep k runs from
                        # tnp1 - dt + k*sub_dt to tnp1 - dt + (k+1)*sub_dt, matching take_forward_step
                        _delta_lambda_rhs, _delta_grad_rhs = time_integrator.take_adjoint_step(self.delta_grad_coeff, self.delta_lambda, self.lambda_k, self.xks[lt-l-1][0], tnp1 - dt + (k + 1) * sub_dt, sub_dt)
                        self.lambda_k.assign(self.lambda_k + self.delta_lambda)
                        delta_lambda.assign(delta_lambda + self.delta_lambda)
                        if has_coeff:
                            delta_grad_coeff.assign(delta_grad_coeff + self.delta_grad_coeff)
                        # accumulate is None-aware: without coefficients delta_grad_rhs stays None, matching
                        # GeneralRK.take_adjoint_step
                        delta_grad_rhs = accumulate(delta_grad_rhs, _delta_grad_rhs)
                        delta_lambda_rhs = delta_lambda_rhs + _delta_lambda_rhs
                    #print(i,k,'grad coeff', _delta_grad_rhs)
                    #print(i,k,'delta lambda',_delta_lambda_rhs)
                    l = l + 1
        return delta_lambda_rhs, delta_grad_rhs

    def _compute_substep_tangents(self, xn, wn, tn, dt):
        """Recompute the substep states xks and tangents wks from x_n and w_n.

        Each sub-integrator advances the state and its tangent together with take_tangent_step,
        so xks matches what take_forward_step computes.
        """
        self.xks[0][0][0].assign(xn[0])
        self.wks[0][0][0].assign(wn[0])
        l = 1
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
                with region(self.profile_labels[i]):
                    time_integrator.reset_internal_vars()
                    time_integrator.take_tangent_step(self.wks[l][0], self.wks[l-1][0], self.xks[l-1][0], tn + k * sub_dt, sub_dt, xnp1=self.xks[l][0])
                l = l + 1

    def take_tangent_step(self, wnp1, wn, xn, tn, dt, xnp1=None):
        """Propagate a tangent-linear perturbation through one split step.

        Same interface as GeneralRK.take_tangent_step: the tangent is composed through the
        substeps of every sub-integrator, and x_{n+1} is also written to xnp1 if given.
        """
        self._compute_substep_tangents(xn, wn, tn, dt)
        wnp1[0].assign(self.wks[-1][0][0])
        if xnp1 is not None:
            xnp1[0].assign(self.xks[-1][0][0])

    def take_second_order_adjoint_step(self, delta_lambda, delta_lambda_dot, lambda_np1, lambda_dot_np1, xn, wn, tnp1, dt):
        """Take one combined first-order and second-order adjoint split step.

        Same interface and return values as GeneralRK.take_second_order_adjoint_step. The substep
        states and tangents are recomputed forward, then lambda and lambda_dot are carried
        backward through the substeps in reverse order, summing the dual increments.
        """
        self.lambda_k.assign(lambda_np1)
        self.lambda_dot_k.assign(lambda_dot_np1)
        delta_lambda.assign(0)
        delta_lambda_dot.assign(0)

        delta_lambda_rhs = None
        delta_grad_rhs = None
        delta_lambda_dot_rhs = None
        delta_hess_coeff_rhs = None

        # recompute the substep states and tangents from (xn, wn)
        with region("substep recompute"):
            self._compute_substep_tangents(xn, wn, tnp1 - dt, dt)

        nintegrators = len(self.time_integrators)
        l = 0
        lt = len(self.xks)-1
        with region("substep adjoints"):
            for i,time_integrator in enumerate(reversed(self.time_integrators)):
                label = self.profile_labels[nintegrators-i-1]
                nsub = self.subcycle_list[nintegrators-i-1]
                sub_dt = dt / nsub
                for k in reversed(range(nsub)):
                    with region(label):
                        time_integrator.reset_internal_vars()
                        self.delta_lambda.assign(0)
                        self.delta_lambda_dot.assign(0)
                        # substep k runs from tnp1 - dt + k*sub_dt to tnp1 - dt + (k+1)*sub_dt
                        increments = time_integrator.take_second_order_adjoint_step(self.delta_lambda, self.delta_lambda_dot,
                            self.lambda_k, self.lambda_dot_k, self.xks[lt-l-1][0], self.wks[lt-l-1][0], tnp1 - dt + (k + 1) * sub_dt, sub_dt)
                        _delta_lambda_rhs, _delta_grad_rhs, _delta_lambda_dot_rhs, _delta_hess_coeff_rhs = increments
                        self.lambda_k.assign(self.lambda_k + self.delta_lambda)
                        self.lambda_dot_k.assign(self.lambda_dot_k + self.delta_lambda_dot)
                        delta_lambda.assign(delta_lambda + self.delta_lambda)
                        delta_lambda_dot.assign(delta_lambda_dot + self.delta_lambda_dot)
                        delta_lambda_rhs = accumulate(delta_lambda_rhs, _delta_lambda_rhs)
                        delta_grad_rhs = accumulate(delta_grad_rhs, _delta_grad_rhs)
                        delta_lambda_dot_rhs = accumulate(delta_lambda_dot_rhs, _delta_lambda_dot_rhs)
                        delta_hess_coeff_rhs = accumulate(delta_hess_coeff_rhs, _delta_hess_coeff_rhs)
                    l = l + 1
        return delta_lambda_rhs, delta_grad_rhs, delta_lambda_dot_rhs, delta_hess_coeff_rhs

    def set_coeff(self, coeff_val):
        for time_integrator in self.time_integrators:
            time_integrator.set_coeff(coeff_val)


    def set_numpy_coeff(self, coeff_val_arr):
        for time_integrator in self.time_integrators:
            time_integrator.set_numpy_coeff(coeff_val_arr)

    def set_coeff_direction(self, coeff_dir_val):
        """Set the coefficient direction theta_dot on every sub-integrator (None zeroes it)."""
        for time_integrator in self.time_integrators:
            time_integrator.set_coeff_direction(coeff_dir_val)

    def set_numpy_coeff_direction(self, coeff_dir_arr):
        """Set the coefficient direction theta_dot on every sub-integrator from a flattened array."""
        for time_integrator in self.time_integrators:
            time_integrator.set_numpy_coeff_direction(coeff_dir_arr)







#RESTORE THIS EVENTUALLY

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





#PROBABLY ADD SOME STORMER-VERLET VARIANTS ALSO- "A" and "B"

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
