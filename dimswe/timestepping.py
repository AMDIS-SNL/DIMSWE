from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, LinearVariationalProblem, LinearVariationalSolver
from .parameters import overall_solver_parameters
import itertools
import numpy as np
from firedrake import Constant, inner, grad, TestFunction, derivative, norm, assemble, Function, TestFunctions, split, TrialFunction, adjoint, action, replace
from .numpy_helpers import set_mixed_function_from_flattened_array, create_flattened_numpy_arr_from_mixed_function, copy_function_values
from ufl.algorithms import extract_coefficients, expand_derivatives
from .profiling import region
from .assembly import AffineSolver, PersistentAssembler, is_direct_solver, petsc_solver_parameters, shared_mass_solver

def add_stage_substitution(mapping, key, value):
    """Add key -> value to a UFL replace() mapping, together with every fixed component of key.

    For a tensor-valued key (for example the ListTensor that split() returns for a vector
    subspace), key[idx] -> value[idx] is also added for every component index idx, so that
    expressions which index the key with fixed integers are replaced as well.
    """
    mapping[key] = value
    for idx in itertools.product(*(range(n) for n in key.ufl_shape)):
        if idx:
            mapping[key[idx]] = value[idx]


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
    # one walk of the form for its coefficients, instead of one per pair (depends_on); the
    # second-order setup calls this for large forms with 2 x nstages + 3 pairs
    present = {id(coefficient) for coefficient in extract_coefficients(form)}
    result = None
    for func, direction in pairs:
        if func is None or id(func) not in present:
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
        # optional: false re-solves the stages in every adjoint substep instead of reusing them
        reuse_stages = parameters['timestepping'].get('reuse_stages', True)

        return lambda model, logger, coeffs: LieSplittingIntegrator(model, logger, timestepper_list, termlist, subcycle_list, solver_parameters, reuse_stages=reuse_stages)
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
        xi_splits = self.build_stage_substitutions()

        #construct residuals for F and aux
        residuals_F = []
        residuals_aux = []
        # the W-dependent left-hand side of each aux residual, whose derivative is the aux operator
        aux_lhss = []
        for i in range(nstages):
        #this sign is due to writing things as dxdt + F(x) = 0
            rhs_Fi = -model.rhs(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
            rhs_Fi = replace(rhs_Fi, xi_splits[i])
            residual_F = inner(xhat[0], self.Fi[i][0][0])*self.dx - rhs_Fi
            #print('resF', i, residual_F)
            #print('xisplits', xi_splits[i])
#MIGHT HAVE TO MODIFY THIS INNER PRODUCT A LITTLE FOR IRK GENERALITY

            residual_aux = 0
            aux_lhs = 0
            if self.model.has_aux():
                aux_expressions = self.model.compute_aux_expressions(self.xk_split, self.t, self.coeff_split, xhat_subs, terms=terms)
                rhs_W = 0
                lhs_W = 0
                for var in self.model.get_aux_var_list(terms=terms):
                    rhs_W = rhs_W + aux_expressions[var][1]
                    lhs_W = lhs_W + aux_expressions[var][0]
                if not (lhs_W == 0):
                    lhs_W = lhs_W + self.unowned_aux_identity(terms, xhat_subs)
                    residual_aux = replace(lhs_W - rhs_W, xi_splits[i])
                    aux_lhs = replace(lhs_W, xi_splits[i])


            residuals_aux.append(residual_aux)
            aux_lhss.append(aux_lhs)
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
        # persistent assemblers of the coefficient increment forms L_delta_grad and
        # L_delta_grad_dot, created on first use (see _assemble_increment); the lambda increments
        # have their own assemblers inside their AffineSolvers
        self._increment_assemblers = {}
        # AffineSolvers of the lambda and lambda_dot increments, keyed by the old right-hand side
        # names L_delta_lambda / L_delta_lambda_dot (see _solve_increment), and of the stage
        # equations, keyed by stage kind and index ('F0', 'aux0', 'mu0', 'tlm_F0', 'soa_mu0', ...)
        self._increment_solvers = {}
        self._stage_solvers = {}
        # factored operator matrices, keyed by mass_solver_key; shared by all sub-integrators of
        # the model, so a LieSplit of four integrators factors the x-space mass matrix once
        self._mass_solvers = model.__dict__.setdefault('_shared_mass_solvers', {})
        # stage values saved by _store_stage_kind, keyed by (slot, kind) with kind 'forward',
        # 'tangent' or 'adjoint'; integer slots are LieSplit substeps within one step (plan item
        # A3), tuple slots are steps of a checkpointed trajectory (plan items A1/A2)
        self._stage_store = {}
        # whether the trajectory stage checkpoint (store_slot / checkpoint_slot arguments) works;
        # only explicit RK keeps its stage values in separate Functions
        self.supports_stage_checkpoint = self.is_explicit


#EVENTUALLY MAKE THESE SPECIFIC TO A GIVEN TYPE OF RK IE SUBCLASS STUFF
        #construct solvers
        if self.is_explicit:
            # operators of the stage equations (see _affine_solver): every x-space residual
            # (F_i, mu_i, delta_lambda and their second-order counterparts) is
            # inner(var, xhat[0])*dx plus terms without var; the aux residuals are lhs_W(W_i)
            # plus terms without W_i, and the aux adjoints inner(nu_i, xhat[1])*dx plus the rest
            self.mass_x = inner(xtrial[0], xhat[0])*self.dx
            self.mass_aux = []
            self.mass_muaux = inner(xtrial[1], xhat[1])*self.dx if self.model.has_aux() else None
            self.Fsolvers = []
            self.auxsolvers = []
            self.musolvers = []
            self.muauxsolvers = []
            for i in range(self.nstages):
                #print('creating solvers', i)
                if (not self.model.has_aux()) or is_zero_form(residuals_aux[i]):
                    self.mass_aux.append(None)
                    self.auxsolvers.append(DummySolver())
                    self.muauxsolvers.append(DummySolver())
                else:
#THIS IS GOING TO FAIL WHEN AUXILIARY VARIABLES HAVE NON-CONSTANT JACOBIANS
#CAN/SHOULD MAYBE SPLIT THESE INTO SEPARATE SOLVERS FOR EACH AUX VAR?
#THIS WILL BE A MESS FOR COMPUTING ADJOINT STUFF, BUT MANAGEABLE!
                    # a small form: only the left-hand side is differentiated
                    self.mass_aux.append(expand_derivatives(derivative(aux_lhss[i], self.Fi[i][0][1], xtrial[1])))
                    self.auxsolvers.append(self._affine_solver('aux'+str(i), residuals_aux[i], self.Fi[i][0][1], self.mass_aux[i], xtrial[1], 'erkstage-aux', 'erk-aux'))
                    self.muauxsolvers.append(self._affine_solver('muaux'+str(i), residuals_muaux[i], self.mui[i][0][1], self.mass_muaux, xtrial[1], 'erkstage-muaux', 'erk-muaux'))

                self.Fsolvers.append(self._affine_solver('F'+str(i), residuals_F[i], self.Fi[i][0][0], self.mass_x, xtrial[0], 'erkstage-f', 'erk-f'))
                self.musolvers.append(self._affine_solver('mu'+str(i), residuals_mu[i], self.mui[i][0][0], self.mass_x, xtrial[0], 'erkstage-mu', 'erk-mu'))
            self._increment_solvers['L_delta_lambda'] = self._affine_solver(None, residual_xk, self.delta_lambda_var, self.mass_x, xtrial[0], 'erk-dlambda', 'erk-lambda', returns_rhs=True)
            if model.has_coeff():
                # delta_grad is never solved for and stays zero, so the old right-hand side
                # action(a, delta_grad) - residual_grad is -residual_grad at delta_grad = 0;
                # _assemble_increment zeroes it before assembling. This avoids a UFL expansion of
                # the whole form in action().
                self.L_delta_grad = -residual_grad
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


    def unowned_aux_identity(self, terms, xhat_subs):
        """Return mass terms setting the aux variables of other terms to zero in this integrator's solve.

        Each integrator solves for the whole aux space, but only the aux variables of its own terms
        have equations. When several split integrators own aux variables (for example two forcing
        terms with aux variables in different LieSplit stages), the other blocks would be empty,
        which makes direct (LU) aux solves singular. Those variables are not used by this
        integrator's terms, so they are set to zero. This matches the adjoint aux residual, which
        uses the full aux mass matrix.
        """
        own = set(self.model.get_aux_var_list(terms=terms))
        identity = 0
        for var in self.model.get_aux_var_list():
            if var not in own:
                identity = identity + inner(xhat_subs[var], self.xk_split[var])*self.dx
        return identity

    def build_stage_substitutions(self):
        """Return, for every RK stage i, the UFL replace() mapping from xk to the stage values.

        The residuals are written in terms of xk_split and then replaced by the stage values:
        x variables by xk + dt * sum_j A[i,j] F_j, aux variables by their stage-i values F_i, and
        t by t + c_i dt. The keys for vector-valued variables are the ListTensors returned by split(),
        but UFL simplifies a fixed-index component such as u[1] (used by rot2D, curl2D, ...) directly
        to the underlying mixed-function component, so the ListTensor key does not catch it. Each
        component is therefore mapped as well; otherwise those components silently keep the values
        stored in xk (for aux variables, zeros, since xk's aux part is never assigned).
        """
        xi_splits = []
        for i in range(self.nstages):
            xi_split = {}
            for var in self.model.get_x_var_list():
                stage_value = self.xk_split[var]
                for j in range(self.nstages):
                    if not (self.A[i,j] == 0):
                        stage_value = stage_value + self.dt*float(self.A[i,j]) * self.Fi[j][2][var]
                add_stage_substitution(xi_split, self.xk_split[var], stage_value)
            for var in self.model.get_aux_var_list():
                add_stage_substitution(xi_split, self.xk_split[var], self.Fi[i][2][var])
            xi_split[self.t] = self.t + float(self.c[i])*self.dt
            xi_splits.append(xi_split)
        return xi_splits

    def get_rhs_expr(self):

        xhat, xhat_subs = self.model.get_full_test_vars(split_x_and_aux=self.is_explicit)

        xi_splits = self.build_stage_substitutions()

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

        # Each residual below is affine in its unknown with the operator of the corresponding
        # first-order solve (mass_x, mass_aux[i], mass_muaux), so they all share the first-order
        # factorizations (_affine_solver). Only the directional derivatives themselves are built
        # here; no action() expansions or LinearVariationalSolvers, which were most of this setup.

        # tangent-linear stage solvers, M_W Wdot_i = ... and M Fdot_i = ...
        self.tlm_Fsolvers = []
        self.tlm_auxsolvers = []
        for i in range(self.nstages):
            if (not has_aux) or is_zero_form(self.residuals_aux[i]):
                self.tlm_auxsolvers.append(DummySolver())
            else:
                residual = directional_derivative(self.residuals_aux[i], forward_pairs)
                self.tlm_auxsolvers.append(self._affine_solver('tlm_aux'+str(i), residual, self.Fdoti[i][0][1], self.mass_aux[i], self.xtrial[1], 'erkstage-aux', 'erk-tlm-aux'))
            residual = directional_derivative(self.residuals_F[i], forward_pairs)
            self.tlm_Fsolvers.append(self._affine_solver('tlm_F'+str(i), residual, self.Fdoti[i][0][0], self.mass_x, self.xtrial[0], 'erkstage-f', 'erk-tlm-f'))

        # second-order adjoint stage solvers, M mu_dot_i = ... and M_W nu_dot_i = ...
        self.soa_musolvers = []
        self.soa_muauxsolvers = []
        for i in range(self.nstages):
            residual = directional_derivative(self.residuals_mu[i], all_pairs)
            self.soa_musolvers.append(self._affine_solver('soa_mu'+str(i), residual, self.mudoti[i][0][0], self.mass_x, self.xtrial[0], 'erkstage-mu', 'erk-soa-mu'))
            # mirror the first-order muauxsolvers, which exist exactly when the aux residual does
            if (not has_aux) or is_zero_form(self.residuals_aux[i]) or is_zero_form(self.residuals_muaux[i]):
                self.soa_muauxsolvers.append(DummySolver())
            else:
                residual = directional_derivative(self.residuals_muaux[i], all_pairs)
                self.soa_muauxsolvers.append(self._affine_solver('soa_muaux'+str(i), residual, self.mudoti[i][0][1], self.mass_muaux, self.xtrial[1], 'erkstage-muaux', 'erk-soa-muaux'))

        # second-order lambda increment: M delta_lambda_dot = rdot^lam, where the returned
        # rdot^lam is the old right-hand side L_delta_lambda_dot
        residual = directional_derivative(self.residual_xk, all_pairs)
        self._increment_solvers['L_delta_lambda_dot'] = self._affine_solver(None, residual, self.delta_lambda_dot_var, self.mass_x, self.xtrial[0], 'erk-dlambda', 'erk-soa-lambda', returns_rhs=True)

        # coefficient Hessian-vector-product increment rdot^theta = L_delta_grad_dot (dual); as
        # for L_delta_grad, delta_grad_dot is never solved for and is zeroed before assembly
        if self.model.has_coeff():
            residual = directional_derivative(self.residual_grad, all_pairs)
            self.L_delta_grad_dot = -residual

    def _affine_solver(self, name, residual, var, mass_form, trial, parameters_key, options_prefix, returns_rhs=False):
        """Return an AffineSolver for residual(var) = 0, registered in _stage_solvers under name (if given).

        residual is affine in var with operator mass_form, so the solve is one assembly of the
        residual at var = 0 and a solve with the model's shared factorization of mass_form
        (dimswe/assembly.py, plan item C1). With iterative solver parameters, or if the residual
        check of the first solves fails, it uses the old LinearVariationalSolver, which is only
        built then. returns_rhs: solve() returns the flattened dual increment (the old
        right-hand side L = action(a, var) - residual at any var), as the increments need.
        """
        solver_parameters = self.solver_parameters[parameters_key]

        def make_reference():
            """The previous solver of this equation (the fallback)."""
            return create_linear_solver_from_residual(residual, var, trial, constant_jacobian=True,
                                                      solver_parameters=petsc_solver_parameters(solver_parameters),
                                                      options_prefix=options_prefix)[0]

        mass_solver = None
        if is_direct_solver(solver_parameters):
            mass_solver = shared_mass_solver(self._mass_solvers, mass_form, solver_parameters)
        solver = AffineSolver(residual, var, mass_solver, make_reference, returns_rhs=returns_rhs)
        if name is not None:
            self._stage_solvers[name] = solver
        return solver

    def _stage_functions(self, kind):
        """Return the Functions holding one kind of stage values, x and aux parts, in stage order.

        kind is 'forward' for (F_i, W_i), 'tangent' for (Fdot_i, Wdot_i), or 'adjoint' for the
        first-order stage adjoints (mu_i, nu_i) followed by lambda_var, the adjoint lambda_{n+1}
        they were solved with, and delta_lambda_var, the solved increment lambda_n - lambda_{n+1}
        (all inputs or unknowns of the first-order part of the second-order adjoint step).
        """
        if kind == 'adjoint':
            stages = self.mui
        elif kind == 'tangent':
            stages = self.Fdoti
        elif kind == 'forward':
            stages = self.Fi
        else:
            raise ValueError('unknown stage kind ' + str(kind))
        funcs = [func for i in range(self.nstages) for func in stages[i][0] if func is not None]
        if kind == 'adjoint':
            funcs += [self.lambda_var, self.delta_lambda_var]
        return funcs

    def _store_stage_kind(self, slot, kind):
        """Copy the current stage values of one kind into storage slot (allocated on first use)."""
        with region("store stages"):
            current = self._stage_functions(kind)
            saved = self._stage_store.get((slot, kind))
            if saved is None:
                self._stage_store[(slot, kind)] = [func.copy(deepcopy=True) for func in current]
            else:
                # plain value copies: an assign per Function cost ~0.3 ms (see copy_function_values)
                for target, func in zip(saved, current):
                    copy_function_values(func, target)

    def _restore_stage_kind(self, slot, kind):
        """Assign the stage values of one kind stored under slot back to the stage Functions."""
        saved = self._stage_store.get((slot, kind))
        if saved is None:
            raise KeyError('no ' + kind + ' stage values stored for slot ' + str(slot))
        for func, value in zip(self._stage_functions(kind), saved):
            copy_function_values(value, func)

    def store_stages(self, slot, tangents=False):
        """Copy the current stage values into storage slot (plan item A3).

        A LieSplit step solves each substep's stages while it recomputes the substep states, and
        the adjoint steps of the same substeps used to solve them again. The split integrator
        stores them here under the substep index (slot) and passes the slot to take_adjoint_step
        or take_second_order_adjoint_step, which restore them instead of re-solving. With
        tangents=True the tangent stages are stored as well. Only explicit RK keeps its stages in
        separate Functions; for other schemes this does nothing and returns False.
        """
        if not self.is_explicit:
            return False
        self._store_stage_kind(slot, 'forward')
        if tangents:
            self._store_stage_kind(slot, 'tangent')
        return True

    def _restore_stages(self, slot, tangents=False):
        """Assign stored stage values (and tangent stages if tangents) back to the stage Functions."""
        self._restore_stage_kind(slot, 'forward')
        if tangents:
            self._restore_stage_kind(slot, 'tangent')

    def _assemble_increment(self, name):
        """Assemble the increment form self.<name> and return it as a rank-local flattened array.

        name is 'L_delta_grad' or 'L_delta_grad_dot' (the lambda increments are assembled by their
        AffineSolvers, see _solve_increment). A bare assemble() of these large forms repeats its per-call setup and
        cost 0.1-0.5 s per substep, so each form gets one PersistentAssembler that reuses its
        assembler and output (plan item B0, docs/hvp_performance_plan.md). The result is a new
        array, as create_flattened_numpy_arr_from_mixed_function(assemble(form)) returned before.
        """
        assembler = self._increment_assemblers.get(name)
        if assembler is None:
            assembler = PersistentAssembler(getattr(self, name))
            self._increment_assemblers[name] = assembler
        # the forms are -residual, equal to the old action(a, delta) - residual only at delta = 0
        # (the unknowns are never solved for, so this only guards that invariant)
        unknown = {'L_delta_grad': self.delta_grad, 'L_delta_grad_dot': self.delta_grad_dot}.get(name)
        if unknown is not None:
            unknown.dat.zero()
        return assembler.assemble_flat()

    def _solve_increment(self, name):
        """Solve the increment equation of right-hand side name; return that right-hand side flattened.

        name is 'L_delta_lambda' (unknown delta_lambda_var) or 'L_delta_lambda_dot' (unknown
        delta_lambda_dot_var). The AffineSolver assembles the increment residual once, at a zero
        unknown, which gives minus the right-hand side, and solves with the model's factored
        x-space mass matrix. The returned array equals the old right-hand side form assembled
        after the solve.
        """
        return self._increment_solvers[name].solve()

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
    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt, store_slot=None):
        """Advance xn by one step of size dt from time tn into xnp1.

        With store_slot, the stage values are also stored under that slot for the trajectory
        stage checkpoint (plan items A1/A2): the adjoint, tangent and second-order adjoint steps
        of the same step then restore them via checkpoint_slot instead of solving them again.
        """
        self.t.assign(tn)
        self.dt.assign(dt)
        self.xk[0].assign(xn[0])
#NEED SOME WAY OF DOING THIS FOR FULL SPACE, SINCE AUX INITIAL GUESS SHOULD GO IN FI...
        if len(self.xk) > 1:
            self.Fi[0][0][1].assign(xn[1])
        # the same stage solves, in the same order, as every recompute of the stages
        self._solve_forward_stages()
        if store_slot is not None:
            self._store_stage_kind(store_slot, 'forward')
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

    def take_adjoint_step(self, delta_grad, delta_lambda, lambda_np1, xn, tnp1, dt, stages_slot=None,
                          store_slot=None, checkpoint_slot=None):
        """Take one first-order adjoint step from lambda_{n+1}, for the step that started at xn.

        Sets delta_lambda to the primal increment lambda_n - lambda_{n+1} and returns the dual
        increments (delta_lambda_rhs, delta_grad_rhs); delta_grad_rhs is None without
        coefficients. The forward stages are re-solved from xn, unless stages_slot names stage
        values saved for exactly this step (store_stages, plan item A3, or take_forward_step's
        store_slot), which are restored instead. With store_slot, the first-order stage adjoints,
        lambda_{n+1} and the solved delta_lambda are stored under it for
        take_second_order_adjoint_step's adjoint_slot.
        checkpoint_slot is shorthand for stages_slot = store_slot = checkpoint_slot, the
        trajectory stage checkpoint of plan items A1/A2.
        """
        if checkpoint_slot is not None:
            stages_slot = store_slot = checkpoint_slot
        self.dt.assign(dt)
        self.t.assign(tnp1 - dt)
        self.lambda_var.assign(lambda_np1)

        #take a forward step from xn to populate Fi (or restore the stages of that step)
        self.xk[0].assign(xn[0])
        if stages_slot is None:
            self._solve_forward_stages()
        else:
            with region("restore stages"):
                self._restore_stage_kind(stages_slot, 'forward')
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

        #compute lambda_n: one assembly of the right-hand side, which is also the dual
        # increment, and a solve with the factored mass matrix (_solve_increment)
        with region("lambda increment solve"):
            delta_lambda_rhs = self._solve_increment('L_delta_lambda')
            delta_lambda.assign(self.delta_lambda_var)
        if store_slot is not None:
            # stage adjoints, lambda_{n+1} and the solved delta_lambda, for adjoint_slot
            self._store_stage_kind(store_slot, 'adjoint')
        if self.model.has_coeff():
            return delta_lambda_rhs, delta_grad_rhs
        else:
            return delta_lambda_rhs, None

    def take_tangent_step(self, wnp1, wn, xn, tn, dt, xnp1=None, stages_slot=None, store_slot=None, checkpoint_slot=None):
        """Propagate a tangent-linear perturbation through one step (the boxed terms of eq. 2.11).

        Computes w_{n+1} = dN/dx(x_n) w_n + dN/dtheta(x_n) theta_dot, where theta_dot is the
        coefficient direction set with set_coeff_direction / set_numpy_coeff_direction. wn, wnp1,
        xn and (optionally) xnp1 are [x, aux] lists as used by take_forward_step; only their x
        entries are used for the tangents. The forward stages are recomputed from xn. If xnp1 is
        given, the forward update x_{n+1} is also written to it, so the forward and tangent
        trajectories can be advanced together without a second forward solve.

        With stages_slot, the forward stages are restored from that slot (stored by
        take_forward_step's store_slot) instead of being solved; with store_slot, the tangent
        stages are stored under it for take_second_order_adjoint_step. checkpoint_slot is
        shorthand for both (plan items A1/A2). The restored stages are the ones the solve would
        produce, so x_{n+1} is unchanged.
        """
        if checkpoint_slot is not None:
            stages_slot = store_slot = checkpoint_slot
        self._setup_second_order()
        self.t.assign(tn)
        self.dt.assign(dt)
        self.xk[0].assign(xn[0])
        self.wk.assign(wn[0])
        if stages_slot is None:
            self._solve_forward_stages()
        else:
            with region("restore stages"):
                self._restore_stage_kind(stages_slot, 'forward')
        self._solve_tangent_stages()
        if store_slot is not None:
            self._store_stage_kind(store_slot, 'tangent')
        with region("update"):
            wnp1[0].assign(wn[0] + self.dt * sum(float(self.b[i]) * self.Fdoti[i][0][0] for i in range(self.nstages)))
            if xnp1 is not None:
                xnp1[0].assign(xn[0] + self.dt * sum(float(self.b[i]) * self.Fi[i][0][0] for i in range(self.nstages)))
                # same aux initial guess as take_forward_step
                if len(xnp1) > 1 and xnp1[1] is not None:
                    xnp1[1].assign(self.Fi[-1][0][1])

    def take_second_order_adjoint_step(self, delta_lambda, delta_lambda_dot, lambda_np1, lambda_dot_np1, xn, wn, tnp1, dt, stages_slot=None,
                                       adjoint_slot=None, checkpoint_slot=None):
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

        If stages_slot is given, the forward and tangent stages are restored from values saved
        for exactly this step (store_stages(stages_slot, tangents=True), plan item A3, or the
        store_slot of take_forward_step and take_tangent_step) instead of being re-solved.

        If adjoint_slot is given, the first-order stage adjoints, lambda_{n+1} and the solved
        delta_lambda are restored from the values take_adjoint_step stored under it (lambda_np1
        is then ignored), and the first-order part is skipped: no first-order solves or
        increment assemblies; delta_lambda is set to the restored increment, and
        delta_lambda_rhs and delta_grad_rhs are returned as None. The caller takes the gradient
        from the gradient sweep that stored them. Every Function the second-order forms can see
        has the value it has in the full step (delta_grad is never solved in either path), so
        the second-order increments are unchanged.
        checkpoint_slot is shorthand for stages_slot = adjoint_slot = checkpoint_slot (plan items
        A1/A2).
        """
        if checkpoint_slot is not None:
            stages_slot = adjoint_slot = checkpoint_slot
        self._setup_second_order()
        self.dt.assign(dt)
        self.t.assign(tnp1 - dt)
        self.lambda_var.assign(lambda_np1)
        self.lambda_dot_var.assign(lambda_dot_np1)
        self.xk[0].assign(xn[0])
        self.wk.assign(wn[0])

        # forward stages, tangent stages and first-order stage adjoints are all inputs to the
        # second-order stage equations
        if stages_slot is None:
            self._solve_forward_stages()
            self._solve_tangent_stages()
        else:
            with region("restore stages"):
                self._restore_stages(stages_slot, tangents=True)
        if adjoint_slot is None:
            self._solve_first_order_adjoint_stages()
        else:
            with region("restore stages"):
                self._restore_stage_kind(adjoint_slot, 'adjoint')

        # second-order stage adjoints, backward over the stages like the first-order ones
        with region("soa stages"):
            for i in range(self.nstages-1,-1,-1):
                self.soa_musolvers[i].solve()
                self.soa_muauxsolvers[i].solve()

        # first-order increments, identical to take_adjoint_step (skipped with adjoint_slot)
        first_order = adjoint_slot is None
        delta_grad_rhs = None
        delta_hess_coeff_rhs = None
        delta_lambda_rhs = None
        if self.model.has_coeff():
            if first_order:
                with region("gradient increments"):
                    delta_grad_rhs = self._assemble_increment('L_delta_grad')
            with region("hessian increments"):
                delta_hess_coeff_rhs = self._assemble_increment('L_delta_grad_dot')
        if first_order:
            with region("lambda increment solve"):
                delta_lambda_rhs = self._solve_increment('L_delta_lambda')
                delta_lambda.assign(self.delta_lambda_var)
        else:
            delta_lambda.assign(self.delta_lambda_var)

        # second-order increments
        with region("lambda_dot increment solve"):
            delta_lambda_dot_rhs = self._solve_increment('L_delta_lambda_dot')
            delta_lambda_dot.assign(self.delta_lambda_dot_var)
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
    def __init__(self, model, logger, timestepper_list, termlist, subcycle_list, solver_parameters, reuse_stages=True):
        """Compose explicit RK sub-integrators, each advancing its terms with subcycling.

        reuse_stages (plan item A3, docs/hvp_performance_plan.md): the adjoint and second-order
        adjoint steps recompute the substep states, which solves every substep's stages; with
        reuse_stages those stage values are kept per substep and restored in the substep adjoint
        steps instead of being solved a second time. False keeps the original recompute (one
        forward pass, and for HVPs one tangent pass, more per substep); the results agree to
        round-off.
        """

        # kept so the coefficient-gradient bookkeeping can be skipped for models without
        # coefficients, for which get_coeff_var returns None (as GeneralRK does)
        self.model = model
        self.reuse_stages = bool(reuse_stages)
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

        # trajectory stage checkpoint (plan items A1/A2): substep states 'x' and tangents 'w'
        # of checkpointed steps, keyed by (slot, substep index, kind); the stage values live in
        # the sub-integrators under the slot (slot, substep index)
        self._substep_store = {}
        self.supports_stage_checkpoint = all(getattr(integrator, 'supports_stage_checkpoint', False)
                                             for integrator in self.time_integrators)

    def _store_substep(self, slot, l, kind, func):
        """Copy a substep state ('x') or tangent ('w') Function into the trajectory store."""
        with region("store stages"):
            saved = self._substep_store.get((slot, l, kind))
            if saved is None:
                self._substep_store[(slot, l, kind)] = func.copy(deepcopy=True)
            else:
                copy_function_values(func, saved)

    def _restore_substeps(self, slot, kind):
        """Assign the stored substep states ('x', into xks) or tangents ('w', into wks) of a step."""
        targets = self.xks if kind == 'x' else self.wks
        with region("restore stages"):
            for l, (funcs, _, _) in enumerate(targets):
                saved = self._substep_store.get((slot, l, kind))
                if saved is None:
                    raise KeyError('no substep ' + kind + ' stored for slot ' + str(slot))
                copy_function_values(saved, funcs[0])

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
    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt, store_slot=None):
        """Advance xn through all substeps into xnp1.

        With store_slot (trajectory stage checkpoint, plan items A1/A2), every substep's stage
        values are stored in its sub-integrator under (store_slot, l), and the substep states
        under (store_slot, l, 'x'), for the checkpoint_slot of the adjoint steps.
        """
        self.xks[0][0][0].assign(xn[0])
        if store_slot is not None:
            self._store_substep(store_slot, 0, 'x', self.xks[0][0][0])
        l = 1
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
                with region(self.profile_labels[i]):
                    time_integrator.reset_internal_vars()
                    sub_slot = None if store_slot is None else (store_slot, l)
                    time_integrator.take_forward_step(self.xks[l][0], self.xks[l][1], self.xks[l-1][0], tn + k * sub_dt, sub_dt,
                                                      store_slot=sub_slot)
                    if store_slot is not None:
                        self._store_substep(store_slot, l, 'x', self.xks[l][0][0])
                l = l +1
        xnp1[0].assign(self.xks[-1][0][0])

    def take_adjoint_step(self, delta_grad_coeff, delta_lambda, lambda_np1, xn, tnp1, dt, checkpoint_slot=None):
        """Take one first-order adjoint split step, carrying lambda backward through the substeps.

        The substep states are recomputed forward from xn (storing their stage values with
        reuse_stages, so the adjoint steps need not solve them again), then the sub-integrators'
        adjoint steps are taken in reverse order. Returns (delta_lambda_rhs, delta_grad_rhs), the summed dual
        increments; delta_grad_rhs is None for a model without coefficients, in which case
        delta_grad_coeff (also None then) is not touched.

        With checkpoint_slot (plan items A1/A2), the step was taken by take_forward_step with
        store_slot=checkpoint_slot: the substep states and stages are restored instead of
        recomputed, and every substep's stage adjoints and lambda are stored for
        take_second_order_adjoint_step(checkpoint_slot=...).
        """
        has_coeff = self.model.has_coeff()
        self.lambda_k.assign(lambda_np1)
        if has_coeff:
            delta_grad_coeff.assign(0)
        delta_lambda.assign(0)

        delta_lambda_rhs = 0
        delta_grad_rhs = None

        # recompute the substep states from xn (checkpoint + recompute within the step); with
        # reuse_stages, substep l's stage values are stored under slot l for its adjoint step
        if checkpoint_slot is not None:
            # trajectory checkpoint: states and stages of this step were stored by the forward pass
            stored = {}
            self._restore_substeps(checkpoint_slot, 'x')
        else:
            stored = self._recompute_substeps(xn, tnp1, dt)

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
                        # this is substep lt-l of the recompute (start state xks[lt-l-1])
                        if checkpoint_slot is not None:
                            # restore the stored stages and store the stage adjoints and lambda
                            slots = {'checkpoint_slot': (checkpoint_slot, lt - l)}
                        else:
                            slots = {'stages_slot': lt - l if stored.get(lt - l) else None}
                        _delta_lambda_rhs, _delta_grad_rhs = time_integrator.take_adjoint_step(self.delta_grad_coeff, self.delta_lambda, self.lambda_k, self.xks[lt-l-1][0], tnp1 - dt + (k + 1) * sub_dt, sub_dt, **slots)
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

    def _recompute_substeps(self, xn, tnp1, dt):
        """Recompute the substep states xks of the step ending at tnp1 from xn (within-step recompute).

        With reuse_stages, substep l's stage values are stored under slot l for its adjoint step
        (plan item A3). Returns {slot: stored} for the substeps whose stages were stored.
        """
        stored = {}
        with region("substep recompute"):
            self.xks[0][0][0].assign(xn[0])
            l = 1
            for i,time_integrator in enumerate(self.time_integrators):
                sub_dt = dt / self.subcycle_list[i]
                for k in range(self.subcycle_list[i]):
                    with region(self.profile_labels[i]):
                        time_integrator.reset_internal_vars()
                        time_integrator.take_forward_step(self.xks[l][0], self.xks[l][1], self.xks[l-1][0], tnp1 - dt + k * sub_dt, sub_dt)
                        if self.reuse_stages:
                            stored[l] = time_integrator.store_stages(l)
                    l = l +1
        return stored

    def _compute_substep_tangents(self, xn, wn, tn, dt, store=False, checkpoint_slot=None):
        """Recompute the substep states xks and tangents wks from x_n and w_n.

        Each sub-integrator advances the state and its tangent together with take_tangent_step,
        so xks matches what take_forward_step computes. With store (and reuse_stages), substep
        l's forward and tangent stage values are stored under slot l for its second-order adjoint
        step. Returns {slot: stored} for the substeps whose stages were stored.

        With checkpoint_slot (plan items A1/A2), the forward stages stored by take_forward_step
        are restored instead of solved, and the tangent stages and substep tangents are stored
        under the slot for take_second_order_adjoint_step(checkpoint_slot=...).
        """
        stored = {}
        self.xks[0][0][0].assign(xn[0])
        self.wks[0][0][0].assign(wn[0])
        if checkpoint_slot is not None:
            self._store_substep(checkpoint_slot, 0, 'w', self.wks[0][0][0])
        l = 1
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
            for k in range(self.subcycle_list[i]):
                with region(self.profile_labels[i]):
                    time_integrator.reset_internal_vars()
                    sub_slot = None if checkpoint_slot is None else (checkpoint_slot, l)
                    time_integrator.take_tangent_step(self.wks[l][0], self.wks[l-1][0], self.xks[l-1][0], tn + k * sub_dt, sub_dt, xnp1=self.xks[l][0],
                                                      checkpoint_slot=sub_slot)
                    if checkpoint_slot is not None:
                        self._store_substep(checkpoint_slot, l, 'w', self.wks[l][0][0])
                    elif store and self.reuse_stages:
                        stored[l] = time_integrator.store_stages(l, tangents=True)
                l = l + 1
        return stored

    def take_tangent_step(self, wnp1, wn, xn, tn, dt, xnp1=None, checkpoint_slot=None):
        """Propagate a tangent-linear perturbation through one split step.

        Same interface as GeneralRK.take_tangent_step: the tangent is composed through the
        substeps of every sub-integrator, and x_{n+1} is also written to xnp1 if given. With
        checkpoint_slot, the stored forward stages of the step are restored instead of solved
        and the tangents are stored (see _compute_substep_tangents).
        """
        self._compute_substep_tangents(xn, wn, tn, dt, checkpoint_slot=checkpoint_slot)
        wnp1[0].assign(self.wks[-1][0][0])
        if xnp1 is not None:
            xnp1[0].assign(self.xks[-1][0][0])

    def take_second_order_adjoint_step(self, delta_lambda, delta_lambda_dot, lambda_np1, lambda_dot_np1, xn, wn, tnp1, dt,
                                       checkpoint_slot=None):
        """Take one combined first-order and second-order adjoint split step.

        Same interface and return values as GeneralRK.take_second_order_adjoint_step. The substep
        states and tangents are recomputed forward (storing the stages with reuse_stages), then
        lambda and lambda_dot are carried backward through the substeps in reverse order,
        summing the dual increments.

        With checkpoint_slot (plan items A1/A2), the substep states, tangents, stages, tangent
        stages, stage adjoints and lambdas stored for this step by take_forward_step,
        take_adjoint_step and take_tangent_step are restored, so only the second-order stage
        adjoints and increments are solved. The first-order part is skipped as for
        GeneralRK's adjoint_slot: delta_lambda is the stored increment and the first-order
        increments are None.
        """
        self.lambda_k.assign(lambda_np1)
        self.lambda_dot_k.assign(lambda_dot_np1)
        delta_lambda.assign(0)
        delta_lambda_dot.assign(0)

        delta_lambda_rhs = None
        delta_grad_rhs = None
        delta_lambda_dot_rhs = None
        delta_hess_coeff_rhs = None

        if checkpoint_slot is not None:
            # trajectory checkpoint: everything first-order and tangent was stored for this step
            stored = {}
            self._restore_substeps(checkpoint_slot, 'x')
            self._restore_substeps(checkpoint_slot, 'w')
        else:
            # recompute the substep states and tangents from (xn, wn)
            with region("substep recompute"):
                stored = self._compute_substep_tangents(xn, wn, tnp1 - dt, dt, store=True)

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
                        # substep lt-l of the recompute; restore its stored stages if available
                        if checkpoint_slot is not None:
                            slots = {'checkpoint_slot': (checkpoint_slot, lt - l)}
                        else:
                            slots = {'stages_slot': lt - l if stored.get(lt - l) else None}
                        increments = time_integrator.take_second_order_adjoint_step(self.delta_lambda, self.delta_lambda_dot,
                            self.lambda_k, self.lambda_dot_k, self.xks[lt-l-1][0], self.wks[lt-l-1][0], tnp1 - dt + (k + 1) * sub_dt, sub_dt,
                            **slots)
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
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn + dt/2.0)
#         self.pre_stepB_solvers()
#         self.stepB_solver.solve()
#         self.xn.assign(self.xn + self.dt * self.F)
#
#         self.tn = self.tn + dt
