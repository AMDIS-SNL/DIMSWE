from firedrake import assemble, inner, norm, LinearVariationalProblem, LinearVariationalSolver, Constant, derivative
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
    xns[0][0].assign(x0[0])
#DO WE NEED TO DO SOMETHING LIKE THIS?
#    xns[0][1].assign(x0[1])
    tns[0].assign(t0)
    for n in range(nsteps):
#WHY DO WE NEED TO DO?
#THERE IS SOME SORT OF DATA BEING CARRIED BETWEEN STEPS...
        timestepper.reset_internal_vars()
        timestepper.take_forward_step(xns[n+1], xn_subs[n+1], xns[n], tns[n], dt)
        tns[n+1].assign(tns[n] + dt)

def create_tangent_states(model, nsteps):
    """Create storage for a tangent-linear trajectory w_0..w_nsteps.

    Each entry is a one-element list [w_n] holding an x-space Function, matching the [x, aux] list
    convention of create_states for the entries take_tangent_step reads (only index 0). Tangents of
    the aux variables are never stored, since the aux variables are recomputed in every stage.
    """
    wns = []
    for i in range(nsteps+1):
        w, _, _ = model.get_x_var('w'+str(i))
        wns.append([w])
    return wns

def compute_states_and_tangents(xns, tns, wns, timestepper, nsteps, dt, x0, w0, t0):
    """Integrate the state and its tangent-linear perturbation together, storing every step.

    x0 is the [x, aux] start state as for compute_states, w0 the x-space start tangent (a
    Function, or None for zero, as for coefficient-only directions), and t0 the start time. The
    coefficient direction must already be set on the timestepper with set_numpy_coeff_direction.
    take_tangent_step also writes the forward update, so the states cost no extra forward solves.
    """
    xns[0][0].assign(x0[0])
    if w0 is None:
        wns[0][0].assign(0)
    else:
        wns[0][0].assign(w0)
    tns[0].assign(t0)
    for n in range(nsteps):
        # same reset per step as compute_states, so the stage solvers see the same initial guesses
        timestepper.reset_internal_vars()
        timestepper.take_tangent_step(wns[n+1], wns[n], xns[n], tns[n], dt, xnp1=xns[n+1])
        tns[n+1].assign(tns[n] + dt)

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
    """Base class for objectives that compare model states against data at observation steps.

    An objective is made of data blocks. Each block is integrated from its own start state for
    nsteps steps, and the model state is compared with data at the steps in
    get_observation_steps(). Subclasses supply the data and the UFL forms used for the loss and
    for the adjoint source term.
    """

    def get_observation_steps(self):
        """Return the sorted step indices (within a block, in 1..nsteps) where data is compared."""
        raise NotImplementedError

    def get_observation(self, block_id, obs_index):
        """Return the data state ([x, aux] list) for observation obs_index of block block_id."""
        raise NotImplementedError

    def build_forms(self, mismatch, mismatch_split, xtest, xtest_subs, dx):
        """Return (loss_form, source_form) for a mismatch Function (model state minus data).

        loss_form is the 0-form contribution of one observation to the loss, and source_form is
        its derivative with respect to the state, as a 1-form in the test function xtest.
        """
        raise NotImplementedError

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

    def evaluate(self, block_id, soln, tns):
#THIS INCLUDES AUX VARIABLES FOR DIRK AND IRK CASES- PROBABLY TRY TO AVOID THOSE IF POSSIBLE?
        residual = soln[0] - self.data_blocks[block_id][1][0]
        return assemble(0.5*inner(residual,residual)*self.dx)

    def get_observation_steps(self):
        """Only the final state of each block is compared."""
        return [self.nsteps]

    def get_observation(self, block_id, obs_index):
        """The data for the final state of a block is the second entry of the block."""
        return self.data_blocks[block_id][1]

    def build_forms(self, mismatch, mismatch_split, xtest, xtest_subs, dx):
        """Unweighted 0.5*||mismatch||^2 loss and its state derivative."""
        return 0.5*inner(mismatch, mismatch)*dx, inner(xtest, mismatch)*dx


class ObservationL2Objective(_Objective):
    """Weighted L2 misfit at an arbitrary set of observation steps within each block.

    This covers the three loss types used by the optimization workflow:
    - final state only: one block, obs_steps = [nsteps]
    - windowed: several blocks restarted from data, obs_steps = [nsteps]
    - trajectory: one block, obs_steps = [K, 2K, ..., nsteps]

    data_blocks[i][0] is the start state of block i, and data_blocks[i][k+1] is the data at step
    obs_steps[k]. Each entry is an [x, aux] list as returned by get_full_var(split_x_and_aux=True).
    t_blocks[i][0] is the start time of block i.

    The loss for one observation is loss_scale * 0.5 * sum_var w_var * ||x_var - d_var||^2. When
    var_weights is None, the unweighted mixed inner product is used instead.
    """

    def __init__(self, data_blocks, t_blocks, nsteps, obs_steps, dx, var_weights=None, loss_scale=1.0):
        self.data_blocks = data_blocks
        self.t_blocks = t_blocks
        self.num_data_blocks = len(self.data_blocks)
        self.nsteps = nsteps
        self.obs_steps = sorted(obs_steps)
        self.dx = dx
        self.var_weights = var_weights
        self.loss_scale = loss_scale

        if len(self.obs_steps) == 0:
            raise ValueError('at least one observation step is required')
        if len(set(self.obs_steps)) != len(self.obs_steps):
            raise ValueError('observation steps must be unique')
        if self.obs_steps[0] < 1 or self.obs_steps[-1] > nsteps:
            raise ValueError('observation steps must lie in 1..nsteps')
        for i, block in enumerate(self.data_blocks):
            if len(block) != len(self.obs_steps) + 1:
                raise ValueError('data block ' + str(i) + ' has ' + str(len(block)) + ' states, expected start state plus ' + str(len(self.obs_steps)) + ' observations')

    def get_observation_steps(self):
        return self.obs_steps

    def get_observation(self, block_id, obs_index):
        return self.data_blocks[block_id][obs_index+1]

    def build_forms(self, mismatch, mismatch_split, xtest, xtest_subs, dx):
        """Build the (optionally per-variable weighted) loss and adjoint source forms."""
        scale = Constant(self.loss_scale)
        if self.var_weights is None:
            return scale*0.5*inner(mismatch, mismatch)*dx, scale*inner(xtest, mismatch)*dx
        loss_form = 0
        source_form = 0
        for var, residual in mismatch_split.items():
            weight = Constant(self.var_weights.get(var, 1.0))
            loss_form = loss_form + scale*weight*0.5*inner(residual, residual)*dx
            source_form = source_form + scale*weight*inner(xtest_subs[var], residual)*dx
        return loss_form, source_form


class ReducedProblem():
    """Wrap a full-space objective/gradient pair into the reduced, scaled space seen by scipy.

    The optimizer variables are z = x[free] / scale[free]; entries of x outside free_mask stay at
    their initial values. fun and jac are memoized on the last evaluated point, so evaluating the
    initial point (or the point reported to a callback) does not trigger extra model runs.

    If full_hessp(x_full, v_full) is given, hessp(z, p) provides reduced Hessian-vector products
    for second-order scipy methods. With x = P S z + x_fixed (P scatters free entries, S the free
    scaling), the reduced Hessian is S P^T H_full P S, so hessp maps p to the full direction
    v_full = P S p (zero on fixed entries) and returns (H_full v_full)[free] * scale[free]. HVP
    calls are counted in nhvp, separately from nfev and njev, and are not memoized, since
    scipy calls hessp with a new direction each time.
    """

    def __init__(self, full_fun, full_jac, x0_full, free_mask=None, scaling_factors=None, bounds=None, full_hessp=None):
        self.full_fun = full_fun
        self.full_jac = full_jac
        self.full_hessp = full_hessp
        self.x0_full = np.array(x0_full, dtype=float).copy()
        n = self.x0_full.size
        self.free_mask = np.ones(n, dtype=bool) if free_mask is None else np.array(free_mask, dtype=bool)
        self.scale = np.ones(n) if scaling_factors is None else np.array(scaling_factors, dtype=float)
        if self.free_mask.shape != (n,) or self.scale.shape != (n,):
            raise ValueError('free_mask and scaling_factors must match the size of the initial guess')
        if not np.any(self.free_mask):
            raise ValueError('at least one parameter must be free')
        if np.any(self.scale[self.free_mask] <= 0):
            raise ValueError('scaling factors must be positive')
        self.bounds = bounds
        self.z0 = self.to_reduced(self.x0_full)
        self.nfev = 0
        self.njev = 0
        self.nhvp = 0
        self._fun_cache = None
        self._jac_cache = None

    def to_full(self, z):
        """Map reduced, scaled variables back to the full parameter vector."""
        x = self.x0_full.copy()
        x[self.free_mask] = np.asarray(z) * self.scale[self.free_mask]
        return x

    def to_reduced(self, x):
        """Map a full parameter vector to reduced, scaled variables."""
        return np.asarray(x)[self.free_mask] / self.scale[self.free_mask]

    def fun(self, z):
        """Objective in reduced variables (memoized on the last point)."""
        if self._fun_cache is not None and np.array_equal(self._fun_cache[0], z):
            return self._fun_cache[1]
        value = float(self.full_fun(self.to_full(z)))
        self.nfev += 1
        self._fun_cache = (np.array(z, copy=True), value)
        return value

    def jac(self, z):
        """Gradient in reduced variables: chain rule through the scaling (memoized)."""
        if self._jac_cache is not None and np.array_equal(self._jac_cache[0], z):
            return self._jac_cache[1]
        g_full = np.asarray(self.full_jac(self.to_full(z)))
        g = g_full[self.free_mask] * self.scale[self.free_mask]
        self.njev += 1
        self._jac_cache = (np.array(z, copy=True), g.copy())
        return g

    def has_hessp(self):
        """Return True if Hessian-vector products are available."""
        return self.full_hessp is not None

    def direction_to_full(self, p):
        """Map a reduced direction p to the full space: v[free] = p * scale[free], zero elsewhere.

        Unlike to_full, this is the linear part of the map only, since fixed entries do not move.
        """
        v = np.zeros_like(self.x0_full)
        v[self.free_mask] = np.asarray(p, dtype=float) * self.scale[self.free_mask]
        return v

    def hessp(self, z, p):
        """Reduced Hessian-vector product H_z p = (H_full(x) v_full)[free] * scale[free].

        x = to_full(z) and v_full = direction_to_full(p). This is the second-order counterpart of
        jac's chain rule through the scaling; the scaling enters twice, once for the direction
        and once for the result, so H_z = S P^T H_full P S is symmetric.
        """
        if self.full_hessp is None:
            raise ValueError('this ReducedProblem has no Hessian-vector product (full_hessp is None)')
        hv_full = np.asarray(self.full_hessp(self.to_full(z), self.direction_to_full(p)), dtype=float)
        if hv_full.shape != self.x0_full.shape:
            raise ValueError('full_hessp returned shape ' + str(hv_full.shape) + ', expected ' + str(self.x0_full.shape))
        self.nhvp += 1
        return hv_full[self.free_mask] * self.scale[self.free_mask]

    def cached_fun(self, z):
        """Return the cached objective at z, or None if z was not the last evaluated point."""
        if self._fun_cache is not None and np.array_equal(self._fun_cache[0], z):
            return self._fun_cache[1]
        return None

    def cached_jac(self, z):
        """Return the cached gradient at z, or None if z was not the last evaluated point."""
        if self._jac_cache is not None and np.array_equal(self._jac_cache[0], z):
            return self._jac_cache[1]
        return None

    def scipy_bounds(self):
        """Return scipy Bounds in reduced variables, or None if unbounded."""
        if self.bounds is None:
            return None
        lb, ub = self.bounds
        lb = np.array(lb, dtype=float)[self.free_mask] / self.scale[self.free_mask]
        ub = np.array(ub, dtype=float)[self.free_mask] / self.scale[self.free_mask]
        return sp.optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)


class ParametricInitialCondition():
    """Differentiable map from analytic initial-condition parameters to the discrete initial state.

    The named entries of parameters['initial-conditions'] are replaced by R-space Functions, and
    the initial condition is rebuilt from these symbolic values, so its expressions depend on the
    parameters through UFL. forward(p) projects the expressions onto the model x space exactly as
    model.initialize does. The tangent dx0/dp_j is the projection of the Gateaux derivative of the
    expressions (computed with ufl.derivative), since projection is linear, and backward(g) applies
    the transpose of the Jacobian to a dual gradient g = dJ/dx0, giving dJ/dp.

    Only the prognostic x variables are differentiated. Quantities that are interpolated once at
    model initialization (coriolis, bottom topography) are held fixed.
    """

    def __init__(self, model, parameters, param_names):
        from copy import deepcopy
        from firedrake import Function, FunctionSpace
        import ufl
        from ufl.algorithms import expand_derivatives
        from ufl.constantvalue import Zero

        self.model = model
        self.param_names = list(param_names)
        if len(self.param_names) == 0:
            raise ValueError('at least one initial-condition parameter is required')
        ic_parameters = parameters['initial-conditions']
        for name in self.param_names:
            if name not in ic_parameters:
                raise KeyError("unknown initial-condition parameter 'initial-conditions." + name + "'")
            if isinstance(ic_parameters[name], bool) or not isinstance(ic_parameters[name], (int, float)):
                raise ValueError("initial-condition parameter '" + name + "' must be numeric")

        self.comm = model.mesh.comm
        R = FunctionSpace(model.mesh, 'R', 0)
        self.param_funcs = [Function(R, name='icparam_' + name) for name in self.param_names]
        self.directions = [Function(R, name='icparam_dir_' + name).assign(1.0) for name in self.param_names]

        # rebuild the model's initial condition with symbolic parameter values
        symbolic_parameters = deepcopy(parameters)
        for name, func in zip(self.param_names, self.param_funcs):
            symbolic_parameters['initial-conditions'][name] = func
        varexpr = model.get_initial_expressions(symbolic_parameters, model.get_t_var())

        self.x_var_list = list(model.get_x_var_list())
        self.exprs = {}
        for var in self.x_var_list:
            expr = varexpr.get(var, 0)
            if not (isinstance(expr, (int, float)) and expr == 0):
                self.exprs[var] = expr

        # symbolic tangent expressions d(expr)/dp_j, None where an expression does not depend on p_j
        self.tangent_exprs = []
        for func, direction in zip(self.param_funcs, self.directions):
            tangents = {}
            for var, expr in self.exprs.items():
                dexpr = expand_derivatives(ufl.derivative(expr, func, direction))
                tangents[var] = None if isinstance(dexpr, Zero) else dexpr
            self.tangent_exprs.append(tangents)
        for name, tangents in zip(self.param_names, self.tangent_exprs):
            if all(dexpr is None for dexpr in tangents.values()):
                raise ValueError("initial condition '" + parameters['initial-conditions']['name'] + "' does not depend on parameter '" + name + "'")

        # direct solves keep the projections (and hence the loss) free of iterative-solver noise,
        # which would otherwise pollute Taylor tests and line searches
        self.projection_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.x0, self.x0_sub, _ = model.get_x_var('icparam_x0')
        self.tangent_vars = [model.get_x_var('icparam_tangent_' + name) for name in self.param_names]
        # second derivatives d^2 x0 / dp_j dp_k, built on first use by second_tangents()
        self._second_tangent_exprs = None
        self._second_tangent_var = None

    def get_initial_values(self, parameters):
        """Return the parameter values in parameters['initial-conditions'] as an array."""
        return np.array([float(parameters['initial-conditions'][name]) for name in self.param_names])

    def set_param_values(self, params):
        """Assign parameter values to the R-space Functions."""
        for func, value in zip(self.param_funcs, params):
            func.assign(float(value))

    def forward(self, params):
        """Return the rank-local flattened discrete initial state for parameter values params."""
        self.set_param_values(params)
        self.x0.assign(0)
        for var, expr in self.exprs.items():
            self.x0_sub[var].project(expr, solver_parameters=self.projection_parameters)
        return create_flattened_numpy_arr_from_mixed_function(self.x0).copy()

    def tangents(self):
        """Return rank-local flattened tangents dx0/dp_j at the current parameter values."""
        tangent_arrs = []
        for (tangent, tangent_sub, _), dexprs in zip(self.tangent_vars, self.tangent_exprs):
            tangent.assign(0)
            for var, dexpr in dexprs.items():
                if dexpr is not None:
                    tangent_sub[var].project(dexpr, solver_parameters=self.projection_parameters)
            tangent_arrs.append(create_flattened_numpy_arr_from_mixed_function(tangent).copy())
        return tangent_arrs

    def backward(self, grad_x0):
        """Map a rank-local dual gradient dJ/dx0 to the global gradient dJ/dp.

        Must be called after forward() at the same parameter values. The pairing of a cofunction
        with a function is a sum over owned dofs, so it is summed across MPI ranks.
        """
        grad_x0 = np.ravel(grad_x0)
        local = np.array([np.dot(grad_x0, tangent) for tangent in self.tangents()])
        return self.comm.allreduce(local)

    def tangent_action(self, direction):
        """Return the rank-local flattened tangent dx0/dp . direction = sum_j direction_j dx0/dp_j.

        Must be called after forward() at the parameter values of interest. direction must be
        identical on every MPI rank, since the parameters are replicated.
        """
        direction = np.ravel(np.asarray(direction, dtype=float))
        result = None
        for value, tangent in zip(direction, self.tangents()):
            result = value * tangent if result is None else result + value * tangent
        return result

    def _build_second_tangent_exprs(self):
        """Build the symbolic second derivatives d^2(expr)/dp_j dp_k for j <= k (None where zero)."""
        import ufl
        from ufl.algorithms import expand_derivatives
        from ufl.constantvalue import Zero

        nparams = len(self.param_names)
        self._second_tangent_exprs = [[None] * nparams for _ in range(nparams)]
        for j in range(nparams):
            for k in range(j, nparams):
                exprs = {}
                for var, dexpr in self.tangent_exprs[j].items():
                    if dexpr is None:
                        exprs[var] = None
                        continue
                    d2expr = expand_derivatives(ufl.derivative(dexpr, self.param_funcs[k], self.directions[k]))
                    exprs[var] = None if isinstance(d2expr, Zero) else d2expr
                self._second_tangent_exprs[j][k] = exprs
        self._second_tangent_var = self.model.get_x_var('icparam_second_tangent')

    def second_tangents(self):
        """Return the rank-local flattened second derivatives d^2 x0 / dp_j dp_k at the current values.

        The result is a symmetric nparams x nparams nested list of arrays, with None for entries
        that are identically zero. As for tangents(), projection is linear, so each entry is the
        projection of the second Gateaux derivative of the initial-condition expressions. Only
        the upper triangle is projected; the lower triangle refers to the same arrays.
        """
        if self._second_tangent_exprs is None:
            self._build_second_tangent_exprs()
        nparams = len(self.param_names)
        second, second_sub, _ = self._second_tangent_var
        result = [[None] * nparams for _ in range(nparams)]
        for j in range(nparams):
            for k in range(j, nparams):
                exprs = self._second_tangent_exprs[j][k]
                if all(d2expr is None for d2expr in exprs.values()):
                    continue
                second.assign(0)
                for var, d2expr in exprs.items():
                    if d2expr is not None:
                        second_sub[var].project(d2expr, solver_parameters=self.projection_parameters)
                arr = create_flattened_numpy_arr_from_mixed_function(second).copy()
                result[j][k] = arr
                result[k][j] = arr
        return result

    def second_order_backward(self, grad_x0, direction):
        """Return the global curvature term of the reduced Hessian-vector product in p.

        For J(p) = Jhat(x0(p)) the Hessian action is
            H_p v = T^T (H_x0 (T v)) + [ sum_j v_j (d^2 x0/dp_j dp_k) . dJ/dx0 ]_k,
        with T = dx0/dp. backward() gives the first term from H_x0 (T v); this returns the second
        one, from the rank-local dual gradient grad_x0 = dJ/dx0 and the direction v. Must be
        called after forward() at the same parameter values; summed across MPI ranks.
        """
        grad_x0 = np.ravel(grad_x0)
        direction = np.ravel(np.asarray(direction, dtype=float))
        nparams = len(self.param_names)
        second = self.second_tangents()
        local = np.zeros(nparams)
        for k in range(nparams):
            for j in range(nparams):
                if second[j][k] is not None and direction[j] != 0:
                    local[k] += direction[j] * np.dot(grad_x0, second[j][k])
        return self.comm.allreduce(local)


# scipy.optimize.minimize methods that consume Hessian-vector products (hessp)
HESSP_METHODS = ('NEWTON-CG', 'TRUST-NCG', 'TRUST-KRYLOV', 'TRUST-CONSTR')
# hessp methods that cannot run without second-order information (scipy raises otherwise);
# Newton-CG falls back to finite differences of jac, trust-constr to a quasi-Newton update
HESSP_REQUIRED_METHODS = ('TRUST-NCG', 'TRUST-KRYLOV')
# methods that honour bounds; scipy only warns for the others and then ignores the bounds
BOUNDED_METHODS = ('L-BFGS-B', 'TNC', 'SLSQP', 'TRUST-CONSTR', 'NELDER-MEAD', 'POWELL', 'COBYLA', 'COBYQA')
# methods for which scipy requires an explicit gradient (CG, BFGS, ... fall back to finite differences)
GRADIENT_METHODS = ('NEWTON-CG', 'TRUST-NCG', 'TRUST-KRYLOV', 'DOGLEG', 'TRUST-EXACT')


def method_uses_hessp(method):
    """Return True if a scipy.optimize.minimize method consumes Hessian-vector products."""
    return method.upper() in HESSP_METHODS


def method_supports_bounds(method):
    """Return True if a scipy.optimize.minimize method honours bounds."""
    return method.upper() in BOUNDED_METHODS


def resolve_hessp_use(method, use_hessp, use_jacobian, has_hessp, has_bounds):
    """Validate a method choice and decide whether Hessian-vector products are passed to scipy.

    use_hessp is True (require HVPs), False (never pass them) or None (pass them whenever the
    method consumes them and the problem provides them). has_hessp says whether the problem can
    compute HVPs, and has_bounds whether bounds would be passed to scipy. Raises ValueError for
    combinations scipy would reject or silently mishandle:
    - HVPs requested for a method that does not use them, or for a problem without them,
    - HVPs (or a gradient-based method) without a gradient,
    - trust-ncg / trust-krylov without HVPs,
    - bounds with an HVP-enabled method that ignores them (Newton-CG, trust-ncg, trust-krylov).
    First-order methods keep their previous behaviour: scipy warns about, and ignores, bounds it
    cannot handle (for example CG or BFGS), so existing configurations keep running unchanged.
    Returns True if hessp should be passed to scipy.
    """
    name = method.upper()
    if use_hessp and not method_uses_hessp(method):
        raise ValueError('method ' + method + ' does not use Hessian-vector products; HVP-enabled methods are ' + ', '.join(HESSP_METHODS))
    if use_hessp and not has_hessp:
        raise ValueError('Hessian-vector products were requested, but this problem does not provide them')
    if not use_jacobian and (use_hessp or name in GRADIENT_METHODS):
        raise ValueError('method ' + method + ' needs the gradient; do not set use_jacobian=False')
    hessp = method_uses_hessp(method) and has_hessp and use_hessp is not False
    if name in HESSP_REQUIRED_METHODS and not hessp:
        raise ValueError('method ' + method + ' requires Hessian-vector products')
    if has_bounds and method_uses_hessp(method) and not method_supports_bounds(method):
        raise ValueError('method ' + method + ' cannot handle bounds (scipy would ignore them); use a bounded method such as '
            'L-BFGS-B or, for Hessian-vector products, trust-constr, or optimize without bounds (use_bounds=False)')
    return hessp


def scipy_options(method, maxiter, gtol, ftol, disp, xtol=None):
    """Return the scipy.optimize.minimize options understood by a method.

    Passing options a method does not understand makes scipy warn, so only the relevant tolerances
    and iteration limits are included. xtol is only used by Newton-CG (its only convergence
    tolerance; scipy's default applies if None) and trust-constr.
    """
    method = method.upper()
    if method == 'L-BFGS-B':
        return {'disp': disp, 'maxiter': maxiter, 'maxfun': maxiter, 'gtol': gtol, 'ftol': ftol}
    if method == 'TNC':
        return {'disp': disp, 'maxfun': maxiter, 'gtol': gtol, 'ftol': ftol}
    if method in ('CG', 'BFGS', 'TRUST-NCG', 'TRUST-KRYLOV'):
        return {'disp': disp, 'maxiter': maxiter, 'gtol': gtol}
    if method == 'NEWTON-CG':
        options = {'disp': disp, 'maxiter': maxiter}
        if xtol is not None:
            options['xtol'] = xtol
        return options
    if method == 'TRUST-CONSTR':
        options = {'disp': disp, 'maxiter': maxiter, 'gtol': gtol}
        if xtol is not None:
            options['xtol'] = xtol
        return options
    if method == 'SLSQP':
        return {'disp': disp, 'maxiter': maxiter, 'ftol': ftol}
    return {'disp': disp, 'maxiter': maxiter}


def taylor_test(problem, direction, eps_list=None):
    """Taylor remainder test of a ReducedProblem gradient at its initial point.

    Returns (eps, remainders, rates), where remainder = |f(z0 + eps*d) - f(z0) - eps*g.d|. A
    correct gradient gives rates close to 2. direction must be in reduced variables and must be
    identical on every MPI rank when the parameters are replicated (coefficients, ic-params).
    """
    if eps_list is None:
        eps_list = list(1e-2 * 0.5 ** np.arange(0, 6))
    z0 = problem.z0.copy()
    f0 = problem.fun(z0)
    g0 = problem.jac(z0)
    dg = float(np.dot(g0, direction))
    remainders = []
    for eps in eps_list:
        fp = problem.fun(z0 + eps*np.asarray(direction))
        remainders.append(abs(fp - f0 - eps*dg))
    rates = []
    for i in range(len(remainders)-1):
        if remainders[i] > 0 and remainders[i+1] > 0:
            rates.append(float(np.log(remainders[i]/remainders[i+1]) / np.log(eps_list[i]/eps_list[i+1])))
        else:
            rates.append(float('nan'))
    # leave the memo pointing at the initial point so optimization starts without recomputation
    problem.fun(z0)
    return np.array(eps_list), np.array(remainders), np.array(rates)


class _ODEConstrainedOptimization():
    def __init__(self, model, timestepper, objective, dt):
        self.model = model
        self.timestepper = timestepper
        self.objective = objective
        self.verbose = True
#THIS IS A CHECKPOINTING/STORAGE CHOICE
        self.states, self.states_sub, self.tns = create_states(model, self.objective.nsteps)
        self.dt = dt
        self.lambda_np1, _, _ = self.model.get_x_var('lambda_np1')
        self.lambda_source, _, _ = self.model.get_x_var('lambda_source')
        self.delta_lambda, _, _ = self.model.get_x_var('delta_lambda')
        self.delta_grad_coeff, _, _ = self.model.get_coeff_var('delta_grad_coeff')
        self.grad_ic, _, _ = self.model.get_x_var('grad_ic')
        self.ic, _, _ = self.model.get_x_var('ic')

        # flattened arrays only hold rank-local dofs, so size them locally rather than globally
        self.local_x_size = create_flattened_numpy_arr_from_mixed_function(self.ic).size
        self.grad_coeffs = np.zeros(self.model.get_coeff_size())
        self.grad_ics = np.zeros((self.objective.num_data_blocks, self.local_x_size))

        self.mismatch, _, self.mismatch_split = self.model.get_x_var('mismatch')
        self.initial_mismatch = self.mismatch
        self.xtest, self.xtest_subs = self.model.get_x_test_vars()
        xtrial, _ = self.model.get_x_trial_vars()

        # loss and adjoint source forms for a single observation, built once from the objective
        self.loss_form, self.source_form = self.objective.build_forms(self.mismatch, self.mismatch_split, self.xtest, self.xtest_subs, self.model.spaces.dx)
        self.obs_steps = list(self.objective.get_observation_steps())
        self.obs_index = {step: k for k, step in enumerate(self.obs_steps)}
        # adjoint sources M^{-1} r_n for each observation, computed right after the forward pass
        self.obs_sources = [self.model.get_x_var('obs_source_' + str(step))[0] for step in self.obs_steps]

        a = inner(self.xtest, xtrial)* self.model.spaces.dx
        lambda_source_projector_problem = LinearVariationalProblem(a, self.source_form, self.lambda_source, constant_jacobian=True)
        self.lambda_np1_projector = LinearVariationalSolver(lambda_source_projector_problem,
            solver_parameters={ 'ksp_type': 'preonly', 'pc_type' : 'lu'}, #{ 'ksp_type': 'cg', 'pc_type' : 'jacobi',},
                options_prefix='opt-proj')

#        self.coeff_scaling_factors = self.model.get_coeff_scaling_factors()

    def reduced_problem(self, initial_guess, opt_type='coeffs', coeffs0=None, bounds=None, scaling_factors=None, free_mask=None, ic_map=None):
        """Build the ReducedProblem for an optimization type.

        opt_type is one of
        - 'coeffs': optimize model coefficients (default bounds from the model)
        - 'ics': optimize the full discrete initial state of every block
        - 'coeffs+ics': both of the above
        - 'ic-params': optimize parameters of the analytic initial condition through ic_map
          (a ParametricInitialCondition), using one data block
        bounds, if given, is a (lb, ub) pair of full-length arrays.
        """
        nblocks = self.objective.num_data_blocks
        if opt_type == 'coeffs':
            full_fun = lambda coeffs: self.obj(coeffs, None)
            full_jac = lambda coeffs: self.jac(coeffs, None)
            if bounds is None:
                bounds = self.model.get_coeff_bounds()
            #scaling_factors = self.coeff_scaling_factors
        elif opt_type == 'ics':
            full_fun = lambda ics: self.obj(None, np.reshape(ics, (nblocks, self.local_x_size)), coeffs0=coeffs0)
            full_jac = lambda ics: self.jac(None, np.reshape(ics, (nblocks, self.local_x_size)), coeffs0=coeffs0)
#FIX THESE PROBABLY...
            #bounds = self.model.get_ic_bounds() * self.objective.num_data_blocks
        elif opt_type == 'coeffs+ics':
            ncoeff = self.model.get_coeff_size()
            full_fun = lambda coeffs_plus_ic: self.obj(coeffs_plus_ic[:ncoeff], np.reshape(coeffs_plus_ic[ncoeff:], (nblocks, self.local_x_size)))
            full_jac = lambda coeffs_plus_ic: self.jac(coeffs_plus_ic[:ncoeff], np.reshape(coeffs_plus_ic[ncoeff:], (nblocks, self.local_x_size)))
            #bounds = self.model.get_coeff_bounds() + self.model.get_ic_bounds() * self.objective.num_data_blocks
        elif opt_type == 'ic-params':
            if ic_map is None:
                raise ValueError("opt_type 'ic-params' requires an ic_map")
            if nblocks != 1:
                raise ValueError("opt_type 'ic-params' requires exactly one data block")
            full_fun = lambda params: self.obj(None, ic_map.forward(params)[None, :], coeffs0=coeffs0)
            # chain rule: dJ/dp = (dx0/dp)^T dJ/dx0, with dx0/dp from UFL derivatives in ic_map
            full_jac = lambda params: ic_map.backward(self.jac(None, ic_map.forward(params)[None, :], coeffs0=coeffs0))
        else:
            raise ValueError('unknown opt_type ' + str(opt_type))
        # Hessian-vector products are always available; they cost nothing until scipy calls hessp
        full_hessp = self.full_hessp(opt_type, coeffs0=coeffs0, ic_map=ic_map) if hasattr(self, 'full_hessp') else None
        return ReducedProblem(full_fun, full_jac, initial_guess, free_mask=free_mask, scaling_factors=scaling_factors, bounds=bounds, full_hessp=full_hessp)

    def optimize(self, initial_guess, method='L-BFGS-B', opt_type='coeffs', use_jacobian=True, coeffs0=None, gtol=1e-5, ftol=1e-5, #cg, bfgs
            maxiter=200000, bounds=None, scaling_factors=None, free_mask=None, ic_map=None, callback=None, return_result=False, problem=None,
            use_hessp=None, use_bounds=True, xtol=None):
        """Run scipy.optimize.minimize and return the optimal full parameter vector.

        callback, if given, is called as callback(iteration, x_full, fun_value, grad_norm) for the
        initial point (iteration 0) and after every optimizer iteration. fun_value/grad_norm are
        None when not available without extra model evaluations. grad_norm is the 2-norm of the
        gradient in the reduced, scaled optimizer variables. If return_result is True, returns
        (x_full, scipy OptimizeResult). A prebuilt ReducedProblem can be passed as problem.

        Second-order methods: for methods that consume Hessian-vector products (HESSP_METHODS:
        Newton-CG, trust-ncg, trust-krylov, trust-constr), problem.hessp is passed to scipy by
        default (use_hessp=None). use_hessp=True requires HVPs and use_hessp=False never passes
        them. Of these methods only trust-constr honours bounds. Since scipy would ignore bounds
        for the others, bounds with Newton-CG, trust-ncg or trust-krylov raise a ValueError;
        use_bounds=False optimizes without bounds instead (note that 'coeffs' uses the model's
        coefficient bounds by default). xtol is the Newton-CG / trust-constr step tolerance.
        Methods that do not use HVPs behave exactly as before.

        The returned OptimizeResult additionally carries the problem's own evaluation counters
        problem_nfev, problem_njev and problem_nhvp (actual model evaluations, after memoization),
        and hessp_used, whether Hessian-vector products were passed to scipy.
        """
        if problem is None:
            problem = self.reduced_problem(initial_guess, opt_type=opt_type, coeffs0=coeffs0, bounds=bounds,
                scaling_factors=scaling_factors, free_mask=free_mask, ic_map=ic_map)

        scipy_bounds = problem.scipy_bounds() if use_bounds else None
        pass_hessp = resolve_hessp_use(method, use_hessp, use_jacobian, problem.has_hessp(), scipy_bounds is not None)

        iteration = [0]
        def report(zk):
            if callback is None:
                return
            f = problem.cached_fun(zk)
            g = problem.cached_jac(zk)
            gnorm = None if g is None else float(np.linalg.norm(g))
            callback(iteration[0], problem.to_full(zk), f, gnorm)

        # trust-constr calls legacy callbacks as callback(xk, state), the other methods as callback(xk)
        def scipy_callback(zk, *args):
            iteration[0] += 1
            report(zk)

        # evaluate the starting point up front (memoized, so this costs nothing extra)
        if callback is not None:
            problem.fun(problem.z0)
            if use_jacobian:
                problem.jac(problem.z0)
            report(problem.z0)

        options = scipy_options(method, maxiter, gtol, ftol, self.verbose, xtol=xtol)
        kwargs = {}
        if use_jacobian:
            kwargs['jac'] = problem.jac
        if pass_hessp:
            kwargs['hessp'] = problem.hessp
        res = sp.optimize.minimize(problem.fun, problem.z0,
            method=method, bounds=scipy_bounds,
            options=options, callback=scipy_callback, **kwargs)

        # scipy's counters differ between methods (and some lack njev/nhev), so also record the
        # problem's counts of actual model evaluations
        res.problem_nfev = problem.nfev
        res.problem_njev = problem.njev
        res.problem_nhvp = problem.nhvp
        # whether hessp was actually passed to scipy (use_hessp=None resolves per method)
        res.hessp_used = bool(pass_hessp)
        if self.verbose:
            print('optimizer success', res.success, res.status, res.message)
            print('optimizer nits', getattr(res, 'nit', None))
            print('optimizer num func evals', problem.nfev)
            if use_jacobian:
                print('optimizer num jac evals', problem.njev)
            if pass_hessp:
                print('optimizer num hessp evals', problem.nhvp)
        x_opt = problem.to_full(res.x)
        if return_result:
            return x_opt, res
        return x_opt

    def _set_coeffs(self, coeffs_arr, coeffs0):
        """Set the timestepper coefficients from a numpy array, or from coeffs0 if it is None."""
        if coeffs_arr is None:
            self.timestepper.set_coeff(coeffs0)
        else:
            #self.timestepper.set_numpy_coeff(coeffs_arr * self.coeff_scaling_factors)
            self.timestepper.set_numpy_coeff(coeffs_arr)

    def _compute_block_states(self, block_id, ics_arr):
        """Integrate one block from its data start state, or from ics_arr if given."""
        if ics_arr is None:
            compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, self.objective.data_blocks[block_id][0], self.objective.t_blocks[block_id][0])
        else:
            set_mixed_function_from_flattened_array(self.ic, ics_arr[block_id,:])
            compute_states(self.states, self.states_sub, self.tns, self.model, self.timestepper, self.objective.nsteps, self.dt, [self.ic,], self.objective.t_blocks[block_id][0])

    def _assign_mismatch(self, block_id, step):
        """Set the mismatch Function to the model state minus data at an observation step."""
        data = self.objective.get_observation(block_id, self.obs_index[step])
        self.mismatch.assign(self.states[step][0] - data[0])

    def obj(self, coeffs_arr, ics_arr, coeffs0=None):
        self._set_coeffs(coeffs_arr, coeffs0)

        l2loss = 0.0
        for i in range(self.objective.num_data_blocks):
#THIS IS REQUIRED TO GET REPEATABILITY FOR OPT
#WHY?????
            self.timestepper.reset_internal_vars()
            self._compute_block_states(i, ics_arr)
            for step in self.obs_steps:
                self._assign_mismatch(i, step)
                l2loss += assemble(self.loss_form)
        return l2loss

#EVENTUALLY ADD A CLASS WITH REGULARIZATION FOR CONSTRAINTS IE AUGMENTED LAGRANGIAN?
class Lagrangian_ODEConstrainedOptimization(_ODEConstrainedOptimization):
    def jac(self, coeffs_arr, ics_arr, coeffs0=None):
        if self.model.has_coeff():
            self.grad_coeffs[:] = 0.0
            self.delta_grad_coeff.assign(0)
        self.grad_ics[:] = 0.0
        self.grad_ic.assign(0)
        self.delta_lambda.assign(0)

        self._set_coeffs(coeffs_arr, coeffs0)

        for i in range(self.objective.num_data_blocks):
            self.lambda_np1.assign(0)
            self.ic.assign(0)

#THIS IS REQUIRED TO GET REPEATABILITY FOR JAC
#WHY?????
            self.timestepper.reset_internal_vars()

            #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE
            self._compute_block_states(i, ics_arr)

            # each observation contributes a dual source r_n (the loss derivative with respect to x_n)
            # to the gradient, and M^{-1} r_n to the adjoint at step n. These are all computed before
            # the backward sweep, since the sweep is not guaranteed to leave self.states untouched.
            for k, step in enumerate(self.obs_steps):
                self._assign_mismatch(i, step)
                self.lambda_np1_projector.solve()
                self.obs_sources[k].assign(self.lambda_source)
                self.grad_ics[i, :] = self.grad_ics[i, :] + create_flattened_numpy_arr_from_mixed_function(assemble(self.source_form))

            for n in range(self.objective.nsteps,0,-1):

                # lambda_n += M^{-1} r_n at observation steps
                if n in self.obs_index:
                    self.lambda_np1.assign(self.lambda_np1 + self.obs_sources[self.obs_index[n]])

                #IF THIS IS NOT DONE THEN THE STATES[N] CHANGES!
                #WHY??
                #BUT THIS ONLY FIXES THE LAST ONE, THE OTHERS ARE STILL BROKEN
                self.timestepper.reset_internal_vars()

                #take a forward step to populate internal variables within timestepper
                #THIS IS A CHOICE/TYPE OF CHECKPOINT + RECOMPUTE\
                #self.timestepper.take_forward_step(self.states[n], self.states_sub[n], self.states[n-1], self.tns[n-1], self.dt)

                delta_lambda_rhs, delta_grad_rhs = self.timestepper.take_adjoint_step(self.delta_grad_coeff, self.delta_lambda, self.lambda_np1, self.states[n-1], self.tns[n], self.dt)
                self.lambda_np1.assign(self.lambda_np1 + self.delta_lambda)

                if self.model.has_coeff():
                    #self.grad_coeffs[:] = self.grad_coeffs[:] + delta_grad_rhs * self.coeff_scaling_factors
                    self.grad_coeffs[:] = self.grad_coeffs[:] + delta_grad_rhs

                self.grad_ics[i, :] = self.grad_ics[i, :] + delta_lambda_rhs
        #print(self.grad_coeffs)
        #print(delta_grad_rhs.dat.data)
        #print(delta_grad_rhs.dat.data)

        if ics_arr is None:
            if self.verbose:
                print('jac norm', np.linalg.norm(self.grad_coeffs), coeffs_arr)
            #print('jac norm', np.linalg.norm(self.grad_coeffs), coeffs_arr * self.coeff_scaling_factors)
            return self.grad_coeffs.copy()
        elif coeffs_arr is None:
            if self.verbose:
                print('jac norm', np.linalg.norm(self.grad_ics))
            return np.ravel(self.grad_ics).copy()
        else:
            if self.verbose:
                print('jac norm', np.linalg.norm(np.hstack([self.grad_coeffs, np.ravel(self.grad_ics)])))
            return np.hstack([self.grad_coeffs, np.ravel(self.grad_ics)])

    def _setup_hvp(self):
        """Allocate the extra storage needed for Hessian-vector products, on first use.

        Storage strategy (see docs/hessian_vector_products.md, section 4): the first-order path
        already stores the full forward trajectory self.states and recomputes stage values inside
        each timestepper step. The HVP adds, per block,
        - self.tangent_states: the full tangent trajectory w_0..w_N (x-space only), because the
          second-order adjoint step at n needs w_{n-1} alongside x_{n-1};
        - self.obs_tangent_sources: M^{-1} sdot_n at each observation step, the second-order
          counterpart of self.obs_sources;
        - single running buffers lambda_dot_np1 / delta_lambda_dot for the second-order adjoint.
        Stage values, stage tangents and stage adjoints are never stored; the timestepper
        recomputes them from (x_{n-1}, w_{n-1}, lambda_n, lambda_dot_n) in every reverse step.
        """
        if getattr(self, '_hvp_ready', False):
            return
        self.tangent_states = create_tangent_states(self.model, self.objective.nsteps)
        self.tangent_ic, _, _ = self.model.get_x_var('tangent_ic')
        self.lambda_dot_np1, _, _ = self.model.get_x_var('lambda_dot_np1')
        self.delta_lambda_dot, _, _ = self.model.get_x_var('delta_lambda_dot')
        self.lambda_dot_source, _, _ = self.model.get_x_var('lambda_dot_source')
        self.obs_tangent_sources = [self.model.get_x_var('obs_tangent_source_' + str(step))[0] for step in self.obs_steps]
        self.hess_coeffs = np.zeros(self.model.get_coeff_size())
        self.hess_ics = np.zeros((self.objective.num_data_blocks, self.local_x_size))

        # second variation of the observation loss: sdot_n = d(source_form)/d(mismatch)[w_n]. For
        # the quadratic objectives this equals source_form with the mismatch replaced by w_n, but
        # differentiating keeps it exact for any loss whose source form depends on the mismatch.
        self.tangent_mismatch, _, _ = self.model.get_x_var('tangent_mismatch')
        self.tangent_source_form = derivative(self.source_form, self.mismatch, self.tangent_mismatch)
        xtrial, _ = self.model.get_x_trial_vars()
        a = inner(self.xtest, xtrial)*self.model.spaces.dx
        tangent_source_problem = LinearVariationalProblem(a, self.tangent_source_form, self.lambda_dot_source, constant_jacobian=True)
        self.lambda_dot_projector = LinearVariationalSolver(tangent_source_problem,
            solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}, options_prefix='opt-proj-tangent')
        self._hvp_ready = True

    def _set_coeff_direction(self, coeffs_dir_arr):
        """Set the timestepper coefficient direction theta_dot (None means zero)."""
        if self.model.has_coeff():
            self.timestepper.set_numpy_coeff_direction(coeffs_dir_arr)

    def _compute_block_states_and_tangents(self, block_id, ics_arr, ics_dir_arr):
        """Integrate one block's states and tangents, storing both trajectories.

        The start state is chosen as in _compute_block_states. The start tangent is
        ics_dir_arr[block_id] (rank-local flattened), or zero if ics_dir_arr is None, which is the
        case for coefficient-only directions. The coefficient direction must already be set.
        """
        if ics_arr is None:
            x0 = self.objective.data_blocks[block_id][0]
        else:
            set_mixed_function_from_flattened_array(self.ic, ics_arr[block_id,:])
            x0 = [self.ic,]
        w0 = None
        if ics_dir_arr is not None:
            set_mixed_function_from_flattened_array(self.tangent_ic, ics_dir_arr[block_id,:])
            w0 = self.tangent_ic
        compute_states_and_tangents(self.states, self.tns, self.tangent_states, self.timestepper,
            self.objective.nsteps, self.dt, x0, w0, self.objective.t_blocks[block_id][0])

    def _compute_observation_sources(self, block_id):
        """Compute the first- and second-order adjoint sources at every observation step.

        Fills self.obs_sources[k] = M^{-1} s_n and self.obs_tangent_sources[k] = M^{-1} sdot_n, with
        s_n the loss derivative at x_n and sdot_n its directional derivative in the direction w_n.
        Returns the rank-local flattened dual sums (sum_n s_n, sum_n sdot_n), which are the direct
        observation contributions to the gradient and the Hessian-vector product in x_0. Like in
        jac, this is done before the reverse sweep, which may overwrite self.states.
        """
        grad_sum = np.zeros(self.local_x_size)
        hess_sum = np.zeros(self.local_x_size)
        for k, step in enumerate(self.obs_steps):
            self._assign_mismatch(block_id, step)
            self.tangent_mismatch.assign(self.tangent_states[step][0])
            self.lambda_np1_projector.solve()
            self.obs_sources[k].assign(self.lambda_source)
            self.lambda_dot_projector.solve()
            self.obs_tangent_sources[k].assign(self.lambda_dot_source)
            grad_sum += create_flattened_numpy_arr_from_mixed_function(assemble(self.source_form))
            hess_sum += create_flattened_numpy_arr_from_mixed_function(assemble(self.tangent_source_form))
        return grad_sum, hess_sum

    def _hvp_sweep(self, coeffs_arr, ics_arr, coeffs_dir_arr, ics_dir_arr, coeffs0=None):
        """Run the forward-over-adjoint HVP sweep and return gradient and Hessian action blocks.

        For every block: integrate the states and tangents together, precompute the first- and
        second-order observation sources, then sweep backward advancing lambda and lambda_dot
        together with take_second_order_adjoint_step (petsc-opt.pdf, eq. 2.11). Arguments are as
        for jac, plus the directions coeffs_dir_arr (flattened, or None for zero) and ics_dir_arr
        ((nblocks, local_x_size), or None for zero).

        Returns (grad_coeffs, grad_ics, hess_coeffs, hess_ics). The ics blocks are rank-local
        (nblocks, local_x_size) dual arrays, grad_ics = M lambda_0 and hess_ics = M lambda_dot_0
        per block; the coefficient blocks are summed over blocks. The gradient is a free
        by-product of the combined sweep and matches jac.
        """
        self._setup_hvp()
        nblocks = self.objective.num_data_blocks
        has_coeff = self.model.has_coeff()
        grad_coeffs = np.zeros(self.model.get_coeff_size())
        hess_coeffs = np.zeros(self.model.get_coeff_size())
        grad_ics = np.zeros((nblocks, self.local_x_size))
        hess_ics = np.zeros((nblocks, self.local_x_size))
        self.delta_lambda.assign(0)
        self.delta_lambda_dot.assign(0)

        self._set_coeffs(coeffs_arr, coeffs0)
        self._set_coeff_direction(coeffs_dir_arr)
        try:
            for i in range(nblocks):
                self.lambda_np1.assign(0)
                self.lambda_dot_np1.assign(0)
                self.ic.assign(0)
                self.tangent_ic.assign(0)
                self.timestepper.reset_internal_vars()

                # forward pass: x_n and w_n for every step (full storage, see _setup_hvp)
                self._compute_block_states_and_tangents(i, ics_arr, ics_dir_arr)

                # observation sources s_n, sdot_n, before the sweep can overwrite self.states
                grad_sum, hess_sum = self._compute_observation_sources(i)
                grad_ics[i, :] += grad_sum
                hess_ics[i, :] += hess_sum

                for n in range(self.objective.nsteps, 0, -1):
                    # lambda_n += M^{-1} s_n and lambda_dot_n += M^{-1} sdot_n at observation steps
                    if n in self.obs_index:
                        k = self.obs_index[n]
                        self.lambda_np1.assign(self.lambda_np1 + self.obs_sources[k])
                        self.lambda_dot_np1.assign(self.lambda_dot_np1 + self.obs_tangent_sources[k])

                    # same per-step reset as jac, for repeatability
                    self.timestepper.reset_internal_vars()
                    delta_lambda_rhs, delta_grad_rhs, delta_lambda_dot_rhs, delta_hess_coeff_rhs = self.timestepper.take_second_order_adjoint_step(
                        self.delta_lambda, self.delta_lambda_dot, self.lambda_np1, self.lambda_dot_np1,
                        self.states[n-1], self.tangent_states[n-1], self.tns[n], self.dt)
                    self.lambda_np1.assign(self.lambda_np1 + self.delta_lambda)
                    self.lambda_dot_np1.assign(self.lambda_dot_np1 + self.delta_lambda_dot)

                    if has_coeff:
                        grad_coeffs += delta_grad_rhs
                        hess_coeffs += delta_hess_coeff_rhs
                    grad_ics[i, :] += delta_lambda_rhs
                    hess_ics[i, :] += delta_lambda_dot_rhs
        finally:
            # leave no stale direction behind for later first-order or HVP calls
            self._set_coeff_direction(None)
        return grad_coeffs, grad_ics, hess_coeffs, hess_ics

    def hessp(self, coeffs_arr, ics_arr, coeffs_dir_arr=None, ics_dir_arr=None, coeffs0=None):
        """Full-space Hessian-vector product of the objective, in the same layout as jac.

        The point is given exactly as for jac: coeffs_arr (flattened coefficients, or None to use
        coeffs0) and ics_arr ((nblocks, local_x_size), or None to start every block from its data).
        The direction has the matching parts coeffs_dir_arr and ics_dir_arr; a part may be None
        (zero), and is ignored for variables that are not being optimized. Returns H v for the
        coefficients if ics_arr is None, for the (raveled) initial states if coeffs_arr is None,
        and their hstack otherwise. The initial-state part is a rank-local dual array, like jac.
        """
        if coeffs_arr is None:
            coeffs_dir_arr = None
        if ics_arr is None:
            ics_dir_arr = None
        _, _, hess_coeffs, hess_ics = self._hvp_sweep(coeffs_arr, ics_arr, coeffs_dir_arr, ics_dir_arr, coeffs0=coeffs0)
        if ics_arr is None:
            result = hess_coeffs
        elif coeffs_arr is None:
            result = np.ravel(hess_ics)
        else:
            result = np.hstack([hess_coeffs, np.ravel(hess_ics)])
        if self.verbose:
            print('hessp norm', np.linalg.norm(result))
        return result.copy()

    def ic_params_hessp(self, ic_map, params, direction, coeffs0=None):
        """Hessian-vector product for analytic initial-condition parameters (opt_type 'ic-params').

        With x0 = x0(p) and T = dx0/dp, H_p v = T^T H_x0 (T v) + (d^2 x0/dp^2 [v])^T dJ/dx0. The first
        term comes from a full-space HVP in the direction T v, and the second from the gradient
        dJ/dx0 = M lambda_0 that the same sweep returns, so one sweep gives both. Requires exactly
        one data block. Returns a global array, identical on every MPI rank.
        """
        if self.objective.num_data_blocks != 1:
            raise ValueError("opt_type 'ic-params' requires exactly one data block")
        x0 = ic_map.forward(params)
        tangent_x0 = ic_map.tangent_action(direction)
        _, grad_ics, _, hess_ics = self._hvp_sweep(None, x0[None, :], None, tangent_x0[None, :], coeffs0=coeffs0)
        # the sweep does not change the parameter Functions, so the tangents in backward() and
        # second_order_backward() are still evaluated at params
        result = ic_map.backward(hess_ics[0]) + ic_map.second_order_backward(grad_ics[0], direction)
        if self.verbose:
            print('hessp norm', np.linalg.norm(result))
        return np.asarray(result, dtype=float)

    def full_hessp(self, opt_type='coeffs', coeffs0=None, ic_map=None):
        """Return full_hessp(x_full, v_full) for an optimization type, matching reduced_problem.

        x_full and v_full are full parameter vectors laid out as for the full_fun / full_jac of
        reduced_problem with the same opt_type ('coeffs', 'ics', 'coeffs+ics' or 'ic-params').
        """
        nblocks = self.objective.num_data_blocks
        reshape = lambda arr: np.reshape(arr, (nblocks, self.local_x_size))
        if opt_type == 'coeffs':
            return lambda coeffs, v: self.hessp(coeffs, None, coeffs_dir_arr=v)
        elif opt_type == 'ics':
            return lambda ics, v: self.hessp(None, reshape(ics), ics_dir_arr=reshape(v), coeffs0=coeffs0)
        elif opt_type == 'coeffs+ics':
            ncoeff = self.model.get_coeff_size()
            return lambda x, v: self.hessp(x[:ncoeff], reshape(x[ncoeff:]), coeffs_dir_arr=v[:ncoeff], ics_dir_arr=reshape(v[ncoeff:]))
        elif opt_type == 'ic-params':
            if ic_map is None:
                raise ValueError("opt_type 'ic-params' requires an ic_map")
            if nblocks != 1:
                raise ValueError("opt_type 'ic-params' requires exactly one data block")
            return lambda params, v: self.ic_params_hessp(ic_map, params, v, coeffs0=coeffs0)
        raise ValueError('unknown opt_type ' + str(opt_type))
