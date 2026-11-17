"""Variable coefficients of the elliptic Hodge Laplacian problems (TODO-elliptic §4.10).

The weighted problem puts coefficients only on the sigma mass, the d u stiffness and a reaction
term; the u inner product stays unweighted (so the kernel and the harmonic-form rule are
unchanged):

    -(mu sigma, tau) + (u, d tau)                          = ...
     (d sigma, v) + (kappa d u, d v) + (c u, v) + (p, v)   = ...

i.e. L u = d(mu^{-1} delta u) + delta(kappa d u) + c u. mu acts on (k-1)-forms (absent for
k = 0), kappa on (k+1)-forms (absent for k = n), c on k-forms. Each is a scalar or, on
vector-proxy forms, an SPD tensor.

cfg (elliptic.coefficients), all entries optional (defaults mu = kappa = 1, c = 0):

    kappa: {kind: constant, value: 1.0, treat_as_coeff: true, bounds: [0.1, 10.0], scaling: 1.0}
    mu:    {kind: field, space: DG0, init: problem, treat_as_coeff: true, log_parametrization: true}
    c:     {kind: constant, value: 0.0}

- kind: constant (value: number, or dim x dim list for rank tensor), problem (the problem-data
  class supplies a UFL expression via coefficient(); never eval'd from the cfg) or field (a
  Function in `space`, initialized from `value` or, with init: problem, from the problem).
- treat_as_coeff: true makes the coefficient a model coefficient (an optimization parameter):
  constants live in R, fields in `space` (DG0 by default); scalars only. The forms then use the
  coefficient Function, so ufl.derivative with respect to it works.
- log_parametrization: the parameter is theta with coefficient exp(theta) (stays positive).
- bounds / scaling: optimizer bounds (in coefficient units; transformed to theta for log) and
  scaling factors, per coefficient dof.
"""
import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy
from firedrake import Constant, Function, FunctionSpace, MixedFunctionSpace, TestFunction, TestFunctions, \
    TrialFunction, TrialFunctions, exp, split

COEFFICIENT_NAMES = ('mu', 'kappa', 'c')
DEFAULT_VALUES = {'mu': 1.0, 'kappa': 1.0, 'c': 0.0}
KINDS = ('constant', 'problem', 'field')
RANKS = ('scalar', 'tensor')


def coefficient_form_degree(name, k):
    """Degree of the forms a coefficient acts on: mu on (k-1)-, kappa on (k+1)-, c on k-forms."""
    return {'mu': k - 1, 'kappa': k + 1, 'c': k}[name]


def coefficient_is_present(name, k, dim):
    """mu needs k > 0 (sigma exists), kappa needs k < dim (d u exists); c is always present."""
    degree = coefficient_form_degree(name, k)
    return 0 <= degree <= dim


def tensor_allowed(name, k, dim):
    """Tensor coefficients need a vector proxy (0 < degree < dim) for the forms they act on."""
    degree = coefficient_form_degree(name, k)
    return 0 < degree < dim


def parse_space(spec):
    """'DG0' -> ('DG', 0), 'CG1' -> ('CG', 1), 'R' -> ('R', 0)."""
    match = re.fullmatch(r'\s*([A-Za-z]+)\s*(\d*)\s*', str(spec))
    if not match:
        raise ValueError("coefficient space must look like 'DG0', 'CG1' or 'R', got " + repr(spec))
    family, degree = match.group(1), match.group(2)
    if family.upper() == 'R':
        return 'R', 0
    if not degree:
        raise ValueError("coefficient space " + repr(spec) + " needs a degree (e.g. DG0)")
    return family.upper(), int(degree)


@dataclass
class EllipticCoefficient:
    """One coefficient (mu, kappa or c) of the weighted Hodge Laplacian (see the module docstring)."""
    name: str
    kind: str = 'constant'
    rank: str = 'scalar'
    value: object = 1.0
    treat_as_coeff: bool = False
    space: str = 'R'
    init: str = 'value'
    bounds: Optional[tuple] = None
    scaling: float = 1.0
    log_parametrization: bool = False

    @property
    def is_default_zero(self):
        """True for a c that is the constant 0 and not a parameter (the reaction term is omitted)."""
        return (self.kind == 'constant' and not self.treat_as_coeff and self.rank == 'scalar'
                and float(self.value) == 0.0)

    def parameter_bounds(self):
        """(lb, ub) of the optimization parameter (theta = log(coefficient) for log)."""
        if self.bounds is None:
            return -numpy.inf, numpy.inf
        lb, ub = self.bounds
        if self.log_parametrization:
            return math.log(lb), math.log(ub)
        return float(lb), float(ub)


def _check_tensor_value(name, value, dim):
    """A dim x dim symmetric matrix, positive definite (mu, kappa) or semidefinite (c)."""
    array = numpy.array(value, dtype=float)
    if array.shape != (dim, dim):
        raise ValueError("tensor coefficient " + name + " needs a " + str(dim) + "x" + str(dim) + " value, got shape "
                         + str(array.shape))
    if not numpy.allclose(array, array.T):
        raise ValueError("tensor coefficient " + name + " must be symmetric")
    smallest = numpy.linalg.eigvalsh(array).min()
    if (name == 'c' and smallest < 0.0) or (name != 'c' and smallest <= 0.0):
        raise ValueError("tensor coefficient " + name + " must be positive " + ("semi" if name == 'c' else "")
                         + "definite")
    return array.tolist()


def parse_coefficients(cfg, k, dim):
    """Parse elliptic.coefficients into {name: EllipticCoefficient} for the blocks that exist.

    Raises ValueError for unknown names/kinds/ranks/keys, a coefficient on an absent block (mu at
    k = 0, kappa at k = n), tensors on scalar-proxy forms or with the wrong shape, non-positive
    mu/kappa (negative c), treat_as_coeff for problem kinds or tensors, log parametrization of a
    non-positive value, and invalid bounds/scaling/space/init entries.
    """
    cfg = cfg or {}
    if not isinstance(cfg, dict):
        raise ValueError("elliptic.coefficients must be a mapping")
    allowed_keys = {'kind', 'rank', 'value', 'treat_as_coeff', 'space', 'init', 'bounds', 'scaling',
                    'log_parametrization'}
    for name in cfg:
        if name not in COEFFICIENT_NAMES:
            raise ValueError("unknown coefficient " + repr(name) + " (known: " + str(list(COEFFICIENT_NAMES)) + ")")
        if not coefficient_is_present(name, k, dim):
            raise ValueError("coefficient " + name + " acts on a block that does not exist for k = " + str(k)
                             + " in " + str(dim) + "D")
    coefficients = {}
    for name in COEFFICIENT_NAMES:
        if not coefficient_is_present(name, k, dim):
            continue
        entry = dict(cfg.get(name) or {})
        unknown = set(entry) - allowed_keys
        if unknown:
            raise ValueError("unknown entries " + str(sorted(unknown)) + " for coefficient " + name)
        coef = EllipticCoefficient(name=name,
                                   kind=entry.get('kind', 'constant'),
                                   rank=entry.get('rank', 'scalar'),
                                   value=entry.get('value', DEFAULT_VALUES[name]),
                                   treat_as_coeff=bool(entry.get('treat_as_coeff', False)),
                                   init=entry.get('init', 'value'),
                                   scaling=float(entry.get('scaling', 1.0)),
                                   log_parametrization=bool(entry.get('log_parametrization', False)))
        if coef.kind not in KINDS:
            raise ValueError("coefficient " + name + ": kind must be one of " + str(list(KINDS)))
        if coef.rank not in RANKS:
            raise ValueError("coefficient " + name + ": rank must be one of " + str(list(RANKS)))
        if coef.rank == 'tensor' and not tensor_allowed(name, k, dim):
            raise ValueError("coefficient " + name + " acts on scalar-proxy forms for k = " + str(k) + " in "
                             + str(dim) + "D; use rank: scalar")
        if coef.init not in ('value', 'problem'):
            raise ValueError("coefficient " + name + ": init must be 'value' or 'problem'")
        if coef.kind == 'problem' and coef.treat_as_coeff:
            raise ValueError("coefficient " + name + ": kind problem cannot be treat_as_coeff (use kind field, init problem)")
        if coef.treat_as_coeff and coef.rank == 'tensor':
            raise ValueError("coefficient " + name + ": tensor coefficients cannot be treat_as_coeff yet")
        if coef.kind == 'constant' and coef.init == 'problem':
            raise ValueError("coefficient " + name + ": init problem needs kind field")
        # Value checks (constants and value-initialized fields).
        if coef.rank == 'tensor' and coef.kind != 'problem':
            coef.value = _check_tensor_value(name, coef.value, dim)
        elif coef.kind != 'problem' and coef.init == 'value':
            value = float(coef.value)
            if (name == 'c' and value < 0.0) or (name != 'c' and value <= 0.0):
                raise ValueError("coefficient " + name + " must be " + ("non-negative" if name == 'c' else "positive"))
            coef.value = value
        if coef.log_parametrization:
            if not coef.treat_as_coeff:
                raise ValueError("coefficient " + name + ": log_parametrization only applies to treat_as_coeff")
            if coef.init == 'value' and float(coef.value) <= 0.0:
                raise ValueError("coefficient " + name + ": log_parametrization needs a positive value")
        # Space: R for constant parameters, a scalar space for fields.
        if coef.kind == 'field':
            coef.space = entry.get('space', 'DG0')
            family, _ = parse_space(coef.space)
            if family == 'R':
                raise ValueError("coefficient " + name + ": fields need a non-R space (use kind constant for R)")
            if coef.rank == 'tensor':
                raise ValueError("coefficient " + name + ": tensor fields are not supported (use kind problem)")
        else:
            coef.space = 'R'
        # Bounds and scaling.
        bounds = entry.get('bounds', None)
        if bounds is not None:
            if len(bounds) != 2 or not float(bounds[0]) < float(bounds[1]):
                raise ValueError("coefficient " + name + ": bounds must be [lb, ub] with lb < ub")
            if coef.log_parametrization and float(bounds[0]) <= 0.0:
                raise ValueError("coefficient " + name + ": log_parametrization needs positive bounds")
            coef.bounds = (float(bounds[0]), float(bounds[1]))
        if coef.scaling <= 0.0:
            raise ValueError("coefficient " + name + ": scaling must be positive")
        coefficients[name] = coef
    return coefficients


class EllipticCoefficients():
    """The coefficients of one model: their Functions, weights and the model-coefficient interface.

    - model_weights(): UFL weights used in the forms; parameters enter through the coefficient
      Function (exp(theta) for log parametrization), so ufl.derivative works.
    - reference_weights(): fixed UFL weights at the initial values, used for the problem data
      (manufactured sources and boundary data must not move with the parameters).
    - The Dynamics-style interface (get_coeff_var, ..., get_coeff_bounds) for the parameters.
    """

    def __init__(self, coefficients, mesh, problem, k, dim, complex):
        """Build the coefficient space (if any parameter exists) and set the initial values."""
        self.coefficients = coefficients
        self.mesh, self.problem = mesh, problem
        self.k, self.dim, self.complex = k, dim, complex
        self.parameters = [c for c in coefficients.values() if c.treat_as_coeff]
        self.coefflist = [c.name for c in self.parameters]
        self.coeffspacelist = [self._space(c) for c in self.parameters]
        if not self.coeffspacelist:
            self.coeffspace = None
        elif len(self.coeffspacelist) == 1:
            self.coeffspace = self.coeffspacelist[0]
        else:
            self.coeffspace = MixedFunctionSpace(self.coeffspacelist)
        self._reference = {name: self._initial_expression(c) for name, c in coefficients.items()}
        self.coeff, self.coeff_sub, self.coeff_split = self.get_coeff_var('coeff')
        if self.coeff is not None:
            self.set_coeffs(self.coeff_sub)

    # --- construction helpers -----------------------------------------------------------------

    def _space(self, coef):
        """Function space of a parameter (R for constants)."""
        family, degree = parse_space(coef.space)
        return FunctionSpace(self.mesh, family, degree)

    def _problem_expression(self, coef):
        """The problem's UFL expression for a coefficient, with its shape checked."""
        expr = self.problem.coefficient(coef.name, self.mesh, self.k, self.dim, self.complex, coef.rank)
        expected = (self.dim, self.dim) if coef.rank == 'tensor' else ()
        if expr.ufl_shape != expected:
            raise ValueError("problem coefficient " + coef.name + " has shape " + str(expr.ufl_shape)
                             + ", expected " + str(expected))
        return expr

    def _initial_expression(self, coef):
        """Fixed UFL value of a coefficient at its initial state."""
        if coef.kind == 'problem':
            return self._problem_expression(coef)
        if coef.kind == 'field':
            source = self._problem_expression(coef) if coef.init == 'problem' else Constant(coef.value)
            return Function(self._space(coef), name=coef.name + '_reference').interpolate(source)
        return Constant(numpy.array(coef.value) if coef.rank == 'tensor' else coef.value)

    # --- weights ------------------------------------------------------------------------------

    def reference_weights(self):
        """{name: UFL} at the initial values (None for the omitted reaction term c = 0)."""
        weights = {}
        for name, coef in self.coefficients.items():
            weights[name] = None if coef.is_default_zero else self._reference[name]
        return weights

    def model_weights(self):
        """{name: UFL} used in the forms; parameters come from the coefficient Function."""
        weights = self.reference_weights()
        for coef in self.parameters:
            theta = self.coeff_split[coef.name]
            weights[coef.name] = exp(theta) if coef.log_parametrization else theta
        return weights

    def reaction_is_positive(self):
        """True if c > 0 everywhere at the initial values (then the problem is non-singular)."""
        coef = self.coefficients['c']
        if coef.is_default_zero:
            return False
        if coef.log_parametrization:
            return True
        if coef.kind == 'constant':
            value = numpy.array(coef.value, dtype=float)
            return bool(numpy.linalg.eigvalsh(value).min() > 0.0) if value.ndim == 2 else bool(value > 0.0)
        # problem/field: sample the scalar (or the tensor diagonal) in a DG space.
        expr = self._reference['c']
        V = FunctionSpace(self.mesh, 'DG', 2)
        samples = [expr] if coef.rank == 'scalar' else [expr[i, i] for i in range(self.dim)]
        minimum = min(float(Function(V).interpolate(s).dat.data_ro.min(initial=numpy.inf)) for s in samples)
        minimum = self.mesh.comm.allreduce(minimum, op=_mpi_min())
        return minimum > 0.0

    # --- model-coefficient interface ----------------------------------------------------------

    def has_coeff(self):
        """True if any coefficient is a model coefficient."""
        return self.coeffspace is not None

    def get_coeff_list(self):
        """Names of the model coefficients."""
        return list(self.coefflist)

    def get_coeff_spaces(self):
        """Spaces of the model coefficients."""
        return list(self.coeffspacelist)

    def get_coeff_var(self, name):
        """(coeff, coeff_sub, coeff_split) like Dynamics.get_coeff_var, or (None, {}, {})."""
        if self.coeffspace is None:
            return None, {}, {}
        coeff = Function(self.coeffspace, name=name)
        if len(self.coeffspacelist) == 1:
            return coeff, {self.coefflist[0]: coeff}, {self.coefflist[0]: coeff}
        pieces = split(coeff)
        coeff_sub = {var: coeff.subfunctions[i] for i, var in enumerate(self.coefflist)}
        coeff_split = {var: pieces[i] for i, var in enumerate(self.coefflist)}
        return coeff, coeff_sub, coeff_split

    def get_coeff_test_trial_vars(self):
        """(test, test_subs, trial, trial_subs) on the coefficient space, or Nones."""
        if self.coeffspace is None:
            return None, {}, None, {}
        test, trial = TestFunction(self.coeffspace), TrialFunction(self.coeffspace)
        if len(self.coeffspacelist) == 1:
            name = self.coefflist[0]
            return test, {name: test}, trial, {name: trial}
        tests, trials = TestFunctions(self.coeffspace), TrialFunctions(self.coeffspace)
        return (test, dict(zip(self.coefflist, tests)), trial, dict(zip(self.coefflist, trials)))

    def set_coeffs(self, coeff_sub):
        """Assign the initial parameter values (theta = log(value) for log parametrization)."""
        for coef in self.parameters:
            target = coeff_sub[coef.name]
            if coef.kind == 'field' and coef.init == 'problem':
                expr = self._problem_expression(coef)
                target.interpolate(_log(expr) if coef.log_parametrization else expr)
            else:
                value = float(coef.value)
                target.assign(math.log(value) if coef.log_parametrization else value)

    def get_coeff_size(self):
        """Global number of parameter dofs (0 without parameters)."""
        return 0 if self.coeffspace is None else self.coeffspace.dim()

    def _per_dof(self, values):
        """Repeat one value per parameter over its local dofs, in flattened-array order."""
        arrays = [numpy.full(_local_size(f), v) for f, v in zip(self._blocks(self.coeff), values)]
        return numpy.hstack(arrays) if arrays else numpy.zeros(0)

    def _blocks(self, function):
        """Per-parameter blocks of a coefficient Function (or Cofunction)."""
        if self.coeffspace is None:
            return []
        if len(self.coeffspacelist) == 1:
            return [function]
        return list(function.subfunctions)

    def get_coeff_bounds(self):
        """(lb, ub) numpy arrays over the flattened parameter dofs (theta bounds for log)."""
        bounds = [c.parameter_bounds() for c in self.parameters]
        return self._per_dof([b[0] for b in bounds]), self._per_dof([b[1] for b in bounds])

    def get_coeff_scaling_factors(self):
        """Scaling factors over the flattened parameter dofs."""
        return self._per_dof([c.scaling for c in self.parameters])

    def flatten(self, function):
        """Flattened numpy array of a coefficient Function/Cofunction (rank-local dofs)."""
        return numpy.hstack([numpy.ravel(f.dat.data_ro) for f in self._blocks(function)]) if self.coeff is not None \
            else numpy.zeros(0)

    def set_from_array(self, array):
        """Set the coefficient Function from a flattened array (inverse of flatten)."""
        offset = 0
        for f in self._blocks(self.coeff):
            n = _local_size(f)
            f.dat.data[:] = numpy.reshape(array[offset:offset + n], f.dat.data.shape)
            offset += n


def _local_size(function):
    """Number of rank-local values of a (non-mixed) Function."""
    return int(numpy.asarray(function.dat.data_ro).size)


def _log(expr):
    """UFL natural logarithm (imported lazily to keep the header short)."""
    from firedrake import ln
    return ln(expr)


def _mpi_min():
    """MPI.MIN (lazy import)."""
    from mpi4py import MPI
    return MPI.MIN
