"""Elliptic Hodge Laplacian problems (d delta + delta d) u = f for k-forms (TODO-elliptic §4.6).

EllipticProblem is a time-invariant model. It is not a subclass of the time-dependent Model,
but implements the subset of its interface used by the driver and Output (see §2), plus solve().

Weak form (AFW mixed formulation, TODO-elliptic §1.1), written with the sigma equation
multiplied by -1 so that the operator is symmetric:

    -(sigma, tau) + (u, d tau)                   = + int_{Gn} B_{k-1}(tau, g_u)
     (d sigma, v) + (d u, d v) + (p, v)          =   (f, v) + int_{Gn} B_k(v, g_du)
     (u, q)                                      =   0

for sigma in V^{k-1} (absent for k = 0), u in V^k and p in the harmonic forms (absent if there
are none). Essential BCs are DirichletBCs on sigma and u; natural BCs are the boundary terms.

The harmonic constraint is enforced in one of two ways (elliptic.harmonic_constraint, §1.5):
- 'lagrange': p is an R-space unknown. Firedrake cannot assemble monolithic matrices with R
  blocks, so this solve is matrix-free GMRES with an assembled LU preconditioner
  (dimswe.harmonic_forms.lagrange_solver_parameters).
- 'nullspace': no p block. The singular system is solved with the harmonic nullspace attached,
  after lifting the essential BCs and projecting the lifted rhs orthogonal to the harmonic
  forms; the harmonic part of u is removed afterwards, and p is reported from the projection.
Either way, p is reported as a Function in V^k (sub['p']).
"""
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy
from firedrake import derivative as fd_derivative, replace as fd_replace
from firedrake import (DirichletBC, Function, LinearSolver, LinearVariationalProblem,
                       LinearVariationalSolver, MixedFunctionSpace, TestFunction, TestFunctions,
                       TrialFunction, TrialFunctions, action, assemble, dx, inner)

from .boundary_conditions import (build_essential_bcs, build_natural_terms, face_bcs_by_direction,
                                  faces_of_type, get_boundary_registry, parse_boundary_conditions)
from .differential_forms import apply_weight, exterior_derivative, form_space, normalize_complex
from .elliptic_coefficients import EllipticCoefficients, parse_coefficients
from .elliptic_problems import BOUNDARY_DATA_MODES, collect_boundary_data, get_elliptic_problem
from .harmonic_forms import (HARMONIC_FORM_MODES, build_harmonic_nullspace, combine_harmonic_forms,
                             get_harmonic_index_sets, harmonic_basis, harmonic_constraint_form,
                             harmonic_multiplier_space, interpolate_or_project, l2_inner,
                             lagrange_preconditioner_form, lagrange_solver_parameters,
                             multiplier_coefficients, orthonormalize_harmonic_basis,
                             project_rhs_orthogonal_to_harmonic, remove_harmonic_component)
from .logger import EmptyLogger
from .meshes import FEEC_FAMILIES, family_elements, get_mesh_and_spaces, get_mesh_type_dimension
from .parameters import elliptic_solver_parameters as ELLIPTIC_SOLVERS

# Allowed values of elliptic.harmonic_constraint and elliptic.solver.
HARMONIC_CONSTRAINTS = ('lagrange', 'nullspace')
SOLVERS = tuple(ELLIPTIC_SOLVERS)
# FEEC families available in DeRhamComplex (TODO-elliptic §4.9); P-/P+ need simplicial cells.
SUPPORTED_FAMILIES = FEEC_FAMILIES
SIMPLICIAL_FAMILIES = ('P-', 'P+')


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------

def _default_if_none(value, default):
    """value, or default when value is None (unlike `value or default`, keeps 0 for validation)."""
    return default if value is None else value


@dataclass
class EllipticConfig:
    """Validated 'elliptic' cfg section (TODO-elliptic §3).

    Attributes:
        dim: spatial dimension.
        k: form degree (elliptic.form_degree).
        complex: 2D complex ('grad-curl' | 'curl-div'), None in 1D/3D.
        boundary_conditions: raw per-face BC cfg (dict face -> 'essential' | 'natural').
        boundary_data: 'exact' | 'problem' | 'homogeneous'.
        harmonic_forms: 'auto' | 'none' | 'numerical'.
        harmonic_constraint: 'lagrange' | 'nullspace'.
        solver: 'elliptic-direct' | 'elliptic-direct-singular' | 'elliptic-minres'.
        solver_options: extra PETSc options merged on top of the selected solver.
        quadrature_degree: quadrature degree for all forms (default 2 * order + 2). Always explicit:
            with UFL's per-form estimate, the lagrange and nullspace paths (which split and assemble
            the forms differently) integrate the source with different rules and disagree at the
            quadrature-error level (seen in 1D, ~1e-5).
        error_quadrature_degree: quadrature degree for error norms (default 2 * order + 4).
        lagrange_regularization: epsilon of the regularized preconditioners a + eps (u, v) used
            by 'lagrange' and by 'nullspace' with a direct solver (see
            harmonic_forms.lagrange_preconditioner_form).
        coefficients: parsed elliptic.coefficients ({name: EllipticCoefficient}, see
            dimswe.elliptic_coefficients); defaults mu = kappa = 1, c = 0.
        registry, face_bcs: face registry and parsed per-face BC types (set when a mesh is given).
    """
    dim: int
    k: int
    complex: Optional[str]
    boundary_conditions: dict
    boundary_data: str = 'exact'
    harmonic_forms: str = 'auto'
    harmonic_constraint: str = 'lagrange'
    solver: str = 'elliptic-direct'
    solver_options: dict = field(default_factory=dict)
    quadrature_degree: Optional[int] = None
    error_quadrature_degree: Optional[int] = None
    lagrange_regularization: float = 1.0
    coefficients: dict = field(default_factory=dict)
    registry: Optional[dict] = None
    face_bcs: Optional[dict] = None

    @classmethod
    def from_parameters(cls, parameters, spaces=None):
        """Build and validate the config; with spaces (a DeRhamComplex) the BCs are parsed too.

        Raises ValueError for: k out of range, invalid 2D complex, lump_mass: true, unknown family,
        family/cell mismatch (P-/P+ need simplicial_cells: true, Q/S-/S false), order below the
        family minimum, invalid option values, and (with spaces) BCs on periodic faces or missing
        BCs on bounded faces; NotImplementedError for serendipity in 3D and for S- (see meshes.family_elements).
        """
        if 'elliptic' not in parameters:
            raise ValueError("elliptic problems need an 'elliptic' cfg section")
        cfg = parameters['elliptic'] or {}
        mesh_cfg = parameters['mesh']
        dim = mesh_cfg.get('dim') or get_mesh_type_dimension(mesh_cfg['type'])
        if dim is None:
            raise ValueError("the dimension of a 'file' mesh is only known once it is loaded; pass spaces")

        disc = parameters['spatial-discretization']
        if disc.get('lump_mass', False):
            raise ValueError("lump_mass: true is not supported for elliptic problems")
        family = disc['family']
        if family not in SUPPORTED_FAMILIES:
            raise ValueError("FEEC family " + str(family) + " is unknown (known: " + str(list(SUPPORTED_FAMILIES)) + ")")
        simplicial = bool(mesh_cfg.get('simplicial_cells', False))
        if dim > 1 and simplicial != (family in SIMPLICIAL_FAMILIES):
            raise ValueError("family " + family + " needs simplicial_cells: " + str(family in SIMPLICIAL_FAMILIES).lower()
                             + " (P-/P+ are simplicial, Q/S-/S use quadrilaterals/hexahedra)")
        order = int(disc['order'])
        if order < 1:
            raise ValueError("spatial-discretization.order must be >= 1")
        # Minimum orders (P+: r >= n, S: r >= 2); serendipity in 3D and S- raise NotImplementedError.
        family_elements(family, dim, order)

        if 'form_degree' not in cfg:
            raise ValueError("elliptic.form_degree (k) is required")
        k = cfg['form_degree']
        if not isinstance(k, int) or k < 0 or k > dim:
            raise ValueError("elliptic.form_degree must be an integer in [0, " + str(dim) + "], got " + repr(k))
        complex = normalize_complex(dim, cfg.get('complex', None))

        config = cls(dim=dim, k=k, complex=complex,
                     boundary_conditions=cfg.get('boundary_conditions', None) or {},
                     boundary_data=cfg.get('boundary_data', 'exact'),
                     harmonic_forms=cfg.get('harmonic_forms', 'auto'),
                     harmonic_constraint=cfg.get('harmonic_constraint', 'lagrange'),
                     solver=cfg.get('solver', 'elliptic-direct'),
                     solver_options=dict(cfg.get('solver_options', None) or {}),
                     quadrature_degree=_default_if_none(cfg.get('quadrature_degree', None), 2 * order + 2),
                     error_quadrature_degree=cfg.get('error_quadrature_degree', 2 * order + 4),
                     lagrange_regularization=float(cfg.get('lagrange_regularization', 1.0)),
                     coefficients=parse_coefficients(cfg.get('coefficients', None), k, dim))
        config.validate_options()
        if spaces is not None:
            config.parse_boundary_conditions(spaces.mesh, mesh_cfg['type'])
        return config

    def validate_options(self):
        """Check the string/number options against their allowed values."""
        checks = [('boundary_data', self.boundary_data, BOUNDARY_DATA_MODES),
                  ('harmonic_forms', self.harmonic_forms, HARMONIC_FORM_MODES),
                  ('harmonic_constraint', self.harmonic_constraint, HARMONIC_CONSTRAINTS),
                  ('solver', self.solver, SOLVERS)]
        for name, value, allowed in checks:
            if value not in allowed:
                raise ValueError("elliptic." + name + " must be one of " + str(list(allowed)) + ", got " + repr(value))
        for name in ('quadrature_degree', 'error_quadrature_degree'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError("elliptic." + name + " must be a positive integer")
        if self.lagrange_regularization <= 0.0:
            raise ValueError("elliptic.lagrange_regularization must be positive")

    def parse_boundary_conditions(self, mesh, mesh_type):
        """Build the face registry of mesh and parse the per-face BCs into self.face_bcs."""
        self.registry = get_boundary_registry(mesh, mesh_type)
        self.face_bcs = parse_boundary_conditions(self.boundary_conditions, self.registry)


def elliptic_solver_parameters(config, num_harmonic, num_blocks, multiplier_indices=()):
    """PETSc options for the elliptic solve (before merging config.solver_options).

    - lagrange with harmonic forms: matrix-free GMRES (harmonic_forms.lagrange_solver_parameters);
      only solver 'elliptic-direct' is meaningful there (its LU is used as preconditioner).
    - nullspace with harmonic forms: 'elliptic-direct' is upgraded to 'elliptic-direct-singular'
      (a plain LU of a singular matrix fails): right-preconditioned GMRES with the nullspace
      attached, preconditioned by the LU of the regularized a + eps (u, v) (see _build_forms).
    - 'elliptic-minres': MINRES; with several blocks, an additive fieldsplit with LU per block of
      the Riesz-map preconditioner (EllipticProblem._riesz_form).
    """
    if config.harmonic_constraint == 'lagrange' and num_harmonic > 0:
        if config.solver != 'elliptic-direct':
            raise ValueError("harmonic_constraint: lagrange with harmonic forms uses its own matrix-free solver; set solver: elliptic-direct (or use harmonic_constraint: nullspace)")
        params = lagrange_solver_parameters(num_blocks, multiplier_indices)
    else:
        name = config.solver
        if name == 'elliptic-direct' and num_harmonic > 0:
            name = 'elliptic-direct-singular'
        params = copy.deepcopy(ELLIPTIC_SOLVERS[name])
        if name == 'elliptic-minres':
            if num_blocks > 1:
                params.update({'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'additive'})
                for i in range(num_blocks):
                    prefix = 'fieldsplit_' + str(i) + '_'
                    params.update({prefix + 'ksp_type': 'preonly', prefix + 'pc_type': 'lu',
                                   prefix + 'pc_factor_mat_solver_type': 'mumps'})
            else:
                params.update({'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
    params.update(config.solver_options)
    return params


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

def _add(form, term):
    """form + term, treating the integer 0 as 'no term' (UFL forms and 0 do not always mix)."""
    if isinstance(term, int) and term == 0:
        return form
    if isinstance(form, int) and form == 0:
        return term
    return form + term


# ---------------------------------------------------------------------------------------------
# Shared builders (used by EllipticProblem and elliptic_eigen.HodgeLaplacianEigenproblem)
# ---------------------------------------------------------------------------------------------

@dataclass
class BlockLayout:
    """Blocks of the mixed space [sigma (k > 0), u, extra blocks (e.g. R multipliers)].

    Attributes:
        W: the (mixed) function space (the single space itself when there is one block).
        block_spaces: list of the block spaces.
        sigma_index, u_index: block indices (sigma_index is None for k = 0).
        extra_indices: block indices of the extra spaces (the scalar R multipliers).
        Vsigma, Vk: spaces of sigma ((k-1)-forms, None for k = 0) and u (k-forms).
    """
    W: object
    block_spaces: list
    sigma_index: Optional[int]
    u_index: int
    extra_indices: list
    Vsigma: object
    Vk: object

    @property
    def num_blocks(self):
        """Number of blocks."""
        return len(self.block_spaces)


def build_block_layout(spaces, k, dim, complex, extra_spaces=()):
    """Mixed space for the Hodge Laplacian of k-forms: [V^{k-1} (k > 0), V^k, *extra_spaces]."""
    blocks = []
    Vsigma, sigma_index = None, None
    if k > 0:
        Vsigma = form_space(spaces, k - 1, dim, complex)
        sigma_index = len(blocks)
        blocks.append(Vsigma)
    Vk = form_space(spaces, k, dim, complex)
    u_index = len(blocks)
    blocks.append(Vk)
    extra_indices = list(range(len(blocks), len(blocks) + len(extra_spaces)))
    blocks.extend(extra_spaces)
    W = blocks[0] if len(blocks) == 1 else MixedFunctionSpace(blocks)
    return BlockLayout(W, blocks, sigma_index, u_index, extra_indices, Vsigma, Vk)


def split_arguments(layout):
    """Trial and test functions per block: dicts with keys 'sigma', 'u' and (lists) 'extra'."""
    if layout.num_blocks == 1:
        trials, tests = (TrialFunction(layout.W),), (TestFunction(layout.W),)
    else:
        trials, tests = TrialFunctions(layout.W), TestFunctions(layout.W)
    trial, test = {}, {}
    for name, index in (('sigma', layout.sigma_index), ('u', layout.u_index)):
        if index is not None:
            trial[name], test[name] = trials[index], tests[index]
    if layout.extra_indices:
        trial['extra'] = [trials[i] for i in layout.extra_indices]
        test['extra'] = [tests[i] for i in layout.extra_indices]
    return trial, test


def build_harmonic_basis(config, mesh, mesh_type, Vk, measure, reaction_positive=False):
    """Harmonic index sets (TODO-elliptic §1.4) and their L2-orthonormal basis in V^k.

    With a reaction term c > 0 everywhere (reaction_positive) the operator is non-singular and
    there are no harmonic forms to constrain: ([], []) is returned.
    """
    if reaction_positive:
        return [], []
    by_direction = face_bcs_by_direction(config.registry, config.face_bcs)
    index_sets = get_harmonic_index_sets(config.harmonic_forms, config.k, config.dim, mesh.periodic_directions,
                                         by_direction, mesh_type)
    basis = orthonormalize_harmonic_basis(harmonic_basis(config.k, config.dim, config.complex, index_sets), Vk,
                                          measure=measure)
    return index_sets, basis


def build_essential_bc_lists(layout, config, g_sigma=None, g_u=None):
    """DirichletBCs on the essential faces, split as (sigma BCs, u BCs).

    g_sigma, g_u: per-face data dicts (or None for homogeneous BCs). The DG u of k = n has no
    trace, so it gets no BC.
    """
    faces = faces_of_type(config.registry, config.face_bcs, 'essential')
    u_index = layout.u_index if config.k < config.dim else None
    bcs_sigma = build_essential_bcs(layout.W, layout.sigma_index, None, faces, g_sigma=g_sigma)
    bcs_u = build_essential_bcs(layout.W, None, u_index, faces, g_u=g_u)
    return bcs_sigma, bcs_u


def mass_sigma_form(sigma, tau, measure, mu=None):
    """-(mu sigma, tau) (sign chosen for a symmetric operator)."""
    return -inner(apply_weight(mu, sigma), tau) * measure


def coupling_form(sigma, tau, u, v, k, dim, complex, measure):
    """(u, d tau) + (d sigma, v) for sigma, tau in V^{k-1}."""
    d = lambda w: exterior_derivative(w, k - 1, dim, complex)
    return (inner(u, d(tau)) + inner(d(sigma), v)) * measure


def stiffness_form(u, v, k, dim, complex, measure, kappa=None):
    """(kappa d u, d v), or 0 for k = dim."""
    if k == dim:
        return 0
    d = lambda w: exterior_derivative(w, k, dim, complex)
    return inner(apply_weight(kappa, d(u)), d(v)) * measure


def reaction_form(u, v, measure, c=None):
    """(c u, v), or 0 when there is no reaction term."""
    if c is None:
        return 0
    return inner(apply_weight(c, u), v) * measure


def hodge_bilinear_form(trial, test, k, dim, complex, measure, weights=None):
    """The symmetric (weighted) Hodge Laplacian form a_0:

    -(mu sigma, tau) + (u, d tau) + (d sigma, v) + (kappa d u, d v) + (c u, v).
    weights: {'mu', 'kappa', 'c'} UFL weights (missing or None: 1, 1 and no reaction term).
    """
    weights = weights or {}
    u, v = trial['u'], test['u']
    a = stiffness_form(u, v, k, dim, complex, measure, weights.get('kappa'))
    a = _add(a, reaction_form(u, v, measure, weights.get('c')))
    if k > 0:
        sigma, tau = trial['sigma'], test['sigma']
        a = _add(a, mass_sigma_form(sigma, tau, measure, weights.get('mu')))
        a = _add(a, coupling_form(sigma, tau, u, v, k, dim, complex, measure))
    return a


def riesz_form(trial, test, k, dim, complex, measure, weights=None):
    """SPD Riesz-map form (mu sigma, tau) + (d sigma, d tau) + ((1 + c) u, v) + (kappa d u, d v).

    The weights follow the problem's so the preconditioner stays robust for variable coefficients.
    """
    weights = weights or {}
    u, v = trial['u'], test['u']
    aP = _add(inner(u, v) * measure, stiffness_form(u, v, k, dim, complex, measure, weights.get('kappa')))
    aP = _add(aP, reaction_form(u, v, measure, weights.get('c')))
    if k > 0:
        sigma, tau = trial['sigma'], test['sigma']
        d = lambda w: exterior_derivative(w, k - 1, dim, complex)
        aP = aP + (inner(apply_weight(weights.get('mu'), sigma), tau) + inner(d(sigma), d(tau))) * measure
    return aP


class EllipticProblem():
    """Time-invariant model solving the Hodge Laplacian for k-forms (see the module docstring).

    Construction: problem data -> mesh/spaces -> config/BCs -> harmonic forms -> spaces and blocks
    -> boundary data -> forms. Call solve() to compute the solution; statistics and diagnostics
    follow the Model interface.
    """

    #: The driver uses this to skip time stepping.
    is_time_invariant = True

    def __init__(self, parameters, logger=None, mesh=None):
        """Set up everything up to (not including) the solve.

        Args:
            parameters: full parameter dictionary (sections mesh, spatial-discretization,
                elliptic, elliptic-problem).
            logger: DIMSWE logger (default: silent).
            mesh: optional existing mesh to reuse (must match the mesh parameters).
        """
        self.parameters = parameters
        self.logger = logger if logger is not None else EmptyLogger()
        self.problem = get_elliptic_problem(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.problem, mesh=mesh)
        self.config = EllipticConfig.from_parameters(parameters, self.spaces)
        self.dim, self.k, self.complex = self.config.dim, self.config.k, self.config.complex
        self.dx = dx(degree=self.config.quadrature_degree)
        self.error_dx = dx(degree=self.config.error_quadrature_degree)
        # Coefficients: the forms use the model weights (parameters enter through the coefficient
        # Function), the problem data use the fixed reference weights (TODO-elliptic §4.10).
        self.coefficients = EllipticCoefficients(self.config.coefficients, self.mesh, self.problem,
                                                 self.k, self.dim, self.complex)
        self.weights = self.coefficients.model_weights()
        self.problem.weights = self.coefficients.reference_weights()

        self._build_harmonic_forms()
        self._build_spaces()
        self._build_arguments()
        self.exact = self.problem.get_exact(self.mesh, self.k, self.dim, self.complex, self.harmonic_index_sets) \
            if self.problem.has_exact else None
        self.boundary_data = collect_boundary_data(self.problem, self.config.boundary_data, self.mesh,
                                                   self.config.registry, self.config.face_bcs,
                                                   self.k, self.dim, self.complex)
        self._build_bcs()
        self._build_forms()

        self.solution = Function(self.W, name='solution')
        self.sub = {}
        self.p_coefficients = []
        self.ksp_iterations = 0
        self.statistics = {}
        self.diagnostics = {}

    # --- construction -------------------------------------------------------------------------

    def _build_harmonic_forms(self):
        """Index sets and L2-orthonormal basis of the harmonic k-forms (TODO-elliptic §1.4)."""
        self.Vk = form_space(self.spaces, self.k, self.dim, self.complex)
        # c > 0 everywhere makes the problem non-singular: no harmonic constraint.
        self.reaction_positive = self.coefficients.reaction_is_positive()
        self.harmonic_index_sets, self.harmonic = build_harmonic_basis(self.config, self.mesh,
                                                                       self.parameters['mesh']['type'], self.Vk, self.dx,
                                                                       reaction_positive=self.reaction_positive)
        self.num_harmonic = len(self.harmonic)
        self.logger.output('elliptic: ' + str(self.num_harmonic) + ' harmonic ' + str(self.k) + '-forms '
                           + str(self.harmonic_index_sets), 0)

    def _build_spaces(self):
        """Blocks [sigma (k > 0), u, p_1..p_m (lagrange with harmonic forms)] and the space W."""
        self.use_multiplier = self.config.harmonic_constraint == 'lagrange' and self.num_harmonic > 0
        # One scalar R block per harmonic form (Firedrake has no vector-valued R arguments).
        self.R = harmonic_multiplier_space(self.mesh, self.num_harmonic) if self.use_multiplier else []
        self.layout = build_block_layout(self.spaces, self.k, self.dim, self.complex, extra_spaces=self.R)
        self.W, self.block_spaces = self.layout.W, self.layout.block_spaces
        self.sigma_index, self.u_index = self.layout.sigma_index, self.layout.u_index
        self.p_indices = self.layout.extra_indices
        self.Vsigma, self.Vk = self.layout.Vsigma, self.layout.Vk
        self.num_blocks = self.layout.num_blocks

    def _build_arguments(self):
        """Trial/test functions per block: self.trial / self.test dicts with keys sigma, u, p."""
        self.trial, self.test = split_arguments(self.layout)
        if 'extra' in self.trial:
            # Lists over the scalar R blocks.
            self.trial['p'], self.test['p'] = self.trial.pop('extra'), self.test.pop('extra')

    def _build_bcs(self):
        """DirichletBCs on the essential faces: tr sigma = g_sigma and tr u = g_u (§1.6)."""
        self.bcs_sigma, self.bcs_u = build_essential_bc_lists(self.layout, self.config,
                                                              g_sigma=self.boundary_data.essential_sigma,
                                                              g_u=self.boundary_data.essential_u)
        self.bcs = self.bcs_sigma + self.bcs_u

    def _d(self, w, j):
        """Exterior derivative of a j-form (None for j = dim)."""
        return exterior_derivative(w, j, self.dim, self.complex)

    def _harmonic(self):
        """Lagrange-multiplier terms sum_i p_i (h_i, v) + q_i (u, h_i), or 0."""
        if not self.use_multiplier:
            return 0
        return harmonic_constraint_form(self.trial['u'], self.test['u'], self.trial['p'], self.test['p'],
                                        self.harmonic, self.dx)

    def _rhs(self):
        """(f, v) with the problem's source."""
        f = self.problem.get_source(self.mesh, self.k, self.dim, self.complex, self.harmonic_index_sets)
        self.source = f
        return inner(f, self.test['u']) * self.dx

    def _natural_terms(self):
        """Boundary functionals of the natural BCs, in the sign convention of this module."""
        faces = faces_of_type(self.config.registry, self.config.face_bcs, 'natural')
        degree = self.config.quadrature_degree
        sigma_term, u_term = build_natural_terms(self.test.get('sigma'), self.test['u'], faces,
                                                 self.boundary_data.natural_u, self.boundary_data.natural_du,
                                                 self.k, self.dim, self.complex, self.spaces.n, degree=degree)
        # build_natural_terms uses the unscaled sigma equation; ours is multiplied by -1.
        if not (isinstance(sigma_term, int) and sigma_term == 0):
            sigma_term = -sigma_term
        return _add(sigma_term, u_term)

    def _bilinear_form_without_harmonic(self):
        """The bilinear form a_0 (everything except the Lagrange-multiplier terms)."""
        return hodge_bilinear_form(self.trial, self.test, self.k, self.dim, self.complex, self.dx, self.weights)

    def _riesz_form(self):
        """SPD Riesz-map preconditioner with the problem's weights (see riesz_form)."""
        return riesz_form(self.trial, self.test, self.k, self.dim, self.complex, self.dx, self.weights)

    def _build_forms(self):
        """Assemble the UFL forms a, L (and the preconditioner form aP when one is needed)."""
        self.a0 = self._bilinear_form_without_harmonic()
        self.a = _add(self.a0, self._harmonic())
        self.L = _add(self._rhs(), self._natural_terms())
        self.aP = None
        if self.use_multiplier:
            self.aP = self.a0 + lagrange_preconditioner_form(self.trial['u'], self.test['u'], self.trial['p'],
                                                             self.test['p'], self.config.lagrange_regularization,
                                                             self.dx)
        elif self.config.solver == 'elliptic-minres':
            self.aP = self._riesz_form()
        elif self.num_harmonic > 0:
            # nullspace approach with a direct solver: precondition with the LU of the regularized,
            # non-singular a + eps (u, v) instead of relying on MUMPS null-pivot detection (which
            # can miss the null pivot and then amplify the nullspace direction enormously).
            self.aP = self.a0 + self.config.lagrange_regularization * inner(self.trial['u'], self.test['u']) * self.dx
        self.solver_parameters = elliptic_solver_parameters(self.config, self.num_harmonic, self.num_blocks,
                                                            self.p_indices)

    # --- solve --------------------------------------------------------------------------------

    def _block(self, function, index):
        """Block index of a (possibly non-mixed) Function or Cofunction."""
        return function if self.num_blocks == 1 else function.subfunctions[index]

    def solve(self):
        """Assemble and solve once; fills self.solution and self.sub ('sigma', 'u', 'p')."""
        if self.num_harmonic > 0 and self.config.harmonic_constraint == 'nullspace':
            self._solve_nullspace()
        else:
            self._solve_variational()
        self.sub = {}
        if self.sigma_index is not None:
            self.sub['sigma'] = self._block(self.solution, self.sigma_index)
        self.sub['u'] = self._block(self.solution, self.u_index)
        if self.num_harmonic > 0:
            self.sub['p'] = combine_harmonic_forms(self.p_coefficients, self.harmonic)
            self.sub['p'].rename('p')
        self.logger.output('elliptic: solved with ' + str(self.ksp_iterations) + ' KSP iterations', 0)
        return self.solution

    def _solve_variational(self):
        """Non-singular system: no harmonic forms, or the 'lagrange' bordered system."""
        problem = LinearVariationalProblem(self.a, self.L, self.solution, bcs=self.bcs, aP=self.aP)
        solver = LinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                         options_prefix='elliptic_')
        solver.solve()
        self.ksp_iterations = solver.snes.ksp.getIterationNumber()
        if self.use_multiplier:
            self.p_coefficients = multiplier_coefficients([self.solution.subfunctions[i] for i in self.p_indices])

    def _homogeneous(self, bcs):
        """Homogeneous copies of DirichletBCs (same space and sub domain)."""
        return [DirichletBC(bc.function_space(), 0, bc.sub_domain) for bc in bcs]

    def _solve_nullspace(self):
        """Singular system with the harmonic nullspace (§1.5), with explicit BC lifting.

        w = w_g + w_0, where w_g carries the essential BC values and w_0 solves the system with
        homogeneous BCs and the lifted rhs L - a(w_g, .). The lifted rhs is made orthogonal to the
        harmonic forms (its coefficients give p) before the solve; Firedrake's own lifting would
        run after the projection and could reintroduce a harmonic component.
        """
        lift = Function(self.W)
        for bc in self.bcs:
            bc.apply(lift)
        rhs = assemble(self.L - action(self.a, lift)) if self.bcs else assemble(self.L)
        hom_sigma, hom_u = self._homogeneous(self.bcs_sigma), self._homogeneous(self.bcs_u)
        # Zero the rhs on the constrained dofs (the correction w_0 vanishes there).
        for index, bcs in ((self.sigma_index, hom_sigma), (self.u_index, hom_u)):
            for bc in bcs:
                self._block(rhs, index).dat.data[bc.nodes] = 0.0
        u_rhs_index = None if self.num_blocks == 1 else self.u_index
        _, self.p_coefficients = project_rhs_orthogonal_to_harmonic(rhs, self.harmonic, u_index=u_rhs_index,
                                                                     measure=self.dx,
                                                                     bcs=hom_u, logger=self.logger)
        bcs0 = hom_sigma + hom_u
        A = assemble(self.a, bcs=bcs0)
        P = assemble(self.aP, bcs=bcs0) if self.aP is not None else None
        nullspace = build_harmonic_nullspace(self.W, self.u_index, self.harmonic)
        solver = LinearSolver(A, P=P, solver_parameters=self.solver_parameters, nullspace=nullspace,
                              transpose_nullspace=nullspace, options_prefix='elliptic_')
        correction = Function(self.W)
        solver.solve(correction, rhs)
        self.ksp_iterations = solver.ksp.getIterationNumber()
        self.solution.assign(lift + correction)
        removed = remove_harmonic_component(self._block(self.solution, self.u_index), self.harmonic,
                                            measure=self.dx)
        self.logger.output('elliptic: removed harmonic part of u with coefficients ' + str(removed), 1)

    # --- sensitivities (TODO-elliptic §4.10) --------------------------------------------------

    def solve_adjoint(self, dJdx):
        """Solve the adjoint system A^T lambda = dJ/dx for the current solution; returns lambda.

        dJdx: Cofunction on W (the derivative of the objective with respect to the solution).
        The operator is symmetric, so the forward operator is reused, with homogeneous essential
        BCs (the BC values do not depend on the coefficients) and the same harmonic treatment:
        - lagrange: the bordered system (the multiplier rows of dJdx are zero);
        - nullspace: the forward map ends with u <- u - P_h u, whose adjoint is the same
          projection of the adjoint source, so dJdx is projected orthogonal to the harmonic forms
          and the nullspace is attached; lambda is normalized with (lambda_u, h_i) = 0.
        """
        rhs = dJdx.copy(deepcopy=True)
        bcs0 = self._homogeneous(self.bcs)
        for index, bcs in ((self.sigma_index, self._homogeneous(self.bcs_sigma)),
                           (self.u_index, self._homogeneous(self.bcs_u))):
            for bc in bcs:
                self._block(rhs, index).dat.data[bc.nodes] = 0.0
        adjoint = Function(self.W, name='adjoint')
        if self.num_harmonic > 0 and self.config.harmonic_constraint == 'nullspace':
            u_rhs_index = None if self.num_blocks == 1 else self.u_index
            project_rhs_orthogonal_to_harmonic(rhs, self.harmonic, u_index=u_rhs_index, measure=self.dx)
            A = assemble(self.a, bcs=bcs0)
            P = assemble(self.aP, bcs=bcs0) if self.aP is not None else None
            nullspace = build_harmonic_nullspace(self.W, self.u_index, self.harmonic)
            solver = LinearSolver(A, P=P, solver_parameters=self.solver_parameters, nullspace=nullspace,
                                  transpose_nullspace=nullspace, options_prefix='elliptic_adjoint_')
            solver.solve(adjoint, rhs)
            remove_harmonic_component(self._block(adjoint, self.u_index), self.harmonic, measure=self.dx)
        else:
            problem = LinearVariationalProblem(self.a, rhs, adjoint, bcs=bcs0, aP=self.aP)
            LinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                    options_prefix='elliptic_adjoint_').solve()
        return adjoint

    def residual_form(self):
        """R(w; m) = a(w, .) - L as a UFL form in the current solution (depends on m via a only)."""
        return action(self.a, self.solution) - self.L

    def coefficient_gradient(self, adjoint, dJdm=None):
        """dJ/dm = dJ/dm|_explicit - lambda^T dR/dm, as a Cofunction on the coefficient space.

        The derivative of the residual with respect to the coefficient Function is taken with
        ufl.derivative and contracted with the adjoint (replace the test functions by lambda).
        dJdm: optional explicit derivative of the objective (Cofunction), added.
        """
        if not self.has_coeff():
            raise ValueError("the model has no coefficients (set treat_as_coeff in elliptic.coefficients)")
        residual = self.residual_form()
        # lambda^T R: replace the test function of the residual by the adjoint. Firedrake's
        # derivative/replace are used (not raw UFL): ufl.derivative without a direction creates a
        # plain UFL Argument, which the Firedrake assembler cannot allocate a result for.
        test = residual.arguments()[0]
        lagrangian = fd_replace(residual, {test: adjoint})
        gradient = assemble(-fd_derivative(lagrangian, self.coefficients.coeff))
        if dJdm is not None:
            gradient += dJdm
        return gradient

    # --- Model interface ----------------------------------------------------------------------

    def get_x_var_list(self):
        """Names of the solution fields: ['sigma', 'u', 'p'] (only those present)."""
        names = ['sigma'] if self.sigma_index is not None else []
        names.append('u')
        if self.num_harmonic > 0:
            names.append('p')
        return names

    def get_output_state(self):
        """(xn, xn_sub) for dimswe.output.Output after solve().

        xn is empty: the mixed solution (which may contain R blocks) is not checkpointed. xn_sub
        maps each name of get_x_var_list() to a plain, named Function in its own space (V^{k-1}
        for sigma, V^k for u and p) that shares its data with the solution, so it can be written
        to the checkpoint and to VTK. p is always a V^k Function, never an R-space function.
        """
        spaces = {'u': self.Vk, 'p': self.Vk}
        if self.sigma_index is not None:
            spaces['sigma'] = self.Vsigma
        xn_sub = {name: Function(spaces[name], val=self.sub[name].dat, name=name) for name in self.get_x_var_list()}
        return [], xn_sub

    def get_aux_var_list(self, terms='all'):
        """No auxiliary variables."""
        return []

    def get_coeff_list(self):
        """Names of the model coefficients (elliptic.coefficients with treat_as_coeff: true)."""
        return self.coefficients.get_coeff_list()

    def get_coeff_spaces(self):
        """Spaces of the model coefficients."""
        return self.coefficients.get_coeff_spaces()

    def get_coeff_var(self, name):
        """(coeff, coeff_sub, coeff_split) of a new coefficient Function, like Dynamics."""
        return self.coefficients.get_coeff_var(name)

    def get_coeff_sub(self):
        """The coefficient Functions used by the forms, by name (for output)."""
        return self.coefficients.coeff_sub

    def get_coeff_test_trial_vars(self):
        """(test, test_subs, trial, trial_subs) on the coefficient space."""
        return self.coefficients.get_coeff_test_trial_vars()

    def get_coeff_size(self):
        """Global number of coefficient dofs."""
        return self.coefficients.get_coeff_size()

    def get_coeff_bounds(self):
        """(lb, ub) over the flattened coefficient dofs (theta bounds for log parametrization)."""
        return self.coefficients.get_coeff_bounds()

    def get_coeff_scaling_factors(self):
        """Scaling factors over the flattened coefficient dofs."""
        return self.coefficients.get_coeff_scaling_factors()

    def get_numpy_coeff(self):
        """Flattened values of the coefficient Function used by the forms."""
        return self.coefficients.flatten(self.coefficients.coeff)

    def set_numpy_coeff(self, array):
        """Set the coefficient Function used by the forms (call solve() afterwards)."""
        self.coefficients.set_from_array(numpy.asarray(array, dtype=float))

    def has_aux(self):
        """No auxiliary variables."""
        return False

    def has_coeff(self):
        """True if any coefficient is a model coefficient."""
        return self.coefficients.has_coeff()

    def set_coeffs(self, parameters, coeff_sub):
        """Assign the cfg initial values of the coefficients to coeff_sub (Model interface)."""
        self.coefficients.set_coeffs(coeff_sub)

    # --- statistics ---------------------------------------------------------------------------

    def get_statistics_list(self):
        """Names of the scalar statistics (arrays of length 1)."""
        names = ['u_norm', 'p_norm', 'harmonic_orthogonality', 'num_harmonic_forms', 'ksp_iterations']
        if self.exact is not None:
            names += ['u_error']
            if self.k > 0:
                names += ['sigma_error', 'dsigma_error']
            if self.k < self.dim:
                names += ['du_error']
            names += ['p_error']
        return names

    def create_statistics(self, *args):
        """Allocate the statistics arrays (one entry each; arguments ignored)."""
        self.statistics = {name: numpy.zeros(1) for name in self.get_statistics_list()}

    def _l2(self, expr):
        """L2 norm with the error quadrature."""
        return assemble(inner(expr, expr) * self.error_dx) ** 0.5

    def reference_u(self):
        """Exact u minus its harmonic part (what u_h approximates), or None."""
        if self.exact is None:
            return None
        u_ref = self.exact['u']
        for h in self.harmonic:
            u_ref = u_ref - float(assemble(inner(self.exact['u'], h) * self.error_dx)) * h
        return u_ref

    def compute_statistics(self, step=0, stat_step=0):
        """Fill the statistics after solve(); errors only if an exact solution is known."""
        if not self.statistics:
            self.create_statistics()
        u_h = self.sub['u']
        p_h = self.sub.get('p')
        stats = self.statistics
        stats['u_norm'][0] = self._l2(u_h)
        stats['p_norm'][0] = self._l2(p_h) if p_h is not None else 0.0
        stats['harmonic_orthogonality'][0] = max([abs(l2_inner(u_h, h)) for h in self.harmonic], default=0.0)
        stats['num_harmonic_forms'][0] = self.num_harmonic
        stats['ksp_iterations'][0] = self.ksp_iterations
        if self.exact is None:
            return
        stats['u_error'][0] = self._l2(u_h - self.reference_u())
        if self.k > 0:
            sigma_h = self.sub['sigma']
            stats['sigma_error'][0] = self._l2(sigma_h - self.exact['sigma'])
            stats['dsigma_error'][0] = self._l2(self._d(sigma_h, self.k - 1) - self.exact['dsigma'])
        if self.k < self.dim:
            stats['du_error'][0] = self._l2(self._d(u_h, self.k) - self.exact['du'])
        p_exact = self.exact['p']
        if p_h is None:
            stats['p_error'][0] = self._l2(p_exact) if p_exact is not None else 0.0
        else:
            stats['p_error'][0] = self._l2(p_h - p_exact) if p_exact is not None else self._l2(p_h)

    def get_statistics(self):
        """Dict statistic name -> numpy array."""
        return self.statistics

    # --- diagnostics --------------------------------------------------------------------------

    def get_diagnostics_list(self):
        """Names of the diagnostic fields (for VTK/checkpoint output)."""
        names = []
        if self.exact is not None:
            names += ['u_exact', 'u_error']
            if self.k > 0:
                names += ['sigma_error']
        if self.k < self.dim:
            names += ['du']
        return names

    def create_diagnostics(self, *args):
        """Allocate the diagnostic Functions (arguments ignored)."""
        spaces = {'u_exact': self.Vk, 'u_error': self.Vk}
        if self.k > 0:
            spaces['sigma_error'] = self.Vsigma
        if self.k < self.dim:
            spaces['du'] = form_space(self.spaces, self.k + 1, self.dim, self.complex)
        self.diagnostics = {name: Function(spaces[name], name=name) for name in self.get_diagnostics_list()}

    def compute_diagnostics(self):
        """Fill the diagnostics after solve() (interpolated, or projected where needed)."""
        if not self.diagnostics:
            self.create_diagnostics()
        diag = self.diagnostics
        if 'u_exact' in diag:
            interpolate_or_project(self.reference_u(), self.Vk, target=diag['u_exact'])
            # sub['u'] lives in W.sub(u_index), not in Vk itself, so do not use Function.assign.
            interpolate_or_project(self.sub['u'] - diag['u_exact'], self.Vk, target=diag['u_error'])
        if 'sigma_error' in diag:
            interpolate_or_project(self.sub['sigma'] - self.exact['sigma'], self.Vsigma, target=diag['sigma_error'])
        if 'du' in diag:
            interpolate_or_project(self._d(self.sub['u'], self.k), diag['du'].function_space(), target=diag['du'])

    def get_diagnostics(self):
        """Dict diagnostic name -> Function."""
        return self.diagnostics
