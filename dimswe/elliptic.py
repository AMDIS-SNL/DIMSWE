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
from firedrake import (DirichletBC, Function, LinearSolver, LinearVariationalProblem,
                       LinearVariationalSolver, MixedFunctionSpace, TestFunction, TestFunctions,
                       TrialFunction, TrialFunctions, action, assemble, dx, inner)

from .boundary_conditions import (build_essential_bcs, build_natural_terms, face_bcs_by_direction,
                                  faces_of_type, get_boundary_registry, parse_boundary_conditions)
from .differential_forms import exterior_derivative, form_space, normalize_complex
from .elliptic_problems import BOUNDARY_DATA_MODES, collect_boundary_data, get_elliptic_problem
from .harmonic_forms import (HARMONIC_FORM_MODES, build_harmonic_nullspace, combine_harmonic_forms,
                             get_harmonic_index_sets, harmonic_basis, harmonic_constraint_form,
                             harmonic_multiplier_space, interpolate_or_project, l2_inner,
                             lagrange_preconditioner_form, lagrange_solver_parameters,
                             multiplier_coefficients, orthonormalize_harmonic_basis,
                             project_rhs_orthogonal_to_harmonic, remove_harmonic_component)
from .logger import EmptyLogger
from .meshes import get_mesh_and_spaces, get_mesh_type_dimension
from .parameters import elliptic_solver_parameters as ELLIPTIC_SOLVERS

# Allowed values of elliptic.harmonic_constraint and elliptic.solver.
HARMONIC_CONSTRAINTS = ('lagrange', 'nullspace')
SOLVERS = tuple(ELLIPTIC_SOLVERS)
# FEEC families currently available in DeRhamComplex (others: TODO-elliptic §4.9).
SUPPORTED_FAMILIES = ('Q',)


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
    registry: Optional[dict] = None
    face_bcs: Optional[dict] = None

    @classmethod
    def from_parameters(cls, parameters, spaces=None):
        """Build and validate the config; with spaces (a DeRhamComplex) the BCs are parsed too.

        Raises ValueError for: k out of range, invalid 2D complex, lump_mass: true, unsupported
        family or cell type, invalid option values, and (with spaces) BCs on periodic faces or
        missing BCs on bounded faces.
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
        if disc['family'] not in SUPPORTED_FAMILIES:
            raise ValueError("FEEC family " + str(disc['family']) + " is not available for elliptic problems yet (TODO-elliptic §4.9)")
        if disc['family'] == 'Q' and mesh_cfg.get('simplicial_cells', False):
            raise ValueError("family Q needs quadrilateral/hexahedral cells (simplicial_cells: false)")
        order = int(disc['order'])
        if order < 1:
            raise ValueError("spatial-discretization.order must be >= 1")

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
                     lagrange_regularization=float(cfg.get('lagrange_regularization', 1.0)))
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
        by_direction = face_bcs_by_direction(self.config.registry, self.config.face_bcs)
        self.harmonic_index_sets = get_harmonic_index_sets(self.config.harmonic_forms, self.k, self.dim,
                                                           self.mesh.periodic_directions, by_direction,
                                                           self.parameters['mesh']['type'])
        self.harmonic = orthonormalize_harmonic_basis(
            harmonic_basis(self.k, self.dim, self.complex, self.harmonic_index_sets), self.Vk, measure=self.dx)
        self.num_harmonic = len(self.harmonic)
        self.logger.output('elliptic: ' + str(self.num_harmonic) + ' harmonic ' + str(self.k) + '-forms '
                           + str(self.harmonic_index_sets), 0)

    def _build_spaces(self):
        """Blocks [sigma (k > 0), u, p_1..p_m (lagrange with harmonic forms)] and the space W."""
        spaces = []
        self.sigma_index = self.u_index = None
        self.p_indices = []
        if self.k > 0:
            self.Vsigma = form_space(self.spaces, self.k - 1, self.dim, self.complex)
            self.sigma_index = len(spaces)
            spaces.append(self.Vsigma)
        self.u_index = len(spaces)
        spaces.append(self.Vk)
        self.use_multiplier = self.config.harmonic_constraint == 'lagrange' and self.num_harmonic > 0
        if self.use_multiplier:
            # One scalar R block per harmonic form (Firedrake has no vector-valued R arguments).
            self.R = harmonic_multiplier_space(self.mesh, self.num_harmonic)
            self.p_indices = list(range(len(spaces), len(spaces) + len(self.R)))
            spaces.extend(self.R)
        self.block_spaces = spaces
        self.W = spaces[0] if len(spaces) == 1 else MixedFunctionSpace(spaces)
        self.num_blocks = len(spaces)

    def _build_arguments(self):
        """Trial/test functions per block: self.trial / self.test dicts with keys sigma, u, p."""
        if self.num_blocks == 1:
            trials, tests = (TrialFunction(self.W),), (TestFunction(self.W),)
        else:
            trials, tests = TrialFunctions(self.W), TestFunctions(self.W)
        self.trial, self.test = {}, {}
        for name, index in (('sigma', self.sigma_index), ('u', self.u_index)):
            if index is not None:
                self.trial[name], self.test[name] = trials[index], tests[index]
        if self.p_indices:
            # Lists over the scalar R blocks.
            self.trial['p'] = [trials[i] for i in self.p_indices]
            self.test['p'] = [tests[i] for i in self.p_indices]

    def _build_bcs(self):
        """DirichletBCs on the essential faces: tr sigma = g_sigma and tr u = g_u (§1.6)."""
        faces = faces_of_type(self.config.registry, self.config.face_bcs, 'essential')
        u_index = self.u_index if self.k < self.dim else None  # DG u has no trace
        self.bcs_sigma = build_essential_bcs(self.W, self.sigma_index, None, faces,
                                             g_sigma=self.boundary_data.essential_sigma)
        self.bcs_u = build_essential_bcs(self.W, None, u_index, faces, g_u=self.boundary_data.essential_u)
        self.bcs = self.bcs_sigma + self.bcs_u

    def _d(self, w, j):
        """Exterior derivative of a j-form (None for j = dim)."""
        return exterior_derivative(w, j, self.dim, self.complex)

    def _mass_sigma(self, sigma, tau):
        """-(sigma, tau) (sign chosen for a symmetric operator)."""
        return -inner(sigma, tau) * self.dx

    def _coupling(self, sigma, tau, u, v):
        """(u, d tau) + (d sigma, v)."""
        return (inner(u, self._d(tau, self.k - 1)) + inner(self._d(sigma, self.k - 1), v)) * self.dx

    def _stiffness(self, u, v):
        """(d u, d v), or 0 for k = dim."""
        if self.k == self.dim:
            return 0
        return inner(self._d(u, self.k), self._d(v, self.k)) * self.dx

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
        u, v = self.trial['u'], self.test['u']
        a = self._stiffness(u, v)
        if self.k > 0:
            sigma, tau = self.trial['sigma'], self.test['sigma']
            a = _add(a, self._mass_sigma(sigma, tau))
            a = _add(a, self._coupling(sigma, tau, u, v))
        return a

    def _riesz_form(self):
        """SPD Riesz-map preconditioner (sigma, tau) + (d sigma, d tau) + (u, v) + (d u, d v)."""
        u, v = self.trial['u'], self.test['u']
        aP = inner(u, v) * self.dx
        aP = _add(aP, self._stiffness(u, v))
        if self.k > 0:
            sigma, tau = self.trial['sigma'], self.test['sigma']
            aP = aP + (inner(sigma, tau) + inner(self._d(sigma, self.k - 1), self._d(tau, self.k - 1))) * self.dx
        return aP

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

    # --- Model interface ----------------------------------------------------------------------

    def get_x_var_list(self):
        """Names of the solution fields: ['sigma', 'u', 'p'] (only those present)."""
        names = ['sigma'] if self.sigma_index is not None else []
        names.append('u')
        if self.num_harmonic > 0:
            names.append('p')
        return names

    def get_aux_var_list(self, terms='all'):
        """No auxiliary variables."""
        return []

    def get_coeff_list(self):
        """No model coefficients yet (TODO-elliptic §4.10)."""
        return []

    def has_aux(self):
        """No auxiliary variables."""
        return False

    def has_coeff(self):
        """No model coefficients yet."""
        return False

    def set_coeffs(self, parameters, coeff_sub):
        """Nothing to set until §4.10."""
        pass

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
