"""Hodge Laplacian eigenproblems for k-forms (TODO-elliptic §4.11), with the variable coefficients of §4.10.

Find lambda and (sigma, u) != 0 with (in the symmetric sign convention of dimswe.elliptic)

    -(sigma, tau) + (u, d tau)              = 0
     (d sigma, v) + (d u, d v)              = lambda (u, v)

i.e. the generalized problem A x = lambda M x, where A is the symmetric indefinite operator of
dimswe.elliptic.hodge_bilinear_form and M is the u-block mass (singular on the sigma block).
With coefficients (elliptic.coefficients) A is the weighted operator
-(mu sigma, tau) + (u, d tau) + (d sigma, v) + (kappa d u, d v) + (c u, v); M stays unweighted.
Eliminating sigma gives (D M_sigma^{-1} D^T + K) u = lambda M_u u, so lambda >= 0; the "infinite"
eigenvalues of the singular M are mapped to 0 by shift-and-invert, far from the target.

Only homogeneous BCs: essential ones are DirichletBCs on sigma and u (restricted out of the
problem, or shifted far away with bc_shift), natural ones need no terms; periodic directions are
handled by the mesh. Harmonic forms are the eigenvalue-0 eigenvectors (0, h); they are either
kept (their count is reported and compared with the Kunneth rule of §1.4) or deflated.

exact_box_eigenvalues gives the exact spectrum on boxes with one BC type per face (pure Python).
"""
from dataclasses import dataclass
from itertools import combinations
from math import pi
from typing import Optional

import numpy
from firedrake import Function, LinearEigenproblem, LinearEigensolver, assemble, dx, inner

from .boundary_conditions import BC_TYPES, FACE_LOCATIONS
from .elliptic import (EllipticConfig, build_block_layout, build_essential_bc_lists, build_harmonic_basis,
                       hodge_bilinear_form, riesz_form, split_arguments)
from .elliptic_solvers import eigen_inner_solver_options
from .numerical_harmonic import physical_eigenpairs
from .elliptic_coefficients import EllipticCoefficients
from .elliptic_problems import get_elliptic_problem
from .logger import EmptyLogger
from .meshes import DIRECTIONS, get_mesh_and_spaces


# ---------------------------------------------------------------------------------------------
# Exact spectra on boxes (pure Python)
# ---------------------------------------------------------------------------------------------

def _direction_kinds(dim, periodic_dirs, face_bcs):
    """Per direction: 'periodic', 'essential', 'natural' or 'mixed' (different types on the two faces).

    face_bcs is per face ({'left': 'essential', ...}) or per direction ({'x': (t0, t1), ...}).
    """
    periodic = set(periodic_dirs)
    by_dir = {}
    if all(key in DIRECTIONS for key in face_bcs):
        by_dir = {d: tuple(t) for d, t in face_bcs.items()}
    else:
        for name, bc in face_bcs.items():
            direction, side = FACE_LOCATIONS[name]
            by_dir.setdefault(direction, [None, None])[side] = bc
    kinds = {}
    for d in DIRECTIONS[:dim]:
        if d in periodic:
            if d in by_dir:
                raise ValueError("boundary conditions given for periodic direction " + d)
            kinds[d] = 'periodic'
            continue
        types = tuple(by_dir.get(d, (None, None)))
        if None in types or any(t not in BC_TYPES for t in types):
            raise ValueError("direction " + d + " needs an essential/natural BC on both faces, got " + str(types))
        kinds[d] = types[0] if types[0] == types[1] else 'mixed'
    return kinds


def direction_modes(length, kind, in_index_set, count):
    """The `count` smallest 1D eigenvalues (with multiplicity) of one direction's factor.

    For the component dx^I and direction x_i (TODO-elliptic §4.11):
    - periodic: (2 pi m / L)^2, m >= 0, one mode at m = 0 and two (cos, sin) otherwise;
    - both faces essential: i in I -> cos, m >= 0; i not in I -> sin, m >= 1, value (m pi / L)^2;
    - both faces natural: i in I -> sin, m >= 1; i not in I -> cos, m >= 0;
    - mixed (essential on one face, natural on the other): quarter waves ((m + 1/2) pi / L)^2.
    """
    values = []
    m = 0
    while len(values) < count:
        if kind == 'periodic':
            values.extend([(2 * pi * m / length) ** 2] * (1 if m == 0 else 2))
        elif kind == 'mixed':
            values.append(((m + 0.5) * pi / length) ** 2)
        else:
            starts_at_zero = (kind == 'essential') == in_index_set  # cos modes include m = 0
            if m > 0 or starts_at_zero:
                values.append((m * pi / length) ** 2)
        m += 1
    return values[:count]


def exact_box_eigenvalues(k, dim, lengths, periodic_dirs, face_bcs, n):
    """The n smallest Hodge Laplacian eigenvalues of k-forms on a box, with multiplicities.

    Eigenforms separate into components f(x) dx^I with |I| = k and f a product of 1D factors
    (direction_modes), so lambda = sum_i lambda_i and the spectrum is the multiset union over I.
    The result does not depend on the 2D complex (the proxies are isometries).

    Args:
        k, dim: form degree and dimension.
        lengths: box side lengths (L_x, ...).
        periodic_dirs: periodic directions.
        face_bcs: BC type per bounded face, or per direction (as in harmonic_forms).
        n: number of eigenvalues.

    Returns:
        sorted list of the n smallest eigenvalues (floats), repeated by multiplicity.
    """
    if k < 0 or k > dim:
        raise ValueError("form degree " + str(k) + " is out of range for dimension " + str(dim))
    kinds = _direction_kinds(dim, periodic_dirs, face_bcs)
    spectrum = []
    for I in combinations(DIRECTIONS[:dim], k):
        factors = [direction_modes(lengths[i], kinds[d], d in I, n) for i, d in enumerate(DIRECTIONS[:dim])]
        sums = [0.0]
        for factor in factors:
            sums = sorted(s + f for s in sums for f in factor)[:n]
        spectrum.extend(sums)
    return sorted(spectrum)[:n]


def constant_scalar_coefficients(coefficients):
    """{'mu', 'kappa', 'c'} values if every coefficient is a constant scalar, else None.

    coefficients: parsed elliptic.coefficients (EllipticConfig.coefficients). Parameters
    (treat_as_coeff) count at their initial values; problem/field kinds and tensors give None.
    """
    values = {'mu': 1.0, 'kappa': 1.0, 'c': 0.0}
    for name, coef in coefficients.items():
        if coef.kind != 'constant' or coef.rank != 'scalar':
            return None
        values[name] = float(coef.value)
    return values


def constant_coefficient_spectrum(k, dim, lengths, periodic_dirs, face_bcs, n, mu=1.0, kappa=1.0, c=0.0):
    """Exact spectrum of -(mu sigma, tau) + ... + (kappa du, dv) + (c u, v) = lambda (u, v) on a box.

    For constant scalar mu, kappa, c the eigenforms are those of the unweighted problem:
    - k = 0 (only d u): lambda = kappa lambda_0 + c;
    - k = dim (only delta u): lambda = lambda_0 / mu + c;
    - 0 < k < dim: lambda_0 = |d u|^2 + |delta u|^2 splits into exact and coexact modes, which a
      box spectrum does not separate; only kappa = 1/mu has a closed form (lambda_0 kappa + c).
    Returns the n smallest eigenvalues, or None when there is no closed form.
    """
    base = exact_box_eigenvalues(k, dim, lengths, periodic_dirs, face_bcs, n)
    if k == 0:
        scale = kappa
    elif k == dim:
        scale = 1.0 / mu
    elif abs(kappa * mu - 1.0) <= 1e-12:
        scale = kappa
    else:
        return None
    return sorted(scale * value + c for value in base)


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------

@dataclass
class EigenConfig:
    """The 'elliptic: eigen' cfg subsection.

    Attributes:
        nev: number of eigenpairs to report.
        ncv: Krylov subspace size (None or too small: max(2 m, m + 16) with m = num_requested()).
        target: shift-and-invert target. The default -1 sits just below the spectrum (lambda >= 0)
            so the smallest eigenvalues, including the zero ones, are found while A - target M
            stays non-singular (target 0 would make it singular when harmonic forms exist).
        tolerance: SLEPc convergence tolerance.
        max_it: SLEPc iteration limit (None: default).
        deflate_harmonic: deflate the known harmonic forms (0, h) instead of computing them.
        zero_tolerance: |lambda| below this counts as a zero eigenvalue.
        bc_shift: eigenvalue given to essential-BC rows when the problem is not restricted
            (Firedrake LinearEigenproblem), chosen far above any target window.
    """
    nev: int = 20
    ncv: Optional[int] = None
    target: float = -1.0
    tolerance: float = 1e-10
    max_it: Optional[int] = None
    deflate_harmonic: bool = False
    zero_tolerance: float = 1e-6
    bc_shift: float = 1e8
    inner_solver: str = 'direct'
    inner_rtol: float = 1e-12
    inner_max_it: int = 1000

    @classmethod
    def from_parameters(cls, parameters):
        """Read and validate parameters['elliptic']['eigen'] (all entries optional)."""
        cfg = (parameters.get('elliptic', {}) or {}).get('eigen', None) or {}
        config = cls(nev=int(cfg.get('nev', 20)),
                     ncv=None if cfg.get('ncv') is None else int(cfg['ncv']),
                     target=float(cfg.get('target', -1.0)),
                     tolerance=float(cfg.get('tolerance', 1e-10)),
                     max_it=None if cfg.get('max_it') is None else int(cfg['max_it']),
                     deflate_harmonic=bool(cfg.get('deflate_harmonic', False)),
                     zero_tolerance=float(cfg.get('zero_tolerance', 1e-6)),
                     bc_shift=float(cfg.get('bc_shift', 1e8)),
                     inner_solver=str(cfg.get('inner_solver', 'direct')),
                     inner_rtol=float(cfg.get('inner_rtol', 1e-12)),
                     inner_max_it=int(cfg.get('inner_max_it', 1000)))
        if config.inner_solver not in ('direct', 'iterative'):
            raise ValueError("elliptic.eigen.inner_solver must be 'direct' or 'iterative'")
        if config.nev < 1:
            raise ValueError("elliptic.eigen.nev must be >= 1")
        if config.ncv is not None and config.ncv < config.nev:
            raise ValueError("elliptic.eigen.ncv must be >= nev")
        if config.tolerance <= 0.0 or config.zero_tolerance <= 0.0:
            raise ValueError("elliptic.eigen tolerances must be positive")
        return config

    def num_requested(self):
        """Eigenpairs requested from SLEPc: nev plus a margin for repeated eigenvalues."""
        return self.nev + max(4, self.nev // 2)

    def subspace_size(self):
        """Krylov subspace size: the configured ncv if large enough, else max(2 m, m + 16)."""
        m = self.num_requested()
        if self.ncv is not None and self.ncv >= m:
            return self.ncv
        return max(2 * m, m + 16)

    def solver_parameters(self, inner_options=None):
        """SLEPc options: Krylov-Schur, generalized non-Hermitian, shift-and-invert.

        The inner solve is MUMPS LU by default; inner_options (st_* keys, e.g. from
        elliptic_solvers.eigen_inner_solver_options) replace it for inner_solver: iterative.
        """
        params = {
            'eps_type': 'krylovschur',
            'eps_gen_non_hermitian': None,
            'eps_target': self.target,
            'eps_target_magnitude': None,
            'eps_tol': self.tolerance,
            'st_type': 'sinvert',
            'st_ksp_type': 'preonly',
            'st_pc_type': 'lu',
            'st_pc_factor_mat_solver_type': 'mumps',
        }
        if self.max_it is not None:
            params['eps_max_it'] = self.max_it
        if inner_options is not None:
            for key in ('st_ksp_type', 'st_pc_type', 'st_pc_factor_mat_solver_type'):
                params.pop(key)
            params.update(inner_options)
        return params


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

class HodgeLaplacianEigenproblem():
    """Time-invariant model computing Hodge Laplacian eigenpairs (model.type: elliptic-eigen).

    Uses the 'elliptic' cfg section for the form degree, complex, BCs and harmonic forms (the
    boundary data are always homogeneous), and 'elliptic: eigen' for the eigensolver; the
    'elliptic-problem' section only provides the domain. After solve(), mode i is written as
    output step i (select_output_step), and the eigenvalues are statistics.
    """

    #: The driver uses this to skip time stepping.
    is_time_invariant = True

    def __init__(self, parameters, logger=None, mesh=None):
        """Build spaces, forms and BCs (no solve yet)."""
        self.parameters = parameters
        self.logger = logger if logger is not None else EmptyLogger()
        self.problem = get_elliptic_problem(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.problem, mesh=mesh)
        self.config = EllipticConfig.from_parameters(parameters, self.spaces)
        self.eigen = EigenConfig.from_parameters(parameters)
        self.dim, self.k, self.complex = self.config.dim, self.config.k, self.config.complex
        self.dx = dx(degree=self.config.quadrature_degree)

        self.layout = build_block_layout(self.spaces, self.k, self.dim, self.complex)
        self.W, self.Vk, self.Vsigma = self.layout.W, self.layout.Vk, self.layout.Vsigma
        self.trial, self.test = split_arguments(self.layout)
        # Variable coefficients (TODO-elliptic §4.10): the same weights as the source problem.
        self.coefficients = EllipticCoefficients(self.config.coefficients, self.mesh, self.problem,
                                                 self.k, self.dim, self.complex)
        self.weights = self.coefficients.model_weights()
        # c > 0 everywhere: no zero eigenvalues, so no harmonic forms to count or deflate.
        self.reaction_positive = self.coefficients.reaction_is_positive()
        self.harmonic_index_sets, self.harmonic = build_harmonic_basis(self.config, self.mesh,
                                                                       parameters['mesh']['type'], self.Vk, self.dx,
                                                                       reaction_positive=self.reaction_positive,
                                                                       spaces=self.spaces, parameters=parameters,
                                                                       logger=self.logger)
        self.num_harmonic = len(self.harmonic)
        bcs_sigma, bcs_u = build_essential_bc_lists(self.layout, self.config)
        self.bcs = bcs_sigma + bcs_u
        self.a = hodge_bilinear_form(self.trial, self.test, self.k, self.dim, self.complex, self.dx, self.weights)
        self.m = inner(self.trial['u'], self.test['u']) * self.dx

        self.eigenvalues = []
        self.imaginary_parts = []
        self.residuals = []
        self.modes = []
        self.num_converged = 0
        self.statistics = {}
        self.output_fields = {}

    # --- solve --------------------------------------------------------------------------------

    def _make_problem(self, restrict):
        """LinearEigenproblem, restricted to the free dofs or not (bc_shift), as requested.

        restrict is always passed explicitly: newer Firedrake restricts by default, which breaks
        anything working on the full space (deflation vectors, the Riesz preconditioner matrix of
        the iterative inner solve). Older Firedrake without the argument only does bc_shift.
        """
        kwargs = {'M': self.m, 'bcs': self.bcs or None, 'bc_shift': self.eigen.bc_shift}
        try:
            return LinearEigenproblem(self.a, restrict=bool(restrict), **kwargs)
        except TypeError:
            return LinearEigenproblem(self.a, **kwargs)  # older Firedrake: no restrict argument

    def _deflation_vectors(self):
        """PETSc vectors of the harmonic eigenvectors (sigma, u) = (0, h_i) in W."""
        vectors = []
        for h in self.harmonic:
            f = Function(self.W)
            block = f if self.layout.num_blocks == 1 else f.subfunctions[self.layout.u_index]
            block.dat.data[:] = h.dat.data_ro
            with f.dat.vec_ro as v:
                vectors.append(v.copy())
        return vectors

    def solve(self):
        """Compute the eigenpairs closest to the target; fills eigenvalues, residuals and modes.

        SLEPc is asked for nev + extra pairs (EigenConfig.num_requested) with a generous subspace,
        and the smallest nev are kept: requesting exactly nev can miss copies of repeated
        eigenvalues at the end of the window.
        """
        deflate = self.eigen.deflate_harmonic and self.num_harmonic > 0
        iterative = self.eigen.inner_solver == 'iterative'
        # Deflation vectors and the iterative inner solve's field index sets live in the full
        # space, so the problem is not restricted then (essential BCs use bc_shift).
        problem = self._make_problem(restrict=not (deflate or iterative))
        nev = self.eigen.nev
        requested = self.eigen.num_requested()
        inner = None
        if iterative:
            inner, _ = eigen_inner_solver_options(self.k, self.dim, self.complex, rtol=self.eigen.inner_rtol,
                                                  max_it=self.eigen.inner_max_it)
        solver = LinearEigensolver(problem, n_evals=requested, solver_parameters=self.eigen.solver_parameters(inner),
                                   ncv=self.eigen.subspace_size(), options_prefix='elliptic_eigen_')
        if deflate:
            solver.es.setDeflationSpace(self._deflation_vectors())
        if iterative:
            self._setup_iterative_inner_solve(solver)
        self.num_converged = solver.solve()
        entries = []
        for i in range(self.num_converged):
            value = complex(solver.eigenvalue(i))
            real, _ = solver.eigenfunction(i)
            entries.append((value.real, value.imag, float(solver.es.computeError(i)), real.copy(deepcopy=True)))
        # Drop the spurious "infinite" eigenvalues of the singular pencil (pure sigma modes, possibly
        # returned as huge values of either sign), then smallest first; keep at most nev.
        physical = physical_eigenpairs([(e[0], e[3]) for e in entries], self.layout, self.dx)
        kept = {id(mode) for _, mode in physical}
        self.num_spurious = len(entries) - len(physical)
        entries = sorted((e for e in entries if id(e[3]) in kept), key=lambda e: e[0])
        entries = entries[:nev]
        self.eigenvalues = [e[0] for e in entries]
        self.imaginary_parts = [e[1] for e in entries]
        self.residuals = [e[2] for e in entries]
        self.modes = [self._normalized(e[3]) for e in entries]
        self.logger.output('elliptic-eigen: ' + str(self.num_converged) + ' converged; eigenvalues '
                           + str(numpy.round(self.eigenvalues, 6).tolist()), 0)
        return self.eigenvalues

    def _setup_iterative_inner_solve(self, solver):
        """Riesz-map preconditioner for the shift-and-invert inner solve (TODO-elliptic §4.12).

        The ST's KSP solves (A - target M) x = b. Its preconditioner matrix is the assembled,
        weighted Riesz matrix (riesz_form: SPD, with the essential BC rows set to the identity), and
        for mixed problems an additive fieldsplit is set up from the field index sets of W (SLEPc's
        KSP has no Firedrake DM, so the fields cannot be inferred). The block solvers are algebraic
        (elliptic_solvers.algebraic_block_pc_options).
        """
        self._st_riesz = assemble(riesz_form(self.trial, self.test, self.k, self.dim, self.complex, self.dx,
                                             self.weights), bcs=self.bcs or None, mat_type='aij')
        st = solver.es.getST()
        st.setPreconditionerMat(self._st_riesz.petscmat)
        if self.layout.num_blocks > 1:
            pc = st.getKSP().getPC()
            pc.setType('fieldsplit')
            pc.setFieldSplitIS(*[(str(i), iset) for i, iset in enumerate(self.W.dof_dset.field_ises)])

    def _normalized(self, mode):
        """Copy of a mode scaled to unit L2 norm of its u part (in a space without BC lifting)."""
        u = mode if self.layout.num_blocks == 1 else mode.subfunctions[self.layout.u_index]
        norm = assemble(inner(u, u) * self.dx) ** 0.5
        if norm > 0.0:
            mode.assign(mode / norm)
        return mode

    def num_zero_eigenvalues(self):
        """Number of computed eigenvalues with |lambda| < zero_tolerance."""
        return sum(1 for value in self.eigenvalues if abs(value) < self.eigen.zero_tolerance)

    def exact_eigenvalues(self):
        """Exact spectrum on the box, or None ('file' meshes, or coefficients without a closed form).

        Without the zeros when the harmonic forms are deflated. Constant scalar coefficients have
        a closed form (constant_coefficient_spectrum).
        """
        bounds = getattr(self.mesh, 'domain_bounds', None)
        if bounds is None:
            return None
        constants = constant_scalar_coefficients(self.config.coefficients)
        if constants is None:
            return None
        lengths = [hi - lo for lo, hi in bounds]
        n = len(self.eigenvalues) + self.num_harmonic
        exact = constant_coefficient_spectrum(self.k, self.dim, lengths, self.mesh.periodic_directions,
                                              self.config.face_bcs, max(n, 1), **constants)
        if exact is None:
            return None
        if self.eigen.deflate_harmonic and self.num_harmonic > 0:
            exact = exact[self.num_harmonic:]
        return exact[:len(self.eigenvalues)]

    # --- Model interface ----------------------------------------------------------------------

    def get_x_var_list(self):
        """Mode fields: ['sigma', 'u'] (only 'u' for k = 0)."""
        return (['sigma'] if self.layout.sigma_index is not None else []) + ['u']

    def get_aux_var_list(self, terms='all'):
        """No auxiliary variables."""
        return []

    def get_coeff_list(self):
        """Coefficients are fixed for eigenproblems (not optimization parameters)."""
        return []

    def has_aux(self):
        """No auxiliary variables."""
        return False

    def has_coeff(self):
        """No model coefficients."""
        return False

    def set_coeffs(self, parameters, coeff_sub):
        """Nothing to set."""
        pass

    def num_output_steps(self):
        """One output step per computed mode."""
        return len(self.modes)

    def get_output_state(self):
        """(xn, xn_sub) for Output: plain named Functions that select_output_step fills per mode."""
        spaces = {'u': self.Vk}
        if self.layout.sigma_index is not None:
            spaces['sigma'] = self.Vsigma
        self.output_fields = {name: Function(spaces[name], name=name) for name in self.get_x_var_list()}
        return [], self.output_fields

    def select_output_step(self, i):
        """Copy mode i into the output Functions."""
        if not self.output_fields:
            self.get_output_state()
        mode = self.modes[i]
        for name, index in (('sigma', self.layout.sigma_index), ('u', self.layout.u_index)):
            if name in self.output_fields:
                block = mode if self.layout.num_blocks == 1 else mode.subfunctions[index]
                self.output_fields[name].dat.data[:] = block.dat.data_ro

    # --- statistics and diagnostics -------------------------------------------------------------

    def get_statistics_list(self):
        """Eigenvalue statistics (arrays over modes) and scalar counts (arrays of length 1)."""
        names = ['eigenvalues', 'eigenvalue_imag', 'eigen_residuals', 'num_converged', 'num_zero_eigenvalues',
                 'num_harmonic_forms', 'deflated']
        if getattr(self.mesh, 'domain_bounds', None) is not None \
                and constant_scalar_coefficients(self.config.coefficients) is not None:
            names += ['exact_eigenvalues', 'eigenvalue_errors']
        return names

    def create_statistics(self, *args):
        """Allocate the statistics (filled by compute_statistics)."""
        self.statistics = {name: numpy.zeros(1) for name in self.get_statistics_list()}

    def compute_statistics(self, step=0, stat_step=0):
        """Fill the statistics after solve(). eigenvalue_errors = |lambda_h - lambda| / max(1, lambda)."""
        if not self.statistics:
            self.create_statistics()
        stats = self.statistics
        stats['eigenvalues'] = numpy.array(self.eigenvalues)
        stats['eigenvalue_imag'] = numpy.array(self.imaginary_parts)
        stats['eigen_residuals'] = numpy.array(self.residuals)
        stats['num_converged'] = numpy.array([self.num_converged])
        stats['num_zero_eigenvalues'] = numpy.array([self.num_zero_eigenvalues()])
        stats['num_harmonic_forms'] = numpy.array([self.num_harmonic])
        stats['deflated'] = numpy.array([1 if self.eigen.deflate_harmonic and self.num_harmonic > 0 else 0])
        exact = self.exact_eigenvalues() if 'exact_eigenvalues' in self.get_statistics_list() else None
        if exact is not None:
            computed = numpy.array(self.eigenvalues[:len(exact)])
            exact = numpy.array(exact)
            stats['exact_eigenvalues'] = exact
            stats['eigenvalue_errors'] = numpy.abs(computed - exact) / numpy.maximum(1.0, numpy.abs(exact))

    def get_statistics(self):
        """Dict statistic name -> numpy array."""
        return self.statistics

    def get_diagnostics_list(self):
        """No diagnostic fields (the modes are the output fields)."""
        return []

    def create_diagnostics(self, *args):
        """Nothing to allocate."""
        pass

    def compute_diagnostics(self):
        """Nothing to compute."""
        pass

    def get_diagnostics(self):
        """No diagnostics."""
        return {}
