"""Problem data for the elliptic Hodge Laplacian problems (TODO-elliptic §4.5).

Mirrors dimswe/initial_conditions.py: get_elliptic_problem(parameters) dispatches on
parameters['elliptic-problem']['name'] to a class that provides
- the box-shaped domain (Lx, Ly, Lz, xc, yc, zc), so the object can be passed to
  dimswe.meshes.get_mesh_and_spaces in place of initial conditions;
- the source f (get_source) and, when known, the exact solution (get_exact);
- per-face boundary data (get_essential_data, get_natural_data), by default derived from the
  exact solution.

Conventions
- Problems are written in normalized coordinates s_i = (x_i - x_i0) / (x_i1 - x_i0) in [0, 1]
  taken from mesh.domain_bounds, so they do not depend on where the mesh is placed.
- Forms are proxies as in dimswe/differential_forms.py: scalars for k = 0 and k = dim, vectors
  otherwise; in 2D the 1-form proxy depends on the complex.
- Boundary data are full fields (the traces/pairings are applied by
  dimswe/boundary_conditions.py): essential data are (g_u, g_sigma) = (u, delta u), natural data
  are (g_u, g_du) = (u, d u).

Compatibility: for any smooth exact solution with exact boundary data the harmonic part of the
right-hand side vanishes (the natural-face terms cancel; on essential faces the BCs are imposed
and the harmonic forms have zero trace), so p = 0. MMS problems are additionally built with zero
mean in every component, so u is orthogonal to every constant (harmonic) form and u_h can be
compared with u directly. IncompatibleSourceProblem adds a harmonic component to f on purpose.
"""
from dataclasses import dataclass, field
from math import pi
from typing import Dict

from firedrake import Constant, SpatialCoordinate, as_vector, exp, sin

from .differential_forms import (codifferential, constant_form_proxy, exterior_derivative,
                                 hodge_laplacian_expr, is_scalar_form, normalize_complex)

# Allowed values of the elliptic 'boundary_data' option.
BOUNDARY_DATA_MODES = ('exact', 'problem', 'homogeneous')


def get_elliptic_problem(parameters):
    """Return the problem-data object named by parameters['elliptic-problem']['name']."""
    cfg = parameters.get('elliptic-problem', {}) or {}
    name = cfg.get('name', 'sine')
    problems = {
        'sine': SineProblem,
        'polynomial': PolynomialProblem,
        'gaussian-source': GaussianSourceProblem,
        'boundary-driven': BoundaryDrivenProblem,
        'incompatible-source': IncompatibleSourceProblem,
    }
    if name not in problems:
        raise ValueError("unknown elliptic problem " + str(name) + " (known: " + str(sorted(problems)) + ")")
    return problems[name](parameters)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _form_from_components(components, k, dim):
    """Wrap a list of dim scalar expressions as a k-form proxy (first entry for scalar forms)."""
    if is_scalar_form(k, dim):
        return components[0]
    return as_vector(components[:dim])


def _sum(terms):
    """Sum of a non-empty list of UFL expressions."""
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


def _product(factors):
    """Product of a non-empty list of UFL expressions."""
    result = factors[0]
    for factor in factors[1:]:
        result = result * factor
    return result


def zero_form(k, dim):
    """A zero k-form proxy that survives in forms.

    UFL folds literal zeros (0.0 * expr) into a domain-less Zero, which would turn inner(f, v)*dx
    into an empty form; a firedrake Constant is kept as a coefficient instead.
    """
    return Constant(0.0) if is_scalar_form(k, dim) else Constant(tuple([0.0] * dim))


# ---------------------------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------------------------

class EllipticProblemData():
    """Base class for elliptic problem data.

    Subclasses implement either exact_u (manufactured solutions; everything else is derived) or
    override source (problems without an exact solution, has_exact = False).
    """

    #: True if get_exact returns an exact solution.
    has_exact = True

    def __init__(self, parameters):
        """Read the domain from parameters['elliptic-problem'] (all entries optional).

        Lx, Ly, Lz default to 1; the centers xc, yc, zc default to L/2, so bounded meshes cover
        [0, L] like the periodic ones.
        """
        self.parameters = parameters
        self.cfg = parameters.get('elliptic-problem', {}) or {}
        self.Lx = float(self.cfg.get('Lx', 1.0))
        self.Ly = float(self.cfg.get('Ly', 1.0))
        self.Lz = float(self.cfg.get('Lz', 1.0))
        self.xc = float(self.cfg.get('xc', self.Lx / 2.))
        self.yc = float(self.cfg.get('yc', self.Ly / 2.))
        self.zc = float(self.cfg.get('zc', self.Lz / 2.))

    # --- geometry -----------------------------------------------------------------------------

    def domain_bounds(self, mesh):
        """((x0, x1), ...) of the mesh (mesh.domain_bounds), or from the problem's own domain."""
        bounds = getattr(mesh, 'domain_bounds', None)
        if bounds is not None:
            return bounds
        dim = mesh.geometric_dimension()
        centers = (self.xc, self.yc, self.zc)[:dim]
        lengths = (self.Lx, self.Ly, self.Lz)[:dim]
        return tuple((c - L / 2., c + L / 2.) for c, L in zip(centers, lengths))

    def normalized_coordinates(self, mesh):
        """UFL coordinates s_i in [0, 1] along each direction of the mesh."""
        X = SpatialCoordinate(mesh)
        return [(X[i] - lo) / (hi - lo) for i, (lo, hi) in enumerate(self.domain_bounds(mesh))]

    # --- solution and source ------------------------------------------------------------------

    def exact_u(self, mesh, k, dim, complex):
        """Exact k-form u (UFL proxy); implemented by manufactured-solution subclasses."""
        raise NotImplementedError(type(self).__name__ + " has no exact solution")

    def harmonic_part(self, mesh, k, dim, complex, harmonic_index_sets=()):
        """Expected harmonic part p = P_h f of the source (None means zero)."""
        return None

    def get_exact(self, mesh, k, dim, complex, harmonic_index_sets=()):
        """Exact fields as a dict, or None if unknown.

        Keys: 'u'; 'sigma' = delta u and 'dsigma' = d sigma (None for k = 0); 'du' = d u (None
        for k = dim); 'f' = (d delta + delta d) u plus any harmonic part; 'p' = the harmonic part
        of f (None for zero).
        """
        if not self.has_exact:
            return None
        complex = normalize_complex(dim, complex)
        u = self.exact_u(mesh, k, dim, complex)
        sigma = codifferential(u, k, dim, complex)
        dsigma = None if sigma is None else exterior_derivative(sigma, k - 1, dim, complex)
        p = self.harmonic_part(mesh, k, dim, complex, harmonic_index_sets)
        f = hodge_laplacian_expr(u, k, dim, complex)
        if p is not None:
            f = f + p
        return {'u': u, 'sigma': sigma, 'dsigma': dsigma,
                'du': exterior_derivative(u, k, dim, complex), 'f': f, 'p': p}

    def get_source(self, mesh, k, dim, complex, harmonic_index_sets=()):
        """Source f (UFL proxy of a k-form)."""
        return self.get_exact(mesh, k, dim, complex, harmonic_index_sets)['f']

    # --- boundary data ------------------------------------------------------------------------

    def get_essential_data(self, mesh, face, k, dim, complex):
        """(g_u, g_sigma) on an essential face; default: (u, delta u) from the exact solution.

        Entries may be None (homogeneous). Without an exact solution both are None.
        """
        exact = self.get_exact(mesh, k, dim, complex) if self.has_exact else None
        if exact is None:
            return None, None
        return exact['u'], exact['sigma']

    def get_natural_data(self, mesh, face, k, dim, complex):
        """(g_u, g_du) on a natural face; default: (u, d u) from the exact solution."""
        exact = self.get_exact(mesh, k, dim, complex) if self.has_exact else None
        if exact is None:
            return None, None
        return exact['u'], exact['du']


# ---------------------------------------------------------------------------------------------
# Manufactured solutions
# ---------------------------------------------------------------------------------------------

class SineProblem(EllipticProblemData):
    """Manufactured trigonometric solution valid for every (dim, k, complex) and every BC set.

    Component j of the proxy is prod_i a_j sin(2 pi m_i s_i + phi_ij), with modes m_i from the
    cfg entry 'modes' (one per direction, default 1) and fixed phases. Every factor covers whole
    periods, so u is periodic in every direction and has zero mean in every component (hence it
    is orthogonal to every harmonic form, and p = 0).
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        modes = self.cfg.get('modes', [1, 1, 1])
        if isinstance(modes, (int, float)):
            modes = [modes]
        modes = [int(m) for m in modes] + [int(modes[-1])] * 3
        if any(m < 1 for m in modes):
            raise ValueError("sine problem modes must be positive integers")
        self.modes = modes[:3]

    def exact_u(self, mesh, k, dim, complex):
        """See the class docstring."""
        s = self.normalized_coordinates(mesh)
        components = []
        for j in range(dim):
            factors = [sin(2 * pi * self.modes[i] * s[i] + 0.3 * (j + 1) + 0.7 * i) for i in range(dim)]
            components.append((1.0 / (j + 1)) * _product(factors))
        return _form_from_components(components, k, dim)


class PolynomialProblem(EllipticProblemData):
    """Low-degree polynomial solution for exactness/consistency tests on bounded meshes.

    Component j is prod_i q_ij(s_i) with q_ij(s) = s^d - 1/(d+1) + c_ij (s - 1/2), d = cfg
    'degree' (default 2): zero mean in each factor (so u is orthogonal to the harmonic forms),
    and of degree d in each direction, so for k = 0 and order >= d the solution lies in the CG
    space. Polynomials are not periodic, so periodic meshes are rejected.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.degree = int(self.cfg.get('degree', 2))
        if self.degree < 1:
            raise ValueError("polynomial problem degree must be >= 1")

    def exact_u(self, mesh, k, dim, complex):
        """See the class docstring."""
        if getattr(mesh, 'periodic_directions', ()):
            raise ValueError("the polynomial problem is not periodic; use a bounded mesh")
        s = self.normalized_coordinates(mesh)
        d = self.degree
        components = []
        for j in range(dim):
            factors = [s[i] ** d - 1.0 / (d + 1) + (0.5 * (i + 1) - 0.25 * j) * (s[i] - 0.5)
                       for i in range(dim)]
            components.append(_product(factors))
        return _form_from_components(components, k, dim)


class IncompatibleSourceProblem(SineProblem):
    """SineProblem with a harmonic component added to f (tests §1.5/§1.6).

    f = Delta u_sine + sum_I a_I dx^I over the harmonic index sets passed to get_exact/get_source
    (amplitudes a_I = cfg 'amplitude' (default 1) times 1, 1/2, 1/3, ...). The exact u is that
    of SineProblem, and the expected multiplier is p = sum_I a_I dx^I. Only harmonic directions
    are added: a non-harmonic constant would change u.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.amplitude = float(self.cfg.get('amplitude', 1.0))

    def harmonic_part(self, mesh, k, dim, complex, harmonic_index_sets=()):
        """p = sum_I a_I dx^I (None if there are no harmonic forms)."""
        if not harmonic_index_sets:
            return None
        terms = [(self.amplitude / (n + 1)) * constant_form_proxy(I, k, dim, complex)
                 for n, I in enumerate(harmonic_index_sets)]
        return _sum(terms)


# ---------------------------------------------------------------------------------------------
# Problems without an exact solution
# ---------------------------------------------------------------------------------------------

class GaussianSourceProblem(EllipticProblemData):
    """Gaussian bump source with homogeneous BCs and no exact solution (demos and plots).

    f = exp(-|s - c|^2 / w^2) times a fixed direction for vector proxies, with center c = cfg
    'center' (default the domain center) and width w = cfg 'width' (default 0.1), in normalized
    coordinates. Any harmonic part of f goes into p (lagrange) or is projected out (nullspace).
    """

    has_exact = False

    def __init__(self, parameters):
        super().__init__(parameters)
        self.width = float(self.cfg.get('width', 0.1))
        self.center = [float(c) for c in self.cfg.get('center', [0.5, 0.5, 0.5])] + [0.5] * 3

    def bump(self, mesh):
        """Scalar Gaussian bump in normalized coordinates."""
        s = self.normalized_coordinates(mesh)
        r2 = _sum([(s[i] - self.center[i]) ** 2 for i in range(len(s))])
        return exp(-r2 / self.width ** 2)

    def get_source(self, mesh, k, dim, complex, harmonic_index_sets=()):
        """See the class docstring."""
        normalize_complex(dim, complex)
        g = self.bump(mesh)
        weights = (1.0, 0.5, 0.25)
        return _form_from_components([weights[j] * g for j in range(dim)], k, dim)


class BoundaryDrivenProblem(EllipticProblemData):
    """f = 0 with nonzero boundary data on one face only (tests the inhomogeneous path).

    The driven face is cfg 'face' (default 'left'); its data are smooth bumps of the right form
    degree (essential: (g_u, g_sigma); natural: (g_u, g_du)), all other faces are homogeneous.
    There is no exact solution; the discrete solution should simply be nonzero.
    """

    has_exact = False

    def __init__(self, parameters):
        super().__init__(parameters)
        self.face = str(self.cfg.get('face', 'left'))
        self.amplitude = float(self.cfg.get('amplitude', 1.0))

    def profile(self, mesh, j, dim):
        """Smooth j-form proxy used as boundary data."""
        s = self.normalized_coordinates(mesh)
        bump = self.amplitude * exp(-_sum([(s[i] - 0.5) ** 2 for i in range(dim)]) / 0.1)
        weights = (1.0, 0.5, 0.25)
        return _form_from_components([weights[i] * bump for i in range(dim)], j, dim)

    def get_source(self, mesh, k, dim, complex, harmonic_index_sets=()):
        """Zero source of the right shape."""
        normalize_complex(dim, complex)
        return zero_form(k, dim)

    def _driven(self, face):
        """True if face is the driven face (face is a BoundaryFace or a name)."""
        return getattr(face, 'name', face) == self.face

    def get_essential_data(self, mesh, face, k, dim, complex):
        """(g_u, g_sigma): bumps on the driven face, None elsewhere."""
        if not self._driven(face):
            return None, None
        g_u = self.profile(mesh, k, dim) if k < dim else None
        g_sigma = self.profile(mesh, k - 1, dim) if k > 0 else None
        return g_u, g_sigma

    def get_natural_data(self, mesh, face, k, dim, complex):
        """(g_u, g_du): bumps on the driven face, None elsewhere."""
        if not self._driven(face):
            return None, None
        g_u = self.profile(mesh, k, dim) if k > 0 else None
        g_du = self.profile(mesh, k + 1, dim) if k < dim else None
        return g_u, g_du


# ---------------------------------------------------------------------------------------------
# Boundary data collection (the elliptic 'boundary_data' option)
# ---------------------------------------------------------------------------------------------

@dataclass
class BoundaryData:
    """Per-face boundary data in the format of dimswe.boundary_conditions.

    essential_u / essential_sigma feed build_essential_bcs(g_u=..., g_sigma=...);
    natural_u / natural_du feed build_natural_terms(g_u=..., g_du=...). Each maps face names to
    UFL expressions; missing faces are homogeneous.
    """
    essential_u: Dict[str, object] = field(default_factory=dict)
    essential_sigma: Dict[str, object] = field(default_factory=dict)
    natural_u: Dict[str, object] = field(default_factory=dict)
    natural_du: Dict[str, object] = field(default_factory=dict)


def collect_boundary_data(problem, mode, mesh, registry, face_bcs, k, dim, complex):
    """Gather the boundary data of every bounded face according to the 'boundary_data' option.

    Args:
        problem: an EllipticProblemData.
        mode: 'exact' (from the exact solution, ignoring subclass overrides; requires one),
            'problem' (the problem's get_essential_data/get_natural_data, which default to the
            exact solution) or 'homogeneous' (all zero).
        mesh, registry, face_bcs: the mesh, its face registry and the parsed per-face BC types.
        k, dim, complex: form degree, dimension and 2D complex.

    Data that cannot be used are dropped: no u trace for k = dim and no sigma for k = 0
    (essential); no u data for k = 0 and no du for k = dim (natural).
    """
    if mode not in BOUNDARY_DATA_MODES:
        raise ValueError("boundary_data must be one of " + str(list(BOUNDARY_DATA_MODES)) + ", got " + str(mode))
    data = BoundaryData()
    if mode == 'homogeneous':
        return data
    if mode == 'exact':
        if not problem.has_exact:
            raise ValueError("boundary_data: exact needs a problem with an exact solution; " + type(problem).__name__ + " has none")
        # Call the base-class methods so subclass overrides do not apply.
        essential = lambda face: EllipticProblemData.get_essential_data(problem, mesh, face, k, dim, complex)
        natural = lambda face: EllipticProblemData.get_natural_data(problem, mesh, face, k, dim, complex)
    else:
        essential = lambda face: problem.get_essential_data(mesh, face, k, dim, complex)
        natural = lambda face: problem.get_natural_data(mesh, face, k, dim, complex)

    for name, face in registry.items():
        if face_bcs[name] == 'essential':
            g_u, g_sigma = essential(face)
            if g_u is not None and k < dim:
                data.essential_u[name] = g_u
            if g_sigma is not None and k > 0:
                data.essential_sigma[name] = g_sigma
        else:
            g_u, g_du = natural(face)
            if g_u is not None and k > 0:
                data.natural_u[name] = g_u
            if g_du is not None and k < dim:
                data.natural_du[name] = g_du
    return data
