"""Unit tests for dimswe/elliptic_problems.py (TODO-elliptic §4.5)."""
from itertools import combinations

import numpy
import pytest
from firedrake import (FacetNormal, Function, SpatialCoordinate, as_vector, assemble, dx, inner,
                       sin)

from dimswe.boundary_conditions import (get_boundary_registry, measure_for,
                                        parse_boundary_conditions)
from dimswe.differential_forms import (boundary_pairing, constant_form_proxy, exterior_derivative,
                                       is_scalar_form)
from dimswe.elliptic_problems import (BoundaryDrivenProblem, GaussianSourceProblem,
                                      IncompatibleSourceProblem, PolynomialProblem, SineProblem,
                                      collect_boundary_data, get_elliptic_problem)
from dimswe.meshes import get_mesh_and_spaces

COMPLEX_FOR_DIM = {1: [None], 2: ['grad-curl', 'curl-div'], 3: [None]}
ALL_CASES = [(dim, cx, k) for dim in (1, 2, 3) for cx in COMPLEX_FOR_DIM[dim] for k in range(dim + 1)]
MESH_FOR_DIM = {1: 'line', 2: 'rectangle', 3: 'box'}
FACES = {1: ['left', 'right'], 2: ['left', 'right', 'bottom', 'top'],
         3: ['left', 'right', 'bottom', 'top', 'back', 'front']}
QDEG = 10


def case_id(case):
    """Readable pytest id for a (dim, complex, k) case."""
    dim, cx, k = case
    return str(dim) + "D-" + (cx + "-" if cx else "") + "k" + str(k)


def make_parameters(name, dim, periodic=(), n=4, order=2, **cfg):
    """Parameters with an off-center, non-cubic domain in the elliptic-problem section."""
    problem = {'name': name, 'Lx': 1.5, 'Ly': 1.0, 'Lz': 0.75, 'xc': 0.25, 'yc': -0.5}
    problem.update(cfg)
    return {'mesh': {'type': MESH_FOR_DIM[dim], 'nx': n, 'ny': n, 'nz': n, 'simplicial_cells': False,
                     'periodic_directions': list(periodic)},
            'spatial-discretization': {'family': 'Q', 'order': order, 'lump_mass': False},
            'elliptic-problem': problem}


def make_problem_and_mesh(name, dim, periodic=(), n=4, order=2, **cfg):
    """Problem object, used directly as the mesh domain (duck-typed contract of build_mesh)."""
    parameters = make_parameters(name, dim, periodic, n, order, **cfg)
    problem = get_elliptic_problem(parameters)
    mesh, spaces = get_mesh_and_spaces(parameters, problem)
    return problem, mesh, spaces


def random_field(X, j, dim, rng):
    """Smooth j-form proxy with seeded random trigonometric components."""
    def component():
        a = rng.uniform(0.5, 2.0, size=dim)
        b = rng.uniform(-1.0, 1.0)
        return sin(sum(float(a[i]) * X[i] for i in range(dim)) + float(b))
    if is_scalar_form(j, dim):
        return component()
    return as_vector([component() for _ in range(dim)])


def boundary_integral(integrand, mesh, dim):
    """Integral over every registry face (whole boundary of a bounded mesh)."""
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    return sum(assemble(integrand * measure_for(face, degree=QDEG)) for face in registry.values())


# ---------------------------------------------------------------------------------------------
# Dispatcher and domain
# ---------------------------------------------------------------------------------------------

def test_get_elliptic_problem():
    """Names map to classes; unknown names are rejected."""
    expected = {'sine': SineProblem, 'polynomial': PolynomialProblem,
                'gaussian-source': GaussianSourceProblem, 'boundary-driven': BoundaryDrivenProblem,
                'incompatible-source': IncompatibleSourceProblem}
    for name, cls in expected.items():
        assert type(get_elliptic_problem({'elliptic-problem': {'name': name}})) is cls
    with pytest.raises(ValueError):
        get_elliptic_problem({'elliptic-problem': {'name': 'nope'}})


def test_domain_defaults_and_cfg():
    """Lengths default to 1 and centers to L/2; cfg values override."""
    default = get_elliptic_problem({'elliptic-problem': {'name': 'sine'}})
    assert (default.Lx, default.Ly, default.Lz, default.xc, default.yc, default.zc) == (1, 1, 1, 0.5, 0.5, 0.5)
    custom = get_elliptic_problem(make_parameters('sine', 2))
    assert (custom.Lx, custom.Ly, custom.xc, custom.yc, custom.zc) == (1.5, 1.0, 0.25, -0.5, 0.375)


def test_problem_as_mesh_domain():
    """A bounded mesh built from a problem is centered at (xc, yc)."""
    _, mesh, _ = make_problem_and_mesh('sine', 2)
    assert mesh.domain_bounds == ((-0.5, 1.0), (-1.0, 0.0))


def test_sine_modes_validation():
    """Modes must be positive integers; a scalar is broadcast."""
    assert get_elliptic_problem(make_parameters('sine', 2, modes=2)).modes == [2, 2, 2]
    assert get_elliptic_problem(make_parameters('sine', 2, modes=[1, 3])).modes == [1, 3, 3]
    with pytest.raises(ValueError):
        get_elliptic_problem(make_parameters('sine', 2, modes=[0, 1]))


# ---------------------------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_exact_fields(case):
    """sigma/dsigma absent for k = 0, du absent for k = dim, p absent for MMS problems."""
    dim, cx, k = case
    problem, mesh, _ = make_problem_and_mesh('sine', dim, n=2, order=1)
    exact = problem.get_exact(mesh, k, dim, cx)
    assert set(exact) == {'u', 'sigma', 'dsigma', 'du', 'f', 'p'}
    assert (exact['sigma'] is None) == (k == 0)
    assert (exact['dsigma'] is None) == (k == 0)
    assert (exact['du'] is None) == (k == dim)
    assert exact['p'] is None
    assert exact['u'].ufl_shape == (() if is_scalar_form(k, dim) else (dim,))


@pytest.mark.parametrize("name", ['sine', 'polynomial', 'incompatible-source'])
@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_exact_solution_satisfies_weak_form(case, name):
    """The exact fields satisfy both equations of §1.1 with full boundary pairings:

    (sigma, tau) - (u, d tau) + int B_{k-1}(tau, u) = 0
    (d sigma, v) + (d u, d v) + (p, v) - (f, v) - int B_k(v, d u) = 0
    for random smooth tau, v; this checks sigma = delta u, d u and f = Delta u (+ p).
    """
    dim, cx, k = case
    problem, mesh, _ = make_problem_and_mesh(name, dim, n=6)
    harmonic_sets = list(combinations(('x', 'y', 'z')[:dim], k))[:2]
    exact = problem.get_exact(mesh, k, dim, cx, harmonic_index_sets=harmonic_sets)
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    rng = numpy.random.default_rng(100 * dim + 10 * k + (1 if cx == 'curl-div' else 0))
    u, sigma, du, f, p = exact['u'], exact['sigma'], exact['du'], exact['f'], exact['p']
    scale = max(1.0, assemble(inner(f, f) * dx(degree=QDEG)) ** 0.5)

    if k > 0:
        tau = random_field(X, k - 1, dim, rng)
        eq1 = assemble((inner(sigma, tau) - inner(u, exterior_derivative(tau, k - 1, dim, cx))) * dx(degree=QDEG)) \
            + boundary_integral(boundary_pairing(tau, u, k - 1, dim, cx, n), mesh, dim)
        assert abs(eq1) < 1e-8 * scale

    v = random_field(X, k, dim, rng)
    volume = -inner(f, v)
    if k > 0:
        volume = volume + inner(exterior_derivative(sigma, k - 1, dim, cx), v)
    if k < dim:
        volume = volume + inner(du, exterior_derivative(v, k, dim, cx))
    if p is not None:
        volume = volume + inner(p, v)
    eq2 = assemble(volume * dx(degree=QDEG))
    if k < dim:
        eq2 -= boundary_integral(boundary_pairing(v, du, k, dim, cx, n), mesh, dim)
    assert abs(eq2) < 1e-8 * scale


@pytest.mark.parametrize("name", ['sine', 'polynomial'])
@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_exact_solution_orthogonal_to_constant_forms(case, name):
    """Every component has zero mean, so u is orthogonal to every constant form dx^I."""
    dim, cx, k = case
    problem, mesh, _ = make_problem_and_mesh(name, dim)
    u = problem.get_exact(mesh, k, dim, cx)['u']
    norm = assemble(inner(u, u) * dx(degree=QDEG)) ** 0.5
    assert norm > 1e-2
    for I in combinations(range(dim), k):
        h = constant_form_proxy(I, k, dim, cx)
        assert abs(assemble(inner(u, h) * dx(degree=QDEG))) < 1e-10 * norm


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sine_is_periodic(dim):
    """u takes the same values on opposite faces (so it also works on periodic meshes)."""
    problem, mesh, _ = make_problem_and_mesh('sine', dim, modes=[1, 2, 1])
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    s = problem.normalized_coordinates(mesh)
    k = 1 if dim > 1 else 0
    cx = 'grad-curl' if dim == 2 else None
    u = problem.get_exact(mesh, k, dim, cx)['u']
    for low, high, axis in (('left', 'right', 0), ('bottom', 'top', 1), ('back', 'front', 2))[:dim]:
        # A weight independent of the normal coordinate distinguishes tangential profiles.
        weight = 1.0 + sum((t + 1) * s[t] for t in range(dim) if t != axis)
        values = [assemble(inner(u, u) * weight * measure_for(registry[name], degree=QDEG))
                  for name in (low, high)]
        assert abs(values[0] - values[1]) < 1e-10 * max(1.0, abs(values[0]))


def test_sine_on_periodic_mesh():
    """The sine problem builds on fully periodic meshes (domain starts at the origin)."""
    problem, mesh, _ = make_problem_and_mesh('sine', 2, periodic=('x', 'y'))
    assert mesh.domain_bounds == ((0.0, 1.5), (0.0, 1.0))
    u = problem.get_exact(mesh, 0, 2, 'grad-curl')['u']
    assert abs(assemble(u * dx(degree=QDEG))) < 1e-10


def test_polynomial_problem():
    """Rejected on periodic meshes; for k = 0 and order >= degree, u lies in the CG space."""
    problem, mesh, _ = make_problem_and_mesh('polynomial', 2, periodic=('x',))
    with pytest.raises(ValueError):
        problem.get_exact(mesh, 0, 2, 'grad-curl')
    for dim in (1, 2, 3):
        problem, mesh, spaces = make_problem_and_mesh('polynomial', dim, n=2, order=2, degree=2)
        u = problem.get_exact(mesh, 0, dim, 'grad-curl')['u']
        u_h = Function(spaces.CG).interpolate(u)
        assert assemble((u_h - u) ** 2 * dx(degree=QDEG)) < 1e-24


def test_incompatible_source_problem():
    """f = f_sine + sum_I a_I dx^I with a_I = amplitude / (n + 1), and p equals that sum."""
    sets = [('x',), ('y',)]
    problem, mesh, _ = make_problem_and_mesh('incompatible-source', 2, amplitude=2.0)
    sine, _, _ = make_problem_and_mesh('sine', 2)
    exact = problem.get_exact(mesh, 1, 2, 'curl-div', harmonic_index_sets=sets)
    expected_p = 2.0 * constant_form_proxy(('x',), 1, 2, 'curl-div') + 1.0 * constant_form_proxy(('y',), 1, 2, 'curl-div')
    # Compare integrals rather than a literal difference (UFL may fold that into a domain-less
    # Zero). curl-div proxies: p = 2 (0, 1) + 1 (-1, 0) = (-1, 2); the area is 1.5.
    area = 1.5
    assert abs(assemble(inner(exact['p'], exact['p']) * dx(domain=mesh)) - 5.0 * area) < 1e-12
    assert abs(assemble(inner(exact['p'], constant_form_proxy(('x',), 1, 2, 'curl-div')) * dx(domain=mesh)) - 2.0 * area) < 1e-12
    f_sine = sine.get_source(mesh, 1, 2, 'curl-div')
    diff_f = exact['f'] - f_sine - expected_p
    assert assemble(inner(diff_f, diff_f) * dx(degree=QDEG)) < 1e-20
    assert problem.get_exact(mesh, 1, 2, 'curl-div')['p'] is None


# ---------------------------------------------------------------------------------------------
# Problems without exact solutions
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_gaussian_source_problem(case):
    """No exact solution; f has the right shape and is nonzero; boundary data are homogeneous."""
    dim, cx, k = case
    problem, mesh, _ = make_problem_and_mesh('gaussian-source', dim, n=2, order=1)
    assert not problem.has_exact and problem.get_exact(mesh, k, dim, cx) is None
    f = problem.get_source(mesh, k, dim, cx)
    assert f.ufl_shape == (() if is_scalar_form(k, dim) else (dim,))
    assert assemble(inner(f, f) * dx(degree=QDEG)) > 1e-3
    assert problem.get_essential_data(mesh, 'left', k, dim, cx) == (None, None)
    assert problem.get_natural_data(mesh, 'left', k, dim, cx) == (None, None)


@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_boundary_driven_problem(case):
    """f = 0; data only on the driven face, of the right form degrees."""
    dim, cx, k = case
    problem, mesh, _ = make_problem_and_mesh('boundary-driven', dim, n=2, order=1, face='right')
    f = problem.get_source(mesh, k, dim, cx)
    assert assemble(inner(f, f) * dx(domain=mesh)) == 0.0
    assert problem.get_essential_data(mesh, 'left', k, dim, cx) == (None, None)
    assert problem.get_natural_data(mesh, 'left', k, dim, cx) == (None, None)
    g_u, g_sigma = problem.get_essential_data(mesh, 'right', k, dim, cx)
    assert (g_u is None) == (k == dim) and (g_sigma is None) == (k == 0)
    g_u, g_du = problem.get_natural_data(mesh, 'right', k, dim, cx)
    assert (g_u is None) == (k == 0) and (g_du is None) == (k == dim)
    if g_du is not None:
        assert g_du.ufl_shape == (() if is_scalar_form(k + 1, dim) else (dim,))


# ---------------------------------------------------------------------------------------------
# collect_boundary_data
# ---------------------------------------------------------------------------------------------

def mixed_bcs(dim):
    """Essential on the low faces, natural on the high faces."""
    return {name: ('essential' if i % 2 == 0 else 'natural') for i, name in enumerate(FACES[dim])}


@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_collect_boundary_data_exact(case):
    """'exact': every usable entry present on the faces of its type, none elsewhere."""
    dim, cx, k = case
    problem, mesh, _ = make_problem_and_mesh('sine', dim, n=2, order=1)
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    face_bcs = parse_boundary_conditions(mixed_bcs(dim), registry)
    data = collect_boundary_data(problem, 'exact', mesh, registry, face_bcs, k, dim, cx)
    essential = {name for name, t in face_bcs.items() if t == 'essential'}
    natural = set(face_bcs) - essential
    assert set(data.essential_u) == (essential if k < dim else set())
    assert set(data.essential_sigma) == (essential if k > 0 else set())
    assert set(data.natural_u) == (natural if k > 0 else set())
    assert set(data.natural_du) == (natural if k < dim else set())


def test_collect_boundary_data_modes():
    """'homogeneous' is empty, 'exact' bypasses overrides, 'problem' uses them; bad input raises."""
    dim, cx, k = 2, 'grad-curl', 1

    class Overridden(SineProblem):
        """Sine problem whose essential data are replaced by a marker."""
        marker = as_vector((3.0, 4.0))

        def get_essential_data(self, mesh, face, k, dim, complex):
            return self.marker, None

    parameters = make_parameters('sine', dim, n=2, order=1)
    problem = Overridden(parameters)
    mesh, _ = get_mesh_and_spaces(parameters, problem)
    registry = get_boundary_registry(mesh, 'rectangle')
    face_bcs = parse_boundary_conditions(mixed_bcs(dim), registry)

    empty = collect_boundary_data(problem, 'homogeneous', mesh, registry, face_bcs, k, dim, cx)
    assert not (empty.essential_u or empty.essential_sigma or empty.natural_u or empty.natural_du)
    exact = collect_boundary_data(problem, 'exact', mesh, registry, face_bcs, k, dim, cx)
    assert all(g is not Overridden.marker for g in exact.essential_u.values())
    assert set(exact.essential_sigma) == {'left', 'bottom'}
    custom = collect_boundary_data(problem, 'problem', mesh, registry, face_bcs, k, dim, cx)
    assert all(g is Overridden.marker for g in custom.essential_u.values())
    assert custom.essential_sigma == {}

    driven, mesh2, _ = make_problem_and_mesh('boundary-driven', dim, n=2, order=1, face='left')
    with pytest.raises(ValueError):
        collect_boundary_data(driven, 'exact', mesh2, registry, face_bcs, k, dim, cx)
    data = collect_boundary_data(driven, 'problem', mesh2, registry, face_bcs, k, dim, cx)
    assert set(data.essential_u) == {'left'} and data.natural_du == {}
    with pytest.raises(ValueError):
        collect_boundary_data(problem, 'sometimes', mesh, registry, face_bcs, k, dim, cx)
