"""Unit tests for dimswe/differential_forms.py (TODO-elliptic §1.2, §1.4, §4.3, §6)."""
from types import SimpleNamespace

from itertools import combinations

import numpy
import pytest
from firedrake import (FacetNormal, Function, SpatialCoordinate, as_vector, assemble, div, dx, grad,
                       inner, sin)

from dimswe.boundary_conditions import get_boundary_registry, measure_for
from dimswe.differential_forms import (boundary_pairing, codifferential, constant_form_proxy,
                                       constant_form_values, exterior_derivative, form_space,
                                       hodge_laplacian_expr, is_scalar_form, normalize_complex,
                                       normalize_index_set)
from dimswe.meshes import get_mesh_and_spaces

# All 12 (dim, complex, k) combinations, and the subsets with d (k < dim) and d^2 (k + 2 <= dim).
COMPLEX_FOR_DIM = {1: [None], 2: ['grad-curl', 'curl-div'], 3: [None]}
ALL_CASES = [(dim, cx, k) for dim in (1, 2, 3) for cx in COMPLEX_FOR_DIM[dim] for k in range(dim + 1)]
D_CASES = [c for c in ALL_CASES if c[2] < c[0]]
DD_CASES = [c for c in ALL_CASES if c[2] + 2 <= c[0]]
MESH_FOR_DIM = {1: 'line', 2: 'rectangle', 3: 'box'}
QDEG = 10  # quadrature degree for the trigonometric test fields


def case_id(case):
    """Readable pytest id for a (dim, complex, k) case."""
    dim, cx, k = case
    return str(dim) + "D-" + (cx + "-" if cx else "") + "k" + str(k)


def make_mesh_and_spaces(dim, n=6, order=2):
    """Bounded Q-family mesh (extruded in 3D) on an off-center, non-square domain."""
    parameters = {'mesh': {'type': MESH_FOR_DIM[dim], 'nx': n, 'ny': n, 'nz': n, 'simplicial_cells': False},
                  'spatial-discretization': {'family': 'Q', 'order': order, 'lump_mass': False}}
    domain = SimpleNamespace(Lx=1.5, Ly=1.0, Lz=0.75, xc=0.25, yc=-0.5, zc=0.0)
    return get_mesh_and_spaces(parameters, domain)


def random_field(X, j, dim, rng):
    """Smooth j-form proxy with seeded random trigonometric components."""
    def component():
        a = rng.uniform(0.5, 2.0, size=dim)
        b = rng.uniform(-1.0, 1.0)
        c = rng.uniform(0.5, 1.5)
        return c * sin(sum(float(a[i]) * X[i] for i in range(dim)) + float(b))
    if is_scalar_form(j, dim):
        return component()
    return as_vector([component() for _ in range(dim)])


def minus_laplacian(u, scalar, dim):
    """Componentwise -Laplacian of a scalar or vector proxy."""
    if scalar:
        return -div(grad(u))
    return as_vector([-div(grad(u[i])) for i in range(dim)])


def full_boundary_integral(integrand, mesh, dim):
    """Integral over the whole boundary using the face registry (handles extruded meshes)."""
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    return sum(assemble(integrand * measure_for(face, degree=QDEG)) for face in registry.values())


# ---------------------------------------------------------------------------------------------
# Pure-Python helpers and validation
# ---------------------------------------------------------------------------------------------

def test_normalize_complex():
    """2D needs a valid complex; 1D/3D ignore it."""
    assert normalize_complex(2, 'curl-div') == 'curl-div'
    assert normalize_complex(1, 'anything') is None
    assert normalize_complex(3, None) is None
    with pytest.raises(ValueError):
        normalize_complex(2, 'div-curl')
    with pytest.raises(ValueError):
        normalize_complex(4, None)


def test_degree_validation():
    """Out-of-range form degrees are rejected by every operator."""
    for func in (exterior_derivative, codifferential, hodge_laplacian_expr):
        with pytest.raises(ValueError):
            func(None, 3, 2, 'grad-curl')
        with pytest.raises(ValueError):
            func(None, -1, 1, None)


def test_normalize_index_set():
    """Index sets are sorted and accept names or axis numbers."""
    assert normalize_index_set(('z', 'x'), 2, 3) == ('x', 'z')
    assert normalize_index_set((2, 0), 2, 3) == ('x', 'z')
    assert normalize_index_set((), 0, 2) == ()
    for bad, k, dim in [(('x', 'x'), 2, 3), (('z',), 1, 2), (('x',), 2, 3), (('w',), 1, 3)]:
        with pytest.raises(ValueError):
            normalize_index_set(bad, k, dim)


@pytest.mark.parametrize("I, k, dim, cx, expected", [
    ((), 0, 2, 'grad-curl', 1.0),
    (('x', 'y'), 2, 2, 'curl-div', -1.0),  # d(x dy) = div(x (-1, 0)) = -1
    (('x', 'y'), 2, 2, 'grad-curl', 1.0),
    (('x', 'y', 'z'), 3, 3, None, 1.0),
    (('x',), 1, 1, None, 1.0),
    (('x',), 1, 2, 'grad-curl', (1.0, 0.0)),
    (('y',), 1, 2, 'grad-curl', (0.0, 1.0)),
    (('x',), 1, 2, 'curl-div', (0.0, 1.0)),
    (('y',), 1, 2, 'curl-div', (-1.0, 0.0)),
    (('z',), 1, 3, None, (0.0, 0.0, 1.0)),
    (('y', 'z'), 2, 3, None, (1.0, 0.0, 0.0)),
    (('z', 'x'), 2, 3, None, (0.0, -1.0, 0.0)),  # dx ^ dz = -(dz ^ dx)
    (('x', 'y'), 2, 3, None, (0.0, 0.0, 1.0)),
])
def test_constant_form_values(I, k, dim, cx, expected):
    """Proxy map of TODO-elliptic §1.4 (with increasing-order index sets)."""
    assert constant_form_values(I, k, dim, cx) == expected


# ---------------------------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_form_space(dim):
    """k-form spaces: CG, Hcurl (grad-curl, 3D 1-forms), Hdiv (curl-div, 3D 2-forms), DG."""
    _, spaces = make_mesh_and_spaces(dim, n=2, order=1)
    if dim == 1:
        expected = {None: [spaces.CG, spaces.DG]}
    elif dim == 2:
        expected = {'grad-curl': [spaces.CG, spaces.Hcurl, spaces.DG],
                    'curl-div': [spaces.CG, spaces.Hdiv, spaces.DG]}
    else:
        expected = {None: [spaces.CG, spaces.Hcurl, spaces.Hdiv, spaces.DG]}
    for cx, expected_spaces in expected.items():
        for k, V in enumerate(expected_spaces):
            assert form_space(spaces, k, dim, cx) is V


# ---------------------------------------------------------------------------------------------
# Operator identities
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_absent_operators(case):
    """d of a top form and delta of a 0-form are None."""
    dim, cx, k = case
    if k == dim:
        assert exterior_derivative(None, k, dim, cx) is None
    if k == 0:
        assert codifferential(None, k, dim, cx) is None
        assert boundary_pairing(None, None, dim, dim, cx, None) == 0


@pytest.mark.parametrize("case", DD_CASES, ids=case_id)
def test_d_squared_is_zero(case):
    """d(d w) = 0 for smooth closed-form w and for random discrete w in V^k."""
    dim, cx, k = case
    mesh, spaces = make_mesh_and_spaces(dim, n=3)
    X = SpatialCoordinate(mesh)
    rng = numpy.random.default_rng(1234 + 10 * dim + k)
    w_smooth = random_field(X, k, dim, rng)
    w_h = Function(form_space(spaces, k, dim, cx))
    w_h.dat.data[:] = rng.standard_normal(w_h.dat.data.shape)
    for w in (w_smooth, w_h):
        dw = exterior_derivative(w, k, dim, cx)
        ddw = exterior_derivative(dw, k + 1, dim, cx)
        assert assemble(inner(dw, dw) * dx(degree=QDEG)) > 1e-6  # not vacuous
        assert assemble(inner(ddw, ddw) * dx(degree=QDEG)) < 1e-16


@pytest.mark.parametrize("case", DD_CASES, ids=case_id)
def test_delta_squared_is_zero(case):
    """delta(delta z) = 0 for a smooth (k+2)-form z."""
    dim, cx, k = case
    mesh, _ = make_mesh_and_spaces(dim, n=3)
    X = SpatialCoordinate(mesh)
    z = random_field(X, k + 2, dim, numpy.random.default_rng(99 + dim + k))
    dz = codifferential(codifferential(z, k + 2, dim, cx), k + 1, dim, cx)
    assert assemble(inner(dz, dz) * dx(degree=QDEG)) < 1e-16


@pytest.mark.parametrize("case", D_CASES, ids=case_id)
def test_integration_by_parts_identity(case):
    """int d w . z - int w . delta z - int_boundary B(w, z) = 0 for random smooth w, z.

    Verifies every sign of the §1.2 table (d, delta and B together). In 3D the boundary integral
    runs over ds_v + ds_t + ds_b of the extruded mesh.
    """
    dim, cx, k = case
    mesh, _ = make_mesh_and_spaces(dim)
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    rng = numpy.random.default_rng(4321 + 10 * dim + k + (5 if cx == 'curl-div' else 0))
    w = random_field(X, k, dim, rng)
    z = random_field(X, k + 1, dim, rng)
    volume = assemble(inner(exterior_derivative(w, k, dim, cx), z) * dx(degree=QDEG)
                      - inner(w, codifferential(z, k + 1, dim, cx)) * dx(degree=QDEG))
    boundary = full_boundary_integral(boundary_pairing(w, z, k, dim, cx, n), mesh, dim)
    assert abs(boundary) > 1e-3  # not vacuous
    assert abs(volume - boundary) < 1e-10 * max(1.0, abs(boundary))


@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_hodge_laplacian_is_minus_vector_laplacian(case):
    """On flat Cartesian domains (d delta + delta d) u = -Laplacian(u) componentwise."""
    dim, cx, k = case
    mesh, _ = make_mesh_and_spaces(dim, n=3)
    X = SpatialCoordinate(mesh)
    u = random_field(X, k, dim, numpy.random.default_rng(7 + 10 * dim + k))
    diff = hodge_laplacian_expr(u, k, dim, cx) - minus_laplacian(u, is_scalar_form(k, dim), dim)
    reference = assemble(inner(u, u) * dx(degree=QDEG))
    assert assemble(inner(diff, diff) * dx(degree=QDEG)) < 1e-16 * max(1.0, reference)


@pytest.mark.parametrize("case", [c for c in ALL_CASES if c[2] >= 1], ids=case_id)
def test_constant_form_proxy_matches_d_of_coordinates(case):
    """dx^I = d(x_{i1} dx^{I'}) with I' = I minus its first index, for every increasing I.

    This ties the proxy map (including the signs of the 2D curl-div and 3D 2-form proxies) to
    the exterior derivative.
    """
    dim, cx, k = case
    mesh, _ = make_mesh_and_spaces(dim, n=2, order=1)
    X = SpatialCoordinate(mesh)
    for I in combinations(range(dim), k):
        lower = X[I[0]] * constant_form_proxy(I[1:], k - 1, dim, cx)
        diff = exterior_derivative(lower, k - 1, dim, cx) - constant_form_proxy(I, k, dim, cx)
        assert assemble(inner(diff, diff) * dx) < 1e-20
