"""Unit tests for dimswe/boundary_conditions.py (face registry, BC parsing, BCs and natural terms)."""
from types import SimpleNamespace

import pytest
from firedrake import (Constant, FacetNormal, Function, MixedFunctionSpace, SpatialCoordinate,
                       as_vector, assemble, curl, div, dot, ds, ds_b, ds_t, ds_v, dx, grad,
                       inner)

from dimswe.boundary_conditions import (build_essential_bcs, build_face_registry,
                                        build_natural_terms, face_bcs_by_direction, faces_of_type,
                                        get_boundary_registry, is_extruded, measure_for,
                                        parse_boundary_conditions, verify_boundary_registry)
from dimswe.meshes import annotate_mesh, build_mesh, get_mesh_and_spaces


def make_domain():
    """Off-center box-shaped domain, so wrong face numbering or bounds are detected."""
    return SimpleNamespace(Lx=2.0, Ly=3.0, Lz=0.5, xc=1.5, yc=-1.0, zc=0.0)


def make_parameters(mesh_type, periodic_directions=None, simplicial=False, family='Q', order=1,
                    nz=2):
    """Minimal parameters for the DIMSWE mesh builder.

    Firedrake's PeriodicBoxMesh needs at least 3 cells per direction, so callers building one
    pass nz=3 (nx=4 and ny=3 already satisfy this).
    """
    mesh = {'type': mesh_type, 'nx': 4, 'ny': 3, 'nz': nz, 'simplicial_cells': simplicial}
    if periodic_directions is not None:
        mesh['periodic_directions'] = periodic_directions
    return {'mesh': mesh, 'spatial-discretization': {'family': family, 'order': order, 'lump_mass': False}}


def make_mesh(mesh_type, periodic_directions=None, simplicial=False, family='Q', nz=2):
    """Build and annotate a mesh without constructing spaces (works for every cell type)."""
    parameters = make_parameters(mesh_type, periodic_directions, simplicial, family, nz=nz)
    domain = make_domain()
    mesh = build_mesh(parameters, domain)
    annotate_mesh(mesh, parameters, domain)
    return mesh, parameters


# ---------------------------------------------------------------------------------------------
# Pure-Python registry and parsing
# ---------------------------------------------------------------------------------------------

def _ids(registry):
    """Summarize a registry as name -> (measure, DirichletBC token)."""
    return {name: (face.measure, face.dirichlet_subdomain) for name, face in registry.items()}


@pytest.mark.parametrize("dim, periodic, extruded, expected", [
    (1, (), False, {'left': ('ds', 1), 'right': ('ds', 2)}),
    (1, ('x',), False, {}),
    (2, (), False, {'left': ('ds', 1), 'right': ('ds', 2), 'bottom': ('ds', 3), 'top': ('ds', 4)}),
    (2, ('x',), False, {'bottom': ('ds', 1), 'top': ('ds', 2)}),
    (2, ('y',), False, {'left': ('ds', 1), 'right': ('ds', 2)}),
    (2, ('x', 'y'), False, {}),
    (3, (), False, {'left': ('ds', 1), 'right': ('ds', 2), 'bottom': ('ds', 3), 'top': ('ds', 4),
                    'back': ('ds', 5), 'front': ('ds', 6)}),
    (3, ('x',), False, {'bottom': ('ds', 3), 'top': ('ds', 4), 'back': ('ds', 5), 'front': ('ds', 6)}),
    (3, ('y', 'z'), False, {'left': ('ds', 1), 'right': ('ds', 2)}),
    (3, ('x', 'z'), False, {'bottom': ('ds', 3), 'top': ('ds', 4)}),
    (3, ('x', 'y', 'z'), False, {}),
    (3, (), True, {'left': ('ds_v', 1), 'right': ('ds_v', 2), 'bottom': ('ds_v', 3), 'top': ('ds_v', 4),
                   'back': ('ds_b', 'bottom'), 'front': ('ds_t', 'top')}),
    (3, ('x',), True, {'bottom': ('ds_v', 1), 'top': ('ds_v', 2), 'back': ('ds_b', 'bottom'),
                       'front': ('ds_t', 'top')}),
    (3, ('z',), True, {'left': ('ds_v', 1), 'right': ('ds_v', 2), 'bottom': ('ds_v', 3), 'top': ('ds_v', 4)}),
    (3, ('x', 'y'), True, {'back': ('ds_b', 'bottom'), 'front': ('ds_t', 'top')}),
    (3, ('x', 'y', 'z'), True, {}),
])
def test_build_face_registry(dim, periodic, extruded, expected):
    """1D/2D and extruded bases: consecutive ids over bounded directions; extruded z faces use
    ds_b/ds_t; non-extruded 3D boxes keep the fixed ids 1-6 (PeriodicBoxMesh convention)."""
    registry = build_face_registry(dim, periodic, extruded)
    assert _ids(registry) == expected
    assert list(registry) == list(expected)
    for face in registry.values():
        assert face.subdomain_ids == ((face.dirichlet_subdomain,) if face.measure in ('ds', 'ds_v') else ())


def test_build_face_registry_rejects_bad_input():
    """Unsupported dimensions and non-3D extruded meshes are rejected."""
    with pytest.raises(ValueError):
        build_face_registry(4, ())
    with pytest.raises(ValueError):
        build_face_registry(2, (), extruded=True)


def test_parse_boundary_conditions_valid():
    """Case-insensitive names and types; result ordered like the registry."""
    registry = build_face_registry(2, ('x',))
    parsed = parse_boundary_conditions({'Top': 'Natural', 'bottom': 'essential'}, registry)
    assert list(parsed.items()) == [('bottom', 'essential'), ('top', 'natural')]
    assert parse_boundary_conditions(None, build_face_registry(2, ('x', 'y'))) == {}


@pytest.mark.parametrize("cfg", [
    {'bottom': 'essential'},                                          # missing top
    {'bottom': 'essential', 'top': 'natural', 'left': 'essential'},   # left is periodic
    {'bottom': 'essential', 'top': 'natural', 'front': 'natural'},    # beyond the dimension
    {'bottom': 'essential', 'top': 'natural', 'north': 'natural'},    # unknown name
    {'bottom': 'dirichlet', 'top': 'natural'},                        # unknown BC type
    {'bottom': 'essential', 'top': 'natural', 'Top': 'natural'},      # duplicate
    ['bottom', 'top'],                                                # not a mapping
])
def test_parse_boundary_conditions_errors(cfg):
    """Every invalid configuration raises ValueError."""
    with pytest.raises(ValueError):
        parse_boundary_conditions(cfg, build_face_registry(2, ('x',)))


def test_faces_of_type_and_by_direction():
    """Faces are split by BC type and grouped per direction (periodic directions absent)."""
    registry = build_face_registry(3, ('y',), extruded=True)
    face_bcs = {'left': 'essential', 'right': 'natural', 'back': 'natural', 'front': 'natural'}
    assert [f.name for f in faces_of_type(registry, face_bcs, 'essential')] == ['left']
    assert [f.name for f in faces_of_type(registry, face_bcs, 'natural')] == ['right', 'back', 'front']
    assert face_bcs_by_direction(registry, face_bcs) == {'x': ('essential', 'natural'),
                                                         'z': ('natural', 'natural')}


# ---------------------------------------------------------------------------------------------
# Registry against the installed Firedrake
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_type, periodic, simplicial, family", [
    ('line', None, False, 'Q'),
    ('line-periodic', None, False, 'Q'),
    ('rectangle', [], False, 'Q'),
    ('rectangle', ['x'], False, 'Q'),
    ('rectangle', ['y'], False, 'Q'),
    ('rectangle', ['x', 'y'], False, 'Q'),
    ('rectangle', [], True, 'Q'),
    ('rectangle', ['x'], True, 'Q'),
    ('rectangle', ['y'], True, 'Q'),
    ('box', [], False, 'Q'),
    ('box', ['x'], False, 'Q'),
    ('box', ['y'], False, 'Q'),
    ('box', ['z'], False, 'Q'),
    ('box', ['x', 'y'], False, 'Q'),
    ('box', ['x', 'z'], False, 'Q'),
    ('box', ['y', 'z'], False, 'Q'),
    ('box', ['x', 'y', 'z'], False, 'Q'),
    ('box', [], True, 'Q'),
    ('box', ['x', 'y', 'z'], True, 'Q'),
    ('box', [], False, 'S-'),
    ('box', ['x'], False, 'S-'),
    ('box', ['y'], False, 'S-'),
    ('box', ['z'], False, 'S-'),
    ('box', ['x', 'y'], False, 'S-'),
    ('box', ['x', 'z'], False, 'S-'),
    ('box', ['y', 'z'], False, 'S-'),
])
def test_registry_matches_firedrake_numbering(mesh_type, periodic, simplicial, family):
    """Each face sits at its logical location, and the faces cover the whole boundary."""
    # PeriodicBoxMesh (fully periodic tets, or serendipity hexes) needs >= 3 cells per direction.
    nz = 3 if (mesh_type == 'box' and (simplicial or family != 'Q') and periodic) else 2
    mesh, parameters = make_mesh(mesh_type, periodic, simplicial, family, nz=nz)
    registry = get_boundary_registry(mesh, mesh_type)
    dim = len(mesh.domain_bounds)
    assert len(registry) == 2 * (dim - len(mesh.periodic_directions))
    verify_boundary_registry(mesh, registry, mesh.domain_bounds)
    # The sum over the registry faces equals the total boundary measure (no face is missing).
    total = sum(assemble(Constant(1.0) * measure_for(face, domain=mesh)) for face in registry.values())
    if is_extruded(mesh):
        everything = assemble(Constant(1.0) * ds_v(domain=mesh))
        if 'z' not in mesh.periodic_directions:
            everything += assemble(Constant(1.0) * ds_t(domain=mesh)) + assemble(Constant(1.0) * ds_b(domain=mesh))
    else:
        everything = assemble(Constant(1.0) * ds(domain=mesh))
    assert abs(total - everything) < 1e-10


def test_registry_rejects_file_meshes():
    """Per-face BCs on general meshes are deferred."""
    mesh, _ = make_mesh('rectangle')
    with pytest.raises(NotImplementedError):
        get_boundary_registry(mesh, 'file')


# ---------------------------------------------------------------------------------------------
# Essential BCs
# ---------------------------------------------------------------------------------------------

def test_essential_bcs_inhomogeneous_scalar():
    """k = 0 primal: u = g on essential faces only, from a single expression or per-face dict."""
    domain = make_domain()
    parameters = make_parameters('rectangle')
    mesh, spaces = get_mesh_and_spaces(parameters, domain)
    registry = get_boundary_registry(mesh, 'rectangle')
    face_bcs = parse_boundary_conditions({'left': 'essential', 'right': 'natural',
                                          'bottom': 'essential', 'top': 'natural'}, registry)
    x, y = SpatialCoordinate(mesh)
    g = x + 2 * y + 1  # linear, so exactly representable in CG1
    faces = faces_of_type(registry, face_bcs, 'essential')
    for data in (g, {'left': g, 'bottom': g}):
        bcs = build_essential_bcs(spaces.CG, None, 0, faces, g_u=data)
        assert len(bcs) == 2
        u = Function(spaces.CG)
        for bc in bcs:
            bc.apply(u)
        for face in registry.values():
            err = assemble((u - g) ** 2 * measure_for(face))
            if face_bcs[face.name] == 'essential':
                assert err < 1e-20
            else:
                assert err > 1e-3


@pytest.mark.parametrize("homogeneous", [True, False])
def test_essential_bcs_mixed_hdiv(homogeneous):
    """2D curl-div k = 2: sigma in Hdiv gets sigma . n = g . n, the DG block u gets no BC.

    RTCF does not support interpolation, so the test data are projected, and the inhomogeneous
    data g = (1 + x, 2 + y) lie in RTCF1, so the (projection-based) BC reproduces g . n exactly.
    """
    domain = make_domain()
    mesh, spaces = get_mesh_and_spaces(make_parameters('rectangle'), domain)
    W = MixedFunctionSpace([spaces.Hdiv, spaces.DG])
    registry = get_boundary_registry(mesh, 'rectangle')
    face_bcs = {'left': 'essential', 'right': 'essential', 'bottom': 'natural', 'top': 'natural'}
    x, y = SpatialCoordinate(mesh)
    g = as_vector((1 + x, 2 + y))
    target = as_vector((0.0, 0.0)) if homogeneous else g
    bcs = build_essential_bcs(W, 0, None, faces_of_type(registry, face_bcs, 'essential'),
                              g_sigma=None if homogeneous else g)
    assert len(bcs) == 2
    # Start from a field whose normal trace differs from the target on every face.
    start = Function(spaces.Hdiv).project(as_vector((3 + x * x, 4 + y * y)))
    w = Function(W)
    w.sub(0).assign(start)
    for bc in bcs:
        bc.apply(w)
    n = FacetNormal(mesh)
    sigma = w.sub(0)
    for face in registry.values():
        mismatch = assemble(dot(sigma - target, n) ** 2 * measure_for(face))
        if face_bcs[face.name] == 'essential':
            assert mismatch < 1e-20
        else:
            assert mismatch > 1e-3


def test_essential_bcs_extruded_top_bottom():
    """Extruded box: the z faces (back/front) use the "bottom"/"top" DirichletBC tokens."""
    domain = make_domain()
    mesh, spaces = get_mesh_and_spaces(make_parameters('box', ['x', 'y']), domain)
    registry = get_boundary_registry(mesh, 'box')
    assert list(registry) == ['back', 'front']
    z = SpatialCoordinate(mesh)[2]
    bcs = build_essential_bcs(spaces.CG, None, 0, list(registry.values()), g_u=z + 1)
    u = Function(spaces.CG)
    for bc in bcs:
        bc.apply(u)
    for face in registry.values():
        assert assemble((u - z - 1) ** 2 * measure_for(face)) < 1e-20


# ---------------------------------------------------------------------------------------------
# Natural boundary terms
# ---------------------------------------------------------------------------------------------

def _skewgrad(w):
    return as_vector((-w.dx(1), w.dx(0)))


def _curl2d(w):
    return w[1].dx(0) - w[0].dx(1)


def _d(w, k, dim, cx):
    """Exterior derivative of a k-form proxy (TODO-elliptic §1.2), local to this test."""
    if dim == 1:
        return w.dx(0)
    if k == 0:
        return _skewgrad(w) if cx == 'curl-div' else grad(w)
    if dim == 2:
        return _curl2d(w) if cx == 'grad-curl' else div(w)
    return curl(w) if k == 1 else div(w)


def _delta(z, k, dim, cx):
    """Formal adjoint of d_k applied to a (k+1)-form proxy z (TODO-elliptic §1.2)."""
    if dim == 1:
        return -z.dx(0)
    if k == 0:
        return -_curl2d(z) if cx == 'curl-div' else -div(z)
    if dim == 2:
        return as_vector((z.dx(1), -z.dx(0))) if cx == 'grad-curl' else -grad(z)
    return curl(z) if k == 1 else -grad(z)


def _poly_form(X, j, dim, which):
    """A smooth non-trivial polynomial j-form proxy (scalar for j in {0, dim}, else vector)."""
    if dim == 1:
        return X[0] ** 2 + 1 if which == 0 else X[0] ** 3 - X[0]
    scalar = j in (0, dim)
    if dim == 2:
        x, y = X
        if scalar:
            return x * x * y + y if which == 0 else x * y + x ** 3
        return as_vector((x * y + 1, x * x - y)) if which == 0 else as_vector((x * y * y, x * x + y ** 3))
    x, y, z = X
    if scalar:
        return x * y * z + z * z if which == 0 else x * x + y * z
    if which == 0:
        return as_vector((y * z + 1, x * x - z, x * y * z))
    return as_vector((x * z * z, y + x * y, z * x - y * y))


CASES = [(1, None, 0),
         (2, 'grad-curl', 0), (2, 'grad-curl', 1),
         (2, 'curl-div', 0), (2, 'curl-div', 1),
         (3, None, 0), (3, None, 1), (3, None, 2)]
MESH_FOR_DIM = {1: 'line', 2: 'rectangle', 3: 'box'}


@pytest.mark.parametrize("dim, cx, k", CASES)
def test_natural_terms_match_integration_by_parts(dim, cx, k):
    """With every face natural, u_term = (d w, z) - (w, delta z) and sigma_term is its negative.

    This checks the signs of every boundary pairing (§1.2) and that the registry faces cover the
    boundary, on non-extruded (1D/2D) and extruded (3D) meshes.
    """
    mesh, _ = make_mesh(MESH_FOR_DIM[dim])
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    w = _poly_form(X, k, dim, 0)
    z = _poly_form(X, k + 1, dim, 1)
    expected = assemble(inner(_d(w, k, dim, cx), z) * dx - inner(w, _delta(z, k, dim, cx)) * dx)
    assert abs(expected) > 1e-3  # the test is not vacuous
    faces = list(registry.values())
    # u equation: v = w is a k-form, g_du = z a (k+1)-form.
    sigma_term, u_term = build_natural_terms(None, w, faces, None, z, k, dim, cx, n)
    assert isinstance(sigma_term, int) and sigma_term == 0
    assert abs(assemble(u_term) - expected) < 1e-9 * max(1.0, abs(expected))
    # sigma equation of the (k+1)-form problem: tau = w, g_u = z.
    sigma_term, u_term = build_natural_terms(w, None, faces, z, None, k + 1, dim, cx, n)
    assert isinstance(u_term, int) and u_term == 0
    assert abs(assemble(sigma_term) + expected) < 1e-9 * max(1.0, abs(expected))


def test_natural_terms_restricted_to_faces():
    """Per-face data and face subsets only integrate over the given faces."""
    mesh, _ = make_mesh('rectangle')
    registry = get_boundary_registry(mesh, 'rectangle')
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    w = _poly_form(X, 0, 2, 0)
    z = _poly_form(X, 1, 2, 1)
    left = registry['left']
    _, u_term = build_natural_terms(None, w, [left, registry['top']], None, {'left': z}, 0, 2, 'grad-curl', n)
    assert abs(assemble(u_term) - assemble(w * dot(z, n) * ds(left.subdomain_ids))) < 1e-12


def test_natural_terms_vanish():
    """Homogeneous data, no faces, and k = n give the integer 0."""
    mesh, _ = make_mesh('rectangle')
    registry = get_boundary_registry(mesh, 'rectangle')
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    faces = list(registry.values())
    w = _poly_form(X, 1, 2, 0)
    assert build_natural_terms(w, w, faces, None, None, 1, 2, 'curl-div', n) == (0, 0)
    assert build_natural_terms(w, w, [], w, w, 1, 2, 'curl-div', n) == (0, 0)
    s = _poly_form(X, 2, 2, 0)
    sigma_term, u_term = build_natural_terms(w, s, faces, s, s, 2, 2, 'curl-div', n)
    # Comparing a UFL Form with == builds an Equation, so test the type instead.
    assert isinstance(u_term, int) and not isinstance(sigma_term, int)
