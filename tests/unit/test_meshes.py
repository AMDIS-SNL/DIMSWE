"""Unit tests for mesh construction in dimswe/meshes.py (periodicity, box meshes, metadata)."""
from types import SimpleNamespace

import pytest
from firedrake import Constant, assemble, ds, ds_v, ds_t, ds_b, dx

from dimswe.meshes import (annotate_mesh, build_mesh, get_mesh_and_spaces, get_periodic_directions,
                           get_domain_bounds, mesh_size, set_dimension)
from dimswe.variables import get_diff_form_space


def make_domain(Lx=2.0, Ly=3.0, Lz=0.5, xc=1.0, yc=-1.0, zc=0.0):
    """Duck-typed domain object (what an elliptic problem provides in place of initcond)."""
    return SimpleNamespace(Lx=Lx, Ly=Ly, Lz=Lz, xc=xc, yc=yc, zc=zc)


def make_parameters(mesh_type, periodic_directions=None, simplicial=False, nx=4, ny=3, nz=2, order=1):
    """Minimal parameter dictionary for get_mesh_and_spaces (no 'diagonal' key on purpose)."""
    mesh = {'type': mesh_type, 'nx': nx, 'ny': ny, 'nz': nz, 'simplicial_cells': simplicial}
    if periodic_directions is not None:
        mesh['periodic_directions'] = periodic_directions
    return {'mesh': mesh,
            'spatial-discretization': {'family': 'Q', 'order': order, 'lump_mass': False}}


@pytest.mark.parametrize("mesh_type, requested, expected", [
    ('line', None, ()),
    ('line-periodic', None, ('x',)),
    ('line', ['x'], ('x',)),
    ('rectangle', None, ()),
    ('rectangle-periodic', None, ('x', 'y')),
    ('rectangle', ['y'], ('y',)),
    ('rectangle', 'x', ('x',)),
    ('rectangle', ['y', 'x', 'x'], ('x', 'y')),
    ('rectangle-periodic', [], ()),
    ('rectangle-periodic', 'none', ()),
    ('box', 'both', ('x', 'y', 'z')),
    ('box-periodic', ['z'], ('z',)),
    ('box', ['x', 'z'], ('x', 'z')),
])
def test_get_periodic_directions(mesh_type, requested, expected):
    """Mesh type defaults and the periodic_directions override are normalized to x, y, z order."""
    mp = {'type': mesh_type}
    if requested is not None:
        mp['periodic_directions'] = requested
    assert get_periodic_directions(mp) == expected


@pytest.mark.parametrize("mesh_type, requested", [
    ('line', ['y']),
    ('rectangle', ['z']),
    ('box', ['w']),
    ('file', ['x']),
])
def test_get_periodic_directions_rejects_invalid(mesh_type, requested):
    """Directions beyond the mesh dimension, unknown names and periodic file meshes are rejected."""
    with pytest.raises(ValueError):
        get_periodic_directions({'type': mesh_type, 'periodic_directions': requested})


def test_get_domain_bounds_conventions():
    """Bounded line/rectangle meshes are centered; periodic and box meshes start at the origin."""
    domain = make_domain()
    assert get_domain_bounds({'type': 'line'}, domain, ()) == ((0.0, 2.0),)
    assert get_domain_bounds({'type': 'rectangle'}, domain, ()) == ((0.0, 2.0), (-2.5, 0.5))
    assert get_domain_bounds({'type': 'rectangle'}, domain, ('x',)) == ((0.0, 2.0), (0.0, 3.0))
    assert get_domain_bounds({'type': 'box'}, domain, ()) == ((0.0, 2.0), (0.0, 3.0), (0.0, 0.5))
    assert get_domain_bounds({'type': 'file'}, domain, ()) is None


def test_set_dimension_file_requires_mesh():
    """The dimension of a 'file' mesh cannot be derived from its type alone."""
    with pytest.raises(ValueError):
        set_dimension({'mesh': {'type': 'file'}})


@pytest.mark.parametrize("periodic, simplicial", [
    ([], False), (['x'], False), (['y'], False), (['x', 'y'], False),
    ([], True), (['x'], True), (['y'], True), (['x', 'y'], True),
])
def test_rectangle_partial_periodicity(periodic, simplicial):
    """The boundary length equals the perimeter of the non-periodic faces only.

    Simplicial meshes are only built and annotated (no DeRhamComplex): the Q family has no
    triangle elements yet (simplicial families arrive in TODO-elliptic §4.9).
    """
    domain = make_domain()
    parameters = make_parameters('rectangle', periodic, simplicial=simplicial)
    if simplicial:
        mesh = build_mesh(parameters, domain)
        annotate_mesh(mesh, parameters, domain)
    else:
        mesh, spaces = get_mesh_and_spaces(parameters, domain)
        assert parameters['mesh']['dim'] == 2
        assert spaces.periodic_directions == tuple(periodic)
    assert mesh.periodic_directions == tuple(periodic)
    expected = (0.0 if 'y' in periodic else 2 * domain.Lx) + (0.0 if 'x' in periodic else 2 * domain.Ly)
    assert abs(assemble(Constant(1.0) * ds(domain=mesh)) - expected) < 1e-12
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - domain.Lx * domain.Ly) < 1e-12
    if periodic:
        assert mesh.dx == domain.Lx / parameters['mesh']['nx']
        assert mesh.dy == domain.Ly / parameters['mesh']['ny']


@pytest.mark.parametrize("periodic", [
    [], ['x'], ['y'], ['z'], ['x', 'y'], ['x', 'z'], ['y', 'z'], ['x', 'y', 'z'],
])
def test_hex_box_is_extruded_for_all_periodicities(periodic):
    """Hexahedral Q boxes are extruded meshes (NCE/NCF available) for all 8 periodicities."""
    domain = make_domain()
    parameters = make_parameters('box', periodic)
    mesh, spaces = get_mesh_and_spaces(parameters, domain)
    assert parameters['mesh']['dim'] == 3
    assert mesh.cell_dimension() == (2, 1)
    assert spaces.Hcurl is not None and spaces.Hdiv is not None
    assert spaces.periodic_directions == tuple(periodic)
    assert spaces.domain_bounds == ((0.0, domain.Lx), (0.0, domain.Ly), (0.0, domain.Lz))
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - domain.Lx * domain.Ly * domain.Lz) < 1e-12
    # Lateral boundary area: non-periodic x faces contribute 2*Ly*Lz, y faces 2*Lx*Lz.
    lateral = (0.0 if 'x' in periodic else 2 * domain.Ly * domain.Lz) + (0.0 if 'y' in periodic else 2 * domain.Lx * domain.Lz)
    assert abs(assemble(Constant(1.0) * ds_v(domain=mesh)) - lateral) < 1e-12
    if 'z' not in periodic:
        cap = domain.Lx * domain.Ly
        assert abs(assemble(Constant(1.0) * ds_t(domain=mesh)) - cap) < 1e-12
        assert abs(assemble(Constant(1.0) * ds_b(domain=mesh)) - cap) < 1e-12
    # k-form spaces come from the existing helper.
    for k, expected in enumerate([spaces.CG, spaces.Hcurl, spaces.Hdiv, spaces.DG]):
        assert get_diff_form_space(spaces, 'S', k, 3, False) is expected


@pytest.mark.parametrize("periodic", [['x'], ['y'], ['z'], ['x', 'y'], ['x', 'z'], ['y', 'z']])
def test_tet_box_partial_periodicity_not_supported(periodic):
    """Firedrake's PeriodicBoxMesh only supports partial periodicity for hexahedral cells."""
    parameters = make_parameters('box', periodic, simplicial=True, nx=4, ny=3, nz=3)
    with pytest.raises(NotImplementedError):
        build_mesh(parameters, make_domain())


@pytest.mark.parametrize("simplicial", [True])
@pytest.mark.parametrize("periodic", [[], ['x', 'y', 'z']])
def test_tet_box_bounded_or_fully_periodic(periodic, simplicial):
    """Tet boxes are either bounded (BoxMesh) or fully periodic (PeriodicBoxMesh)."""
    domain = make_domain()
    parameters = make_parameters('box', periodic, simplicial=simplicial, nx=4, ny=3, nz=3)
    mesh = build_mesh(parameters, domain)
    annotate_mesh(mesh, parameters, domain)
    assert mesh.periodic_directions == tuple(periodic)
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - domain.Lx * domain.Ly * domain.Lz) < 1e-12
    expected = 0.0 if periodic else 2 * (domain.Ly * domain.Lz + domain.Lx * domain.Lz + domain.Lx * domain.Ly)
    assert abs(assemble(Constant(1.0) * ds(domain=mesh)) - expected) < 1e-12


@pytest.mark.parametrize("periodic", [
    [], ['x'], ['y'], ['z'], ['x', 'y'], ['x', 'z'], ['y', 'z'], ['x', 'y', 'z'],
])
@pytest.mark.parametrize("simplicial, family", [(False, 'S-')])
def test_nonextruded_box_partial_periodicity(periodic, simplicial, family):
    """Genuine hex boxes (serendipity families) support all 8 periodicities.

    Only the mesh is built (no DeRhamComplex: the serendipity families do not exist yet).
    PeriodicBoxMesh needs >= 3 cells per direction.
    """
    domain = make_domain()
    parameters = make_parameters('box', periodic, simplicial=simplicial, nx=4, ny=3, nz=3)
    parameters['spatial-discretization']['family'] = family
    mesh = build_mesh(parameters, domain)
    annotate_mesh(mesh, parameters, domain)
    assert not isinstance(mesh.cell_dimension(), tuple)  # not extruded
    assert mesh.periodic_directions == tuple(periodic)
    assert mesh.domain_bounds == ((0.0, domain.Lx), (0.0, domain.Ly), (0.0, domain.Lz))
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - domain.Lx * domain.Ly * domain.Lz) < 1e-12
    # Boundary area: each non-periodic direction contributes its two faces.
    face_area = {'x': domain.Ly * domain.Lz, 'y': domain.Lx * domain.Lz, 'z': domain.Lx * domain.Ly}
    expected = sum(2 * area for d, area in face_area.items() if d not in periodic)
    assert abs(assemble(Constant(1.0) * ds(domain=mesh)) - expected) < 1e-12


@pytest.mark.parametrize("mesh_type, periodic", [('line', None), ('line-periodic', None), ('line', ['x'])])
def test_line_meshes(mesh_type, periodic):
    """1D meshes are either fully periodic or bounded; 0/1-forms are CG/DG."""
    domain = make_domain()
    parameters = make_parameters(mesh_type, periodic)
    mesh, spaces = get_mesh_and_spaces(parameters, domain)
    assert parameters['mesh']['dim'] == 1
    expected_periodic = ('x',) if (periodic or mesh_type == 'line-periodic') else ()
    assert spaces.periodic_directions == expected_periodic
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - domain.Lx) < 1e-12
    assert get_diff_form_space(spaces, 'S', 0, 1, False) is spaces.CG
    assert get_diff_form_space(spaces, 'S', 1, 1, False) is spaces.DG


def test_reused_mesh_is_annotated():
    """Passing an existing mesh records the same metadata as building it."""
    domain = make_domain()
    parameters = make_parameters('rectangle', ['x'])
    mesh, _ = get_mesh_and_spaces(parameters, domain)
    for attr in ('periodic_directions', 'domain_bounds', 'dx', 'dy'):
        delattr(mesh, attr)
    mesh2, spaces2 = get_mesh_and_spaces(parameters, domain, mesh=mesh)
    assert mesh2 is mesh
    assert spaces2.periodic_directions == ('x',)
    assert mesh.dx == domain.Lx / parameters['mesh']['nx']


def test_mesh_size():
    """mesh_size returns the cell diagonal of a uniform quadrilateral mesh."""
    domain = make_domain(Lx=1.0, Ly=1.0)
    mesh, _ = get_mesh_and_spaces(make_parameters('rectangle', nx=4, ny=4), domain)
    assert abs(mesh_size(mesh) - (2 * 0.25**2) ** 0.5) < 1e-10
