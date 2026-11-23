"""The MeshSpec registry and Firedrake's utility meshes (TODO-elliptic §4.14, stages 1-2)."""
from types import SimpleNamespace

import pytest
from firedrake import Constant, SpatialCoordinate, assemble, ds, ds_b, ds_t, ds_v, dx, sqrt
from math import pi

from dimswe.boundary_conditions import get_boundary_registry, is_extruded, measure_for, parse_boundary_conditions
from dimswe.meshes import (BOX_MESH_TYPES, LINE_MESH_TYPES, MESH_TYPES, RECTANGLE_MESH_TYPES, build_mesh,
                           get_domain_bounds, get_mesh_spec, get_mesh_type_dimension, get_periodic_directions,
                           get_mesh_and_spaces, is_box_mesh_type)

DOMAIN = SimpleNamespace(Lx=2.0, Ly=3.0, Lz=0.5, xc=1.0, yc=-1.0, zc=0.0)


def parameters(mesh_type, simplicial=False, family='Q', **mesh):
    """Minimal parameters for build_mesh / get_mesh_and_spaces."""
    mesh_cfg = {'type': mesh_type, 'nx': 4, 'ny': 3, 'nz': 3, 'simplicial_cells': simplicial}
    mesh_cfg.update(mesh)
    return {'mesh': mesh_cfg, 'spatial-discretization': {'family': family, 'order': 1, 'lump_mass': False}}


def test_registry_backwards_compatible():
    """The historic types and grouping tuples are unchanged."""
    for name in ('line', 'line-periodic'):
        assert name in LINE_MESH_TYPES and get_mesh_type_dimension(name) == 1
    for name in ('rectangle', 'rectangle-periodic'):
        assert name in RECTANGLE_MESH_TYPES and get_mesh_type_dimension(name) == 2
    for name in ('box', 'box-periodic'):
        assert name in BOX_MESH_TYPES and get_mesh_type_dimension(name) == 3
    assert get_mesh_type_dimension('file') is None
    assert get_periodic_directions({'type': 'rectangle-periodic'}) == ('x', 'y')
    assert get_periodic_directions({'type': 'rectangle'}) == ()
    with pytest.raises(ValueError):
        get_mesh_spec('hexagon')


def test_box_aliases():
    """Unit aliases use the unit box; periodic aliases default to all directions; others the domain."""
    assert get_domain_bounds({'type': 'unit-square'}, DOMAIN, ()) == ((0.0, 1.0), (0.0, 1.0))
    assert get_domain_bounds({'type': 'unit-cube'}, DOMAIN, ()) == ((0.0, 1.0),) * 3
    assert get_domain_bounds({'type': 'square'}, DOMAIN, ()) == ((0.0, 2.0), (-2.5, 0.5))
    assert get_periodic_directions({'type': 'periodic-unit-square'}) == ('x', 'y')
    assert get_periodic_directions({'type': 'periodic-unit-square', 'periodic_directions': ['x']}) == ('x',)
    assert is_box_mesh_type('unit-square') and not is_box_mesh_type('annulus') and not is_box_mesh_type('file')
    with pytest.raises(ValueError):
        get_periodic_directions({'type': 'annulus', 'periodic_directions': ['x']})


@pytest.mark.parametrize("mesh_type, simplicial", [
    ('unit-interval', False), ('periodic-unit-interval', False),
    ('unit-square', False), ('unit-square', True), ('periodic-unit-square', False), ('square', False),
    ('periodic-square', True), ('unit-cube', False), ('unit-cube', True), ('periodic-unit-cube', False), ('cube', False),
])
def test_box_aliases_build(mesh_type, simplicial):
    """Every box alias builds, has the expected volume and keeps the face registry."""
    prm = parameters(mesh_type, simplicial=simplicial)
    mesh, spaces = get_mesh_and_spaces(prm, DOMAIN) if not simplicial else (build_mesh(prm, DOMAIN), None)
    if simplicial:
        from dimswe.meshes import annotate_mesh
        annotate_mesh(mesh, prm, DOMAIN)
    bounds = mesh.domain_bounds
    volume = 1.0
    for lo, hi in bounds:
        volume *= hi - lo
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - volume) < 1e-10
    registry = get_boundary_registry(mesh, mesh_type)
    assert len(registry) == 2 * (len(bounds) - len(mesh.periodic_directions))


def test_tensor_meshes():
    """Graded boxes from coordinate lists; bounds come from the coordinates."""
    prm = parameters('tensor-rectangle', x_coords=[0.0, 0.1, 0.4, 1.0], y_coords=[-1.0, 0.0, 2.0])
    mesh, _ = get_mesh_and_spaces(prm, DOMAIN)
    assert mesh.domain_bounds == ((0.0, 1.0), (-1.0, 2.0))
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - 3.0) < 1e-12
    assert set(get_boundary_registry(mesh, 'tensor-rectangle')) == {'left', 'right', 'bottom', 'top'}
    with pytest.raises(ValueError):
        build_mesh(parameters('tensor-rectangle', x_coords=[0.0, 0.0, 1.0], y_coords=[0.0, 1.0]), DOMAIN)
    with pytest.raises(ValueError):
        build_mesh(parameters('tensor-rectangle', x_coords=[0.0, 1.0]), DOMAIN)
    box = build_mesh(parameters('tensor-box', simplicial=True, x_coords=[0, 1], y_coords=[0, 0.5, 1],
                                z_coords=[0, 2]), DOMAIN)
    assert abs(assemble(Constant(1.0) * dx(domain=box)) - 2.0) < 1e-12
    with pytest.raises(NotImplementedError):
        build_mesh(parameters('tensor-box', x_coords=[0, 1], y_coords=[0, 1], z_coords=[0, 1]), DOMAIN)


@pytest.mark.parametrize("mesh_type, simplicial, faces, dim", [
    ('unit-disk', True, ['outer'], 2),
    ('annulus', False, ['inner', 'outer'], 2),
    ('unit-ball', True, ['outer'], 3),
])
def test_radial_meshes_and_faces(mesh_type, simplicial, faces, dim):
    """Radial meshes build, have no box bounds, and their faces sit at the right radius."""
    prm = parameters(mesh_type, simplicial=simplicial, family='P-' if simplicial else 'Q', nr=4, nt=16)
    mesh = build_mesh(prm, DOMAIN)
    from dimswe.meshes import annotate_mesh
    annotate_mesh(mesh, prm, DOMAIN)
    assert mesh.domain_bounds is None and mesh.mesh_topology == 'radial'
    registry = get_boundary_registry(mesh, mesh_type)
    assert list(registry) == faces
    X = SpatialCoordinate(mesh)
    r = sqrt(sum(X[i] ** 2 for i in range(dim)))
    expected = {'outer': 1.0, 'inner': 0.5}
    for name, face in registry.items():
        measure = measure_for(face, domain=mesh)
        mean = assemble(r * measure) / assemble(Constant(1.0) * measure)
        # Polygonal/polyhedral approximations of circles and spheres sit slightly inside them.
        assert abs(mean - expected[name]) < 5e-2, (name, mean)
    total = sum(assemble(Constant(1.0) * measure_for(f, domain=mesh)) for f in registry.values())
    if is_extruded(mesh):
        # The quadrilateral annulus is a radial interval extruded periodically around the circle:
        # its boundary is the side (ds_v); bottom/top layers only count if the extrusion is not periodic.
        boundary = assemble(Constant(1.0) * ds_v(domain=mesh))
        if not getattr(mesh, 'extruded_periodic', False):
            boundary += sum(assemble(Constant(1.0) * m(domain=mesh)) for m in (ds_b, ds_t))
    else:
        boundary = assemble(Constant(1.0) * ds(domain=mesh))
    assert abs(total - boundary) < 1e-10
    assert parse_boundary_conditions({name: 'essential' for name in faces}, registry) == \
        {name: 'essential' for name in faces}
    with pytest.raises(ValueError):
        parse_boundary_conditions({'left': 'essential'}, registry)


def test_radial_volume_approximates_disk():
    """The polygonal unit disk approaches pi under refinement."""
    errors = []
    for refinement in (2, 3):
        mesh = build_mesh(parameters('unit-disk', simplicial=True, refinement=refinement), DOMAIN)
        errors.append(abs(assemble(Constant(1.0) * dx(domain=mesh)) - pi))
    assert errors[1] < errors[0]


def test_radial_mesh_rules():
    """Simplicial-only meshes reject quads; hierarchies need non-extruded coarse meshes."""
    with pytest.raises(ValueError):
        build_mesh(parameters('unit-disk', simplicial=False), DOMAIN)
    with pytest.raises(NotImplementedError):   # the quadrilateral annulus is extruded
        build_mesh(parameters('annulus', simplicial=False, refinement_levels=1, nr=4, nt=16), DOMAIN)
    with pytest.raises(ValueError):
        build_mesh(parameters('annulus', inner_radius=1.0, outer_radius=0.5), DOMAIN)
    assert all(spec.topology in ('box', 'radial', 'manifold') for spec in MESH_TYPES.values())
