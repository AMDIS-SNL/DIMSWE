"""Refinement hierarchies on curved and manifold meshes (TODO-elliptic §4.14 stage 4).

- every refined level of a disk/ball/sphere/torus/cylinder hierarchy is snapped to the exact
  geometry (so the geometric error shrinks with the level, unlike plain uniform refinement);
- manifold levels keep outward cell orientations;
- riesz-gmg on these hierarchies converges, matches the direct solve, and its iteration count
  stays bounded under refinement;
- extruded coarse meshes (the quadrilateral annulus) reject refinement_levels.
"""
import copy
import math

import numpy
import pytest
from firedrake import COMM_WORLD, CellNormal, Constant, SpatialCoordinate, assemble, dot, dx, inner, sqrt

from dimswe.elliptic import EllipticProblem
from dimswe.meshes import build_mesh

pytestmark = pytest.mark.skipif(COMM_WORLD.size > 1, reason="snapping checks read rank-local coordinates")


def mesh_parameters(mesh_type, simplicial=True, family='P-', levels=0, **mesh):
    """Parameters of a curved mesh with refinement_levels."""
    cfg = {'type': mesh_type, 'simplicial_cells': simplicial, 'refinement_levels': levels}
    cfg.update(mesh)
    return {'mesh': cfg, 'spatial-discretization': {'family': family, 'order': 1, 'lump_mass': False}}


def max_radius_error(mesh, radius=1.0, boundary_only=False, axis=None):
    """max | |x| - radius | over the vertices (or over the boundary vertices)."""
    from dimswe.meshes import _boundary_vertex_nodes
    coords = mesh.coordinates.dat.data_ro
    if boundary_only:
        coords = coords[_boundary_vertex_nodes(mesh)]
    values = numpy.linalg.norm(coords if axis is None else coords[:, :axis], axis=1)
    return float(numpy.max(numpy.abs(values - radius)))


@pytest.mark.parametrize("mesh_type, simplicial, family, extra, boundary_only, axis", [
    ('icosahedral-sphere', True, 'P-', {'refinement': 1}, False, None),
    ('octahedral-sphere', True, 'P-', {'refinement': 1}, False, None),
    ('cubed-sphere', False, 'Q', {'refinement': 1}, False, None),
    ('unit-disk', True, 'P-', {'refinement': 1}, True, None),
    ('unit-ball', True, 'P-', {'refinement': 0}, True, None),
    ('cylinder-surface', False, 'Q', {'nr': 8, 'nl': 2}, False, 2),
])
def test_levels_are_snapped(mesh_type, simplicial, family, extra, boundary_only, axis):
    """The finest level lies on the exact surface/boundary (to round-off), and the area/volume
    error decreases with refinement instead of staying at the coarse polygon's."""
    mesh = build_mesh(mesh_parameters(mesh_type, simplicial, family, levels=2, **extra), None)
    assert mesh.refinement_levels == 2 and len(mesh.mesh_hierarchy) == 3
    assert max_radius_error(mesh, boundary_only=boundary_only, axis=axis) < 1e-12
    if mesh_type.endswith('sphere'):
        areas = [assemble(Constant(1.0) * dx(domain=level)) for level in mesh.mesh_hierarchy]
        errors = [abs(a - 4 * math.pi) for a in areas]
        assert errors[2] < errors[1] < errors[0], errors
        X = SpatialCoordinate(mesh)
        assert assemble(dot(CellNormal(mesh), X / sqrt(dot(X, X))) * dx) > 0.95 * 4 * math.pi


def test_torus_snapping():
    """Every torus vertex ends up on the tube of radius r around the core circle."""
    mesh = build_mesh(mesh_parameters('torus', True, 'P-', levels=1, nR=8, nr=4, R=2.0, r=0.5), None)
    coords = mesh.coordinates.dat.data_ro
    rho = numpy.linalg.norm(coords[:, :2], axis=1)
    tube = numpy.sqrt((rho - 2.0) ** 2 + coords[:, 2] ** 2)
    assert numpy.max(numpy.abs(tube - 0.5)) < 1e-12


def test_extruded_coarse_mesh_rejected():
    """The quadrilateral annulus is extruded; refinement_levels needs simplicial cells there."""
    with pytest.raises(NotImplementedError):
        build_mesh(mesh_parameters('annulus', simplicial=False, family='Q', levels=1, nr=4, nt=16), None)


def gmg_parameters(mesh_type, k, complex, bcs, levels, problem, **mesh):
    """riesz-gmg elliptic parameters on a curved hierarchy."""
    parameters = mesh_parameters(mesh_type, levels=levels, **mesh)
    parameters['spatial-discretization']['order'] = 1
    parameters.update({'model': {'type': 'elliptic'},
                       'elliptic': {'form_degree': k, 'complex': complex, 'boundary_conditions': bcs,
                                    'boundary_data': 'exact', 'harmonic_forms': 'auto',
                                    'harmonic_constraint': 'nullspace', 'solver': 'riesz-gmg', 'ksp_rtol': 1e-9},
                       'elliptic-problem': {'name': problem}, 'output': {'loglevel': 0}})
    return parameters


def solve(parameters, mesh=None):
    """Solve and compute statistics."""
    model = EllipticProblem(parameters, mesh=mesh)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    return model


@pytest.mark.parametrize("mesh_type, k, complex, bcs, problem, extra", [
    ('unit-disk', 0, 'grad-curl', {'outer': 'essential'}, 'radial', {'refinement': 1}),
    ('unit-disk', 1, 'curl-div', {'outer': 'natural'}, 'radial', {'refinement': 1}),
    ('icosahedral-sphere', 0, 'grad-curl', {}, 'spherical-harmonic', {'refinement': 1}),
    ('icosahedral-sphere', 1, 'curl-div', {}, 'spherical-harmonic', {'refinement': 1}),
])
def test_gmg_on_curved_hierarchy(mesh_type, k, complex, bcs, problem, extra):
    """riesz-gmg converges on a two-level curved hierarchy and matches the direct solve."""
    model = solve(gmg_parameters(mesh_type, k, complex, bcs, 2, problem, **extra))
    assert model.ksp_reason > 0 and model.ksp_iterations < 200
    direct_parameters = copy.deepcopy(model.parameters)
    direct_parameters['elliptic']['solver'] = 'elliptic-direct'
    direct = solve(direct_parameters, mesh=model.mesh)
    for name in direct.get_x_var_list():
        diff = model.sub[name] - direct.sub[name]
        assert assemble(inner(diff, diff) * dx) ** 0.5 < 1e-6 * max(1.0, assemble(
            inner(direct.sub[name], direct.sub[name]) * dx) ** 0.5), name


@pytest.mark.parametrize("mesh_type, k, complex, bcs, problem", [
    ('unit-disk', 1, 'curl-div', {'outer': 'natural'}, 'radial'),
    ('icosahedral-sphere', 1, 'curl-div', {}, 'spherical-harmonic'),
])
def test_gmg_iterations_bounded_on_curved_hierarchy(mesh_type, k, complex, bcs, problem):
    """Iteration counts stay bounded over three hierarchy depths."""
    counts = []
    for levels in (1, 2, 3):
        parameters = gmg_parameters(mesh_type, k, complex, bcs, levels, problem, refinement=1)
        parameters['elliptic']['ksp_rtol'] = 1e-8
        model = solve(parameters)
        assert model.ksp_reason > 0
        counts.append(model.ksp_iterations)
    assert counts[-1] <= 1.5 * counts[0] + 5, counts
