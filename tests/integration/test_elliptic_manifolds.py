"""Immersed 2-manifolds (TODO-elliptic §4.14 stage 3): spheres, torus, cylinder surface.

The operators on surfaces use the outward cell normal (dimswe.differential_forms.surface_normal):
skewgrad = n x grad, curl2D v = div(v x n), a x b = n . (a x b). These tests check
- every mesh builds, is a 2-manifold in 3D with outward cell orientations, and has the right area;
- d^2 = 0 on discrete fields and the discrete integration-by-parts identity of both complexes
  (closed surfaces have no boundary terms), which pins down every sign;
- Betti numbers (sphere 1, 0, 1; torus 1, 2, 1) and the numerical harmonic forms;
- MMS convergence on the sphere for k = 0, 1, 2 (spherical harmonic, closed-form fields);
- the k = 0 eigenvalues of the sphere, l (l + 1) / R^2 with multiplicity 2 l + 1;
- the open cylinder surface with bottom/top boundary circles.
"""
import math

import numpy
import pytest
from firedrake import (COMM_WORLD, CellNormal, Constant, Function, SpatialCoordinate, assemble, dot, dx,
                       inner, sqrt)

from dimswe.boundary_conditions import get_boundary_registry
from dimswe.differential_forms import exterior_derivative, form_space
from dimswe.elliptic import EllipticConfig, EllipticProblem
from dimswe.elliptic_eigen import HodgeLaplacianEigenproblem
from dimswe.meshes import build_mesh, get_mesh_and_spaces
from dimswe.numerical_harmonic import numerical_harmonic_basis
from dimswe.ufl_helpers import ufl_attribute
from dimswe.topology import betti_number

pytestmark = pytest.mark.skipif(COMM_WORLD.size > 1, reason="serial Betti numbers")

MESHES = [
    ('icosahedral-sphere', True, 'P-', 4 * math.pi),
    ('octahedral-sphere', True, 'P-', 4 * math.pi),
    ('cubed-sphere', False, 'Q', 4 * math.pi),
    ('torus', False, 'Q', 4 * math.pi ** 2 * 2.0 * 0.5),
    ('torus', True, 'P-', 4 * math.pi ** 2 * 2.0 * 0.5),
]


def mesh_parameters(mesh_type, simplicial, family, order=1, **mesh):
    """Parameters for a manifold mesh."""
    cfg = {'type': mesh_type, 'simplicial_cells': simplicial}
    cfg.update(mesh)
    return {'mesh': cfg, 'spatial-discretization': {'family': family, 'order': order, 'lump_mass': False}}


def elliptic_parameters(mesh_type, k, complex, simplicial=True, family='P-', order=1, bcs=None, **mesh):
    """Elliptic parameters on a manifold with the spherical-harmonic problem."""
    parameters = mesh_parameters(mesh_type, simplicial, family, order, **mesh)
    parameters.update({'model': {'type': 'elliptic'},
                       'elliptic': {'form_degree': k, 'complex': complex, 'boundary_conditions': bcs or {},
                                    'boundary_data': 'exact', 'harmonic_forms': 'auto',
                                    'harmonic_constraint': 'nullspace', 'solver': 'elliptic-direct'},
                       'elliptic-problem': {'name': 'spherical-harmonic'},
                       'output': {'loglevel': 0}})
    return parameters


@pytest.mark.parametrize("mesh_type, simplicial, family, area", MESHES)
def test_manifold_meshes(mesh_type, simplicial, family, area):
    """Builds; 2-manifold in 3D; outward normals; area close to the smooth surface's."""
    mesh, spaces = get_mesh_and_spaces(mesh_parameters(mesh_type, simplicial, family, refinement=3), None)
    assert spaces.dim == 2 and ufl_attribute(mesh, 'geometric_dimension') == 3
    assert mesh.mesh_topology == 'manifold' and mesh.domain_bounds is None
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - area) < 0.05 * area
    X = SpatialCoordinate(mesh)
    n = CellNormal(mesh)
    if mesh_type.endswith('sphere'):
        outward = assemble(dot(n, X / sqrt(dot(X, X))) * dx)
        assert outward > 0.95 * area
    assert get_boundary_registry(mesh, mesh_type) == {}


@pytest.mark.parametrize("cx", ['grad-curl', 'curl-div'])
@pytest.mark.parametrize("mesh_type, simplicial, family", [('icosahedral-sphere', True, 'P-'),
                                                          ('cubed-sphere', False, 'Q')])
def test_operator_identities_on_sphere(mesh_type, simplicial, family, cx):
    """d^2 = 0 for random discrete w, and (d w, z) = (w, delta z) for discrete w, z (closed surface)."""
    mesh, spaces = get_mesh_and_spaces(mesh_parameters(mesh_type, simplicial, family, order=2, refinement=2), None)
    rng = numpy.random.default_rng(3)
    w0 = Function(form_space(spaces, 0, 2, cx))
    w0.dat.data[:] = rng.standard_normal(w0.dat.data.shape)
    dw = exterior_derivative(w0, 0, 2, cx)
    ddw = exterior_derivative(dw, 1, 2, cx)
    assert assemble(inner(dw, dw) * dx) > 1e-6
    assert assemble(inner(ddw, ddw) * dx) < 1e-14 * assemble(inner(dw, dw) * dx)
    # d_0 w is tangential (curl-div: n x grad w; grad-curl: the surface gradient).
    n = CellNormal(mesh)
    assert assemble(dot(dw, n) ** 2 * dx) < 1e-14 * assemble(inner(dw, dw) * dx)
    # Integration by parts on a closed surface (no boundary): (d_1 v, q) + ... the k = 1 -> 2 pairing
    # (d v, q) = (v, delta q) with delta_2 q = -skewgrad q (grad-curl) or -grad q (curl-div), all discrete.
    v = Function(form_space(spaces, 1, 2, cx))
    v.dat.data[:] = rng.standard_normal(v.dat.data.shape)
    q = Function(form_space(spaces, 0, 2, cx))   # a smooth (CG) test field for the top-form pairing
    q.dat.data[:] = rng.standard_normal(q.dat.data.shape)
    from dimswe.differential_forms import codifferential
    lhs = assemble(inner(exterior_derivative(v, 1, 2, cx), q) * dx)
    rhs = assemble(inner(v, codifferential(q, 2, 2, cx)) * dx)
    assert abs(lhs - rhs) < 1e-9 * max(1.0, abs(lhs))


def test_betti_numbers_and_harmonic_forms():
    """Sphere: b = (1, 0, 1); torus: b = (1, 2, 1); numerical harmonic 1-forms on the torus."""
    for mesh_type, simplicial, family, expected in (('icosahedral-sphere', True, 'P-', {0: 1, 1: 0, 2: 1}),
                                                    ('torus', False, 'Q', {0: 1, 1: 2, 2: 1})):
        parameters = mesh_parameters(mesh_type, simplicial, family, order=1, refinement=2, nR=12, nr=6)
        mesh, spaces = get_mesh_and_spaces(parameters, None)
        parameters['mesh']['dim'] = 2
        for k, count in expected.items():
            config = EllipticConfig(dim=2, k=k, complex='curl-div', boundary_conditions={}, registry={}, face_bcs={})
            assert betti_number(mesh, parameters, config) == count, (mesh_type, k)
            if mesh_type == 'torus' and k == 1:
                basis, info = numerical_harmonic_basis(spaces, config, mesh, parameters,
                                                       form_space(spaces, 1, 2, 'curl-div'))
                assert len(basis) == 2 and info.gap_ratio > 1e6 and info.max_sigma_ratio < 1e-6
                n = CellNormal(mesh)
                for h in basis:
                    assert abs(assemble(dot(h, n) * dx)) < 1e-8   # tangential


@pytest.mark.parametrize("k, complex", [(0, 'grad-curl'), (0, 'curl-div'), (1, 'curl-div'), (2, 'curl-div')])
def test_sphere_mms(k, complex):
    """Spherical-harmonic MMS on the icosahedral sphere (order 1): u converges at >= ~2 (k = 0) / ~1."""
    errors = []
    for refinement in (2, 3):
        model = EllipticProblem(elliptic_parameters('icosahedral-sphere', k, complex, refinement=refinement,
                                                    degree=2))
        model.solve()
        model.create_statistics()
        model.compute_statistics()
        errors.append(model.get_statistics()['u_error'][0])
    rate = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert rate >= (1.7 if k == 0 else 0.7), (errors, rate)


def test_sphere_laplace_beltrami_spectrum():
    """k = 0 eigenvalues on the unit sphere: 0 (x1), 2 (x3), 6 (x5), within discretization error."""
    parameters = elliptic_parameters('icosahedral-sphere', 0, 'grad-curl', order=2, refinement=3, degree=2)
    parameters['model']['type'] = 'elliptic-eigen'
    parameters['elliptic']['eigen'] = {'nev': 9, 'target': -1.0, 'tolerance': 1e-10}
    model = HodgeLaplacianEigenproblem(parameters)
    model.solve()
    expected = [0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    assert numpy.allclose(model.eigenvalues, expected, atol=2e-2), model.eigenvalues


def test_cylinder_surface_faces():
    """The open cylinder surface has bottom and top boundary circles at z = 0 and z = height."""
    parameters = mesh_parameters('cylinder-surface', False, 'Q', nr=16, nl=4, height=2.0)
    mesh, _ = get_mesh_and_spaces(parameters, None)
    registry = get_boundary_registry(mesh, 'cylinder-surface')
    assert list(registry) == ['bottom', 'top']
    X = SpatialCoordinate(mesh)
    from dimswe.boundary_conditions import measure_for
    z = {name: assemble(X[2] * measure_for(face, domain=mesh)) / assemble(Constant(1.0) * measure_for(face, domain=mesh))
         for name, face in registry.items()}
    assert z['bottom'] < z['top']


def test_manifold_mesh_rules():
    """Cell-type restrictions (hierarchies on surfaces are tested in test_curved_hierarchies.py)."""
    with pytest.raises(ValueError):
        build_mesh(mesh_parameters('icosahedral-sphere', False, 'Q'), None)
    with pytest.raises(ValueError):
        build_mesh(mesh_parameters('cubed-sphere', True, 'P-'), None)
    with pytest.raises(ValueError, match="spherical-harmonic"):
        EllipticProblem(dict(elliptic_parameters('icosahedral-sphere', 0, 'grad-curl', refinement=1),
                             **{'elliptic-problem': {'name': 'sine'}}))


def test_sphere_constant_coefficients():
    """Constant mu, kappa, c rescale the closed-form fields: the error converges like the unweighted
    case (a wrong scaling would make the source inconsistent and the error stall). Variable
    coefficients raise."""
    errors = []
    for refinement in (2, 3):
        parameters = elliptic_parameters('icosahedral-sphere', 1, 'curl-div', refinement=refinement, degree=2)
        parameters['elliptic']['coefficients'] = {'mu': {'value': 2.0}, 'c': {'value': 0.5}}
        model = EllipticProblem(parameters)
        model.solve()
        model.create_statistics()
        model.compute_statistics()
        errors.append(model.get_statistics()['u_error'][0])
    rate = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert rate >= 0.7, (errors, rate)
    parameters['elliptic']['coefficients'] = {'mu': {'kind': 'problem'}}
    with pytest.raises(NotImplementedError):
        EllipticProblem(parameters)
