"""Unit tests for dimswe/harmonic_forms.py (TODO-elliptic §1.4, §1.5, §4.4, §6)."""
from math import comb, pi, sqrt
from types import SimpleNamespace

import numpy
import pytest
from firedrake import (Cofunction, Function, LinearVariationalProblem, LinearVariationalSolver,
                       MixedFunctionSpace, SpatialCoordinate, TestFunction,
                       TestFunctions, TrialFunction, TrialFunctions, as_vector, assemble, cos, dx,
                       grad, inner, sin, solve)

from dimswe.boundary_conditions import (build_essential_bcs, faces_of_type, face_bcs_by_direction,
                                        get_boundary_registry, parse_boundary_conditions)
from dimswe.differential_forms import exterior_derivative, form_space
from dimswe.harmonic_forms import (build_harmonic_nullspace, combine_harmonic_forms,
                                   get_harmonic_index_sets, harmonic_basis,
                                   harmonic_constraint_form, harmonic_index_sets,
                                   harmonic_multiplier_space, l2_inner,
                                   lagrange_preconditioner_form, lagrange_solver_parameters,
                                   multiplier_coefficients,
                                   orthonormalize_harmonic_basis, project_rhs_orthogonal_to_harmonic,
                                   remove_harmonic_component)
from dimswe.meshes import get_mesh_and_spaces

MESH_FOR_DIM = {1: 'line', 2: 'rectangle', 3: 'box'}
LU = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
SINGULAR_LU = {'ksp_type': 'gmres', 'ksp_rtol': 1e-12, 'ksp_atol': 1e-14,
               'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'mat_mumps_icntl_24': 1}


# ---------------------------------------------------------------------------------------------
# Selection rule (pure Python)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_harmonic_index_sets_fully_periodic(dim):
    """Fully periodic: every dx^I, i.e. C(n, k) forms of degree k."""
    periodic = ('x', 'y', 'z')[:dim]
    for k in range(dim + 1):
        sets = harmonic_index_sets(k, dim, periodic, {})
        assert len(sets) == comb(dim, k)
        assert all(len(I) == k for I in sets)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_harmonic_index_sets_bounded(dim):
    """All-natural: constants at k = 0 only; all-essential: the volume form at k = n only."""
    names = ['left', 'right', 'bottom', 'top', 'back', 'front'][:2 * dim]
    natural = {name: 'natural' for name in names}
    essential = {name: 'essential' for name in names}
    for k in range(dim + 1):
        assert harmonic_index_sets(k, dim, (), natural) == ([()] if k == 0 else [])
        expected = [('x', 'y', 'z')[:dim]] if k == dim else []
        assert harmonic_index_sets(k, dim, (), essential) == expected


def test_harmonic_index_sets_mismatched_faces():
    """Different BC types on the two faces of a bounded direction: no harmonic forms."""
    for k in range(3):
        assert harmonic_index_sets(k, 2, ('x',), {'bottom': 'essential', 'top': 'natural'}) == []
        assert harmonic_index_sets(k, 2, (), {'left': 'essential', 'right': 'natural',
                                              'bottom': 'essential', 'top': 'essential'}) == []


@pytest.mark.parametrize("bc, expected", [
    ('natural', {0: [()], 1: [('x',)], 2: []}),
    ('essential', {0: [], 1: [('y',)], 2: [('x', 'y')]}),
])
def test_harmonic_index_sets_cylinder(bc, expected):
    """Periodic in x, bounded in y (TODO-elliptic §6 cylinder examples)."""
    for k, sets in expected.items():
        assert harmonic_index_sets(k, 2, ('x',), {'bottom': bc, 'top': bc}) == sets


def test_harmonic_index_sets_3d_partial():
    """P = {x}, E = {y}, N = {z}: I = {y} or {x, y}."""
    face_bcs = {'bottom': 'essential', 'top': 'essential', 'back': 'natural', 'front': 'natural'}
    expected = {0: [], 1: [('y',)], 2: [('x', 'y')], 3: []}
    for k, sets in expected.items():
        assert harmonic_index_sets(k, 3, ('x',), face_bcs) == sets
    # Per-direction input gives the same answer.
    by_dir = {'y': ('essential', 'essential'), 'z': ('natural', 'natural')}
    assert harmonic_index_sets(2, 3, ('x',), by_dir) == [('x', 'y')]


@pytest.mark.parametrize("periodic, face_bcs", [
    (('x',), {'left': 'natural', 'right': 'natural', 'bottom': 'natural', 'top': 'natural'}),  # BC on periodic
    ((), {'left': 'natural', 'right': 'natural', 'bottom': 'natural'}),                        # missing face
    ((), {'left': 'natural', 'right': 'natural', 'bottom': 'robin', 'top': 'robin'}),          # bad type
    ((), {'left': 'natural', 'right': 'natural', 'bottom': 'natural', 'top': 'natural',
          'back': 'natural', 'front': 'natural'}),                                             # beyond 2D
    (('z',), {}),                                                                              # bad periodic dir
])
def test_harmonic_index_sets_errors(periodic, face_bcs):
    """Inconsistent inputs raise ValueError."""
    with pytest.raises(ValueError):
        harmonic_index_sets(1, 2, periodic, face_bcs)


def test_get_harmonic_index_sets_modes():
    """'none' gives nothing, 'auto' the rule, 'numerical' and 'file' meshes are not available."""
    assert get_harmonic_index_sets('none', 0, 2, ('x', 'y'), {}, 'rectangle') == []
    assert get_harmonic_index_sets('auto', 1, 2, ('x', 'y'), {}, 'rectangle') == [('x',), ('y',)]
    with pytest.raises(NotImplementedError):
        get_harmonic_index_sets('numerical', 1, 2, ('x', 'y'), {}, 'rectangle')
    with pytest.raises(NotImplementedError):
        get_harmonic_index_sets('auto', 1, 2, (), {}, 'file')
    with pytest.raises(ValueError):
        get_harmonic_index_sets('sometimes', 1, 2, (), {}, 'rectangle')


# ---------------------------------------------------------------------------------------------
# Firedrake helpers
# ---------------------------------------------------------------------------------------------

def make_setup(dim, periodic, bc_cfg, n=3, order=1):
    """Mesh, spaces, registry and parsed BCs for a Q-family box-shaped mesh."""
    parameters = {'mesh': {'type': MESH_FOR_DIM[dim], 'nx': n, 'ny': n, 'nz': n,
                           'simplicial_cells': False, 'periodic_directions': list(periodic)},
                  'spatial-discretization': {'family': 'Q', 'order': order, 'lump_mass': False}}
    domain = SimpleNamespace(Lx=1.0, Ly=1.5, Lz=0.5, xc=0.0, yc=0.0, zc=0.0)
    mesh, spaces = get_mesh_and_spaces(parameters, domain)
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    face_bcs = parse_boundary_conditions(bc_cfg, registry)
    return mesh, spaces, registry, face_bcs


def make_basis(spaces, registry, face_bcs, mesh, k, dim, cx):
    """Orthonormal harmonic basis in V^k for the given setup."""
    sets = harmonic_index_sets(k, dim, mesh.periodic_directions, face_bcs_by_direction(registry, face_bcs))
    V = form_space(spaces, k, dim, cx)
    return sets, V, orthonormalize_harmonic_basis(harmonic_basis(k, dim, cx, sets), V)


ALL_FACES = {1: ['left', 'right'], 2: ['left', 'right', 'bottom', 'top'],
             3: ['left', 'right', 'bottom', 'top', 'back', 'front']}

# (dim, complex, periodic directions, BC per bounded face (one type for all of them))
HARMONIC_CONFIGS = [
    (1, None, ('x',), None),
    (1, None, (), 'natural'),
    (1, None, (), 'essential'),
    (2, 'grad-curl', ('x', 'y'), None),
    (2, 'curl-div', ('x', 'y'), None),
    (2, 'grad-curl', ('x',), 'natural'),
    (2, 'grad-curl', ('x',), 'essential'),
    (2, 'curl-div', ('y',), 'natural'),
    (2, 'curl-div', ('y',), 'essential'),
    (3, None, ('x', 'y', 'z'), None),
    (3, None, ('x', 'y'), 'essential'),
    (3, None, ('z',), 'natural'),
]


def config_id(config):
    """Readable pytest id."""
    dim, cx, periodic, bc = config
    return str(dim) + "D-" + (cx + "-" if cx else "") + "P" + "".join(periodic) + "-" + str(bc)


def bc_cfg_for(dim, periodic, bc):
    """The same BC type on every bounded face."""
    if bc is None:
        return {}
    axis_of = {'left': 'x', 'right': 'x', 'bottom': 'y', 'top': 'y', 'back': 'z', 'front': 'z'}
    return {name: bc for name in ALL_FACES[dim] if axis_of[name] not in periodic}


# ---------------------------------------------------------------------------------------------
# Basis properties
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("config", HARMONIC_CONFIGS, ids=config_id)
def test_harmonic_basis_is_discretely_harmonic(config):
    """For each basis form h: orthonormal, d h = 0, (h, d tau) = 0 for tau in V^{k-1} with the
    essential BCs applied, and h satisfies the homogeneous essential BCs of u."""
    dim, cx, periodic, bc = config
    mesh, spaces, registry, face_bcs = make_setup(dim, periodic, bc_cfg_for(dim, periodic, bc))
    essential = faces_of_type(registry, face_bcs, 'essential')
    for k in range(dim + 1):
        sets, V, basis = make_basis(spaces, registry, face_bcs, mesh, k, dim, cx)
        assert len(basis) == len(sets)
        gram = numpy.array([[l2_inner(a, b) for b in basis] for a in basis])
        assert numpy.allclose(gram, numpy.eye(len(basis)), atol=1e-10)
        for h in basis:
            if k < dim:
                dh = exterior_derivative(h, k, dim, cx)
                assert assemble(inner(dh, dh) * dx) < 1e-20
                # h satisfies the homogeneous essential BCs of u (no change when applied).
                g = h.copy(deepcopy=True)
                for bc_obj in build_essential_bcs(V, None, 0, essential):
                    bc_obj.apply(g)
                diff = Function(V).assign(g - h)
                assert l2_inner(diff, diff) < 1e-20
            if k > 0:
                Vm = form_space(spaces, k - 1, dim, cx)
                tau = TestFunction(Vm)
                vec = assemble(inner(h, exterior_derivative(tau, k - 1, dim, cx)) * dx)
                for bc_obj in build_essential_bcs(Vm, None, 0, essential):
                    vec.dat.data[bc_obj.nodes] = 0.0
                assert numpy.max(numpy.abs(vec.dat.data_ro)) < 1e-10


def test_orthonormalize_rejects_dependent_basis():
    """A repeated form is linearly dependent."""
    mesh, spaces, _, _ = make_setup(2, ('x', 'y'), {})
    proxies = harmonic_basis(1, 2, 'grad-curl', [('x',), ('x',)])
    with pytest.raises(ValueError):
        orthonormalize_harmonic_basis(proxies, spaces.Hcurl)


# ---------------------------------------------------------------------------------------------
# rhs projection and removal
# ---------------------------------------------------------------------------------------------

def periodic_2d_k1():
    """Fully periodic 2D grad-curl 1-forms: two harmonic forms (dx, dy)."""
    mesh, spaces, registry, face_bcs = make_setup(2, ('x', 'y'), {}, n=4, order=2)
    _, V, basis = make_basis(spaces, registry, face_bcs, mesh, 1, 2, 'grad-curl')
    return mesh, spaces, V, basis


def mean_free_field(mesh):
    """A smooth periodic vector field with zero mean in both components."""
    x, y = SpatialCoordinate(mesh)
    (_, Lx), (_, Ly) = mesh.domain_bounds
    return as_vector((sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly), cos(4 * pi * y / Ly)))


def test_project_rhs_orthogonal_to_harmonic_ufl():
    """UFL source: removes (f, h_i), is idempotent, and leaves orthogonal data unchanged."""
    mesh, _, _, basis = periodic_2d_k1()
    g = mean_free_field(mesh)
    f = g + as_vector((2.0, -3.0))
    projected, coefficients = project_rhs_orthogonal_to_harmonic(f, basis)
    area = 1.0 * 1.5
    assert numpy.allclose(coefficients, [2.0 * sqrt(area), -3.0 * sqrt(area)], atol=1e-8)
    assert all(abs(l2_inner(projected, h)) < 1e-10 for h in basis)
    _, again = project_rhs_orthogonal_to_harmonic(projected, basis)
    assert numpy.allclose(again, 0.0, atol=1e-10)
    same, zero = project_rhs_orthogonal_to_harmonic(g, basis)
    assert numpy.allclose(zero, 0.0, atol=1e-10)
    assert assemble(inner(same - g, same - g) * dx) < 1e-18


def test_project_rhs_orthogonal_to_harmonic_cofunction():
    """Assembled rhs on V^k: b(h_i) = 0 afterwards, and a second projection changes nothing."""
    mesh, _, V, basis = periodic_2d_k1()
    f = mean_free_field(mesh) + as_vector((2.0, -3.0))
    b = assemble(inner(f, TestFunction(V)) * dx)
    out, coefficients = project_rhs_orthogonal_to_harmonic(b, basis)
    assert out is b
    assert numpy.max(numpy.abs(coefficients)) > 1.0
    _, again = project_rhs_orthogonal_to_harmonic(b, basis)
    assert numpy.allclose(again, 0.0, atol=1e-10)
    before = b.dat.data_ro.copy()
    project_rhs_orthogonal_to_harmonic(b, basis)
    assert numpy.allclose(b.dat.data_ro, before, atol=1e-12)


def test_project_rhs_orthogonal_to_harmonic_mixed():
    """Mixed rhs: only the u block changes, and it becomes orthogonal to the basis."""
    mesh, spaces, V, basis = periodic_2d_k1()
    W = MixedFunctionSpace([spaces.CG, V])
    tau, v = TestFunctions(W)
    x, y = SpatialCoordinate(mesh)
    f = mean_free_field(mesh) + as_vector((2.0, -3.0))
    b = assemble(sin(2 * pi * x) * tau * dx + inner(f, v) * dx)
    sigma_before = b.subfunctions[0].dat.data_ro.copy()
    _, coefficients = project_rhs_orthogonal_to_harmonic(b, basis, u_index=1)
    assert numpy.max(numpy.abs(coefficients)) > 1.0
    assert numpy.allclose(b.subfunctions[0].dat.data_ro, sigma_before)
    _, again = project_rhs_orthogonal_to_harmonic(b, basis, u_index=1)
    assert numpy.allclose(again, 0.0, atol=1e-10)


def test_project_rhs_keeps_bc_rows():
    """Cylinder (periodic x, essential y), grad-curl k = 1, h = dy: BC rows of the rhs stay."""
    mesh, spaces, registry, face_bcs = make_setup(2, ('x',), {'bottom': 'essential', 'top': 'essential'}, n=4)
    _, V, basis = make_basis(spaces, registry, face_bcs, mesh, 1, 2, 'grad-curl')
    assert len(basis) == 1
    bcs = build_essential_bcs(V, None, 0, faces_of_type(registry, face_bcs, 'essential'))
    b = assemble(inner(as_vector((1.0, 2.0)), TestFunction(V)) * dx)
    for bc in bcs:
        b.dat.data[bc.nodes] = 7.0  # stand-in for lifted BC values
    project_rhs_orthogonal_to_harmonic(b, basis, bcs=bcs)
    for bc in bcs:
        assert numpy.allclose(b.dat.data_ro[bc.nodes], 7.0)
    _, again = project_rhs_orthogonal_to_harmonic(b, basis, bcs=bcs)
    assert numpy.allclose(again, 0.0, atol=1e-10)


def test_remove_harmonic_component():
    """The result is orthogonal to the basis, and the removed coefficients are returned."""
    mesh, _, V, basis = periodic_2d_k1()
    u = Function(V).project(mean_free_field(mesh), solver_parameters=LU)
    u.assign(u + 2.0 * basis[0] - 0.5 * basis[1])
    coefficients = remove_harmonic_component(u, basis)
    assert numpy.allclose(coefficients, [2.0, -0.5], atol=1e-8)
    assert all(abs(l2_inner(u, h)) < 1e-12 for h in basis)
    combined = combine_harmonic_forms([1.0, 2.0], basis)
    assert numpy.allclose([l2_inner(combined, h) for h in basis], [1.0, 2.0])


# ---------------------------------------------------------------------------------------------
# Lagrange multipliers vs nullspace
# ---------------------------------------------------------------------------------------------

def solve_lagrange(a, aP, L, w, multiplier_index):
    """Matrix-free GMRES solve of the bordered 'lagrange' system with the assembled aP."""
    problem = LinearVariationalProblem(a, L, w, aP=aP)
    params = lagrange_solver_parameters(len(w.function_space()), multiplier_index)
    LinearVariationalSolver(problem, solver_parameters=params).solve()


def test_lagrange_and_nullspace_agree_primal_k0():
    """Fully periodic Poisson (k = 0): both approaches give the same u and p = P_h f."""
    mesh, spaces, registry, face_bcs = make_setup(2, ('x', 'y'), {}, n=6, order=2)
    _, V, basis = make_basis(spaces, registry, face_bcs, mesh, 0, 2, 'grad-curl')
    assert len(basis) == 1
    x, y = SpatialCoordinate(mesh)
    f = sin(2 * pi * x) * cos(2 * pi * y / 1.5) + 3.0

    # lagrange: W = V x R
    R = harmonic_multiplier_space(mesh, 1)
    assert len(R) == 1
    W = MixedFunctionSpace([V, *R])
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a0 = inner(grad(u), grad(v)) * dx
    a = a0 + harmonic_constraint_form(u, v, p, q, basis)
    w = Function(W)
    solve_lagrange(a, a0 + lagrange_preconditioner_form(u, v, p, q), inner(f, v) * dx, w, 1)
    u_lagrange, p_h = w.subfunctions
    p_coeffs = multiplier_coefficients(p_h)
    assert numpy.allclose(p_coeffs, [3.0 * sqrt(1.5)], rtol=1e-8)
    assert abs(l2_inner(u_lagrange, basis[0])) < 1e-10

    # nullspace: singular V system, projected rhs, CG with the nullspace attached
    ut, vt = TrialFunction(V), TestFunction(V)
    A = assemble(inner(grad(ut), grad(vt)) * dx)
    b = assemble(inner(f, vt) * dx)
    _, coefficients = project_rhs_orthogonal_to_harmonic(b, basis)
    nullspace = build_harmonic_nullspace(V, None, basis)
    u_null = Function(V)
    solve(A, u_null, b, nullspace=nullspace, transpose_nullspace=nullspace,
          solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_rtol': 1e-13})
    remove_harmonic_component(u_null, basis)
    assert numpy.allclose(coefficients, p_coeffs, rtol=1e-8)
    diff = Function(V).assign(u_null - u_lagrange)
    assert l2_inner(diff, diff) ** 0.5 < 1e-8 * l2_inner(u_lagrange, u_lagrange) ** 0.5


def test_lagrange_and_nullspace_agree_mixed_top_form():
    """Fully periodic 2D grad-curl k = 2 (sigma in Hcurl, u in DG): both approaches agree."""
    mesh, spaces, registry, face_bcs = make_setup(2, ('x', 'y'), {}, n=4, order=1)
    _, V, basis = make_basis(spaces, registry, face_bcs, mesh, 2, 2, 'grad-curl')
    S = form_space(spaces, 1, 2, 'grad-curl')
    x, y = SpatialCoordinate(mesh)
    f = cos(2 * pi * x) * sin(2 * pi * y / 1.5) - 1.0

    def d1(s):
        return exterior_derivative(s, 1, 2, 'grad-curl')

    # lagrange: W = Hcurl x DG x R
    W = MixedFunctionSpace([S, V, *harmonic_multiplier_space(mesh, 1)])
    sigma, u, p = TrialFunctions(W)
    tau, v, q = TestFunctions(W)
    a0 = (inner(sigma, tau) - inner(u, d1(tau)) + inner(d1(sigma), v)) * dx
    a = a0 + harmonic_constraint_form(u, v, p, q, basis)
    w = Function(W)
    solve_lagrange(a, a0 + lagrange_preconditioner_form(u, v, p, q), inner(f, v) * dx, w, 2)
    sigma_l, u_l, p_l = w.subfunctions

    # nullspace: W2 = Hcurl x DG, kernel (0, h)
    W2 = MixedFunctionSpace([S, V])
    sigma, u = TrialFunctions(W2)
    tau, v = TestFunctions(W2)
    A = assemble((inner(sigma, tau) - inner(u, d1(tau)) + inner(d1(sigma), v)) * dx)
    b = assemble(inner(f, v) * dx)
    assert isinstance(b, Cofunction)
    _, coefficients = project_rhs_orthogonal_to_harmonic(b, basis, u_index=1)
    nullspace = build_harmonic_nullspace(W2, 1, basis)
    w2 = Function(W2)
    solve(A, w2, b, nullspace=nullspace, transpose_nullspace=nullspace, solver_parameters=SINGULAR_LU)
    sigma_n, u_n = w2.subfunctions
    remove_harmonic_component(u_n, basis)

    assert numpy.allclose(coefficients, multiplier_coefficients(p_l), rtol=1e-8)
    for a_h, b_h in ((sigma_l, sigma_n), (u_l, u_n)):
        diff = a_h - b_h
        assert assemble(inner(diff, diff) * dx) ** 0.5 < 1e-8 * max(1.0, l2_inner(a_h, a_h) ** 0.5)
