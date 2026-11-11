"""Tests for the cheap mass-matrix inverses of dimswe/assembly.py MassSolver (solver parameter mass_inverse).

A MassSolver inverts the mass matrix shared by the RK stage solves. With mass_inverse 'auto' it
divides pointwise by a diagonal matrix (a lumped mass: continuous spectral elements with GLL
quadrature), uses the cellwise (Slate) inverse for discontinuous fields, treats the fields of an
uncoupled mixed space separately, and LU-factors everything else. These tests check, on small
quadrilateral meshes, which inverse is chosen for each kind of space and mode, and that every
choice solves M x = b to round-off, as an LU factorization does.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from firedrake import (Cofunction, Function, FunctionSpace, TestFunction, TestFunctions,  # noqa: E402
                       TrialFunction, TrialFunctions, UnitSquareMesh, VectorFunctionSpace, dx, inner)

from dimswe.assembly import MassSolver  # noqa: E402
from dimswe.meshes import gauss_lobatto_legendre_cube_rule  # noqa: E402

DIRECT = {'ksp_type': 'preonly', 'pc_type': 'lu'}
# the cheap inverses and LU solve the same system exactly, so they agree to round-off
RTOL = 1e-11
DEGREE = 3


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(4, 4, quadrilateral=True)


def gll_dx():
    """The lumped-mass measure of the model (dimswe/meshes.py, lump_mass: true)."""
    return dx(scheme=gauss_lobatto_legendre_cube_rule(dimension=2, degree=DEGREE))


def spaces(mesh):
    """Return the test spaces by name: spectral CG (lumpable), DG, spectral DG, vector spectral CG, plain CG.

    'dg' uses equispaced nodes, so its mass matrix is block diagonal but not diagonal. Firedrake's
    default DG variant on quadrilaterals has Gauss-Legendre nodes, whose mass matrix is diagonal
    with exact quadrature ('dgs').
    """
    return {
        'cg': FunctionSpace(mesh, 'CG', DEGREE, variant='spectral'),
        'dg': FunctionSpace(mesh, 'DG', DEGREE - 1, variant='equispaced'),
        'dgs': FunctionSpace(mesh, 'DG', DEGREE - 1),
        'cgv': VectorFunctionSpace(mesh, 'CG', DEGREE, variant='spectral'),
        'cg2': FunctionSpace(mesh, 'CG', 2),
    }


def solve_and_compare(form, space, mode, seed):
    """Solve M x = b with a MassSolver in mode and with LU ('factor'); return (solver, relative difference)."""
    rng = np.random.default_rng(seed)
    b = Cofunction(space.dual())
    for dat in b.dat.split:
        dat.data[:] = rng.standard_normal(dat.data.shape)
    solver = MassSolver(form, dict(DIRECT, mass_inverse=mode))
    reference = MassSolver(form, dict(DIRECT, mass_inverse='factor'))
    x, x_reference = Function(space), Function(space)
    # two solves: the first sets up and checks the inverse, the second reuses it
    for _ in range(2):
        solver.solve(x, b, scale=-1.0)
    reference.solve(x_reference, b, scale=-1.0)
    assert reference.method == 'ksp'
    with x.dat.vec_ro as xv, x_reference.dat.vec_ro as rv:
        difference = xv.copy()
        difference.axpy(-1.0, rv)
        relative = difference.norm() / rv.norm()
    return solver, relative


def single_space_cases(mesh):
    """Yield (label, form, space, mode, expected method, expected note substring or None)."""
    V = spaces(mesh)
    for name, measure in (('cg', gll_dx()), ('cgv', gll_dx())):
        space = V[name]
        form = inner(TrialFunction(space), TestFunction(space)) * measure
        yield name + ' lumped auto', form, space, 'auto', 'diagonal', None
        yield name + ' lumped diagonal', form, space, 'diagonal', 'diagonal', None
        yield name + ' lumped factor', form, space, 'factor', 'ksp', None
    dg = V['dg']
    dg_form = inner(TrialFunction(dg), TestFunction(dg)) * dx
    yield 'dg auto', dg_form, dg, 'auto', 'cellwise', 'not diagonal'
    yield 'dg cellwise', dg_form, dg, 'cellwise', 'cellwise', 'not diagonal'
    yield 'dg diagonal only', dg_form, dg, 'diagonal', 'ksp', 'not diagonal'
    dgs = V['dgs']
    yield 'spectral dg is diagonal', inner(TrialFunction(dgs), TestFunction(dgs)) * dx, dgs, 'auto', 'diagonal', None
    cg2 = V['cg2']
    cg2_form = inner(TrialFunction(cg2), TestFunction(cg2)) * dx
    yield 'cg exact quadrature', cg2_form, cg2, 'auto', 'ksp', 'not discontinuous'
    yield 'cg cellwise requested', cg2_form, cg2, 'cellwise', 'ksp', 'not discontinuous'


@pytest.mark.parametrize("index", range(12))
def test_single_space_inverse_choice_and_accuracy(mesh, index):
    label, form, space, mode, expected, note = list(single_space_cases(mesh))[index]
    solver, relative = solve_and_compare(form, space, mode, seed=100 + index)
    assert solver.method == expected, (label, solver.method, solver.inverse_notes)
    if note is not None:
        assert any(note in text for text in solver.inverse_notes), (label, solver.inverse_notes)
    assert relative <= RTOL, (label, relative)


def test_uncoupled_mixed_space_is_inverted_field_by_field(mesh):
    V = spaces(mesh)
    # the model's x-space shape: lumped continuous fields and a DG tracer, all with GLL quadrature
    W = V['cgv'] * V['cg'] * V['dg']
    form = sum(inner(u, v) for u, v in zip(TrialFunctions(W), TestFunctions(W))) * gll_dx()
    solver, relative = solve_and_compare(form, W, 'auto', seed=7)
    assert solver.method == 'fieldsplit(diagonal, diagonal, cellwise)', (solver.method, solver.inverse_notes)
    assert relative <= RTOL, relative
    # an all-lumped mixed space is diagonal as a whole
    Wd = V['cgv'] * V['cg']
    form_d = sum(inner(u, v) for u, v in zip(TrialFunctions(Wd), TestFunctions(Wd))) * gll_dx()
    solver_d, relative_d = solve_and_compare(form_d, Wd, 'auto', seed=8)
    assert solver_d.method == 'diagonal' and relative_d <= RTOL


def test_mixed_spaces_without_a_cheap_field_or_with_coupling_are_factored(mesh):
    V = spaces(mesh)
    W = V['cg2'] * V['cg2']
    u0, u1 = TrialFunctions(W)
    v0, v1 = TestFunctions(W)
    plain = (inner(u0, v0) + inner(u1, v1)) * dx
    solver, relative = solve_and_compare(plain, W, 'auto', seed=9)
    assert solver.method == 'ksp' and relative <= RTOL
    assert any('no field has a cheap inverse' in note for note in solver.inverse_notes), solver.inverse_notes
    coupled = (inner(u0, v0) + inner(u1, v1) + 0.1 * inner(u1, v0) + 0.1 * inner(u0, v1)) * dx
    solver_c, relative_c = solve_and_compare(coupled, W, 'auto', seed=10)
    assert solver_c.method == 'ksp' and relative_c <= RTOL
    assert any('couples fields' in note for note in solver_c.inverse_notes), solver_c.inverse_notes


def test_an_unknown_mode_is_rejected(mesh):
    space = spaces(mesh)['dg']
    form = inner(TrialFunction(space), TestFunction(space)) * dx
    with pytest.raises(ValueError, match="mass_inverse"):
        MassSolver(form, dict(DIRECT, mass_inverse='lumped'))
    # the reference path of the stage solves never sees mass_inverse
    from dimswe.assembly import petsc_solver_parameters
    assert 'mass_inverse' not in petsc_solver_parameters(dict(DIRECT, mass_inverse='auto'))
