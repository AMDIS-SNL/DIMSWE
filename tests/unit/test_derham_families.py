"""Unit tests for the FEEC families of DeRhamComplex (TODO-elliptic §4.9).

The element names and degree conventions in meshes.family_elements are checked against the
installed Firedrake/FIAT here: every family must build on its cells, and form a subcomplex
(d V^k ⊂ V^{k+1}, checked by L2-projecting d of a random discrete function into V^{k+1}).
"""
from types import SimpleNamespace

import numpy
import pytest
from firedrake import Function, FunctionSpace, VectorFunctionSpace, assemble, dx, inner

from dimswe.differential_forms import exterior_derivative, form_space
from dimswe.harmonic_forms import interpolate_or_project
from dimswe.meshes import FEEC_FAMILIES, family_elements, get_mesh_and_spaces

MESH_FOR_DIM = {1: 'line', 2: 'rectangle', 3: 'box'}

# (family, dim, order) combinations built on Firedrake meshes; 3D cases are the most expensive.
BUILD_CASES = [
    ('Q', 1, 2), ('Q', 2, 2), ('Q', 3, 1),
    ('P-', 1, 2), ('P-', 2, 2), ('P-', 3, 1),
    ('P+', 2, 2), ('P+', 2, 3), ('P+', 3, 3),
    ('S-', 1, 2),
    ('S', 2, 2), ('S', 2, 3),
]


def case_id(case):
    """Readable pytest id."""
    family, dim, order = case
    return family + "-" + str(dim) + "D-r" + str(order)


def build(family, dim, order, n=2):
    """Mesh and spaces for a family (simplicial cells for P-/P+)."""
    parameters = {'mesh': {'type': MESH_FOR_DIM[dim], 'nx': n, 'ny': n, 'nz': n,
                           'simplicial_cells': family in ('P-', 'P+')},
                  'spatial-discretization': {'family': family, 'order': order, 'lump_mass': False}}
    domain = SimpleNamespace(Lx=1.0, Ly=1.25, Lz=0.75, xc=0.5, yc=0.625, zc=0.375)
    return get_mesh_and_spaces(parameters, domain)


# ---------------------------------------------------------------------------------------------
# Pure-Python table
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family, dim, r, expected", [
    ('Q', 2, 3, {'CG': ('CG', 3, 'spectral'), 'Hcurl': ('RTCE', 3, None), 'Hdiv': ('RTCF', 3, None),
                 'DG': ('DG', 2, 'spectral')}),
    ('Q', 3, 2, {'CG': ('CG', 2, 'spectral'), 'Hcurl': ('NCE', 2, None), 'Hdiv': ('NCF', 2, None),
                 'DG': ('DG', 1, 'spectral')}),
    ('P-', 3, 2, {'CG': ('CG', 2, None), 'Hcurl': ('N1curl', 2, None), 'Hdiv': ('RT', 2, None),
                  'DG': ('DG', 1, None)}),
    ('P+', 2, 3, {'CG': ('CG', 3, None), 'Hcurl': ('N2curl', 2, None), 'Hdiv': ('BDM', 2, None),
                  'DG': ('DG', 1, None)}),
    ('P+', 3, 3, {'CG': ('CG', 3, None), 'Hcurl': ('N2curl', 2, None), 'Hdiv': ('BDM', 1, None),
                  'DG': ('DG', 0, None)}),
    ('S', 2, 3, {'CG': ('S', 3, None), 'Hcurl': ('BDMCE', 2, None), 'Hdiv': ('BDMCF', 2, None),
                 'DG': ('DPC', 1, None)}),
])
def test_family_elements_table(family, dim, r, expected):
    """The documented element table."""
    assert family_elements(family, dim, r) == expected


@pytest.mark.parametrize("family", FEEC_FAMILIES)
def test_family_elements_1d_reduces_to_cg_dg(family):
    """Every family is CG r -> DG r-1 in 1D, with the existing Hdiv = CG convention."""
    table = family_elements(family, 1, 2)
    assert table['CG'][:2] == ('CG', 2) and table['DG'][:2] == ('DG', 1)
    assert table['Hdiv'] == table['CG'] and table['Hcurl'] is None


def test_family_elements_rejections():
    """Unknown families, too-low orders and 3D full serendipity are rejected."""
    with pytest.raises(ValueError):
        family_elements('N', 2, 1)
    with pytest.raises(ValueError):
        family_elements('P+', 3, 2)   # needs r >= n
    with pytest.raises(ValueError):
        family_elements('S', 2, 1)    # DPC r-2 needs r >= 2
    with pytest.raises(ValueError):
        family_elements('P-', 2, 0)
    with pytest.raises(NotImplementedError):
        family_elements('S', 3, 3)    # Firedrake: only Q/DQ elements on hexahedra
    with pytest.raises(NotImplementedError):
        family_elements('S-', 3, 1)
    with pytest.raises(NotImplementedError):
        family_elements('S-', 2, 2)   # FIAT trimmed serendipity lacks dual_basis()


# ---------------------------------------------------------------------------------------------
# Construction against the installed Firedrake
# ---------------------------------------------------------------------------------------------

def subcomplex_defect(spaces, k, dim, cx, seed):
    """Relative L2 distance of d w_h (w_h random in V^k) from V^{k+1}."""
    V = form_space(spaces, k, dim, cx)
    W = form_space(spaces, k + 1, dim, cx)
    w = Function(V)
    w.dat.data[:] = numpy.random.default_rng(seed).standard_normal(w.dat.data.shape)
    dw = exterior_derivative(w, k, dim, cx)
    # interpolate_or_project: several of these elements cannot be interpolated into (and must not
    # go through Function.project either); data in W are reproduced exactly either way.
    projected = interpolate_or_project(dw, W)
    diff = projected - dw
    norm = assemble(inner(dw, dw) * dx) ** 0.5
    return assemble(inner(diff, diff) * dx) ** 0.5 / norm, norm


@pytest.mark.parametrize("case", BUILD_CASES, ids=case_id)
def test_family_builds_and_is_a_subcomplex(case):
    """All spaces exist and d maps each discrete space into the next one."""
    family, dim, order = case
    mesh, spaces = build(family, dim, order)
    assert spaces.family == family
    for name in ('CG', 'DG', 'CGV', 'DGV', 'Hdiv'):
        assert getattr(spaces, name) is not None
    assert (spaces.Hcurl is None) == (dim == 1)
    complexes = ['grad-curl', 'curl-div'] if dim == 2 else [None]
    for cx in complexes:
        for k in range(dim):
            defect, norm = subcomplex_defect(spaces, k, dim, cx, seed=10 * dim + k)
            assert norm > 1e-8
            assert defect < 1e-8, (family, dim, cx, k, defect)


def test_q_family_unchanged():
    """The Q family builds exactly the spaces of the original implementation."""
    for dim, r in ((1, 2), (2, 3), (3, 2)):
        mesh, spaces = build('Q', dim, r)
        expected = {
            'CG': FunctionSpace(mesh, 'CG', r, variant='spectral'),
            'DG': FunctionSpace(mesh, 'DG', r - 1, variant='spectral'),
        }
        if dim > 1:
            expected['CGV'] = VectorFunctionSpace(mesh, 'CG', r, variant='spectral')
            expected['DGV'] = VectorFunctionSpace(mesh, 'DG', r - 1, variant='spectral')
            curl, div = ('RTCE', 'RTCF') if dim == 2 else ('NCE', 'NCF')
            expected['Hcurl'] = FunctionSpace(mesh, curl, r)
            expected['Hdiv'] = FunctionSpace(mesh, div, r)
        for name, V in expected.items():
            assert getattr(spaces, name).ufl_element() == V.ufl_element(), (dim, name)
        if dim == 1:
            assert spaces.Hdiv is spaces.CG and spaces.CGV is spaces.CG and spaces.DGV is spaces.DG


@pytest.mark.parametrize("family, simplicial, dim", [
    ('P-', False, 2),   # simplicial family on quadrilaterals
    ('P+', False, 3),   # ... and on the extruded box
    ('S', True, 2),     # serendipity on triangles
    ('Q', True, 2),     # Q on triangles
])
def test_family_cell_mismatch_is_rejected(family, simplicial, dim):
    """Families only build on their cells."""
    parameters = {'mesh': {'type': MESH_FOR_DIM[dim], 'nx': 2, 'ny': 2, 'nz': 2, 'simplicial_cells': simplicial},
                  'spatial-discretization': {'family': family, 'order': 3, 'lump_mass': False}}
    domain = SimpleNamespace(Lx=1.0, Ly=1.0, Lz=1.0, xc=0.5, yc=0.5, zc=0.5)
    with pytest.raises(ValueError):
        get_mesh_and_spaces(parameters, domain)


def test_lump_mass_only_for_q():
    """GLL lumping is only defined for the Q family."""
    parameters = {'mesh': {'type': 'rectangle', 'nx': 2, 'ny': 2, 'simplicial_cells': True},
                  'spatial-discretization': {'family': 'P-', 'order': 1, 'lump_mass': True}}
    domain = SimpleNamespace(Lx=1.0, Ly=1.0, Lz=1.0, xc=0.5, yc=0.5, zc=0.5)
    with pytest.raises(ValueError):
        get_mesh_and_spaces(parameters, domain)
