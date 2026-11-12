"""Shared pytest fixtures."""
import pytest

# Mesh type and logical faces per dimension, for the elliptic parameter builder.
_ELLIPTIC_MESH_TYPES = {1: 'line', 2: 'rectangle', 3: 'box'}
_ELLIPTIC_FACES = {1: [('left', 'x'), ('right', 'x')],
                   2: [('left', 'x'), ('right', 'x'), ('bottom', 'y'), ('top', 'y')],
                   3: [('left', 'x'), ('right', 'x'), ('bottom', 'y'), ('top', 'y'),
                       ('back', 'z'), ('front', 'z')]}
# Named BC sets: (periodic directions, BC on low faces, BC on high faces); None = all directions.
ELLIPTIC_BC_SETS = {
    'periodic': (None, None, None),
    'essential': ((), 'essential', 'essential'),
    'natural': ((), 'natural', 'natural'),
    'mixed': ((), 'essential', 'natural'),
    'partial-essential': (('x',), 'essential', 'essential'),
    'partial-natural': (('x',), 'natural', 'natural'),
}


def make_elliptic_parameters(dim, complex, k, bcs, n=8, order=2, problem='sine',
                             harmonic_constraint='lagrange', solver='elliptic-direct', **problem_cfg):
    """Parameter dictionary for an EllipticProblem (no Firedrake objects are created).

    Args:
        dim, complex, k: dimension, 2D complex ('grad-curl' | 'curl-div', ignored otherwise) and
            form degree.
        bcs: a name from ELLIPTIC_BC_SETS ('mixed' = essential on the low faces, natural on the
            high faces; 'partial-*' = periodic in x, the given type elsewhere), or a tuple
            (periodic directions, {face: type}).
        n, order: cells per direction and polynomial order.
        problem: elliptic-problem name; extra keyword arguments go into that section.
    """
    if isinstance(bcs, tuple):
        periodic, face_cfg = bcs
    else:
        periodic, low, high = ELLIPTIC_BC_SETS[bcs]
        periodic = ('x', 'y', 'z')[:dim] if periodic is None else periodic
        face_cfg = {}
        for i, (name, direction) in enumerate(_ELLIPTIC_FACES[dim]):
            if direction not in periodic:
                face_cfg[name] = low if i % 2 == 0 else high
    if periodic and n < 3:
        # Firedrake's periodic meshes need at least 3 cells per periodic direction.
        raise ValueError("periodic elliptic test meshes need n >= 3, got n = " + str(n))
    problem_section = {'name': problem, 'Lx': 1.0, 'Ly': 1.25, 'Lz': 0.75}
    problem_section.update(problem_cfg)
    return {
        'mesh': {'type': _ELLIPTIC_MESH_TYPES[dim], 'nx': n, 'ny': n, 'nz': n, 'simplicial_cells': False,
                 'periodic_directions': list(periodic)},
        'spatial-discretization': {'family': 'Q', 'order': order, 'lump_mass': False},
        'model': {'type': 'elliptic'},
        'elliptic': {'form_degree': k, 'complex': complex, 'boundary_conditions': face_cfg,
                     'boundary_data': 'exact', 'harmonic_forms': 'auto',
                     'harmonic_constraint': harmonic_constraint, 'solver': solver},
        'elliptic-problem': problem_section,
        'output': {'loglevel': 0},
    }


@pytest.fixture
def elliptic_parameters():
    """Builder for elliptic parameter dictionaries (see make_elliptic_parameters)."""
    return make_elliptic_parameters
