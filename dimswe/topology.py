"""Betti numbers of the discrete de Rham complex (TODO-elliptic §4.13).

The dimension of the discrete harmonic k-forms is the k-th Betti number of the complex relative to
the essential boundary Gamma_e:

    b_k = dim V^k_0 - rank d_k - rank d_{k-1},

where V^j_0 is the j-form space with zero trace on Gamma_e and d_j : V^j_0 -> V^{j+1}. Cohomology
does not depend on the polynomial order, so the ranks are computed on the lowest-order complex of
the same mesh (Q -> Q_1, P-/P+ -> P-_1, S -> Q_1), which keeps the matrices small. The rank of d_j
equals the rank of the rectangular matrix B_j = (d phi, psi) (phi in V^j_0, psi in V^{j+1}),
because the V^{j+1} mass matrix is non-singular. Ranks use a dense SVD, so this is serial and
limited in size (max_entries); it returns None when it cannot be computed, and callers then rely
on the spectral gap of the eigen-solve (dimswe.numerical_harmonic).
"""
import copy

import numpy
from firedrake import DirichletBC, TestFunction, TrialFunction, assemble, dx, inner

from .boundary_conditions import faces_of_type
from .differential_forms import exterior_derivative, form_space

# Largest dense B_j (rows x free columns) the SVD rank is computed for (~160 MB of doubles).
DEFAULT_MAX_ENTRIES = 20_000_000


class _TooLarge(Exception):
    """Raised internally when a dense incidence matrix would exceed max_entries."""


def lowest_order_parameters(parameters):
    """Copy of parameters with the lowest-order complex of the same mesh and cell type."""
    lowest = copy.deepcopy(parameters)
    disc = lowest['spatial-discretization']
    family = disc['family']
    disc['family'] = 'P-' if family in ('P-', 'P+') else 'Q'
    disc['order'] = 1
    disc['lump_mass'] = False
    return lowest


def free_dofs(V, faces, essential):
    """Indices of the dofs of V without an essential trace constraint (all of them if not essential)."""
    n = V.dim()
    if not essential or not faces:
        return numpy.arange(n)
    constrained = set()
    for face in faces:
        constrained.update(int(i) for i in DirichletBC(V, 0, face.dirichlet_subdomain).nodes)
    return numpy.array(sorted(set(range(n)) - constrained), dtype=int)


def incidence_rank(spaces, j, dim, complex, faces, max_entries=DEFAULT_MAX_ENTRIES):
    """rank of d_j : V^j_0 -> V^{j+1} on the given complex (0 for j < 0 or j >= dim)."""
    if j < 0 or j >= dim:
        return 0
    Vj, Vj1 = form_space(spaces, j, dim, complex), form_space(spaces, j + 1, dim, complex)
    columns = free_dofs(Vj, faces, essential=True)
    if columns.size == 0:
        return 0
    if Vj1.dim() * columns.size > max_entries:
        raise _TooLarge()
    B = assemble(inner(exterior_derivative(TrialFunction(Vj), j, dim, complex), TestFunction(Vj1)) * dx)
    dense = B.petscmat.convert('dense').getDenseArray()   # rows: test (V^{j+1}), columns: trial (V^j)
    return int(numpy.linalg.matrix_rank(dense[:, columns]))


def betti_number(mesh, parameters, config, max_entries=DEFAULT_MAX_ENTRIES):
    """b_k of the complex on mesh relative to the essential faces of config, or None.

    None in parallel (the dense rank is serial) and when a matrix exceeds max_entries. config is an
    EllipticConfig with parsed boundary conditions (registry, face_bcs). Cached on the mesh.
    """
    if mesh.comm.size > 1:
        return None
    from .meshes import DeRhamComplex
    k, dim, complex = config.k, config.dim, config.complex
    faces = faces_of_type(config.registry, config.face_bcs, 'essential')
    key = (k, dim, complex, parameters['spatial-discretization']['family'], tuple(sorted(config.face_bcs.items())))
    cache = getattr(mesh, '_dimswe_betti', None)
    if cache is None:
        cache = {}
        mesh._dimswe_betti = cache
    if key not in cache:
        lowest = DeRhamComplex(mesh, lowest_order_parameters(parameters))
        try:
            n_free = free_dofs(form_space(lowest, k, dim, complex), faces, essential=k < dim).size
            cache[key] = n_free - incidence_rank(lowest, k, dim, complex, faces, max_entries) \
                - incidence_rank(lowest, k - 1, dim, complex, faces, max_entries)
        except _TooLarge:
            cache[key] = None
    return cache[key]
