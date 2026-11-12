"""Harmonic forms for the elliptic Hodge Laplacian problems (TODO-elliptic §1.4, §1.5, §4.4).

For the box-shaped meshes DIMSWE builds (each direction periodic, or bounded with one BC type per
face), the discrete harmonic k-forms are spanned by constant forms dx^I (Künneth). With
P = periodic directions, E = directions whose two faces are both essential and N = directions
whose two faces are both natural:
- if a bounded direction has different BC types on its two faces, there are none;
- otherwise they are dx^I for E ⊆ I ⊆ E ∪ P, |I| = k (dimension C(|P|, k - |E|)).

This module provides
- the selection rule (pure Python): harmonic_index_sets, get_harmonic_index_sets;
- the basis: harmonic_basis (UFL constant proxies), orthonormalize_harmonic_basis (L2-orthonormal
  Functions in V^k);
- the rhs/solution helpers for the 'nullspace' approach: project_rhs_orthogonal_to_harmonic,
  remove_harmonic_component, combine_harmonic_forms, build_harmonic_nullspace;
- the Lagrange-multiplier pieces for the 'lagrange' approach: harmonic_multiplier_space,
  harmonic_constraint_form, multiplier_coefficients, and (since Firedrake cannot assemble
  monolithic matrices with R blocks) the matrix-free solve: lagrange_preconditioner_form,
  lagrange_solver_parameters.

Everything below takes the basis as a list of L2-orthonormal Functions in V^k (the output of
orthonormalize_harmonic_basis), so it also works unchanged for numerical bases (§8).
"""
from itertools import combinations

from firedrake import (Cofunction, Function, FunctionSpace, MixedVectorSpaceBasis, TestFunction,
                       VectorSpaceBasis, assemble, dx, inner)

from .boundary_conditions import BC_TYPES, FACE_LOCATIONS
from .differential_forms import constant_form_proxy, normalize_complex
from .meshes import DIRECTIONS

# Allowed values of the elliptic 'harmonic_forms' option.
HARMONIC_FORM_MODES = ('auto', 'none', 'numerical')

# Direct solver used to project constant proxies into spaces that do not support interpolation.
_PROJECTION_SOLVER = {'ksp_type': 'preonly', 'pc_type': 'lu'}


# ---------------------------------------------------------------------------------------------
# Selection rule (pure Python)
# ---------------------------------------------------------------------------------------------

def _bcs_by_direction(face_bcs):
    """Normalize per-face or per-direction BC types to {direction: (type side 0, type side 1)}.

    face_bcs is either {face name: type} (as parse_boundary_conditions returns) or
    {direction: (type0, type1)} (as face_bcs_by_direction returns).
    """
    if all(key in DIRECTIONS for key in face_bcs):
        return {d: tuple(types) for d, types in face_bcs.items()}
    grouped = {}
    for name, bc_type in face_bcs.items():
        if name not in FACE_LOCATIONS:
            raise ValueError("unknown boundary face '" + str(name) + "'")
        direction, side = FACE_LOCATIONS[name]
        grouped.setdefault(direction, [None, None])[side] = bc_type
    return {d: tuple(types) for d, types in grouped.items()}


def harmonic_index_sets(k, dim, periodic_dirs, face_bcs):
    """Index sets I (tuples of directions, increasing order) of the harmonic constant forms dx^I.

    Args:
        k: form degree, 0 <= k <= dim.
        dim: spatial dimension.
        periodic_dirs: periodic directions (subset of 'x', 'y', 'z' within dim).
        face_bcs: BC types of the bounded faces, per face ({'left': 'essential', ...}) or per
            direction ({'x': ('essential', 'natural'), ...}). Every bounded direction needs both
            faces, periodic directions none.

    Returns:
        list of tuples I with E ⊆ I ⊆ E ∪ P and |I| = k, in lexicographic order of the
        direction indices; empty if a bounded direction has mismatched BC types.
    """
    if dim not in (1, 2, 3):
        raise ValueError("dimension " + str(dim) + " is not supported (1, 2 or 3)")
    if k < 0 or k > dim:
        raise ValueError("form degree " + str(k) + " is out of range for dimension " + str(dim))
    valid = DIRECTIONS[:dim]
    periodic = set(periodic_dirs)
    if not periodic <= set(valid):
        raise ValueError("periodic directions " + str(sorted(periodic)) + " are not valid in " + str(dim) + "D")
    by_dir = _bcs_by_direction(face_bcs)

    essential = []
    for d in valid:
        types = by_dir.get(d, None)
        if d in periodic:
            if types is not None:
                raise ValueError("boundary conditions given for periodic direction " + d)
            continue
        if types is None or len(types) != 2 or None in types:
            raise ValueError("both faces of bounded direction " + d + " need a boundary condition")
        for bc_type in types:
            if bc_type not in BC_TYPES:
                raise ValueError("boundary condition " + str(bc_type) + " must be one of " + str(list(BC_TYPES)))
        if types[0] != types[1]:
            # Mixed essential/natural in one direction: no harmonic forms at all.
            return []
        if types[0] == 'essential':
            essential.append(d)
    for d in by_dir:
        if d not in valid:
            raise ValueError("boundary conditions given for direction " + d + " beyond dimension " + str(dim))

    # I = E ∪ J with J ⊆ P, |J| = k - |E|.
    free = [d for d in valid if d in periodic]
    m = k - len(essential)
    if m < 0 or m > len(free):
        return []
    sets = []
    for J in combinations(free, m):
        chosen = set(essential) | set(J)
        sets.append(tuple(d for d in valid if d in chosen))
    return sorted(sets, key=lambda I: [valid.index(d) for d in I])


def get_harmonic_index_sets(mode, k, dim, periodic_dirs, face_bcs, mesh_type):
    """Apply the elliptic 'harmonic_forms' option.

    - 'auto': the Künneth rule (harmonic_index_sets); not available on 'file' meshes;
    - 'none': no harmonic forms (for users who know the harmonic space is trivial);
    - 'numerical': eigen-solver based bases (TODO-elliptic §8), not implemented yet.
    """
    if mode not in HARMONIC_FORM_MODES:
        raise ValueError("harmonic_forms must be one of " + str(list(HARMONIC_FORM_MODES)) + ", got " + str(mode))
    if mode == 'none':
        return []
    if mode == 'numerical':
        raise NotImplementedError("numerical harmonic forms are not implemented yet (TODO-elliptic §8)")
    if mesh_type == 'file':
        raise NotImplementedError("harmonic_forms: auto needs a DIMSWE box mesh; use 'none' for 'file' meshes until numerical harmonic forms exist")
    return harmonic_index_sets(k, dim, periodic_dirs, face_bcs)


# ---------------------------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------------------------

def harmonic_basis(k, dim, complex, index_sets):
    """UFL constant proxies of the harmonic forms dx^I, one per index set (see constant_form_proxy)."""
    complex = normalize_complex(dim, complex)
    return [constant_form_proxy(I, k, dim, complex) for I in index_sets]


def interpolate_or_project(expr, V, target=None):
    """Interpolate expr into V, or L2-project with a direct solver if V cannot interpolate.

    The Q-family Hcurl/Hdiv elements (RTCE/RTCF/NCE/NCF) have no dual basis; for the constant
    proxies used here the projection is exact up to round-off, since V contains constants.
    target: optional Function in V to fill (a new one is created otherwise); it is returned.
    """
    f = Function(V) if target is None else target
    try:
        f.interpolate(expr)
    except NotImplementedError:
        f.project(expr, solver_parameters=_PROJECTION_SOLVER)
    return f


def _to_space(expr, V):
    """Backwards-compatible alias of interpolate_or_project (new Function)."""
    return interpolate_or_project(expr, V)


def l2_inner(a, b, measure=dx):
    """L2 inner product (a, b) assembled to a Python float.

    The cast matters: assemble returns a numpy float64, and numpy_scalar * Function builds an
    object array instead of a UFL expression.
    """
    return float(assemble(inner(a, b) * measure))


def orthonormalize_harmonic_basis(basis, V, measure=dx, tol=1e-10):
    """Turn basis expressions into L2-orthonormal Functions in V (modified Gram-Schmidt, twice).

    Constant forms on boxes are already orthogonal; the procedure normalizes them and keeps the
    code general for numerical bases. Raises ValueError if the basis is (numerically) linearly
    dependent, i.e. a vector loses all but a fraction tol of its norm.
    """
    result = []
    for expr in basis:
        h = _to_space(expr, V)
        norm0 = l2_inner(h, h, measure) ** 0.5
        # Two passes of modified Gram-Schmidt for robustness against round-off.
        for _ in range(2):
            for q in result:
                c = l2_inner(h, q, measure)
                _axpy(h, -c, q)
        norm = l2_inner(h, h, measure) ** 0.5
        if norm0 == 0.0 or norm <= tol * norm0:
            raise ValueError("harmonic basis is linearly dependent (or contains a zero form)")
        h.assign(h / norm)
        result.append(h)
    return result


def _axpy(y, alpha, x):
    """In-place y <- y + alpha x for Functions/Cofunctions with the same dof layout."""
    with y.dat.vec as yv, x.dat.vec_ro as xv:
        yv.axpy(alpha, xv)


def _pairing(b, h):
    """Dual pairing b(h) of a Cofunction (or a block of one) with a Function in the same space."""
    with b.dat.vec_ro as bv, h.dat.vec_ro as hv:
        return float(bv.dot(hv))


def combine_harmonic_forms(coefficients, basis):
    """Function sum_i c_i h_i in the space of the (non-empty) basis.

    Used to report p = P_h(rhs) from the coefficients of either constraint approach.
    """
    if len(coefficients) != len(basis):
        raise ValueError("need one coefficient per basis form")
    if not basis:
        raise ValueError("cannot combine an empty harmonic basis")
    result = Function(basis[0].function_space())
    for c, h in zip(coefficients, basis):
        _axpy(result, float(c), h)
    return result


# ---------------------------------------------------------------------------------------------
# Right-hand side projection and solution post-processing ('nullspace' approach)
# ---------------------------------------------------------------------------------------------

def _rhs_block(rhs, u_index):
    """The u block of a (possibly mixed) Cofunction."""
    if u_index is None:
        return rhs
    return rhs.subfunctions[u_index]


def project_rhs_orthogonal_to_harmonic(rhs, basis, u_index=None, bcs=(), measure=dx, logger=None):
    """Remove the harmonic component of a right-hand side (the rhs projection helper of §1.5).

    Args:
        rhs: either a UFL source expression f (a k-form proxy), or an assembled Cofunction (on
            V^k, or on a mixed space with V^k as block u_index), which may include natural
            boundary functionals and essential BC values.
        basis: L2-orthonormal Functions h_i in V^k.
        u_index: block index of u for mixed Cofunctions (None for a Cofunction on V^k).
        bcs: DirichletBCs on the u block that were applied to the assembled rhs; the correction
            is zeroed on their nodes so the BC rows are left untouched (h_i vanishes there).
        measure: integration measure for UFL sources.
        logger: optional DIMSWE logger; the size of the removed component is reported at level 1.

    Returns:
        (projected, coefficients), with coefficients c_i = (f, h_i) or c_i = b(h_i) before the
        projection:
        - UFL source: projected = f - sum_i c_i h_i (a new UFL expression);
        - Cofunction: the rhs itself, modified in place by b <- b - sum_i b(h_i) M h_i, where M
          is the V^k mass matrix, so that b(h_i) = 0 for every i afterwards.
    """
    if not basis:
        return rhs, []
    if isinstance(rhs, Cofunction):
        block = _rhs_block(rhs, u_index)
        coefficients = [_pairing(block, h) for h in basis]
        for c, h in zip(coefficients, basis):
            Mh = assemble(inner(h, TestFunction(h.function_space())) * measure)
            for bc in bcs:
                # Leave the BC rows of the rhs alone.
                Mh.dat.data[bc.nodes] = 0.0
            _axpy(block, -c, Mh)
        projected = rhs
    else:
        coefficients = [l2_inner(rhs, h, measure) for h in basis]
        projected = rhs
        for c, h in zip(coefficients, basis):
            projected = projected - float(c) * h
    if logger is not None:
        size = sum(c * c for c in coefficients) ** 0.5
        logger.output('removed harmonic component of the rhs with l2 size ' + str(size), 1)
    return projected, coefficients


def remove_harmonic_component(u_h, basis, measure=dx):
    """In place u_h <- u_h - sum_i (u_h, h_i) h_i; returns the removed coefficients.

    u_h may be a Function in V^k or the u subfunction of a mixed solution.
    """
    coefficients = [l2_inner(u_h, h, measure) for h in basis]
    for c, h in zip(coefficients, basis):
        _axpy(u_h, -c, h)
    return coefficients


def build_harmonic_nullspace(W, u_index, basis):
    """PETSc nullspace {(0, h) : h harmonic} of the singular system ('nullspace' approach).

    Args:
        W: the problem's function space (mixed, or just V^k for the primal k = 0 problem).
        u_index: block index of u in W (ignored for a non-mixed W).
        basis: harmonic Functions in V^k (copied; the copies are l2-orthonormalized in the PETSc
            sense, as PETSc requires).

    Returns:
        a VectorSpaceBasis (non-mixed W) or MixedVectorSpaceBasis, or None for an empty basis.
        The operator is symmetric, so the same object serves as nullspace and transpose
        nullspace.
    """
    if not basis:
        return None
    vecs = [h.copy(deepcopy=True) for h in basis]
    block = VectorSpaceBasis(vecs=vecs)
    block.orthonormalize()
    if len(W) == 1:
        return block
    bases = [block if i == u_index else W.sub(i) for i in range(len(W))]
    return MixedVectorSpaceBasis(W, bases)


# ---------------------------------------------------------------------------------------------
# Lagrange multipliers ('lagrange' approach)
# ---------------------------------------------------------------------------------------------

def harmonic_multiplier_space(mesh, m):
    """Spaces of the multiplier p = sum_i p_i h_i: a list of m scalar R spaces (one per form).

    Firedrake supports neither arguments on vector-valued R spaces nor monolithic matrices with R
    blocks, so each coefficient p_i gets its own scalar R block; append them to the mixed space
    (e.g. MixedFunctionSpace([V_sigma, V_u, *harmonic_multiplier_space(mesh, m)])).
    """
    return [FunctionSpace(mesh, "R", 0) for _ in range(m)]


def _as_list(x):
    """Wrap a single multiplier argument/Function/index in a list (lists are passed through)."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def harmonic_constraint_form(u, v, p, q, basis, measure=dx):
    """Bilinear form sum_i [p_i (h_i, v) + q_i (u, h_i)] coupling u and the multiplier p.

    u/v are the trial/test functions of u; p/q are lists of the trial/test functions of the m
    scalar R blocks (a single argument is accepted when m = 1). Adding this to the system gives
    the '(p, v)' term of the u equation and the constraint rows (u, h_i) = 0.
    """
    p, q = _as_list(p), _as_list(q)
    if len(p) != len(basis) or len(q) != len(basis):
        raise ValueError("need one multiplier block per harmonic basis form")
    form = 0
    for p_i, q_i, h in zip(p, q, basis):
        form = form + (p_i * inner(h, v) + q_i * inner(u, h)) * measure
    return form


def lagrange_preconditioner_form(u, v, p, q, regularization=1.0, measure=dx):
    """Extra preconditioner terms for the 'lagrange' system: regularization (u, v) + sum_i p_i q_i.

    Firedrake cannot assemble monolithic matrices with R-space blocks, so the bordered system is
    solved matrix-free (lagrange_solver_parameters) and preconditioned by an assembled form
    aP = a_0 + lagrange_preconditioner_form(...), where a_0 is the problem's bilinear form
    *without* harmonic_constraint_form. The regularization makes the (sigma, u) block of aP
    non-singular (it is singular on the harmonic forms otherwise); the p_i q_i terms give the R
    blocks an identity-like preconditioner. p and q are lists over the R blocks (or single
    arguments when m = 1).

    On non-harmonic modes the preconditioned operator has eigenvalues lambda / (lambda + eps)
    (lambda the Hodge Laplacian eigenvalues), and the m harmonic directions coupled to p add 2m
    eigenvalues +-(eps |Omega|)^(-1/2), so GMRES converges in few iterations. eps should be small
    compared to the smallest non-zero eigenvalue (pi^2 / L^2 for a domain of size L).
    """
    form = regularization * inner(u, v)
    for p_i, q_i in zip(_as_list(p), _as_list(q)):
        form = form + p_i * q_i
    return form * measure


def lagrange_solver_parameters(num_blocks, multiplier_indices, rtol=1e-12, atol=1e-14, max_it=500):
    """PETSc options for the matrix-free 'lagrange' solve with the aP of lagrange_preconditioner_form.

    GMRES (right-preconditioned, so the stopping test uses the true residual) on the matrix-free
    operator, preconditioned by an additive fieldsplit: all non-R blocks together, assembled from
    aP with firedrake.AssembledPC and factored by MUMPS LU; all R blocks together with no
    preconditioner (their aP block is the identity-like sum_i p_i q_i).

    Args:
        num_blocks: number of blocks of the mixed space (including the R blocks).
        multiplier_indices: block indices of the R spaces (a single int is accepted).
    """
    multipliers = [int(i) for i in _as_list(multiplier_indices)]
    others = ','.join(str(i) for i in range(num_blocks) if i not in multipliers)
    return {
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'ksp_type': 'gmres',
        'ksp_pc_side': 'right',
        'ksp_gmres_restart': 200,
        'ksp_rtol': rtol,
        'ksp_atol': atol,
        'ksp_max_it': max_it,
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'additive',
        'pc_fieldsplit_0_fields': others,
        'pc_fieldsplit_1_fields': ','.join(str(i) for i in multipliers),
        'fieldsplit_0_ksp_type': 'preonly',
        'fieldsplit_0_pc_type': 'python',
        'fieldsplit_0_pc_python_type': 'firedrake.AssembledPC',
        'fieldsplit_0_assembled_pc_type': 'lu',
        'fieldsplit_0_assembled_pc_factor_mat_solver_type': 'mumps',
        'fieldsplit_1_ksp_type': 'preonly',
        'fieldsplit_1_pc_type': 'none',
    }


def multiplier_coefficients(p):
    """Values p_i of the multiplier Functions (list of scalar R Functions, or a single one)."""
    values = []
    for p_i in _as_list(p):
        values.extend(float(x) for x in p_i.dat.data_ro.reshape(-1))
    return values