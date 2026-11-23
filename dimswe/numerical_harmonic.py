"""Numerical harmonic forms (TODO-elliptic §4.13).

The discrete harmonic k-forms are the kernel of the mixed Hodge Laplacian with homogeneous BCs:
the eigenvectors (sigma, u) = (0, h) of A x = lambda M x with lambda = 0 (dimswe.elliptic_eigen),
where A is the symmetric operator of dimswe.elliptic.hodge_bilinear_form (unweighted: the kernel
does not depend on mu, kappa) and M the u-block mass. Their number is the Betti number
(dimswe.topology.betti_number), which is computed independently and used as a check.

Two methods (elliptic.harmonic_method):
- 'eigen' (default): SLEPc shift-and-invert eigen-solve near 0 (target -1), keeping the eigenvalues
  below zero_tolerance times the largest computed eigenvalue. Without a known count the number of
  requested eigenpairs is doubled until a clearly non-zero eigenvalue is found.
- 'kernel': block inverse (subspace) iteration with the regularized, non-singular A + eps M, then
  Rayleigh-Ritz (no SLEPc needed). It needs the count, from the Betti number or
  elliptic.harmonic_forms_count, and carries two extra vectors to estimate the spectral gap.

Acceptance checks (both methods): the number of zero eigenvalues equals the expected count (if
known); sigma is negligible against u for every accepted mode (true harmonic forms have
sigma = 0); and the gap ratio (first non-zero eigenvalue over the largest accepted one) is large
(a warning below MIN_GAP_RATIO). The accepted u parts are L2-orthonormalized, and the result
is cached on the mesh, since it depends only on the mesh, spaces, k and BC types.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy
from firedrake import (Cofunction, Function, LinearEigenproblem, LinearEigensolver, LinearSolver, assemble, dx,
                       inner)

from .harmonic_forms import l2_inner, orthonormalize_harmonic_basis

HARMONIC_METHODS = ('eigen', 'kernel')
DEFAULT_ZERO_TOLERANCE = 1e-8
# Absolute floor of the zero threshold: eigenvalues of size round-off are zero even when every
# computed eigenvalue is tiny (relative tests alone cannot tell 0 from 1e-15).
ABSOLUTE_ZERO = 1e-10
MIN_GAP_RATIO = 1e6
MAX_REQUEST = 64


@dataclass
class NumericalHarmonicInfo:
    """Diagnostics of a numerical harmonic-form computation."""
    method: str
    count: int
    expected: Optional[int]
    eigenvalues: list = field(default_factory=list)
    gap_ratio: float = float('inf')
    max_sigma_ratio: float = 0.0
    warnings: list = field(default_factory=list)


def classify_zero_eigenvalues(eigenvalues, expected=None, zero_tolerance=DEFAULT_ZERO_TOLERANCE):
    """Number of zero eigenvalues and the gap ratio, from sorted non-negative eigenvalues (pure Python).

    An eigenvalue is zero if it is below max(zero_tolerance times the largest eigenvalue given,
    ABSOLUTE_ZERO); pass only physical eigenvalues (see physical_eigenpairs). The
    gap ratio is the first non-zero eigenvalue over the largest zero one (inf if the zero ones are
    exactly 0, nan if there is no non-zero eigenvalue to compare with). Raises ValueError if
    expected is given and differs from the count.
    """
    values = sorted(float(v) for v in eigenvalues)
    if not values:
        raise ValueError("no eigenvalues to classify")
    scale = max(abs(v) for v in values)
    threshold = max(zero_tolerance * scale, ABSOLUTE_ZERO)
    count = sum(1 for v in values if abs(v) <= threshold)
    if count < len(values):
        largest_zero = max([abs(v) for v in values[:count]], default=0.0)
        gap = float('inf') if largest_zero == 0.0 else values[count] / largest_zero
    else:
        gap = float('nan')
    if expected is not None and count != expected:
        raise ValueError("found " + str(count) + " zero eigenvalues, but the Betti number is " + str(expected)
                         + "; computed spectrum " + str(values))
    return count, gap


def physical_eigenpairs(pairs, layout, measure, relative_u=1e-8, negative_tolerance=1e-6):
    """Drop the spurious eigenpairs of the singular pencil (A, M) and sort by eigenvalue.

    M is the u-block mass, zero on the sigma block, so the pencil has "infinite" eigenvalues whose
    eigenvectors are pure sigma modes (u = 0). Shift-and-invert maps them near zero, and SLEPc
    can return them as huge values of either sign (seen: -1e16 ... +3e16 in 1D). The Hodge
    Laplacian has lambda >= 0 and every physical eigenvector has u != 0, so pairs with a negative
    eigenvalue (beyond round-off) or a negligible u part are removed. pairs: (value, Function).
    """
    kept = []
    for value, mode in pairs:
        u = _u_part(mode, layout)
        u_norm = l2_inner(u, u, measure) ** 0.5
        total = sum(l2_inner(sub, sub, measure) for sub in (mode.subfunctions if layout.num_blocks > 1 else [mode])) ** 0.5
        if total == 0.0 or u_norm <= relative_u * total:
            continue  # pure sigma mode: an infinite eigenvalue of the pencil
        if value < -negative_tolerance:
            continue  # the Hodge Laplacian has no negative eigenvalues (beyond round-off)
        kept.append((value, mode))
    return sorted(kept, key=lambda pair: pair[0])


def _problem_forms(spaces, config, measure):
    """Unweighted mixed operator, u mass and homogeneous essential BCs on the [sigma, u] space."""
    from .elliptic import build_block_layout, build_essential_bc_lists, hodge_bilinear_form, split_arguments
    layout = build_block_layout(spaces, config.k, config.dim, config.complex)
    trial, test = split_arguments(layout)
    a = hodge_bilinear_form(trial, test, config.k, config.dim, config.complex, measure)
    m = inner(trial['u'], test['u']) * measure
    bcs_sigma, bcs_u = build_essential_bc_lists(layout, config)
    return layout, a, m, bcs_sigma + bcs_u


def _u_part(function, layout):
    """The u block of a Function on the [sigma, u] space."""
    return function if layout.num_blocks == 1 else function.subfunctions[layout.u_index]


def _sigma_ratio(function, layout, measure):
    """||sigma|| / ||u|| of a mode (0 for k = 0)."""
    if layout.sigma_index is None:
        return 0.0
    sigma = function.subfunctions[layout.sigma_index]
    u = _u_part(function, layout)
    return (l2_inner(sigma, sigma, measure) / max(l2_inner(u, u, measure), 1e-300)) ** 0.5


def _eigen_modes(layout, a, m, bcs, expected, zero_tolerance, measure, logger):
    """Zero modes by SLEPc shift-and-invert; returns (modes, eigenvalues, count, gap)."""
    from .elliptic_eigen import EigenConfig
    eigen = EigenConfig(nev=1, target=-1.0, tolerance=1e-12)
    try:
        problem = LinearEigenproblem(a, M=m, bcs=bcs or None, bc_shift=1e8, restrict=True)
    except TypeError:
        problem = LinearEigenproblem(a, M=m, bcs=bcs or None, bc_shift=1e8)
    request = (expected if expected is not None else 1) + 4
    while True:
        solver = LinearEigensolver(problem, n_evals=request, solver_parameters=eigen.solver_parameters(),
                                   ncv=max(2 * request, request + 16), options_prefix='harmonic_eigen_')
        nconv = solver.solve()
        raw = [(float(complex(solver.eigenvalue(i)).real), solver.eigenfunction(i)[0].copy(deepcopy=True))
               for i in range(nconv)]
        pairs = physical_eigenpairs(raw, layout, measure)
        values = [value for value, _ in pairs]
        if not values:
            raise RuntimeError("the harmonic-form eigen-solve converged no physical eigenpairs (raw spectrum "
                               + str(sorted(v for v, _ in raw)) + ")")
        count, gap = classify_zero_eigenvalues(values, None, zero_tolerance)
        if count < len(values) or request >= MAX_REQUEST:
            break
        request = min(2 * request, MAX_REQUEST)   # all requested eigenvalues are zero: ask for more
        logger.output('numerical harmonic forms: all ' + str(len(values)) + ' eigenvalues are zero, requesting '
                      + str(request), 1)
    if expected is not None:
        classify_zero_eigenvalues(values, expected, zero_tolerance)   # raises with the spectrum
    return [pair[1].copy(deepcopy=True) for pair in pairs[:count]], values, count, gap


def _kernel_modes(layout, a, m, bcs, expected, zero_tolerance, measure, regularization=1e-4, iterations=8,
                  extra=2, seed=2024):
    """Zero modes by block inverse iteration x <- (A + eps M)^{-1} M x, then Rayleigh-Ritz.

    A + eps M is non-singular (the kernel of A is in the u block, where M is positive definite).
    The harmonic modes are amplified by 1/eps per sweep against at most 1/(lambda_1 + eps) for the
    rest, so a few sweeps suffice; `extra` additional vectors give the first non-zero Ritz values,
    i.e. the gap estimate. Vectors are kept zero on the essential-BC rows.
    """
    if expected is None:
        raise ValueError("harmonic_method: kernel needs the number of harmonic forms; set "
                         "elliptic.harmonic_forms_count (the Betti number could not be computed)")
    if expected == 0:
        return [], [], 0, float('inf')
    W = layout.W
    A = assemble(a, bcs=bcs or None).petscmat
    M = assemble(m).petscmat
    shifted = assemble(a + regularization * m, bcs=bcs or None)
    solver = LinearSolver(shifted, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu',
                                                      'pc_factor_mat_solver_type': 'mumps'},
                          options_prefix='harmonic_kernel_')
    bc_nodes = [(_bc_block(bc, layout), bc.nodes) for bc in bcs]
    rng = numpy.random.default_rng(seed)
    block = []
    for _ in range(expected + extra):
        f = Function(W)
        for sub in (f.subfunctions if layout.num_blocks > 1 else [f]):
            sub.dat.data[:] = rng.standard_normal(sub.dat.data.shape)
        _zero_rows(f, bc_nodes, layout)
        block.append(f)
    for _ in range(iterations):
        new = []
        for f in block:
            rhs = Cofunction(W.dual())
            with f.dat.vec_ro as x, rhs.dat.vec_wo as b:
                M.mult(x, b)
            _zero_rows(rhs, bc_nodes, layout)
            out = Function(W)
            solver.solve(out, rhs)
            new.append(out)
        block = _m_orthonormalize(new, M)
    values, block = _rayleigh_ritz(block, A, M)
    count, gap = classify_zero_eigenvalues(values, expected, zero_tolerance)
    return block[:count], values, count, gap


def _bc_block(bc, layout):
    """Block index of the space a DirichletBC acts on (0 for a non-mixed space)."""
    if layout.num_blocks == 1:
        return 0
    index = getattr(bc.function_space(), 'index', None)
    return 0 if index is None else index


def _zero_rows(function, bc_nodes, layout):
    """Set the essential-BC rows of a (Co)Function to zero."""
    for block, nodes in bc_nodes:
        target = function if layout.num_blocks == 1 else function.subfunctions[block]
        target.dat.data[nodes] = 0.0


def _m_orthonormalize(block, M):
    """Modified Gram-Schmidt in the (semi-definite) M inner product, dropping dependent vectors."""
    result = []
    for f in block:
        with f.dat.vec as x:
            for q in result:
                with q.dat.vec_ro as y:
                    Mx = x.duplicate()
                    M.mult(y, Mx)
                    x.axpy(-x.dot(Mx), y)
            Mx = x.duplicate()
            M.mult(x, Mx)
            norm = x.dot(Mx) ** 0.5
            if norm <= 1e-14:
                continue
            x.scale(1.0 / norm)
        result.append(f)
    return result


def _rayleigh_ritz(block, A, M):
    """Ritz values and vectors of A x = lambda M x in span(block), sorted ascending."""
    import scipy.linalg
    n = len(block)
    Ap, Mp = numpy.zeros((n, n)), numpy.zeros((n, n))
    vecs = []
    for f in block:
        with f.dat.vec_ro as x:
            vecs.append(x.copy())
    tmp = vecs[0].duplicate()
    for j in range(n):
        A.mult(vecs[j], tmp)
        for i in range(n):
            Ap[i, j] = vecs[i].dot(tmp)
        M.mult(vecs[j], tmp)
        for i in range(n):
            Mp[i, j] = vecs[i].dot(tmp)
    Ap, Mp = 0.5 * (Ap + Ap.T), 0.5 * (Mp + Mp.T)
    values, coefficients = scipy.linalg.eigh(Ap, Mp)
    ritz = []
    for c in coefficients.T:
        f = Function(block[0].function_space())
        with f.dat.vec as x:
            x.set(0.0)
            for weight, v in zip(c, vecs):
                x.axpy(float(weight), v)
        ritz.append(f)
    return [float(v) for v in values], ritz


def numerical_harmonic_basis(spaces, config, mesh, parameters, Vk, measure=dx, method='eigen', expected=None,
                             zero_tolerance=DEFAULT_ZERO_TOLERANCE, sigma_tolerance=1e-6, logger=None):
    """L2-orthonormal basis of the discrete harmonic k-forms in V^k, and NumericalHarmonicInfo.

    expected: number of harmonic forms if known (elliptic.harmonic_forms_count); otherwise the
    Betti number is computed (dimswe.topology), or None in parallel / for very large meshes.
    Raises ValueError for a count mismatch or a mode with non-negligible sigma.
    """
    from .logger import EmptyLogger
    from .topology import betti_number
    logger = logger if logger is not None else EmptyLogger()
    if method not in HARMONIC_METHODS:
        raise ValueError("harmonic_method must be one of " + str(list(HARMONIC_METHODS)) + ", got " + repr(method))
    if expected is None:
        expected = betti_number(mesh, parameters, config)
    key = (method, config.k, config.complex, spaces.family, spaces.order, tuple(sorted(config.face_bcs.items())),
           expected, zero_tolerance)
    cache = getattr(mesh, '_dimswe_numerical_harmonic', None)
    if cache is None:
        cache = {}
        mesh._dimswe_numerical_harmonic = cache
    if key not in cache:
        layout, a, m, bcs = _problem_forms(spaces, config, measure)
        if expected == 0:
            modes, values, count, gap = [], [], 0, float('inf')
        elif method == 'eigen':
            modes, values, count, gap = _eigen_modes(layout, a, m, bcs, expected, zero_tolerance, measure, logger)
        else:
            modes, values, count, gap = _kernel_modes(layout, a, m, bcs, expected, zero_tolerance, measure)
        ratios = [_sigma_ratio(mode, layout, measure) for mode in modes]
        info = NumericalHarmonicInfo(method=method, count=count, expected=expected, eigenvalues=values,
                                     gap_ratio=gap, max_sigma_ratio=max(ratios, default=0.0))
        if info.max_sigma_ratio > sigma_tolerance:
            raise ValueError("a zero mode has ||sigma|| / ||u|| = " + str(info.max_sigma_ratio)
                             + "; it is not a harmonic form (spectrum " + str(values) + ")")
        if count > 0 and gap < MIN_GAP_RATIO:
            info.warnings.append("small spectral gap " + str(gap) + " between zero and non-zero eigenvalues")
        data = [_u_part(mode, layout).dat.data_ro.copy() for mode in modes]
        cache[key] = (data, info)
    data, info = cache[key]
    # Rebuild in the caller's V^k (same mesh and element: same dof numbering), then orthonormalize.
    functions = []
    for values in data:
        h = Function(Vk)
        h.dat.data[:] = values
        functions.append(h)
    basis = orthonormalize_harmonic_basis(functions, Vk, measure=measure)
    for warning in info.warnings:
        logger.output('numerical harmonic forms: WARNING ' + warning, 0)
    logger.output('numerical harmonic forms (' + info.method + '): ' + str(info.count) + ' forms, expected '
                  + str(info.expected) + ', gap ratio ' + str(info.gap_ratio), 0)
    return basis, info
