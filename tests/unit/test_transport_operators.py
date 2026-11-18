"""Unit tests of the DG Lie derivative operators in dimswe.transport_operators.

The tests check CVLieDerivative, the weak Lie derivative of a covector-valued volume form (such as
the momentum density m in the Lie-Poisson bracket), against the strong form
(L_u m)_i = d_j(u_j m_i) + m_j d_i u_j, and check its antisymmetry in (u, mtest).
"""

import numpy as np
from firedrake import (
    Function,
    PeriodicRectangleMesh,
    SpatialCoordinate,
    VectorFunctionSpace,
    FacetNormal,
    as_vector,
    assemble,
    cos,
    div,
    dot,
    dS,
    dx,
    grad,
    inner,
    outer,
    pi,
    sin,
)

from dimswe.transport_operators import CVLieDerivative

ORDER = 2


def periodic_mesh():
    """Return a small periodic unit-square quadrilateral mesh."""
    return PeriodicRectangleMesh(8, 8, 1.0, 1.0, quadrilateral=True)


def smooth_fields(mesh):
    """Return smooth, mutually non-parallel vector expressions (u, m, w) on the unit square.

    m is deliberately not parallel to u, so outer(u, m) is not symmetric and an index
    transposition in the volume terms cannot cancel out.
    """
    x, y = SpatialCoordinate(mesh)
    u = as_vector([sin(2 * pi * x) * cos(2 * pi * y) + 0.3, 0.5 * cos(2 * pi * x)])
    m = as_vector([cos(2 * pi * y) + 0.2 * sin(2 * pi * x), 1.0 + sin(2 * pi * (x + y))])
    w = as_vector([sin(2 * pi * (x - y)), cos(2 * pi * x) * sin(2 * pi * y) - 0.4])
    return u, m, w


def lie_form(u, m, w, mesh):
    """Return the CVLieDerivative form for velocity u, momentum m and test field w (alpha_s = 1)."""
    n = FacetNormal(mesh)
    return CVLieDerivative(2, 2, u, m, w, 1.0, 1.0, u, 1.0, n, ORDER, dx, dS)


def test_cv_lie_derivative_matches_strong_form():
    """For continuous fields, CVLieDerivative equals int w . L_u m dx with the correct index order.

    With continuous u, m, w all facet terms vanish, so this isolates the volume terms. On a
    periodic mesh integration by parts has no boundary terms, so the weak and strong forms must
    agree up to round-off.
    """
    mesh = periodic_mesh()
    V = VectorFunctionSpace(mesh, 'CG', ORDER)
    u_expr, m_expr, w_expr = smooth_fields(mesh)
    u = Function(V).interpolate(u_expr)
    m = Function(V).interpolate(m_expr)
    w = Function(V).interpolate(w_expr)

    weak = assemble(lie_form(u, m, w, mesh))
    # Strong form: d_j(m_i u_j) is div(outer(m, u)), and m_j d_i u_j is grad(u)^T . m.
    strong = assemble(inner(w, div(outer(m, u)) + dot(grad(u).T, m)) * dx)

    assert abs(strong) > 1e-3, 'test fields are too degenerate to detect errors'
    assert np.isclose(weak, strong, rtol=1e-10, atol=1e-12), (weak, strong)

    # The previous (buggy) implementation used outer(u, m) and outer(w, m), which corresponds to
    # d_j(u_i m_j) + m_j d_j u_i. It gives a measurably different value, so this test catches it.
    transposed = assemble(inner(w, div(outer(u, m)) + dot(grad(u), m)) * dx)
    assert not np.isclose(weak, transposed, rtol=1e-3), (weak, transposed)


def test_cv_lie_derivative_antisymmetric_for_discontinuous_fields():
    """CVLieDerivative(u, m, w) = -CVLieDerivative(w, m, u) for discontinuous DG fields.

    The upwinding of m uses sign(u . n), so the upwind velocity is fixed to u in both
    evaluations; antisymmetry then has to hold including the facet terms.
    """
    mesh = periodic_mesh()
    n = FacetNormal(mesh)
    DGV = VectorFunctionSpace(mesh, 'DG', ORDER - 1)
    rng = np.random.default_rng(1234)
    u, m, w = Function(DGV), Function(DGV), Function(DGV)
    for f in (u, m, w):
        f.dat.data[:] = rng.standard_normal(f.dat.data.shape)

    forward = assemble(CVLieDerivative(2, 2, u, m, w, 1.0, 1.0, u, 1.0, n, ORDER, dx, dS))
    swapped = assemble(CVLieDerivative(2, 2, w, m, u, 1.0, 1.0, u, 1.0, n, ORDER, dx, dS))

    assert abs(forward) > 1e-6, 'test fields are too degenerate to check antisymmetry'
    assert np.isclose(forward, -swapped, rtol=1e-10, atol=1e-12), (forward, swapped)
