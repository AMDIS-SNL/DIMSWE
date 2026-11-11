"""Bernstein coefficients of finite element fields on axis-aligned cells, written in UFL.

On each cell a polynomial field a of degree p (tensor-product Q_p, continuous or discontinuous) is
written in the Bernstein basis of the local box coordinates xi in [0, 1]^d. The Bernstein basis is
non-negative and a partition of unity, so the coefficients bound the field on the whole cell:

    min_k b_k <= a(x) <= max_k b_k   for every x in the cell,

with equality of the minimum for p <= 1 (the coefficients are then the vertex values). The cell mean
is the average of the coefficients. The coefficients are linear in a,

    b_k = (1/|K|) int_K a psi_k dx,

with psi_k the L2-dual basis of the Bernstein basis (computed exactly in rational arithmetic), i.e.
an L2 projection onto a DG0 vector space with a diagonal mass matrix. Used by the positivity
diagnostics; only axis-aligned boxes (intervals, rectangle-mesh quadrilaterals) are supported.
"""
import itertools
from fractions import Fraction
from functools import reduce
from math import comb

import numpy as np
from firedrake import (Function, FunctionSpace, SpatialCoordinate, TestFunction, VectorFunctionSpace,
                       as_vector, assemble, inner, min_value)
import ufl


def bernstein_polynomial(p, i, t):
    """Return the 1D Bernstein polynomial B_i^p(t) = C(p, i) t^i (1 - t)^(p - i) on [0, 1].

    Zero powers are left out rather than written as t**0, so UFL never sees (or differentiates)
    a power with exponent zero.
    """
    value = comb(p, i)
    if i > 0:
        value = value * t**i
    if p - i > 0:
        value = value * (1 - t)**(p - i)
    return value


def bernstein_gram_matrix(p):
    """Return the exact Gram matrix G_mn = int_0^1 B_m^p B_n^p dt as Fractions."""
    def entry(m, n):
        # int_0^1 t^(m+n) (1-t)^(2p-m-n) dt = (m+n)! (2p-m-n)! / (2p+1)!  (a Beta function)
        return Fraction(comb(p, m) * comb(p, n), (2 * p + 1) * comb(2 * p, m + n))
    return [[entry(m, n) for n in range(p + 1)] for m in range(p + 1)]


def invert_exact(matrix):
    """Return the inverse of a square matrix of Fractions by exact Gauss-Jordan elimination."""
    n = len(matrix)
    augmented = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if augmented[r][col] != 0)
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for r in range(n):
            if r != col and augmented[r][col] != 0:
                factor = augmented[r][col]
                augmented[r] = [a - factor * b for a, b in zip(augmented[r], augmented[col])]
    return [row[n:] for row in augmented]


def dual_bernstein_coefficients(p):
    """Return D with D[i][m] such that d_i = sum_m D[i][m] B_m^p satisfies int_0^1 d_i B_j = delta_ij."""
    inverse = invert_exact(bernstein_gram_matrix(p))
    return [[float(value) for value in row] for row in inverse]


def dual_bernstein_polynomial(p, i, t, dual=None):
    """Return the 1D dual Bernstein polynomial d_i(t) (a UFL expression if t is one)."""
    dual = dual_bernstein_coefficients(p) if dual is None else dual
    return sum(dual[i][m] * bernstein_polynomial(p, m, t) for m in range(p + 1))


def ufl_attribute(obj, name):
    """Return obj.name, calling it if it is a method.

    UFL and Firedrake have turned several of these methods into properties (for example
    Cell.cellname), so this supports both the older and the newer versions.
    """
    value = getattr(obj, name)
    return value() if callable(value) else value


class CellBoxGeometry():
    """Local box coordinates of axis-aligned cells, xi_d = (x_d - x0_d) / w_d in [0, 1].

    The lower corners x0 and widths w are DG0 Functions computed once from the mesh coordinates
    (centre by interpolation, width from the cell variance of each coordinate, w^2 = 12 var), so they
    do not depend on the state. On periodic meshes the coordinate field is discontinuous, so every
    cell sees its own unwrapped coordinates.
    """
    def __init__(self, mesh):
        cellname = ufl_attribute(mesh.ufl_cell(), 'cellname')
        if cellname not in ('interval', 'quadrilateral'):
            raise NotImplementedError(f'Bernstein coefficients need interval or quadrilateral cells, not {cellname}')
        self.mesh = mesh
        self.dim = ufl_attribute(mesh, 'geometric_dimension')
        X = SpatialCoordinate(mesh)
        V0 = FunctionSpace(mesh, 'DG', 0)
        self.V0 = V0
        self.cell_volume = assemble(TestFunction(V0) * ufl.dx(domain=mesh)).dat.data_ro.copy()
        centres = [Function(V0).interpolate(X[d]) for d in range(self.dim)]
        widths = []
        for d in range(self.dim):
            variance = self.cell_mean((X[d] - centres[d])**2)
            widths.append(Function(V0, name=f'cell_width_{d}'))
            widths[d].dat.data[:] = np.sqrt(12.0 * np.maximum(variance, 0.0))
        self.check_axis_aligned(X, centres, widths)
        self.lower = [Function(V0, name=f'cell_lower_{d}').interpolate(centres[d] - 0.5 * widths[d]) for d in range(self.dim)]
        self.widths = widths
        self.xi = [(X[d] - self.lower[d]) / self.widths[d] for d in range(self.dim)]

    def cell_mean(self, f):
        """Return the cell means (1/|K|) int_K f dx as a local array (exact up to quadrature)."""
        return assemble(TestFunction(self.V0) * f * ufl.dx(domain=self.mesh)).dat.data_ro / self.cell_volume

    def check_axis_aligned(self, X, centres, widths):
        """Raise unless every cell is an axis-aligned box (zero coordinate covariance, volume = prod w)."""
        box_volume = np.prod([w.dat.data_ro for w in widths], axis=0)
        ok = bool(np.allclose(box_volume, self.cell_volume, rtol=1e-8, atol=0.0))
        if self.dim == 2:
            covariance = self.cell_mean((X[0] - centres[0]) * (X[1] - centres[1]))
            scale = widths[0].dat.data_ro * widths[1].dat.data_ro
            ok = ok and bool(np.all(np.abs(covariance) <= 1e-8 * scale))
        ok = bool(self.mesh.comm.allreduce(ok, op=_mpi_land()))
        if not ok:
            raise NotImplementedError('Bernstein coefficients need axis-aligned (affine box) cells')


def _mpi_land():
    """Return the MPI logical-and reduction."""
    from mpi4py import MPI
    return MPI.LAND


def get_cell_box_geometry(spaces):
    """Return the (cached) CellBoxGeometry of the mesh of a DeRhamComplex."""
    geometry = getattr(spaces, '_bernstein_geometry', None)
    if geometry is None:
        geometry = CellBoxGeometry(spaces.mesh)
        spaces._bernstein_geometry = geometry
    return geometry


def has_cell_box_geometry(spaces):
    """Return whether the mesh cells are axis-aligned boxes, so Bernstein coefficients are available."""
    try:
        get_cell_box_geometry(spaces)
    except NotImplementedError:
        return False
    return True


def field_degree(space):
    """Return the polynomial degree per direction of a scalar function space (max over directions)."""
    degree = ufl_attribute(space.ufl_element(), 'degree')
    return max(degree) if isinstance(degree, tuple) else degree


class BernsteinCoefficients():
    """Bernstein coefficients of scalar fields of a given degree, as a DG0 vector Function.

    coefficient_expressions() gives the [lhs, rhs] of the (diagonal) projection; mean() and
    minimum() give the cell mean and the whole-cell lower bound from the coefficients.
    """
    def __init__(self, spaces, degree):
        self.degree = degree
        self.geometry = get_cell_box_geometry(spaces)
        self.dim = self.geometry.dim
        self.multi_indices = list(itertools.product(range(degree + 1), repeat=self.dim))
        self.ncoeff = len(self.multi_indices)
        self.space = VectorFunctionSpace(spaces.mesh, 'DG', 0, dim=self.ncoeff)
        # a and psi_k have degree p in each local coordinate, so degree 2p integrates a psi_k exactly
        self.dx = ufl.Measure('dx', domain=spaces.mesh, metadata={'quadrature_degree': 2 * degree})
        dual = dual_bernstein_coefficients(degree)
        self.dual_functions = [
            reduce(lambda f, g: f * g, [dual_bernstein_polynomial(degree, k[d], self.geometry.xi[d], dual) for d in range(self.dim)])
            for k in self.multi_indices
        ]

    def coefficient_expressions(self, a, bhat, b):
        """Return [lhs, rhs] defining the Bernstein coefficients b of a (bhat the test function)."""
        moments = as_vector([a * psi for psi in self.dual_functions])
        return [inner(bhat, b) * self.dx, inner(bhat, moments) * self.dx]

    def mean(self, b):
        """Return the cell mean of the field, the average of its Bernstein coefficients."""
        return sum(b[k] for k in range(self.ncoeff)) / self.ncoeff

    def minimum(self, b):
        """Return the smallest Bernstein coefficient, a lower bound of the field on the whole cell."""
        return reduce(min_value, [b[k] for k in range(self.ncoeff)])
