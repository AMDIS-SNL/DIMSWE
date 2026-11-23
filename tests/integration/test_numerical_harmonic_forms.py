"""Numerical harmonic forms (TODO-elliptic §4.13).

- Betti numbers from the incidence ranks match the Künneth rule on boxes;
- the numerical basis spans the same space as the Künneth basis (compared through projectors,
  since bases differ by a rotation) for both methods, on every harmonic box configuration;
- a domain with a hole (a triangulated square with a square hole, built through DMPlex) has one
  harmonic 1-form for natural BCs, which the Künneth rule cannot give;
- the full model runs with harmonic_forms: numerical (MMS, lagrange vs nullspace);
- forcing a wrong count raises.
"""
import numpy
import pytest
from firedrake import COMM_WORLD, Function, Mesh, SpatialCoordinate, assemble, dx, inner, sin
from firedrake.petsc import PETSc

from dimswe.differential_forms import exterior_derivative, form_space
from dimswe.elliptic import EllipticConfig, EllipticProblem
from dimswe.harmonic_forms import l2_inner
from dimswe.meshes import DeRhamComplex
from dimswe.numerical_harmonic import numerical_harmonic_basis
from dimswe.topology import betti_number
from tests.conftest import make_elliptic_parameters

pytestmark = pytest.mark.skipif(COMM_WORLD.size > 1, reason="Betti numbers and these checks are serial")

# (dim, complex, k, bcs) cases with and without harmonic forms (Künneth reference available).
BOX_CASES = [
    (1, None, 0, 'periodic'), (1, None, 0, 'natural'), (1, None, 1, 'essential'), (1, None, 0, 'mixed'),
    (2, 'grad-curl', 1, 'periodic'), (2, 'curl-div', 1, 'periodic'), (2, 'grad-curl', 0, 'natural'),
    (2, 'curl-div', 2, 'essential'), (2, 'grad-curl', 1, 'partial-essential'), (2, 'curl-div', 1, 'partial-natural'),
    (2, 'grad-curl', 1, 'mixed'), (2, 'curl-div', 2, 'periodic'),
]


def case_id(case):
    """Readable pytest id."""
    dim, cx, k, bcs = case
    return str(dim) + "D-" + (cx + "-" if cx else "") + "k" + str(k) + "-" + bcs


def build(case, mode, n=4, order=2, **elliptic):
    """Model parameters for a box case with the given harmonic_forms mode."""
    dim, cx, k, bcs = case
    parameters = make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order)
    parameters['elliptic']['harmonic_forms'] = mode
    parameters['elliptic'].update(elliptic)
    return parameters


def projector_distance(basis_a, basis_b, V, seed=0, samples=3):
    """max over random u of ||P_a u - P_b u|| / ||u|| (P = L2 projector onto the span)."""
    rng = numpy.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        u = Function(V)
        u.dat.data[:] = rng.standard_normal(u.dat.data.shape)
        diff = Function(V)
        for basis, sign in ((basis_a, 1.0), (basis_b, -1.0)):
            for h in basis:
                diff.assign(diff + sign * l2_inner(u, h) * h)
        worst = max(worst, l2_inner(diff, diff) ** 0.5 / l2_inner(u, u) ** 0.5)
    return worst


@pytest.mark.parametrize("case", BOX_CASES, ids=case_id)
def test_betti_number_matches_kunneth(case):
    """b_k from the incidence ranks equals the Künneth count on boxes."""
    auto = EllipticProblem(build(case, 'auto'))
    assert betti_number(auto.mesh, auto.parameters, auto.config) == auto.num_harmonic


@pytest.mark.parametrize("method", ['eigen', 'kernel'])
@pytest.mark.parametrize("case", BOX_CASES, ids=case_id)
def test_numerical_basis_spans_kunneth_basis(case, method):
    """Same dimension and same span as the Künneth basis; orthonormal; discretely harmonic."""
    auto = EllipticProblem(build(case, 'auto'))
    numerical = EllipticProblem(build(case, 'numerical', harmonic_method=method), mesh=auto.mesh)
    assert numerical.num_harmonic == auto.num_harmonic
    if auto.num_harmonic == 0:
        return
    V = auto.Vk
    copies = []
    for h in numerical.harmonic:          # same mesh and element: copy the dofs into auto's V^k
        g = Function(V)
        g.dat.data[:] = h.dat.data_ro
        copies.append(g)
    assert projector_distance(auto.harmonic, copies, V) < 1e-7
    gram = numpy.array([[l2_inner(a, b) for b in copies] for a in copies])
    assert numpy.allclose(gram, numpy.eye(len(copies)), atol=1e-8)
    dim, cx, k, _ = case
    if k < dim:
        for h in copies:
            dh = exterior_derivative(h, k, dim, cx)
            assert assemble(inner(dh, dh) * dx) < 1e-16
    stats_model = numerical
    stats_model.solve()
    stats_model.create_statistics()
    stats_model.compute_statistics()
    stats = stats_model.get_statistics()
    assert stats['harmonic_gap_ratio'][0] > 1e6 and stats['harmonic_max_sigma_ratio'][0] < 1e-6


def test_numerical_model_mms_and_constraints():
    """harmonic_forms: numerical gives the Künneth MMS error, and lagrange == nullspace."""
    case = (2, 'curl-div', 1, 'partial-natural')
    errors = {}
    models = {}
    for mode in ('auto', 'numerical'):
        for constraint in ('lagrange', 'nullspace'):
            parameters = build(case, mode, n=6, harmonic_constraint=constraint)
            mesh = next(iter(models.values())).mesh if models else None
            model = EllipticProblem(parameters, mesh=mesh)
            model.solve()
            model.create_statistics()
            model.compute_statistics()
            models[(mode, constraint)] = model
            errors[(mode, constraint)] = model.get_statistics()['u_error'][0]
    reference = errors[('auto', 'nullspace')]
    for key, value in errors.items():
        assert abs(value - reference) < 1e-8 * max(1.0, reference), (key, errors)
    a, b = models[('numerical', 'lagrange')].sub['u'], models[('numerical', 'nullspace')].sub['u']
    assert assemble(inner(a - b, a - b) * dx) ** 0.5 < 1e-7


def test_wrong_count_raises():
    """Forcing a wrong number of harmonic forms is detected."""
    with pytest.raises(ValueError, match="Betti number"):
        EllipticProblem(build((2, 'grad-curl', 1, 'periodic'), 'numerical', harmonic_forms_count=3))
    with pytest.raises(ValueError, match="Betti number"):
        EllipticProblem(build((2, 'grad-curl', 1, 'periodic'), 'numerical', harmonic_method='kernel',
                              harmonic_forms_count=1))


# ---------------------------------------------------------------------------------------------
# A domain with a hole
# ---------------------------------------------------------------------------------------------

def square_with_hole(n=6, hole=(2, 4)):
    """Triangulated unit square without the cells in [hole[0], hole[1])^2 (an annulus topologically)."""
    h = 1.0 / n
    keep = [(i, j) for i in range(n) for j in range(n)
            if not (hole[0] <= i < hole[1] and hole[0] <= j < hole[1])]
    used = sorted({(i + a, j + b) for i, j in keep for a in (0, 1) for b in (0, 1)})
    index = {v: c for c, v in enumerate(used)}
    coords = numpy.array([[i * h, j * h] for i, j in used], dtype=float)
    cells = []
    for i, j in keep:
        v00, v10, v11, v01 = index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]
        cells += [[v00, v10, v11], [v00, v11, v01]]
    plex = PETSc.DMPlex().createFromCellList(2, numpy.array(cells, dtype=numpy.int32), coords, comm=COMM_WORLD)
    return Mesh(plex)


@pytest.mark.parametrize("method", ['eigen', 'kernel'])
@pytest.mark.parametrize("cx", ['grad-curl', 'curl-div'])
def test_hole_has_one_harmonic_one_form(cx, method):
    """Natural BCs on a square with a hole: b_0 = 1, b_1 = 1, b_2 = 0; the 1-form circulates."""
    mesh = square_with_hole()
    parameters = {'mesh': {'type': 'file', 'dim': 2},
                  'spatial-discretization': {'family': 'P-', 'order': 2, 'lump_mass': False}}
    spaces = DeRhamComplex(mesh, parameters)
    expected = {0: 1, 1: 1, 2: 0}
    for k, count in expected.items():
        config = EllipticConfig(dim=2, k=k, complex=cx, boundary_conditions={}, registry={}, face_bcs={})
        assert betti_number(mesh, parameters, config) == count
        V = form_space(spaces, k, 2, cx)
        basis, info = numerical_harmonic_basis(spaces, config, mesh, parameters, V, method=method)
        assert len(basis) == count and info.count == count
        if k == 1:
            h = basis[0]
            assert assemble(inner(exterior_derivative(h, 1, 2, cx), exterior_derivative(h, 1, 2, cx)) * dx) < 1e-16
            # Harmonic, not exact: (h, d w) = 0 for every discrete w in V^0 (natural BCs: no constraint
            # on w), checked for interpolated smooth potentials. d V^0 is in V^1, so this is exact.
            X = SpatialCoordinate(mesh)
            for expr in (X[0] * X[1], sin(3 * X[0]) + X[1] ** 2):
                w = Function(spaces.CG).interpolate(expr)
                dw = exterior_derivative(w, 0, 2, cx)
                assert abs(assemble(inner(h, dw) * dx)) < 1e-9 * assemble(inner(dw, dw) * dx) ** 0.5
            assert info.gap_ratio > 1e6
