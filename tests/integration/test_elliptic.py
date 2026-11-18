"""Integration tests for EllipticProblem (TODO-elliptic §4.6, §6).

Covers MMS convergence over the (dim, complex, k) x BC matrix (with inhomogeneous exact boundary
data), lagrange vs nullspace agreement, incompatible sources, the MINRES solver, problems without
exact solutions, and the statistics/diagnostics interface. 3D cases are marked slow.
"""
import math

import pytest
from firedrake import assemble, dx, inner

from dimswe.elliptic import EllipticProblem
from tests.conftest import make_elliptic_parameters

COMPLEX_FOR_DIM = {1: [None], 2: ['grad-curl', 'curl-div'], 3: [None]}
ALL_CASES = [(dim, cx, k) for dim in (1, 2, 3) for cx in COMPLEX_FOR_DIM[dim] for k in range(dim + 1)]
# Resolutions and order per dimension for the convergence tests.
RESOLUTION = {1: (8, 16, 2), 2: (6, 12, 2), 3: (3, 6, 1)}
BC_SETS = ['periodic', 'essential', 'natural', 'mixed', 'partial-essential', 'partial-natural']


def case_id(case):
    """Readable pytest id for a (dim, complex, k) case."""
    dim, cx, k = case
    return str(dim) + "D-" + (cx + "-" if cx else "") + "k" + str(k)


def slow_if_3d(cases):
    """Attach the slow marker to 3D cases."""
    return [pytest.param(*c, marks=pytest.mark.slow) if c[0] == 3 else pytest.param(*c) for c in cases]


def run(parameters, mesh=None):
    """Build, solve and compute statistics/diagnostics; returns the model.

    Pass the mesh of an earlier model to compare fields of two models: forms cannot mix
    functions from different meshes.
    """
    model = EllipticProblem(parameters, mesh=mesh)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    model.create_diagnostics()
    model.compute_diagnostics()
    return model


def l2_diff(a, b):
    """L2 norm of a - b."""
    return assemble(inner(a - b, a - b) * dx) ** 0.5


def l2(a):
    """L2 norm of a."""
    return assemble(inner(a, a) * dx) ** 0.5


# ---------------------------------------------------------------------------------------------
# MMS convergence
# ---------------------------------------------------------------------------------------------

CONVERGENCE_CASES = [(dim, cx, k, bcs) for (dim, cx, k) in ALL_CASES for bcs in BC_SETS
                     if not (dim == 1 and bcs.startswith('partial'))]


@pytest.mark.parametrize("dim, cx, k, bcs", slow_if_3d(CONVERGENCE_CASES))
def test_mms_convergence(dim, cx, k, bcs):
    """Observed L2 rates >= expected - 0.3: u: order + 1 for k = 0, order otherwise; sigma, d sigma,
    d u: order. Boundary data are the (inhomogeneous) exact traces; the solution stays orthogonal
    to the harmonic forms."""
    n0, n1, order = RESOLUTION[dim]
    stats = []
    for n in (n0, n1):
        model = run(make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order))
        stats.append(model.get_statistics())
        assert model.get_statistics()['harmonic_orthogonality'][0] < 1e-8
    expected = {'u_error': order + 1 if k == 0 else order}
    if k > 0:
        expected.update({'sigma_error': order, 'dsigma_error': order})
    if k < dim:
        expected['du_error'] = order
    errors = {name: (stats[0][name][0], stats[1][name][0]) for name in expected}
    for name, rate in expected.items():
        e0, e1 = errors[name]
        if e1 < 1e-11:
            continue  # exact up to round-off
        observed = math.log(e0 / e1) / math.log(n1 / n0)
        assert observed >= rate - 0.3, (name, observed, rate, errors)


# ---------------------------------------------------------------------------------------------
# Harmonic constraint: lagrange vs nullspace
# ---------------------------------------------------------------------------------------------

HARMONIC_CASES = [(dim, cx, k, 'periodic') for (dim, cx, k) in ALL_CASES] + [
    (1, None, 0, 'natural'), (1, None, 1, 'essential'),
    (2, 'grad-curl', 0, 'natural'), (2, 'curl-div', 2, 'essential'),
    (2, 'grad-curl', 1, 'partial-essential'), (2, 'curl-div', 1, 'partial-natural'),
    (3, None, 1, 'partial-natural'), (3, None, 2, 'partial-essential'),
]


@pytest.mark.parametrize("dim, cx, k, bcs", slow_if_3d(HARMONIC_CASES))
def test_lagrange_and_nullspace_agree(dim, cx, k, bcs):
    """Incompatible source: both approaches give the same (sigma, u, p); u is orthogonal to the
    harmonic forms; on fully periodic meshes p matches the added harmonic component."""
    n, order = (4, 1) if dim == 3 else (6, 2)
    models = {}
    for constraint in ('lagrange', 'nullspace'):
        parameters = make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order, problem='incompatible-source',
                                              harmonic_constraint=constraint, amplitude=2.0)
        mesh = models['lagrange'].mesh if models else None
        models[constraint] = run(parameters, mesh=mesh)
    lagrange, nullspace = models['lagrange'], models['nullspace']
    assert lagrange.num_harmonic > 0
    assert lagrange.get_x_var_list() == nullspace.get_x_var_list()
    for name in lagrange.get_x_var_list():
        a, b = lagrange.sub[name], nullspace.sub[name]
        assert l2_diff(a, b) < 1e-7 * max(1.0, l2(a)), name
    for model in models.values():
        stats = model.get_statistics()
        assert stats['harmonic_orthogonality'][0] < 1e-8
        assert stats['p_norm'][0] > 0.5
        if bcs == 'periodic':
            assert stats['p_error'][0] < 1e-8 * stats['p_norm'][0]


def test_constraint_choice_irrelevant_without_harmonic_forms():
    """With no harmonic forms both constraint settings run the same non-singular solve."""
    results = []
    for constraint in ('lagrange', 'nullspace'):
        mesh = results[0].mesh if results else None
        model = run(make_elliptic_parameters(2, 'grad-curl', 1, 'mixed', n=4, harmonic_constraint=constraint),
                    mesh=mesh)
        assert model.num_harmonic == 0 and model.get_x_var_list() == ['sigma', 'u']
        results.append(model)
    assert l2_diff(results[0].sub['u'], results[1].sub['u']) < 1e-12


@pytest.mark.parametrize("cx, k, bcs", [('grad-curl', 1, 'periodic'), ('curl-div', 1, 'periodic'),
                                         ('grad-curl', 0, 'natural'), ('curl-div', 1, 'essential')])
def test_minres_matches_direct(cx, k, bcs):
    """The MINRES/Riesz solver reproduces the direct solve (nullspace approach where needed)."""
    reference = run(make_elliptic_parameters(2, cx, k, bcs, n=4, harmonic_constraint='nullspace'))
    minres = run(make_elliptic_parameters(2, cx, k, bcs, n=4, harmonic_constraint='nullspace',
                                          solver='elliptic-minres'), mesh=reference.mesh)
    for name in reference.get_x_var_list():
        assert l2_diff(reference.sub[name], minres.sub[name]) < 1e-7 * max(1.0, l2(reference.sub[name])), name


# ---------------------------------------------------------------------------------------------
# Problems without exact solutions
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cx, k", [('grad-curl', 0), ('grad-curl', 1), ('curl-div', 1), ('curl-div', 2)])
def test_boundary_driven_problem(cx, k):
    """f = 0 with data on one face only gives a nonzero solution; no error statistics."""
    parameters = make_elliptic_parameters(2, cx, k, 'mixed', n=4, problem='boundary-driven', face='left')
    parameters['elliptic']['boundary_data'] = 'problem'
    model = run(parameters)
    assert model.get_statistics()['u_norm'][0] > 1e-3
    assert 'u_error' not in model.get_statistics_list()


def test_gaussian_source_problem():
    """Gaussian source with homogeneous data (natural k = 0 has constants: p takes the mean)."""
    parameters = make_elliptic_parameters(2, 'grad-curl', 0, 'natural', n=4, problem='gaussian-source')
    parameters['elliptic']['boundary_data'] = 'homogeneous'
    model = run(parameters)
    stats = model.get_statistics()
    assert stats['u_norm'][0] > 1e-4 and stats['p_norm'][0] > 1e-4
    assert stats['harmonic_orthogonality'][0] < 1e-8


def test_exact_boundary_data_requires_exact_solution():
    """boundary_data: exact is rejected for problems without an exact solution."""
    parameters = make_elliptic_parameters(2, 'grad-curl', 0, 'essential', n=2, problem='gaussian-source')
    with pytest.raises(ValueError):
        EllipticProblem(parameters)


# ---------------------------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------------------------

def test_model_interface():
    """Variable, statistics and diagnostics lists, and diagnostic spaces."""
    model = run(make_elliptic_parameters(2, 'curl-div', 1, 'partial-essential', n=4))
    assert model.is_time_invariant
    assert model.get_x_var_list() == ['sigma', 'u', 'p']
    assert model.get_aux_var_list() == [] and model.get_coeff_list() == []
    assert not model.has_aux() and not model.has_coeff()
    stats = model.get_statistics()
    assert set(stats) == set(model.get_statistics_list())
    assert all(len(v) == 1 for v in stats.values())
    assert {'u_error', 'sigma_error', 'dsigma_error', 'du_error', 'p_error'} <= set(stats)
    diagnostics = model.get_diagnostics()
    assert set(diagnostics) == {'u_exact', 'u_error', 'sigma_error', 'du'}
    assert diagnostics['du'].function_space().ufl_element() == model.spaces.DG.ufl_element()
    # u_h - Pi u_exact may superconverge, so only check that the diagnostics were filled sensibly.
    assert math.isfinite(l2(diagnostics['u_error'])) and l2(diagnostics['u_exact']) > 1e-2
    assert l2(diagnostics['du']) > 1e-2
