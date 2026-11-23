"""Unit tests for variable coefficients (TODO-elliptic §4.10): parsing, interface, weighted MMS."""
import math
from types import SimpleNamespace

import numpy
import pytest
from firedrake import FacetNormal, SpatialCoordinate, as_matrix, as_vector, assemble, dx, inner, sin

from dimswe.boundary_conditions import get_boundary_registry, measure_for
from dimswe.differential_forms import (apply_weight, boundary_pairing, exterior_derivative, hodge_laplacian_expr,
                                       is_scalar_form, weighted_hodge_fields)
from dimswe.elliptic_coefficients import (EllipticCoefficients, coefficient_is_present, parse_coefficients,
                                          parse_space, tensor_allowed)
from dimswe.elliptic_eigen import constant_coefficient_spectrum, constant_scalar_coefficients, exact_box_eigenvalues
from dimswe.meshes import get_mesh_and_spaces

QDEG = 10


# ---------------------------------------------------------------------------------------------
# Parsing (pure Python)
# ---------------------------------------------------------------------------------------------

def test_presence_and_tensor_rules():
    """mu needs k > 0, kappa needs k < dim; tensors only on vector-proxy forms."""
    assert not coefficient_is_present('mu', 0, 2) and coefficient_is_present('mu', 1, 2)
    assert not coefficient_is_present('kappa', 2, 2) and coefficient_is_present('kappa', 1, 2)
    assert coefficient_is_present('c', 0, 1)
    assert tensor_allowed('kappa', 0, 2) and not tensor_allowed('kappa', 1, 2)   # kappa on 2-forms (scalar)
    assert tensor_allowed('mu', 2, 3) and tensor_allowed('c', 1, 3) and not tensor_allowed('c', 0, 3)


def test_parse_space():
    """Space specs."""
    assert parse_space('DG0') == ('DG', 0) and parse_space('cg1') == ('CG', 1) and parse_space('R') == ('R', 0)
    for bad in ('DG', '0', 'D G1'):
        with pytest.raises(ValueError):
            parse_space(bad)


def test_defaults():
    """Missing coefficients default to mu = kappa = 1, c = 0; absent blocks are skipped."""
    coefficients = parse_coefficients(None, 1, 2)
    assert {name: c.value for name, c in coefficients.items()} == {'mu': 1.0, 'kappa': 1.0, 'c': 0.0}
    assert coefficients['c'].is_default_zero and not coefficients['mu'].treat_as_coeff
    assert set(parse_coefficients({}, 0, 2)) == {'kappa', 'c'}
    assert set(parse_coefficients({}, 2, 2)) == {'mu', 'c'}


def test_parameters_and_bounds():
    """treat_as_coeff constants live in R, fields in DG0 by default; log bounds are transformed."""
    coefficients = parse_coefficients({
        'kappa': {'kind': 'constant', 'value': 2.0, 'treat_as_coeff': True, 'bounds': [0.1, 10.0], 'scaling': 2.0,
                  'log_parametrization': True},
        'mu': {'kind': 'field', 'init': 'problem', 'treat_as_coeff': True},
        'c': {'kind': 'constant', 'value': 0.5},
    }, 1, 2)
    kappa, mu = coefficients['kappa'], coefficients['mu']
    assert kappa.space == 'R' and mu.space == 'DG0'
    assert numpy.allclose(kappa.parameter_bounds(), (math.log(0.1), math.log(10.0)))
    assert mu.parameter_bounds() == (-numpy.inf, numpy.inf)
    assert not coefficients['c'].is_default_zero


@pytest.mark.parametrize("cfg, k, dim", [
    ({'mu': {}}, 0, 2),                                         # mu at k = 0
    ({'kappa': {}}, 2, 2),                                      # kappa at k = n
    ({'nu': {}}, 1, 2),                                         # unknown name
    ({'kappa': {'kind': 'random'}}, 1, 2),
    ({'kappa': {'rank': 'matrix'}}, 1, 2),
    ({'kappa': {'colour': 'red'}}, 1, 2),                       # unknown entry
    ({'kappa': {'rank': 'tensor', 'value': [[1, 0], [0, 1]]}}, 1, 2),  # kappa on 2D 2-forms is scalar
    ({'kappa': {'rank': 'tensor', 'value': [[1, 0], [0, 1]]}}, 0, 3),  # wrong shape (needs 3x3)
    ({'kappa': {'rank': 'tensor', 'value': [[1, 2], [0, 1]]}}, 0, 2),  # not symmetric
    ({'kappa': {'rank': 'tensor', 'value': [[1, 0], [0, -1]]}}, 0, 2),  # not SPD
    ({'kappa': {'value': 0.0}}, 1, 2),                          # must be positive
    ({'c': {'value': -1.0}}, 1, 2),                             # must be non-negative
    ({'kappa': {'kind': 'problem', 'treat_as_coeff': True}}, 1, 2),
    ({'kappa': {'rank': 'tensor', 'value': [[1, 0], [0, 1]], 'treat_as_coeff': True}}, 0, 2),
    ({'kappa': {'log_parametrization': True}}, 1, 2),           # log without treat_as_coeff
    ({'kappa': {'kind': 'field', 'space': 'R'}}, 1, 2),
    ({'kappa': {'init': 'problem'}}, 1, 2),                     # init problem needs a field
    ({'kappa': {'bounds': [2.0, 1.0]}}, 1, 2),
    ({'kappa': {'treat_as_coeff': True, 'log_parametrization': True, 'bounds': [0.0, 1.0]}}, 1, 2),
    ({'kappa': {'scaling': 0.0}}, 1, 2),
])
def test_invalid_coefficients(cfg, k, dim):
    """Every invalid coefficient configuration raises ValueError."""
    with pytest.raises(ValueError):
        parse_coefficients(cfg, k, dim)


def test_constant_coefficient_spectrum():
    """k = 0: kappa lambda + c; k = n: lambda / mu + c; middle k: only kappa = 1/mu; else None."""
    bcs = {'left': 'natural', 'right': 'natural', 'bottom': 'natural', 'top': 'natural'}
    base0 = exact_box_eigenvalues(0, 2, [1.0, 1.0], (), bcs, 5)
    assert numpy.allclose(constant_coefficient_spectrum(0, 2, [1.0, 1.0], (), bcs, 5, kappa=2.0, c=0.5),
                          [2.0 * b + 0.5 for b in base0])
    base2 = exact_box_eigenvalues(2, 2, [1.0, 1.0], (), bcs, 5)
    assert numpy.allclose(constant_coefficient_spectrum(2, 2, [1.0, 1.0], (), bcs, 5, mu=4.0),
                          [b / 4.0 for b in base2])
    assert constant_coefficient_spectrum(1, 2, [1.0, 1.0], (), bcs, 5, mu=2.0, kappa=2.0) is None
    assert constant_coefficient_spectrum(1, 2, [1.0, 1.0], (), bcs, 5, mu=0.5, kappa=2.0) is not None
    coefficients = parse_coefficients({'kappa': {'kind': 'problem'}}, 0, 2)
    assert constant_scalar_coefficients(coefficients) is None
    assert constant_scalar_coefficients(parse_coefficients({'kappa': {'value': 3.0}}, 0, 2)) == \
        {'mu': 1.0, 'kappa': 3.0, 'c': 0.0}


# ---------------------------------------------------------------------------------------------
# Firedrake: weighted fields and the coefficient interface
# ---------------------------------------------------------------------------------------------

COMPLEX_FOR_DIM = {1: [None], 2: ['grad-curl', 'curl-div'], 3: [None]}
ALL_CASES = [(dim, cx, k) for dim in (1, 2, 3) for cx in COMPLEX_FOR_DIM[dim] for k in range(dim + 1)]
MESH_FOR_DIM = {1: 'line', 2: 'rectangle', 3: 'box'}


def make_mesh(dim, n=4, order=2):
    """Bounded Q-family mesh on the unit box."""
    parameters = {'mesh': {'type': MESH_FOR_DIM[dim], 'nx': n, 'ny': n, 'nz': n, 'simplicial_cells': False},
                  'spatial-discretization': {'family': 'Q', 'order': order, 'lump_mass': False}}
    domain = SimpleNamespace(Lx=1.0, Ly=1.0, Lz=1.0, xc=0.5, yc=0.5, zc=0.5)
    return get_mesh_and_spaces(parameters, domain)


def smooth_form(X, j, dim, shift):
    """Smooth j-form proxy."""
    components = [sin(1.3 * X[0] + 0.7 * (i + 1) + shift) * (1.0 + 0.2 * X[min(i, dim - 1)]) for i in range(dim)]
    return components[0] if is_scalar_form(j, dim) else as_vector(components)


def weight(X, j, dim, tensor):
    """Positive scalar or SPD tensor weight acting on j-forms."""
    s = 1.5 + 0.5 * sin(2.0 * X[0])
    if not tensor:
        return s
    entries = [[(s if i == m else 0.0) + (0.3 if i != m else 0.0) for m in range(dim)] for i in range(dim)]
    return as_matrix(entries)


def boundary_integral(integrand, mesh, dim):
    """Integral over the whole boundary."""
    registry = get_boundary_registry(mesh, MESH_FOR_DIM[dim])
    return sum(assemble(integrand * measure_for(face, degree=QDEG)) for face in registry.values())


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: str(c[0]) + "D-" + (c[1] or "") + "-k" + str(c[2]))
def test_weighted_fields_satisfy_weak_form(case):
    """With variable (tensor where allowed) weights, the exact fields satisfy the weighted weak form:

    -(mu sigma, tau) + (u, d tau) - int B_{k-1}(tau, u) = 0
    (d sigma, v) + (kappa du, dv) + (c u, v) - (f, v) - int B_k(v, kappa du) = 0
    """
    dim, cx, k = case
    mesh, _ = make_mesh(dim)
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    mu = weight(X, k - 1, dim, 0 < k - 1 < dim) if k > 0 else None
    kappa = weight(X, k + 1, dim, 0 < k + 1 < dim) if k < dim else None
    c = 0.25 + 0.1 * X[0]
    u = smooth_form(X, k, dim, 0.0)
    fields = weighted_hodge_fields(u, k, dim, cx, mu=mu, kappa=kappa, c=c)
    sigma, flux, f = fields['sigma'], fields['flux'], fields['f']
    scale = max(1.0, assemble(inner(f, f) * dx(degree=QDEG)) ** 0.5)
    if k > 0:
        tau = smooth_form(X, k - 1, dim, 0.4)
        eq1 = assemble((-inner(apply_weight(mu, sigma), tau) + inner(u, exterior_derivative(tau, k - 1, dim, cx)))
                       * dx(degree=QDEG)) - boundary_integral(boundary_pairing(tau, u, k - 1, dim, cx, n), mesh, dim)
        assert abs(eq1) < 1e-8 * scale
    v = smooth_form(X, k, dim, 0.9)
    volume = inner(apply_weight(c, u), v) - inner(f, v)
    if k > 0:
        volume = volume + inner(exterior_derivative(sigma, k - 1, dim, cx), v)
    if k < dim:
        volume = volume + inner(flux, exterior_derivative(v, k, dim, cx))
    eq2 = assemble(volume * dx(degree=QDEG))
    if k < dim:
        eq2 -= boundary_integral(boundary_pairing(v, flux, k, dim, cx, n), mesh, dim)
    assert abs(eq2) < 1e-8 * scale


def test_unweighted_fields_reduce_to_hodge_laplacian():
    """No weights: f = hodge_laplacian_expr and flux = d u."""
    mesh, _ = make_mesh(2)
    X = SpatialCoordinate(mesh)
    u = smooth_form(X, 1, 2, 0.0)
    fields = weighted_hodge_fields(u, 1, 2, 'curl-div')
    diff = fields['f'] - hodge_laplacian_expr(u, 1, 2, 'curl-div')
    assert assemble(inner(diff, diff) * dx(degree=QDEG)) < 1e-20
    assert assemble((fields['flux'] - fields['du']) ** 2 * dx(degree=QDEG)) < 1e-20


class _Problem:
    """Minimal problem data providing coefficient expressions."""

    def coefficient(self, name, mesh, k, dim, complex, rank):
        X = SpatialCoordinate(mesh)
        return 1.0 + 0.5 * X[0] if rank == 'scalar' else as_matrix([[2.0, 0.1], [0.1, 1.0]])


def test_coefficient_interface():
    """Spaces, initial values (log theta), weights, bounds, scaling, flatten/set round trip."""
    mesh, _ = make_mesh(2, n=3, order=1)
    parsed = parse_coefficients({
        'kappa': {'kind': 'constant', 'value': 2.0, 'treat_as_coeff': True, 'bounds': [0.5, 4.0],
                  'log_parametrization': True, 'scaling': 3.0},
        'mu': {'kind': 'field', 'init': 'problem', 'treat_as_coeff': True},
        'c': {'kind': 'problem', 'rank': 'tensor'},
    }, 1, 2)
    coefficients = EllipticCoefficients(parsed, mesh, _Problem(), 1, 2, 'grad-curl')
    assert coefficients.has_coeff() and coefficients.get_coeff_list() == ['mu', 'kappa']
    assert coefficients.get_coeff_size() == 9 + 1      # DG0 on 3x3 cells + one R dof
    theta = coefficients.coeff_sub['kappa'].dat.data_ro.reshape(-1)[0]
    assert math.isclose(theta, math.log(2.0))
    lb, ub = coefficients.get_coeff_bounds()
    assert numpy.allclose(lb[-1], math.log(0.5)) and numpy.allclose(ub[-1], math.log(4.0))
    assert numpy.all(numpy.isinf(lb[:-1]))
    assert numpy.allclose(coefficients.get_coeff_scaling_factors(), [1.0] * 9 + [3.0])
    weights = coefficients.model_weights()
    assert abs(assemble(weights['kappa'] * dx(domain=mesh)) - 2.0) < 1e-12
    reference = coefficients.reference_weights()
    assert reference['c'].ufl_shape == (2, 2)
    array = coefficients.flatten(coefficients.coeff)
    coefficients.set_from_array(array * 0.5)
    assert numpy.allclose(coefficients.flatten(coefficients.coeff), array * 0.5)
    # The reference weights do not move with the parameters.
    assert abs(assemble(reference['kappa'] * dx(domain=mesh)) - 2.0) < 1e-12
    assert coefficients.reaction_is_positive()
