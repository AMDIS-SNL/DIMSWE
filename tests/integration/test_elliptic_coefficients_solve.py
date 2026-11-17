"""Variable coefficients in the elliptic model and eigenproblem (TODO-elliptic §4.10, §6).

MMS convergence with variable/tensor coefficients, c > 0 switching off the harmonic constraint,
Taylor tests of the coefficient gradient (scalar R and DG0 fields; k = 0, k = n, middle k; both
harmonic treatments), steady-state optimization (recovering kappa and a DG0 mu field), and
weighted eigenproblems (closed-form spectra for constants, min-max bounds for variable weights).
"""
import math

import numpy
import pytest
from firedrake import Function, assemble, dx, inner

from dimswe.elliptic import EllipticProblem
from dimswe.elliptic_eigen import HodgeLaplacianEigenproblem
from dimswe.optimize import SteadyStateConstrainedOptimization, taylor_test
from tests.conftest import make_elliptic_parameters


def coefficient_parameters(dim, cx, k, bcs, coefficients, n=6, order=2, problem='variable-coefficient-sine',
                           **kwargs):
    """Elliptic parameters with an elliptic.coefficients section."""
    parameters = make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order, problem=problem, **kwargs)
    parameters['elliptic']['coefficients'] = coefficients
    return parameters


def run(parameters, mesh=None):
    """Solve and compute statistics."""
    model = EllipticProblem(parameters, mesh=mesh)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    return model


# ---------------------------------------------------------------------------------------------
# MMS with variable coefficients
# ---------------------------------------------------------------------------------------------

def variable_coefficients(dim, k):
    """kind: problem coefficients for every present block; tensors where allowed, c > 0 included."""
    def entry(degree):
        return {'kind': 'problem', 'rank': 'tensor' if 0 < degree < dim else 'scalar'}
    coefficients = {'c': entry(k)}
    if k > 0:
        coefficients['mu'] = entry(k - 1)
    if k < dim:
        coefficients['kappa'] = entry(k + 1)
    return coefficients


MMS_CASES = [(1, None, 0), (1, None, 1), (2, 'grad-curl', 0), (2, 'grad-curl', 1), (2, 'curl-div', 1),
             (2, 'curl-div', 2), (3, None, 1), (3, None, 2)]


@pytest.mark.parametrize("dim, cx, k", [pytest.param(*c, marks=[pytest.mark.slow] if c[0] == 3 else [])
                                        for c in MMS_CASES])
@pytest.mark.parametrize("bcs", ['essential', 'natural', 'mixed'])
def test_variable_coefficient_mms(dim, cx, k, bcs):
    """Rates with smooth anisotropic coefficients match the constant-coefficient rates."""
    order = 1 if dim == 3 else 2
    # d sigma involves third derivatives of u and derivatives of 1/mu, so it needs finer meshes to
    # reach the asymptotic range than the constant-coefficient tests ((6, 12) gave 1.6 for 2D
    # curl-div k = 1).
    ns = (3, 6) if dim == 3 else (12, 24)
    stats = [run(coefficient_parameters(dim, cx, k, bcs, variable_coefficients(dim, k), n=n, order=order))
             .get_statistics() for n in ns]
    expected = {'u_error': order + 1 if k == 0 else order}
    if k > 0:
        expected.update({'sigma_error': order, 'dsigma_error': order})
    if k < dim:
        expected['du_error'] = order
    errors = {name: (stats[0][name][0], stats[1][name][0]) for name in expected}
    for name, rate in expected.items():
        e0, e1 = errors[name]
        if e1 < 1e-11:
            continue
        observed = math.log(e0 / e1) / math.log(ns[1] / ns[0])
        assert observed >= rate - 0.3, (name, observed, rate, errors)


def test_positive_reaction_switches_off_harmonic_constraint():
    """Fully periodic k = 1 has 2 harmonic forms, but with c > 0 there are none and no p block."""
    parameters = coefficient_parameters(2, 'grad-curl', 1, 'periodic', {'c': {'value': 0.5}}, problem='sine')
    model = run(parameters)
    assert model.num_harmonic == 0 and model.get_x_var_list() == ['sigma', 'u']
    plain = run(make_elliptic_parameters(2, 'grad-curl', 1, 'periodic', n=6))
    assert plain.num_harmonic == 2
    # Still an MMS solution with the reaction term in the source.
    assert model.get_statistics()['u_error'][0] < 2.0 * plain.get_statistics()['u_error'][0]


# ---------------------------------------------------------------------------------------------
# Coefficient gradients (Taylor tests)
# ---------------------------------------------------------------------------------------------

def make_optimizer(parameters, truth=None, fields=('u',)):
    """Optimizer whose data come from the same forward map at the true parameters.

    truth: callable model -> flattened true parameter array (default: the initial parameters).
    The problem data (source, boundary data) are fixed by the cfg's initial coefficient values
    (reference weights), so only the coefficient Function differs between truth and model; the
    truth fields are copied into plain Functions and the parameters are reset afterwards.
    """
    model = EllipticProblem(parameters)
    x0 = model.get_numpy_coeff().copy()
    model.set_numpy_coeff(x0 if truth is None else truth(model))
    model.solve()
    spaces = {'u': model.Vk, 'sigma': model.Vsigma}
    data = {}
    for name in fields:
        data[name] = Function(spaces[name], name=name + '_data')
        data[name].dat.data[:] = model.sub[name].dat.data_ro
    model.set_numpy_coeff(x0)
    optimizer = SteadyStateConstrainedOptimization(model, data, fields=fields)
    optimizer.verbose = False
    return optimizer, model


def assert_second_order(rates):
    """Leading Taylor rates close to 2."""
    finite = rates[numpy.isfinite(rates)]
    assert finite.size >= 3 and numpy.allclose(finite[:3], 2.0, atol=0.2), rates


GRADIENT_CASES = [
    # (dim, complex, k, bcs, harmonic_constraint, coefficients)
    (2, 'grad-curl', 0, 'natural', 'lagrange',
     {'kappa': {'value': 1.5, 'treat_as_coeff': True}, 'c': {'value': 0.0}}),
    (2, 'curl-div', 2, 'essential', 'nullspace',
     {'mu': {'kind': 'field', 'space': 'DG0', 'init': 'problem', 'treat_as_coeff': True}}),
    (2, 'grad-curl', 1, 'mixed', 'lagrange',
     {'mu': {'value': 0.8, 'treat_as_coeff': True, 'log_parametrization': True},
      'kappa': {'kind': 'field', 'space': 'DG0', 'init': 'problem', 'treat_as_coeff': True}}),
    (2, 'curl-div', 1, 'periodic', 'nullspace',
     {'kappa': {'value': 1.2, 'treat_as_coeff': True}, 'mu': {'value': 0.9, 'treat_as_coeff': True}}),
    (2, 'curl-div', 1, 'periodic', 'lagrange',
     {'kappa': {'value': 1.2, 'treat_as_coeff': True}, 'mu': {'value': 0.9, 'treat_as_coeff': True}}),
]


@pytest.mark.parametrize("dim, cx, k, bcs, constraint, coefficients", GRADIENT_CASES)
def test_coefficient_gradient_taylor(dim, cx, k, bcs, constraint, coefficients):
    """Adjoint coefficient gradient: Taylor remainder of second order (R and DG0 parameters)."""
    parameters = coefficient_parameters(dim, cx, k, bcs, coefficients, n=4, harmonic_constraint=constraint)
    optimizer, model = make_optimizer(parameters)
    x0 = model.get_numpy_coeff().copy()
    # Move away from the truth (where the gradient vanishes) and test along a random direction.
    rng = numpy.random.default_rng(7)
    start = x0 * (1.0 + 0.2 * rng.uniform(0.5, 1.0, size=x0.size)) + 0.05
    problem = optimizer.reduced_problem(start, use_model_bounds=False, scaling_factors=numpy.ones(x0.size))
    direction = rng.standard_normal(x0.size)
    _, remainders, rates = taylor_test(problem, direction)
    assert numpy.max(remainders) > 0.0
    assert_second_order(rates)


# ---------------------------------------------------------------------------------------------
# Steady-state optimization
# ---------------------------------------------------------------------------------------------

def test_recover_scalar_kappa():
    """Recover kappa = 2 from truth data, starting from kappa = 1 (log parametrization, bounds)."""
    coefficients = {'kappa': {'value': 1.0, 'treat_as_coeff': True, 'log_parametrization': True,
                              'bounds': [0.1, 10.0]}}
    parameters = coefficient_parameters(2, 'grad-curl', 0, 'essential', coefficients, n=6, problem='sine')
    optimizer, model = make_optimizer(parameters, truth=lambda m: numpy.array([math.log(2.0)]))
    x_opt = optimizer.optimize(gtol=1e-10, ftol=1e-14, maxiter=100)
    assert abs(math.exp(x_opt[0]) - 2.0) < 1e-3
    assert optimizer.obj(x_opt) < 1e-10


def test_recover_mu_field():
    """Recover a smooth DG0 mu field (k = n, mixed Darcy) from sigma and u data, with a small
    Tikhonov term; the loss decreases and the recovered field is close to the truth."""
    coefficients = {'mu': {'kind': 'field', 'space': 'DG0', 'value': 1.0, 'treat_as_coeff': True,
                           'log_parametrization': True}}
    parameters = coefficient_parameters(2, 'curl-div', 2, 'essential', coefficients, n=4)

    def mu_true_values(model):
        mu = Function(model.coefficients.coeff.function_space()).interpolate(
            model.problem.coefficient('mu', model.mesh, 2, 2, 'curl-div', 'scalar'))
        return numpy.log(mu.dat.data_ro.copy())

    optimizer, model = make_optimizer(parameters, truth=mu_true_values, fields=('sigma', 'u'))
    optimizer.regularization = 1e-8
    x0 = model.get_numpy_coeff().copy()
    loss0 = optimizer.obj(x0)
    x_opt = optimizer.optimize(gtol=1e-12, ftol=1e-16, maxiter=300)
    assert optimizer.obj(x_opt) < 1e-3 * loss0
    mu_true = Function(model.coefficients.coeff.function_space()).interpolate(
        model.problem.coefficient('mu', model.mesh, 2, 2, 'curl-div', 'scalar'))
    recovered = numpy.exp(x_opt)
    relative = numpy.max(numpy.abs(recovered - mu_true.dat.data_ro) / mu_true.dat.data_ro)
    assert relative < 0.05, relative


# ---------------------------------------------------------------------------------------------
# Weighted eigenproblems
# ---------------------------------------------------------------------------------------------

def eigen_parameters(dim, cx, k, bcs, coefficients, n=10, order=2, nev=6):
    """elliptic-eigen parameters with coefficients (the problem supplies kind: problem coefficients)."""
    parameters = coefficient_parameters(dim, cx, k, bcs, coefficients, n=n, order=order,
                                        problem='variable-coefficient-sine')
    parameters['model']['type'] = 'elliptic-eigen'
    parameters['elliptic']['eigen'] = {'nev': nev, 'target': -1.0, 'tolerance': 1e-10}
    return parameters


def solve_eigen(parameters):
    """Eigenproblem solve with statistics."""
    model = HodgeLaplacianEigenproblem(parameters)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    return model


@pytest.mark.parametrize("cx, k, bcs, coefficients", [
    ('grad-curl', 0, 'natural', {'kappa': {'value': 2.0}, 'c': {'value': 0.5}}),
    ('curl-div', 2, 'essential', {'mu': {'value': 4.0}}),
    ('grad-curl', 1, 'periodic', {'mu': {'value': 0.5}, 'kappa': {'value': 2.0}}),
    ('curl-div', 1, 'mixed', {'mu': {'value': 0.5}, 'kappa': {'value': 2.0}, 'c': {'value': 1.0}}),
])
def test_constant_coefficient_eigenvalues(cx, k, bcs, coefficients):
    """Constant coefficients: computed eigenvalues match the closed-form scaled spectrum."""
    model = solve_eigen(eigen_parameters(2, cx, k, bcs, coefficients))
    stats = model.get_statistics()
    assert 'exact_eigenvalues' in stats
    assert numpy.max(stats['eigenvalue_errors']) < 5e-2, (stats['eigenvalues'], stats['exact_eigenvalues'])
    if 'c' in coefficients and coefficients['c']['value'] > 0:
        assert model.num_harmonic == 0 and stats['num_zero_eigenvalues'][0] == 0


@pytest.mark.parametrize("cx, k, bcs", [('grad-curl', 0, 'essential'), ('curl-div', 1, 'natural'),
                                         ('grad-curl', 2, 'natural')])
def test_variable_coefficient_eigenvalue_bounds(cx, k, bcs):
    """Variable coefficients between bounds: by min-max, every eigenvalue lies between those of the
    constant-coefficient problems with the extreme weights.

    Eliminating sigma, the operator is D M_mu^{-1} D^T + K_kappa, which grows (in the operator
    order) when kappa grows or mu shrinks; so the lower bound uses (kappa_min, mu_max) and the
    upper bound (kappa_max, mu_min), also when both vary (middle k).
    """
    # Variable mu (k > 0) and kappa (k < n), as kind: problem scalars.
    variable = {}
    if k > 0:
        variable['mu'] = {'kind': 'problem'}
    if k < 2:
        variable['kappa'] = {'kind': 'problem'}
    model = solve_eigen(eigen_parameters(2, cx, k, bcs, variable))
    assert 'exact_eigenvalues' not in model.get_statistics()
    computed = numpy.array(model.eigenvalues)
    # VariableCoefficientSineProblem scalar weights: SCALE * (1 + 0.5 * prod sin) in [0.5, 1.5] * SCALE.
    lo, hi = 0.5, 1.5
    kappa_scale, mu_scale = 1.5, 1.0
    low_coeffs, high_coeffs = {}, {}
    if 'kappa' in variable:
        low_coeffs['kappa'], high_coeffs['kappa'] = {'value': lo * kappa_scale}, {'value': hi * kappa_scale}
    if 'mu' in variable:
        low_coeffs['mu'], high_coeffs['mu'] = {'value': hi * mu_scale}, {'value': lo * mu_scale}
    low = solve_eigen(eigen_parameters(2, cx, k, bcs, low_coeffs)).eigenvalues
    high = solve_eigen(eigen_parameters(2, cx, k, bcs, high_coeffs)).eigenvalues
    tol = 1e-8
    assert numpy.all(computed >= numpy.array(low) - tol * (1 + numpy.abs(low))), (computed, low)
    assert numpy.all(computed <= numpy.array(high) + tol * (1 + numpy.abs(high))), (computed, high)
