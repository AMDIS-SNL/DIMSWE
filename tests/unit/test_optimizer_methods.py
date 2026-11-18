"""Pure-Python tests of optimizer-method selection and the HVP-enabled optimize() path.

resolve_hessp_use and scipy_options are tested directly. optimize() is exercised on a quadratic
ReducedProblem with known minimizer, by calling it with a minimal stand-in for self: with a
prebuilt problem, optimize() only reads self.verbose. No model runs are involved.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from dimswe.optimize import (
    HESSP_METHODS,
    ReducedProblem,
    _ODEConstrainedOptimization,
    method_supports_bounds,
    method_uses_hessp,
    resolve_hessp_use,
    scipy_options,
)

N = 5


def quadratic_problem(bounds=None, with_hessp=True, scaling_factors=None):
    """Return (problem, x_star) for f(x) = 0.5 x.A x + b.x with seeded SPD A."""
    generator = np.random.default_rng(0)
    M = generator.standard_normal((N, N))
    A = M @ M.T + N * np.eye(N)
    b = generator.standard_normal(N)
    x_star = np.linalg.solve(A, -b)
    problem = ReducedProblem(lambda x: 0.5 * x @ A @ x + b @ x, lambda x: A @ x + b, np.zeros(N),
        scaling_factors=scaling_factors, bounds=bounds, full_hessp=(lambda x, v: A @ v) if with_hessp else None)
    return problem, x_star


def run_optimize(problem, method, **kwargs):
    """Call _ODEConstrainedOptimization.optimize with a prebuilt problem and a stand-in self.

    Both gtol and ftol are tight: optimize() defaults to ftol=1e-5, which stops L-BFGS-B on its
    relative-reduction test well before the minimizer is reached to the asserted accuracy.
    """
    stand_in = SimpleNamespace(verbose=False)
    return _ODEConstrainedOptimization.optimize(stand_in, None, method=method, problem=problem,
        return_result=True, gtol=1e-10, ftol=1e-15, maxiter=200, **kwargs)


def test_method_capabilities():
    for method in ('Newton-CG', 'trust-ncg', 'trust-krylov', 'trust-constr'):
        assert method_uses_hessp(method)
    for method in ('L-BFGS-B', 'TNC', 'CG', 'BFGS', 'SLSQP'):
        assert not method_uses_hessp(method)
    assert method_supports_bounds('L-BFGS-B') and method_supports_bounds('trust-constr')
    assert not method_supports_bounds('Newton-CG')
    assert set(HESSP_METHODS) == {'NEWTON-CG', 'TRUST-NCG', 'TRUST-KRYLOV', 'TRUST-CONSTR'}


@pytest.mark.parametrize('method', ['Newton-CG', 'trust-ncg', 'trust-krylov', 'trust-constr'])
def test_resolve_auto_uses_hessp_for_second_order_methods(method):
    assert resolve_hessp_use(method, None, True, has_hessp=True, has_bounds=False) is True


@pytest.mark.parametrize('method', ['L-BFGS-B', 'CG', 'BFGS', 'TNC'])
def test_resolve_auto_leaves_first_order_methods_unchanged(method):
    assert resolve_hessp_use(method, None, True, has_hessp=True, has_bounds=False) is False


def test_resolve_first_order_methods_keep_ignoring_bounds():
    # scipy warns about and ignores bounds for CG/BFGS; this is unchanged for compatibility
    assert resolve_hessp_use('CG', None, True, has_hessp=True, has_bounds=True) is False


def test_resolve_newton_cg_without_hessp_falls_back():
    # Newton-CG can use finite differences of the gradient instead
    assert resolve_hessp_use('Newton-CG', False, True, has_hessp=True, has_bounds=False) is False
    assert resolve_hessp_use('Newton-CG', None, True, has_hessp=False, has_bounds=False) is False


@pytest.mark.parametrize('args, message', [
    (('L-BFGS-B', True, True, True, False), 'does not use Hessian-vector products'),
    (('Newton-CG', True, True, False, False), 'does not provide them'),
    (('Newton-CG', None, False, True, False), 'needs the gradient'),
    (('trust-ncg', False, True, True, False), 'requires Hessian-vector products'),
    (('trust-krylov', None, True, False, False), 'requires Hessian-vector products'),
    (('Newton-CG', None, True, True, True), 'cannot handle bounds'),
    (('trust-ncg', None, True, True, True), 'cannot handle bounds'),
])
def test_resolve_rejects_invalid_combinations(args, message):
    with pytest.raises(ValueError, match=message):
        resolve_hessp_use(*args)


def test_resolve_trust_constr_accepts_bounds():
    assert resolve_hessp_use('trust-constr', None, True, has_hessp=True, has_bounds=True) is True


def test_scipy_options_for_second_order_methods():
    assert scipy_options('Newton-CG', 10, 1e-6, 1e-7, False) == {'disp': False, 'maxiter': 10}
    assert scipy_options('Newton-CG', 10, 1e-6, 1e-7, False, xtol=1e-9)['xtol'] == 1e-9
    assert scipy_options('trust-ncg', 10, 1e-6, 1e-7, False) == {'disp': False, 'maxiter': 10, 'gtol': 1e-6}
    assert scipy_options('trust-krylov', 10, 1e-6, 1e-7, False)['gtol'] == 1e-6
    assert scipy_options('trust-constr', 10, 1e-6, 1e-7, False, xtol=1e-9) == {'disp': False, 'maxiter': 10, 'gtol': 1e-6, 'xtol': 1e-9}
    # first-order options are unchanged
    assert scipy_options('L-BFGS-B', 10, 1e-6, 1e-7, False) == {'disp': False, 'maxiter': 10, 'maxfun': 10, 'gtol': 1e-6, 'ftol': 1e-7}


@pytest.mark.parametrize('method', ['Newton-CG', 'trust-ncg', 'trust-krylov', 'trust-constr'])
def test_second_order_methods_converge_and_count_hvps(method):
    problem, x_star = quadratic_problem(scaling_factors=np.linspace(0.5, 2.0, N))
    x_opt, res = run_optimize(problem, method, xtol=1e-12 if method in ('Newton-CG', 'trust-constr') else None)
    assert np.allclose(x_opt, x_star, atol=1e-6)
    assert problem.nhvp > 0
    assert res.problem_nhvp == problem.nhvp
    assert res.problem_njev == problem.njev and res.problem_nfev == problem.nfev
    assert res.hessp_used is True


def test_trust_constr_with_bounds_and_callback():
    problem, x_star = quadratic_problem(bounds=(np.full(N, -0.05), np.full(N, 0.05)))
    calls = []
    x_opt, _ = run_optimize(problem, 'trust-constr', callback=lambda it, x, f, g: calls.append(it))
    assert np.all(x_opt >= -0.05 - 1e-8) and np.all(x_opt <= 0.05 + 1e-8)
    assert problem.nhvp > 0
    # the initial point is reported as iteration 0, then every trust-constr iteration
    assert calls[0] == 0 and len(calls) > 1


def test_first_order_method_does_not_call_hessp():
    problem, x_star = quadratic_problem()
    x_opt, res = run_optimize(problem, 'L-BFGS-B')
    assert np.allclose(x_opt, x_star, atol=1e-5)
    assert problem.nhvp == 0 and res.problem_nhvp == 0
    assert res.hessp_used is False


def test_newton_cg_with_bounds_is_rejected_unless_disabled():
    problem, x_star = quadratic_problem(bounds=(np.full(N, -10.0), np.full(N, 10.0)))
    with pytest.raises(ValueError, match='cannot handle bounds'):
        run_optimize(problem, 'Newton-CG')
    x_opt, _ = run_optimize(problem, 'Newton-CG', use_bounds=False, xtol=1e-12)
    assert np.allclose(x_opt, x_star, atol=1e-6)
