"""Pure-Python tests of ReducedProblem Hessian-vector products.

A quadratic full-space objective f(x) = 0.5 x.A x + b.x with known Hessian A is wrapped in a
ReducedProblem, so the reduced Hessian must be exactly S P^T A P S (S the free scaling, P the
scatter of free entries). No model runs are involved.
"""

import numpy as np
import pytest

from dimswe.optimize import ReducedProblem

N = 6


def quadratic(seed=0):
    """Return (A, b, fun, jac, hessp) for a seeded symmetric quadratic objective."""
    generator = np.random.default_rng(seed)
    M = generator.standard_normal((N, N))
    A = M @ M.T + N * np.eye(N)
    b = generator.standard_normal(N)
    fun = lambda x: 0.5 * x @ A @ x + b @ x
    jac = lambda x: A @ x + b
    hessp = lambda x, v: A @ v
    return A, b, fun, jac, hessp


def make_problem(free_mask=None, scaling_factors=None, with_hessp=True):
    """Return (problem, A) for the quadratic objective with optional mask and scaling."""
    A, _, fun, jac, hessp = quadratic()
    x0 = np.linspace(1.0, 2.0, N)
    problem = ReducedProblem(fun, jac, x0, free_mask=free_mask, scaling_factors=scaling_factors,
        full_hessp=hessp if with_hessp else None)
    return problem, A


def test_hessp_matches_scaled_reduced_hessian():
    free_mask = np.array([True, False, True, True, False, True])
    scale = np.array([2.0, 7.0, 0.5, 3.0, 11.0, 1e-3])
    problem, A = make_problem(free_mask=free_mask, scaling_factors=scale)
    S = np.diag(scale[free_mask])
    expected_hessian = S @ A[np.ix_(free_mask, free_mask)] @ S
    p = np.random.default_rng(1).standard_normal(free_mask.sum())
    assert np.allclose(problem.hessp(problem.z0, p), expected_hessian @ p)


def test_hessp_matches_reduced_jacobian_differences():
    # for a quadratic, jac(z + p) - jac(z) = H_z p exactly, including scaling and free_mask
    free_mask = np.array([True, True, False, True, True, True])
    scale = np.array([0.1, 4.0, 1.0, 2.5, 1.0, 9.0])
    problem, _ = make_problem(free_mask=free_mask, scaling_factors=scale)
    p = np.random.default_rng(2).standard_normal(free_mask.sum())
    z = problem.z0 + 0.3
    difference = problem.jac(z + p) - problem.jac(z)
    assert np.allclose(problem.hessp(z, p), difference)


def test_hessp_direction_is_zero_on_fixed_entries():
    free_mask = np.array([False, True, True, False, True, False])
    seen = []

    def full_hessp(x, v):
        seen.append(v.copy())
        return v

    problem = ReducedProblem(lambda x: 0.0, lambda x: np.zeros(N), np.ones(N), free_mask=free_mask,
        scaling_factors=np.full(N, 2.0), full_hessp=full_hessp)
    problem.hessp(problem.z0, np.ones(free_mask.sum()))
    assert np.all(seen[0][~free_mask] == 0)
    assert np.allclose(seen[0][free_mask], 2.0)


def test_hessp_is_evaluated_at_the_full_point():
    free_mask = np.array([True, False, True, False, True, True])
    scale = np.array([3.0, 1.0, 0.25, 1.0, 5.0, 2.0])
    points = []

    def full_hessp(x, v):
        points.append(x.copy())
        return np.zeros(N)

    problem = ReducedProblem(lambda x: 0.0, lambda x: np.zeros(N), np.arange(1.0, N + 1), free_mask=free_mask,
        scaling_factors=scale, full_hessp=full_hessp)
    z = problem.z0 + 1.0
    problem.hessp(z, np.ones(free_mask.sum()))
    assert np.allclose(points[0], problem.to_full(z))


def test_hessp_counts_separately_from_fun_and_jac():
    problem, _ = make_problem()
    p = np.ones(N)
    problem.fun(problem.z0)
    problem.jac(problem.z0)
    for _ in range(3):
        problem.hessp(problem.z0, p)
    # HVPs are not memoized: every call is a new model evaluation
    assert problem.nhvp == 3
    assert problem.nfev == 1
    assert problem.njev == 1


def test_hessp_unavailable_raises():
    problem, _ = make_problem(with_hessp=False)
    assert not problem.has_hessp()
    with pytest.raises(ValueError, match='no Hessian-vector product'):
        problem.hessp(problem.z0, np.ones(N))
    assert problem.nhvp == 0


def test_hessp_rejects_wrong_full_shape():
    problem = ReducedProblem(lambda x: 0.0, lambda x: np.zeros(N), np.ones(N), full_hessp=lambda x, v: np.zeros(N - 1))
    with pytest.raises(ValueError, match='full_hessp returned shape'):
        problem.hessp(problem.z0, np.ones(N))
