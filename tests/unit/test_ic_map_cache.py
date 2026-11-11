"""Tests for the point cache of ParametricInitialCondition (docs/hvp_performance_plan.md).

The initial state, the tangents dx0/dp and the second tangents d^2 x0/dp^2 depend only on the
parameter values. The map caches them for the most recent point, so a gradient followed by HVPs
at that point projects each of them once. The cached values must be exactly what a fresh
projection gives, callers must not be able to corrupt them through the returned arrays, and a new
point must invalidate them.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from tests.gradient_helpers import setup_problem  # noqa: E402

PARAM_NAMES = ["ox", "oy"]
SCALING = np.array([0.1, 0.1])


@pytest.fixture(scope="module")
def setup():
    return setup_problem(3, obs_steps=[1, 3])


def make_problem(setup, cache):
    """Return (ic_map, ReducedProblem, full starting point) for 'ic-params' with or without the cache."""
    from dimswe.optimize import ParametricInitialCondition

    ic_map = ParametricInitialCondition(setup.model, setup.parameters, PARAM_NAMES, cache=cache)
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
    problem = setup.optimizer.reduced_problem(params, opt_type="ic-params", coeffs0=setup.coeff, ic_map=ic_map,
                                              scaling_factors=SCALING)
    return ic_map, problem, problem.to_full(problem.z0)


def count_projections(ic_map):
    """Wrap the map's projection methods so their calls are counted; return the counter dict."""
    counts = {"x0": 0, "tangents": 0, "second_tangents": 0}
    for key, name in (("x0", "_project_x0"), ("tangents", "_project_tangents"),
                      ("second_tangents", "_project_second_tangents")):
        original = getattr(ic_map, name)

        def counted(_key=key, _original=original):
            counts[_key] += 1
            return _original()
        setattr(ic_map, name, counted)
    return counts


def directions(seed):
    """Return two seeded parameter-space directions."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(2), rng.standard_normal(2)


def test_gradient_and_hvps_at_one_point_project_each_quantity_once(setup):
    ic_map, problem, x = make_problem(setup, cache=True)
    counts = count_projections(ic_map)
    d1, d2 = directions(11)
    problem.full_fun(x)
    problem.full_jac(x)
    problem.full_hessp(x, d1)
    problem.full_hessp(x, d2)
    problem.full_jac(x)
    assert counts == {"x0": 1, "tangents": 1, "second_tangents": 1}
    # a new point invalidates everything, and the old point is recomputed when it comes back
    problem.full_jac(x + np.array([1e-3, 0.0]))
    problem.full_jac(x)
    assert counts == {"x0": 3, "tangents": 3, "second_tangents": 1}


def test_cached_results_equal_uncached_ones(setup):
    _, cached_problem, x = make_problem(setup, cache=True)
    _, plain_problem, _ = make_problem(setup, cache=False)
    d1, d2 = directions(12)
    for problem_x in (x, x + np.array([2e-3, -1e-3])):
        np.testing.assert_allclose(cached_problem.full_fun(problem_x), plain_problem.full_fun(problem_x),
                                   rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(cached_problem.full_jac(problem_x), plain_problem.full_jac(problem_x),
                                   rtol=1e-12, atol=0.0)
        for direction in (d1, d2):
            np.testing.assert_allclose(cached_problem.full_hessp(problem_x, direction),
                                       plain_problem.full_hessp(problem_x, direction), rtol=1e-12, atol=0.0)


def test_returned_arrays_are_copies_of_the_cache(setup):
    ic_map, _, x = make_problem(setup, cache=True)
    x0 = ic_map.forward(x)
    tangents = ic_map.tangents()
    second = ic_map.second_tangents()
    expected = (x0.copy(), [t.copy() for t in tangents],
                [[None if a is None else a.copy() for a in row] for row in second])
    x0 += 1.0
    for tangent in tangents:
        tangent *= -3.0
    for row in second:
        for arr in row:
            if arr is not None:
                arr += 5.0
    np.testing.assert_array_equal(ic_map.forward(x), expected[0])
    for tangent, original in zip(ic_map.tangents(), expected[1]):
        np.testing.assert_array_equal(tangent, original)
    for row, original_row in zip(ic_map.second_tangents(), expected[2]):
        for arr, original in zip(row, original_row):
            assert (arr is None) == (original is None)
            if arr is not None:
                np.testing.assert_array_equal(arr, original)


def test_clear_cache_and_disabled_cache_recompute(setup):
    ic_map, _, x = make_problem(setup, cache=True)
    counts = count_projections(ic_map)
    ic_map.forward(x)
    ic_map.tangents()
    ic_map.clear_cache()
    ic_map.forward(x)
    ic_map.tangents()
    assert counts["x0"] == 2 and counts["tangents"] == 2

    plain_map, _, _ = make_problem(setup, cache=False)
    plain_counts = count_projections(plain_map)
    for _ in range(2):
        plain_map.forward(x)
        plain_map.tangents()
    assert plain_counts["x0"] == 2 and plain_counts["tangents"] == 2
