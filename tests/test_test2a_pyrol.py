"""Cheap arbitrary-pytree PyROL adapter and exact derivative certification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip(
    "pyrol",
    reason="PyROL is optional; install rol-python for Test 2A optimizer tests",
)

from pyrol import Problem, Solver
from pyrol.vectors import NumPyVector

from dimswe.rol_adapter import bound_constrained_lbfgs_parameters
from dimswe.test2a_pyrol import (
    CallbackPytreeObjective,
    JAXPytreeObjective,
    build_test2a_lbfgs_parameters,
)


def _parameters():
    return {
        "first": jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        "nested": {"second": jnp.asarray([[0.5]], dtype=jnp.float64)},
    }


def _objective(parameters):
    first = parameters["first"]
    second = parameters["nested"]["second"]
    return jnp.sum((first - jnp.asarray([0.25, -0.75])) ** 2) + 2.0 * jnp.sum(
        (second - 1.5) ** 2
    )


def test_pytree_vector_roundtrip_and_callback_derivatives():
    parameters = _parameters()
    adapter = JAXPytreeObjective(_objective, parameters, use_jit=False)
    control = adapter.vector_from_pytree(parameters)
    restored = adapter.pytree_from_vector(control)
    assert adapter.dimension == 3
    for expected, actual in zip(
        jax.tree_util.tree_leaves(parameters), jax.tree_util.tree_leaves(restored)
    ):
        np.testing.assert_array_equal(actual, expected)

    gradient = NumPyVector(np.empty(3, dtype=np.float64))
    adapter.gradient(gradient, control, 0.0)
    direction = NumPyVector(np.asarray([0.3, -0.4, 0.7], dtype=np.float64))
    hvp = NumPyVector(np.empty(3, dtype=np.float64))
    adapter.hessVec(hvp, direction, control, 0.0)
    step = 1.0e-6
    plus = NumPyVector(control.array + step * direction.array)
    minus = NumPyVector(control.array - step * direction.array)
    plus_gradient = NumPyVector(np.empty(3, dtype=np.float64))
    minus_gradient = NumPyVector(np.empty(3, dtype=np.float64))
    adapter.gradient(plus_gradient, plus, 0.0)
    adapter.gradient(minus_gradient, minus, 0.0)
    centered = (plus_gradient.array - minus_gradient.array) / (2.0 * step)
    np.testing.assert_allclose(centered, hvp.array, rtol=1.0e-9, atol=1.0e-9)
    assert adapter.hvp_evaluations == 1


def test_callbacks_accept_uninitialized_output_storage_and_fill_it_finitely():
    adapter = JAXPytreeObjective(_objective, _parameters(), use_jit=False)
    control = adapter.vector_from_pytree(_parameters())
    gradient = NumPyVector(np.full(3, np.nan, dtype=np.float64))
    adapter.gradient(gradient, control, 0.0)
    assert np.isfinite(gradient.array).all()
    direction = NumPyVector(np.ones(3, dtype=np.float64))
    action = NumPyVector(np.full(3, np.nan, dtype=np.float64))
    adapter.hessVec(action, direction, control, 0.0)
    assert np.isfinite(action.array).all()


def test_rol_line_search_lbfgs_optimizes_arbitrary_pytree_with_exact_gradient():
    adapter = JAXPytreeObjective(_objective, _parameters(), use_jit=True)
    control = adapter.vector_from_pytree(_parameters())
    initial = adapter.value(control, 0.0)
    problem = Problem(adapter, control)
    solver = Solver(
        problem,
        bound_constrained_lbfgs_parameters(
            gradient_tolerance=1.0e-11,
            step_tolerance=1.0e-14,
            iteration_limit=30,
        ),
    )
    solver.solve()
    final = adapter.value(control, 0.0)
    restored = adapter.pytree_from_vector(control)
    assert final < 1.0e-20 * initial
    np.testing.assert_allclose(restored["first"], [0.25, -0.75], atol=1.0e-10)
    np.testing.assert_allclose(restored["nested"]["second"], [[1.5]], atol=1.0e-10)
    assert adapter.gradient_evaluations > 0
    assert adapter.hvp_evaluations == 0


def test_adapter_rejects_wrong_vector_dimension_and_pytree_structure():
    adapter = JAXPytreeObjective(_objective, _parameters(), use_jit=False)
    with pytest.raises(ValueError, match="dimension"):
        adapter.value(NumPyVector(np.zeros(2, dtype=np.float64)), 0.0)
    with pytest.raises(ValueError, match="structure"):
        adapter.vector_from_pytree({"different": jnp.zeros(3, dtype=jnp.float64)})
    with pytest.raises(ValueError, match="leaf shapes"):
        adapter.vector_from_pytree(
            {
                "first": jnp.zeros((1, 2), dtype=jnp.float64),
                "nested": {"second": jnp.zeros((1,), dtype=jnp.float64)},
            }
        )


def test_selected_lbfgs_policy_sets_memory_explicitly():
    parameters = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": 1.0e-8,
            "step_tolerance": 1.0e-12,
            "iteration_limit": 100,
            "maximum_secant_storage": 10,
        }
    )
    storage = parameters.sublist("General").sublist("Secant").get(
        "Maximum Storage"
    )
    assert storage == 10


def test_external_callback_adapter_reuses_identical_pytree_vector_convention():
    parameters = _parameters()
    exact_gradient = jax.grad(_objective)
    adapter = CallbackPytreeObjective(_objective, exact_gradient, parameters)
    control = adapter.vector_from_pytree(parameters)
    output = NumPyVector(np.full(adapter.dimension, np.nan, dtype=np.float64))
    adapter.gradient(output, control, 0.0)
    expected = adapter.vector_from_pytree(exact_gradient(parameters))
    np.testing.assert_array_equal(output.array, expected.array)
    assert adapter.value(control, 0.0) == pytest.approx(float(_objective(parameters)))
    assert adapter.gradient_evaluations == 1
    assert adapter.hvp_evaluations == 0
