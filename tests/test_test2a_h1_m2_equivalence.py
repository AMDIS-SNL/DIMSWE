"""Cheap algebraic tests for the Test-2A H=1/Method-2 audit."""

import jax
import jax.numpy as jnp
import numpy as np

from dimswe.test2a_discrete_offline import (
    DiscreteOfflineObservation,
    DiscretePredictionCache,
)
from dimswe.test2a_h1_m2_equivalence import (
    WeightedFixedStateObjective,
    h1_structural_source_error,
    h1_tendency_loss_coefficient,
    parameter_gradient_relation,
)
from dimswe.test2a_trajectory import reset_windows


def test_h1_source_defect_preserves_exact_structural_relations_and_cancels_r():
    h = np.asarray([2.0, 3.0], dtype=np.float64)
    delta_a = np.asarray([-0.25, 0.5], dtype=np.float64)
    beta2 = 7.0
    v, depth, entropy, qv, qc, qr = h1_structural_source_error(
        h, delta_a, beta2
    )
    np.testing.assert_array_equal(v, 0.0)
    np.testing.assert_array_equal(depth, 0.0)
    np.testing.assert_array_equal(qr, 0.0)
    np.testing.assert_allclose(qv + qc + qr, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(entropy - beta2 * qv, 0.0, rtol=0.0, atol=0.0)


def test_h1_coefficient_contains_exact_half_dt_squared_and_target_normalizer():
    assert h1_tendency_loss_coefficient(100.0, 25.0, 0.4) == 80.0


def test_canonical_h1_uses_80_postprefix_transitions_without_future_truth():
    windows = reset_windows(range(80), 1, "endpoint", (1.0,))
    assert tuple(window.start_step for window in windows) == tuple(range(80))
    assert tuple(window.target_steps[0] for window in windows) == tuple(range(1, 81))
    assert max(step for window in windows for step in window.target_steps) == 80


class _Operations:
    @staticmethod
    def predict(parameters, observation):
        return DiscretePredictionCache(
            tendency=observation.payload * parameters["w"]
        )

    @staticmethod
    def subtract(left, right, name):
        del name
        return left - right

    @staticmethod
    def squared_mass_norm(value):
        return jnp.dot(value, jnp.asarray([2.0, 0.5]) * value)

    def gradient_contribution(self, parameters, observation, prediction, residual):
        del prediction, residual
        return jax.grad(
            lambda active: self.squared_mass_norm(
                observation.payload * active["w"] - observation.target_tendency
            )
        )(parameters)


def test_weighted_fixed_state_objective_matches_explicit_quadratic_and_gradient():
    observations = (
        DiscreteOfflineObservation(0, jnp.asarray([1.0, 2.0]), jnp.asarray([0.2, -0.1]), jnp.ones(2)),
        DiscreteOfflineObservation(1, jnp.asarray([-0.5, 0.7]), jnp.asarray([0.4, 0.3]), jnp.ones(2)),
    )
    coefficients = (0.25, 1.5)
    operations = _Operations()
    objective = WeightedFixedStateObjective(observations, operations, coefficients)
    parameters = {"w": jnp.asarray(0.8, dtype=jnp.float64)}

    def oracle(active):
        total = 0.0
        for observation, coefficient in zip(observations, coefficients):
            residual = observation.payload * active["w"] - observation.target_tendency
            total = total + coefficient * operations.squared_mass_norm(residual)
        return total

    value, gradient = objective.value_and_gradient(parameters)
    np.testing.assert_allclose(value, oracle(parameters), rtol=2.0e-15, atol=0.0)
    expected = jax.grad(oracle)(parameters)
    np.testing.assert_allclose(
        gradient["w"], expected["w"], rtol=2.0e-15, atol=0.0
    )


def test_gradient_relation_recovers_positive_scaling_and_detects_nonproportionality():
    right = {"x": jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)}
    left = {"x": 7.25 * right["x"]}
    relation = parameter_gradient_relation(left, right)
    np.testing.assert_allclose(relation["best_alpha_left_over_right"], 7.25)
    np.testing.assert_allclose(relation["gradient_cosine"], 1.0)
    np.testing.assert_allclose(
        relation["relative_nonproportional_residual"], 0.0, atol=2.0e-16
    )

    nonproportional = {"x": left["x"] + jnp.asarray([0.0, 0.1, 0.4])}
    relation = parameter_gradient_relation(nonproportional, right)
    assert relation["relative_nonproportional_residual"] > 0.0
