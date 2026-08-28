"""Cheap algebraic tests for Test 2A-3A deployed-discrete semantics."""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from dimswe.learned_physics.parameters import tree_axpy, tree_dot, tree_norm
from dimswe.test2a_discrete_offline import (
    DeployedDiscreteOfflineObjective,
    DiscreteOfflineObservation,
    DiscretePredictionCache,
    load_discrete_offline_configuration,
    objective_gradient_comparison,
    require_training_steps,
)
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2a_operator import (
    initialize_mlp,
    load_selected_configuration,
    mlp_configuration_from_record,
)


CONFIGURATION = "dimswe/configs/test2a_deployed_discrete_offline.json"
SELECTED_OPERATOR = "dimswe/configs/test2a_selected_operator.json"


def _parameters():
    return {
        "w": jnp.asarray([0.2, -0.35], dtype=jnp.float64),
        "nested": {"b": jnp.asarray(0.08, dtype=jnp.float64)},
    }


def _rate(parameters, features):
    return jnp.tanh(features @ parameters["w"] + parameters["nested"]["b"])


class _SyntheticOperations:
    """Fixed-state linear deployed maps with a shared nonzero R contribution."""

    def __init__(self):
        self.predict_calls = []
        self.original_r_evaluations = 0

    def predict(self, parameters, observation):
        payload = observation.payload
        self.predict_calls.append(payload["state_id"])
        self.original_r_evaluations += 1
        tendency = payload["B"] @ _rate(parameters, payload["features"])
        tendency = tendency + payload["shared_R_update"]
        return DiscretePredictionCache(tendency=tendency)

    @staticmethod
    def subtract(left, right, name):
        del name
        return left - right

    @staticmethod
    def squared_mass_norm(value):
        mass = jnp.asarray([[1.7, 0.2], [0.2, 0.9]], dtype=jnp.float64)
        return value @ mass @ value

    def _local_squared(self, parameters, observation):
        prediction = self.predict(parameters, observation).tendency
        residual = prediction - observation.target_tendency
        return self.squared_mass_norm(residual)

    def gradient_contribution(
        self, parameters, observation, prediction, residual
    ):
        del prediction, residual
        return jax.grad(
            lambda value: self._local_squared(value, observation)
        )(parameters)

    def hvp_contribution(
        self, parameters, direction, observation, prediction, residual
    ):
        del prediction, residual
        gradient = jax.grad(
            lambda value: self._local_squared(value, observation)
        )
        return jax.jvp(
            gradient,
            (parameters,),
            (direction,),
        )[1]


def _problem():
    operations = _SyntheticOperations()
    observations = []
    truth_rates = []
    for step, shift in enumerate((0.0, 0.25, -0.15)):
        features = jnp.asarray(
            [[1.0 + shift, -0.4], [0.3, 0.8 - shift], [-0.6, 0.5]],
            dtype=jnp.float64,
        )
        truth = jnp.asarray([0.25, -0.3, 0.45], dtype=jnp.float64) + shift
        deployed = jnp.asarray(
            [[1.0, 0.3, -0.2], [0.1, 0.7 + 0.1 * step, 0.5]],
            dtype=jnp.float64,
        )
        shared_r = jnp.asarray([0.04, -0.02], dtype=jnp.float64)
        payload = {
            "state_id": step,
            "features": features,
            "B": deployed,
            "shared_R_update": shared_r,
        }
        a_tendency = deployed @ truth
        observations.append(
            DiscreteOfflineObservation(
                step=step,
                payload=payload,
                target_tendency=a_tendency + shared_r,
                analytical_a_tendency=a_tendency,
            )
        )
        truth_rates.append(truth)
    return (
        DeployedDiscreteOfflineObjective(observations, operations),
        operations,
        tuple(truth_rates),
    )


def _relative_tree_error(actual, expected):
    actual_flat, _ = ravel_pytree(actual)
    expected_flat, _ = ravel_pytree(expected)
    return float(jnp.linalg.norm(actual_flat - expected_flat)) / max(
        float(jnp.linalg.norm(expected_flat)), np.finfo(np.float64).tiny
    )


def test_selected_contract_uses_seed0_and_forbids_future_truth():
    record = load_discrete_offline_configuration(CONFIGURATION)
    assert record["truth"]["state_indices"] == [0, 80]
    assert record["truth"]["states_after_80_forbidden"] is True
    assert record["physics"]["recursive_model_state_propagation"] is False
    assert record["optimizer"]["maximum_secant_storage"] == 20
    selected = load_selected_configuration(SELECTED_OPERATOR)
    initial = initialize_mlp(mlp_configuration_from_record(selected["model"]))
    assert parameter_pytree_sha256(initial) == record["model"][
        "canonical_initial_parameter_sha256"
    ]
    assert require_training_steps(range(81)) == tuple(range(81))
    with pytest.raises(ValueError, match="states after 80 are forbidden"):
        require_training_steps(range(82))


def test_global_normalization_matches_manual_mixed_mass_ratio():
    objective, operations, _ = _problem()
    parameters = _parameters()
    numerator = 0.0
    denominator = 0.0
    for observation in objective.observations:
        prediction = operations.predict(parameters, observation).tendency
        numerator += float(
            operations.squared_mass_norm(
                prediction - observation.target_tendency
            )
        )
        denominator += float(
            operations.squared_mass_norm(observation.analytical_a_tendency)
        )
    assert objective.value(parameters) == pytest.approx(
        numerator / denominator, rel=2.0e-15
    )
    assert objective.normalizer == pytest.approx(denominator, rel=2.0e-15)
    assert len(set(objective.normalization_terms)) > 1


def test_fixed_states_are_independent_and_original_r_is_evaluated_each_time():
    objective, operations, _ = _problem()
    before = tuple(
        np.asarray(observation.target_tendency).copy()
        for observation in objective.observations
    )
    objective.value(_parameters())
    assert operations.predict_calls[-3:] == [0, 1, 2]
    assert operations.original_r_evaluations >= 3
    for observation, expected in zip(objective.observations, before):
        np.testing.assert_array_equal(observation.target_tendency, expected)


def test_exact_gradient_matches_centered_directional_difference():
    objective, _, _ = _problem()
    parameters = _parameters()
    direction = {
        "w": jnp.asarray([0.3, -0.7], dtype=jnp.float64),
        "nested": {"b": jnp.asarray(0.2, dtype=jnp.float64)},
    }
    gradient = objective.gradient(parameters)
    epsilon = 2.0e-6
    centered = (
        objective.value(tree_axpy(parameters, epsilon, direction))
        - objective.value(tree_axpy(parameters, -epsilon, direction))
    ) / (2.0 * epsilon)
    assert centered == pytest.approx(
        float(tree_dot(gradient, direction)), rel=2.0e-9, abs=2.0e-10
    )
    assert jax.tree.structure(gradient) == jax.tree.structure(parameters)
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(gradient))


def test_exact_hvp_matches_centered_gradient_difference():
    objective, _, _ = _problem()
    parameters = _parameters()
    direction = {
        "w": jnp.asarray([-0.2, 0.4], dtype=jnp.float64),
        "nested": {"b": jnp.asarray(0.15, dtype=jnp.float64)},
    }
    action = objective.hess_vec(parameters, direction)
    epsilon = 2.0e-5
    plus = objective.gradient(tree_axpy(parameters, epsilon, direction))
    minus = objective.gradient(tree_axpy(parameters, -epsilon, direction))
    centered = jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon), plus, minus
    )
    assert _relative_tree_error(action, centered) < 3.0e-9


def test_operator_and_discrete_comparison_detects_nonisotropic_reweighting():
    objective, _, truths = _problem()
    parameters = _parameters()

    def operator_loss(value):
        numerator = jnp.float64(0.0)
        denominator = jnp.float64(0.0)
        for observation, truth in zip(objective.observations, truths):
            error = _rate(value, observation.payload["features"]) - truth
            numerator = numerator + jnp.vdot(error, error)
            denominator = denominator + jnp.vdot(truth, truth)
        return numerator / denominator

    operator_value, operator_gradient = jax.value_and_grad(operator_loss)(parameters)
    discrete_value, discrete_gradient = objective.value_and_gradient(parameters)
    comparison = objective_gradient_comparison(
        operator_value,
        operator_gradient,
        discrete_value,
        discrete_gradient,
    )
    assert comparison["gradient_cosine_similarity"] < 0.999
    assert comparison["relative_nonproportional_gradient_residual"] > 1.0e-3
    assert comparison["operator_value"] != pytest.approx(
        comparison["deployed_discrete_value"], rel=1.0e-3
    )


def test_zero_global_a_sensitive_normalizer_is_rejected():
    operations = _SyntheticOperations()
    observation = DiscreteOfflineObservation(
        step=0,
        payload={
            "state_id": 0,
            "features": jnp.zeros((3, 2), dtype=jnp.float64),
            "B": jnp.zeros((2, 3), dtype=jnp.float64),
            "shared_R_update": jnp.ones(2, dtype=jnp.float64),
        },
        target_tendency=jnp.ones(2, dtype=jnp.float64),
        analytical_a_tendency=jnp.zeros(2, dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="normalization must be positive"):
        DeployedDiscreteOfflineObjective((observation,), operations)
