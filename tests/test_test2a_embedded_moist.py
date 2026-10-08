"""Pure-JAX certification of the frozen-neural-A/original-R local physics."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from dimswe.jax_moist import moist_rates_and_source_density_jax
from dimswe.learned_physics.parameters import tree_axpy, tree_dot, tree_norm
from dimswe.test2a_embedded_moist import (
    FrozenNeuralAMoistPhysics,
    certify_frozen_network_prediction_parity,
    load_frozen_neural_a_physics,
    load_test2a_embedding_configuration,
    parameter_pytree_sha256,
)
from dimswe.test2a_operator import (
    FEATURE_ORDER,
    MLPConfiguration,
    NormalizationMetadata,
    initialize_mlp,
)


EMBEDDING_CONFIGURATION = Path("dimswe/configs/test2a_embedded_neural_a.json")
FROZEN_ARTIFACT = Path(
    "external-results/test2a/optimizer-study/continuation-m20-plus45000/"
    "continuation_final_parameters.npz"
)
OPERATOR_DATASET = Path(
    "external-results/test2a/dataset/doublevortex_A_operator.npz"
)
STATE_KEYS = ("h", "S", "Qv", "Qc")
SOURCE_KEYS = ("S", "Qv", "Qc", "Qr")


def _normalization():
    return NormalizationMetadata(
        feature_order=FEATURE_ORDER,
        input_offset=np.asarray([750.0, 7354.62, 1.5, 0.75, 0.0]),
        input_scale=np.asarray([20.0, 200.0, 0.3, 0.2, 1.0]),
        input_zero_scale=(False, False, False, False, True),
        output_scale=1.0e-8,
        output_scale_method="uncentered_training_rms",
        output_candidates={"rms": 1.0e-8, "standard_deviation": 1.0e-8, "max_abs": 1.0e-7},
        output_zero_scale=False,
    )


def _physics(use_jit=False):
    configuration = MLPConfiguration(hidden_layers=(4,), seed=7)
    return FrozenNeuralAMoistPhysics(
        initialize_mlp(configuration),
        configuration,
        _normalization(),
        provenance={"test": "tiny deterministic"},
        use_jit=use_jit,
    )


def _inputs(shape=(2, 3)):
    modulation = np.linspace(0.98, 1.02, np.prod(shape)).reshape(shape)
    h = 750.0 * modulation
    state = {
        "h": jnp.asarray(h),
        "S": jnp.asarray(h * 9.80616 * (1.0 + 2.0e-4 * modulation)),
        "Qv": jnp.asarray(h * (0.0015 + 8.0e-4 * modulation)),
        "Qc": jnp.asarray(h * (0.00015 + 5.0e-4 * modulation)),
    }
    fields = {"B": jnp.zeros(shape, dtype=jnp.float64)}
    parameters = {
        "g": jnp.asarray(9.80616),
        "q0": jnp.asarray(0.002),
        "H0": jnp.asarray(750.0),
        "gamma_r": jnp.asarray(0.001),
        "qprecip": jnp.asarray(0.0001),
        "L": jnp.asarray(10.0),
        "configured_dt": jnp.asarray(100.0),
    }
    return state, fields, parameters


def _state_direction(state):
    scales = {"h": 0.2, "S": -1.1, "Qv": 3.0e-4, "Qc": -2.0e-4}
    return {
        key: jnp.full_like(value, scales[key], dtype=jnp.float64)
        for key, value in state.items()
    }


def _source_covector(source):
    scales = {"S": 0.3, "Qv": -0.7, "Qc": 1.1, "Qr": -0.4}
    return {
        key: jnp.full_like(value, scales[key], dtype=jnp.float64)
        for key, value in source.items()
    }


def _tree_dot(left, right):
    return float(tree_dot(left, right))


def _tree_relative_error(actual, expected):
    actual_flat, _ = ravel_pytree(actual)
    expected_flat, _ = ravel_pytree(expected)
    return float(
        jnp.linalg.norm(actual_flat - expected_flat)
        / max(float(jnp.linalg.norm(expected_flat)), np.finfo(np.float64).tiny)
    )


def test_selected_embedding_configuration_is_opt_in_and_training_normalized():
    record = load_test2a_embedding_configuration(EMBEDDING_CONFIGURATION)
    assert record["physics"]["default"] is False
    assert record["physics"]["mode"] == "neural_A_original_R"
    assert record["normalization_contract"]["refit_during_embedding"] is False
    assert record["truth_access"]["allowed_state_indices"] == [0, 80]


@pytest.mark.skipif(not FROZEN_ARTIFACT.exists(), reason="external frozen artifact absent")
def test_exact_selected_frozen_artifact_loads_with_expected_pytree():
    physics = load_frozen_neural_a_physics(EMBEDDING_CONFIGURATION, use_jit=False)
    assert physics.model_configuration.parameter_count == 1281
    assert parameter_pytree_sha256(physics.parameters) == (
        "f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56"
    )
    assert physics.provenance["states_after_80_accessed"] is False


@pytest.mark.skipif(
    not (FROZEN_ARTIFACT.exists() and OPERATOR_DATASET.exists()),
    reason="external frozen artifact or training-only dataset absent",
)
def test_frozen_predictions_match_test2a1_on_all_training_samples(tmp_path):
    output = tmp_path / "prediction_parity.json"
    result = certify_frozen_network_prediction_parity(
        EMBEDDING_CONFIGURATION, OPERATOR_DATASET, output
    )
    assert result["sample_count"] == 331_776
    assert result["truth_state_indices"] == [0, 80]
    assert result["states_after_80_accessed"] is False
    assert result["bitwise_equal"] is True
    assert result["maximum_absolute_difference"] == 0.0
    assert result["relative_l2_difference"] == 0.0


@pytest.mark.skipif(not FROZEN_ARTIFACT.exists(), reason="external frozen artifact absent")
def test_frozen_loader_rejects_architecture_reinterpretation(tmp_path):
    record = json.loads(EMBEDDING_CONFIGURATION.read_text(encoding="utf-8"))
    record["frozen_operator"]["architecture"]["hidden_layers"] = [31, 32]
    record["frozen_operator"]["architecture"]["parameter_count"] = 1249
    changed = tmp_path / "changed_architecture.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="leaf .* shape changed"):
        load_frozen_neural_a_physics(changed, use_jit=False)


@pytest.mark.skipif(not FROZEN_ARTIFACT.exists(), reason="external frozen artifact absent")
def test_frozen_loader_rejects_changed_normalization_provenance(tmp_path):
    record = json.loads(EMBEDDING_CONFIGURATION.read_text(encoding="utf-8"))
    record["dataset_metadata_sha256"] = "0" * 64
    changed = tmp_path / "changed_normalization.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset metadata fingerprint mismatch"):
        load_frozen_neural_a_physics(changed, use_jit=False)


def test_hybrid_source_identities_and_original_R_are_exact():
    physics = _physics()
    state, fields, parameters = _inputs()
    hybrid = physics.combined_kernel(state, fields, parameters)
    analytical = moist_rates_and_source_density_jax(state, fields, parameters)
    np.testing.assert_array_equal(hybrid["rates"]["R"], analytical["rates"]["R"])
    assert float(jnp.max(jnp.abs(hybrid["rates"]["R"]))) > 0.0
    h = state["h"]
    a_rate = hybrid["rates"]["A"]
    r_rate = hybrid["rates"]["R"]
    beta2 = parameters["g"] * parameters["L"]
    np.testing.assert_array_equal(hybrid["source"]["Qv"], h * a_rate)
    np.testing.assert_array_equal(hybrid["source"]["Qc"], -h * (a_rate + r_rate))
    np.testing.assert_array_equal(hybrid["source"]["Qr"], h * r_rate)
    np.testing.assert_array_equal(hybrid["source"]["S"], h * beta2 * a_rate)
    np.testing.assert_allclose(
        hybrid["source"]["Qv"]
        + hybrid["source"]["Qc"]
        + hybrid["source"]["Qr"],
        0.0,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        hybrid["source"]["S"] - beta2 * hybrid["source"]["Qv"],
        0.0,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    )


def test_analytical_path_remains_unchanged_and_distinct():
    physics = _physics()
    state, fields, parameters = _inputs()
    before = moist_rates_and_source_density_jax(state, fields, parameters)
    hybrid = physics.combined_kernel(state, fields, parameters)
    after = moist_rates_and_source_density_jax(state, fields, parameters)
    for group in ("rates", "source"):
        for key in before[group]:
            np.testing.assert_array_equal(before[group][key], after[group][key])
    assert not np.array_equal(hybrid["rates"]["A"], before["rates"]["A"])
    np.testing.assert_array_equal(hybrid["rates"]["R"], before["rates"]["R"])


def test_state_jvp_and_vjp_are_exact_local_transposes_and_match_fd():
    physics = _physics()
    state, fields, parameters = _inputs()
    direction = _state_direction(state)
    source, tangent = physics.state_jvp_kernel(
        state, direction, fields, parameters
    )
    epsilon = 1.0e-4
    plus = physics.combined_kernel(
        tree_axpy(state, epsilon, direction), fields, parameters
    )["source"]
    minus = physics.combined_kernel(
        tree_axpy(state, -epsilon, direction), fields, parameters
    )["source"]
    centered = jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon), plus, minus
    )
    assert _tree_relative_error(centered, tangent) < 2.0e-8
    covector = _source_covector(source)
    pullback = physics.state_vjp_kernel(state, covector, fields, parameters)
    assert _tree_dot(tangent, covector) == pytest.approx(
        _tree_dot(direction, pullback), rel=5.0e-13, abs=5.0e-13
    )


def test_parameter_jvp_and_vjp_cover_every_pytree_leaf():
    physics = _physics()
    state, fields, parameters = _inputs()
    direction = jax.tree.map(
        lambda value: jnp.linspace(-0.3, 0.4, value.size).reshape(value.shape)
        / max(1, value.size),
        physics.parameters,
    )
    source, tangent = physics.parameter_jvp(
        state, direction, fields, parameters
    )
    epsilon = 1.0e-4
    plus = physics.combined_with_parameters(
        state, fields, parameters, tree_axpy(physics.parameters, epsilon, direction)
    )["source"]
    minus = physics.combined_with_parameters(
        state, fields, parameters, tree_axpy(physics.parameters, -epsilon, direction)
    )["source"]
    centered = jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon), plus, minus
    )
    assert _tree_relative_error(centered, tangent) < 2.0e-8
    covector = _source_covector(source)
    pullback = physics.parameter_vjp(state, covector, fields, parameters)
    assert len(jax.tree.leaves(pullback)) == len(jax.tree.leaves(physics.parameters))
    assert _tree_dot(tangent, covector) == pytest.approx(
        _tree_dot(direction, pullback), rel=5.0e-13, abs=5.0e-13
    )


def test_joint_differentiated_vjp_matches_centered_vjp_difference():
    physics = _physics()
    state, fields, parameters = _inputs()
    state_direction = _state_direction(state)
    parameter_direction = jax.tree.map(
        lambda value: jnp.full_like(value, 1.0e-3), physics.parameters
    )
    source = physics.combined_kernel(state, fields, parameters)["source"]
    covector = _source_covector(source)
    covector_direction = jax.tree.map(
        lambda value: -0.2 * value, covector
    )
    ordinary, incremental = physics.joint_differentiated_vjp(
        state,
        covector,
        state_direction,
        parameter_direction,
        covector_direction,
        fields,
        parameters,
    )

    def vjp_at(local_state, local_parameters, local_covector):
        source_map = lambda q, theta: physics.combined_with_parameters(
            q, fields, parameters, theta
        )["source"]
        _, pullback = jax.vjp(source_map, local_state, local_parameters)
        return pullback(local_covector)

    epsilon = 2.0e-4
    plus = vjp_at(
        tree_axpy(state, epsilon, state_direction),
        tree_axpy(physics.parameters, epsilon, parameter_direction),
        tree_axpy(covector, epsilon, covector_direction),
    )
    minus = vjp_at(
        tree_axpy(state, -epsilon, state_direction),
        tree_axpy(physics.parameters, -epsilon, parameter_direction),
        tree_axpy(covector, -epsilon, covector_direction),
    )
    centered = jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon), plus, minus
    )
    assert _tree_relative_error(centered, incremental) < 2.0e-7
    assert float(tree_norm(ordinary[0])) > 0.0
    assert float(tree_norm(ordinary[1])) > 0.0
