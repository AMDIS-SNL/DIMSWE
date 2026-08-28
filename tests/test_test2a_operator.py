"""Cheap pure-JAX certification for Test 2A-1 local operator learning."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from dimswe.test2a_operator import (
    DenseMLP,
    FEATURE_ORDER,
    HybridAMoistOutputMap,
    MLPConfiguration,
    SELECTED_SAMPLE_COUNT,
    TRAINING_STEPS,
    assemble_operator_dataset,
    build_learned_a_model,
    deployed_sample_count,
    diagnostic_baselines,
    fit_normalization,
    initialize_mlp,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    normalized_operator_objective,
    operator_metrics,
    require_training_steps,
    save_mlp_parameters,
    save_operator_dataset,
)


CONFIGURATION = Path("dimswe/configs/test2a_selected_operator.json")


def _training_records(cells=1):
    records = {}
    point_index = np.arange(cells * 16, dtype=np.float64).reshape(cells, 16)
    for step in TRAINING_STEPS:
        offset = np.float64(step / 100.0)
        records[step] = {
            "h": np.asarray(1.0 + offset + 0.01 * point_index, dtype=np.float64),
            "S": np.asarray(2.0 + offset + 0.02 * point_index, dtype=np.float64),
            "Qv": np.asarray(0.1 + offset + 0.001 * point_index, dtype=np.float64),
            "Qc": np.asarray(0.01 + offset + 0.002 * point_index, dtype=np.float64),
            "B": np.zeros((cells, 16), dtype=np.float64),
            "A": np.asarray(-0.2 + offset + 0.003 * point_index, dtype=np.float64),
        }
    return records


def test_exact_selected_deployed_sample_accounting():
    assert deployed_sample_count(81, 16, 16, 16) == SELECTED_SAMPLE_COUNT
    assert SELECTED_SAMPLE_COUNT == 331_776
    dataset = assemble_operator_dataset(_training_records(), cells=1)
    assert dataset.sample_count == 81 * 16
    assert dataset.steps == tuple(range(81))


def test_future_state_access_is_rejected():
    with pytest.raises(ValueError, match="after 80"):
        require_training_steps(range(82))
    records = _training_records()
    records[81] = records[80]
    with pytest.raises(ValueError, match="after 80"):
        assemble_operator_dataset(records, cells=1)


def test_feature_order_and_cell_local_repetition_are_exact():
    assert FEATURE_ORDER == ("h", "S", "Qv", "Qc", "B")
    records = _training_records()
    dataset = assemble_operator_dataset(records, cells=1)
    np.testing.assert_array_equal(
        dataset.features[0],
        [
            records[0]["h"][0, 0],
            records[0]["S"][0, 0],
            records[0]["Qv"][0, 0],
            records[0]["Qc"][0, 0],
            records[0]["B"][0, 0],
        ],
    )
    np.testing.assert_array_equal(dataset.features[15, :], [1.15, 2.3, 0.115, 0.04, 0.0])
    assert dataset.features[16, 0] == records[1]["h"][0, 0]


def test_training_only_normalization_roundtrip_and_output_choice():
    dataset = assemble_operator_dataset(_training_records(), cells=1)
    normalization = fit_normalization(dataset.features, dataset.targets)
    normalized = normalization.normalize_features(dataset.features)
    reconstructed = normalization.inverse_features(normalized)
    np.testing.assert_allclose(reconstructed, dataset.features, rtol=1.0e-14, atol=1.0e-14)
    normalized_a = normalization.normalize_a(dataset.targets)
    np.testing.assert_allclose(
        normalization.inverse_a(normalized_a), dataset.targets, rtol=1.0e-14, atol=1.0e-14
    )
    expected_rms = np.sqrt(np.mean(dataset.targets.reshape(-1) ** 2))
    assert normalization.output_scale_method == "uncentered_training_rms"
    assert normalization.output_scale == expected_rms
    assert normalization.input_zero_scale[-1]
    assert normalization.input_scale[-1] == 1.0
    assert normalization.to_record()["fitted_truth_state_indices"] == [0, 80]


def test_zero_target_normalization_is_explicit_and_invertible():
    features = np.ones((8, 5), dtype=np.float64)
    targets = np.zeros((8, 1), dtype=np.float64)
    normalization = fit_normalization(features, targets)
    assert normalization.output_zero_scale
    assert normalization.output_scale == 1.0
    np.testing.assert_array_equal(normalization.normalize_a(targets), targets)


def test_default_model_is_deterministic_configurable_and_has_1281_parameters():
    selected = load_selected_configuration(CONFIGURATION)
    configuration = MLPConfiguration(
        input_dimension=selected["model"]["input_dimension"],
        hidden_layers=tuple(selected["model"]["hidden_layers"]),
        output_dimension=selected["model"]["output_dimension"],
        activation=selected["model"]["activation"],
        dtype=selected["model"]["dtype"],
        seed=selected["model"]["seed"],
    )
    assert configuration.parameter_count == 1281
    assert selected["model"]["parameter_count"] == 1281
    left = initialize_mlp(configuration)
    right = initialize_mlp(configuration)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    ):
        assert left_leaf.dtype == jnp.float64
        np.testing.assert_array_equal(left_leaf, right_leaf)
    alternative = MLPConfiguration(hidden_layers=(4, 3, 2), activation="gelu", seed=4)
    assert alternative.parameter_count == (5 * 4 + 4) + (4 * 3 + 3) + (3 * 2 + 2) + (2 + 1)
    output = DenseMLP(alternative)(initialize_mlp(alternative), jnp.ones((7, 5), dtype=jnp.float64))
    assert output.shape == (7, 1)


def test_hybrid_output_replaces_only_A_and_retains_nonzero_original_R():
    features = np.arange(50, dtype=np.float64).reshape(10, 5) + 1.0
    targets = np.linspace(-2.0, 3.0, 10, dtype=np.float64).reshape(-1, 1)
    normalization = fit_normalization(features, targets)
    configuration = MLPConfiguration(hidden_layers=(), seed=0)
    parameters = initialize_mlp(configuration)
    parameters["layers"][0]["weight"] = jnp.zeros((5, 1), dtype=jnp.float64)
    parameters["layers"][0]["bias"] = jnp.asarray([2.0], dtype=jnp.float64)
    learned = build_learned_a_model(configuration, normalization)
    state = {
        "h": jnp.asarray([2.0, 3.0], dtype=jnp.float64),
        "S": jnp.asarray([1.0, 1.5], dtype=jnp.float64),
        "Qv": jnp.asarray([0.2, 0.3], dtype=jnp.float64),
        "Qc": jnp.asarray([0.04, 0.05], dtype=jnp.float64),
    }
    context = {
        "B": jnp.asarray([0.0, 0.1], dtype=jnp.float64),
        "beta2": jnp.asarray(7.0, dtype=jnp.float64),
    }
    original_r = jnp.asarray([0.25, 0.5], dtype=jnp.float64)
    result = learned(parameters, state, context, {"R": original_r})
    expected_a = np.full(2, 2.0 * normalization.output_scale)
    np.testing.assert_allclose(result["A"], expected_a)
    np.testing.assert_array_equal(result["R"], original_r)
    np.testing.assert_allclose(result["source"]["Qv"], state["h"] * expected_a)
    np.testing.assert_allclose(
        result["source"]["Qc"], -state["h"] * (expected_a + original_r)
    )
    np.testing.assert_allclose(result["source"]["Qr"], state["h"] * original_r)
    np.testing.assert_allclose(
        result["source"]["S"], state["h"] * context["beta2"] * expected_a
    )
    assert np.any(np.asarray(result["source"]["Qr"]) != 0.0)


def _tiny_objective_case():
    configuration = MLPConfiguration(hidden_layers=(3,), seed=6)
    parameters = initialize_mlp(configuration)
    model = DenseMLP(configuration)
    features = jnp.asarray(
        np.linspace(-1.0, 1.0, 40, dtype=np.float64).reshape(8, 5)
    )
    targets = jnp.asarray(
        np.linspace(-0.4, 0.7, 8, dtype=np.float64).reshape(8, 1)
    )

    def objective(value):
        return normalized_operator_objective(value, model, features, targets)

    return parameters, objective


def test_local_operator_gradient_directional_derivative():
    parameters, objective = _tiny_objective_case()
    flat, unravel = ravel_pytree(parameters)
    direction = jnp.linspace(-0.7, 0.9, flat.size, dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    flat_objective = lambda value: objective(unravel(value))
    gradient = jax.grad(flat_objective)(flat)
    exact = float(jnp.vdot(gradient, direction))
    step = 2.0e-5
    centered = float(
        (flat_objective(flat + step * direction) - flat_objective(flat - step * direction))
        / (2.0 * step)
    )
    np.testing.assert_allclose(centered, exact, rtol=2.0e-8, atol=2.0e-10)


def test_local_operator_exact_hvp_matches_gradient_difference():
    parameters, objective = _tiny_objective_case()
    flat, unravel = ravel_pytree(parameters)
    direction = jnp.linspace(0.2, 1.1, flat.size, dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    flat_objective = lambda value: objective(unravel(value))
    gradient = jax.grad(flat_objective)
    exact = jax.jvp(gradient, (flat,), (direction,))[1]
    step = 2.0e-5
    centered = (gradient(flat + step * direction) - gradient(flat - step * direction)) / (2.0 * step)
    np.testing.assert_allclose(centered, exact, rtol=2.0e-8, atol=2.0e-9)


def test_metrics_and_trivial_baselines_cover_active_strata():
    dataset = assemble_operator_dataset(_training_records(), cells=1)
    normalization = fit_normalization(dataset.features, dataset.targets)
    baselines = diagnostic_baselines(dataset.features, dataset.targets, normalization)
    assert set(baselines) == {"zero", "constant_training_mean", "affine_normalized_five_input"}
    perfect = operator_metrics(dataset.targets, dataset.targets)
    assert perfect["normalized_mse"] == 0.0
    assert perfect["relative_rms_error"] == 0.0
    assert perfect["sign_accuracy"] == 1.0
    assert perfect["magnitude_strata"]["abs_A_gt_1e-03_max_abs_A"]["sample_count"] > 0
    for record in perfect["sign_accuracy_strata"].values():
        assert record["accuracy"] == 1.0


def test_sign_accuracy_is_stratified_by_physical_activity():
    target = np.asarray([1.0, -0.2, 0.02, -0.002, 1.0e-7], dtype=np.float64)
    prediction = np.asarray([1.0, -0.2, -0.02, 0.002, -1.0e-7], dtype=np.float64)
    metrics = operator_metrics(prediction, target)
    strata = metrics["sign_accuracy_strata"]
    assert strata["abs_A_gt_1e-03_max_abs_A"]["sample_count"] == 4
    assert strata["abs_A_gt_1e-03_max_abs_A"]["accuracy"] == pytest.approx(0.5)
    assert strata["abs_A_gt_1e-02_max_abs_A"]["sample_count"] == 3
    assert strata["abs_A_gt_1e-02_max_abs_A"]["accuracy"] == pytest.approx(2.0 / 3.0)
    assert strata["abs_A_gt_1e-01_max_abs_A"]["sample_count"] == 2
    assert strata["abs_A_gt_1e-01_max_abs_A"]["accuracy"] == 1.0


def test_dataset_and_parameter_roundtrip_without_pickle(tmp_path):
    dataset = assemble_operator_dataset(_training_records(), cells=1)
    normalization = fit_normalization(dataset.features, dataset.targets)
    dataset_path = tmp_path / "operator_dataset.npz"
    save_operator_dataset(
        dataset,
        dataset_path,
        {"normalization": normalization.to_record()},
    )
    restored, metadata = load_operator_dataset(dataset_path)
    np.testing.assert_array_equal(restored.features, dataset.features)
    np.testing.assert_array_equal(restored.targets, dataset.targets)
    assert metadata["states_after_80_accessed"] is False

    configuration = MLPConfiguration(hidden_layers=(4,), seed=3)
    parameters = initialize_mlp(configuration)
    parameter_path = tmp_path / "parameters.npz"
    save_mlp_parameters(parameter_path, parameters, configuration)
    loaded, loaded_configuration = load_mlp_parameters(parameter_path)
    assert loaded_configuration == configuration
    for expected, actual in zip(
        jax.tree_util.tree_leaves(parameters), jax.tree_util.tree_leaves(loaded)
    ):
        np.testing.assert_array_equal(actual, expected)


def test_selected_configuration_freezes_only_test2a1_choices():
    selected = load_selected_configuration(CONFIGURATION)
    assert selected["data"]["truth_state_indices"] == [0, 80]
    assert selected["data"]["sample_count"] == 331_776
    assert selected["physics"]["learned"] == "A only"
    assert selected["physics"]["original_R_retained"] is True
    assert selected["optimizer"]["library"] == "PyROL/ROL"
    assert selected["optimizer"]["method"] == "line-search L-BFGS"
    assert selected["optimizer"]["maximum_secant_storage"] == 10
    assert selected["optimizer"]["production_HVP"] is False
    assert "coordinates" not in selected["data"]["feature_order"]
    assert json.loads(CONFIGURATION.read_text())["format_version"] == 1
