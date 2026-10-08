"""Focused pure-JAX contracts for the J4A learned-physics framework."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimswe.learned_physics import (
    DiscreteOfflineExample,
    ExperimentDefinition,
    Float64TreeError,
    LearnedPhysicsModel,
    LocalOfflineExample,
    LossAccumulation,
    RolloutExample,
    TrainingMode,
    TruthDataset,
    TruthMetadata,
    TruthResetWindow,
    apriori_offline,
    discrete_offline,
    load_truth_dataset,
    objective_for_mode,
    rollout,
    save_truth_dataset,
    tree_all_finite,
    tree_axpy,
    tree_copy,
    tree_dot,
    tree_norm,
    tree_zeros,
    truth_reset,
)


def _f64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def test_arbitrary_pytree_parameter_utilities_are_float64_and_owned():
    parameters = {
        "dense": (_f64([1.0, -2.0]), {"bias": _f64(3.0)}),
        "scalar": _f64(4.0),
    }
    copied = tree_copy(parameters)
    zeros = tree_zeros(parameters)
    np.testing.assert_allclose(np.asarray(tree_dot(parameters, parameters)), 30.0)
    np.testing.assert_allclose(np.asarray(tree_norm(parameters)), np.sqrt(30.0))
    updated = tree_axpy(parameters, _f64(2.0), zeros)
    np.testing.assert_allclose(np.asarray(updated["scalar"]), 4.0)
    assert bool(tree_all_finite(copied))
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(copied))
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(zeros))


def test_model_composition_is_feature_then_model_then_output_map():
    calls = []

    def feature(state, context):
        calls.append("feature")
        return state["x"] + context["forcing"]

    def parameterized(parameters, features):
        calls.append("model")
        return parameters["weight"] * features

    def output(state, context, baseline, raw):
        calls.append("output")
        return {"source": baseline["source"] + raw}

    learned = LearnedPhysicsModel(feature, parameterized, output)
    result = learned(
        {"weight": _f64(2.0)},
        {"x": _f64(3.0)},
        {"forcing": _f64(4.0)},
        {"source": _f64(5.0)},
    )
    assert calls == ["feature", "model", "output"]
    np.testing.assert_allclose(np.asarray(result["source"]), 19.0)


def test_composition_does_not_mutate_caller_owned_containers():
    parameters = {"weight": _f64(2.0)}
    state = {"x": _f64(3.0)}
    context = {"forcing": _f64(4.0)}
    baseline = {"source": _f64(5.0)}

    def mutating_feature(owned_state, owned_context):
        owned_state["added"] = _f64(99.0)
        owned_context["forcing"] = _f64(-1.0)
        return owned_state["x"]

    learned = LearnedPhysicsModel(
        mutating_feature,
        lambda owned_parameters, features: owned_parameters["weight"] * features,
        lambda owned_state, owned_context, owned_baseline, raw: raw,
    )
    learned(parameters, state, context, baseline)
    assert set(parameters) == {"weight"}
    assert set(state) == {"x"}
    np.testing.assert_allclose(np.asarray(context["forcing"]), 4.0)
    np.testing.assert_allclose(np.asarray(baseline["source"]), 5.0)


def test_float32_is_rejected_at_every_pytree_boundary():
    with pytest.raises(Float64TreeError, match="float64"):
        tree_copy({"bad": jnp.asarray(1.0, dtype=jnp.float32)})
    learned = LearnedPhysicsModel(
        lambda state, context: state,
        lambda parameters, features: features,
        lambda state, context, baseline, raw: raw,
    )
    with pytest.raises(Float64TreeError, match="float64"):
        learned(_f64(1.0), jnp.asarray(1.0, dtype=jnp.float32))


def _definition(mode=TrainingMode.APRIORI_OFFLINE):
    return ExperimentDefinition(
        benchmark="hidden_c0",
        truth_configuration={"truth_c0": 0.14, "backend": "production"},
        baseline_configuration={"initial_c0": 0.07},
        model_configuration={"parameter": "normalized_z"},
        training_mode=mode,
        observation_definition={"kind": "weak_operator"},
        rollout_horizon=3,
        seed=7,
        optimizer_configuration={"iterations": 4},
        evaluation_metrics=("c0_error", "rollout_error"),
    )


def test_configuration_serialization_is_deterministic_and_roundtrips():
    first = _definition().to_json()
    second = ExperimentDefinition.from_dict(json.loads(first)).to_json()
    assert first == second
    assert first.index('"baseline_configuration"') < first.index('"benchmark"')


def _metadata():
    return TruthMetadata(
        benchmark="hidden_c0",
        solver_backend="production",
        timestep=100.0,
        num_steps=2,
        initial_condition={"name": "tiny"},
        physical_parameters={"s": 3.2},
        truth_c0=0.14,
        moist_backend="ufl",
        random_seed=0,
        state_control_convention={"fields": ["v", "h"], "control": "z"},
        solver_configuration={"serial_only": True},
    )


def test_truth_npz_json_metadata_roundtrip(tmp_path):
    dataset = TruthDataset(
        states=np.arange(12, dtype=np.float64).reshape(3, 4),
        times=np.array([0.0, 100.0, 200.0]),
        metadata=_metadata(),
    )
    paths = save_truth_dataset(dataset, tmp_path / "truth")
    loaded = load_truth_dataset(tmp_path / "truth")
    np.testing.assert_array_equal(loaded.states, dataset.states)
    np.testing.assert_array_equal(loaded.times, dataset.times)
    assert loaded.metadata.to_dict() == dataset.metadata.to_dict()
    assert paths[0].suffix == ".npz" and paths[1].suffix == ".json"
    assert not loaded.states.flags.writeable


def test_objective_mode_dispatch_calls_each_explicit_api():
    local = (LocalOfflineExample(_f64(1.0), None, _f64(2.0)),)
    discrete = (DiscreteOfflineExample(_f64(1.0), None, _f64(3.0)),)
    reset_windows = (TruthResetWindow(_f64(1.0), (None,), (_f64(2.0),)),)
    rollout_example = RolloutExample(_f64(1.0), (None,), (_f64(2.0),))
    parameter = _f64(1.0)
    assert float(
        objective_for_mode(
            TrainingMode.APRIORI_OFFLINE,
            parameter,
            examples=local,
            predict_physics=lambda p, x, c: p + x,
        )
    ) == 0.0
    assert float(
        objective_for_mode(
            TrainingMode.DISCRETE_OFFLINE,
            parameter,
            examples=discrete,
            predict_physics=lambda p, x, c: p,
            discrete_map=lambda x, c, physics: x + physics,
        )
    ) == 0.5
    assert float(
        objective_for_mode(
            TrainingMode.TRUTH_RESET,
            parameter,
            windows=reset_windows,
            transition=lambda p, x, c: x + p,
        )
    ) == 0.0
    assert float(
        objective_for_mode(
            TrainingMode.ROLLOUT,
            parameter,
            example=rollout_example,
            transition=lambda p, x, c: x + p,
        )
    ) == 0.0


def test_apriori_offline_never_calls_solver():
    calls = {"solver": 0}

    def forbidden_solver(*args):
        calls["solver"] += 1
        raise AssertionError("solver was called")

    examples = (LocalOfflineExample(_f64(2.0), None, _f64(6.0)),)
    value = apriori_offline(
        _f64(3.0), examples, lambda p, state, context: p * state
    )
    assert float(value) == 0.0
    assert calls["solver"] == 0
    assert callable(forbidden_solver)


def test_discrete_offline_uses_each_truth_state_without_recursion():
    seen = []
    examples = tuple(
        DiscreteOfflineExample(_f64(state), None, _f64(state + 1.0))
        for state in (10.0, 20.0, 30.0)
    )

    def fixed_discrete_map(truth_state, context, physics):
        seen.append(float(truth_state))
        return truth_state + physics

    value = discrete_offline(
        _f64(1.0),
        examples,
        lambda p, state, context: p,
        fixed_discrete_map,
    )
    assert float(value) == 0.0
    assert seen == [10.0, 20.0, 30.0]


def test_truth_reset_restarts_every_window_from_trusted_state():
    seen = []
    windows = (
        TruthResetWindow(_f64(10.0), (None, None), (_f64(11.0), _f64(12.0))),
        TruthResetWindow(_f64(100.0), (None, None), (_f64(101.0), _f64(102.0))),
    )

    def transition(parameter, state, context):
        seen.append(float(state))
        return state + parameter

    value = truth_reset(
        _f64(1.0), windows, transition, accumulation=LossAccumulation.ACCUMULATED
    )
    assert float(value) == 0.0
    assert seen == [10.0, 11.0, 100.0, 101.0]


def test_rollout_recursively_uses_its_own_previous_prediction():
    seen = []
    example = RolloutExample(
        _f64(10.0),
        (None, None, None),
        (_f64(11.0), _f64(12.0), _f64(13.0)),
    )

    def transition(parameter, state, context):
        seen.append(float(state))
        return state + parameter

    value = rollout(_f64(1.0), example, transition)
    assert float(value) == 0.0
    assert seen == [10.0, 11.0, 12.0]


def test_four_modes_remain_distinct_stable_record_values():
    assert tuple(mode.value for mode in TrainingMode) == (
        "apriori_offline",
        "discrete_offline",
        "truth_reset",
        "rollout",
    )
    assert len(set(TrainingMode)) == 4
