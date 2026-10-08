"""Focused production checks for J4A Benchmark 1: hidden-c0 recovery."""

from __future__ import annotations

import numpy as np
import pytest

from dimswe.hidden_c0 import (
    DEFAULT_INITIAL_C0,
    DEFAULT_TRUTH_C0,
    ScalarOptimizerConfiguration,
    build_hidden_c0_case,
    default_hidden_c0_scan,
    evaluate_hidden_c0,
    generate_hidden_c0_truth,
    optimize_hidden_c0,
    prepare_hidden_c0_objectives,
)
from dimswe.learned_physics import TrainingMode


def _flat(value):
    return np.hstack(tuple(np.ravel(block) for block in value.dat.data)).copy()


@pytest.fixture(scope="module")
def hidden_c0_case():
    return build_hidden_c0_case()


@pytest.fixture(scope="module")
def hidden_c0_truth(hidden_c0_case):
    return generate_hidden_c0_truth(hidden_c0_case, num_steps=3)


@pytest.fixture(scope="module")
def hidden_c0_suite(hidden_c0_case, hidden_c0_truth):
    return prepare_hidden_c0_objectives(hidden_c0_case, hidden_c0_truth)


def test_truth_generation_is_reproducible_finite_and_does_not_mutate_ic(
    hidden_c0_case, hidden_c0_truth
):
    initial_before = _flat(hidden_c0_case.initial_state)
    repeated = generate_hidden_c0_truth(hidden_c0_case, num_steps=3)
    np.testing.assert_array_equal(repeated.dataset.states, hidden_c0_truth.dataset.states)
    np.testing.assert_array_equal(repeated.dataset.times, hidden_c0_truth.dataset.times)
    np.testing.assert_array_equal(_flat(hidden_c0_case.initial_state), initial_before)
    assert np.all(np.isfinite(hidden_c0_truth.dataset.states))
    assert not hidden_c0_truth.dataset.states.flags.writeable


def test_hidden_truth_differs_from_initial_guess_and_only_c0_is_hidden(
    hidden_c0_truth,
):
    metadata = hidden_c0_truth.dataset.metadata
    assert metadata.truth_c0 == DEFAULT_TRUTH_C0
    assert metadata.truth_c0 != DEFAULT_INITIAL_C0
    assert metadata.state_control_convention["physical_map"] == "c0 = 0.07 z"
    assert metadata.solver_configuration["serial_only"] is True
    assert metadata.physical_parameters["hyperviscosity"]["c0"] == DEFAULT_TRUTH_C0


@pytest.mark.parametrize("mode", tuple(TrainingMode))
def test_each_objective_has_a_strict_minimum_at_truth(hidden_c0_case, hidden_c0_suite, mode):
    objective = hidden_c0_suite[mode]
    truth_z = DEFAULT_TRUTH_C0 / hidden_c0_case.c0_scale
    left = objective.value(0.8 * truth_z)
    center = objective.value(truth_z)
    right = objective.value(1.2 * truth_z)
    assert center <= 1.0e-24
    assert center < left and center < right


@pytest.mark.parametrize("mode", tuple(TrainingMode))
def test_normalized_gradients_agree_with_centered_differences(
    hidden_c0_case, hidden_c0_suite, mode
):
    objective = hidden_c0_suite[mode]
    z = 0.112 / hidden_c0_case.c0_scale
    epsilon = 2.0e-5
    exact = objective.gradient(z)
    centered = (
        objective.value(z + epsilon) - objective.value(z - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(exact, centered, rtol=3.0e-6, atol=2.0e-9)


@pytest.mark.parametrize("mode", tuple(TrainingMode))
def test_normalized_hvp_agrees_with_centered_gradient_difference(
    hidden_c0_case, hidden_c0_suite, mode
):
    objective = hidden_c0_suite[mode]
    z = 0.112 / hidden_c0_case.c0_scale
    epsilon = 2.0e-5
    exact = objective.hess_vec(z, 1.0)
    centered = (
        objective.gradient(z + epsilon) - objective.gradient(z - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(exact, centered, rtol=8.0e-5, atol=2.0e-7)


@pytest.mark.parametrize("mode", tuple(TrainingMode))
def test_each_mode_recovers_hidden_c0_strictly(hidden_c0_suite, mode):
    configuration = ScalarOptimizerConfiguration(
        max_iterations=6,
        gradient_tolerance=1.0e-9,
        use_hvp=True,
    )
    result = optimize_hidden_c0(
        hidden_c0_suite[mode], configuration=configuration
    )
    assert result.success, result.failure_reason
    assert abs(result.recovered_c0 - DEFAULT_TRUTH_C0) <= 2.0e-10
    assert result.objective_history[-1] < result.objective_history[0]
    assert result.counts.gradient_evaluations >= 2
    assert result.counts.hvp_evaluations >= 1
    if mode in (TrainingMode.APRIORI_OFFLINE, TrainingMode.DISCRETE_OFFLINE):
        assert result.counts.solver_calls == 0
    else:
        assert result.counts.solver_calls > 0


def test_dense_objective_scan_has_expected_minimum(hidden_c0_suite):
    for mode in TrainingMode:
        scan = default_hidden_c0_scan(
            hidden_c0_suite[mode], truth_c0=DEFAULT_TRUTH_C0, points=5
        )
        assert scan.minimum_physical_c0 == DEFAULT_TRUTH_C0
        assert int(np.argmin(scan.objective)) == 2


def test_cross_evaluation_favors_recovered_parameter_and_is_repeatable(
    hidden_c0_case, hidden_c0_truth, hidden_c0_suite
):
    recovered = optimize_hidden_c0(
        hidden_c0_suite[TrainingMode.ROLLOUT]
    ).recovered_c0
    truth_before = hidden_c0_truth.dataset.states.copy()
    metrics = evaluate_hidden_c0(
        hidden_c0_case,
        hidden_c0_truth,
        hidden_c0_suite,
        recovered,
    )
    for values in metrics["objectives_under_all_training_modes"].values():
        assert values["recovered"] < values["initial"]
    assert metrics["one_step_state_prediction_error"] < 1.0e-11
    assert metrics["short_autonomous_rollout_error"] < 1.0e-11
    assert metrics["final_state_error"] < 1.0e-11
    assert metrics["accumulated_trajectory_error"] < 1.0e-11
    assert all(
        value < 1.0e-11
        for block in metrics["state_field_block_errors"].values()
        for value in block.values()
    )
    assert metrics["stability"]["all_states_finite"]
    assert metrics["repeatability"]["exact_repeated_state_vectors"]
    np.testing.assert_array_equal(hidden_c0_truth.dataset.states, truth_before)


def test_repeated_offline_fit_is_bitwise_deterministic(hidden_c0_suite):
    first = optimize_hidden_c0(hidden_c0_suite[TrainingMode.APRIORI_OFFLINE])
    second = optimize_hidden_c0(hidden_c0_suite[TrainingMode.APRIORI_OFFLINE])
    assert first.recovered_c0 == second.recovered_c0
    assert first.objective_history == second.objective_history
    assert first.gradient_norms == second.gradient_norms
