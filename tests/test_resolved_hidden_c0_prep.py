"""Cheap, Firedrake-free checks for J4B resolved hidden-c0 preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from dimswe.resolved_hidden_c0 import (
    C0_SCALE,
    CANDIDATE_CASES,
    LateTimeGrowthConfiguration,
    ObjectiveScanConfiguration,
    ResolvedInferenceConfiguration,
    ResolvedPilotConfiguration,
    RolloutLoss,
    SolverLossNormalization,
    STATE_FIELDS,
    build_inference_index_plan,
    fieldwise_normalized_separation,
    late_time_growth_indicator,
    normalized_separation,
    paired_pilot_configurations,
    read_json_record,
    resolved_truth_state_indices,
    scan_scalar_objective,
    shell_averaged_vector_spectrum,
    write_json_record,
)


def test_pilot_configuration_is_deterministic_and_supports_repository_candidates():
    first = ResolvedPilotConfiguration()
    second = ResolvedPilotConfiguration.from_dict(first.to_dict())
    assert first == second
    assert first.to_json() == second.to_json()
    assert set(CANDIDATE_CASES) == {"doublevortex", "TC5", "TC2"}
    assert first.output_steps == tuple(range(0, 21, 2))
    assert first.sampling_shape == (32, 32)


def test_paired_configs_differ_only_in_c0_and_operational_output_path(tmp_path):
    base = ResolvedPilotConfiguration(
        output_directory=str(tmp_path / "ignored"), nx=8, ny=6
    )
    left, right = paired_pilot_configurations(
        base, 0.05, 0.16, parent_directory=tmp_path / "pair"
    )
    assert left.c0 == 0.05 and right.c0 == 0.16
    assert left.output_directory != right.output_directory
    assert left.physics_configuration(include_c0=False) == right.physics_configuration(
        include_c0=False
    )
    differing = {
        key
        for key in left.physics_configuration()
        if left.physics_configuration()[key] != right.physics_configuration()[key]
    }
    assert differing == {"c0"}


def test_metadata_record_roundtrip_and_atomic_path_handling(tmp_path):
    destination = tmp_path / "nested" / "metadata.json"
    record = {
        "restartability": {
            "kind": "experiment/state restart snapshots",
            "adjoint_checkpointing": False,
        },
        "fields": STATE_FIELDS,
        "completed_output_steps": (0, 2, 4),
    }
    assert write_json_record(destination, record) == destination
    assert read_json_record(destination) == {
        "completed_output_steps": [0, 2, 4],
        "fields": list(STATE_FIELDS),
        "restartability": {
            "adjoint_checkpointing": False,
            "kind": "experiment/state restart snapshots",
        },
    }
    assert not destination.with_name("metadata.json.tmp").exists()


def test_normalized_and_fieldwise_separation_do_not_mutate_inputs():
    right = np.arange(1.0, 10.0)
    left = right.copy()
    left[3:6] += 0.5
    left_before = left.copy()
    right_before = right.copy()
    slices = {
        name: (index, index + 1)
        for index, name in enumerate(STATE_FIELDS)
    }
    # Extend the final block to cover the remaining synthetic coefficients.
    slices["Qr"] = (5, 9)
    result = fieldwise_normalized_separation(left, right, slices)
    assert result["v"] == 0.0
    assert result["S"] == 0.0
    assert result["Qv"] > 0.0
    assert normalized_separation(left, right) > 0.0
    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)


def test_weighted_separation_matches_diagonal_mass_formula():
    left = np.array([2.0, 1.0])
    right = np.array([1.0, 1.0])
    weights = np.array([4.0, 1.0])
    expected = np.sqrt(4.0 / 5.0)
    assert normalized_separation(left, right, weights=weights) == expected


def test_uniform_periodic_spectrum_has_parseval_normalization_and_known_shell():
    nx = ny = 32
    x = (np.arange(nx) + 0.5) / nx
    velocity = np.zeros((ny, nx, 2), dtype=np.float64)
    velocity[:, :, 0] = np.sin(2.0 * np.pi * 3.0 * x)[None, :]
    original = velocity.copy()
    spectrum = shell_averaged_vector_spectrum(velocity, lx=2.0, ly=3.0)
    assert int(np.argmax(spectrum.shell_energy_sum[1:]) + 1) == 3
    np.testing.assert_allclose(
        spectrum.parseval_mean_kinetic_energy, 0.25, rtol=0.0, atol=2.0e-15
    )
    np.testing.assert_array_equal(velocity, original)
    assert 0.0 <= spectrum.high_wavenumber_fraction <= 1.0


def test_finite_histories_are_not_automatically_called_stable():
    from dimswe.analyze_resolved_hidden_c0 import _growth_classification

    times = np.arange(8.0)
    history = np.ones(8)
    diagnostic = late_time_growth_indicator(
        times, history, growth_factor=2.0
    )
    assert not diagnostic["suspicious_late_time_growth"]
    classification = _growth_classification(True, {"metric": diagnostic})
    assert classification == (
        "no suspicious late-time growth detected; stability is not proved"
    )


def test_late_time_growth_flags_finite_explosive_synthetic_histories_without_mutation():
    configuration = LateTimeGrowthConfiguration()
    times = np.arange(8.0)
    history = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 5.0, 20.0, 40.0])
    before = history.copy()
    diagnostic = late_time_growth_indicator(
        times,
        history,
        growth_factor=configuration.hyperviscosity_tendency_factor,
        baseline_fraction=configuration.baseline_fraction,
        tail_fraction=configuration.tail_fraction,
        absolute_floor=configuration.absolute_floor,
    )
    assert np.all(np.isfinite(history))
    assert diagnostic["suspicious_late_time_growth"]
    assert diagnostic["tail_to_baseline_ratio"] == 30.0
    np.testing.assert_array_equal(history, before)


@pytest.mark.parametrize("stride", (1, 2, 5, 10))
def test_observation_cadences_and_horizons_are_explicit(stride):
    configuration = ResolvedInferenceConfiguration(
        truth_run_directory="selected_truth",
        training_start_step=0,
        training_stop_step=20,
        heldout_stop_step=40,
        observation_stride=stride,
        truth_reset_horizon=stride,
        rollout_horizon=stride,
        rollout_loss=RolloutLoss.TERMINAL,
    )
    plan = build_inference_index_plan(configuration)
    assert plan.training_observations[0] == 0
    assert plan.training_observations[-1] == 20
    assert plan.heldout_observations[0] == 20
    assert plan.heldout_observations[-1] == 40
    assert plan.rollout_prefixes == (stride,)
    assert all(stop - start == stride for start, stop in plan.truth_reset_windows)
    assert set(configuration.training_transition_steps).isdisjoint(
        configuration.heldout_transition_steps
    )


def test_terminal_and_accumulated_rollout_plans_are_not_collapsed():
    common = dict(
        truth_run_directory="selected_truth",
        training_stop_step=10,
        heldout_stop_step=20,
        observation_stride=1,
        truth_reset_horizon=3,
        rollout_horizon=5,
    )
    terminal_configuration = ResolvedInferenceConfiguration(
        **common,
        rollout_loss=RolloutLoss.TERMINAL,
    )
    terminal = build_inference_index_plan(terminal_configuration)
    accumulated = build_inference_index_plan(
        ResolvedInferenceConfiguration(
            **common,
            rollout_loss=RolloutLoss.ACCUMULATED,
        )
    )
    assert terminal.rollout_prefixes == (5,)
    assert accumulated.rollout_prefixes == (1, 2, 3, 4, 5)
    assert resolved_truth_state_indices(terminal_configuration) == tuple(range(11))
    assert resolved_truth_state_indices(
        terminal_configuration, include_heldout=True
    ) == tuple(range(21))


def test_reset_window_stride_and_loss_remain_configurable_ablation_controls():
    common = dict(
        truth_run_directory="selected_truth",
        training_stop_step=10,
        heldout_stop_step=20,
        observation_stride=1,
        truth_reset_horizon=2,
        truth_reset_window_stride=2,
        rollout_horizon=4,
        rollout_loss=RolloutLoss.ACCUMULATED,
    )
    terminal = build_inference_index_plan(
        ResolvedInferenceConfiguration(
            **common,
            truth_reset_loss=RolloutLoss.TERMINAL,
        )
    )
    accumulated = build_inference_index_plan(
        ResolvedInferenceConfiguration(
            **common,
            truth_reset_loss=RolloutLoss.ACCUMULATED,
        )
    )
    assert terminal.truth_reset_windows == (
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 10),
    )
    assert terminal.truth_reset_target_steps == (2, 4, 6, 8, 10)
    assert accumulated.truth_reset_target_steps == tuple(range(1, 11))


def test_solver_loss_normalizations_are_explicit_and_distinct():
    initial_guess = ResolvedInferenceConfiguration(
        truth_run_directory="selected_truth",
        training_stop_step=5,
        heldout_stop_step=10,
    )
    shared_target = ResolvedInferenceConfiguration(
        truth_run_directory="selected_truth",
        training_stop_step=5,
        heldout_stop_step=10,
        solver_loss_normalization=SolverLossNormalization.TRUTH_TARGET_MASS,
    )
    assert (
        initial_guess.solver_loss_normalization
        is SolverLossNormalization.INITIAL_GUESS_RESIDUAL
    )
    assert (
        shared_target.solver_loss_normalization
        is SolverLossNormalization.TRUTH_TARGET_MASS
    )
    assert initial_guess.to_dict() != shared_target.to_dict()


def test_incompatible_observation_cadence_and_horizon_is_rejected():
    with pytest.raises(ValueError, match="horizon must be divisible"):
        ResolvedInferenceConfiguration(
            truth_run_directory="selected_truth",
            training_stop_step=10,
            heldout_stop_step=20,
            observation_stride=2,
            truth_reset_horizon=1,
            rollout_horizon=4,
        )


@dataclass(frozen=True)
class _Counts:
    objective_evaluations: int
    gradient_evaluations: int
    hvp_evaluations: int
    solver_calls: int


class _QuadraticObjective:
    c0_scale = C0_SCALE

    def __init__(self):
        self.objective_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.solver_calls = 0

    def counts(self):
        return _Counts(
            self.objective_evaluations,
            self.gradient_evaluations,
            self.hvp_evaluations,
            self.solver_calls,
        )

    def value_and_gradient(self, z):
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        c0 = self.c0_scale * z
        return 0.5 * (c0 - 0.14) ** 2, self.c0_scale * (c0 - 0.14)

    def hess_vec(self, z, direction):
        self.hvp_evaluations += 1
        return self.c0_scale**2 * direction


def test_objective_landscape_records_physical_derivatives_and_costs():
    objective = _QuadraticObjective()
    result = scan_scalar_objective(
        objective,
        ObjectiveScanConfiguration(
            physical_lower=0.10, physical_upper=0.18, points=3
        ),
    )
    center = result[1]
    assert center.physical_c0 == 0.14
    assert center.objective == 0.0
    assert center.physical_gradient == 0.0
    np.testing.assert_allclose(center.physical_hessian, 1.0)
    assert center.objective_evaluations == 1
    assert center.gradient_evaluations == 1
    assert center.hvp_evaluations == 1
    assert center.solver_calls == 0
    assert center.forward_steps == 0
    assert center.reverse_steps == 0
    assert center.tangent_steps == 0
    assert center.incremental_reverse_steps == 0
    assert all(point.finite for point in result)


def test_objective_landscape_restart_skips_completed_points():
    configuration = ObjectiveScanConfiguration(
        physical_lower=0.10, physical_upper=0.18, points=3
    )
    first = scan_scalar_objective(_QuadraticObjective(), configuration)
    resumed_objective = _QuadraticObjective()
    resumed = scan_scalar_objective(
        resumed_objective,
        configuration,
        completed_points=tuple(point.__dict__ for point in first),
    )
    assert resumed == first
    assert resumed_objective.counts() == _Counts(0, 0, 0, 0)


def test_test1a_certified_constants_and_tiny_config_remain_unchanged():
    repository = Path(__file__).resolve().parents[1]
    source = (repository / "dimswe" / "hidden_c0.py").read_text(encoding="utf-8")
    config = (
        repository / "dimswe" / "configs" / "hidden_c0_tiny.cfg"
    ).read_text(encoding="utf-8")
    assert "DEFAULT_TRUTH_C0 = 0.14" in source
    assert "DEFAULT_INITIAL_C0 = 0.07" in source
    assert "C0_SCALE = 0.07" in source
    assert "nx: 2" in config and "ny: 2" in config
    assert "dt: 100.0" in config
    assert "num_steps: 4" in config
    assert "subcycle_list: [2, 1, 2, 1]" in config
