"""Cheap contracts for Test-1B autonomous post-fit certification."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import dimswe.resolved_hidden_c0_inference as inference
from dimswe.learned_physics.objectives import TrainingMode
from dimswe.resolved_hidden_c0 import ResolvedInferenceConfiguration
from dimswe.test1b_gate4 import (
    STRATEGIES,
    aggregate_gate4_records,
    plot_gate4_summary,
)


def _configuration():
    return ResolvedInferenceConfiguration(
        truth_run_directory="selected_truth",
        training_start_step=0,
        training_stop_step=80,
        heldout_stop_step=160,
        observation_stride=1,
        truth_reset_horizon=5,
        truth_reset_window_stride=5,
        rollout_horizon=80,
        truth_reset_loss="accumulated",
        rollout_loss="accumulated",
        solver_loss_normalization="truth_target_mass",
    )


def _fit_record(mode=TrainingMode.ROLLOUT, *, success=True):
    configuration = _configuration()
    return {
        "status": "complete",
        "intent": {
            "command": "fit",
            "configuration": configuration.to_dict(),
            "training_mode": mode.value,
        },
        "result": {
            "success": success,
            "failure_reason": None if success else "iteration limit reached",
            "starting_c0": 0.07,
            "recovered_c0": 0.14,
            "counts": {
                "objective_evaluations": 3,
                "gradient_evaluations": 2,
                "hvp_evaluations": 1,
            },
            "wall_time_seconds": 1.25,
        },
        "fit_summary": {
            "recovered_c0": 0.14,
            "accepted_optimizer_steps": 1,
        },
        "termination_claim": "converged" if success else "not converged",
    }


def test_fit_validation_requires_complete_success_and_returns_recovered_c0():
    configuration = _configuration()
    mode, starting, recovered = inference._validated_fit_result(
        _fit_record(), configuration
    )
    assert mode is TrainingMode.ROLLOUT
    assert starting == 0.07
    assert recovered == 0.14

    unsuccessful = _fit_record(success=False)
    with pytest.raises(ValueError, match="successful fit"):
        inference._validated_fit_result(unsuccessful, configuration)

    incomplete = _fit_record()
    incomplete["status"] = "running"
    with pytest.raises(ValueError, match="complete fit"):
        inference._validated_fit_result(incomplete, configuration)


def test_relative_error_zero_reference_is_explicit():
    both_zero = inference._relative_error_record(0.0, 0.0)
    assert both_zero["relative_error"] == 0.0
    assert both_zero["reference_norm_zero"]
    assert both_zero["relative_error_defined"]

    undefined = inference._relative_error_record(4.0, 0.0)
    assert undefined["absolute_error"] == 2.0
    assert undefined["relative_error"] is None
    assert undefined["reference_norm_zero"]
    assert not undefined["relative_error_defined"]


def test_diagnostic_mismatch_records_absolute_relative_and_extrema():
    result = inference._diagnostic_mismatch(
        np.array([2.0, 4.0]),
        np.array([1.0, 2.0]),
        (81, 82),
        (8100.0, 8200.0),
    )
    assert result["absolute_mismatch"] == (1.0, 2.0)
    assert result["relative_mismatch"] == (1.0, 1.0)
    assert result["maximum_absolute_mismatch"] == 2.0
    assert result["final_relative_mismatch"] == 1.0
    assert result["relative_history_l2_mismatch"] == 1.0


def test_heldout_autonomous_map_advances_once_from_state_80(monkeypatch):
    calls = []
    trusted_x80 = object()

    def fake_advance(case, initial, c0, nsteps, *, start_time, prefix):
        calls.append((initial, c0, nsteps, start_time, prefix))
        return tuple(f"predicted-{index}" for index in range(nsteps + 1))

    monkeypatch.setattr(inference, "_advance", fake_advance)
    case = type("Case", (), {"t0": 0.0, "dt": 100.0})()
    states = inference._autonomous_map(
        case, trusted_x80, 0.14, 80, 160, "gate4_test"
    )
    assert calls == [(trusted_x80, 0.14, 80, 8000.0, "gate4_test")]
    assert tuple(states) == tuple(range(80, 161))
    assert states[81] == "predicted-1"
    assert states[160] == "predicted-80"


def _gate4_record(strategy):
    times = tuple(float(step * 100) for step in range(81, 161))
    zeros = (0.0,) * 80
    diagnostic = {
        "truth": (1.0,) * 80,
        "predicted": (1.0,) * 80,
        "maximum_absolute_mismatch": 0.0,
        "final_absolute_mismatch": 0.0,
        "maximum_relative_mismatch": 0.0,
        "final_relative_mismatch": 0.0,
    }
    return {
        "status": "complete",
        "fitted_training_mode": strategy,
        "fit_provenance": {
            "success": True,
            "starting_c0": 0.07,
            "recovered_c0": 0.14,
        },
        "evaluation": {
            "heldout_autonomous_trajectory_error": {
                "times": times,
                "relative_mass_norm_error": zeros,
                "maximum": 0.0,
                "final": 0.0,
            },
            "kinetic_energy_mismatch": deepcopy(diagnostic),
            "projected_enstrophy_mismatch": deepcopy(diagnostic),
            "heldout_deployment_contract": {"complete_production_steps": 80},
            "wall_time_seconds": 12.0,
            "gate4_certification": {"passed": True},
        },
    }


def test_gate4_aggregate_requires_four_passed_strategies_and_is_plot_ready():
    records = {strategy: _gate4_record(strategy) for strategy in STRATEGIES}
    summary = aggregate_gate4_records(records)
    assert summary["status"] == "complete"
    assert tuple(row["strategy"] for row in summary["strategies"]) == STRATEGIES
    assert all(row["complete_solver_steps"] == 80 for row in summary["strategies"])
    assert all(row["passed"] for row in summary["strategies"])
    assert set(summary["plot_data"]) == set(STRATEGIES)

    failed = deepcopy(records)
    failed[STRATEGIES[0]]["evaluation"]["gate4_certification"]["passed"] = False
    with pytest.raises(ValueError, match="did not pass"):
        aggregate_gate4_records(failed)


def test_gate4_summary_writes_three_requested_plots(tmp_path):
    pytest.importorskip("matplotlib")
    records = {strategy: _gate4_record(strategy) for strategy in STRATEGIES}
    plot_gate4_summary(aggregate_gate4_records(records), tmp_path)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "gate4_mixed_state_error.png",
        "gate4_kinetic_energy.png",
        "gate4_projected_enstrophy.png",
    }
