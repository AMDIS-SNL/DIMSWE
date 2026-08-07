"""Cheap, solver-free checks for the selected Test 1B execution plan."""

from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

from dimswe.resolved_hidden_c0 import (
    ScanDerivativeLevel,
    build_inference_index_plan,
    read_json_record,
    resolved_truth_state_indices,
    write_json_record,
)
from dimswe.selected_test1b import (
    DEFAULT_SELECTED_PLAN,
    audit_selected_truth,
    fitting_and_heldout_index_record,
    load_selected_test1b_plan,
)


def _fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_selected_plan_records_exact_scientific_choice_and_pilot_evidence():
    plan, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    assert (
        selected.case,
        selected.nx,
        selected.ny,
        selected.dt,
        selected.nsteps,
        selected.final_time,
    ) == ("doublevortex", 16, 16, 100.0, 160, 16000.0)
    assert (
        selected.c0_truth,
        selected.c0_initial,
        selected.c0_scale,
        selected.s,
    ) == (0.14, 0.07, 0.07, 3.2)
    evidence = plan["pilot_evidence"]
    assert evidence["accepted_16x16_dt100"][
        "maximum_mixed_c0_pair_separation"
    ] == 1.643758942633589e-4
    assert not evidence["accepted_16x16_dt100"][
        "numerical_stability_heuristic_warning"
    ]
    assert evidence["rejected_32x32_dt400"]["sigma"] == 2.5734163936944783
    assert not evidence["run_64x64"]
    assert (
        ScanDerivativeLevel(plan["objective_scan"]["derivative_level"])
        is ScanDerivativeLevel.OBJECTIVE_ONLY
    )


def test_selected_split_and_fitting_data_boundary_are_exact():
    _, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    inference = selected.inference_configuration("selected_truth")
    indices = fitting_and_heldout_index_record(selected)
    assert selected.training_state_indices == tuple(range(81))
    assert selected.heldout_target_state_indices == tuple(range(81, 161))
    assert indices["training_state_indices"] == (0, 80)
    assert indices["training_transition_start_indices"] == (0, 79)
    assert indices["heldout_boundary_initial_state_index"] == 80
    assert indices["heldout_target_state_indices"] == (81, 160)
    assert indices["heldout_transition_start_indices"] == (80, 159)
    assert not indices["heldout_states_available_during_fitting"]
    assert resolved_truth_state_indices(inference) == tuple(range(81))
    assert resolved_truth_state_indices(
        inference, include_heldout=True
    ) == tuple(range(161))


def test_selected_offline_objectives_use_all_training_transitions_once():
    _, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    inference = selected.inference_configuration("selected_truth")
    plan = build_inference_index_plan(inference)
    assert plan.offline_transition_starts == tuple(range(80))
    assert inference.training_observation_steps[:-1] == tuple(range(80))


def test_selected_reset_windows_are_disjoint_and_cover_each_target_once():
    _, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    inference = selected.inference_configuration("selected_truth")
    plan = build_inference_index_plan(inference)
    expected_starts = tuple(range(0, 80, 5))
    assert plan.truth_reset_windows == tuple(
        (start, start + 5) for start in expected_starts
    )
    assert selected.reset_window_starts == expected_starts
    assert selected.reset_window_length == 5
    assert selected.reset_window_stride == 5
    assert plan.truth_reset_target_steps == tuple(range(1, 81))
    assert Counter(plan.truth_reset_target_steps) == Counter(range(1, 81))
    target_sets = [
        set(range(start + 1, stop + 1))
        for start, stop in plan.truth_reset_windows
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(target_sets)
        for right in target_sets[index + 1 :]
    )


def test_selected_rollout_covers_full_training_interval_once():
    _, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    inference = selected.inference_configuration("selected_truth")
    plan = build_inference_index_plan(inference)
    assert plan.rollout_start_step == 0
    assert plan.rollout_prefixes == tuple(range(1, 81))
    assert plan.rollout_target_steps == tuple(range(1, 81))
    assert Counter(plan.rollout_target_steps) == Counter(range(1, 81))
    assert selected.rollout_training_length == 80


def test_selected_reset_and_rollout_have_identical_target_multisets():
    plan_record, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    inference = selected.inference_configuration("selected_truth")
    plan = build_inference_index_plan(inference)
    assert Counter(plan.truth_reset_target_steps) == Counter(
        plan.rollout_target_steps
    )
    assert max(stop - start for start, stop in plan.truth_reset_windows) == 5
    assert max(plan.rollout_prefixes) == 80
    assert inference.truth_reset_loss.value == "accumulated"
    assert inference.rollout_loss.value == "accumulated"
    assert inference.solver_loss_normalization.value == "truth_target_mass"
    weighting = plan_record["canonical_comparison"]["solver_loss_weighting"]
    assert weighting["per_target_outer_weight"].startswith("1/80")
    assert "inner(error,error)" in weighting["state_metric"]
    assert "rollout_horizon" not in plan_record["canonical_comparison"]
    assert "fairness_limitation" not in plan_record["canonical_comparison"]


def test_selected_plan_never_constructs_heldout_fitting_targets():
    _, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    plan = build_inference_index_plan(
        selected.inference_configuration("selected_truth")
    )
    fitting_targets = (
        plan.truth_reset_target_steps + plan.rollout_target_steps
    )
    assert fitting_targets
    assert max(fitting_targets) == 80
    assert set(fitting_targets).isdisjoint(range(81, 161))


def _write_synthetic_selected_truth(directory):
    plan, selected = load_selected_test1b_plan(DEFAULT_SELECTED_PLAN)
    for name in ("restart", "checkpoints", "diagnostics", "spectra"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    pilot = selected.pilot_configuration(directory)
    field_sizes = (2, 2, 2, 1, 1, 1)
    slices = {
        "v": (0, 2),
        "h": (2, 4),
        "S": (4, 6),
        "Qv": (6, 7),
        "Qc": (7, 8),
        "Qr": (8, 9),
    }
    diagnostics = []
    for step in range(161):
        state = np.ones(sum(field_sizes), dtype=np.float64)
        state[2:4] = 750.0
        np.save(directory / "restart" / f"step_{step:08d}.npy", state)
        (directory / "checkpoints" / f"step_{step:08d}.h5").touch()
        (directory / "diagnostics" / f"step_{step:08d}.json").touch()
        (directory / "spectra" / f"step_{step:08d}.npz").touch()
        diagnostics.append(
            {
                "step": step,
                "time": float(100 * step),
                "kinetic_energy": 10.0,
                "projected_enstrophy": 2.0,
                "hyperviscosity_tendency_mass_norm": 3.0,
                "velocity_high_wavenumber_energy_fraction": 1.0e-8,
                "all_state_coefficients_finite": True,
                "minimum_height_coefficient": 750.0,
            }
        )
    configuration = pilot.to_dict()
    metadata = {
        "status": "complete",
        "configuration": configuration,
        "paired_non_c0_physics_sha256": _fingerprint(
            pilot.physics_configuration(include_c0=False)
        ),
        "physical_parameters": {"c0": 0.14, "s": 3.2},
        "moist_backend": "ufl",
        "random_seed": 0,
        "completed_output_steps": tuple(range(161)),
        "time": {
            "dt": 100.0,
            "nsteps": 160,
            "final_time": 16000.0,
            "output_times": tuple(float(100 * step) for step in range(161)),
        },
        "state_convention": {
            "field_sizes": field_sizes,
            "field_slices": slices,
        },
        "diagnostics": diagnostics,
    }
    write_json_record(directory / "metadata.json", metadata)
    return plan


def test_truth_audit_checks_all_161_states_and_rejects_late_growth(tmp_path):
    _write_synthetic_selected_truth(tmp_path)
    audit = audit_selected_truth(tmp_path)
    assert audit["passed"]
    assert audit["state_count"] == 161
    assert audit["snapshot_file_counts"] == {
        "restart": 161,
        "checkpoint": 161,
        "diagnostic": 161,
        "spectrum": 161,
    }
    assert audit["all_states_finite"]
    assert audit["minimum_height_admissible"]
    assert not audit["any_numerical_stability_heuristic_warning"]

    metadata = read_json_record(tmp_path / "metadata.json")
    for record in metadata["diagnostics"][-41:]:
        record["hyperviscosity_tendency_mass_norm"] = 100.0
    write_json_record(tmp_path / "metadata.json", metadata)
    failed = audit_selected_truth(tmp_path)
    assert not failed["passed"]
    assert failed["any_numerical_stability_heuristic_warning"]
    assert any("late-time" in reason for reason in failed["failure_reasons"])
