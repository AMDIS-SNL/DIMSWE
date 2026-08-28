"""Final evaluation-only postprocessor for Test2B Representation B.

This module has no optimizer entry point.  It verifies immutable completed
parameter artifacts, evaluates the frozen objective ladder, and performs the
common 160-step autonomous rollout with both A and R learned on the current
model state.  Shared state, mass, flow, and trajectory routines are imported
from the certified Representation-A postprocessor.
"""

from __future__ import annotations

import argparse
from gc import collect
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from .hidden_c0 import STATE_FIELDS, _copy_function
from .jax_moist import moist_rates_jax
from .learned_physics.parameters import tree_copy
from .resolved_hidden_c0 import ResolvedPilotConfiguration, write_json_record
from .resolved_hidden_c0_driver import ResolvedDiagnosticEvaluator
from .resolved_hidden_c0_inference import (
    _diagnostic_mismatch,
    _field_squared_norms,
    _state_squared_difference,
)
from .test2a_apriori_autonomous import rain_activity_diagnostic
from .test2a_problem_b_campaign import ProblemBDiagnosticConfiguration
from .test2b_rain_learning import (
    build_model,
    load_parameters,
    source_invariant_diagnostics,
)
from .test2b_rain_learning_campaign import (
    FixedObjective,
    OperatorObjective,
    build_neural_case,
    load_configuration,
    load_preparation,
)
from .test2b_representation_a_postprocess import (
    DIRECTORIES,
    EXPECTED_STAGES,
    LABELS,
    REGIMES,
    _field_integral,
    _file_sha256,
    _make_trajectory_objective,
    _relative_record,
    _summarize_boundary,
    _weighted_metrics,
)


REPRESENTATION = "B"
SIGMA_A = 9.052258655848717e-8
SIGMA_R = 1.9902871261559996e-11


def _verify_artifacts(root):
    verified = {}
    parameters = {}
    for label in LABELS:
        directory = Path(root) / DIRECTORIES[label]
        result_path = directory / "fit_result.json"
        progress_path = directory / "fit_progress.json"
        parameter_path = directory / "final_parameters.npz"
        if not all(
            path.is_file()
            for path in (result_path, progress_path, parameter_path)
        ):
            raise FileNotFoundError(f"incomplete final artifact {label}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if result != progress or result.get("status") != "complete":
            raise RuntimeError(f"{label} final progress/result mismatch")
        if result.get("representation") != REPRESENTATION:
            raise ValueError(f"{label} representation identity changed")
        if result.get("stage") != EXPECTED_STAGES[label]:
            raise ValueError(f"{label} stage identity changed")
        if "MAXITER" not in result.get("termination_reason", ""):
            raise ValueError(f"{label} did not terminate at its recorded cap")
        loaded, sidecar = load_parameters(parameter_path, REPRESENTATION)
        if (
            sidecar["parameter_pytree_sha256"]
            != result["final_parameter_pytree_sha256"]
        ):
            raise ValueError(f"{label} parameter fingerprint mismatch")
        if sidecar["metadata"] != {
            "stage": EXPECTED_STAGES[label],
            "accepted_iteration": result["accepted_iterations"],
        }:
            raise ValueError(f"{label} parameter provenance mismatch")
        parameters[label] = tree_copy(loaded)
        verified[label] = {
            **result,
            "parameter_npz_sha256": _file_sha256(parameter_path),
            "fit_result_sha256": _file_sha256(result_path),
            "fit_progress_matches_result": True,
        }
    return verified, parameters


def _predict_rates(parameters, normalized_features, normalization):
    scaled = np.asarray(
        build_model(REPRESENTATION)(
            parameters, jnp.asarray(normalized_features)
        ),
        dtype=np.float64,
    ) * normalization.output_scales(REPRESENTATION)
    return scaled[..., 0], scaled[..., 1]


def _truth_rate_arrays(case, truth, normalization):
    helper = case.helper.moist_helper
    primal = helper.primal_helper
    _, topography = primal.interpolate_and_pack(
        primal.term.B, "test2b_representation_b_reference_B"
    )
    fields = {"B": topography}
    parameters = primal._parameters(None)
    result = {
        name: [] for name in ("features", "A", "R", "h", "Qr")
    }
    for step in range(161):
        packed = helper.state_interpolation(truth[step])
        rates = moist_rates_jax(packed, fields, parameters)
        physical = np.stack(
            tuple(
                np.asarray(packed[name]).reshape(-1)
                for name in ("h", "S", "Qv", "Qc")
            )
            + (np.asarray(topography).reshape(-1),),
            axis=-1,
        )
        _, qr = primal.interpolate_and_pack(
            truth[step].sub(5), f"test2b_representation_b_truth_Qr_{step}"
        )
        result["features"].append(
            np.asarray(
                normalization.normalize_features(physical), dtype=np.float64
            )
        )
        result["A"].append(
            np.asarray(rates["A"], dtype=np.float64).reshape(-1)
        )
        result["R"].append(
            np.asarray(rates["R"], dtype=np.float64).reshape(-1)
        )
        result["h"].append(
            np.asarray(packed["h"], dtype=np.float64).reshape(-1)
        )
        result["Qr"].append(np.asarray(qr, dtype=np.float64).reshape(-1))
    return {name: np.stack(values) for name, values in result.items()}


def _activity_masks(
    rate, h, qr, comparison_rate_scale, *, dt=100.0,
):
    rate = np.asarray(rate, dtype=np.float64)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), rate.shape)
    qr = np.broadcast_to(np.asarray(qr, dtype=np.float64), rate.shape)
    numerical_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(float(comparison_rate_scale), np.finfo(np.float64).tiny)
    )
    qr_rms = np.sqrt(np.mean(qr * qr, axis=-1, keepdims=True))
    physical_tolerance = 1.0e-12 * qr_rms
    increments = float(dt) * h * rate
    meaningful_positive = (rate > numerical_tolerance) & (
        increments > physical_tolerance
    )
    meaningful_negative = (rate < -numerical_tolerance) & (
        -increments > physical_tolerance
    )
    return {
        "exact_positive": rate > 0.0,
        "exact_negative": rate < 0.0,
        "meaningful_positive": meaningful_positive,
        "meaningful_negative": meaningful_negative,
        "numerical_tolerance": float(numerical_tolerance),
    }


def _masked_metrics(prediction, target, weights, mask, scale):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    broadcast = np.broadcast_to(np.asarray(weights), np.asarray(target).shape)
    return _weighted_metrics(
        np.asarray(prediction)[mask],
        np.asarray(target)[mask],
        broadcast[mask],
        scale,
    )


def _r_metrics(
    prediction, target, weights, h, qr, comparison_rate_scale,
):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    truth_masks = _activity_masks(
        target, h, qr, comparison_rate_scale
    )
    predicted_masks = _activity_masks(
        prediction, h, qr, comparison_rate_scale
    )
    truth_active = truth_masks["meaningful_positive"]
    predicted_active = predicted_masks["meaningful_positive"]
    predicted_negative = predicted_masks["meaningful_negative"]
    truth_inactive = ~truth_active
    false_positive = predicted_active & truth_inactive
    false_negative = truth_active & ~predicted_active
    active_count = int(np.count_nonzero(truth_active))
    inactive_count = int(np.count_nonzero(truth_inactive))
    return {
        "all_samples": _weighted_metrics(
            prediction, target, weights, SIGMA_R
        ),
        "truth_active_samples": _masked_metrics(
            prediction, target, weights, truth_active, SIGMA_R
        ),
        "truth_active_sample_count": active_count,
        "truth_active_sample_fraction": float(np.mean(truth_active)),
        "predicted_meaningful_positive_fraction": float(
            np.mean(predicted_active)
        ),
        "predicted_meaningful_negative_fraction": float(
            np.mean(predicted_negative)
        ),
        "predicted_exact_positive_fraction": float(
            np.mean(predicted_masks["exact_positive"])
        ),
        "predicted_exact_negative_fraction": float(
            np.mean(predicted_masks["exact_negative"])
        ),
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_fraction_all_samples": float(
            np.mean(false_positive)
        ),
        "false_positive_rate_given_truth_inactive": None
        if inactive_count == 0
        else float(np.count_nonzero(false_positive) / inactive_count),
        "false_negative_count": int(np.count_nonzero(false_negative)),
        "false_negative_fraction_all_samples": float(
            np.mean(false_negative)
        ),
        "false_negative_rate_given_truth_active": None
        if active_count == 0
        else float(np.count_nonzero(false_negative) / active_count),
        "maximum_predicted_R": float(np.max(prediction)),
        "minimum_predicted_R": float(np.min(prediction)),
        "activity_numerical_tolerance": truth_masks[
            "numerical_tolerance"
        ],
        "sample_count": int(target.size),
    }


def _first_activation_step(mask):
    active_by_step = np.any(np.asarray(mask), axis=1)
    locations = np.flatnonzero(active_by_step)
    return None if locations.size == 0 else int(locations[0])


def _direct_metrics(
    parameters, arrays, weights, normalization, comparison_rate_scale,
):
    predicted = [
        _predict_rates(parameters, feature, normalization)
        for feature in arrays["features"]
    ]
    prediction_a = np.stack([row[0] for row in predicted])
    prediction_r = np.stack([row[1] for row in predicted])
    selections = {
        "TRAINING_OVERALL": tuple(range(0, 81)),
        **REGIMES,
    }
    a_result = {}
    r_result = {}
    for name, steps in selections.items():
        index = np.asarray(steps)
        local_weights = np.broadcast_to(
            weights, arrays["A"][index].shape
        )
        a_result[name] = _weighted_metrics(
            prediction_a[index], arrays["A"][index], local_weights, SIGMA_A
        )
        r_result[name] = _r_metrics(
            prediction_r[index], arrays["R"][index], local_weights,
            arrays["h"][index], arrays["Qr"][index],
            comparison_rate_scale,
        )
        a_result[name]["state_indices"] = [int(steps[0]), int(steps[-1])]
        r_result[name]["state_indices"] = [int(steps[0]), int(steps[-1])]

    predicted_masks = _activity_masks(
        prediction_r, arrays["h"], arrays["Qr"], comparison_rate_scale
    )
    truth_masks = _activity_masks(
        arrays["R"], arrays["h"], arrays["Qr"], comparison_rate_scale
    )
    activation = {
        "first_exact_positive_predicted_R_step": _first_activation_step(
            predicted_masks["exact_positive"]
        ),
        "first_meaningful_positive_predicted_R_step": _first_activation_step(
            predicted_masks["meaningful_positive"]
        ),
        "first_meaningful_positive_truth_R_step": _first_activation_step(
            truth_masks["meaningful_positive"]
        ),
    }
    for key, value in tuple(activation.items()):
        if key.endswith("_step"):
            activation[key.removesuffix("_step") + "_time"] = (
                None if value is None else float(100.0 * value)
            )
    return a_result, r_result, activation


def _source_record(
    case, moist, step, weights, comparison_rate_scale, normalization,
):
    primal = case.helper.moist_helper.primal_helper
    packed = moist.packed_state
    fields = moist.packed_fields
    analytical = moist_rates_jax(packed, fields, moist.parameters)
    predicted_a = np.asarray(moist.rates["A"], dtype=np.float64).reshape(-1)
    analytical_a = np.asarray(analytical["A"], dtype=np.float64).reshape(-1)
    predicted_r = np.asarray(moist.rates["R"], dtype=np.float64).reshape(-1)
    analytical_r = np.asarray(analytical["R"], dtype=np.float64).reshape(-1)
    h = np.asarray(packed["h"], dtype=np.float64).reshape(-1)
    qc = np.asarray(packed["Qc"], dtype=np.float64).reshape(-1) / h
    _, qr = primal.interpolate_and_pack(
        moist.stage_state.sub(5), f"test2b_representation_b_Qr_{step}"
    )
    qr = np.asarray(qr, dtype=np.float64).reshape(-1)
    rain = rain_activity_diagnostic(
        predicted_r, h, qr, case.dt, comparison_rate_scale,
        float64_scale_multiplier=64.0,
        physical_increment_relative_threshold=1.0e-12,
    )
    predicted_masks = _activity_masks(
        predicted_r[None, :], h[None, :], qr[None, :],
        comparison_rate_scale,
    )
    truth_masks = _activity_masks(
        analytical_r[None, :], h[None, :], qr[None, :],
        comparison_rate_scale,
    )
    predicted_positive = predicted_masks["meaningful_positive"].reshape(-1)
    predicted_negative = predicted_masks["meaningful_negative"].reshape(-1)
    truth_active = truth_masks["meaningful_positive"].reshape(-1)
    result = {
        "step": int(step),
        "time": float(case.t0 + step * case.dt),
        "applied_to_trajectory": bool(step < 160),
        "specific_Qc_maximum": float(np.max(qc)),
        "specific_Qc_rms": float(np.sqrt(np.mean(qc * qc))),
        "A_metrics": _weighted_metrics(
            predicted_a, analytical_a, weights, normalization.sigma_a
        ),
        "R_metrics": _r_metrics(
            predicted_r, analytical_r, weights, h, qr,
            comparison_rate_scale,
        ),
        "R_maximum_positive": float(np.max(predicted_r)),
        "R_minimum": float(np.min(predicted_r)),
        "R_maximum_absolute": rain["maximum_absolute_R"],
        "R_rms": rain["rms_R"],
        "R_exact_positive_fraction": float(np.mean(predicted_r > 0.0)),
        "R_exact_negative_fraction": float(np.mean(predicted_r < 0.0)),
        "meaningful_positive_R_fraction": float(
            np.mean(predicted_positive)
        ),
        "meaningful_negative_R_fraction": float(
            np.mean(predicted_negative)
        ),
        "truth_meaningful_R_fraction": float(np.mean(truth_active)),
        "false_positive_R_count": int(
            np.count_nonzero(predicted_positive & ~truth_active)
        ),
        "false_negative_R_count": int(
            np.count_nonzero(truth_active & ~predicted_positive)
        ),
        "truth_active_R_count": int(np.count_nonzero(truth_active)),
        "sample_count": int(predicted_r.size),
        "maximum_absolute_Qr_increment": rain[
            "maximum_absolute_Qr_increment"
        ],
        "rain_source_mass_rate": _field_integral(case, moist.tendency.sub(5)),
        "source_invariants": source_invariant_diagnostics(
            moist.source_density, 98.0616
        ),
    }
    result["rain_source_mass_increment"] = (
        case.dt * result["rain_source_mass_rate"]
    )
    return result


def _summarize_rain(records, boundary, truth_audit):
    exact = [row for row in records if row["R_maximum_positive"] > 0.0]
    meaningful = [
        row for row in records
        if row["meaningful_positive_R_fraction"] > 0.0
    ]
    negative = [
        row for row in records
        if row["meaningful_negative_R_fraction"] > 0.0
    ]
    peak_qc = max(records, key=lambda row: row["specific_Qc_maximum"])
    integrated = sum(
        row["rain_source_mass_increment"] for row in records
        if row["applied_to_trajectory"]
    )
    qr = np.asarray(boundary["Qr_mass"], dtype=np.float64)
    truth_qr = np.asarray(
        [row["rain_water_mass"] for row in truth_audit["records"]],
        dtype=np.float64,
    )
    truth_summary = truth_audit["summary"]
    exact_step = None if not exact else exact[0]["step"]
    meaningful_step = None if not meaningful else meaningful[0]["step"]
    negative_step = None if not negative else negative[0]["step"]
    pre = [row for row in records if row["step"] <= 50]
    pre_count = sum(row["false_positive_R_count"] for row in pre)
    pre_samples = sum(row["sample_count"] for row in pre)
    active_count = sum(row["truth_active_R_count"] for row in records)
    false_negative_count = sum(
        row["false_negative_R_count"] for row in records
    )
    return {
        "first_exact_positive_R_step": exact_step,
        "first_exact_positive_R_time": None
        if exact_step is None else float(exact_step * 100.0),
        "first_physically_meaningful_positive_R_step": meaningful_step,
        "first_physically_meaningful_positive_R_time": None
        if meaningful_step is None else float(meaningful_step * 100.0),
        "meaningful_onset_time_error_vs_truth": None
        if meaningful_step is None else float(meaningful_step * 100.0 - 5100.0),
        "first_physically_meaningful_negative_R_step": negative_step,
        "first_physically_meaningful_negative_R_time": None
        if negative_step is None else float(negative_step * 100.0),
        "pre_truth_onset_false_positive_R_count": int(pre_count),
        "pre_truth_onset_false_positive_R_fraction": float(
            pre_count / pre_samples
        ),
        "false_negative_active_R_count": int(false_negative_count),
        "truth_active_R_count": int(active_count),
        "false_negative_rate_given_truth_active": None
        if active_count == 0 else float(false_negative_count / active_count),
        "maximum_specific_Qc": peak_qc["specific_Qc_maximum"],
        "maximum_specific_Qc_step": peak_qc["step"],
        "maximum_specific_Qc_time": peak_qc["time"],
        "maximum_specific_Qc_error": peak_qc["specific_Qc_maximum"]
        - truth_summary["maximum_specific_Qc"],
        "maximum_R": float(max(row["R_maximum_positive"] for row in records)),
        "minimum_R": float(min(row["R_minimum"] for row in records)),
        "maximum_R_error": float(
            max(row["R_maximum_positive"] for row in records)
            - truth_summary["maximum_R"]
        ),
        "time_integrated_rain_source_mass": float(integrated),
        "time_integrated_rain_source_mass_error": float(
            integrated - truth_summary["time_integrated_rain_source_mass"]
        ),
        "final_Qr_mass": float(qr[-1]),
        "maximum_Qr_mass": float(np.max(qr)),
        "final_Qr_mass_error": float(qr[-1] - truth_qr[-1]),
        "maximum_Qr_mass_error": float(
            np.max(qr) - truth_summary["maximum_rain_water_mass"]
        ),
        "maximum_absolute_Qr_mass_error": float(
            np.max(np.abs(qr - truth_qr))
        ),
        "records": records,
    }


def _evaluate_rollout(
    case, truth, parameters, weights, diagnostic_configuration,
    truth_diagnostics, truth_audit, label, normalization,
):
    evaluator = ResolvedDiagnosticEvaluator(case, diagnostic_configuration)
    current = _copy_function(truth[0], f"test2b_B_{label}_autonomous_0")
    zero = case.new_state(f"test2b_B_{label}_zero")
    zero.assign(0)
    mixed_records = {}
    field_records = {name: {} for name in STATE_FIELDS}
    predicted_diagnostics = []
    boundary = {
        "total_water_mass": [], "S_minus_beta2_Qv_mass": [],
        "Qv_mass": [], "Qc_mass": [], "Qr_mass": [],
        "minimum_field_coefficients": {name: [] for name in STATE_FIELDS},
        "all_state_coefficients_finite": [],
    }
    source_records = []
    rate_accumulators = {
        rate: {
            name: {"prediction": [], "target": [], "h": [], "Qr": []}
            for name in (*REGIMES, "ALL")
        }
        for rate in ("A", "R")
    }
    maximum_source_residuals = {
        "water_maximum_absolute": 0.0,
        "S_minus_beta2_Qv_maximum_absolute": 0.0,
    }
    comparison_scale = float(
        truth_audit["activity_tolerance"]["comparison_rate_scale"]
    )
    for step in range(161):
        target = truth[step]
        numerator = _state_squared_difference(
            case, current, target, f"test2b_B_{label}_mixed_residual_{step}"
        )
        denominator = _state_squared_difference(
            case, target, zero, f"test2b_B_{label}_mixed_target_{step}"
        )
        mixed_records[step] = _relative_record(numerator, denominator)
        for index, name in enumerate(STATE_FIELDS):
            field_numerator, field_denominator = _field_squared_norms(
                case, current, target, index,
                f"test2b_B_{label}_{name}_{step}",
            )
            field_records[name][step] = (
                None if field_denominator <= np.finfo(np.float64).tiny
                else float(np.sqrt(field_numerator / field_denominator))
            )
            boundary["minimum_field_coefficients"][name].append(
                float(np.min(current.dat.data[index]))
            )
        predicted_diagnostics.append(
            evaluator.evaluate(current, step, step * case.dt)[0]
        )
        boundary["all_state_coefficients_finite"].append(
            bool(
                all(
                    np.all(np.isfinite(current.dat.data[index]))
                    for index in range(len(STATE_FIELDS))
                )
            )
        )
        qv_mass = _field_integral(case, current.sub(3))
        qc_mass = _field_integral(case, current.sub(4))
        qr_mass = _field_integral(case, current.sub(5))
        boundary["Qv_mass"].append(qv_mass)
        boundary["Qc_mass"].append(qc_mass)
        boundary["Qr_mass"].append(qr_mass)
        boundary["total_water_mass"].append(qv_mass + qc_mass + qr_mass)
        boundary["S_minus_beta2_Qv_mass"].append(
            _field_integral(case, current.sub(2)) - 98.0616 * qv_mass
        )

        if step < 160:
            cache = case.helper.take_forward_step_cached(
                current, step * case.dt, case.dt,
                neural_parameters=parameters,
            )
            moist = cache.children[-1].cache
            next_state = _copy_function(
                cache.state_out, f"test2b_B_{label}_autonomous_{step + 1}"
            )
        else:
            prefix = case.helper.take_fixed_prefix_cached(
                current, step * case.dt, case.dt
            )
            moist = case.helper.moist_helper.take_forward_step_cached(
                prefix.state_out, step * case.dt, case.dt,
                neural_parameters=parameters,
            )
            next_state = None
        record = _source_record(
            case, moist, step, weights, comparison_scale, normalization
        )
        source_records.append(record)
        for name in maximum_source_residuals:
            maximum_source_residuals[name] = max(
                maximum_source_residuals[name],
                record["source_invariants"][name],
            )
        if step < 160:
            target_step = step + 1
            regime = next(
                name for name, steps in REGIMES.items()
                if target_step in steps
            )
            analytical = moist_rates_jax(
                moist.packed_state, moist.packed_fields, moist.parameters
            )
            h = np.asarray(moist.packed_state["h"]).reshape(-1)
            _, qr = case.helper.moist_helper.primal_helper.interpolate_and_pack(
                moist.stage_state.sub(5),
                f"test2b_B_{label}_rate_Qr_{step}",
            )
            for rate in ("A", "R"):
                prediction = np.asarray(moist.rates[rate]).reshape(-1)
                target_rate = np.asarray(analytical[rate]).reshape(-1)
                for name in (regime, "ALL"):
                    accumulator = rate_accumulators[rate][name]
                    accumulator["prediction"].append(prediction)
                    accumulator["target"].append(target_rate)
                    accumulator["h"].append(h)
                    accumulator["Qr"].append(np.asarray(qr).reshape(-1))
            current = next_state

    mixed, fieldwise, conservation = _summarize_boundary(
        case, boundary, mixed_records, field_records
    )
    steps = tuple(range(161))
    times = tuple(float(step * case.dt) for step in steps)
    flow = {
        key: _diagnostic_mismatch(
            [row[key] for row in predicted_diagnostics],
            [row[key] for row in truth_diagnostics],
            steps, times,
        )
        for key in (
            "kinetic_energy", "projected_enstrophy",
            "velocity_high_wavenumber_energy_fraction",
        )
    }
    a_metrics = {}
    r_metrics = {}
    for name in (*REGIMES, "ALL"):
        a_values = rate_accumulators["A"][name]
        a_target = np.stack(a_values["target"])
        a_metrics[name] = _weighted_metrics(
            np.stack(a_values["prediction"]), a_target,
            np.broadcast_to(weights, a_target.shape), normalization.sigma_a,
        )
        r_values = rate_accumulators["R"][name]
        r_target = np.stack(r_values["target"])
        r_metrics[name] = _r_metrics(
            np.stack(r_values["prediction"]), r_target,
            np.broadcast_to(weights, r_target.shape),
            np.stack(r_values["h"]), np.stack(r_values["Qr"]),
            comparison_scale,
        )
    return {
        "mixed_state_error": mixed,
        "fieldwise_state_error": fieldwise,
        "rain": _summarize_rain(source_records, boundary, truth_audit),
        "A_error_on_model_postprefix_states": a_metrics,
        "R_error_on_model_postprefix_states": r_metrics,
        "conservation_and_stability": {
            **conservation,
            "maximum_source_residuals": maximum_source_residuals,
        },
        "flow": flow,
        "boundary_timeseries": boundary,
    }


def postprocess(configuration_path, preparation_path, artifact_root, output_path):
    started = perf_counter()
    configuration = load_configuration(configuration_path)
    metadata, normalization, data, matrices = load_preparation(preparation_path)
    verified, parameters = _verify_artifacts(artifact_root)
    weights = np.asarray(data["carrier_weights"], dtype=np.float64)
    repeated_weights = np.broadcast_to(weights, data["x_A"].shape)
    fixed = {
        "J_M1": OperatorObjective(
            REPRESENTATION, data["x_features"], data["x_A"], data["x_R"],
            repeated_weights, normalization,
        ),
        "J_M2_X": FixedObjective(
            REPRESENTATION, data["x_features"], data["x_A"], data["x_R"],
            matrices, metadata["m2x_denominator"], normalization,
        ),
        "J_H1": FixedObjective(
            REPRESENTATION, data["y_features"], data["y_A"], data["y_R"],
            matrices, metadata["common_horizon_denominator"] / 10000.0,
            normalization,
        ),
    }
    case, truth, _ = build_neural_case(
        configuration, normalization, REPRESENTATION, parameters["M1"], 160
    )
    truth_arrays = _truth_rate_arrays(case, truth, normalization)
    truth_root = Path(configuration["truth"]["run_directory"])
    truth_metadata = json.loads(
        (truth_root / "metadata.json").read_text(encoding="utf-8")
    )
    truth_audit = json.loads(
        (truth_root / "rain_activity_audit.json").read_text(encoding="utf-8")
    )
    truth_diagnostics = [
        json.loads(
            (truth_root / "diagnostics" / f"step_{step:08d}.json").read_text(
                encoding="utf-8"
            )
        )
        for step in range(161)
    ]
    pilot = ResolvedPilotConfiguration(**{
        **truth_metadata["configuration"],
        "output_directory": "/tmp/test2b-representation-b-postprocess-no-output",
    })
    diagnostic_configuration = ProblemBDiagnosticConfiguration.from_resolved_pilot(
        pilot
    )
    comparison_scale = float(
        truth_audit["activity_tolerance"]["comparison_rate_scale"]
    )
    result = {
        "status": "in_progress",
        "representation": REPRESENTATION,
        "evaluation_only": True,
        "optimizer_instantiated": False,
        "truth_generated": False,
        "configuration": str(Path(configuration_path).resolve()),
        "preparation": {
            "path": str(Path(preparation_path).resolve()),
            "sha256": _file_sha256(preparation_path),
            "truth_states": [0, 80],
            "heldout_states": [81, 160],
        },
        "artifacts": verified,
        "objective_matrix": {},
        "direct_A": {},
        "direct_R": {},
        "direct_R_activation_on_truth_states": {},
        "autonomous": {},
    }
    for label in LABELS:
        print(f"Representation B evaluation: {label} fixed/direct", flush=True)
        local = parameters[label]
        result["objective_matrix"][label] = {
            name: objective.value(local) for name, objective in fixed.items()
        }
        direct_a, direct_r, activation = _direct_metrics(
            local, truth_arrays, weights, normalization, comparison_scale
        )
        result["direct_A"][label] = direct_a
        result["direct_R"][label] = direct_r
        result["direct_R_activation_on_truth_states"][label] = activation
    del fixed, repeated_weights, data, matrices, truth_arrays
    collect()
    for label in LABELS:
        local = parameters[label]
        for horizon in (2, 5):
            print(
                f"Representation B evaluation: {label} H{horizon} objective",
                flush=True,
            )
            objective = _make_trajectory_objective(
                case, truth, metadata, horizon
            )
            result["objective_matrix"][label][f"J_H{horizon}"] = (
                objective.value(local)
            )
            objective.clear_parameter_tape()
            del objective
            collect()
        print(f"Representation B evaluation: {label} autonomous", flush=True)
        result["autonomous"][label] = _evaluate_rollout(
            case, truth, local, weights, diagnostic_configuration,
            truth_diagnostics, truth_audit, label, normalization,
        )
        print(f"Representation B evaluation: {label} complete", flush=True)
    result["truth_reference"] = {
        "configuration_sha256": truth_metadata["configuration_sha256"],
        "rain_audit_sha256": _file_sha256(
            truth_root / "rain_activity_audit.json"
        ),
        "rain_summary": truth_audit["summary"],
        "activity_tolerance": truth_audit["activity_tolerance"],
        "training_steps": [0, 80],
        "heldout_steps": [81, 160],
    }
    result["total_optimization_accounting"] = {
        "accepted_iterations": int(sum(
            row["accepted_iterations"] for row in verified.values()
        )),
        "objective_evaluations": int(sum(
            row["objective_evaluations"] for row in verified.values()
        )),
        "gradient_evaluations": int(sum(
            row["gradient_evaluations"] for row in verified.values()
        )),
        "wall_seconds": float(sum(
            row["wall_seconds"] for row in verified.values()
        )),
        "all_termination_reasons": sorted(set(
            row["termination_reason"] for row in verified.values()
        )),
        "stationary_optimum_demonstrated": False,
    }
    result["evaluation_wall_seconds"] = float(perf_counter() - started)
    result["status"] = "complete"
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    write_json_record(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    postprocess(
        args.configuration, args.preparation, args.artifact_root, args.output
    )


if __name__ == "__main__":
    main()
