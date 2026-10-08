"""Final evaluation-only postprocessor for Test2B Representation C.

This module has no optimizer entry point.  It verifies immutable completed
parameter artifacts, evaluates the frozen objective ladder, and performs the
common 160-step autonomous rollout with four unconstrained neural moist
tendencies.  Projection onto the physical two-rate manifold is diagnostic
only and is never applied to a trajectory.
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
from .test2a_problem_b_campaign import ProblemBDiagnosticConfiguration
from .test2b_rain_learning import (
    SOURCE_ORDER,
    build_model,
    canonical_sha256,
    load_parameters,
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
from .test2b_representation_b_postprocess import (
    SIGMA_A,
    SIGMA_R,
    _activity_masks,
    _first_activation_step,
    _r_metrics,
)


REPRESENTATION = "C"
BETA2 = 98.0616


def _verify_artifacts(root):
    verified = {}
    parameters = {}
    for label in LABELS:
        directory = Path(root) / DIRECTORIES[label]
        result_path = directory / "fit_result.json"
        progress_path = directory / "fit_progress.json"
        parameter_path = directory / "final_parameters.npz"
        sidecar_path = parameter_path.with_suffix(".json")
        paths = (result_path, progress_path, parameter_path, sidecar_path)
        if not all(path.is_file() for path in paths):
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
        if sidecar["parameter_pytree_sha256"] != result[
            "final_parameter_pytree_sha256"
        ]:
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
            "parameter_sidecar_sha256": _file_sha256(sidecar_path),
            "fit_result_sha256": _file_sha256(result_path),
            "fit_progress_matches_result": True,
            "completion_provenance_uses_log": False,
        }
    return verified, parameters


def _predict_source(parameters, normalized_features, normalization):
    return np.asarray(
        build_model(REPRESENTATION)(
            parameters, jnp.asarray(normalized_features)
        ),
        dtype=np.float64,
    ) * normalization.output_scales(REPRESENTATION)


def _truth_source(a, r, h):
    return np.stack(
        (h * BETA2 * a, h * a, -h * (a + r), h * r), axis=-1
    )


def _truth_arrays(case, truth, normalization):
    helper = case.helper.moist_helper
    primal = helper.primal_helper
    _, topography = primal.interpolate_and_pack(
        primal.term.B, "test2b_representation_c_reference_B"
    )
    fields = {"B": topography}
    moist_parameters = primal._parameters(None)
    result = {
        name: [] for name in ("features", "A", "R", "h", "Qr", "source")
    }
    for step in range(161):
        packed = helper.state_interpolation(truth[step])
        rates = moist_rates_jax(packed, fields, moist_parameters)
        physical = np.stack(
            tuple(
                np.asarray(packed[name]).reshape(-1)
                for name in ("h", "S", "Qv", "Qc")
            )
            + (np.asarray(topography).reshape(-1),),
            axis=-1,
        )
        _, qr = primal.interpolate_and_pack(
            truth[step].sub(5), f"test2b_representation_c_truth_Qr_{step}"
        )
        h = np.asarray(packed["h"], dtype=np.float64).reshape(-1)
        a = np.asarray(rates["A"], dtype=np.float64).reshape(-1)
        r = np.asarray(rates["R"], dtype=np.float64).reshape(-1)
        result["features"].append(np.asarray(
            normalization.normalize_features(physical), dtype=np.float64
        ))
        result["A"].append(a)
        result["R"].append(r)
        result["h"].append(h)
        result["Qr"].append(np.asarray(qr, dtype=np.float64).reshape(-1))
        result["source"].append(_truth_source(a, r, h))
    return {name: np.stack(values) for name, values in result.items()}


def _component_metrics(prediction, target, weights, scales):
    return {
        name: _weighted_metrics(
            prediction[..., index], target[..., index],
            np.broadcast_to(weights, prediction[..., index].shape),
            scales[index],
        )
        for index, name in enumerate(SOURCE_ORDER)
    }


def _defect_metrics(values, weights, *, dt=100.0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    local_weights = np.broadcast_to(
        np.asarray(weights, dtype=np.float64), values.shape
    )
    weight_sum = float(np.sum(local_weights))
    signed_mean = float(np.sum(local_weights * values) / weight_sum)
    rms = float(np.sqrt(np.sum(local_weights * values**2) / weight_sum))
    spatial_signed = np.sum(local_weights * values, axis=-1)
    spatial_absolute = np.sum(local_weights * np.abs(values), axis=-1)
    integrated_signed = float(dt * np.sum(spatial_signed))
    integrated_absolute = float(dt * np.sum(spatial_absolute))
    return {
        "RMS": rms,
        "maximum_absolute": float(np.max(np.abs(values))),
        "signed_mass_weighted_mean": signed_mean,
        "spatial_integral_minimum": float(np.min(spatial_signed)),
        "spatial_integral_maximum": float(np.max(spatial_signed)),
        "time_sampled_signed_integral": integrated_signed,
        "time_sampled_absolute_integral": integrated_absolute,
        "signed_to_absolute_ratio": integrated_signed
        / max(integrated_absolute, np.finfo(np.float64).tiny),
        "sampled_state_count": int(values.shape[0]),
    }


def _projection_diagnostics(
    prediction, truth, h, weights, source_scales,
):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), prediction.shape[:-1])
    scales = np.asarray(source_scales, dtype=np.float64)
    if prediction.shape[-1] != 4 or truth.shape != prediction.shape:
        raise ValueError("source projection shape changed")
    flattened = prediction.reshape(-1, 4)
    flattened_truth = truth.reshape(-1, 4)
    flat_h = h.reshape(-1)
    local_weights = np.broadcast_to(
        np.asarray(weights, dtype=np.float64), prediction.shape[:-1]
    ).reshape(-1)
    normalized = flattened / scales
    normalized_truth = flattened_truth / scales
    basis_physical = np.asarray(
        ((BETA2, 1.0, -1.0, 0.0), (0.0, 0.0, -1.0, 1.0)),
        dtype=np.float64,
    ).T
    basis = basis_physical / scales[:, None]
    coefficients = normalized @ basis @ np.linalg.inv(basis.T @ basis)
    projected_normalized = coefficients @ basis.T
    residual_normalized = normalized - projected_normalized
    projected_physical = projected_normalized * scales
    residual_physical = flattened - projected_physical
    weight_sum = float(np.sum(local_weights))

    def vector_rms(values):
        return float(np.sqrt(
            np.sum(local_weights[:, None] * np.asarray(values) ** 2)
            / weight_sum
        ))

    residual_norm = vector_rms(residual_normalized)
    source_norm = vector_rms(normalized)
    truth_norm = vector_rms(normalized_truth)
    a_projected = coefficients[:, 0] / flat_h
    r_projected = coefficients[:, 1] / flat_h
    truth_a = flattened_truth[:, 1] / flat_h
    truth_r = flattened_truth[:, 3] / flat_h
    return {
        "metric": {
            "component_order": list(SOURCE_ORDER),
            "source_scales": scales.tolist(),
            "basis": basis.tolist(),
            "sha256": canonical_sha256({
                "component_order": list(SOURCE_ORDER),
                "source_scales": scales.tolist(),
                "basis": basis.tolist(),
            }),
        },
        "normalized_off_manifold_RMS": residual_norm,
        "normalized_source_RMS": source_norm,
        "normalized_truth_source_RMS": truth_norm,
        "off_manifold_fraction_of_source_magnitude": residual_norm
        / max(source_norm, np.finfo(np.float64).tiny),
        "physical_off_manifold_vector_RMS": vector_rms(residual_physical),
        "physical_off_manifold_component_bias": {
            name: float(np.sum(local_weights * residual_physical[:, index])
                        / weight_sum)
            for index, name in enumerate(SOURCE_ORDER)
        },
        "projected_A": _weighted_metrics(
            a_projected, truth_a, local_weights, SIGMA_A
        ),
        "projected_R": _weighted_metrics(
            r_projected, truth_r, local_weights, SIGMA_R
        ),
    }


def _effective_rate_diagnostics(
    prediction, truth_a, truth_r, h, qr, weights, comparison_scale,
):
    prediction = np.asarray(prediction, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    truth_a = np.asarray(truth_a, dtype=np.float64)
    truth_r = np.asarray(truth_r, dtype=np.float64)
    a_qv = prediction[..., 1] / h
    a_s = prediction[..., 0] / (h * BETA2)
    r_qr = prediction[..., 3] / h
    cloud_total = -prediction[..., 2] / h
    local_weights = np.broadcast_to(weights, h.shape)
    return {
        "A_from_Qv": _weighted_metrics(
            a_qv, truth_a, local_weights, SIGMA_A
        ),
        "A_from_S": _weighted_metrics(
            a_s, truth_a, local_weights, SIGMA_A
        ),
        "R_from_Qr": _r_metrics(
            r_qr, truth_r, local_weights, h, qr, comparison_scale
        ),
        "cloud_total_rate_minus_truth_A_plus_R": _weighted_metrics(
            cloud_total, truth_a + truth_r, local_weights, SIGMA_A
        ),
        "A_S_minus_A_Qv": _weighted_metrics(
            a_s, a_qv, local_weights, SIGMA_A
        ),
        "cloud_closure_minus_A_Qv_plus_R_Qr": _weighted_metrics(
            cloud_total, a_qv + r_qr, local_weights, SIGMA_A
        ),
    }


def _source_diagnostics(
    prediction, truth, a, r, h, qr, weights, normalization,
    comparison_scale,
):
    water = prediction[..., 1] + prediction[..., 2] + prediction[..., 3]
    thermo = prediction[..., 0] - BETA2 * prediction[..., 1]
    return {
        "component_errors": _component_metrics(
            prediction, truth, weights, normalization.source_scales
        ),
        "physical_two_rate_projection": _projection_diagnostics(
            prediction, truth, h, weights, normalization.source_scales
        ),
        "effective_rates": _effective_rate_diagnostics(
            prediction, a, r, h, qr, weights, comparison_scale
        ),
        "water_source_defect": _defect_metrics(water, weights),
        "thermodynamic_source_defect": _defect_metrics(thermo, weights),
        "negative_Qr_t_fraction": float(np.mean(prediction[..., 3] < 0.0)),
        "positive_Qr_t_fraction": float(np.mean(prediction[..., 3] > 0.0)),
    }


def _direct_metrics(
    parameters, arrays, weights, normalization, comparison_scale,
):
    prediction = np.stack([
        _predict_source(parameters, features, normalization)
        for features in arrays["features"]
    ])
    selections = {"TRAINING_OVERALL": tuple(range(0, 81)), **REGIMES}
    result = {}
    for name, steps in selections.items():
        index = np.asarray(steps)
        result[name] = _source_diagnostics(
            prediction[index], arrays["source"][index], arrays["A"][index],
            arrays["R"][index], arrays["h"][index], arrays["Qr"][index],
            weights, normalization, comparison_scale,
        )
        result[name]["state_indices"] = [int(steps[0]), int(steps[-1])]
    masks = _activity_masks(
        prediction[..., 3] / arrays["h"], arrays["h"], arrays["Qr"],
        comparison_scale,
    )
    truth_masks = _activity_masks(
        arrays["R"], arrays["h"], arrays["Qr"], comparison_scale
    )
    activation = {
        "diagnostic_rate": "effective R_Qr = predicted Qr_t / h",
        "first_exact_positive_predicted_step": _first_activation_step(
            masks["exact_positive"]
        ),
        "first_meaningful_positive_predicted_step": _first_activation_step(
            masks["meaningful_positive"]
        ),
        "first_meaningful_negative_predicted_step": _first_activation_step(
            masks["meaningful_negative"]
        ),
        "first_meaningful_positive_truth_step": _first_activation_step(
            truth_masks["meaningful_positive"]
        ),
    }
    for key, value in tuple(activation.items()):
        if key.endswith("_step"):
            activation[key.removesuffix("_step") + "_time"] = (
                None if value is None else float(100.0 * value)
            )
    return result, activation


def _source_record(
    case, moist, step, weights, normalization, comparison_scale,
):
    predicted = np.stack([
        np.asarray(moist.source_density[name], dtype=np.float64).reshape(-1)
        for name in SOURCE_ORDER
    ], axis=-1)
    packed = moist.packed_state
    analytical_rates = moist_rates_jax(
        packed, moist.packed_fields, moist.parameters
    )
    h = np.asarray(packed["h"], dtype=np.float64).reshape(-1)
    a = np.asarray(analytical_rates["A"], dtype=np.float64).reshape(-1)
    r = np.asarray(analytical_rates["R"], dtype=np.float64).reshape(-1)
    truth = _truth_source(a, r, h)
    primal = case.helper.moist_helper.primal_helper
    _, qr = primal.interpolate_and_pack(
        moist.stage_state.sub(5), f"test2b_C_source_Qr_{step}"
    )
    qr = np.asarray(qr, dtype=np.float64).reshape(-1)
    qc = np.asarray(packed["Qc"], dtype=np.float64).reshape(-1) / h
    water = predicted[:, 1] + predicted[:, 2] + predicted[:, 3]
    thermo = predicted[:, 0] - BETA2 * predicted[:, 1]
    r_qr = predicted[:, 3] / h
    predicted_masks = _activity_masks(
        r_qr[None, :], h[None, :], qr[None, :], comparison_scale
    )
    truth_masks = _activity_masks(
        r[None, :], h[None, :], qr[None, :], comparison_scale
    )
    positive = predicted_masks["meaningful_positive"].reshape(-1)
    negative = predicted_masks["meaningful_negative"].reshape(-1)
    truth_active = truth_masks["meaningful_positive"].reshape(-1)
    actual_water_rate = _field_integral(
        case,
        moist.tendency.sub(3) + moist.tendency.sub(4)
        + moist.tendency.sub(5),
    )
    actual_thermo_rate = _field_integral(
        case, moist.tendency.sub(2) - BETA2 * moist.tendency.sub(3)
    )
    local = _source_diagnostics(
        predicted, truth, a, r, h, qr, weights, normalization,
        comparison_scale,
    )
    result = {
        "step": int(step),
        "time": float(step * case.dt),
        "applied_to_trajectory": bool(step < 160),
        "sample_count": int(h.size),
        "specific_Qc_maximum": float(np.max(qc)),
        "specific_Qc_rms": float(np.sqrt(np.mean(qc * qc))),
        "effective_R_Qr_maximum": float(np.max(r_qr)),
        "effective_R_Qr_minimum": float(np.min(r_qr)),
        "effective_R_Qr_RMS": float(np.sqrt(np.mean(r_qr * r_qr))),
        "effective_R_Qr_metrics": local["effective_rates"]["R_from_Qr"],
        "meaningful_positive_R_Qr_fraction": float(np.mean(positive)),
        "meaningful_negative_R_Qr_fraction": float(np.mean(negative)),
        "truth_meaningful_R_fraction": float(np.mean(truth_active)),
        "false_positive_R_Qr_count": int(np.count_nonzero(
            positive & ~truth_active
        )),
        "false_negative_R_Qr_count": int(np.count_nonzero(
            truth_active & ~positive
        )),
        "truth_active_R_count": int(np.count_nonzero(truth_active)),
        "source_diagnostics": local,
        "local_water_source_spatial_integral": float(np.sum(weights * water)),
        "local_absolute_water_source_spatial_integral": float(
            np.sum(weights * np.abs(water))
        ),
        "local_thermo_source_spatial_integral": float(
            np.sum(weights * thermo)
        ),
        "local_absolute_thermo_source_spatial_integral": float(
            np.sum(weights * np.abs(thermo))
        ),
        "discrete_water_tendency_mass_rate": actual_water_rate,
        "discrete_thermo_tendency_mass_rate": actual_thermo_rate,
        "rain_source_mass_rate": _field_integral(case, moist.tendency.sub(5)),
    }
    for name in (
        "discrete_water_tendency_mass_rate",
        "discrete_thermo_tendency_mass_rate",
        "rain_source_mass_rate",
    ):
        result[name.removesuffix("_rate") + "_increment"] = (
            case.dt * result[name]
        )
    return result


def _summarize_rain(records, boundary, truth_audit):
    exact = [row for row in records if row["effective_R_Qr_maximum"] > 0.0]
    meaningful = [
        row for row in records
        if row["meaningful_positive_R_Qr_fraction"] > 0.0
    ]
    negative = [
        row for row in records
        if row["meaningful_negative_R_Qr_fraction"] > 0.0
    ]
    applied = [row for row in records if row["applied_to_trajectory"]]
    pre = [row for row in applied if row["step"] <= 50]
    peak_qc = max(records, key=lambda row: row["specific_Qc_maximum"])
    qr = np.asarray(boundary["Qr_mass"], dtype=np.float64)
    truth_qr = np.asarray(
        [row["rain_water_mass"] for row in truth_audit["records"]],
        dtype=np.float64,
    )
    summary = truth_audit["summary"]
    exact_step = None if not exact else exact[0]["step"]
    meaningful_step = None if not meaningful else meaningful[0]["step"]
    negative_step = None if not negative else negative[0]["step"]
    integrated = float(sum(
        row["rain_source_mass_increment"] for row in applied
    ))
    pre_count = sum(row["false_positive_R_Qr_count"] for row in pre)
    pre_samples = sum(row["sample_count"] for row in pre)
    active_count = sum(row["truth_active_R_count"] for row in applied)
    false_negative = sum(row["false_negative_R_Qr_count"] for row in applied)
    return {
        "diagnostic_rate": "effective R_Qr = predicted Qr_t / h",
        "first_exact_positive_source_step": exact_step,
        "first_exact_positive_source_time": None
        if exact_step is None else float(exact_step * case_dt(records)),
        "first_meaningful_positive_source_step": meaningful_step,
        "first_meaningful_positive_source_time": None
        if meaningful_step is None else float(meaningful_step * case_dt(records)),
        "meaningful_onset_time_error_vs_truth": None
        if meaningful_step is None
        else float(meaningful_step * case_dt(records) - 5100.0),
        "first_meaningful_negative_source_step": negative_step,
        "first_meaningful_negative_source_time": None
        if negative_step is None else float(negative_step * case_dt(records)),
        "pre_truth_onset_false_positive_count": int(pre_count),
        "pre_truth_onset_false_positive_fraction": float(
            pre_count / pre_samples
        ),
        "false_negative_active_count": int(false_negative),
        "truth_active_count": int(active_count),
        "false_negative_rate_given_truth_active": None
        if active_count == 0 else float(false_negative / active_count),
        "maximum_effective_R_Qr": float(max(
            row["effective_R_Qr_maximum"] for row in records
        )),
        "minimum_effective_R_Qr": float(min(
            row["effective_R_Qr_minimum"] for row in records
        )),
        "maximum_specific_Qc": peak_qc["specific_Qc_maximum"],
        "maximum_specific_Qc_step": peak_qc["step"],
        "maximum_specific_Qc_time": peak_qc["time"],
        "time_integrated_rain_source_mass": integrated,
        "time_integrated_rain_source_mass_error": float(
            integrated - summary["time_integrated_rain_source_mass"]
        ),
        "pre_truth_onset_signed_rain_source_mass": float(sum(
            row["rain_source_mass_increment"] for row in pre
        )),
        "pre_truth_onset_absolute_rain_source_mass": float(sum(
            abs(row["rain_source_mass_increment"]) for row in pre
        )),
        "final_Qr_mass": float(qr[-1]),
        "maximum_Qr_mass": float(np.max(qr)),
        "minimum_Qr_mass": float(np.min(qr)),
        "first_positive_Qr_mass_step": _first_positive_mass_step(qr),
        "first_positive_Qr_mass_time": None
        if _first_positive_mass_step(qr) is None
        else float(_first_positive_mass_step(qr) * case_dt(records)),
        "final_Qr_mass_error": float(qr[-1] - truth_qr[-1]),
        "maximum_absolute_Qr_mass_error": float(np.max(np.abs(qr - truth_qr))),
        "records": records,
    }


def case_dt(records):
    if len(records) < 2:
        return 100.0
    return float(records[1]["time"] - records[0]["time"])


def _first_positive_mass_step(values):
    locations = np.flatnonzero(np.asarray(values) > 0.0)
    return None if locations.size == 0 else int(locations[0])


def _signed_budget(boundary, truth_water, records):
    water = np.asarray(boundary["total_water_mass"], dtype=np.float64)
    truth_water = np.asarray(truth_water, dtype=np.float64)
    model_drift = water - water[0]
    truth_drift = truth_water - truth_water[0]
    error_drift = model_drift - truth_drift
    applied = [row for row in records if row["applied_to_trajectory"]]
    discrete_increments = np.asarray([
        row["discrete_water_tendency_mass_increment"] for row in applied
    ])
    local_increments = np.asarray([
        case_dt(records) * row["local_water_source_spatial_integral"]
        for row in applied
    ])
    local_absolute = float(sum(
        case_dt(records) * row["local_absolute_water_source_spatial_integral"]
        for row in applied
    ))
    cumulative_discrete = np.concatenate(([0.0], np.cumsum(discrete_increments)))
    cumulative_local = np.concatenate(([0.0], np.cumsum(local_increments)))
    scale = abs(float(truth_water[0]))
    return {
        "initial_model_total_water_mass": float(water[0]),
        "initial_truth_total_water_mass": float(truth_water[0]),
        "final_signed_model_drift": float(model_drift[-1]),
        "final_signed_truth_relative_drift": float(error_drift[-1]),
        "maximum_positive_truth_relative_drift": float(np.max(error_drift)),
        "most_negative_truth_relative_drift": float(np.min(error_drift)),
        "maximum_absolute_truth_relative_drift": float(np.max(np.abs(error_drift))),
        "final_relative_signed_drift": float(error_drift[-1] / scale),
        "maximum_relative_absolute_drift": float(np.max(np.abs(error_drift)) / scale),
        "truth_relative_drift_maximum_step": int(np.argmax(np.abs(error_drift))),
        "cumulative_discrete_signed_source_defect": float(cumulative_discrete[-1]),
        "cumulative_local_signed_source_defect": float(cumulative_local[-1]),
        "cumulative_local_absolute_source_defect": local_absolute,
        "local_signed_to_absolute_ratio": float(cumulative_local[-1])
        / max(local_absolute, np.finfo(np.float64).tiny),
        "state_minus_discrete_source_closure": float(
            model_drift[-1] - cumulative_discrete[-1]
        ),
        "boundary_model_total_water_mass": water.tolist(),
        "boundary_truth_total_water_mass": truth_water.tolist(),
        "boundary_truth_relative_signed_drift": error_drift.tolist(),
        "cumulative_discrete_source_defect": cumulative_discrete.tolist(),
        "cumulative_local_source_defect": cumulative_local.tolist(),
    }


def _thermodynamic_budget(boundary, truth_values, records):
    values = np.asarray(
        boundary["S_minus_beta2_Qv_mass"], dtype=np.float64
    )
    truth_values = np.asarray(truth_values, dtype=np.float64)
    model_drift = values - values[0]
    truth_drift = truth_values - truth_values[0]
    error_drift = model_drift - truth_drift
    applied = [row for row in records if row["applied_to_trajectory"]]
    discrete = float(sum(
        row["discrete_thermo_tendency_mass_increment"] for row in applied
    ))
    local_signed = float(sum(
        case_dt(records) * row["local_thermo_source_spatial_integral"]
        for row in applied
    ))
    local_absolute = float(sum(
        case_dt(records) * row[
            "local_absolute_thermo_source_spatial_integral"
        ]
        for row in applied
    ))
    scale = abs(float(truth_values[0]))
    return {
        "initial_model_value": float(values[0]),
        "initial_truth_value": float(truth_values[0]),
        "final_signed_model_drift": float(model_drift[-1]),
        "final_truth_relative_signed_drift": float(error_drift[-1]),
        "maximum_absolute_truth_relative_drift": float(
            np.max(np.abs(error_drift))
        ),
        "final_relative_signed_drift": float(error_drift[-1] / scale),
        "maximum_relative_absolute_drift": float(
            np.max(np.abs(error_drift)) / scale
        ),
        "cumulative_discrete_signed_source_defect": discrete,
        "cumulative_local_signed_source_defect": local_signed,
        "cumulative_local_absolute_source_defect": local_absolute,
        "local_signed_to_absolute_ratio": local_signed
        / max(local_absolute, np.finfo(np.float64).tiny),
        "state_minus_discrete_source_closure": float(
            model_drift[-1] - discrete
        ),
    }


def _summarize_accumulator(values, weights, normalization, comparison_scale):
    prediction = np.stack(values["prediction"])
    truth = np.stack(values["truth"])
    a = np.stack(values["A"])
    r = np.stack(values["R"])
    h = np.stack(values["h"])
    qr = np.stack(values["Qr"])
    return _source_diagnostics(
        prediction, truth, a, r, h, qr, weights, normalization,
        comparison_scale,
    )


def _evaluate_rollout(
    case, truth, parameters, weights, diagnostic_configuration,
    truth_diagnostics, truth_audit, label, normalization,
):
    evaluator = ResolvedDiagnosticEvaluator(case, diagnostic_configuration)
    current = _copy_function(truth[0], f"test2b_C_{label}_autonomous_0")
    zero = case.new_state(f"test2b_C_{label}_zero")
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
    truth_water = []
    truth_thermo = []
    source_records = []
    accumulators = {
        name: {
            key: [] for key in ("prediction", "truth", "A", "R", "h", "Qr")
        }
        for name in (*REGIMES, "ALL")
    }
    comparison_scale = float(
        truth_audit["activity_tolerance"]["comparison_rate_scale"]
    )
    for step in range(161):
        target = truth[step]
        numerator = _state_squared_difference(
            case, current, target, f"test2b_C_{label}_mixed_residual_{step}"
        )
        denominator = _state_squared_difference(
            case, target, zero, f"test2b_C_{label}_mixed_target_{step}"
        )
        mixed_records[step] = _relative_record(numerator, denominator)
        for index, name in enumerate(STATE_FIELDS):
            field_numerator, field_denominator = _field_squared_norms(
                case, current, target, index,
                f"test2b_C_{label}_{name}_{step}",
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
        boundary["all_state_coefficients_finite"].append(bool(all(
            np.all(np.isfinite(current.dat.data[index]))
            for index in range(len(STATE_FIELDS))
        )))
        qv_mass = _field_integral(case, current.sub(3))
        qc_mass = _field_integral(case, current.sub(4))
        qr_mass = _field_integral(case, current.sub(5))
        boundary["Qv_mass"].append(qv_mass)
        boundary["Qc_mass"].append(qc_mass)
        boundary["Qr_mass"].append(qr_mass)
        boundary["total_water_mass"].append(qv_mass + qc_mass + qr_mass)
        boundary["S_minus_beta2_Qv_mass"].append(
            _field_integral(case, current.sub(2)) - BETA2 * qv_mass
        )
        truth_qv = _field_integral(case, target.sub(3))
        truth_qc = _field_integral(case, target.sub(4))
        truth_qr = _field_integral(case, target.sub(5))
        truth_water.append(truth_qv + truth_qc + truth_qr)
        truth_thermo.append(
            _field_integral(case, target.sub(2)) - BETA2 * truth_qv
        )

        if step < 160:
            cache = case.helper.take_forward_step_cached(
                current, step * case.dt, case.dt,
                neural_parameters=parameters,
            )
            moist = cache.children[-1].cache
            next_state = _copy_function(
                cache.state_out, f"test2b_C_{label}_autonomous_{step + 1}"
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
            case, moist, step, weights, normalization, comparison_scale
        )
        source_records.append(record)
        if step < 160:
            target_step = step + 1
            regime = next(
                name for name, steps in REGIMES.items()
                if target_step in steps
            )
            prediction = np.stack([
                np.asarray(moist.source_density[name]).reshape(-1)
                for name in SOURCE_ORDER
            ], axis=-1)
            analytical = moist_rates_jax(
                moist.packed_state, moist.packed_fields, moist.parameters
            )
            h = np.asarray(moist.packed_state["h"]).reshape(-1)
            a = np.asarray(analytical["A"]).reshape(-1)
            r = np.asarray(analytical["R"]).reshape(-1)
            _, qr = case.helper.moist_helper.primal_helper.interpolate_and_pack(
                moist.stage_state.sub(5),
                f"test2b_C_{label}_rate_Qr_{step}",
            )
            values = {
                "prediction": prediction,
                "truth": _truth_source(a, r, h),
                "A": a,
                "R": r,
                "h": h,
                "Qr": np.asarray(qr).reshape(-1),
            }
            for name in (regime, "ALL"):
                for key, value in values.items():
                    accumulators[name][key].append(value)
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
    source_summary = {
        name: _summarize_accumulator(
            accumulators[name], weights, normalization, comparison_scale
        )
        for name in (*REGIMES, "ALL")
    }
    rain = _summarize_rain(source_records, boundary, truth_audit)
    rain_mass_floor = (
        128.0 * np.finfo(np.float64).eps
        * abs(boundary["total_water_mass"][0])
    )
    qr_values = np.asarray(boundary["Qr_mass"])
    meaningful_qr_steps = np.flatnonzero(qr_values > rain_mass_floor)
    rain["Qr_mass_meaningful_floor"] = float(rain_mass_floor)
    rain["first_meaningful_positive_Qr_mass_step"] = (
        None if meaningful_qr_steps.size == 0
        else int(meaningful_qr_steps[0])
    )
    rain["first_meaningful_positive_Qr_mass_time"] = (
        None if meaningful_qr_steps.size == 0
        else float(case.dt * meaningful_qr_steps[0])
    )
    return {
        "mixed_state_error": mixed,
        "fieldwise_state_error": fieldwise,
        "rain_source_and_partition": rain,
        "source_diagnostics_on_model_postprefix_states": source_summary,
        "signed_total_water_budget": _signed_budget(
            boundary, truth_water, source_records
        ),
        "signed_thermodynamic_budget": _thermodynamic_budget(
            boundary, truth_thermo, source_records
        ),
        "conservation_and_stability": conservation,
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
    arrays = _truth_arrays(case, truth, normalization)
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
        "output_directory": "/tmp/test2b-representation-c-postprocess-no-output",
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
        "projection_applied_to_trajectory": False,
        "conservation_repair_applied": False,
        "configuration": str(Path(configuration_path).resolve()),
        "preparation": {
            "path": str(Path(preparation_path).resolve()),
            "sha256": _file_sha256(preparation_path),
            "truth_states": [0, 80],
            "heldout_states": [81, 160],
            "source_scales": normalization.source_scales.tolist(),
        },
        "artifacts": verified,
        "objective_matrix": {},
        "direct_source_diagnostics": {},
        "direct_effective_rain_activation": {},
        "autonomous": {},
    }
    for label in LABELS:
        print(f"Representation C evaluation: {label} fixed/direct", flush=True)
        local = parameters[label]
        result["objective_matrix"][label] = {
            name: objective.value(local) for name, objective in fixed.items()
        }
        direct, activation = _direct_metrics(
            local, arrays, weights, normalization, comparison_scale
        )
        result["direct_source_diagnostics"][label] = direct
        result["direct_effective_rain_activation"][label] = activation
    del fixed, repeated_weights, data, matrices, arrays
    collect()
    for label in LABELS:
        local = parameters[label]
        for horizon in (2, 5):
            print(
                f"Representation C evaluation: {label} H{horizon} objective",
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
        print(f"Representation C evaluation: {label} autonomous", flush=True)
        result["autonomous"][label] = _evaluate_rollout(
            case, truth, local, weights, diagnostic_configuration,
            truth_diagnostics, truth_audit, label, normalization,
        )
        print(f"Representation C evaluation: {label} complete", flush=True)
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
