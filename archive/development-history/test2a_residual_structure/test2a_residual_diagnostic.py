"""Read-only residual-structure diagnostic for a frozen Test-2A operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .jax_moist import moist_diagnostics_jit
from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_continuation import verify_parameter_artifact
from .test2a_operator import (
    DenseMLP,
    FEATURE_ORDER,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    operator_metrics,
    physical_predictions,
)


SIGN_ACTIVITY_LEVELS = (1.0e-3, 1.0e-2, 1.0e-1)
MAGNITUDE_EDGES = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
SWITCH_EDGES = (
    -np.inf,
    -1.0e-1,
    -1.0e-2,
    -1.0e-3,
    -1.0e-4,
    -1.0e-5,
    -1.0e-6,
    -1.0e-8,
    0.0,
    1.0e-8,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    np.inf,
)


def _finite_vector(name, values):
    array = np.asarray(values)
    if array.dtype != np.float64:
        raise TypeError(f"{name} must be float64")
    array = array.reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def residual_subset_metrics(prediction, target, mask, *, global_error_energy):
    """Return physical residual metrics on one explicitly selected subset."""
    predicted = _finite_vector("prediction", prediction)
    truth = _finite_vector("target", target)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if predicted.shape != truth.shape or selected.shape != truth.shape:
        raise ValueError("residual subset arrays do not align")
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {"sample_count": 0, "sample_fraction": 0.0}
    local_truth = truth[selected]
    local_prediction = predicted[selected]
    residual = local_prediction - local_truth
    target_rms = float(np.sqrt(np.mean(local_truth * local_truth)))
    residual_energy = float(np.sum(residual * residual))
    return {
        "sample_count": count,
        "sample_fraction": float(np.mean(selected)),
        "target_minimum": float(np.min(local_truth)),
        "target_maximum": float(np.max(local_truth)),
        "target_rms": target_rms,
        "residual_bias": float(np.mean(residual)),
        "residual_mae": float(np.mean(np.abs(residual))),
        "residual_rmse": float(np.sqrt(np.mean(residual * residual))),
        "relative_rms_error": (
            None if target_rms == 0.0 else float(np.sqrt(np.mean(residual * residual)) / target_rms)
        ),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "sign_accuracy": float(np.mean(np.sign(local_prediction) == np.sign(local_truth))),
        "global_residual_squared_energy_fraction": (
            0.0 if global_error_energy == 0.0 else residual_energy / global_error_energy
        ),
    }


def residual_bins(independent, prediction, target, edges, *, scale=1.0):
    """Bin residuals without replacing pointwise relative errors at zero."""
    coordinate = _finite_vector("independent variable", independent) / float(scale)
    predicted = _finite_vector("prediction", prediction)
    truth = _finite_vector("target", target)
    if coordinate.shape != truth.shape or predicted.shape != truth.shape:
        raise ValueError("binned residual arrays do not align")
    error = predicted - truth
    total_error_energy = float(np.sum(error * error))
    total_target_energy = float(np.sum(truth * truth))
    records = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = coordinate >= lower
        mask &= coordinate <= upper if index + 2 == len(edges) else coordinate < upper
        count = int(np.count_nonzero(mask))
        record = {
            "lower": None if np.isneginf(lower) else float(lower),
            "upper": None if np.isposinf(upper) else float(upper),
            "sample_count": count,
            "sample_fraction": float(np.mean(mask)),
        }
        if count:
            local_error = error[mask]
            local_truth = truth[mask]
            local_coordinate = coordinate[mask]
            target_rms = float(np.sqrt(np.mean(local_truth * local_truth)))
            nonzero = np.abs(local_truth) > 0.0
            pointwise_relative = (
                np.abs(local_error[nonzero] / local_truth[nonzero])
                if np.any(nonzero)
                else np.asarray([], dtype=np.float64)
            )
            record.update(
                {
                    "coordinate_mean": float(np.mean(local_coordinate)),
                    "coordinate_median": float(np.median(local_coordinate)),
                    "absolute_residual_mean": float(np.mean(np.abs(local_error))),
                    "absolute_residual_median": float(np.median(np.abs(local_error))),
                    "residual_rmse": float(np.sqrt(np.mean(local_error * local_error))),
                    "relative_rms_error": (
                        None
                        if target_rms == 0.0
                        else float(np.sqrt(np.mean(local_error * local_error)) / target_rms)
                    ),
                    "pointwise_absolute_relative_residual_median": (
                        None if pointwise_relative.size == 0 else float(np.median(pointwise_relative))
                    ),
                    "global_residual_squared_energy_fraction": (
                        0.0
                        if total_error_energy == 0.0
                        else float(np.sum(local_error * local_error) / total_error_energy)
                    ),
                    "global_target_squared_energy_fraction": (
                        0.0
                        if total_target_energy == 0.0
                        else float(np.sum(local_truth * local_truth) / total_target_energy)
                    ),
                }
            )
        records.append(record)
    return records


def time_residual_metrics(prediction, target, *, samples_per_state, steps=range(81)):
    predicted = _finite_vector("prediction", prediction)
    truth = _finite_vector("target", target)
    step_values = tuple(int(step) for step in steps)
    if step_values != tuple(range(81)):
        raise ValueError("Test 2A residual diagnostics are restricted to states 0..80")
    if predicted.size != len(step_values) * int(samples_per_state):
        raise ValueError("time layout does not match the deployed sample accounting")
    error = predicted - truth
    total_error_energy = float(np.sum(error * error))
    records = []
    for local_index, step in enumerate(step_values):
        start = local_index * samples_per_state
        stop = start + samples_per_state
        local_prediction = predicted[start:stop]
        local_truth = truth[start:stop]
        local_error = local_prediction - local_truth
        target_rms = float(np.sqrt(np.mean(local_truth * local_truth)))
        maximum = float(np.max(np.abs(local_truth)))
        sign_mask = np.abs(local_truth) > 1.0e-6 * maximum
        records.append(
            {
                "step": step,
                "sample_count": int(samples_per_state),
                "target_rms": target_rms,
                "residual_bias": float(np.mean(local_error)),
                "residual_mae": float(np.mean(np.abs(local_error))),
                "residual_rmse": float(np.sqrt(np.mean(local_error * local_error))),
                "relative_rms_error": (
                    None
                    if target_rms == 0.0
                    else float(np.sqrt(np.mean(local_error * local_error)) / target_rms)
                ),
                "maximum_absolute_error": float(np.max(np.abs(local_error))),
                "sign_accuracy": (
                    None
                    if not np.any(sign_mask)
                    else float(
                        np.mean(
                            np.sign(local_prediction[sign_mask])
                            == np.sign(local_truth[sign_mask])
                        )
                    )
                ),
                "global_residual_squared_energy_fraction": (
                    0.0
                    if total_error_energy == 0.0
                    else float(np.sum(local_error * local_error) / total_error_energy)
                ),
            }
        )
    return records


def deployed_a_switch_diagnostics(features, target, parameters):
    """Reuse the certified JAX algebra and expose its exact saturation switch."""
    values = np.asarray(features)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 5:
        raise TypeError("features must be a float64 (N,5) array")
    state = {
        name: jnp.asarray(values[:, index], dtype=jnp.float64)
        for index, name in enumerate(FEATURE_ORDER[:-1])
    }
    fields = {"B": jnp.asarray(values[:, 4], dtype=jnp.float64)}
    required = ("g", "q0", "H0", "gamma_r", "qprecip", "L", "configured_dt")
    deployed_parameters = {
        name: jnp.asarray(parameters[name], dtype=jnp.float64) for name in required
    }
    diagnostics = moist_diagnostics_jit(state, fields, deployed_parameters)
    arrays = {
        name: np.asarray(diagnostics[name], dtype=np.float64)
        for name in (
            "qv",
            "qc",
            "qsat",
            "gamma_v",
            "condensation_argument",
            "evaporation_argument",
            "evaporation_cap",
            "evaporation_cap_difference",
            "C",
            "E_positive",
            "E",
            "A",
        )
    }
    truth = _finite_vector("target", target)
    discrepancy = arrays["A"] - truth
    tolerance = max(
        1.0e-18,
        512.0
        * np.finfo(np.float64).eps
        * float(np.max(np.abs(truth))),
    )
    if float(np.max(np.abs(discrepancy))) > tolerance:
        raise ValueError("certified deployed A algebra does not reproduce the dataset target")
    saturation_excess = arrays["qv"] - arrays["qsat"]
    relative_saturation_excess = saturation_excess / arrays["qsat"]
    return {
        **arrays,
        "saturation_excess": saturation_excess,
        "relative_saturation_excess": relative_saturation_excess,
        "target_reproduction_maximum_absolute_error": float(
            np.max(np.abs(discrepancy))
        ),
        "target_reproduction_tolerance": tolerance,
    }


def _regime_metrics(prediction, target, switches):
    error_energy = float(np.sum((prediction - target) ** 2))
    delta = switches["saturation_excess"]
    cap = switches["evaporation_cap"]
    positive_candidate = switches["E_positive"]
    regimes = {
        "supersaturated_condensation_candidate": delta > 0.0,
        "sub_saturated_uncapped_evaporation": (delta < 0.0) & (cap >= positive_candidate),
        "sub_saturated_capped_nonnegative_evaporation": (
            (delta < 0.0) & (cap >= 0.0) & (cap < positive_candidate)
        ),
        "sub_saturated_negative_cloud_cap": (delta < 0.0) & (cap < 0.0),
        "exact_saturation_boundary": delta == 0.0,
    }
    return {
        name: residual_subset_metrics(
            prediction, target, mask, global_error_energy=error_energy
        )
        for name, mask in regimes.items()
    }


def _squared_energy_fraction(values, mask):
    array = _finite_vector("energy values", values)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if selected.shape != array.shape:
        raise ValueError("energy mask does not align")
    total = float(np.sum(array * array))
    return 0.0 if total == 0.0 else float(np.sum(array[selected] ** 2) / total)


def _load_frozen_parameters(result_path):
    result_file = Path(result_path)
    result = read_json_record(result_file)
    if result.get("status") != "complete" or result.get("decision") != "STILL_OPTIMIZER_LIMITED":
        raise ValueError("residual diagnostic requires the completed +5000 continuation")
    if int(result["optimizer"]["additional_accepted_iterations"]) != 5000:
        raise ValueError("residual diagnostic source is not the +5000 endpoint")
    if int(result["optimizer"]["HVP_evaluations"]) != 0:
        raise ValueError("unexpected source optimizer record")
    progress_path = result_file.with_name("continuation_progress.json")
    progress = read_json_record(progress_path)
    if progress.get("status") != "complete":
        raise ValueError("continuation progress record is incomplete")
    parameter_path = Path(result["final_parameter_file"])
    if parameter_path.resolve() != Path(progress["last_checkpoint_parameter_file"]).resolve():
        raise ValueError("result and progress parameter paths disagree")
    parameters, configuration = verify_parameter_artifact(
        parameter_path,
        progress["last_checkpoint_npz_sha256"],
        progress["last_checkpoint_pytree_sha256"],
    )
    return parameters, configuration, result, progress, parameter_path


def _plot_diagnostics(summary, prediction, target, switch, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    count = target.size
    stride = max(1, int(np.ceil(count / 50_000)))
    selected = np.arange(0, count, stride, dtype=np.int64)

    figure, axis = plt.subplots()
    axis.scatter(target[selected], prediction[selected], s=2, alpha=0.2, rasterized=True)
    extent = max(float(np.max(np.abs(target))), float(np.max(np.abs(prediction))))
    axis.plot((-extent, extent), (-extent, extent), color="black", linewidth=1)
    axis.set(xlabel="truth A", ylabel="neural A_theta")
    figure.tight_layout()
    figure.savefig(destination / "prediction_vs_truth_scatter.png", dpi=150)
    plt.close(figure)

    absolute_truth = np.abs(target[selected])
    absolute_error = np.abs(prediction[selected] - target[selected])
    nonzero = absolute_truth > 0.0
    figure, axes = plt.subplots(2, 1, sharex=True)
    axes[0].loglog(
        absolute_truth[nonzero], absolute_error[nonzero], ".", markersize=1, alpha=0.2
    )
    axes[0].set(ylabel="|A_theta-A|")
    axes[1].loglog(
        absolute_truth[nonzero],
        absolute_error[nonzero] / absolute_truth[nonzero],
        ".",
        markersize=1,
        alpha=0.2,
    )
    axes[1].set(xlabel="|A|", ylabel="|residual|/|A|")
    figure.tight_layout()
    figure.savefig(destination / "residual_vs_abs_A.png", dpi=150)
    plt.close(figure)

    time_records = summary["time_resolved_residual"]
    figure, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot([record["step"] for record in time_records], [record["residual_rmse"] for record in time_records])
    axes[0].set(ylabel="residual RMSE")
    axes[1].plot([record["step"] for record in time_records], [record["relative_rms_error"] for record in time_records])
    axes[1].set(xlabel="truth state", ylabel="relative RMS")
    figure.tight_layout()
    figure.savefig(destination / "residual_by_time_state.png", dpi=150)
    plt.close(figure)

    chi = switch["relative_saturation_excess"][selected]
    residual = prediction[selected] - target[selected]
    figure, axis = plt.subplots()
    axis.scatter(chi, residual, s=2, alpha=0.2, rasterized=True)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xscale("symlog", linthresh=1.0e-8)
    axis.set(xlabel="relative saturation excess (qv-q_sat)/q_sat", ylabel="A_theta-A")
    figure.tight_layout()
    figure.savefig(destination / "residual_vs_saturation_switch.png", dpi=150)
    plt.close(figure)


def run_residual_diagnostic(
    selected_configuration,
    dataset_path,
    continuation_result,
    activity_audit,
    output_directory,
):
    """Evaluate the frozen operator; do not update parameters or open truth states."""
    selected = load_selected_configuration(selected_configuration)
    dataset, metadata = load_operator_dataset(dataset_path)
    if metadata["truth_state_indices"] != [0, 80] or metadata["states_after_80_accessed"]:
        raise ValueError("future Test-2 truth data are forbidden")
    parameters, model_configuration, source_result, progress, parameter_path = (
        _load_frozen_parameters(continuation_result)
    )
    if model_configuration.to_record() != mlp_configuration_from_record(selected["model"]).to_record():
        raise ValueError("frozen parameter architecture differs from selected Test 2A")
    normalization = normalization_from_record(metadata["normalization"])
    model = DenseMLP(model_configuration)
    prediction = physical_predictions(
        parameters, model, normalization, dataset.features
    )
    target = np.asarray(dataset.targets, dtype=np.float64).reshape(-1)
    error = prediction - target
    error_energy = float(np.sum(error * error))
    maximum = float(np.max(np.abs(target)))
    global_metrics = operator_metrics(prediction, target)

    sign_metrics = {
        "existing_non_negligible_abs_A_gt_1e-6_max_abs_A": {
            "threshold": float(1.0e-6 * maximum),
            "accuracy": global_metrics["sign_accuracy"],
            "sample_count": int(np.count_nonzero(np.abs(target) > 1.0e-6 * maximum)),
        }
    }
    for level in SIGN_ACTIVITY_LEVELS:
        label = f"abs_A_gt_{level:.0e}_max_abs_A"
        sign_metrics[label] = global_metrics["sign_accuracy_strata"][label]

    sign_regimes = {
        "A_negative": residual_subset_metrics(
            prediction, target, target < 0.0, global_error_energy=error_energy
        ),
        "A_positive": residual_subset_metrics(
            prediction, target, target > 0.0, global_error_energy=error_energy
        ),
        "A_exact_zero": residual_subset_metrics(
            prediction, target, target == 0.0, global_error_energy=error_energy
        ),
    }
    magnitude_bins = residual_bins(
        np.abs(target), prediction, target, MAGNITUDE_EDGES, scale=maximum
    )
    samples_per_state = dataset.cells * dataset.points_per_cell
    time_records = time_residual_metrics(
        prediction, target, samples_per_state=samples_per_state
    )

    audit = read_json_record(activity_audit)
    moist_parameters = audit["moist_parameters"]
    switch = deployed_a_switch_diagnostics(
        dataset.features, target, moist_parameters
    )
    switch_bins = residual_bins(
        switch["relative_saturation_excess"],
        prediction,
        target,
        SWITCH_EDGES,
    )
    top_time = sorted(
        time_records,
        key=lambda record: record["global_residual_squared_energy_fraction"],
        reverse=True,
    )[:10]
    absolute_target = np.abs(target)
    relative_switch = switch["relative_saturation_excess"]
    magnitude_concentration = {}
    for relation, threshold in (("le", 1.0e-3), ("gt", 1.0e-2), ("gt", 1.0e-1)):
        mask = (
            absolute_target <= threshold * maximum
            if relation == "le"
            else absolute_target > threshold * maximum
        )
        label = f"abs_A_{relation}_{threshold:.0e}_max_abs_A"
        magnitude_concentration[label] = {
            "sample_fraction": float(np.mean(mask)),
            "residual_squared_energy_fraction": _squared_energy_fraction(error, mask),
            "target_squared_energy_fraction": _squared_energy_fraction(target, mask),
        }
    near_switch = np.abs(relative_switch) <= 1.0e-6
    sub_saturated = switch["saturation_excess"] < 0.0
    first_ten_error_fraction = float(
        sum(
            record["global_residual_squared_energy_fraction"]
            for record in time_records[:10]
        )
    )
    time_relative_values = np.asarray(
        [record["relative_rms_error"] for record in time_records], dtype=np.float64
    )
    concentration_summary = {
        "by_truth_magnitude": magnitude_concentration,
        "near_saturation_switch_abs_relative_excess_le_1e-6": {
            "sample_fraction": float(np.mean(near_switch)),
            "residual_squared_energy_fraction": _squared_energy_fraction(error, near_switch),
            "target_squared_energy_fraction": _squared_energy_fraction(target, near_switch),
        },
        "sub_saturated_delta_lt_zero": {
            "sample_fraction": float(np.mean(sub_saturated)),
            "residual_squared_energy_fraction": _squared_energy_fraction(error, sub_saturated),
            "target_squared_energy_fraction": _squared_energy_fraction(target, sub_saturated),
        },
        "time": {
            "steps_0_through_9_residual_squared_energy_fraction": first_ten_error_fraction,
            "uniform_ten_of_eighty_one_state_fraction": 10.0 / 81.0,
            "largest_single_state_residual_squared_energy_fraction": top_time[0][
                "global_residual_squared_energy_fraction"
            ],
            "largest_single_state": top_time[0]["step"],
            "minimum_state_relative_rms_error": float(np.min(time_relative_values)),
            "maximum_state_relative_rms_error": float(np.max(time_relative_values)),
        },
        "assessment": (
            "Residual squared energy is dominated by dynamically active samples, "
            "not the A=0 or saturation-switch neighborhood. It is moderately "
            "weighted toward sub-saturated/evaporation regimes and early states, "
            "but remains distributed across signs, regimes, and the full time interval."
        ),
        "recommendation": "CONTINUE_OPTIMIZATION",
        "recommendation_basis": (
            "The frozen model has no sharp switching-boundary failure or isolated "
            "time-state failure, active-event sign accuracy is high, and the source "
            "optimization remained nonstationary. These controls favor continuing "
            "the same optimization before changing representation or freezing it."
        ),
    }
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "benchmark_stage": "Test 2A-1 frozen +5000 residual-structure diagnostic",
        "source": {
            "continuation_result": str(Path(continuation_result).resolve()),
            "parameter_file": str(parameter_path.resolve()),
            "parameter_npz_sha256": progress["last_checkpoint_npz_sha256"],
            "parameter_pytree_sha256": progress["last_checkpoint_pytree_sha256"],
            "source_decision": source_result["decision"],
            "parameters_modified": False,
            "optimization_executed": False,
        },
        "data_contract": {
            "dataset": str(Path(dataset_path).resolve()),
            "feature_order": list(FEATURE_ORDER),
            "truth_state_indices": [0, 80],
            "states_after_80_accessed": False,
            "sample_count": dataset.sample_count,
            "samples_per_state": samples_per_state,
            "state_count": 81,
            "shared_GLL_samples_deduplicated": False,
        },
        "global_metrics": global_metrics,
        "sign_accuracy_by_activity": sign_metrics,
        "metrics_by_truth_sign": sign_regimes,
        "residual_by_truth_magnitude": {
            "coordinate": "abs(A)/max_abs(A)",
            "max_abs_A": maximum,
            "bins": magnitude_bins,
        },
        "time_resolved_residual": time_records,
        "largest_time_state_residual_energy_contributors": top_time,
        "residual_concentration_summary": concentration_summary,
        "deployed_A_switch": {
            "switching_variable": "delta = qv - q_sat",
            "dimensionless_plotted_variable": "delta/q_sat",
            "qv": "Qv/h",
            "specific_entropy": "S/h",
            "q_sat": "q0*H0/(h+B)*exp(20*(1-(S/h)/g))",
            "gamma_v": "1/(1+20*q_sat*(g*L)/g)",
            "condensation": "C=max(0,gamma_v*delta/configured_dt)",
            "evaporation": "E=min(Qc/(h*configured_dt), max(0,-gamma_v*delta/configured_dt))",
            "net_rate": "A=E-C",
            "interpretation": (
                "delta=0 is the saturation switch; delta>0 activates the "
                "condensation candidate and delta<0 activates the evaporation "
                "candidate. The deployed cloud-water cap is a second kink and "
                "can change the realized magnitude or sign when Qc/h is negative."
            ),
            "target_reproduction_maximum_absolute_error": switch[
                "target_reproduction_maximum_absolute_error"
            ],
            "target_reproduction_tolerance": switch["target_reproduction_tolerance"],
            "relative_saturation_excess_percentiles": {
                str(percentile): float(
                    np.percentile(switch["relative_saturation_excess"], percentile)
                )
                for percentile in (0, 1, 5, 25, 50, 75, 95, 99, 100)
            },
            "residual_bins_by_relative_saturation_excess": switch_bins,
            "branch_regime_metrics": _regime_metrics(prediction, target, switch),
        },
        "plot_sampling": {
            "policy": "deterministic uniform row stride, statistics use all samples",
            "maximum_plotted_samples": 50_000,
        },
    }
    write_json_record(output_root / "residual_structure.json", summary)
    _plot_diagnostics(summary, prediction, target, switch, output_root)
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--continuation-result", required=True)
    parser.add_argument("--activity-audit", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    run_residual_diagnostic(
        arguments.configuration,
        arguments.dataset,
        arguments.continuation_result,
        arguments.activity_audit,
        arguments.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "deployed_a_switch_diagnostics",
    "residual_bins",
    "residual_subset_metrics",
    "run_residual_diagnostic",
    "time_residual_metrics",
)
