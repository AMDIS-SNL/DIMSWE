"""Post-hoc signed total-water audit for completed Test-2A Problem-B fits.

This module is evaluation-only.  It loads immutable final parameters, replays
the accepted six-child autonomous step on truth support 0..80, and diagnoses
signed total-water creation/destruction separately from redistribution into
the learned ``Qr`` component.  It never constructs an optimizer.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .hidden_c0 import _copy_function
from .test2a_problem_b import load_problem_b_parameters
from .test2a_problem_b_campaign import (
    PRODUCTION_ARTIFACT_STAGES,
    _build_problem_b_case,
    _file_sha256,
    _verify_completed_training_artifact,
    load_preparation,
    load_problem_b_configuration,
)


EXPECTED_COMPARISON_SHA256 = (
    "c9bba696f957b34e40a1f95d29e645463bbdd4456144abd536b2fed4d24561d2"
)
EXPECTED_LABELS = tuple(PRODUCTION_ARTIFACT_STAGES)


def _canonical_sha256(record):
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json_atomic(path, record):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _extreme_record(values, steps, times, mode):
    values = np.asarray(values, dtype=np.float64)
    if mode == "maximum":
        index = int(np.argmax(values))
    elif mode == "minimum":
        index = int(np.argmin(values))
    elif mode == "maximum_absolute":
        index = int(np.argmax(np.abs(values)))
    else:
        raise ValueError(f"unknown extreme mode {mode!r}")
    return {
        "value": float(values[index]),
        "step": int(steps[index]),
        "time": float(times[index]),
    }


def _signed_extreme_record(values, steps, times, sign):
    values = np.asarray(values, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    if sign == "positive":
        eligible = np.flatnonzero(values > 0.0)
        if eligible.size == 0:
            return None
        index = int(eligible[np.argmax(values[eligible])])
    elif sign == "negative":
        eligible = np.flatnonzero(values < 0.0)
        if eligible.size == 0:
            return None
        index = int(eligible[np.argmin(values[eligible])])
    else:
        raise ValueError(f"unknown signed extreme {sign!r}")
    return {
        "value": float(values[index]),
        "step": int(steps[index]),
        "time": float(times[index]),
    }


def summarize_state_drift(values, truth_values, steps, times, normalization):
    """Summarize signed model drift after subtracting the truth drift."""
    model = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth_values, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    if not (model.shape == truth.shape == steps.shape == times.shape):
        raise ValueError("state-drift series have inconsistent shapes")
    if model.ndim != 1 or model.size < 2:
        raise ValueError("state-drift series must contain at least two boundaries")
    scale = float(normalization)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("state-drift normalization must be positive")
    model_drift = model - model[0]
    truth_drift = truth - truth[0]
    error = model_drift - truth_drift
    relative = error / scale
    return {
        "initial_model_total_water": float(model[0]),
        "final_model_total_water": float(model[-1]),
        "final_signed_model_drift": float(model_drift[-1]),
        "final_signed_truth_relative_drift": float(error[-1]),
        "final_relative_signed_truth_relative_drift": float(relative[-1]),
        "maximum_positive_truth_relative_drift": _signed_extreme_record(
            error, steps, times, "positive"
        ),
        "most_negative_truth_relative_drift": _signed_extreme_record(
            error, steps, times, "negative"
        ),
        "maximum_absolute_truth_relative_drift": _extreme_record(
            error, steps, times, "maximum_absolute"
        ),
        "maximum_relative_absolute_truth_relative_drift": float(
            np.max(np.abs(relative))
        ),
    }


def summarize_source(source_integrals, source_steps, source_times, dt):
    """Summarize signed applied child-6 source integrals."""
    values = np.asarray(source_integrals, dtype=np.float64)
    steps = np.asarray(source_steps, dtype=np.int64)
    times = np.asarray(source_times, dtype=np.float64)
    if not (values.shape == steps.shape == times.shape):
        raise ValueError("source-integral series have inconsistent shapes")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("source-integral series must be nonempty")
    step_size = float(dt)
    return {
        "final_applied_source_integral": float(values[-1]),
        "maximum_positive_applied_source_integral": _signed_extreme_record(
            values, steps, times, "positive"
        ),
        "most_negative_applied_source_integral": _signed_extreme_record(
            values, steps, times, "negative"
        ),
        "time_integrated_signed_source_defect": float(step_size * np.sum(values)),
        "time_integrated_absolute_source_defect": float(
            step_size * np.sum(np.abs(values))
        ),
    }


def _truth_summary(water, qr, steps, times):
    water = np.asarray(water, dtype=np.float64)
    drift = water - water[0]
    relative = drift / water[0]
    return {
        "initial_total_water": float(water[0]),
        "final_total_water": float(water[-1]),
        "total_water_by_step": water.tolist(),
        "signed_drift_by_step": drift.tolist(),
        "maximum_absolute_drift": _extreme_record(
            drift, steps, times, "maximum_absolute"
        ),
        "maximum_relative_absolute_drift": float(np.max(np.abs(relative))),
        "final_total_Qr_mass": float(qr[-1]),
        "maximum_absolute_total_Qr_mass": float(np.max(np.abs(qr))),
        "total_Qr_mass_by_step": np.asarray(qr, dtype=np.float64).tolist(),
    }


def _carrier_weights(dataset):
    weights = np.asarray(dataset.spatial_weights, dtype=np.float64)
    if weights.size % 81 != 0:
        raise ValueError("Problem-B carrier weights do not contain 81 states")
    blocks = weights.reshape(81, -1)
    if not np.array_equal(blocks, np.broadcast_to(blocks[0], blocks.shape)):
        raise ValueError("fixed broken-GLL weights differ between truth states")
    return np.array(blocks[0], copy=True)


def _integrate_state(case, state, component):
    from firedrake import assemble

    if component == "water":
        expression = state.sub(3) + state.sub(4) + state.sub(5)
    elif component == "Qr":
        expression = state.sub(5)
    else:
        raise ValueError(f"unknown state integral component {component!r}")
    return float(assemble(expression * case.model.spaces.dx))


def _integrate_source_with_weights(source_density, weights):
    water = sum(
        np.asarray(source_density[name], dtype=np.float64)
        for name in ("Qv", "Qc", "Qr")
    )
    flattened = water.reshape(-1)
    if flattened.shape != weights.shape:
        raise ValueError("source samples and carrier weights are inconsistent")
    return float(weights @ flattened), water


def _integrate_source_with_firedrake(case, water_values, name):
    from firedrake import assemble

    helper = case.helper.moist_helper.primal_helper
    carrier = helper.unpack_carrier(water_values, name)
    return float(assemble(carrier * case.model.spaces.dx))


def _classification(final_error, threshold, maximum_qr_mass):
    if final_error > threshold:
        water = "artificial_net_water_creation"
    elif final_error < -threshold:
        water = "artificial_net_water_destruction"
    else:
        water = "net_water_drift_at_numerical_floor"
    rain = (
        "spurious_Qr_partition_present"
        if abs(float(maximum_qr_mass)) > threshold
        else "Qr_partition_at_numerical_floor"
    )
    return {"water_budget": water, "rain_partition": rain}


def _write_csv(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "network",
        "step",
        "time",
        "W_model",
        "W_truth",
        "signed_water_drift",
        "truth_signed_water_drift",
        "signed_water_drift_error",
        "relative_signed_water_drift_error",
        "applied_source_step",
        "applied_source_time",
        "applied_source_integral",
        "cumulative_integrated_source_defect",
        "cumulative_integrated_absolute_source_defect",
        "total_Qr_mass",
        "truth_total_Qr_mass",
    )
    temporary = destination.with_name(destination.name + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _write_plots(directory, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    selected = ("M1", "H1", "H2", "H5")
    by_label = {
        label: [row for row in rows if row["network"] == label]
        for label in selected
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for label, records in by_label.items():
        axis.plot(
            [row["time"] for row in records],
            [row["relative_signed_water_drift_error"] for row in records],
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("time")
    axis.set_ylabel("signed total-water drift / W_truth(0)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "signed_relative_total_water_drift_vs_time.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for label, records in by_label.items():
        axis.plot(
            [row["time"] for row in records],
            [row["total_Qr_mass"] for row in records],
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("time")
    axis.set_ylabel("integrated Qr")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "total_Qr_mass_vs_time.png", dpi=180)
    plt.close(fig)


def signed_water_budget(
    configuration_path,
    preparation_path,
    comparison_path,
    output_path,
    csv_path,
    plot_directory,
):
    """Replay completed artifacts and write the signed-water audit."""
    configuration = load_problem_b_configuration(configuration_path)
    comparison_path = Path(comparison_path)
    comparison_sha = _file_sha256(comparison_path)
    if comparison_sha != EXPECTED_COMPARISON_SHA256:
        raise ValueError("authoritative Problem-B comparison SHA256 changed")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("status") != "complete" or comparison.get(
        "states_after_80_accessed", True
    ):
        raise ValueError("Problem-B comparison is incomplete or violates truth support")
    if set(comparison["artifacts"]) != set(EXPECTED_LABELS):
        raise ValueError("Problem-B comparison does not contain the six frozen models")

    preparation_metadata, dataset, _, _ = load_preparation(preparation_path)
    weights = _carrier_weights(dataset)
    artifact_audit = {}
    parameters = {}
    for label in EXPECTED_LABELS:
        path = comparison["artifacts"][label]["parameter_file"]
        artifact_audit[label] = _verify_completed_training_artifact(label, path)
        parameters[label], _, sidecar = load_problem_b_parameters(path)
        if sidecar["parameter_pytree_sha256"] != comparison["artifacts"][label][
            "parameter_pytree_sha256"
        ]:
            raise ValueError(f"{label} comparison and parameter SHA differ")

    first_parameters = parameters[EXPECTED_LABELS[0]]
    case, truth, _, _ = _build_problem_b_case(
        configuration, dataset.normalization, first_parameters, 80
    )
    steps = np.arange(81, dtype=np.int64)
    times = np.asarray(
        [case.t0 + int(step) * case.dt for step in steps], dtype=np.float64
    )
    truth_water = np.asarray(
        [_integrate_state(case, truth[int(step)], "water") for step in steps],
        dtype=np.float64,
    )
    truth_qr = np.asarray(
        [_integrate_state(case, truth[int(step)], "Qr") for step in steps],
        dtype=np.float64,
    )
    truth_summary = _truth_summary(truth_water, truth_qr, steps, times)
    numerical_floor = max(
        float(truth_summary["maximum_absolute_drift"]["value"]),
        float(128.0 * np.finfo(np.float64).eps * abs(truth_water[0])),
    )

    records = {}
    csv_rows = []
    maximum_source_assembly_discrepancy = 0.0
    maximum_stored_water_discrepancy = 0.0
    for label in EXPECTED_LABELS:
        print(f"signed-water evaluation: {label}", flush=True)
        generated = [_copy_function(truth[0], f"problem_b_water_{label}_0")]
        source_integrals = []
        source_steps = []
        source_times = []
        source_increment = []
        moist_state_increment = []
        prefix_state_increment = []
        moist_source_closure = []
        water_values = [_integrate_state(case, generated[0], "water")]
        qr_values = [_integrate_state(case, generated[0], "Qr")]
        with case.physical_c0(float(configuration["truth"]["c0"])):
            for step in range(80):
                time = float(case.t0 + step * case.dt)
                cache = case.helper.take_forward_step_cached(
                    generated[-1], time, case.dt,
                    neural_parameters=parameters[label],
                )
                moist = cache.children[-1].cache
                source_integral, source_values = _integrate_source_with_weights(
                    moist.source_density, weights
                )
                assembled_integral = _integrate_source_with_firedrake(
                    case, source_values, f"problem_b_water_source_{label}_{step}"
                )
                maximum_source_assembly_discrepancy = max(
                    maximum_source_assembly_discrepancy,
                    abs(source_integral - assembled_integral),
                )
                water_before = _integrate_state(case, cache.state_in, "water")
                water_prefix = _integrate_state(case, moist.state_in, "water")
                water_after = _integrate_state(case, moist.state_out, "water")
                source_integrals.append(source_integral)
                source_steps.append(step)
                source_times.append(time)
                source_increment.append(float(case.dt * source_integral))
                moist_state_increment.append(water_after - water_prefix)
                prefix_state_increment.append(water_prefix - water_before)
                moist_source_closure.append(
                    (water_after - water_prefix) - case.dt * source_integral
                )
                generated.append(
                    _copy_function(
                        cache.state_out, f"problem_b_water_{label}_{step + 1}"
                    )
                )
                water_values.append(_integrate_state(case, generated[-1], "water"))
                qr_values.append(_integrate_state(case, generated[-1], "Qr"))

        water_values = np.asarray(water_values, dtype=np.float64)
        qr_values = np.asarray(qr_values, dtype=np.float64)
        stored = np.asarray(
            comparison["artifacts"][label][
                "autonomous_training_support_posthoc"
            ]["total_water_integral"],
            dtype=np.float64,
        )
        stored_difference = float(np.max(np.abs(water_values - stored)))
        maximum_stored_water_discrepancy = max(
            maximum_stored_water_discrepancy, stored_difference
        )
        stored_tolerance = float(
            128.0 * np.finfo(np.float64).eps * max(np.max(np.abs(stored)), 1.0)
        )
        if stored_difference > stored_tolerance:
            raise RuntimeError(
                f"{label} replay differs from accepted postprocessor water history"
            )

        drift = summarize_state_drift(
            water_values, truth_water, steps, times, truth_water[0]
        )
        source = summarize_source(
            source_integrals, source_steps, source_times, case.dt
        )
        cumulative_source = np.concatenate(
            ([0.0], np.cumsum(np.asarray(source_increment, dtype=np.float64)))
        )
        cumulative_absolute_source = np.concatenate(
            ([0.0], np.cumsum(np.abs(np.asarray(source_increment, dtype=np.float64))))
        )
        model_drift = water_values - water_values[0]
        truth_drift = truth_water - truth_water[0]
        drift_error = model_drift - truth_drift
        qr_summary = {
            "initial_total_Qr_mass": float(qr_values[0]),
            "final_total_Qr_mass": float(qr_values[-1]),
            "maximum_total_Qr_mass": _extreme_record(
                qr_values, steps, times, "maximum"
            ),
            "most_negative_total_Qr_mass": _extreme_record(
                qr_values, steps, times, "minimum"
            ),
            "maximum_absolute_total_Qr_mass": _extreme_record(
                qr_values, steps, times, "maximum_absolute"
            ),
            "maximum_absolute_Qr_t": float(
                comparison["artifacts"][label][
                    "autonomous_training_support_posthoc"
                ]["maximum_absolute_predicted_Qr_t"]
            ),
        }
        closure = {
            "time_integrated_source_minus_final_model_drift": float(
                cumulative_source[-1] - model_drift[-1]
            ),
            "time_integrated_source_minus_final_truth_relative_drift": float(
                cumulative_source[-1] - drift_error[-1]
            ),
            "summed_prefix_total_water_change": float(np.sum(prefix_state_increment)),
            "summed_moist_mass_solve_closure_residual": float(
                np.sum(moist_source_closure)
            ),
            "maximum_absolute_per_step_moist_mass_solve_closure_residual": float(
                np.max(np.abs(moist_source_closure))
            ),
            "maximum_replay_vs_stored_water_integral_discrepancy": stored_difference,
        }
        records[label] = {
            "parameter_pytree_sha256": artifact_audit[label][
                "parameter_pytree_sha256"
            ],
            "state_water_budget": drift,
            "source_water_budget": source,
            "rain_budget": qr_summary,
            "source_to_state_closure": closure,
            "classification": _classification(
                drift["final_signed_truth_relative_drift"],
                numerical_floor,
                qr_summary["maximum_absolute_total_Qr_mass"]["value"],
            ),
            "total_water_by_step": water_values.tolist(),
            "total_Qr_mass_by_step": qr_values.tolist(),
            "applied_source_integral_by_transition": list(
                map(float, source_integrals)
            ),
            "cumulative_integrated_source_defect_by_step": cumulative_source.tolist(),
        }
        for index, step in enumerate(steps):
            csv_rows.append(
                {
                    "network": label,
                    "step": int(step),
                    "time": float(times[index]),
                    "W_model": float(water_values[index]),
                    "W_truth": float(truth_water[index]),
                    "signed_water_drift": float(model_drift[index]),
                    "truth_signed_water_drift": float(truth_drift[index]),
                    "signed_water_drift_error": float(drift_error[index]),
                    "relative_signed_water_drift_error": float(
                        drift_error[index] / truth_water[0]
                    ),
                    "applied_source_step": "" if index == 0 else int(step - 1),
                    "applied_source_time": (
                        "" if index == 0 else float(source_times[index - 1])
                    ),
                    "applied_source_integral": (
                        "" if index == 0 else float(source_integrals[index - 1])
                    ),
                    "cumulative_integrated_source_defect": float(
                        cumulative_source[index]
                    ),
                    "cumulative_integrated_absolute_source_defect": float(
                        cumulative_absolute_source[index]
                    ),
                    "total_Qr_mass": float(qr_values[index]),
                    "truth_total_Qr_mass": float(truth_qr[index]),
                }
            )

    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A Problem B signed post-hoc water budget",
        "evaluation_only": True,
        "optimizer_or_training_instantiated": False,
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "autonomous_metrics_used_for_selection": False,
        "comparison_file": str(comparison_path.resolve()),
        "comparison_sha256": comparison_sha,
        "preparation_file": str(Path(preparation_path).resolve()),
        "preparation_sha256": preparation_metadata["preparation_npz_sha256"],
        "integration_contract": {
            "state": "Firedrake assemble((Qv+Qc+Qr)*dx) on the accepted spaces",
            "rain": "Firedrake assemble(Qr*dx) on the accepted spaces",
            "source": (
                "exact broken-CG3 GLL carrier mass weights applied to the actual "
                "child-6 source density retained by each complete-step cache"
            ),
            "source_samples_per_step": int(weights.size),
            "carrier_weight_sum": float(np.sum(weights)),
            "carrier_weight_sha256": _array_sha256(weights),
            "source_times": "child-6 source applied on transitions n->n+1 at t_n",
            "relative_state_drift_normalization": "W_truth(0)",
            "numerical_floor_definition": (
                "max(truth maximum absolute drift, 128 eps |W_truth(0)|)"
            ),
            "numerical_floor_absolute": numerical_floor,
            "maximum_GLL_weighted_vs_Firedrake_source_integral_discrepancy": (
                maximum_source_assembly_discrepancy
            ),
            "maximum_replay_vs_accepted_postprocessor_water_discrepancy": (
                maximum_stored_water_discrepancy
            ),
        },
        "truth": truth_summary,
        "artifact_audit": artifact_audit,
        "networks": records,
    }
    result["result_sha256"] = _canonical_sha256(result)
    _write_json_atomic(output_path, result)
    _write_csv(csv_path, csv_rows)
    _write_plots(plot_directory, csv_rows)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot-directory", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = signed_water_budget(
        args.configuration,
        args.preparation,
        args.comparison,
        args.output,
        args.csv,
        args.plot_directory,
    )
    headline = {
        "status": result["status"],
        "truth_maximum_absolute_drift": result["truth"][
            "maximum_absolute_drift"
        ],
        "networks": {
            label: {
                "final_signed_truth_relative_drift": record[
                    "state_water_budget"
                ]["final_signed_truth_relative_drift"],
                "classification": record["classification"],
            }
            for label, record in result["networks"].items()
        },
    }
    print(json.dumps(headline, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
