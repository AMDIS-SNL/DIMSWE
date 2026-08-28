"""Final evaluation-only postprocessor for Test2B Representation A.

This module has no optimizer entry point.  It verifies immutable completed
parameter artifacts, evaluates the frozen objective ladder, and performs the
common 160-step autonomous rollout with learned A and analytical R evaluated
on the current model state.
"""

from __future__ import annotations

import argparse
from gc import collect
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from .hidden_c0 import STATE_FIELDS, _copy_function, _serial_solver_parameters
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
    production_windows,
)


LABELS = (
    "M1",
    "M2-X-independent",
    "M1-to-M2-X",
    "H1",
    "H2",
    "H5",
)
DIRECTORIES = {
    "M1": "m1-seed0-m20-10k",
    "M2-X-independent": "m2x-seed0-m20-10k",
    "M1-to-M2-X": "m1-to-m2x-m20-5k",
    "H1": "h1-from-m1-m20-5k",
    "H2": "h2-from-h1-m20-20",
    "H5": "h5-from-h2-m20-20",
}
EXPECTED_STAGES = {
    "M1": "M1",
    "M2-X-independent": "M2-X",
    "M1-to-M2-X": "M2-X",
    "H1": "H1",
    "H2": "H2",
    "H5": "H5",
}
REGIMES = {
    "PRE_RAIN": tuple(range(0, 51)),
    "ONSET": tuple(range(51, 61)),
    "TRAINING_SUSTAINED_RAIN": tuple(range(61, 81)),
    "HELDOUT_MATURE_RAIN": tuple(range(81, 161)),
}


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predict_a(parameters, normalized_features, sigma_a):
    values = build_model("A")(parameters, jnp.asarray(normalized_features))
    return np.asarray(values[..., 0], dtype=np.float64) * float(sigma_a)


def _weighted_metrics(prediction, target, weights, sigma_a):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    weights = np.broadcast_to(
        np.asarray(weights, dtype=np.float64), prediction.shape
    ).reshape(-1)
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    total = float(np.sum(weights))
    mean = lambda values: float(np.sum(weights * values) / total)
    error = prediction - target
    rmse = float(np.sqrt(mean(error * error)))
    target_rms = float(np.sqrt(mean(target * target)))
    prediction_mean = mean(prediction)
    target_mean = mean(target)
    covariance = mean(
        (prediction - prediction_mean) * (target - target_mean)
    )
    prediction_std = float(np.sqrt(mean((prediction - prediction_mean) ** 2)))
    target_std = float(np.sqrt(mean((target - target_mean) ** 2)))
    dot = mean(prediction * target)
    prediction_rms = float(np.sqrt(mean(prediction * prediction)))
    return {
        "normalized_RMS_error": rmse / float(sigma_a),
        "relative_RMS_error": rmse
        / max(target_rms, np.finfo(np.float64).tiny),
        "physical_RMS_error": rmse,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "signed_mass_weighted_bias": mean(error),
        "correlation": None
        if prediction_std == 0.0 or target_std == 0.0
        else covariance / (prediction_std * target_std),
        "cosine": dot
        / max(prediction_rms * target_rms, np.finfo(np.float64).tiny),
        "target_RMS": target_rms,
        "sample_count": int(error.size),
    }


def _summarize_metric_series(records, selected_steps):
    selected = [records[int(step)] for step in selected_steps]
    numerators = np.asarray([row["numerator"] for row in selected])
    denominators = np.asarray([row["denominator"] for row in selected])
    relative = np.asarray([row["relative_error"] for row in selected])
    return {
        "steps": [int(step) for step in selected_steps],
        "final": float(relative[-1]),
        "maximum": float(np.max(relative)),
        "maximum_step": int(selected_steps[int(np.argmax(relative))]),
        "accumulated": float(
            np.sqrt(np.sum(numerators) / np.sum(denominators))
        ),
    }


def _relative_record(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "absolute_error": float(np.sqrt(numerator)),
        "reference_norm": float(np.sqrt(denominator)),
        "relative_error": float(np.sqrt(numerator / denominator)),
    }


def _verify_artifacts(root):
    verified = {}
    parameters = {}
    for label in LABELS:
        directory = Path(root) / DIRECTORIES[label]
        result_path = directory / "fit_result.json"
        progress_path = directory / "fit_progress.json"
        parameter_path = directory / "final_parameters.npz"
        if not all(path.is_file() for path in (result_path, progress_path, parameter_path)):
            raise FileNotFoundError(f"incomplete final artifact {label}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if result != progress or result.get("status") != "complete":
            raise RuntimeError(f"{label} final progress/result mismatch")
        if result.get("stage") != EXPECTED_STAGES[label]:
            raise ValueError(f"{label} stage identity changed")
        if "MAXITER" not in result.get("termination_reason", ""):
            raise ValueError(f"{label} did not terminate at its recorded cap")
        loaded, sidecar = load_parameters(parameter_path, "A")
        if sidecar["parameter_pytree_sha256"] != result["final_parameter_pytree_sha256"]:
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


def _direct_truth_arrays(case, truth, normalization, weights):
    helper = case.helper.moist_helper
    primal = helper.primal_helper
    _, topography = primal.interpolate_and_pack(
        primal.term.B, "test2b_representation_a_reference_B"
    )
    fields = {"B": topography}
    parameters = primal._parameters(None)
    features = {}
    targets = {}
    for step in range(81, 161):
        packed = helper.state_interpolation(truth[step])
        rates = moist_rates_jax(packed, fields, parameters)
        physical = np.stack(
            tuple(np.asarray(packed[name]).reshape(-1) for name in ("h", "S", "Qv", "Qc"))
            + (np.asarray(topography).reshape(-1),),
            axis=-1,
        )
        features[step] = np.asarray(
            normalization.normalize_features(physical), dtype=np.float64
        )
        targets[step] = np.asarray(rates["A"], dtype=np.float64).reshape(-1)
        if targets[step].shape != np.asarray(weights).shape:
            raise ValueError("held-out rate samples and carrier weights differ")
    return features, targets


def _direct_metrics(parameters, data, normalization, weights, heldout):
    sigma = float(normalization.sigma_a)
    training_prediction = np.stack(
        [
            _predict_a(parameters, data["x_features"][step], sigma)
            for step in range(81)
        ]
    )
    training_target = np.asarray(data["x_A"], dtype=np.float64)
    result = {
        "TRAINING_OVERALL": _weighted_metrics(
            training_prediction, training_target,
            np.broadcast_to(weights, training_target.shape), sigma,
        )
    }
    for name, steps in REGIMES.items():
        if name == "HELDOUT_MATURE_RAIN":
            prediction = np.stack(
                [_predict_a(parameters, heldout[0][step], sigma) for step in steps]
            )
            target = np.stack([heldout[1][step] for step in steps])
        else:
            prediction = training_prediction[np.asarray(steps)]
            target = training_target[np.asarray(steps)]
        result[name] = _weighted_metrics(
            prediction, target, np.broadcast_to(weights, target.shape), sigma
        )
        result[name]["state_indices"] = [int(steps[0]), int(steps[-1])]
    return result


def _make_trajectory_objective(case, truth, metadata, horizon):
    from .test2a_trajectory import GlobalMixedMassMetric, NeuralTrajectoryObjective

    denominator = float(metadata["common_horizon_denominator"])
    metric = GlobalMixedMassMetric(
        case.helper,
        denominator,
        denominator_sha256=metadata["denominator_fingerprints"]["H1-H2-H5"],
    )
    return NeuralTrajectoryObjective(
        case,
        truth,
        production_windows(horizon),
        metric=metric,
        c0=0.14,
        use_fixed_prefix=True,
    )


def _field_integral(case, field):
    from firedrake import assemble

    return float(assemble(field * case.model.spaces.dx))


def _source_record(case, moist, step, weights, comparison_scale):
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
        moist.stage_state.sub(5), f"test2b_representation_a_Qr_{step}"
    )
    rain = rain_activity_diagnostic(
        predicted_r, h, qr, case.dt, comparison_scale,
        float64_scale_multiplier=64.0,
        physical_increment_relative_threshold=1.0e-12,
    )
    result = {
        "step": int(step),
        "time": float(case.t0 + step * case.dt),
        "applied_to_trajectory": bool(step < 160),
        "specific_Qc_maximum": float(np.max(qc)),
        "specific_Qc_rms": float(np.sqrt(np.mean(qc * qc))),
        "A_metrics": _weighted_metrics(
            predicted_a, analytical_a, weights,
            sigma_a=9.052258655848717e-8,
        ),
        "R_maximum_absolute": rain["maximum_absolute_R"],
        "R_rms": rain["rms_R"],
        "R_exact_nonzero_fraction": rain["exact_nonzero_fraction"],
        "physically_meaningful_R_fraction": rain["physically_meaningful_fraction"],
        "maximum_absolute_Qr_increment": rain["maximum_absolute_Qr_increment"],
        "analytical_R_maximum_absolute_discrepancy": float(
            np.max(np.abs(predicted_r - analytical_r))
        ),
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
    exact = [row for row in records if row["R_maximum_absolute"] > 0.0]
    meaningful = [
        row for row in records
        if row["physically_meaningful_R_fraction"] > 0.0
    ]
    peak_qc = max(records, key=lambda row: row["specific_Qc_maximum"])
    maximum_r = max(row["R_maximum_absolute"] for row in records)
    integrated = sum(
        row["rain_source_mass_increment"] for row in records
        if row["applied_to_trajectory"]
    )
    qr = np.asarray(boundary["Qr_mass"], dtype=np.float64)
    truth_summary = truth_audit["summary"]
    exact_step = None if not exact else exact[0]["step"]
    meaningful_step = None if not meaningful else meaningful[0]["step"]
    return {
        "first_exact_nonzero_R_step": exact_step,
        "first_exact_nonzero_R_time": None
        if exact_step is None else float(exact_step * 100.0),
        "first_physically_meaningful_R_step": meaningful_step,
        "first_physically_meaningful_R_time": None
        if meaningful_step is None else float(meaningful_step * 100.0),
        "meaningful_onset_time_error_vs_truth": None
        if meaningful_step is None else float(meaningful_step * 100.0 - 5100.0),
        "maximum_specific_Qc": peak_qc["specific_Qc_maximum"],
        "maximum_specific_Qc_step": peak_qc["step"],
        "maximum_specific_Qc_time": peak_qc["time"],
        "maximum_specific_Qc_error": peak_qc["specific_Qc_maximum"]
        - truth_summary["maximum_specific_Qc"],
        "maximum_R": maximum_r,
        "maximum_R_error": maximum_r - truth_summary["maximum_R"],
        "time_integrated_rain_source_mass": float(integrated),
        "time_integrated_rain_source_mass_error": float(
            integrated - truth_summary["time_integrated_rain_source_mass"]
        ),
        "final_Qr_mass": float(qr[-1]),
        "maximum_Qr_mass": float(np.max(qr)),
        "final_Qr_mass_error": float(
            qr[-1] - truth_audit["records"][-1]["rain_water_mass"]
        ),
        "maximum_Qr_mass_error": float(
            np.max(qr) - truth_summary["maximum_rain_water_mass"]
        ),
        "records": records,
    }


def _summarize_boundary(case, boundary, mixed_records, field_records):
    overall_steps = tuple(range(0, 161))
    mixed = {"ALL": _summarize_metric_series(mixed_records, overall_steps)}
    for name, steps in REGIMES.items():
        mixed[name] = _summarize_metric_series(mixed_records, steps)
    fieldwise = {}
    for field in STATE_FIELDS:
        values = field_records[field]
        all_relative = [values[step] for step in overall_steps]
        fieldwise[field] = {
            "final": all_relative[-1],
            "maximum": None
            if not any(value is not None for value in all_relative)
            else float(max(value for value in all_relative if value is not None)),
            "by_regime": {},
        }
        for name, steps in REGIMES.items():
            selected = [values[step] for step in steps]
            finite = [value for value in selected if value is not None]
            fieldwise[field]["by_regime"][name] = {
                "final": selected[-1],
                "maximum": None if not finite else float(max(finite)),
            }
    water = np.asarray(boundary["total_water_mass"])
    thermo = np.asarray(boundary["S_minus_beta2_Qv_mass"])
    return mixed, fieldwise, {
        "initial_total_water_mass": float(water[0]),
        "final_total_water_mass": float(water[-1]),
        "relative_maximum_total_water_drift": float(
            np.max(np.abs(water - water[0])) / abs(water[0])
        ),
        "maximum_absolute_total_water_drift": float(
            np.max(np.abs(water - water[0]))
        ),
        "maximum_absolute_S_minus_beta2_Qv_drift": float(
            np.max(np.abs(thermo - thermo[0]))
        ),
        "partition": {
            name: {
                "initial": float(np.asarray(boundary[f"{name}_mass"])[0]),
                "final": float(np.asarray(boundary[f"{name}_mass"])[-1]),
                "minimum": float(np.min(boundary[f"{name}_mass"])),
                "maximum": float(np.max(boundary[f"{name}_mass"])),
            }
            for name in ("Qv", "Qc", "Qr")
        },
        "minimum_field_coefficients": {
            name: float(np.min(values))
            for name, values in boundary["minimum_field_coefficients"].items()
        },
        "all_state_coefficients_finite": bool(
            all(boundary["all_state_coefficients_finite"])
        ),
    }


def _evaluate_rollout(
    case, truth, parameters, weights, diagnostic_configuration,
    truth_diagnostics, truth_audit, label,
):
    evaluator = ResolvedDiagnosticEvaluator(case, diagnostic_configuration)
    current = _copy_function(truth[0], f"test2b_{label}_autonomous_0")
    zero = case.new_state(f"test2b_{label}_zero")
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
    a_accumulators = {
        name: {"prediction": [], "target": []}
        for name in (*REGIMES, "ALL")
    }
    maximum_source_residuals = {
        "water_maximum_absolute": 0.0,
        "S_minus_beta2_Qv_maximum_absolute": 0.0,
        "analytical_R_maximum_absolute_discrepancy": 0.0,
    }
    comparison_scale = float(
        truth_audit["activity_tolerance"]["comparison_rate_scale"]
    )
    for step in range(161):
        target = truth[step]
        numerator = _state_squared_difference(
            case, current, target, f"test2b_{label}_mixed_residual_{step}"
        )
        denominator = _state_squared_difference(
            case, target, zero, f"test2b_{label}_mixed_target_{step}"
        )
        mixed_records[step] = _relative_record(numerator, denominator)
        for index, name in enumerate(STATE_FIELDS):
            field_numerator, field_denominator = _field_squared_norms(
                case, current, target, index,
                f"test2b_{label}_{name}_{step}",
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
                cache.state_out, f"test2b_{label}_autonomous_{step + 1}"
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
            case, moist, step, weights, comparison_scale
        )
        source_records.append(record)
        invariants = record["source_invariants"]
        for name in (
            "water_maximum_absolute",
            "S_minus_beta2_Qv_maximum_absolute",
        ):
            maximum_source_residuals[name] = max(
                maximum_source_residuals[name], invariants[name]
            )
        maximum_source_residuals[
            "analytical_R_maximum_absolute_discrepancy"
        ] = max(
            maximum_source_residuals[
                "analytical_R_maximum_absolute_discrepancy"
            ],
            record["analytical_R_maximum_absolute_discrepancy"],
        )
        if step < 160:
            target_step = step + 1
            regime = next(
                name for name, steps in REGIMES.items()
                if target_step in steps
            )
            predicted_a = np.asarray(moist.rates["A"]).reshape(-1)
            analytical_a = np.asarray(
                moist_rates_jax(
                    moist.packed_state, moist.packed_fields, moist.parameters
                )["A"]
            ).reshape(-1)
            for name in (regime, "ALL"):
                a_accumulators[name]["prediction"].append(predicted_a)
                a_accumulators[name]["target"].append(analytical_a)
            current = next_state

    mixed, fieldwise, conservation = _summarize_boundary(
        case, boundary, mixed_records, field_records
    )
    steps = tuple(range(161))
    times = tuple(float(step * case.dt) for step in steps)
    flow = {}
    for key in (
        "kinetic_energy", "projected_enstrophy",
        "velocity_high_wavenumber_energy_fraction",
    ):
        flow[key] = _diagnostic_mismatch(
            [row[key] for row in predicted_diagnostics],
            [row[key] for row in truth_diagnostics],
            steps, times,
        )
    a_metrics = {
        name: _weighted_metrics(
            np.stack(values["prediction"]),
            np.stack(values["target"]),
            np.broadcast_to(
                weights, np.stack(values["target"]).shape
            ),
            9.052258655848717e-8,
        )
        for name, values in a_accumulators.items()
    }
    return {
        "mixed_state_error": mixed,
        "fieldwise_state_error": fieldwise,
        "rain": _summarize_rain(
            source_records, boundary, truth_audit
        ),
        "A_error_on_model_postprefix_states": a_metrics,
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
            "A", data["x_features"], data["x_A"], data["x_R"],
            repeated_weights, normalization,
        ),
        "J_M2_X": FixedObjective(
            "A", data["x_features"], data["x_A"], data["x_R"],
            matrices, metadata["m2x_denominator"], normalization,
        ),
        "J_H1": FixedObjective(
            "A", data["y_features"], data["y_A"], data["y_R"],
            matrices, metadata["common_horizon_denominator"] / 10000.0,
            normalization,
        ),
    }
    case, truth, _ = build_neural_case(
        configuration, normalization, "A", parameters["M1"], 160
    )
    heldout = _direct_truth_arrays(
        case, truth, normalization, weights
    )
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
        "output_directory": "/tmp/test2b-representation-a-postprocess-no-output",
    })
    diagnostic_configuration = ProblemBDiagnosticConfiguration.from_resolved_pilot(
        pilot
    )
    result = {
        "status": "in_progress",
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
        "autonomous": {},
    }
    for label in LABELS:
        print(f"Representation A evaluation: {label} fixed/direct", flush=True)
        local = parameters[label]
        result["objective_matrix"][label] = {
            name: objective.value(local) for name, objective in fixed.items()
        }
        result["direct_A"][label] = _direct_metrics(
            local, data, normalization, weights, heldout
        )
    del fixed, repeated_weights, data, matrices, heldout
    collect()
    for label in LABELS:
        local = parameters[label]
        for horizon in (2, 5):
            print(
                f"Representation A evaluation: {label} H{horizon} objective",
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
        print(f"Representation A evaluation: {label} autonomous", flush=True)
        result["autonomous"][label] = _evaluate_rollout(
            case, truth, local, weights, diagnostic_configuration,
            truth_diagnostics, truth_audit, label,
        )
        print(f"Representation A evaluation: {label} complete", flush=True)
    result["truth_reference"] = {
        "configuration_sha256": truth_metadata["configuration_sha256"],
        "rain_audit_sha256": _file_sha256(truth_root / "rain_activity_audit.json"),
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
