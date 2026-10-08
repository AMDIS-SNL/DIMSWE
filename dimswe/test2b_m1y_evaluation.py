"""Matched evaluation-only comparison of frozen Test-2B M1-X and M1-Y.

This module does not instantiate an optimizer.  It evaluates the accepted
historical M1-X checkpoint and newly trained M1-Y checkpoint on both X- and
Y-state analytical supports, applies the existing objective ladder, and runs
the exact existing representation-specific 160-step hybrid postprocessor for
M1-Y.  The already frozen standard M1-X rollout remains the comparison source.
"""

from __future__ import annotations

import argparse
from gc import collect
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .resolved_hidden_c0 import ResolvedPilotConfiguration, read_json_record, write_json_record
from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
from .test2b_m1y_campaign import (
    CAMPAIGN_ID,
    EXPECTED_HEAD,
    EXPECTED_HISTORICAL_PARAMETER_SHA,
    EXPECTED_NORMALIZATION,
    EXPECTED_STAGE,
    _array_sha256,
    _load_historical,
    _pack_evaluation,
    _postprefix,
    _resolved,
    load_m1y_configuration,
    load_m1y_preparation,
    repository_root,
    representation_target,
)
from .test2b_rain_learning import RainMLPConfiguration, load_parameters
from .test2b_rain_learning_campaign import (
    FixedObjective,
    OperatorObjective,
    _analytical_case,
    build_neural_case,
    load_preparation,
)
from .test2b_rain_learning_prepare import file_sha256
from .test2b_representation_a_postprocess import (
    _direct_metrics as _a_direct_metrics,
    _evaluate_rollout as _a_evaluate_rollout,
    _make_trajectory_objective,
)
from .test2b_representation_b_postprocess import (
    _direct_metrics as _b_direct_metrics,
    _evaluate_rollout as _b_evaluate_rollout,
    _truth_rate_arrays,
)
from .test2b_representation_c_postprocess import (
    _direct_metrics as _c_direct_metrics,
    _evaluate_rollout as _c_evaluate_rollout,
    _truth_source,
)
from .test2a_problem_b_campaign import ProblemBDiagnosticConfiguration


HISTORICAL_COMPARISONS = {
    "A": "external-results/test2b-rain-active-learning/production/representation-A/representation_a_final_comparison.json",
    "B": "external-results/test2b-rain-active-learning/production/representation-B/representation_b_final_comparison.json",
    "C": "external-results/test2b-rain-active-learning/production/representation-C/representation_c_final_comparison.json",
}
HISTORICAL_CHECKPOINTS = {
    representation: (
        f"external-results/test2b-rain-active-learning/production/"
        f"representation-{representation}/m1-seed0-m20-10k/final_parameters.npz"
    )
    for representation in "ABC"
}
M1Y_CHECKPOINTS = {
    representation: (
        f"external-results/m1y-test2b-20260828/production/"
        f"representation-{representation}/m1y-seed0-m20-10k/final_parameters.npz"
    )
    for representation in "ABC"
}
M1Y_FIT_RESULTS = {
    representation: (
        f"external-results/m1y-test2b-20260828/production/"
        f"representation-{representation}/m1y-seed0-m20-10k/fit_result.json"
    )
    for representation in "ABC"
}


def _load_extended_analytical_case(configuration):
    case, truth, adapter = _analytical_case(configuration)
    truth_root = _resolved(repository_root(), configuration["truth"]["run_directory"])
    for step in range(81, 161):
        truth[step] = case.state_from_values(
            np.load(
                truth_root / "restart" / f"step_{step:08d}.npy",
                allow_pickle=False,
            ),
            f"m1y_evaluation_truth_{step}",
        )
    return case, truth, adapter


def prepare_y_heldout(configuration_path, preparation_path, output_path):
    configuration_source, campaign = load_m1y_configuration(configuration_path)
    historical = _load_historical(configuration_source, campaign)
    training_metadata, training = load_m1y_preparation(preparation_path)
    destination = Path(output_path)
    sidecar = destination.with_suffix(".json")
    progress = destination.parent / "heldout_preparation_progress.json"
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite M1-Y held-out cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalization = historical["normalization"]
    case, truth, adapter = _load_extended_analytical_case(
        historical["configuration"]
    )
    features, target_a, target_r, qr_values = [], [], [], []
    started = perf_counter()
    for local_index, step in enumerate(range(81, 161)):
        y_state = _postprefix(case, truth[step], step)
        evaluated = adapter.evaluate(y_state, case.dt)
        physical = _pack_evaluation(evaluated)
        _, qr = adapter.interpolate_and_pack(
            y_state.sub(5), f"m1y_evaluation_Qr_{step}"
        )
        features.append(np.asarray(normalization.normalize_features(physical)))
        target_a.append(np.asarray(evaluated.rates["A"]).reshape(-1))
        target_r.append(np.asarray(evaluated.rates["R"]).reshape(-1))
        qr_values.append(np.asarray(qr).reshape(-1))
        write_json_record(progress, {
            "status": "in_progress",
            "campaign_id": CAMPAIGN_ID,
            "completed_truth_state": int(step),
            "completed_count": int(local_index + 1),
            "required_count": 80,
            "used_for_training_or_model_selection": False,
            "elapsed_wall_seconds": float(perf_counter() - started),
        })
    arrays = {
        "heldout_y_features": np.stack(features).astype(np.float64, copy=False),
        "heldout_y_A": np.stack(target_a).astype(np.float64, copy=False),
        "heldout_y_R": np.stack(target_r).astype(np.float64, copy=False),
        "heldout_y_Qr": np.stack(qr_values).astype(np.float64, copy=False),
    }
    expected_shapes = {
        "heldout_y_features": (80, 65536, 5),
        "heldout_y_A": (80, 65536),
        "heldout_y_R": (80, 65536),
        "heldout_y_Qr": (80, 65536),
    }
    for name, value in arrays.items():
        if value.shape != expected_shapes[name] or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid held-out Y array {name}: {value.shape}")
    if not np.array_equal(
        arrays["heldout_y_features"][..., 4],
        np.zeros_like(arrays["heldout_y_features"][..., 4]),
    ):
        raise ValueError("held-out flat-case B is not exact zero")
    incomplete = destination.with_name(destination.name + ".incomplete")
    with incomplete.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    incomplete.replace(destination)
    metadata = {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "source_head": EXPECTED_HEAD,
        "truth_state_indices": [81, 160],
        "state_count": 80,
        "samples_per_state": 65536,
        "feature_state": "Y_n*=P(X_n*)",
        "target_state": "Y_n*=P(X_n*)",
        "prefix_implementation": "case.helper.take_forward_step_cached(X_n*, n*dt, dt).boundary_states[-2]",
        "evaluation_only": True,
        "used_for_training_or_model_selection": False,
        "normalization_refitted_on_Y": False,
        "normalization_provenance_sha256": EXPECTED_NORMALIZATION["provenance_sha256"],
        "training_preparation": str(Path(preparation_path).resolve()),
        "training_preparation_sha256": file_sha256(preparation_path),
        "training_preparation_state_indices": training_metadata["training_state_indices"],
        "carrier_weights_sha256": training_metadata["arrays"]["carrier_weights"]["sha256"],
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for name, value in arrays.items()
        },
        "heldout_npz_sha256": file_sha256(destination),
        "wall_seconds": float(perf_counter() - started),
        "configuration": str(configuration_source),
        "configuration_sha256": file_sha256(configuration_source),
        "command": [sys.executable, "-m", "dimswe.test2b_m1y_evaluation", *sys.argv[1:]],
    }
    write_json_record(sidecar, metadata)
    write_json_record(progress, {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "completed_truth_state": 160,
        "completed_count": 80,
        "required_count": 80,
        "used_for_training_or_model_selection": False,
        "elapsed_wall_seconds": metadata["wall_seconds"],
        "heldout_npz": str(destination.resolve()),
        "heldout_npz_sha256": metadata["heldout_npz_sha256"],
    })
    return metadata


def load_y_evaluation_arrays(training_path, heldout_path, normalization):
    training_metadata, training = load_m1y_preparation(training_path)
    source = Path(heldout_path).resolve()
    heldout_metadata = read_json_record(source.with_suffix(".json"))
    if (
        heldout_metadata.get("status") != "complete"
        or heldout_metadata.get("campaign_id") != CAMPAIGN_ID
        or heldout_metadata.get("truth_state_indices") != [81, 160]
        or not heldout_metadata.get("evaluation_only")
        or heldout_metadata.get("used_for_training_or_model_selection")
        or heldout_metadata.get("normalization_refitted_on_Y")
        or heldout_metadata.get("training_preparation_sha256")
        != file_sha256(training_path)
        or file_sha256(source) != heldout_metadata.get("heldout_npz_sha256")
    ):
        raise ValueError("invalid M1-Y held-out evaluation cache")
    with np.load(source, allow_pickle=False) as archive:
        heldout = {
            name: np.array(archive[name], copy=True)
            for name in (
                "heldout_y_features", "heldout_y_A", "heldout_y_R",
                "heldout_y_Qr",
            )
        }
    for name, value in heldout.items():
        expected = heldout_metadata["arrays"][name]
        if (
            list(value.shape) != expected["shape"]
            or str(value.dtype) != expected["dtype"]
            or _array_sha256(value) != expected["sha256"]
        ):
            raise ValueError(f"held-out Y array changed: {name}")
    features = np.concatenate(
        (training["m1y_features"], heldout["heldout_y_features"]), axis=0
    )
    a = np.concatenate((training["m1y_A"], heldout["heldout_y_A"]), axis=0)
    r = np.concatenate((training["m1y_R"], heldout["heldout_y_R"]), axis=0)
    qr = np.concatenate((training["m1y_Qr"], heldout["heldout_y_Qr"]), axis=0)
    h = features[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
    return training_metadata, heldout_metadata, {
        "features": features,
        "A": a,
        "R": r,
        "Qr": qr,
        "h": h,
        "source": representation_target("C", features, a, r, normalization),
    }


def _verified_parameters(representation, root):
    historical_path = _resolved(root, HISTORICAL_CHECKPOINTS[representation])
    m1y_path = _resolved(root, M1Y_CHECKPOINTS[representation])
    historical_parameters, historical_sidecar = load_parameters(
        historical_path, representation
    )
    m1y_parameters, m1y_sidecar = load_parameters(m1y_path, representation)
    historical_result = read_json_record(historical_path.parent / "fit_result.json")
    m1y_result = read_json_record(_resolved(root, M1Y_FIT_RESULTS[representation]))
    if (
        historical_result.get("status") != "complete"
        or historical_result.get("stage") != "M1"
        or historical_result.get("accepted_iterations") != 10000
        or historical_result.get("final_parameter_pytree_sha256")
        != EXPECTED_HISTORICAL_PARAMETER_SHA[representation]
        or historical_sidecar["parameter_pytree_sha256"]
        != EXPECTED_HISTORICAL_PARAMETER_SHA[representation]
    ):
        raise ValueError(f"historical {representation}-M1-X artifact changed")
    expected_m1y_metadata = m1y_sidecar["metadata"]
    if (
        m1y_result.get("status") != "complete"
        or m1y_result.get("stage") != EXPECTED_STAGE
        or m1y_result.get("evaluation_state") != "Y_n*=P(X_n*)"
        or m1y_result.get("accepted_iterations") != 10000
        or m1y_result.get("final_parameter_pytree_sha256")
        != m1y_sidecar["parameter_pytree_sha256"]
        or expected_m1y_metadata.get("stage") != EXPECTED_STAGE
        or expected_m1y_metadata.get("evaluation_state") != "Y_n*=P(X_n*)"
    ):
        raise ValueError(f"invalid {representation}-M1-Y artifact")
    artifacts = {
        "M1-X": {
            **historical_result,
            "checkpoint": str(historical_path),
            "checkpoint_npz_sha256": file_sha256(historical_path),
            "checkpoint_sidecar_sha256": file_sha256(historical_path.with_suffix(".json")),
            "architecture": historical_sidecar["architecture"],
            "evaluation_state": "X_n*",
        },
        "M1-Y": {
            **m1y_result,
            "checkpoint": str(m1y_path),
            "checkpoint_npz_sha256": file_sha256(m1y_path),
            "checkpoint_sidecar_sha256": file_sha256(m1y_path.with_suffix(".json")),
            "architecture": m1y_sidecar["architecture"],
        },
    }
    return artifacts, {"M1-X": historical_parameters, "M1-Y": m1y_parameters}


def _a_direct(parameters, arrays, normalization, weights):
    training = {
        "x_features": arrays["features"][:81],
        "x_A": arrays["A"][:81],
    }
    heldout = (
        {step: arrays["features"][step] for step in range(81, 161)},
        {step: arrays["A"][step] for step in range(81, 161)},
    )
    return _a_direct_metrics(parameters, training, normalization, weights, heldout)


def _direct_metrics(representation, parameters, arrays, normalization, weights, comparison_scale):
    if representation == "A":
        return {"A": _a_direct(parameters, arrays, normalization, weights)}
    if representation == "B":
        a, r, activation = _b_direct_metrics(
            parameters, arrays, weights, normalization, comparison_scale
        )
        return {"A": a, "R": r, "R_activation": activation}
    source, activation = _c_direct_metrics(
        parameters, arrays, weights, normalization, comparison_scale
    )
    return {"source": source, "effective_R_activation": activation}


def _historical_comparison(representation, root, artifact_sha):
    path = _resolved(root, HISTORICAL_COMPARISONS[representation])
    record = read_json_record(path)
    if (
        record.get("status") != "complete"
        or record["artifacts"]["M1"]["final_parameter_pytree_sha256"]
        != artifact_sha
    ):
        raise ValueError(f"invalid historical {representation} comparison")
    return path, record


def _historical_direct(record, representation):
    if representation == "A":
        return {"A": record["direct_A"]["M1"]}
    if representation == "B":
        return {
            "A": record["direct_A"]["M1"],
            "R": record["direct_R"]["M1"],
            "R_activation": record["direct_R_activation_on_truth_states"]["M1"],
        }
    return {
        "source": record["direct_source_diagnostics"]["M1"],
        "effective_R_activation": record["direct_effective_rain_activation"]["M1"],
    }


def _selected_direct_discrepancy(representation, computed, frozen):
    if representation == "A":
        pairs = ((computed["A"]["TRAINING_OVERALL"], frozen["A"]["TRAINING_OVERALL"]),)
    elif representation == "B":
        pairs = (
            (computed["A"]["TRAINING_OVERALL"], frozen["A"]["TRAINING_OVERALL"]),
            (computed["R"]["TRAINING_OVERALL"]["all_samples"], frozen["R"]["TRAINING_OVERALL"]["all_samples"]),
        )
    else:
        pairs = tuple(
            (
                computed["source"]["TRAINING_OVERALL"]["component_errors"][name],
                frozen["source"]["TRAINING_OVERALL"]["component_errors"][name],
            )
            for name in ("S", "Qv", "Qc", "Qr")
        )
    keys = ("physical_RMS_error", "relative_RMS_error", "maximum_absolute_error")
    differences = [
        abs(float(left[key]) - float(right[key]))
        for left, right in pairs for key in keys
    ]
    return {
        "maximum_absolute_metric_difference": float(max(differences, default=0.0)),
        "metrics_compared": int(len(differences)),
        "passed": bool(max(differences, default=0.0) <= 1.0e-15),
    }


def evaluate_representation(
    configuration_path, preparation_path, heldout_path, representation,
    output_path,
):
    started = perf_counter()
    if representation not in "ABC":
        raise ValueError("evaluation supports only A, B, or C")
    configuration_source, campaign = load_m1y_configuration(configuration_path)
    historical = _load_historical(configuration_source, campaign)
    historical_metadata, normalization, historical_data, matrices = load_preparation(
        historical["historical_preparation_path"]
    )
    training_metadata, heldout_metadata, y_arrays = load_y_evaluation_arrays(
        preparation_path, heldout_path, normalization
    )
    root = repository_root()
    artifacts, parameters = _verified_parameters(representation, root)
    comparison_path, frozen = _historical_comparison(
        representation, root,
        artifacts["M1-X"]["final_parameter_pytree_sha256"],
    )
    weights = np.asarray(historical_data["carrier_weights"], dtype=np.float64)
    repeated_x_weights = np.broadcast_to(weights, historical_data["x_A"].shape)
    repeated_y_weights = np.broadcast_to(weights, y_arrays["A"][:81].shape)
    fixed = {
        "J_M1_X": OperatorObjective(
            representation, historical_data["x_features"],
            historical_data["x_A"], historical_data["x_R"],
            repeated_x_weights, normalization,
        ),
        "J_M1_Y": OperatorObjective(
            representation, y_arrays["features"][:81], y_arrays["A"][:81],
            y_arrays["R"][:81], repeated_y_weights, normalization,
        ),
        "J_M2_X": FixedObjective(
            representation, historical_data["x_features"],
            historical_data["x_A"], historical_data["x_R"], matrices,
            historical_metadata["m2x_denominator"], normalization,
        ),
        "J_H1": FixedObjective(
            representation, historical_data["y_features"],
            historical_data["y_A"], historical_data["y_R"], matrices,
            historical_metadata["common_horizon_denominator"] / 10000.0,
            normalization,
        ),
    }
    objective_matrix = {
        label: {name: objective.value(parameters[label]) for name, objective in fixed.items()}
        for label in ("M1-X", "M1-Y")
    }
    if objective_matrix["M1-X"]["J_M1_X"] != artifacts["M1-X"]["final_objective"]:
        raise RuntimeError("recomputed historical M1-X objective changed")
    if objective_matrix["M1-Y"]["J_M1_Y"] != artifacts["M1-Y"]["final_objective"]:
        raise RuntimeError("recomputed M1-Y objective changed")

    case, truth, _ = build_neural_case(
        historical["configuration"], normalization, representation,
        parameters["M1-Y"], 160,
    )
    x_arrays = _truth_rate_arrays(case, truth, normalization)
    if representation == "A":
        # The accepted Representation-A postprocessor uses the frozen
        # preparation arrays for training states and independently evaluates
        # only held-out states.  Preserve that exact summation/input path for
        # the historical metric-parity gate.
        x_arrays["features"][:81] = historical_data["x_features"]
        x_arrays["A"][:81] = historical_data["x_A"]
        x_arrays["R"][:81] = historical_data["x_R"]
        x_arrays["h"][:81] = (
            historical_data["x_features"][..., 0]
            * normalization.input_scale[0]
            + normalization.input_offset[0]
        )
    x_arrays["source"] = _truth_source(x_arrays["A"], x_arrays["R"], x_arrays["h"])
    truth_root = _resolved(root, historical["configuration"]["truth"]["run_directory"])
    truth_metadata = read_json_record(truth_root / "metadata.json")
    truth_audit = read_json_record(truth_root / "rain_activity_audit.json")
    comparison_scale = float(truth_audit["activity_tolerance"]["comparison_rate_scale"])
    direct = {
        state: {
            label: _direct_metrics(
                representation, parameters[label], arrays, normalization,
                weights, comparison_scale,
            )
            for label in ("M1-X", "M1-Y")
        }
        for state, arrays in (("X", x_arrays), ("Y", y_arrays))
    }
    frozen_direct = _historical_direct(frozen, representation)
    frozen_parity = _selected_direct_discrepancy(
        representation, direct["X"]["M1-X"], frozen_direct
    )
    if not frozen_parity["passed"]:
        raise RuntimeError(f"historical direct metric parity failed: {frozen_parity}")

    del fixed, repeated_x_weights, repeated_y_weights, historical_data, matrices, x_arrays, y_arrays
    collect()
    for horizon in (2, 5):
        print(f"Representation {representation} M1-Y H{horizon} objective", flush=True)
        objective = _make_trajectory_objective(
            case, truth, historical_metadata, horizon
        )
        objective_matrix["M1-Y"][f"J_H{horizon}"] = objective.value(parameters["M1-Y"])
        objective.clear_parameter_tape()
        del objective
        collect()
    objective_matrix["M1-X"]["J_H2"] = frozen["objective_matrix"]["M1"]["J_H2"]
    objective_matrix["M1-X"]["J_H5"] = frozen["objective_matrix"]["M1"]["J_H5"]

    truth_diagnostics = [
        read_json_record(truth_root / "diagnostics" / f"step_{step:08d}.json")
        for step in range(161)
    ]
    pilot = ResolvedPilotConfiguration(**{
        **truth_metadata["configuration"],
        "output_directory": f"/tmp/test2b-m1y-{representation}-evaluation-no-output",
    })
    diagnostic_configuration = ProblemBDiagnosticConfiguration.from_resolved_pilot(pilot)
    print(f"Representation {representation} M1-Y autonomous", flush=True)
    if representation == "A":
        autonomous_m1y = _a_evaluate_rollout(
            case, truth, parameters["M1-Y"], weights,
            diagnostic_configuration, truth_diagnostics, truth_audit, "M1_Y",
        )
    elif representation == "B":
        autonomous_m1y = _b_evaluate_rollout(
            case, truth, parameters["M1-Y"], weights,
            diagnostic_configuration, truth_diagnostics, truth_audit, "M1_Y",
            normalization,
        )
    else:
        autonomous_m1y = _c_evaluate_rollout(
            case, truth, parameters["M1-Y"], weights,
            diagnostic_configuration, truth_diagnostics, truth_audit, "M1_Y",
            normalization,
        )
    print(f"Representation {representation} M1-Y autonomous complete", flush=True)
    result = {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "source_head": EXPECTED_HEAD,
        "representation": representation,
        "evaluation_only": True,
        "optimizer_instantiated": False,
        "truth_generated": False,
        "heldout_used_for_training_or_model_selection": False,
        "feature_order": list(RainMLPConfiguration(representation).to_record()["features"]),
        "normalization": normalization.to_record(),
        "normalization_refitted_on_Y": False,
        "artifacts": artifacts,
        "objective_matrix": objective_matrix,
        "direct_cross_evaluation": direct,
        "historical_standard_M1_X": {
            "comparison_path": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
            "direct_metrics": frozen_direct,
            "objective_matrix": frozen["objective_matrix"]["M1"],
            "autonomous": frozen["autonomous"]["M1"],
            "recomputed_direct_metric_parity": frozen_parity,
        },
        "standard_M1_Y": {
            "autonomous": autonomous_m1y,
        },
        "truth_reference": {
            "configuration_sha256": truth_metadata["configuration_sha256"],
            "rain_audit_sha256": file_sha256(truth_root / "rain_activity_audit.json"),
            "truth_manifest_sha256": campaign["historical"]["truth_manifest_sha256"],
            "training_steps": [0, 80],
            "heldout_steps": [81, 160],
            "activity_tolerance": truth_audit["activity_tolerance"],
            "rain_summary": truth_audit["summary"],
        },
        "M1_Y_training_preparation": {
            "path": str(Path(preparation_path).resolve()),
            "sha256": file_sha256(preparation_path),
            "state_indices": training_metadata["training_state_indices"],
        },
        "M1_Y_heldout_preparation": {
            "path": str(Path(heldout_path).resolve()),
            "sha256": file_sha256(heldout_path),
            "state_indices": heldout_metadata["truth_state_indices"],
            "evaluation_only": True,
        },
        "evaluation_wall_seconds": float(perf_counter() - started),
        "command": [sys.executable, "-m", "dimswe.test2b_m1y_evaluation", *sys.argv[1:]],
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite matched M1-X/M1-Y evaluation")
    write_json_record(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-heldout")
    prepare.add_argument("--configuration", required=True)
    prepare.add_argument("--preparation", required=True)
    prepare.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--configuration", required=True)
    evaluate.add_argument("--preparation", required=True)
    evaluate.add_argument("--heldout", required=True)
    evaluate.add_argument("--representation", choices=("A", "B", "C"), required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-heldout":
        prepare_y_heldout(args.configuration, args.preparation, args.output)
    else:
        evaluate_representation(
            args.configuration, args.preparation, args.heldout,
            args.representation, args.output,
        )


if __name__ == "__main__":
    main()
