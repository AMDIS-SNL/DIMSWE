#!/usr/bin/env python3
"""Post-hoc fixed-array direct histories for the accepted Test-2B models.

No optimizer, prefix, timestep, or rollout is constructed.  Network outputs
are evaluated only at saved checkpoints on immutable cached X/Y arrays.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from portable_paths import (
    AUDIT_ROOT as AUDIT,
    M1Y_REPOSITORY as M1Y,
    PACKAGE_ROOT,
    REFERENCE_REPOSITORY as AUTH,
)

OUTPUT = PACKAGE_ROOT / "data"
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"
FEATURE_ORDER = ("h", "S", "Qv", "Qc", "B")
SOURCE_ORDER = ("S", "Qv", "Qc", "Qr")
BETA2 = 98.0616
DT = 100.0
INPUT_OFFSET = np.asarray(
    (749.6487720807651, 7376.434989735685, 1.4193153609575624,
     0.06015957787413514, 0.0), dtype=np.float64,
)
INPUT_SCALE = np.asarray(
    (16.913638066122523, 133.5602198531373, 0.21326095651874272,
     0.012412402653357142, 1.0), dtype=np.float64,
)
SIGMA_A = 9.052258655848717e-8
SIGMA_R = 1.9902871261559996e-11
SOURCE_SCALES = np.asarray(
    (0.006671477765500949, 6.803353979030477e-5,
     6.80335397581467e-5, 1.5076498196845062e-8), dtype=np.float64,
)
NORMALIZATION_SHA = "794e074b2d3149f58025a7e6a74856374d86adab1e3ee518a64fe6f30ff0dd79"
COMPARISON_RATE_SCALE = 8.528669963488885e-7
NUMERICAL_RATE_TOLERANCE = (
    64.0 * np.finfo(np.float64).eps * COMPARISON_RATE_SCALE
)
CHECKPOINT_RE = re.compile(r"checkpoint_(\d{9})_parameters\.npz$")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ScalarAccumulator:
    sum_weight: float = 0.0
    sum_error2: float = 0.0
    sum_target2: float = 0.0
    sum_prediction2: float = 0.0
    sum_prediction_target: float = 0.0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    maximum_absolute_error: float = 0.0
    sample_count: int = 0

    def update(self, prediction, target, weights) -> None:
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if prediction.shape != target.shape or weights.shape != target.shape:
            raise ValueError("scalar metric shape mismatch")
        if prediction.size == 0:
            return
        error = prediction - target
        self.sum_weight += float(np.sum(weights))
        self.sum_error2 += float(np.sum(weights * error * error))
        self.sum_target2 += float(np.sum(weights * target * target))
        self.sum_prediction2 += float(np.sum(weights * prediction * prediction))
        self.sum_prediction_target += float(np.sum(weights * prediction * target))
        self.sum_prediction += float(np.sum(weights * prediction))
        self.sum_target += float(np.sum(weights * target))
        self.maximum_absolute_error = max(
            self.maximum_absolute_error, float(np.max(np.abs(error)))
        )
        self.sample_count += int(error.size)

    def finish(self, scale: float) -> dict:
        if self.sum_weight <= 0.0:
            return {}
        prediction_mean = self.sum_prediction / self.sum_weight
        target_mean = self.sum_target / self.sum_weight
        prediction_var = max(
            self.sum_prediction2 / self.sum_weight - prediction_mean**2, 0.0
        )
        target_var = max(
            self.sum_target2 / self.sum_weight - target_mean**2, 0.0
        )
        covariance = (
            self.sum_prediction_target / self.sum_weight
            - prediction_mean * target_mean
        )
        rmse = float(np.sqrt(self.sum_error2 / self.sum_weight))
        target_rms = float(np.sqrt(self.sum_target2 / self.sum_weight))
        prediction_rms = float(np.sqrt(self.sum_prediction2 / self.sum_weight))
        return {
            "normalized_RMS_error": rmse / float(scale),
            "relative_RMS_error": rmse / max(target_rms, np.finfo(np.float64).tiny),
            "physical_RMS_error": rmse,
            "maximum_absolute_error": self.maximum_absolute_error,
            "signed_mass_weighted_bias": (
                self.sum_prediction - self.sum_target
            ) / self.sum_weight,
            "correlation": None
            if prediction_var == 0.0 or target_var == 0.0
            else covariance / np.sqrt(prediction_var * target_var),
            "cosine": self.sum_prediction_target
            / max(
                np.sqrt(self.sum_prediction2 * self.sum_target2),
                np.finfo(np.float64).tiny,
            ),
            "target_RMS": target_rms,
            "prediction_RMS": prediction_rms,
            "sample_count": self.sample_count,
        }


def model_label(run_id: str) -> str:
    suffix = run_id.split("-", 2)[2]
    return {
        "m1x": "M1-X",
        "m1y": "M1-Y",
        "m2x-independent": "M2-X-independent",
        "m2x-warm": "warm M2-X",
        "h1": "H1",
        "h2": "H2",
        "h5": "H5",
    }[suffix]


def accepted_label(label: str) -> str:
    return {
        "M1-X": "M1",
        "warm M2-X": "M1-to-M2-X",
    }.get(label, label)


def truth_source(a, r, h):
    return np.stack(
        (h * BETA2 * a, h * a, -h * (a + r), h * r), axis=-1
    )


def load_support(evaluation_state: str, support: str) -> dict:
    fixed_path = M1Y / (
        "external-results/test2b-rain-active-learning/preparation/"
        "fixed_learning_data.npz"
    )
    if evaluation_state == "X" and support == "training":
        with np.load(fixed_path, allow_pickle=False) as archive:
            features = np.array(archive["x_features"], copy=True)
            a = np.array(archive["x_A"], copy=True)
            r = np.array(archive["x_R"], copy=True)
            weights = np.array(archive["carrier_weights"], copy=True)
        with np.load(OUTPUT / "training_x_carriers_test2b.npz", allow_pickle=False) as archive:
            h = np.array(archive["training_x_h"], copy=True)
            qr = np.array(archive["training_x_Qr"], copy=True)
        first = 0
    elif evaluation_state == "X" and support == "heldout":
        with np.load(OUTPUT / "heldout_x_test2b.npz", allow_pickle=False) as archive:
            features = np.array(archive["heldout_x_features"], copy=True)
            a = np.array(archive["heldout_x_A"], copy=True)
            r = np.array(archive["heldout_x_R"], copy=True)
            h = np.array(archive["heldout_x_h"], copy=True)
            qr = np.array(archive["heldout_x_Qr"], copy=True)
            weights = np.array(archive["carrier_weights"], copy=True)
        first = 81
    elif evaluation_state == "Y" and support == "training":
        path = M1Y / "external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz"
        with np.load(path, allow_pickle=False) as archive:
            features = np.array(archive["m1y_features"], copy=True)
            a = np.array(archive["m1y_A"], copy=True)
            r = np.array(archive["m1y_R"], copy=True)
            qr = np.array(archive["m1y_Qr"], copy=True)
            weights = np.array(archive["carrier_weights"], copy=True)
        h = features[..., 0] * INPUT_SCALE[0] + INPUT_OFFSET[0]
        first = 0
    elif evaluation_state == "Y" and support == "heldout":
        path = M1Y / "external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz"
        with np.load(path, allow_pickle=False) as archive:
            features = np.array(archive["heldout_y_features"], copy=True)
            a = np.array(archive["heldout_y_A"], copy=True)
            r = np.array(archive["heldout_y_R"], copy=True)
            qr = np.array(archive["heldout_y_Qr"], copy=True)
        with np.load(fixed_path, allow_pickle=False) as archive:
            weights = np.array(archive["carrier_weights"], copy=True)
        h = features[..., 0] * INPUT_SCALE[0] + INPUT_OFFSET[0]
        first = 81
    else:
        raise ValueError((evaluation_state, support))
    if features.shape[1:] != (65536, 5):
        raise ValueError("support feature shape changed")
    if not all(value.dtype == np.float64 for value in (features, a, r, h, qr, weights)):
        raise TypeError("support dtype changed")
    if not np.all(np.isfinite(features)):
        raise ValueError("non-finite support features")
    return {
        "features": features,
        "A": a,
        "R": r,
        "h": h,
        "Qr": qr,
        "weights": weights,
        "state_first": first,
        "state_last": first + features.shape[0] - 1,
    }


def evaluate_checkpoint(representation, parameters, model, data) -> dict:
    weights = data["weights"]
    a_metric = ScalarAccumulator()
    r_all_metric = ScalarAccumulator()
    r_active_metric = ScalarAccumulator()
    component_metrics = [ScalarAccumulator() for _ in SOURCE_ORDER]
    effective_a_metric = ScalarAccumulator()
    effective_r_metric = ScalarAccumulator()
    operator_numerator = 0.0
    operator_denominator = 0.0
    total_weight = 0.0
    off_manifold_sum = 0.0
    normalized_prediction2 = 0.0
    normalized_target2 = 0.0
    normalized_dot = 0.0
    water_defect2 = 0.0
    thermo_defect2 = 0.0
    source_sample_count = 0
    negative_qr_count = 0
    positive_qr_count = 0
    truth_active_count = 0
    predicted_active_count = 0
    false_positive_count = 0
    false_negative_count = 0
    predicted_negative_count = 0
    predicted_exact_positive_count = 0
    predicted_exact_negative_count = 0
    first_predicted_active = None
    first_truth_active = None
    maximum_predicted_r = -np.inf
    minimum_predicted_r = np.inf
    physical_basis = np.asarray(
        ((BETA2, 1.0, -1.0, 0.0), (0.0, 0.0, -1.0, 1.0)),
        dtype=np.float64,
    ).T
    normalized_basis = physical_basis / SOURCE_SCALES[:, None]
    projection_map = normalized_basis @ np.linalg.inv(
        normalized_basis.T @ normalized_basis
    )

    for local_step, features in enumerate(data["features"]):
        raw = np.asarray(model(parameters, jnp.asarray(features)), dtype=np.float64)
        target_a = data["A"][local_step]
        target_r = data["R"][local_step]
        h = data["h"][local_step]
        qr = data["Qr"][local_step]
        if representation == "A":
            prediction_a = raw[:, 0] * SIGMA_A
            normalized_target = (target_a / SIGMA_A)[:, None]
            a_metric.update(prediction_a, target_a, weights)
        elif representation == "B":
            prediction_a = raw[:, 0] * SIGMA_A
            prediction_r = raw[:, 1] * SIGMA_R
            normalized_target = np.stack(
                (target_a / SIGMA_A, target_r / SIGMA_R), axis=-1
            )
            a_metric.update(prediction_a, target_a, weights)
        else:
            prediction_source = raw * SOURCE_SCALES
            target_source = truth_source(target_a, target_r, h)
            normalized_target = target_source / SOURCE_SCALES
            for index in range(4):
                component_metrics[index].update(
                    prediction_source[:, index], target_source[:, index], weights
                )
            coefficients = raw @ projection_map
            projected = coefficients @ normalized_basis.T
            residual = raw - projected
            prediction_a = coefficients[:, 0] / h
            prediction_r = coefficients[:, 1] / h
            effective_a_metric.update(prediction_a, target_a, weights)
            effective_r_metric.update(prediction_r, target_r, weights)
            local_weight = float(np.sum(weights))
            total_weight += local_weight
            off_manifold_sum += float(
                np.sum(weights[:, None] * residual * residual)
            )
            normalized_prediction2 += float(
                np.sum(weights[:, None] * raw * raw)
            )
            normalized_target2 += float(
                np.sum(weights[:, None] * normalized_target * normalized_target)
            )
            normalized_dot += float(
                np.sum(weights[:, None] * raw * normalized_target)
            )
            water = (
                prediction_source[:, 1]
                + prediction_source[:, 2]
                + prediction_source[:, 3]
            )
            thermo = prediction_source[:, 0] - BETA2 * prediction_source[:, 1]
            water_defect2 += float(np.sum(weights * water * water))
            thermo_defect2 += float(np.sum(weights * thermo * thermo))
            negative_qr_count += int(np.count_nonzero(prediction_source[:, 3] < 0.0))
            positive_qr_count += int(np.count_nonzero(prediction_source[:, 3] > 0.0))
            source_sample_count += int(prediction_source.shape[0])

        difference = raw - normalized_target
        operator_numerator += float(
            np.sum(weights[:, None] * difference * difference)
        )
        operator_denominator += float(
            np.sum(weights[:, None] * normalized_target * normalized_target)
        )

        if representation in ("B", "C"):
            qr_rms = float(np.sqrt(np.mean(qr * qr)))
            physical_tolerance = 1.0e-12 * qr_rms
            truth_active = (target_r > NUMERICAL_RATE_TOLERANCE) & (
                DT * h * target_r > physical_tolerance
            )
            predicted_active = (prediction_r > NUMERICAL_RATE_TOLERANCE) & (
                DT * h * prediction_r > physical_tolerance
            )
            predicted_negative = (prediction_r < -NUMERICAL_RATE_TOLERANCE) & (
                -DT * h * prediction_r > physical_tolerance
            )
            false_positive = predicted_active & ~truth_active
            false_negative = truth_active & ~predicted_active
            if representation == "B":
                r_all_metric.update(prediction_r, target_r, weights)
                r_active_metric.update(
                    prediction_r[truth_active], target_r[truth_active], weights[truth_active]
                )
            truth_active_count += int(np.count_nonzero(truth_active))
            predicted_active_count += int(np.count_nonzero(predicted_active))
            predicted_negative_count += int(np.count_nonzero(predicted_negative))
            predicted_exact_positive_count += int(np.count_nonzero(prediction_r > 0.0))
            predicted_exact_negative_count += int(np.count_nonzero(prediction_r < 0.0))
            false_positive_count += int(np.count_nonzero(false_positive))
            false_negative_count += int(np.count_nonzero(false_negative))
            maximum_predicted_r = max(maximum_predicted_r, float(np.max(prediction_r)))
            minimum_predicted_r = min(minimum_predicted_r, float(np.min(prediction_r)))
            global_step = data["state_first"] + local_step
            if first_predicted_active is None and np.any(predicted_active):
                first_predicted_active = global_step
            if first_truth_active is None and np.any(truth_active):
                first_truth_active = global_step

    result = {
        "operator": {
            "normalized_objective": operator_numerator / operator_denominator,
            "numerator": operator_numerator,
            "denominator": operator_denominator,
        },
        "A": a_metric.finish(SIGMA_A),
    }
    if representation == "B":
        total_count = int(data["features"].shape[0] * 65536)
        inactive_count = total_count - truth_active_count
        result.update({
            "R_all": r_all_metric.finish(SIGMA_R),
            "R_truth_active": r_active_metric.finish(SIGMA_R),
            "R_activation": {
                "truth_active_sample_count": truth_active_count,
                "truth_active_sample_fraction": truth_active_count / total_count,
                "predicted_meaningful_positive_fraction": predicted_active_count / total_count,
                "predicted_meaningful_negative_fraction": predicted_negative_count / total_count,
                "predicted_exact_positive_fraction": predicted_exact_positive_count / total_count,
                "predicted_exact_negative_fraction": predicted_exact_negative_count / total_count,
                "false_positive_count": false_positive_count,
                "false_positive_fraction_all_samples": false_positive_count / total_count,
                "false_positive_rate_given_truth_inactive": (
                    None if inactive_count == 0 else false_positive_count / inactive_count
                ),
                "false_negative_count": false_negative_count,
                "false_negative_fraction_all_samples": false_negative_count / total_count,
                "false_negative_rate_given_truth_active": (
                    None if truth_active_count == 0 else false_negative_count / truth_active_count
                ),
                "first_meaningful_positive_predicted_step": first_predicted_active,
                "first_meaningful_positive_truth_step": first_truth_active,
                "maximum_predicted_R": maximum_predicted_r,
                "minimum_predicted_R": minimum_predicted_r,
                "activity_numerical_tolerance": NUMERICAL_RATE_TOLERANCE,
                "sample_count": total_count,
            },
        })
    elif representation == "C":
        total_count = int(data["features"].shape[0] * 65536)
        inactive_count = total_count - truth_active_count
        result.update({
            "source_components": {
                name: component_metrics[index].finish(SOURCE_SCALES[index])
                for index, name in enumerate(SOURCE_ORDER)
            },
            "source_vector": {
                "normalized_vector_RMS_error": float(
                    np.sqrt(operator_numerator / total_weight)
                ),
                "normalized_vector_relative_RMS_error": float(
                    np.sqrt(operator_numerator / operator_denominator)
                ),
                "normalized_source_vector_cosine": normalized_dot
                / max(
                    np.sqrt(normalized_prediction2 * normalized_target2),
                    np.finfo(np.float64).tiny,
                ),
            },
            "effective_A": effective_a_metric.finish(SIGMA_A),
            "effective_R": effective_r_metric.finish(SIGMA_R),
            "source_structure": {
                "normalized_off_manifold_RMS": float(
                    np.sqrt(off_manifold_sum / total_weight)
                ),
                "water_source_defect_RMS": float(
                    np.sqrt(water_defect2 / total_weight)
                ),
                "S_minus_beta2_Qv_defect_RMS": float(
                    np.sqrt(thermo_defect2 / total_weight)
                ),
                "negative_Qr_source_fraction": negative_qr_count / source_sample_count,
                "positive_Qr_source_fraction": positive_qr_count / source_sample_count,
            },
            "effective_R_activation": {
                "truth_active_sample_count": truth_active_count,
                "truth_active_sample_fraction": truth_active_count / total_count,
                "predicted_meaningful_positive_fraction": predicted_active_count / total_count,
                "predicted_meaningful_negative_fraction": predicted_negative_count / total_count,
                "false_positive_count": false_positive_count,
                "false_positive_rate_given_truth_inactive": (
                    None if inactive_count == 0 else false_positive_count / inactive_count
                ),
                "false_negative_count": false_negative_count,
                "false_negative_rate_given_truth_active": (
                    None if truth_active_count == 0 else false_negative_count / truth_active_count
                ),
                "first_meaningful_positive_predicted_step": first_predicted_active,
                "first_meaningful_positive_truth_step": first_truth_active,
                "maximum_predicted_effective_R": maximum_predicted_r,
                "minimum_predicted_effective_R": minimum_predicted_r,
                "activity_numerical_tolerance": NUMERICAL_RATE_TOLERANCE,
                "sample_count": total_count,
            },
        })
    return result


def flatten_metric_rows(entry, evaluation_state, support, data, metrics):
    base = {
        "run_id": entry["run_id"],
        "physical_case": "Test 2B",
        "representation": entry["representation"],
        "model_label": entry["model_label"],
        "trained_objective": entry["objective"],
        "checkpoint_iteration": entry["iteration"],
        "checkpoint_parameter_pytree_sha256": entry["parameter_sha256"],
        "checkpoint_npz_sha256": entry["npz_sha256"],
        "checkpoint_path": entry["path"],
        "evaluation_state": evaluation_state,
        "support": (
            "TRAINING TRUTH SUPPORT" if support == "training"
            else "HELD-OUT TRUTH SUPPORT"
        ),
        "state_first": data["state_first"],
        "state_last": data["state_last"],
        "support_use": "POST-HOC CHECKPOINT EVALUATION",
    }
    rows = []
    for quantity, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            if isinstance(value, dict):
                for nested_metric, nested_value in value.items():
                    if not isinstance(nested_value, (dict, list)):
                        rows.append({
                            **base,
                            "quantity": f"{quantity}.{metric}",
                            "metric": nested_metric,
                            "value": "" if nested_value is None else nested_value,
                        })
            elif not isinstance(value, list):
                rows.append({
                    **base,
                    "quantity": quantity,
                    "metric": metric,
                    "value": "" if value is None else value,
                })
    return rows


def compare_metric(parity, label, actual, expected, keys):
    for key in keys:
        if key not in actual or key not in expected:
            continue
        if actual[key] is None or expected[key] is None:
            if actual[key] != expected[key]:
                parity.append({"label": f"{label}.{key}", "passed": False})
            continue
        difference = abs(float(actual[key]) - float(expected[key]))
        tolerance = max(2.0e-15, 2.0e-11 * abs(float(expected[key])))
        parity.append({
            "label": f"{label}.{key}",
            "actual": float(actual[key]),
            "accepted": float(expected[key]),
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": bool(difference <= tolerance),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    if output != OUTPUT.resolve():
        raise ValueError(f"output directory must be {OUTPUT}")
    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        "csv": output / "checkpoint_direct_histories.csv",
        "json": output / "checkpoint_direct_histories.json",
        "operator_csv": output / "checkpoint_training_objectives_operator.csv",
        "j_m1y_csv": output / "j_m1y_diagnostics.csv",
        "j_m1y_json": output / "j_m1y_diagnostics.json",
        "checkpoints": output / "checkpoint_hash_manifest.json",
        "validation": output / "direct_history_validation.json",
    }
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("refusing to overwrite direct-history products")

    sys.path.insert(0, str(M1Y))
    from dimswe.test2b_rain_learning import build_model, load_parameters  # noqa: PLC0415

    started = perf_counter()
    with (AUDIT / "ML_RUN_INVENTORY.csv").open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    rows = [
        row for row in inventory
        if row["physical_case"] == "Test 2B"
        and row["representation"] in SOURCE_ORDER[:0] + ("A", "B", "C")
        and row["run_id"].split("-", 2)[2]
        in {"m1x", "m1y", "m2x-independent", "m2x-warm", "h1", "h2", "h5"}
    ]
    if len(rows) != 21:
        raise RuntimeError(f"expected 21 main Test-2B runs, found {len(rows)}")

    checkpoint_entries = []
    parameter_cache = {}
    run_records = {}
    for row in rows:
        run_id = row["run_id"]
        representation = row["representation"]
        directory = Path(row["checkpoint_path"]).parent
        fit_result = read_json(Path(row["fit_result_path"]))
        if fit_result.get("status") != "complete":
            raise RuntimeError(f"incomplete accepted run {run_id}")
        final_parameters, final_sidecar = load_parameters(
            Path(row["checkpoint_path"]), representation
        )
        final_sha = final_sidecar["parameter_pytree_sha256"]
        if final_sha != fit_result["final_parameter_pytree_sha256"]:
            raise RuntimeError(f"final parameter provenance changed for {run_id}")
        checkpoints = []
        for path in sorted(directory.glob("checkpoint_*_parameters.npz")):
            match = CHECKPOINT_RE.search(path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            parameters, sidecar = load_parameters(path, representation)
            parameter_sha = sidecar["parameter_pytree_sha256"]
            cache_key = (representation, parameter_sha)
            parameter_cache.setdefault(cache_key, parameters)
            entry = {
                "run_id": run_id,
                "representation": representation,
                "model_label": model_label(run_id),
                "objective": row["objective"],
                "iteration": iteration,
                "path": str(path),
                "npz_sha256": file_sha256(path),
                "sidecar_path": str(path.with_suffix(".json")),
                "sidecar_sha256": file_sha256(path.with_suffix(".json")),
                "parameter_sha256": parameter_sha,
                "stage": sidecar["metadata"]["stage"],
            }
            checkpoints.append(entry)
            checkpoint_entries.append(entry)
        expected_iterations = json.loads(row["intermediate_checkpoint_iterations"])
        actual_iterations = [item["iteration"] for item in checkpoints]
        if actual_iterations != expected_iterations:
            raise RuntimeError(
                f"checkpoint cadence changed for {run_id}: {actual_iterations}"
            )
        final_iteration = int(row["accepted_iterations_this_run"])
        final_entry = next(
            (item for item in checkpoints if item["iteration"] == final_iteration), None
        )
        if final_entry is None or final_entry["parameter_sha256"] != final_sha:
            raise RuntimeError(f"terminal checkpoint mismatch for {run_id}")
        run_records[run_id] = {
            "inventory": row,
            "checkpoints": checkpoints,
            "final_entry": final_entry,
            "final_parameters_npz_sha256": file_sha256(Path(row["checkpoint_path"])),
            "final_parameter_pytree_sha256": final_sha,
        }

    models = {representation: jax.jit(build_model(representation)) for representation in "ABC"}
    evaluations = {}
    direct_rows = []
    operator_rows = []
    support_plan = (
        ("X", "training", "all"),
        ("X", "heldout", "all"),
        ("Y", "training", "m1-plus-finals"),
        ("Y", "heldout", "m1-only"),
    )
    for evaluation_state, support, selection in support_plan:
        print(f"loading {evaluation_state} {support}", flush=True)
        data = load_support(evaluation_state, support)
        for representation in "ABC":
            tasks = []
            for run_id, run in run_records.items():
                if run["inventory"]["representation"] != representation:
                    continue
                label = run["final_entry"]["model_label"]
                if selection == "all":
                    tasks.extend(run["checkpoints"])
                elif selection == "m1-only":
                    if label in ("M1-X", "M1-Y"):
                        tasks.extend(run["checkpoints"])
                elif label in ("M1-X", "M1-Y"):
                    tasks.extend(run["checkpoints"])
                else:
                    tasks.append(run["final_entry"])
            unique = {}
            for entry in tasks:
                unique.setdefault(entry["parameter_sha256"], entry)
            computed = {}
            for number, (parameter_sha, representative) in enumerate(unique.items(), 1):
                print(
                    f"{evaluation_state}-{support} {representation} "
                    f"{number}/{len(unique)} {representative['model_label']} "
                    f"i={representative['iteration']}",
                    flush=True,
                )
                computed[parameter_sha] = evaluate_checkpoint(
                    representation,
                    parameter_cache[(representation, parameter_sha)],
                    models[representation],
                    data,
                )
            for entry in tasks:
                metric = computed[entry["parameter_sha256"]]
                key = (
                    entry["run_id"], entry["iteration"], evaluation_state, support
                )
                evaluations[key] = metric
                direct_rows.extend(
                    flatten_metric_rows(
                        entry, evaluation_state, support, data, metric
                    )
                )
                if support == "training":
                    fitted = (
                        entry["model_label"] == "M1-X" and evaluation_state == "X"
                    ) or (
                        entry["model_label"] == "M1-Y" and evaluation_state == "Y"
                    )
                    if fitted:
                        operator_rows.append({
                            "run_id": entry["run_id"],
                            "physical_case": "Test 2B",
                            "representation": representation,
                            "model_label": entry["model_label"],
                            "trained_objective": entry["objective"],
                            "checkpoint_iteration": entry["iteration"],
                            "objective": (
                                "J_M1_X" if evaluation_state == "X" else "J_M1_Y"
                            ),
                            "evaluation_state": evaluation_state,
                            "support": "TRAINING TRUTH SUPPORT",
                            "value": metric["operator"]["normalized_objective"],
                            "history_kind": "POST-HOC FIXED-ARRAY CHECKPOINT EVALUATION",
                            "fitted_objective": True,
                            "checkpoint_parameter_pytree_sha256": entry["parameter_sha256"],
                            "checkpoint_path": entry["path"],
                        })
        del data

    j_m1y_rows = []
    for run_id, run in run_records.items():
        entry = run["final_entry"]
        metric = evaluations[(run_id, entry["iteration"], "Y", "training")]
        j_m1y_rows.append({
            "run_id": run_id,
            "physical_case": "Test 2B",
            "representation": entry["representation"],
            "model_label": entry["model_label"],
            "trained_objective": entry["objective"],
            "J_M1_Y": metric["operator"]["normalized_objective"],
            "fitted": entry["model_label"] == "M1-Y",
            "evaluation_kind": (
                "FITTED OBJECTIVE" if entry["model_label"] == "M1-Y"
                else "POST-HOC DIAGNOSTIC OBJECTIVE"
            ),
            "evaluation_state": "Y_n*=P(X_n*)",
            "state_indices": "0..80",
            "checkpoint_parameter_pytree_sha256": entry["parameter_sha256"],
            "checkpoint_path": entry["path"],
        })

    metric_keys = (
        "physical_RMS_error", "relative_RMS_error", "normalized_RMS_error",
        "maximum_absolute_error", "signed_mass_weighted_bias", "correlation",
    )
    parity = []
    for representation in "ABC":
        historical_path = AUTH / (
            "external-results/test2b-rain-active-learning/production/"
            f"representation-{representation}/representation_"
            f"{representation.lower()}_final_comparison.json"
        )
        frozen = read_json(historical_path)
        for run_id, run in run_records.items():
            entry = run["final_entry"]
            if entry["representation"] != representation or entry["model_label"] == "M1-Y":
                continue
            label = accepted_label(entry["model_label"])
            for support, accepted_support in (
                ("training", "TRAINING_OVERALL"),
                ("heldout", "HELDOUT_MATURE_RAIN"),
            ):
                actual = evaluations[(run_id, entry["iteration"], "X", support)]
                prefix = f"{run_id}.X.{support}"
                if representation == "A":
                    compare_metric(parity, prefix + ".A", actual["A"], frozen["direct_A"][label][accepted_support], metric_keys)
                elif representation == "B":
                    compare_metric(parity, prefix + ".A", actual["A"], frozen["direct_A"][label][accepted_support], metric_keys)
                    expected_r = frozen["direct_R"][label][accepted_support]
                    compare_metric(parity, prefix + ".R_all", actual["R_all"], expected_r["all_samples"], metric_keys)
                    if expected_r["truth_active_samples"] is not None:
                        compare_metric(parity, prefix + ".R_active", actual["R_truth_active"], expected_r["truth_active_samples"], metric_keys)
                else:
                    expected = frozen["direct_source_diagnostics"][label][accepted_support]["component_errors"]
                    for name in SOURCE_ORDER:
                        compare_metric(parity, prefix + f".{name}", actual["source_components"][name], expected[name], metric_keys)

        matched_path = M1Y / (
            "external-results/m1y-test2b-20260828/evaluation/"
            f"representation_{representation}_matched.json"
        )
        matched = read_json(matched_path)
        for label in ("M1-X", "M1-Y"):
            run_id = f"t2b-{representation.lower()}-{'m1x' if label == 'M1-X' else 'm1y'}"
            entry = run_records[run_id]["final_entry"]
            for state in ("X", "Y"):
                for support, accepted_support in (
                    ("training", "TRAINING_OVERALL"),
                    ("heldout", "HELDOUT_MATURE_RAIN"),
                ):
                    actual = evaluations[(run_id, entry["iteration"], state, support)]
                    expected = matched["direct_cross_evaluation"][state][label]
                    prefix = f"matched.{representation}.{label}.{state}.{support}"
                    if representation == "A":
                        compare_metric(parity, prefix + ".A", actual["A"], expected["A"][accepted_support], metric_keys)
                    elif representation == "B":
                        compare_metric(parity, prefix + ".A", actual["A"], expected["A"][accepted_support], metric_keys)
                        compare_metric(parity, prefix + ".R_all", actual["R_all"], expected["R"][accepted_support]["all_samples"], metric_keys)
                        expected_active = expected["R"][accepted_support]["truth_active_samples"]
                        if expected_active is not None:
                            compare_metric(parity, prefix + ".R_active", actual["R_truth_active"], expected_active, metric_keys)
                    else:
                        expected_components = expected["source"][accepted_support]["component_errors"]
                        for name in SOURCE_ORDER:
                            compare_metric(parity, prefix + f".{name}", actual["source_components"][name], expected_components[name], metric_keys)

    objective_parity = []
    for row in rows:
        if row["objective"] not in ("M1-X", "M1-Y"):
            continue
        run = run_records[row["run_id"]]
        entry = run["final_entry"]
        state = "X" if row["objective"] == "M1-X" else "Y"
        actual = evaluations[(row["run_id"], entry["iteration"], state, "training")]["operator"]["normalized_objective"]
        expected = float(row["final_training_objective"])
        difference = abs(actual - expected)
        tolerance = max(2.0e-15, 2.0e-11 * abs(expected))
        objective_parity.append({
            "run_id": row["run_id"],
            "actual": actual,
            "accepted": expected,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": bool(difference <= tolerance),
        })
    if not all(item.get("passed", False) for item in parity + objective_parity):
        failed = [item for item in parity + objective_parity if not item.get("passed", False)]
        raise RuntimeError(f"accepted final-metric parity failed: {failed[:10]}")

    direct_fields = [
        "run_id", "physical_case", "representation", "model_label",
        "trained_objective", "checkpoint_iteration",
        "checkpoint_parameter_pytree_sha256", "checkpoint_npz_sha256",
        "checkpoint_path", "evaluation_state", "support", "state_first",
        "state_last", "support_use", "quantity", "metric", "value",
    ]
    operator_fields = [
        "run_id", "physical_case", "representation", "model_label",
        "trained_objective", "checkpoint_iteration", "objective",
        "evaluation_state", "support", "value", "history_kind",
        "fitted_objective", "checkpoint_parameter_pytree_sha256",
        "checkpoint_path",
    ]
    j_fields = list(j_m1y_rows[0])
    write_csv(destinations["csv"], direct_rows, direct_fields)
    write_json(destinations["json"], {
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "history_kind": "POST-HOC CHECKPOINT EVALUATION",
        "not_validation_during_training": True,
        "normalization_provenance_sha256": NORMALIZATION_SHA,
        "feature_order": list(FEATURE_ORDER),
        "state_supports": {
            "training": [0, 80],
            "heldout": [81, 160],
        },
        "records": direct_rows,
    })
    write_csv(destinations["operator_csv"], operator_rows, operator_fields)
    write_csv(destinations["j_m1y_csv"], j_m1y_rows, j_fields)
    write_json(destinations["j_m1y_json"], {
        "status": "complete",
        "definition": "J_M1-Y fixed-array normalized operator objective on Y states 0..80",
        "records": j_m1y_rows,
    })
    write_json(destinations["checkpoints"], {
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "run_count": len(run_records),
        "checkpoint_reference_count": len(checkpoint_entries),
        "unique_parameter_count": len(parameter_cache),
        "runs": run_records,
    })
    write_json(destinations["validation"], {
        "status": "passed",
        "accepted_direct_metric_comparisons": parity,
        "accepted_direct_metric_comparison_count": len(parity),
        "maximum_absolute_direct_metric_difference": max(
            item.get("absolute_difference", 0.0) for item in parity
        ),
        "accepted_fitted_objective_comparisons": objective_parity,
        "normalization": {
            "feature_order": list(FEATURE_ORDER),
            "input_offset": INPUT_OFFSET.tolist(),
            "input_scale": INPUT_SCALE.tolist(),
            "sigma_A": SIGMA_A,
            "sigma_R": SIGMA_R,
            "source_scales": SOURCE_SCALES.tolist(),
            "provenance_sha256": NORMALIZATION_SHA,
        },
        "carrier_weights": "frozen production carrier-mass weights",
        "optimizer_instantiated": False,
        "truth_generated": False,
        "prefix_integrated": False,
        "rollout_performed": False,
        "wall_seconds": float(perf_counter() - started),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    })
    print(json.dumps({
        "status": "complete",
        "runs": len(run_records),
        "checkpoint_references": len(checkpoint_entries),
        "direct_rows": len(direct_rows),
        "operator_rows": len(operator_rows),
        "J_M1_Y_rows": len(j_m1y_rows),
        "parity_comparisons": len(parity),
        "wall_seconds": float(perf_counter() - started),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
