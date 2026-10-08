"""Build compact machine-readable results for the Test-2B M1-Y campaign."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys


CAMPAIGN_ID = "m1y_test2b_20260828"
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json_record(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_record(path, record):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


EVALUATIONS = {
    representation: (
        f"external-results/m1y-test2b-20260828/evaluation/"
        f"representation_{representation}_matched.json"
    )
    for representation in "ABC"
}


def _relative_change(new, old):
    return float(new / old - 1.0)


def _improvement(new, old):
    return float(1.0 - new / old)


def _load_evaluations():
    root = repository_root()
    records = {}
    for representation, relative in EVALUATIONS.items():
        path = root / relative
        record = read_json_record(path)
        if (
            record.get("status") != "complete"
            or record.get("campaign_id") != CAMPAIGN_ID
            or record.get("representation") != representation
            or not record["historical_standard_M1_X"]["recomputed_direct_metric_parity"]["passed"]
        ):
            raise ValueError(f"invalid matched evaluation {path}")
        records[representation] = (path, record)
    return records


def _nominal_direct(representation, label, record):
    state = "X" if label == "M1-X" else "Y"
    values = record["direct_cross_evaluation"][state][label]
    if representation == "A":
        a = values["A"]
        return {
            "A_training_relative_RMS": a["TRAINING_OVERALL"]["relative_RMS_error"],
            "A_training_physical_RMS": a["TRAINING_OVERALL"]["physical_RMS_error"],
            "A_heldout_relative_RMS": a["HELDOUT_MATURE_RAIN"]["relative_RMS_error"],
            "A_heldout_physical_RMS": a["HELDOUT_MATURE_RAIN"]["physical_RMS_error"],
            "A_training_maximum_absolute": a["TRAINING_OVERALL"]["maximum_absolute_error"],
            "A_heldout_maximum_absolute": a["HELDOUT_MATURE_RAIN"]["maximum_absolute_error"],
        }
    if representation == "B":
        a = values["A"]
        r = values["R"]
        return {
            "A_training_relative_RMS": a["TRAINING_OVERALL"]["relative_RMS_error"],
            "A_training_physical_RMS": a["TRAINING_OVERALL"]["physical_RMS_error"],
            "A_heldout_relative_RMS": a["HELDOUT_MATURE_RAIN"]["relative_RMS_error"],
            "A_heldout_physical_RMS": a["HELDOUT_MATURE_RAIN"]["physical_RMS_error"],
            "R_training_relative_RMS_all": r["TRAINING_OVERALL"]["all_samples"]["relative_RMS_error"],
            "R_training_relative_RMS_active": r["TRAINING_OVERALL"]["truth_active_samples"]["relative_RMS_error"],
            "R_training_physical_RMS_all": r["TRAINING_OVERALL"]["all_samples"]["physical_RMS_error"],
            "R_heldout_relative_RMS_all": r["HELDOUT_MATURE_RAIN"]["all_samples"]["relative_RMS_error"],
            "R_heldout_relative_RMS_active": r["HELDOUT_MATURE_RAIN"]["truth_active_samples"]["relative_RMS_error"],
            "R_heldout_physical_RMS_all": r["HELDOUT_MATURE_RAIN"]["all_samples"]["physical_RMS_error"],
            "R_training_false_positive_rate": r["TRAINING_OVERALL"]["false_positive_rate_given_truth_inactive"],
            "R_heldout_false_positive_rate": r["HELDOUT_MATURE_RAIN"]["false_positive_rate_given_truth_inactive"],
            "R_training_false_negative_rate": r["TRAINING_OVERALL"]["false_negative_rate_given_truth_active"],
            "R_heldout_false_negative_rate": r["HELDOUT_MATURE_RAIN"]["false_negative_rate_given_truth_active"],
        }
    source = values["source"]
    result = {}
    for regime, prefix in (
        ("TRAINING_OVERALL", "source_training"),
        ("HELDOUT_MATURE_RAIN", "source_heldout"),
    ):
        local = source[regime]
        for component, metrics in local["component_errors"].items():
            result[f"{prefix}_{component}_relative_RMS"] = metrics["relative_RMS_error"]
            result[f"{prefix}_{component}_physical_RMS"] = metrics["physical_RMS_error"]
        result[f"{prefix}_off_manifold_fraction"] = local[
            "physical_two_rate_projection"
        ]["off_manifold_fraction_of_source_magnitude"]
        result[f"{prefix}_projected_R_relative_RMS"] = local[
            "physical_two_rate_projection"
        ]["projected_R"]["relative_RMS_error"]
    return result


def _hybrid(representation, label, record):
    auto = (
        record["historical_standard_M1_X"]["autonomous"]
        if label == "M1-X"
        else record["standard_M1_Y"]["autonomous"]
    )
    mixed = auto["mixed_state_error"]["ALL"]
    result = {
        "hybrid_mixed_accumulated": mixed["accumulated"],
        "hybrid_mixed_final": mixed["final"],
        "hybrid_mixed_maximum": mixed["maximum"],
        "hybrid_mixed_maximum_step": mixed["maximum_step"],
    }
    if representation == "A":
        held = auto["A_error_on_model_postprefix_states"]["HELDOUT_MATURE_RAIN"]
        rain = auto["rain"]
        result.update({
            "hybrid_model_state_A_heldout_relative_RMS": held["relative_RMS_error"],
            "hybrid_final_Qr_mass": rain["final_Qr_mass"],
            "hybrid_final_Qr_mass_error": rain["final_Qr_mass_error"],
            "hybrid_maximum_R": rain["maximum_R"],
            "hybrid_maximum_R_error": rain["maximum_R_error"],
        })
    elif representation == "B":
        held_a = auto["A_error_on_model_postprefix_states"]["HELDOUT_MATURE_RAIN"]
        held_r = auto["R_error_on_model_postprefix_states"]["HELDOUT_MATURE_RAIN"]
        rain = auto["rain"]
        result.update({
            "hybrid_model_state_A_heldout_relative_RMS": held_a["relative_RMS_error"],
            "hybrid_model_state_R_heldout_relative_RMS_all": held_r["all_samples"]["relative_RMS_error"],
            "hybrid_model_state_R_heldout_relative_RMS_active": held_r["truth_active_samples"]["relative_RMS_error"],
            "hybrid_final_Qr_mass": rain["final_Qr_mass"],
            "hybrid_final_Qr_mass_error": rain["final_Qr_mass_error"],
            "hybrid_maximum_R": rain["maximum_R"],
            "hybrid_maximum_R_error": rain["maximum_R_error"],
            "hybrid_pre_onset_R_false_positive_fraction": rain["pre_truth_onset_false_positive_R_fraction"],
            "hybrid_R_false_negative_rate": rain["false_negative_rate_given_truth_active"],
        })
    else:
        held = auto["source_diagnostics_on_model_postprefix_states"]["HELDOUT_MATURE_RAIN"]
        rain = auto["rain_source_and_partition"]
        for component, metrics in held["component_errors"].items():
            result[f"hybrid_model_state_{component}_heldout_relative_RMS"] = metrics["relative_RMS_error"]
        result["hybrid_model_state_off_manifold_fraction"] = held[
            "physical_two_rate_projection"
        ]["off_manifold_fraction_of_source_magnitude"]
        result["hybrid_model_state_projected_R_heldout_relative_RMS"] = held[
            "physical_two_rate_projection"
        ]["projected_R"]["relative_RMS_error"]
        result.update({
            "hybrid_final_Qr_mass": rain["final_Qr_mass"],
            "hybrid_final_Qr_mass_error": rain["final_Qr_mass_error"],
        })
    return result


def _row(representation, label, record):
    artifact = record["artifacts"][label]
    architecture = artifact["architecture"]
    objective = record["objective_matrix"][label]
    result = {
        "representation": representation,
        "objective": label,
        "evaluation_state": "X_n*" if label == "M1-X" else "Y_n*=P(X_n*)",
        "training_support": "truth states 0..80 inclusive (81 states; 5,308,416 samples)",
        "architecture": "->".join(str(value) for value in architecture["layers"]),
        "parameter_count": architecture["parameter_count"],
        "activation": architecture["activation"],
        "dtype": architecture["dtype"],
        "seed": architecture["seed"],
        "optimizer_budget": "PyROL line-search L-BFGS m20; max 10000; gtol 1e-8; stol 1e-12",
        "termination_status": artifact["termination_reason"],
        "accepted_iterations": artifact["accepted_iterations"],
        "objective_evaluations": artifact["objective_evaluations"],
        "gradient_evaluations": artifact["gradient_evaluations"],
        "final_normalized_training_loss": artifact["final_objective"],
        "J_M1_X": objective["J_M1_X"],
        "J_M1_Y": objective["J_M1_Y"],
        "J_M2_X": objective["J_M2_X"],
        "J_H1": objective["J_H1"],
        "J_H2": objective["J_H2"],
        "J_H5": objective["J_H5"],
        "checkpoint": artifact["checkpoint"],
        "checkpoint_pytree_sha256": artifact["final_parameter_pytree_sha256"],
        "checkpoint_npz_sha256": artifact["checkpoint_npz_sha256"],
    }
    result.update(_nominal_direct(representation, label, record))
    result.update(_hybrid(representation, label, record))
    return result


def build_results(output_directory):
    root = repository_root()
    configuration = read_json_record(
        root / "dimswe/configs/test2b_m1y_20260828.json"
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "M1Y_RESULTS.json"
    csv_path = output / "M1Y_RESULTS.csv"
    manifest_path = output / "manifest.json"
    for path in (json_path, csv_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    evaluations = _load_evaluations()
    rows = [
        _row(representation, label, record)
        for representation, (_, record) in evaluations.items()
        for label in ("M1-X", "M1-Y")
    ]
    comparisons = {}
    for representation, (_, record) in evaluations.items():
        x = next(row for row in rows if row["representation"] == representation and row["objective"] == "M1-X")
        y = next(row for row in rows if row["representation"] == representation and row["objective"] == "M1-Y")
        comparison = {
            "normalized_training_loss_relative_change": _relative_change(
                y["final_normalized_training_loss"], x["final_normalized_training_loss"]
            ),
            "J_H1_improvement_fraction": _improvement(y["J_H1"], x["J_H1"]),
            "J_H2_improvement_fraction": _improvement(y["J_H2"], x["J_H2"]),
            "J_H5_improvement_fraction": _improvement(y["J_H5"], x["J_H5"]),
            "hybrid_accumulated_improvement_fraction": _improvement(
                y["hybrid_mixed_accumulated"], x["hybrid_mixed_accumulated"]
            ),
            "hybrid_final_improvement_fraction": _improvement(
                y["hybrid_mixed_final"], x["hybrid_mixed_final"]
            ),
            "hybrid_maximum_improvement_fraction": _improvement(
                y["hybrid_mixed_maximum"], x["hybrid_mixed_maximum"]
            ),
        }
        if representation == "A":
            comparison.update({
                "nominal_training_A_improvement_fraction": _improvement(
                    y["A_training_relative_RMS"], x["A_training_relative_RMS"]
                ),
                "nominal_heldout_A_improvement_fraction": _improvement(
                    y["A_heldout_relative_RMS"], x["A_heldout_relative_RMS"]
                ),
            })
        elif representation == "B":
            comparison.update({
                "nominal_training_A_improvement_fraction": _improvement(
                    y["A_training_relative_RMS"], x["A_training_relative_RMS"]
                ),
                "nominal_training_R_active_improvement_fraction": _improvement(
                    y["R_training_relative_RMS_active"], x["R_training_relative_RMS_active"]
                ),
                "nominal_heldout_R_active_improvement_fraction": _improvement(
                    y["R_heldout_relative_RMS_active"], x["R_heldout_relative_RMS_active"]
                ),
                "hybrid_model_state_R_heldout_improvement_fraction": _improvement(
                    y["hybrid_model_state_R_heldout_relative_RMS_all"],
                    x["hybrid_model_state_R_heldout_relative_RMS_all"],
                ),
                "hybrid_Qr_mass_error_magnitude_improvement_fraction": _improvement(
                    abs(y["hybrid_final_Qr_mass_error"]), abs(x["hybrid_final_Qr_mass_error"])
                ),
            })
        else:
            comparison.update({
                f"nominal_training_{component}_improvement_fraction": _improvement(
                    y[f"source_training_{component}_relative_RMS"],
                    x[f"source_training_{component}_relative_RMS"],
                )
                for component in ("S", "Qv", "Qc", "Qr")
            })
            comparison["hybrid_Qr_mass_error_magnitude_improvement_fraction"] = _improvement(
                abs(y["hybrid_final_Qr_mass_error"]), abs(x["hybrid_final_Qr_mass_error"])
            )
        comparisons[representation] = comparison
    result = {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "source_head": EXPECTED_HEAD,
        "comparison": "historical M1-X versus new M1-Y",
        "rows": rows,
        "relative_comparisons": comparisons,
        "scientific_interpretation": {
            "direct_fit": (
                "M1-Y improves nominal training fit for A and the dominant C "
                "source components, but does not uniformly improve held-out direct "
                "fit and worsens learned-R/Qr direct errors for B/C."
            ),
            "hybrid": (
                "M1-Y strongly improves all H1/H2/H5 objectives and the standard "
                "hybrid accumulated state error, with an exceptionally large B "
                "improvement; A final/maximum error is mixed and C rain partition "
                "is worse despite better state error."
            ),
            "inference_limit": (
                "The controlled state-location change supports timestep-location "
                "consistency as important, especially for B, but optimization and "
                "finite-support extrapolation prevent a universal causal claim."
            ),
        },
    }
    write_json_record(json_path, result)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = []
    candidate_paths = [
        root / "dimswe/configs/test2b_m1y_20260828.json",
        root / "dimswe/test2b_m1y_campaign.py",
        root / "dimswe/test2b_m1y_evaluation.py",
        root / "dimswe/test2b_m1y_report.py",
        root / "tests/test_test2b_m1y_campaign.py",
        root / "external-results/m1y-test2b-20260828/preparation/immutable_inputs.json",
        root / "external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz",
        root / "external-results/m1y-test2b-20260828/preparation/m1y_learning_data.json",
        root / "external-results/m1y-test2b-20260828/preparation/pretraining_validation.json",
        root / "external-results/m1y-test2b-20260828/preparation/objective_certification.json",
        root / "external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz",
        root / "external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.json",
        *[path for path, _ in evaluations.values()],
        json_path,
        csv_path,
    ]
    for representation in "ABC":
        directory = root / f"external-results/m1y-test2b-20260828/production/representation-{representation}/m1y-seed0-m20-10k"
        candidate_paths.extend((
            directory / "fit_result.json",
            directory / "final_parameters.npz",
            directory / "final_parameters.json",
        ))
    for path in candidate_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        })
    manifest = {
        "status": "results_complete_pending_final_authoritative_recheck",
        "campaign_id": CAMPAIGN_ID,
        "source_head": EXPECTED_HEAD,
        "workspace": str(root),
        "authoritative_repository": configuration["reference"][
            "authoritative_repository"
        ],
        "artifacts": artifacts,
        "runs": {
            representation: {
                "feature_map": ["h", "S", "Qv", "Qc", "B"],
                "evaluation_state": "Y_n*=P(X_n*)",
                "config": str((root / "dimswe/configs/test2b_m1y_20260828.json").resolve()),
                "seed": 0,
                "checkpoint": next(row["checkpoint"] for row in rows if row["representation"] == representation and row["objective"] == "M1-Y"),
                "checkpoint_pytree_sha256": next(row["checkpoint_pytree_sha256"] for row in rows if row["representation"] == representation and row["objective"] == "M1-Y"),
                "metrics": str(evaluations[representation][0].resolve()),
                "truth_artifact": str((root / "external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000").resolve()),
                "training_support": [0, 80],
                "heldout_support": [81, 160],
            }
            for representation in "ABC"
        },
        "report_command": [sys.executable, "-m", "dimswe.test2b_m1y_report", *sys.argv[1:]],
    }
    write_json_record(manifest_path, manifest)
    return result, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    build_results(args.output_directory)


if __name__ == "__main__":
    main()
