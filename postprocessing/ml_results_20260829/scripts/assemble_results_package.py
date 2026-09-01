#!/usr/bin/env python3
"""Assemble audited ML-results data and publication table drafts.

Parsing and array algebra only.  This script does not import DIMSWE, JAX,
Firedrake, an optimizer, or a timestepper.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from portable_paths import (
    AUDIT_ROOT as AUDIT,
    GROUND_TRUTH_PACKAGE as GT,
    M1Y_REPOSITORY as M1Y,
    PACKAGE_ROOT,
    REFERENCE_REPOSITORY as AUTH,
)

ROOT = PACKAGE_ROOT
DATA = ROOT / "data"
TABLES = ROOT / "tables"
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"
MAIN_SUFFIXES = {
    "m1x", "m1y", "m2x-independent", "m2x-warm", "h1", "h2", "h5"
}
MODEL_ORDER = (
    "M1-X", "M1-Y", "M2-X-independent", "warm M2-X", "H1", "H2", "H5"
)
OBJECTIVE_COLUMNS = ("J_M1_X", "J_M1_Y", "J_M2_X", "J_H1", "J_H2", "J_H5")
SOURCE_ORDER = ("S", "Qv", "Qc", "Qr")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_label(run_id: str) -> str:
    suffix = run_id.split("-", 2)[2].removesuffix("-accepted")
    return {
        "m1x": "M1-X",
        "m1y": "M1-Y",
        "m2x-independent": "M2-X-independent",
        "m2x-warm": "warm M2-X",
        "h1": "H1",
        "h2": "H2",
        "h5": "H5",
    }[suffix]


def fitted_column(label: str) -> str:
    return {
        "M1-X": "J_M1_X",
        "M1-Y": "J_M1_Y",
        "M2-X-independent": "J_M2_X",
        "warm M2-X": "J_M2_X",
        "H1": "J_H1",
        "H2": "J_H2",
        "H5": "J_H5",
    }[label]


def fmt(value, digits=3):
    if value in (None, "", "NOT_EVALUATED", "NOT_DEFINED"):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "0"
    if abs(number) < 1.0e-2 or abs(number) >= 1.0e4:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def tex_table(headers, rows, alignment=None):
    alignment = alignment or ("l" * len(headers))
    lines = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(latex_escape(x) for x in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_escape(x) for x in row) + r" \\" for row in rows
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main_inventory():
    rows = read_csv(AUDIT / "ML_RUN_INVENTORY.csv")
    return [
        row for row in rows
        if (
            row["physical_case"] == "Test 2B"
            and row["representation"] in ("A", "B", "C")
            and row["run_id"].split("-", 2)[2] in MAIN_SUFFIXES
        ) or (
            row["physical_case"] == "Test 2A"
            and row["run_id"] in {
                "t2a-a-m1x-accepted", "t2a-a-m2x-independent-accepted",
                "t2a-a-m2x-warm", "t2a-a-h1", "t2a-a-h2", "t2a-a-h5",
                "t2a-c-m1x", "t2a-c-m2x-independent", "t2a-c-m2x-warm",
                "t2a-c-h1", "t2a-c-h2", "t2a-c-h5",
            }
        )
    ]


def complete_objective_matrix(inventory):
    audit_rows = read_csv(AUDIT / "FROZEN_MODEL_OBJECTIVE_MATRIX.csv")
    j_m1y = {
        row["run_id"]: row["J_M1_Y"]
        for row in read_csv(DATA / "j_m1y_diagnostics.csv")
    }
    main_ids = {row["run_id"] for row in inventory}
    completed = []
    for source in audit_rows:
        if source["run_id"] not in main_ids:
            continue
        row = dict(source)
        label = model_label(row["run_id"])
        if row["physical_case"] == "Test 2B":
            row["J_M1_Y"] = j_m1y[row["run_id"]]
        else:
            row["J_M1_Y"] = "NOT_DEFINED"
        row["model_label"] = label
        row["fitted_column"] = fitted_column(label)
        for column in OBJECTIVE_COLUMNS:
            row[f"{column}_role"] = (
                "FITTED" if column == row["fitted_column"] else
                "NOT_DEFINED" if row[column] in ("", "NOT_DEFINED", "NOT_EVALUATED") else
                "POST_HOC_DIAGNOSTIC"
            )
        completed.append(row)
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    completed.sort(key=lambda row: (
        row["physical_case"], row["representation"], order[row["model_label"]]
    ))
    destination = DATA / "completed_objective_matrix.csv"
    if destination.exists():
        existing = read_csv(destination)
        if len(existing) != len(completed):
            raise RuntimeError("existing completed objective matrix is incomplete")
        return existing
    write_csv(destination, completed)
    return completed


def add_history(rows, **record):
    base = {
        "physical_case": record.pop("physical_case"),
        "run_id": record.pop("run_id"),
        "representation": record.pop("representation"),
        "model_label": record.pop("model_label"),
        "trained_objective": record.pop("trained_objective"),
        "checkpoint_iteration": int(record.pop("checkpoint_iteration")),
        "objective": record.pop("objective"),
        "evaluation_state": record.pop("evaluation_state"),
        "support": record.pop("support"),
        "value": float(record.pop("value")),
        "history_kind": record.pop("history_kind"),
        "fitted_objective": True,
        "checkpoint_parameter_pytree_sha256": record.pop(
            "checkpoint_parameter_pytree_sha256", ""
        ),
        "checkpoint_path": record.pop("checkpoint_path", ""),
    }
    if record:
        raise ValueError(record)
    rows.append(base)


def assemble_histories(inventory):
    rows = read_csv(DATA / "checkpoint_training_objectives_operator.csv")
    rows.extend(read_csv(DATA / "checkpoint_training_objectives_fixed.csv"))
    by_id = {row["run_id"]: row for row in inventory}

    # Test-2B recursive objectives: recorded endpoints only; no reconstructed
    # recursive checkpoint history is introduced.
    existing_matrix = read_csv(DATA / "completed_objective_matrix.csv")
    matrix_by_id = {row["run_id"]: row for row in existing_matrix}
    for run_id, item in by_id.items():
        if item["physical_case"] != "Test 2B" or item["objective"] not in ("H2", "H5"):
            continue
        fit = read_json(Path(item["fit_result_path"]))
        predecessor_suffix = "h1" if item["objective"] == "H2" else "h2"
        predecessor = f"t2b-{item['representation'].lower()}-{predecessor_suffix}"
        initial_value = matrix_by_id[predecessor][f"J_{item['objective']}"]
        for iteration, value, history_kind in (
            (0, initial_value, "EXISTING DIAGNOSTIC INITIAL ENDPOINT"),
            (
                int(item["accepted_iterations_this_run"]),
                fit["final_objective"],
                "RECORDED TRAINING FINAL ENDPOINT",
            ),
        ):
            add_history(
                rows,
                physical_case="Test 2B", run_id=run_id,
                representation=item["representation"],
                model_label=model_label(run_id), trained_objective=item["objective"],
                checkpoint_iteration=iteration, objective=f"J_{item['objective']}",
                evaluation_state="recursive model state",
                support="TRAINING TRUTH WINDOWS",
                value=value, history_kind=history_kind,
            )

    # Test-2A Representation-A fair long fits (recorded sparse diagnostics).
    fair_specs = (
        (
            "t2a-a-m1x-accepted", "operator-seed0-m20-200k", "J_op", "J_M1_X",
            "M1-X", "X_n*",
        ),
        (
            "t2a-a-m2x-independent-accepted", "discrete-seed0-m20-200k", "J_disc", "J_M2_X",
            "M2-X-independent", "X_n*",
        ),
    )
    for run_id, directory_name, key, objective, label, state in fair_specs:
        directory = AUTH / "external-results/test2a/fair-longfit" / directory_name
        for path in sorted(directory.glob("checkpoint_iter_*.json")):
            record = read_json(path)
            add_history(
                rows, physical_case="Test 2A", run_id=run_id,
                representation="A", model_label=label,
                trained_objective=by_id[run_id]["objective"],
                checkpoint_iteration=record["accepted_iteration"],
                objective=objective, evaluation_state=state,
                support="TRAINING SUPPORT", value=record[key],
                history_kind="RECORDED SPARSE TRAINING CHECKPOINT",
                checkpoint_parameter_pytree_sha256=record["parameter_pytree_sha256"],
                checkpoint_path=record["parameter_file"],
            )

    warm = read_json(
        AUTH / "external-results/test2a/m1-to-m2-finetune/postprocess/checkpoint_metrics.json"
    )
    for record in warm["checkpoints"]:
        add_history(
            rows, physical_case="Test 2A", run_id="t2a-a-m2x-warm",
            representation="A", model_label="warm M2-X", trained_objective="M2-X",
            checkpoint_iteration=record["accepted_iteration"], objective="J_M2_X",
            evaluation_state="X_n*", support="TRAINING SUPPORT",
            value=record["J_disc"], history_kind="RECORDED SPARSE TRAINING CHECKPOINT",
            checkpoint_parameter_pytree_sha256=record["parameter_pytree_sha256"],
            checkpoint_path=record["parameter_file"],
        )

    for label, directory_name, run_id in (
        ("H1", "h1-from-m1-200k", "t2a-a-h1"),
        ("H2", "h2-from-h1", "t2a-a-h2"),
        ("H5", "h5-from-h2", "t2a-a-h5"),
    ):
        directory = AUTH / "external-results/test2a/horizon-curriculum-h1-h2-h5" / directory_name
        for path in sorted(directory.glob("checkpoint_iter_*.json")):
            record = read_json(path)
            add_history(
                rows, physical_case="Test 2A", run_id=run_id,
                representation="A", model_label=label, trained_objective=label,
                checkpoint_iteration=record["accepted_iteration"], objective=f"J_{label}",
                evaluation_state=("Y_n*=P(X_n*)" if label == "H1" else "recursive model state"),
                support="TRAINING TRUTH WINDOWS", value=record["J_active"],
                history_kind="RECORDED SPARSE TRAINING CHECKPOINT",
                checkpoint_parameter_pytree_sha256=record["parameter_pytree_sha256"],
                checkpoint_path=record["parameter_file"],
            )

    # Test-2A Representation-C checkpoints were recorded in fit_progress.
    c_specs = {
        "t2a-c-m1x": ("m1-seed0-m20-200k", "J_M1_X", "X_n*"),
        "t2a-c-m2x-independent": ("m2x-seed0-m20-200k", "J_M2_X", "X_n*"),
        "t2a-c-m2x-warm": ("m1-to-m2x-m20-50k", "J_M2_X", "X_n*"),
        "t2a-c-h1": ("h1-from-m1", "J_H1", "Y_n*=P(X_n*)"),
        "t2a-c-h2": ("h2-from-h1", "J_H2", "recursive model state"),
        "t2a-c-h5": ("h5-from-h2", "J_H5", "recursive model state"),
    }
    root = AUTH / "external-results/test2a/problem-b/production"
    for run_id, (directory_name, objective, state) in c_specs.items():
        progress = read_json(root / directory_name / "fit_progress.json")
        for record in progress["checkpoints"]:
            add_history(
                rows, physical_case="Test 2A", run_id=run_id,
                representation="C", model_label=model_label(run_id),
                trained_objective=by_id[run_id]["objective"],
                checkpoint_iteration=record["accepted_iteration"], objective=objective,
                evaluation_state=state,
                support=("TRAINING SUPPORT" if state == "X_n*" else "TRAINING TRUTH WINDOWS"),
                value=record["objective"], history_kind="RECORDED SPARSE TRAINING CHECKPOINT",
                checkpoint_parameter_pytree_sha256=record["parameter_pytree_sha256"],
                checkpoint_path=record["parameter_file"],
            )

    rows.sort(key=lambda row: (
        row["physical_case"], row["representation"],
        MODEL_ORDER.index(row["model_label"]), int(row["checkpoint_iteration"])
    ))
    fields = [
        "physical_case", "run_id", "representation", "model_label",
        "trained_objective", "checkpoint_iteration", "objective",
        "evaluation_state", "support", "value", "history_kind",
        "fitted_objective", "checkpoint_parameter_pytree_sha256", "checkpoint_path",
    ]
    write_csv(DATA / "checkpoint_training_objectives.csv", rows, fields)
    return rows


def assemble_final_direct(inventory):
    direct = read_csv(DATA / "checkpoint_direct_histories.csv")
    checkpoint_manifest = read_json(DATA / "checkpoint_hash_manifest.json")
    finals = {
        run_id: int(record["final_entry"]["iteration"])
        for run_id, record in checkpoint_manifest["runs"].items()
    }
    rows = []
    for row in direct:
        if int(row["checkpoint_iteration"]) != finals[row["run_id"]]:
            continue
        rows.append({
            **row,
            "metric_provenance": "POST-HOC FINAL CHECKPOINT EVALUATION",
            "nominal_state_for_model": (
                "true" if (
                    (row["model_label"] == "M1-X" and row["evaluation_state"] == "X")
                    or (row["model_label"] == "M1-Y" and row["evaluation_state"] == "Y")
                    or (row["model_label"] not in ("M1-X", "M1-Y") and row["evaluation_state"] == "X")
                ) else "false"
            ),
        })

    # Test-2A A training-support metrics already stored by accepted postprocessors.
    a_metrics = {}
    fair = read_json(
        AUTH / "external-results/test2a/fair-longfit/comparison/fair_longfit_comparison.json"
    )["cross_objectives_and_direct_A"]["cross_objective_table"]
    a_metrics["t2a-a-m1x-accepted"] = fair["theta_op_long"]["direct_A_metrics"]
    a_metrics["t2a-a-m2x-independent-accepted"] = fair["theta_disc_long"]["direct_A_metrics"]
    warm = read_json(
        AUTH / "external-results/test2a/m1-to-m2-finetune/postprocess/checkpoint_metrics.json"
    )
    a_metrics["t2a-a-m2x-warm"] = warm["checkpoints"][-1]["direct_A_metrics"]
    horizon = read_json(
        AUTH / "external-results/test2a/horizon-curriculum-h1-h2-h5/postprocess/horizon_curriculum_report.json"
    )
    horizon_map = {"H1-final": "t2a-a-h1", "H2-final": "t2a-a-h2", "H5-final": "t2a-a-h5"}
    for item in horizon["entries"]:
        if item["label"] in horizon_map:
            a_metrics[horizon_map[item["label"]]] = item["offline_diagnostics"]["direct_A_metrics"]
    for run_id, metric in a_metrics.items():
        mapping = {
            "physical_RMS_error": metric["physical_rmse_A"],
            "relative_RMS_error": metric["relative_rms_error"],
            "maximum_absolute_error": metric["maximum_absolute_error"],
            "correlation": metric["correlation"],
            "physical_MAE": metric["physical_mae_A"],
            "sign_accuracy": metric["sign_accuracy"],
        }
        item = next(row for row in inventory if row["run_id"] == run_id)
        for name, value in mapping.items():
            rows.append({
                "run_id": run_id, "physical_case": "Test 2A", "representation": "A",
                "model_label": model_label(run_id), "trained_objective": item["objective"],
                "checkpoint_iteration": item["accepted_iterations_this_run"],
                "checkpoint_parameter_pytree_sha256": "", "checkpoint_npz_sha256": "",
                "checkpoint_path": item["checkpoint_path"], "evaluation_state": "X",
                "support": "TRAINING SUPPORT", "state_first": 0, "state_last": 80,
                "support_use": "FINAL STORED DIAGNOSTIC", "quantity": "A",
                "metric": name, "value": value,
                "metric_provenance": "FINAL STORED DIAGNOSTIC",
                "nominal_state_for_model": "true",
            })

    comparison = read_json(
        AUTH / "external-results/test2a/problem-b/production/problem_b_comparison.json"
    )
    c_run = {
        "M1": "t2a-c-m1x", "M2-X-independent": "t2a-c-m2x-independent",
        "M1-to-M2-X": "t2a-c-m2x-warm", "H1": "t2a-c-h1",
        "H2": "t2a-c-h2", "H5": "t2a-c-h5",
    }
    for accepted, run_id in c_run.items():
        diagnostics = comparison["artifacts"][accepted]["structural_diagnostics_on_boundary_truth_support"]
        item = next(row for row in inventory if row["run_id"] == run_id)
        for component, value in diagnostics["component_physical_rms_error"].items():
            rows.append({
                "run_id": run_id, "physical_case": "Test 2A", "representation": "C",
                "model_label": model_label(run_id), "trained_objective": item["objective"],
                "checkpoint_iteration": item["accepted_iterations_this_run"],
                "checkpoint_parameter_pytree_sha256": comparison["artifacts"][accepted]["parameter_pytree_sha256"],
                "checkpoint_npz_sha256": "", "checkpoint_path": item["checkpoint_path"],
                "evaluation_state": "X", "support": "TRAINING SUPPORT",
                "state_first": 0, "state_last": 80,
                "support_use": "FINAL STORED DIAGNOSTIC",
                "quantity": f"source_components.{component}",
                "metric": "physical_RMS_error", "value": value,
                "metric_provenance": "FINAL STORED DIAGNOSTIC",
                "nominal_state_for_model": "true",
            })
        for metric, source_key in (
            ("normalized_off_manifold_RMS", "normalized_manifold_residual_rms"),
            ("water_source_defect_RMS", "water_source_defect_rms"),
            ("S_minus_beta2_Qv_defect_RMS", "beta_source_defect_rms"),
        ):
            rows.append({
                "run_id": run_id, "physical_case": "Test 2A", "representation": "C",
                "model_label": model_label(run_id), "trained_objective": item["objective"],
                "checkpoint_iteration": item["accepted_iterations_this_run"],
                "checkpoint_parameter_pytree_sha256": comparison["artifacts"][accepted]["parameter_pytree_sha256"],
                "checkpoint_npz_sha256": "", "checkpoint_path": item["checkpoint_path"],
                "evaluation_state": "X", "support": "TRAINING SUPPORT",
                "state_first": 0, "state_last": 80,
                "support_use": "FINAL STORED DIAGNOSTIC", "quantity": "source_structure",
                "metric": metric, "value": diagnostics[source_key],
                "metric_provenance": "FINAL STORED DIAGNOSTIC",
                "nominal_state_for_model": "true",
            })

    write_csv(DATA / "final_direct_metrics.csv", rows)
    cross = [
        row for row in rows
        if row["physical_case"] == "Test 2B"
        and row["model_label"] in ("M1-X", "M1-Y")
    ]
    write_csv(DATA / "m1_cross_state_final_metrics.csv", cross)
    return rows


def load_test2b_autonomous():
    result = []
    historical_sources = {}
    for representation in "ABC":
        path = AUTH / (
            "external-results/test2b-rain-active-learning/production/"
            f"representation-{representation}/representation_"
            f"{representation.lower()}_final_comparison.json"
        )
        record = read_json(path)
        historical_sources[representation] = path
        mapping = {
            "M1": "M1-X", "M2-X-independent": "M2-X-independent",
            "M1-to-M2-X": "warm M2-X", "H1": "H1", "H2": "H2", "H5": "H5",
        }
        for accepted, label in mapping.items():
            result.append((representation, label, record["autonomous"][accepted], path))
        m1y_path = M1Y / (
            "external-results/m1y-test2b-20260828/evaluation/"
            f"representation_{representation}_matched.json"
        )
        m1y = read_json(m1y_path)
        result.append((representation, "M1-Y", m1y["standard_M1_Y"]["autonomous"], m1y_path))
    return result


def assemble_deployed_and_trajectories():
    truth_csv = GT / "outputs/ground_truth_figures_20260829/data/test2b_temporal_diagnostics.csv"
    truth = read_csv(truth_csv)
    if [int(row["step"]) for row in truth] != list(range(161)):
        raise RuntimeError("truth trajectory indices changed")
    truth_qc = np.asarray([float(row["Qc_mass"]) for row in truth])
    truth_qr = np.asarray([float(row["rain_water_mass"]) for row in truth])
    truth_water = np.asarray([float(row["total_water_mass"]) for row in truth])
    truth_rain_rate = np.asarray([float(row["rain_source_mass_rate"]) for row in truth])
    diagnostics_rows = []
    rain_rows = [{
        "physical_case": "Test 2B", "representation": "truth", "model_label": "Truth",
        "run_id": "truth", "rain_diagnostic_kind": "analytical truth R",
        "first_meaningful_rain_time_s": 5100.0, "onset_error_s": 0.0,
        "pre_truth_onset_false_positive_count": 0,
        "pre_truth_onset_false_positive_fraction": 0.0,
        "false_negative_rate_given_truth_active": 0.0,
        "time_integrated_rain_source_mass": float(np.sum(truth_rain_rate[:-1]) * 100.0),
        "time_integrated_rain_source_mass_error": 0.0,
        "final_Qr_mass": float(truth_qr[-1]), "final_Qr_mass_error": 0.0,
        "final_Qc_mass": float(truth_qc[-1]), "final_Qc_mass_error": 0.0,
        "source_path": str(truth_csv), "quantity_kind": "GROUND TRUTH",
    }]
    trajectory_rows = []
    truth_ke_reference = None
    truth_vort_reference = None
    validation = []

    for representation, label, auto, source in load_test2b_autonomous():
        run_suffix = {
            "M1-X": "m1x", "M1-Y": "m1y", "M2-X-independent": "m2x-independent",
            "warm M2-X": "m2x-warm", "H1": "h1", "H2": "h2", "H5": "h5",
        }[label]
        run_id = f"t2b-{representation.lower()}-{run_suffix}"
        boundary = auto["boundary_timeseries"]
        conservation = auto["conservation_and_stability"]
        flow_ke = auto["flow"]["kinetic_energy"]
        flow_vort = auto["flow"]["projected_enstrophy"]
        steps = list(map(int, flow_ke["steps"]))
        times = list(map(float, flow_ke["times"]))
        if steps != list(range(161)) or times != [100.0 * step for step in range(161)]:
            raise RuntimeError(f"flow time alignment changed for {run_id}")
        for key in ("Qc_mass", "Qr_mass", "Qv_mass", "total_water_mass"):
            if len(boundary[key]) != 161:
                raise RuntimeError(f"boundary trajectory incomplete for {run_id}:{key}")
        truth_ke = np.asarray(flow_ke["truth"], dtype=np.float64)
        truth_vort = np.asarray(flow_vort["truth"], dtype=np.float64)
        if truth_ke_reference is None:
            truth_ke_reference = truth_ke
            truth_vort_reference = truth_vort
        validation.append({
            "run_id": run_id,
            "kinetic_truth_max_abs_difference": float(np.max(np.abs(truth_ke - truth_ke_reference))),
            "vorticity_truth_max_abs_difference": float(np.max(np.abs(truth_vort - truth_vort_reference))),
            "time_indices_match": True,
        })
        predicted_ke = np.asarray(flow_ke["predicted"], dtype=np.float64)
        predicted_vort = np.asarray(flow_vort["predicted"], dtype=np.float64)
        qc = np.asarray(boundary["Qc_mass"], dtype=np.float64)
        qr = np.asarray(boundary["Qr_mass"], dtype=np.float64)
        water = np.asarray(boundary["total_water_mass"], dtype=np.float64)
        for step in range(161):
            trajectory_rows.append({
                "run_id": run_id, "representation": representation,
                "model_label": label, "step": step, "time_s": 100.0 * step,
                "model_Qc_mass": qc[step], "truth_Qc_mass": truth_qc[step],
                "model_Qr_mass": qr[step], "truth_Qr_mass": truth_qr[step],
                "model_total_water_mass": water[step], "truth_total_water_mass": truth_water[step],
                "model_relative_total_water_drift": (water[step] - water[0]) / water[0],
                "truth_relative_total_water_drift": (truth_water[step] - truth_water[0]) / truth_water[0],
                "model_kinetic_energy": predicted_ke[step], "truth_kinetic_energy": truth_ke[step],
                "kinetic_energy_relative_error": (predicted_ke[step] - truth_ke[step]) / truth_ke[step],
                "model_projected_relative_vorticity_squared": predicted_vort[step],
                "truth_projected_relative_vorticity_squared": truth_vort[step],
                "projected_relative_vorticity_squared_relative_error": (
                    predicted_vort[step] - truth_vort[step]
                ) / truth_vort[step],
                "source_path": str(source),
                "quantity_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC",
            })

        rain_key = "rain" if representation in ("A", "B") else "rain_source_and_partition"
        rain = auto[rain_key]
        if representation == "A":
            onset = rain.get("first_physically_meaningful_R_time")
            fp_count = "NOT_STORED_ANALYTICAL_R_ON_MODEL_STATE"
            fp_fraction = "NOT_STORED_ANALYTICAL_R_ON_MODEL_STATE"
            fn_rate = "NOT_STORED_ANALYTICAL_R_ON_MODEL_STATE"
            rain_kind = "analytical R evaluated on model-generated state"
        elif representation == "B":
            onset = rain.get("first_physically_meaningful_positive_R_time")
            fp_count = rain.get("pre_truth_onset_false_positive_R_count")
            fp_fraction = rain.get("pre_truth_onset_false_positive_R_fraction")
            fn_rate = rain.get("false_negative_rate_given_truth_active")
            rain_kind = "direct learned R"
        else:
            onset = rain.get("first_meaningful_positive_source_time")
            fp_count = rain.get("pre_truth_onset_false_positive_count")
            fp_fraction = rain.get("pre_truth_onset_false_positive_fraction")
            fn_rate = rain.get("false_negative_rate_given_truth_active")
            rain_kind = "effective R_Qr=predicted Qr source/h"

        partition = conservation["partition"]
        mixed = auto["mixed_state_error"]["ALL"]
        minima = conservation["minimum_field_coefficients"]
        if representation in ("A", "B"):
            maximum_source = conservation["maximum_source_residuals"]
            water_source_rms = 0.0
            thermo_source_rms = 0.0
            source_status = "STRUCTURALLY ENFORCED"
            source_max_water = maximum_source["water_maximum_absolute"]
            source_max_thermo = maximum_source["S_minus_beta2_Qv_maximum_absolute"]
        else:
            source_diag = auto["source_diagnostics_on_model_postprefix_states"]["ALL"]
            water_source_rms = source_diag["water_source_defect"]["RMS"]
            thermo_source_rms = source_diag["thermodynamic_source_defect"]["RMS"]
            source_max_water = source_diag["water_source_defect"]["maximum_absolute"]
            source_max_thermo = source_diag["thermodynamic_source_defect"]["maximum_absolute"]
            source_status = "LEARNED / NOT ENFORCED"
        diagnostics_rows.append({
            "physical_case": "Test 2B", "run_id": run_id,
            "representation": representation, "model_label": label,
            "maximum_absolute_total_water_drift": conservation["maximum_absolute_total_water_drift"],
            "relative_maximum_total_water_drift": conservation["relative_maximum_total_water_drift"],
            "final_total_water_drift": conservation["final_total_water_mass"] - conservation["initial_total_water_mass"],
            "final_mixed_state_error": mixed["final"],
            "maximum_mixed_state_error": mixed["maximum"],
            "minimum_Qv_coefficient": minima["Qv"],
            "minimum_Qc_coefficient": minima["Qc"],
            "minimum_Qr_coefficient": minima["Qr"],
            "minimum_moisture_coefficient": min(minima["Qv"], minima["Qc"], minima["Qr"]),
            "final_Qc_mass": partition["Qc"]["final"],
            "final_Qc_mass_error": partition["Qc"]["final"] - truth_qc[-1],
            "final_Qr_mass": partition["Qr"]["final"],
            "final_Qr_mass_error": partition["Qr"]["final"] - truth_qr[-1],
            "rain_onset_time_s": onset,
            "rain_onset_error_s": None if onset is None else onset - 5100.0,
            "pre_truth_onset_false_positive_count": fp_count,
            "pre_truth_onset_false_positive_fraction": fp_fraction,
            "false_negative_rate_given_truth_active": fn_rate,
            "time_integrated_rain_source_mass": rain["time_integrated_rain_source_mass"],
            "time_integrated_rain_source_mass_error": rain["time_integrated_rain_source_mass_error"],
            "water_source_identity_status": source_status,
            "water_source_defect_RMS": water_source_rms,
            "water_source_defect_maximum_absolute": source_max_water,
            "thermodynamic_source_identity_status": source_status,
            "S_minus_beta2_Qv_source_defect_RMS": thermo_source_rms,
            "S_minus_beta2_Qv_source_defect_maximum_absolute": source_max_thermo,
            "all_state_coefficients_finite": conservation["all_state_coefficients_finite"],
            "source_path": str(source),
            "quantity_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC",
        })
        rain_rows.append({
            "physical_case": "Test 2B", "representation": representation,
            "model_label": label, "run_id": run_id,
            "rain_diagnostic_kind": rain_kind,
            "first_meaningful_rain_time_s": onset,
            "onset_error_s": None if onset is None else onset - 5100.0,
            "pre_truth_onset_false_positive_count": fp_count,
            "pre_truth_onset_false_positive_fraction": fp_fraction,
            "false_negative_rate_given_truth_active": fn_rate,
            "time_integrated_rain_source_mass": rain["time_integrated_rain_source_mass"],
            "time_integrated_rain_source_mass_error": rain["time_integrated_rain_source_mass_error"],
            "final_Qr_mass": partition["Qr"]["final"],
            "final_Qr_mass_error": partition["Qr"]["final"] - truth_qr[-1],
            "final_Qc_mass": partition["Qc"]["final"],
            "final_Qc_mass_error": partition["Qc"]["final"] - truth_qc[-1],
            "source_path": str(source),
            "quantity_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC",
        })

    if any(
        item["kinetic_truth_max_abs_difference"] != 0.0
        or item["vorticity_truth_max_abs_difference"] != 0.0
        for item in validation
    ):
        raise RuntimeError("embedded truth flow trajectories differ across models")
    write_csv(DATA / "deployed_diagnostics.csv", diagnostics_rows)
    write_csv(DATA / "rain_event_diagnostics.csv", rain_rows)
    trajectory_path = DATA / "global_trajectories/test2b_global_trajectories.csv"
    write_csv(trajectory_path, trajectory_rows)
    write_json(DATA / "global_trajectories/test2b_global_trajectories.json", {
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "record_count": len(trajectory_rows),
        "model_count": 21,
        "state_indices": [0, 160],
        "time_seconds": [0.0, 16000.0],
        "truth_source": str(truth_csv),
        "truth_source_sha256": file_sha256(truth_csv),
        "validation": validation,
        "definitions": {
            "Qc_mass": "domain integral of conservative cloud water Qc",
            "Qr_mass": "domain integral of conservative rain water Qr",
            "total_water_mass": "domain integral of Qv+Qc+Qr",
            "kinetic_energy": "0.5 integral h |v|^2 dA; not total energy",
            "projected_relative_vorticity_squared": (
                "0.5 integral zeta_h^2 dA for CG(3) L2-projected relative vorticity; "
                "not potential enstrophy"
            ),
            "mixed_state_error": (
                "only final/max/regime summaries are stored; a time series is not available "
                "without rerollout"
            ),
        },
        "quantity_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC",
        "rollout_performed": False,
    })
    return diagnostics_rows, rain_rows, trajectory_rows


def build_table1():
    headers = [
        "Case", "objective family", "mesh / dt", "fitting support",
        "evaluation / target state", "features", "scaling / weighting", "evaluation terminology",
    ]
    rows = [
        ["Test 2A", "M1-X / M2-X", "16x16 / 100 s", "states 0--80; 4096 samples/state", "X; local law or fixed deployed child", "(h,S,Qv,Qc,B)", "X 0--80 statistics; objective-specific output scale; packed/deployed mass metric", "training support only"],
        ["Test 2A", "H1", "16x16 / 100 s", "starts 0--79", "Y=P(X); one-step truth boundary target", "(h,S,Qv,Qc,B)", "same frozen X scaling; deployed mixed-state metric", "training-support windows"],
        ["Test 2A", "H2 / H5", "16x16 / 100 s", "40 H2 / 16 H5 windows in 0--80", "recursive model state; truth boundary targets", "(h,S,Qv,Qc,B)", "same frozen scaling; accumulated mixed-state metric", "no held-out/test set; 81--160 unused by recorded learning"],
        ["Test 2B", "M1-X / M2-X", "64x64 / 100 s", "states 0--80; 65,536 samples/state", "X; local law or fixed deployed child", "(h,S,Qv,Qc,B)", "X 0--80 statistics/output scales; carrier-mass weighting", "training truth support"],
        ["Test 2B", "M1-Y / H1", "64x64 / 100 s", "M1-Y states 0--80; H1 starts 0--79", "Y=P(X); local law or one-step truth boundary target", "(h,S,Qv,Qc,B)", "historical X scaling retained; carrier/mixed-state weighting", "training truth support"],
        ["Test 2B", "H2 / H5", "64x64 / 100 s", "40 H2 / 16 H5 windows in 0--80", "recursive model state; truth boundary targets", "(h,S,Qv,Qc,B)", "same frozen X scaling; accumulated mixed-state metric", "states 81--160 are post-hoc held-out truth support; temporally adjacent"],
    ]
    (TABLES / "table1_data_supports.md").write_text(
        "# Table 1. Data supports and evaluation protocol\n\n" + md_table(headers, rows)
        + "\nTest 2A has 161 stored states but no historically defined held-out/test set. "
        "Test 2B held-out values reported here are post-hoc and were not stopping or selection signals.\n",
        encoding="utf-8",
    )
    (TABLES / "table1_data_supports.tex").write_text(
        tex_table(headers, rows, "llllllll"), encoding="utf-8"
    )


def build_table2(inventory):
    headers = [
        "Case", "Rep.", "model", "evaluation state", "target",
        "architecture (params)", "initialization", "budget / accepted",
        "optimizer", "stop",
    ]
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    selected = sorted(inventory, key=lambda row: (
        row["physical_case"], row["representation"], order[model_label(row["run_id"])]
    ))
    rows = []
    for item in selected:
        optimizer = item["optimizer"] or "PyROL/ROL line-search L-BFGS"
        memory = item["lbfgs_memory"] or "20"
        rows.append([
            item["physical_case"], item["representation"], model_label(item["run_id"]),
            item["training_evaluation_state"], item["target_state"],
            f"{item['architecture']} ({item['parameter_count']})",
            item["initialization_kind"] + (f" from {item['initialization_source']}" if item["initialization_source"] else ""),
            f"{item['iteration_budget']} / {item['accepted_iterations_this_run']}",
            f"{optimizer}; m={memory}", item["stopping_reason"],
        ])
    (TABLES / "table2_training_contracts.md").write_text(
        "# Table 2. Scientific training-run contracts\n\n" + md_table(headers, rows)
        + "\nAll models use float64 and seed 0. H1/H2/H5 are sequential warm starts; "
        "objective, initialization history, and budget therefore change together.\n",
        encoding="utf-8",
    )
    (TABLES / "table2_training_contracts.tex").write_text(
        tex_table(headers, rows, "llllllllll"), encoding="utf-8"
    )


def metric_lookup(rows):
    return {
        (row["run_id"], row["evaluation_state"], row["support"], row["quantity"], row["metric"]): row["value"]
        for row in rows
    }


def build_table3(final_direct):
    lookup = metric_lookup(final_direct)
    lines_md = ["# Table 3. Final direct prediction accuracy", ""]
    lines_tex = []
    for representation in "ABC":
        if representation == "A":
            headers = ["Model", "Support", "A RMS", "A rel. RMS", "A bias", "A corr."]
        elif representation == "B":
            headers = ["Model", "Support", "A rel. RMS", "R rel. RMS", "active-R rel. RMS", "R FP rate", "R FN rate"]
        else:
            headers = ["Model", "Support", "S RMS", "Qv RMS", "Qc RMS", "Qr RMS", "effective A rel.", "effective R rel.", "off-manifold"]
        table_rows = []
        for label in MODEL_ORDER:
            run_id = f"t2b-{representation.lower()}-{ {'M1-X':'m1x','M1-Y':'m1y','M2-X-independent':'m2x-independent','warm M2-X':'m2x-warm','H1':'h1','H2':'h2','H5':'h5'}[label] }"
            for support, support_label in (
                ("TRAINING TRUTH SUPPORT", "train 0--80"),
                ("HELD-OUT TRUTH SUPPORT", "held-out 81--160"),
            ):
                key = lambda quantity, metric: lookup.get((run_id, "X", support, quantity, metric), "")
                if representation == "A":
                    row = [label, support_label, fmt(key("A", "physical_RMS_error")), fmt(key("A", "relative_RMS_error")), fmt(key("A", "signed_mass_weighted_bias")), fmt(key("A", "correlation"))]
                elif representation == "B":
                    row = [label, support_label, fmt(key("A", "relative_RMS_error")), fmt(key("R_all", "relative_RMS_error")), fmt(key("R_truth_active", "relative_RMS_error")), fmt(key("R_activation", "false_positive_rate_given_truth_inactive")), fmt(key("R_activation", "false_negative_rate_given_truth_active"))]
                else:
                    row = [label, support_label] + [fmt(key(f"source_components.{name}", "physical_RMS_error")) for name in SOURCE_ORDER] + [fmt(key("effective_A", "relative_RMS_error")), fmt(key("effective_R", "relative_RMS_error")), fmt(key("source_structure", "normalized_off_manifold_RMS"))]
                table_rows.append(row)
        lines_md.extend((f"## Representation {representation}", "", md_table(headers, table_rows)))
        lines_tex.extend((f"% Representation {representation}", tex_table(headers, table_rows)))
    lines_md.extend((
        "## Test 2A scope", "",
        "Test 2A accuracy is available only on training support. Representation A has complete direct-A diagnostics; Representation C has stored component RMS and structural diagnostics but not a historically matched held-out suite. These values remain in `data/final_direct_metrics.csv` and are not mixed into the Test 2B panels.",
    ))
    (TABLES / "table3_final_direct_accuracy.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    (TABLES / "table3_final_direct_accuracy.tex").write_text("\n".join(lines_tex), encoding="utf-8")


def build_table4(matrix):
    lines_md = ["# Table 4. Frozen-model objective matrix", ""]
    lines_tex = []
    for case in ("Test 2B", "Test 2A"):
        for representation in ("A", "B", "C"):
            panel = [row for row in matrix if row["physical_case"] == case and row["representation"] == representation]
            if not panel:
                continue
            columns = OBJECTIVE_COLUMNS if case == "Test 2B" else tuple(x for x in OBJECTIVE_COLUMNS if x != "J_M1_Y")
            headers = ["Model"] + [column.replace("_", "-") for column in columns]
            table_rows = []
            for row in panel:
                values = []
                for column in columns:
                    value = fmt(row[column])
                    if row[f"{column}_role"] == "FITTED" and value != "--":
                        value = value + " *"
                    values.append(value)
                table_rows.append([row["model_label"], *values])
            lines_md.extend((f"## {case}, Representation {representation}", "", md_table(headers, table_rows)))
            lines_tex.extend((f"% {case}, Representation {representation}", tex_table(headers, table_rows)))
    lines_md.append("`*` marks the fitted objective. Every other populated cell is a post-hoc diagnostic evaluation. M1-Y is not defined for the historical Test 2A campaign.")
    (TABLES / "table4_objective_matrix.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    (TABLES / "table4_objective_matrix.tex").write_text("\n".join(lines_tex), encoding="utf-8")


def build_table5(rain_rows):
    headers = [
        "Rep.", "Model", "rain diagnostic", "onset (s)", "onset error (s)",
        "pre-onset FP", "truth-active FN rate", "integrated rain error", "final Qr error", "final Qc error",
    ]
    rows = []
    for item in rain_rows:
        rows.append([
            item["representation"], item["model_label"], item["rain_diagnostic_kind"],
            fmt(item["first_meaningful_rain_time_s"]), fmt(item["onset_error_s"]),
            fmt(item["pre_truth_onset_false_positive_fraction"]),
            fmt(item["false_negative_rate_given_truth_active"]),
            fmt(item["time_integrated_rain_source_mass_error"]),
            fmt(item["final_Qr_mass_error"]), fmt(item["final_Qc_mass_error"]),
        ])
    (TABLES / "table5_rain_events.md").write_text(
        "# Table 5. Rain-event and water-partition diagnostics\n\n" + md_table(headers, rows)
        + "\nRepresentation A uses analytical R on the model-generated state; a learned-rate FP/FN count is therefore not stored. Representation B learns R directly. Representation C is labeled by the effective Qr-source rate.\n",
        encoding="utf-8",
    )
    (TABLES / "table5_rain_events.tex").write_text(
        tex_table(headers, rows, "llllllllll"), encoding="utf-8"
    )


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    inventory = main_inventory()
    if len(inventory) != 33:
        raise RuntimeError(f"expected 33 main runs, found {len(inventory)}")
    matrix = complete_objective_matrix(inventory)
    histories = assemble_histories(inventory)
    final_direct = assemble_final_direct(inventory)
    deployed, rain, trajectories = assemble_deployed_and_trajectories()
    build_table1()
    build_table2(inventory)
    build_table3(final_direct)
    build_table4(matrix)
    build_table5(rain)
    write_json(DATA / "assembled_data_validation.json", {
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "main_run_count": len(inventory),
        "objective_matrix_rows": len(matrix),
        "checkpoint_training_objective_rows": len(histories),
        "final_direct_metric_rows": len(final_direct),
        "deployed_diagnostic_rows": len(deployed),
        "rain_rows_including_truth": len(rain),
        "global_trajectory_rows": len(trajectories),
        "no_training": True,
        "no_truth_generation": True,
        "no_rollout": True,
        "recursive_H2_H5_history_reconstruction": False,
    })
    print(json.dumps({
        "status": "complete", "main_runs": len(inventory),
        "objective_matrix": len(matrix), "objective_histories": len(histories),
        "final_direct": len(final_direct), "deployed": len(deployed),
        "rain": len(rain), "trajectories": len(trajectories),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
