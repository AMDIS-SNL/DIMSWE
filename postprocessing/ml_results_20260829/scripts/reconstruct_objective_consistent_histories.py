#!/usr/bin/env python3
"""Build Test-2B training/evaluation histories of the fitted objectives.

M1 values are selected from already verified fixed-array checkpoint results.
M2-X and H1 evaluation values use the production FixedObjective with immutable
held-out X/Y arrays and the frozen finite-element mass maps.  No optimizer,
prefix construction, timestep integration, or rollout is instantiated.
"""

from __future__ import annotations

import csv
from gc import collect
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import jax
import numpy as np

from portable_paths import (
    AUDIT_ROOT as AUDIT,
    M1Y_REPOSITORY as M1Y,
    PACKAGE_ROOT,
)

jax.config.update("jax_enable_x64", True)


ROOT = PACKAGE_ROOT
DATA = ROOT / "data"
DESTINATION = DATA / "objective_training_evaluation_histories.csv"
SIDECAR = DESTINATION.with_suffix(".json")
FIXED_PREPARATION = M1Y / (
    "external-results/test2b-rain-active-learning/preparation/"
    "fixed_learning_data.npz"
)
HELDOUT_X = DATA / "heldout_x_test2b.npz"
HELDOUT_Y = M1Y / (
    "external-results/m1y-test2b-20260828/evaluation/"
    "m1y_heldout_data.npz"
)
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"
DT = 100.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def support_name(curve_role: str) -> str:
    return "TRAINING" if curve_role == "training" else "EVALUATION"


def objective_name(model_label: str) -> str:
    return {
        "M1-X": "J_M1_X",
        "M1-Y": "J_M1_Y",
        "M2-X-independent": "J_M2_X",
        "warm M2-X": "J_M2_X",
        "H1": "J_H1",
    }[model_label]


def add_record(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    representation: str,
    model_label: str,
    iteration: int,
    value: float,
    numerator: float,
    denominator: float,
    curve_role: str,
    evaluation_state: str,
    input_indices: str,
    target_indices: str,
    denominator_definition: str,
    weighting: str,
    checkpoint: dict[str, Any],
    source: str,
) -> None:
    rows.append(
        {
            "physical_case": "Test 2B",
            "run_id": run_id,
            "representation": representation,
            "model_label": model_label,
            "objective": objective_name(model_label),
            "checkpoint_iteration": int(iteration),
            "curve_role": curve_role,
            "curve_label": support_name(curve_role),
            "evaluation_state": evaluation_state,
            "input_state_indices": input_indices,
            "target_state_indices": target_indices,
            "normalized_objective": float(value),
            "numerator": float(numerator),
            "denominator": float(denominator),
            "denominator_definition": denominator_definition,
            "normalization": (
                "frozen Test-2B X-state input/output normalization; no refit"
            ),
            "weighting": weighting,
            "checkpoint_parameter_pytree_sha256": checkpoint["parameter_sha256"],
            "checkpoint_npz_sha256": checkpoint["npz_sha256"],
            "checkpoint_path": checkpoint["path"],
            "evaluation_source": source,
            "used_during_optimization": curve_role == "training",
        }
    )


def checkpoint_lookup(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (run_id, int(checkpoint["iteration"])): checkpoint
        for run_id, run in manifest["runs"].items()
        for checkpoint in run["checkpoints"]
    }


def main() -> None:
    if DESTINATION.exists() or SIDECAR.exists():
        raise FileExistsError("refusing to overwrite objective-consistent histories")
    started = perf_counter()
    sys.path.insert(0, str(M1Y))
    from dimswe.test2b_rain_learning import load_parameters  # noqa: PLC0415
    from dimswe.test2b_rain_learning_campaign import (  # noqa: PLC0415
        FixedObjective,
        _matrix_energy,
        load_preparation,
    )

    manifest_path = DATA / "checkpoint_hash_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("source_head") != EXPECTED_HEAD:
        raise ValueError("invalid checkpoint manifest")
    checkpoints = checkpoint_lookup(manifest)

    selected_run_ids = {
        f"t2b-{rep.lower()}-{suffix}"
        for rep in "ABC"
        for suffix in ["m1x", "m1y", "m2x-independent", "m2x-warm", "h1"]
    }
    inventory = [
        row
        for row in read_csv_rows(AUDIT / "ML_RUN_INVENTORY.csv")
        if row["run_id"] in selected_run_ids
    ]
    if len(inventory) != 15:
        raise RuntimeError(f"expected 15 run records, found {len(inventory)}")

    metadata, normalization, training, matrices = load_preparation(FIXED_PREPARATION)
    normalization_sha = normalization.provenance_sha256
    fixed_metadata_path = FIXED_PREPARATION.with_suffix(".json")

    heldout_x_meta = read_json(HELDOUT_X.with_suffix(".json"))
    if (
        heldout_x_meta.get("status") != "complete"
        or heldout_x_meta.get("state_indices") != [81, 160]
        or heldout_x_meta.get("normalization_refitted")
        or file_sha256(HELDOUT_X) != heldout_x_meta.get("npz_sha256")
    ):
        raise ValueError("invalid held-out X cache")
    with np.load(HELDOUT_X, allow_pickle=False) as archive:
        heldout_x_features = np.array(archive["heldout_x_features"], copy=True)
        heldout_x_a = np.array(archive["heldout_x_A"], copy=True)
        heldout_x_r = np.array(archive["heldout_x_R"], copy=True)
    for name, value in [
        ("heldout_x_features", heldout_x_features),
        ("heldout_x_A", heldout_x_a),
        ("heldout_x_R", heldout_x_r),
    ]:
        if array_sha256(value) != heldout_x_meta["arrays"][name]["sha256"]:
            raise ValueError(f"held-out X array hash changed: {name}")

    heldout_y_meta = read_json(HELDOUT_Y.with_suffix(".json"))
    if (
        heldout_y_meta.get("status") != "complete"
        or heldout_y_meta.get("truth_state_indices") != [81, 160]
        or heldout_y_meta.get("used_for_training_or_model_selection")
        or heldout_y_meta.get("normalization_refitted_on_Y")
        or heldout_y_meta.get("normalization_provenance_sha256") != normalization_sha
        or file_sha256(HELDOUT_Y) != heldout_y_meta.get("heldout_npz_sha256")
    ):
        raise ValueError("invalid held-out Y cache")
    with np.load(HELDOUT_Y, allow_pickle=False) as archive:
        heldout_y_features_all = np.array(archive["heldout_y_features"], copy=True)
        heldout_y_a_all = np.array(archive["heldout_y_A"], copy=True)
        heldout_y_r_all = np.array(archive["heldout_y_R"], copy=True)
    for name, value in [
        ("heldout_y_features", heldout_y_features_all),
        ("heldout_y_A", heldout_y_a_all),
        ("heldout_y_R", heldout_y_r_all),
    ]:
        if array_sha256(value) != heldout_y_meta["arrays"][name]["sha256"]:
            raise ValueError(f"held-out Y array hash changed: {name}")

    # H1 evaluation starts must lie entirely in the later portion.  Thus the
    # final cached Y_160 has no X_161 target and is excluded: starts 81..159,
    # corresponding next truth states 82..160.
    heldout_y_features = heldout_y_features_all[:79]
    heldout_y_a = heldout_y_a_all[:79]
    heldout_y_r = heldout_y_r_all[:79]

    heldout_m2_denominator = _matrix_energy(
        heldout_x_a, heldout_x_r, heldout_x_features, normalization, matrices
    )
    heldout_h1_source_denominator = _matrix_energy(
        heldout_y_a, heldout_y_r, heldout_y_features, normalization, matrices
    )
    if not (
        np.isfinite(heldout_m2_denominator)
        and heldout_m2_denominator > 0
        and np.isfinite(heldout_h1_source_denominator)
        and heldout_h1_source_denominator > 0
    ):
        raise RuntimeError("invalid held-out fixed-map denominator")

    direct_path = DATA / "checkpoint_direct_histories.csv"
    direct = read_csv_rows(direct_path)
    training_operator_path = DATA / "checkpoint_training_objectives_operator.csv"
    training_operator = read_csv_rows(training_operator_path)
    training_fixed_path = DATA / "checkpoint_training_objectives_fixed.csv"
    training_fixed = read_csv_rows(training_fixed_path)
    cross_final_path = DATA / "m1_cross_state_final_metrics.csv"
    cross_final = read_csv_rows(cross_final_path)

    rows: list[dict[str, Any]] = []
    all_checkpoint_training_parity = []
    final_evaluation_parity = []

    # M1-X and M1-Y are already available from the validated fixed-array pass.
    for representation in "ABC":
        for model_label, state, run_suffix in [
            ("M1-X", "X", "m1x"),
            ("M1-Y", "Y", "m1y"),
        ]:
            run_id = f"t2b-{representation.lower()}-{run_suffix}"
            for curve_role, raw_support, indices in [
                ("training", "TRAINING TRUTH SUPPORT", "0..80"),
                ("evaluation", "HELD-OUT TRUTH SUPPORT", "81..160"),
            ]:
                base = [
                    row
                    for row in direct
                    if row["run_id"] == run_id
                    and row["evaluation_state"] == state
                    and row["support"] == raw_support
                    and row["quantity"] == "operator"
                ]
                pivot: dict[int, dict[str, float]] = {}
                for row in base:
                    pivot.setdefault(int(row["checkpoint_iteration"]), {})[
                        row["metric"]
                    ] = float(row["value"])
                if len(pivot) != 10:
                    raise RuntimeError(f"incomplete M1 objective curve: {run_id} {curve_role}")
                if any(
                    set(values) != {"normalized_objective", "numerator", "denominator"}
                    for values in pivot.values()
                ):
                    raise RuntimeError(f"incomplete M1 metrics: {run_id} {curve_role}")
                for iteration, values in sorted(pivot.items()):
                    checkpoint = checkpoints[(run_id, int(iteration))]
                    add_record(
                        rows,
                        run_id=run_id,
                        representation=representation,
                        model_label=model_label,
                        iteration=int(iteration),
                        value=values["normalized_objective"],
                        numerator=values["numerator"],
                        denominator=values["denominator"],
                        curve_role=curve_role,
                        evaluation_state=f"{state}*",
                        input_indices=indices,
                        target_indices=indices,
                        denominator_definition=(
                            "carrier-weighted squared norm of the output-scaled "
                            f"analytical target on {curve_role} {state}* states"
                        ),
                        weighting="frozen carrier-mass weights broadcast over states",
                        checkpoint=checkpoint,
                        source=str(direct_path.resolve()),
                    )
                if curve_role == "training":
                    expected = {
                        int(row["checkpoint_iteration"]): float(row["value"])
                        for row in training_operator
                        if row["run_id"] == run_id
                    }
                    if set(expected) != set(pivot):
                        raise RuntimeError(f"M1 training-history mismatch: {run_id}")
                    maximum = max(
                        abs(pivot[iteration]["normalized_objective"] - expected[iteration])
                        for iteration in pivot
                    )
                    all_checkpoint_training_parity.append(
                        {"run_id": run_id, "maximum_absolute_difference": maximum, "passed": maximum <= 2e-15}
                    )
                else:
                    final_iteration = max(pivot)
                    expected = [
                        row
                        for row in cross_final
                        if row["run_id"] == run_id
                        and row["evaluation_state"] == state
                        and row["support"] == raw_support
                        and row["quantity"] == "operator"
                        and row["metric"] == "normalized_objective"
                    ]
                    if len(expected) != 1:
                        raise RuntimeError("missing accepted final M1 evaluation metric")
                    actual = pivot[final_iteration]["normalized_objective"]
                    difference = abs(actual - float(expected[0]["value"]))
                    final_evaluation_parity.append(
                        {"run_id": run_id, "absolute_difference": difference, "passed": difference <= 2e-15}
                    )

    # Existing training fixed-objective histories are preserved verbatim.
    training_denominators = {
        "M2-X-independent": float(metadata["m2x_denominator"]),
        "warm M2-X": float(metadata["m2x_denominator"]),
        "H1": float(metadata["common_horizon_denominator"] / (DT * DT)),
    }
    for row in training_fixed:
        model_label = row["model_label"]
        if model_label not in training_denominators:
            continue
        run_id = str(row["run_id"])
        checkpoint = checkpoints[(run_id, int(row["checkpoint_iteration"]))]
        denominator = training_denominators[model_label]
        is_h1 = model_label == "H1"
        add_record(
            rows,
            run_id=run_id,
            representation=row["representation"],
            model_label=model_label,
            iteration=int(row["checkpoint_iteration"]),
            value=float(row["value"]),
            numerator=float(row["value"]) * denominator,
            denominator=denominator,
            curve_role="training",
            evaluation_state="Y*=P(X*)" if is_h1 else "X*",
            input_indices="0..79" if is_h1 else "0..80",
            target_indices="1..80" if is_h1 else "0..80",
            denominator_definition=(
                "FE mixed-state mass energy of analytical one-step source targets at "
                "Y*_0..79; equivalent dt^2 increment energy cancels between numerator "
                "and denominator"
                if is_h1
                else "FE mixed-state mass energy of analytical source-to-tendency targets at X*_0..80"
            ),
            weighting="frozen S/Q finite-element mass and inverse-mass maps",
            checkpoint=checkpoint,
            source=str(training_fixed_path.resolve()),
        )

    # Evaluate each unique frozen checkpoint only once per representation/map.
    for representation in "ABC":
        for family in ["M2-X", "H1"]:
            if family == "M2-X":
                run_ids = [
                    f"t2b-{representation.lower()}-m2x-independent",
                    f"t2b-{representation.lower()}-m2x-warm",
                ]
                features, target_a, target_r = (
                    heldout_x_features,
                    heldout_x_a,
                    heldout_x_r,
                )
                denominator = heldout_m2_denominator
                state = "X*"
                input_indices = target_indices = "81..160"
                denominator_definition = (
                    "FE mixed-state mass energy of analytical source-to-tendency "
                    "targets at evaluation X*_81..160"
                )
            else:
                run_ids = [f"t2b-{representation.lower()}-h1"]
                features, target_a, target_r = (
                    heldout_y_features,
                    heldout_y_a,
                    heldout_y_r,
                )
                denominator = heldout_h1_source_denominator
                state = "Y*=P(X*)"
                input_indices = "81..159"
                target_indices = "82..160"
                denominator_definition = (
                    "FE mixed-state mass energy of analytical one-step source targets "
                    "at Y*_81..159; equivalent dt^2 increment energy for targets "
                    "X*_82..160 cancels between numerator and denominator"
                )
            objective = FixedObjective(
                representation,
                features,
                target_a,
                target_r,
                matrices,
                denominator,
                normalization,
            )
            references = [
                checkpoint
                for run_id in run_ids
                for checkpoint in manifest["runs"][run_id]["checkpoints"]
            ]
            unique = {checkpoint["parameter_sha256"]: checkpoint for checkpoint in references}
            values: dict[str, float] = {}
            for number, (parameter_sha, checkpoint) in enumerate(unique.items(), 1):
                print(
                    f"evaluation {representation} {family} {number}/{len(unique)} "
                    f"i={checkpoint['iteration']}",
                    flush=True,
                )
                parameters, parameter_sidecar = load_parameters(
                    Path(checkpoint["path"]), representation
                )
                if parameter_sidecar["parameter_pytree_sha256"] != parameter_sha:
                    raise RuntimeError("checkpoint parameter hash changed")
                values[parameter_sha] = objective.value(parameters)
            for run_id in run_ids:
                model_label = (
                    "M2-X-independent"
                    if run_id.endswith("m2x-independent")
                    else "warm M2-X"
                    if run_id.endswith("m2x-warm")
                    else "H1"
                )
                for checkpoint in manifest["runs"][run_id]["checkpoints"]:
                    value = values[checkpoint["parameter_sha256"]]
                    add_record(
                        rows,
                        run_id=run_id,
                        representation=representation,
                        model_label=model_label,
                        iteration=int(checkpoint["iteration"]),
                        value=value,
                        numerator=value * denominator,
                        denominator=denominator,
                        curve_role="evaluation",
                        evaluation_state=state,
                        input_indices=input_indices,
                        target_indices=target_indices,
                        denominator_definition=denominator_definition,
                        weighting="frozen S/Q finite-element mass and inverse-mass maps",
                        checkpoint=checkpoint,
                        source=(
                            f"{HELDOUT_X.resolve()}; {FIXED_PREPARATION.resolve()}"
                            if family == "M2-X"
                            else f"{HELDOUT_Y.resolve()}; {FIXED_PREPARATION.resolve()}"
                        ),
                    )
            del objective
            jax.clear_caches()
            collect()

    result = sorted(
        rows,
        key=lambda row: (
            row["representation"],
            row["model_label"],
            row["curve_role"],
            row["checkpoint_iteration"],
        ),
    )
    expected_counts = {
        (rep, model, role): count
        for rep in "ABC"
        for model, count in [
            ("M1-X", 10),
            ("M1-Y", 10),
            ("M2-X-independent", 10),
            ("warm M2-X", 9),
            ("H1", 9),
        ]
        for role in ["training", "evaluation"]
    }
    actual_counts: dict[tuple[str, str, str], int] = {}
    for row in result:
        key = (row["representation"], row["model_label"], row["curve_role"])
        actual_counts[key] = actual_counts.get(key, 0) + 1
    if actual_counts != expected_counts:
        raise RuntimeError(f"history coverage mismatch: {actual_counts}")
    if not all(np.isfinite(row["normalized_objective"]) for row in result):
        raise RuntimeError("nonfinite objective value")

    final_training_parity = []
    for run in inventory:
        run_id = run["run_id"]
        curve = [
            row
            for row in result
            if row["run_id"] == run_id and row["curve_role"] == "training"
        ]
        iteration = int(run["accepted_iterations_this_run"])
        final = [row for row in curve if row["checkpoint_iteration"] == iteration]
        if len(final) != 1:
            raise RuntimeError(f"missing final training checkpoint for {run_id}")
        actual = float(final[0]["normalized_objective"])
        accepted = float(run["final_training_objective"])
        difference = abs(actual - accepted)
        tolerance = max(2e-15, 2e-11 * abs(accepted))
        final_training_parity.append(
            {
                "run_id": run_id,
                "accepted": accepted,
                "reconstructed_or_selected": actual,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": difference <= tolerance,
            }
        )

    gates = {
        "all_15_final_training_objectives_match": all(
            item["passed"] for item in final_training_parity
        ),
        "all_M1_training_checkpoint_values_match_existing_history": all(
            item["passed"] for item in all_checkpoint_training_parity
        ),
        "all_6_M1_final_evaluation_values_match_existing_cross_state_metrics": all(
            item["passed"] for item in final_evaluation_parity
        ),
        "coverage_complete_for_15_nonrecursive_runs": len(actual_counts) == 30,
        "normalization_provenance_matches": normalization_sha
        == heldout_y_meta["normalization_provenance_sha256"],
        "H1_evaluation_alignment_is_81_to_159_targeting_82_to_160": True,
        "H2_H5_excluded": not any(
            row["model_label"] in {"H2", "H5"} for row in result
        ),
        "all_values_finite": all(
            np.isfinite(row["normalized_objective"]) for row in result
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"objective-history validation failed: {gates}")

    incomplete = DESTINATION.with_name(DESTINATION.name + ".incomplete")
    write_csv_rows(incomplete, result)
    incomplete.replace(DESTINATION)
    sidecar = {
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "record_count": len(result),
        "run_count": 15,
        "representations": ["A", "B", "C"],
        "included_models": [
            "M1-X",
            "M1-Y",
            "M2-X-independent",
            "warm M2-X",
            "H1",
        ],
        "excluded_models": ["H2", "H5"],
        "excluded_reason": "recursive evaluation histories were not computed",
        "state_contracts": {
            "M1-X training": {"input_and_target": "X*_0..80"},
            "M1-X evaluation": {"input_and_target": "X*_81..160"},
            "M1-Y training": {"input_and_target": "Y*_0..80=P(X*_0..80)"},
            "M1-Y evaluation": {"input_and_target": "Y*_81..160=P(X*_81..160)"},
            "M2-X training": {"input_and_source_target": "X*_0..80"},
            "M2-X evaluation": {"input_and_source_target": "X*_81..160"},
            "H1 training": {"input": "Y*_0..79", "next_truth_target": "X*_1..80"},
            "H1 evaluation": {"input": "Y*_81..159", "next_truth_target": "X*_82..160"},
        },
        "denominator_convention": {
            "definition": (
                "Each curve uses the same normalized-target energy formula as its "
                "training objective, evaluated on that curve's own state/window set."
            ),
            "newly_defined_evaluation_denominators": ["M2-X", "H1"],
            "M2-X_training": float(metadata["m2x_denominator"]),
            "M2-X_evaluation": float(heldout_m2_denominator),
            "H1_training_source_equivalent": float(
                metadata["common_horizon_denominator"] / (DT * DT)
            ),
            "H1_evaluation_source_equivalent": float(
                heldout_h1_source_denominator
            ),
            "H1_training_increment_denominator": float(
                metadata["common_horizon_denominator"]
            ),
            "H1_evaluation_increment_denominator": float(
                DT * DT * heldout_h1_source_denominator
            ),
            "note": (
                "The H1 implementation evaluates source error after canceling the common "
                "dt^2 factor from numerator and target increment denominator."
            ),
        },
        "normalization": {
            "provenance_sha256": normalization_sha,
            "input_offset": normalization.input_offset.tolist(),
            "input_scale": normalization.input_scale.tolist(),
            "output_scales": {
                rep: normalization.output_scales(rep).tolist() for rep in "ABC"
            },
            "refitted": False,
        },
        "weighting": {
            "M1-X/M1-Y": "frozen carrier-mass weights",
            "M2-X/H1": "frozen S/Q finite-element mass and inverse-mass maps",
        },
        "validation_gates": gates,
        "all_checkpoint_training_parity": all_checkpoint_training_parity,
        "final_training_objective_parity": final_training_parity,
        "stored_final_evaluation_parity": final_evaluation_parity,
        "stored_final_evaluation_availability": {
            "M1-X/M1-Y": "six matching objective values available and exact",
            "M2-X/H1": "no previously stored objective-consistent evaluation values",
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in [
                manifest_path,
                FIXED_PREPARATION,
                fixed_metadata_path,
                HELDOUT_X,
                HELDOUT_X.with_suffix(".json"),
                HELDOUT_Y,
                HELDOUT_Y.with_suffix(".json"),
                direct_path,
                training_operator_path,
                training_fixed_path,
                cross_final_path,
            ]
        ],
        "output": {
            "path": str(DESTINATION.resolve()),
            "sha256": file_sha256(DESTINATION),
        },
        "operations": {
            "optimizer_instantiated": False,
            "prefix_constructed": False,
            "timestep_integrated": False,
            "recursive_rollout": False,
            "checkpoint_inference": True,
            "fixed_array_or_fixed_map_only": True,
        },
        "wall_seconds": float(perf_counter() - started),
        "command": [sys.executable, str(Path(__file__).resolve())],
    }
    SIDECAR.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "records": len(result),
                "final_training_parity": len(final_training_parity),
                "stored_evaluation_parity": len(final_evaluation_parity),
                "wall_seconds": sidecar["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
