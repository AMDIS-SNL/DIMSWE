#!/usr/bin/env python3
"""Reconstruct cheap fixed-map M2-X and H1 fitted objectives at checkpoints.

This script deliberately excludes H2/H5: those objectives are recursive and
are outside the cost gate for this pass.  No optimizer or timestep integrator
is instantiated here.
"""

from __future__ import annotations

import argparse
import csv
from gc import collect
import json
from pathlib import Path
import sys
from time import perf_counter

import jax

from portable_paths import (
    AUDIT_ROOT as AUDIT,
    M1Y_REPOSITORY as M1Y,
    PACKAGE_ROOT,
)

OUTPUT = PACKAGE_ROOT / "data"
PREPARATION = M1Y / (
    "external-results/test2b-rain-active-learning/preparation/"
    "fixed_learning_data.npz"
)
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    if destination != (OUTPUT / "checkpoint_training_objectives_fixed.csv").resolve():
        raise ValueError("unexpected output path")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    sys.path.insert(0, str(M1Y))
    from dimswe.test2b_rain_learning import load_parameters  # noqa: PLC0415
    from dimswe.test2b_rain_learning_campaign import (  # noqa: PLC0415
        FixedObjective,
        load_preparation,
    )

    started = perf_counter()
    with (AUDIT / "ML_RUN_INVENTORY.csv").open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    selected = {
        row["run_id"]: row for row in inventory
        if row["physical_case"] == "Test 2B"
        and row["representation"] in ("A", "B", "C")
        and row["run_id"].split("-", 2)[2]
        in ("m2x-independent", "m2x-warm", "h1")
    }
    if len(selected) != 9:
        raise RuntimeError(f"expected 9 fixed-objective runs, found {len(selected)}")
    checkpoint_manifest = read_json(OUTPUT / "checkpoint_hash_manifest.json")
    metadata, normalization, data, matrices = load_preparation(PREPARATION)
    rows = []
    validation = []

    for representation in "ABC":
        for family in ("M2-X", "H1"):
            if family == "M2-X":
                run_ids = (
                    f"t2b-{representation.lower()}-m2x-independent",
                    f"t2b-{representation.lower()}-m2x-warm",
                )
                objective = FixedObjective(
                    representation,
                    data["x_features"], data["x_A"], data["x_R"], matrices,
                    metadata["m2x_denominator"], normalization,
                )
                objective_name = "J_M2_X"
                evaluation_state = "X_n*"
                state_indices = "0..80"
            else:
                run_ids = (f"t2b-{representation.lower()}-h1",)
                objective = FixedObjective(
                    representation,
                    data["y_features"], data["y_A"], data["y_R"], matrices,
                    metadata["common_horizon_denominator"] / 10000.0,
                    normalization,
                )
                objective_name = "J_H1"
                evaluation_state = "Y_n*=P(X_n*)"
                state_indices = "0..79"

            unique = {}
            references = []
            for run_id in run_ids:
                run = checkpoint_manifest["runs"][run_id]
                for checkpoint in run["checkpoints"]:
                    references.append((run_id, checkpoint))
                    unique.setdefault(checkpoint["parameter_sha256"], checkpoint)
            values = {}
            for number, (parameter_sha, checkpoint) in enumerate(unique.items(), 1):
                print(
                    f"{representation} {family} {number}/{len(unique)} "
                    f"i={checkpoint['iteration']}", flush=True,
                )
                parameters, sidecar = load_parameters(
                    Path(checkpoint["path"]), representation
                )
                if sidecar["parameter_pytree_sha256"] != parameter_sha:
                    raise RuntimeError("checkpoint hash changed during fixed-objective pass")
                values[parameter_sha] = objective.value(parameters)
            for run_id, checkpoint in references:
                row = selected[run_id]
                value = values[checkpoint["parameter_sha256"]]
                record = {
                    "run_id": run_id,
                    "physical_case": "Test 2B",
                    "representation": representation,
                    "model_label": (
                        "M2-X-independent" if run_id.endswith("m2x-independent")
                        else "warm M2-X" if run_id.endswith("m2x-warm")
                        else "H1"
                    ),
                    "trained_objective": row["objective"],
                    "checkpoint_iteration": checkpoint["iteration"],
                    "objective": objective_name,
                    "evaluation_state": evaluation_state,
                    "support": "TRAINING TRUTH SUPPORT",
                    "state_indices": state_indices,
                    "value": value,
                    "history_kind": "POST-HOC FIXED-MAP CHECKPOINT EVALUATION",
                    "fitted_objective": True,
                    "checkpoint_parameter_pytree_sha256": checkpoint["parameter_sha256"],
                    "checkpoint_path": checkpoint["path"],
                }
                rows.append(record)
                if int(checkpoint["iteration"]) == int(row["accepted_iterations_this_run"]):
                    accepted = float(row["final_training_objective"])
                    difference = abs(value - accepted)
                    tolerance = max(2.0e-15, 2.0e-11 * abs(accepted))
                    validation.append({
                        "run_id": run_id,
                        "reconstructed": value,
                        "accepted": accepted,
                        "absolute_difference": difference,
                        "tolerance": tolerance,
                        "passed": bool(difference <= tolerance),
                    })
            del objective
            jax.clear_caches()
            collect()

    if len(validation) != 9 or not all(item["passed"] for item in validation):
        raise RuntimeError(f"fixed objective parity failed: {validation}")
    fields = list(rows[0])
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = destination.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "record_count": len(rows),
        "run_count": 9,
        "reconstructed_objectives": ["J_M2_X", "J_H1"],
        "excluded_recursive_histories": ["J_H2", "J_H5"],
        "excluded_reason": "cost gate: recursive checkpoint evaluation deferred",
        "optimizer_instantiated": False,
        "timestep_integrated": False,
        "fixed_array_or_map_only": True,
        "final_endpoint_validation": validation,
        "wall_seconds": float(perf_counter() - started),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "records": len(rows),
        "validated_endpoints": len(validation),
        "wall_seconds": float(perf_counter() - started),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
