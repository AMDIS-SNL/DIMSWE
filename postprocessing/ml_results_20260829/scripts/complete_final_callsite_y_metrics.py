#!/usr/bin/env python3
"""Complete final-model direct physical-law metrics on truth-derived Y*.

Existing training-Y metrics and M1-Y evaluation-Y metrics are copied exactly
from the accepted checkpoint-history artifact.  Only the nine missing held-out
Y* endpoints (H1/H2/H5 for A/B/C) are evaluated.  This is fixed-array network
inference: no prefix, timestep, recursive rollout, optimizer, or solver is
constructed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import jax
import numpy as np

from portable_paths import M1Y_REPOSITORY as M1Y, PACKAGE_ROOT

jax.config.update("jax_enable_x64", True)


ROOT = PACKAGE_ROOT
DATA = ROOT / "data"
DESTINATION = DATA / "final_callsite_y_metrics.csv"
SIDECAR = DESTINATION.with_suffix(".json")
DIRECT = DATA / "checkpoint_direct_histories.csv"
DIRECT_VALIDATION = DATA / "direct_history_validation.json"
MANIFEST = DATA / "checkpoint_hash_manifest.json"
TRAINING_Y = M1Y / (
    "external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz"
)
EVALUATION_Y = M1Y / (
    "external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz"
)
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"
EXPECTED_MODELS = ["M1-Y", "H1", "H2", "H5"]
EXPECTED_METRIC_ROWS = {"A": 13, "B": 51, "C": 85}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_y_cache(npz_path: Path, *, training: bool) -> dict[str, Any]:
    metadata_path = npz_path.with_suffix(".json")
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("source_head") != EXPECTED_HEAD:
        raise RuntimeError(f"invalid Y* metadata: {metadata_path}")
    expected_npz_sha = (
        metadata["preparation_npz_sha256"]
        if training
        else metadata["heldout_npz_sha256"]
    )
    if sha256_file(npz_path) != expected_npz_sha:
        raise RuntimeError(f"Y* cache hash changed: {npz_path}")
    if training:
        if metadata.get("training_state_indices") != [0, 80]:
            raise RuntimeError("training Y* indices changed")
        if metadata.get("feature_state") != "Y_n*=P(X_n*)":
            raise RuntimeError("training feature state is not Y*")
        normalization_sha = metadata["normalization"]["provenance_sha256"]
        keys = {
            "m1y_features": "m1y_features",
            "m1y_A": "m1y_A",
            "m1y_R": "m1y_R",
            "m1y_Qr": "m1y_Qr",
            "carrier_weights": "carrier_weights",
        }
    else:
        if metadata.get("truth_state_indices") != [81, 160]:
            raise RuntimeError("evaluation Y* indices changed")
        if metadata.get("feature_state") != "Y_n*=P(X_n*)":
            raise RuntimeError("evaluation feature state is not Y*")
        if metadata.get("normalization_refitted_on_Y"):
            raise RuntimeError("evaluation normalization was refitted")
        if metadata.get("used_for_training_or_model_selection"):
            raise RuntimeError("evaluation Y* states influenced optimization")
        normalization_sha = metadata["normalization_provenance_sha256"]
        keys = {
            "heldout_y_features": "heldout_y_features",
            "heldout_y_A": "heldout_y_A",
            "heldout_y_R": "heldout_y_R",
            "heldout_y_Qr": "heldout_y_Qr",
        }
    with np.load(npz_path, allow_pickle=False) as archive:
        for key, metadata_key in keys.items():
            value = np.asarray(archive[key])
            expected = metadata["arrays"][metadata_key]
            if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise RuntimeError(f"Y* array shape/dtype changed: {key}")
            if sha256_array(value) != expected["sha256"]:
                raise RuntimeError(f"Y* array hash changed: {key}")
    return {
        "path": str(npz_path.resolve()),
        "sha256": sha256_file(npz_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "normalization_provenance_sha256": normalization_sha,
    }


def main() -> None:
    if DESTINATION.exists() or SIDECAR.exists():
        raise FileExistsError("refusing to overwrite call-site metric outputs")
    started = perf_counter()

    training_cache = validate_y_cache(TRAINING_Y, training=True)
    evaluation_cache = validate_y_cache(EVALUATION_Y, training=False)
    if (
        training_cache["normalization_provenance_sha256"]
        != evaluation_cache["normalization_provenance_sha256"]
    ):
        raise RuntimeError("training/evaluation normalization provenance differs")

    validation = read_json(DIRECT_VALIDATION)
    if validation.get("status") != "passed":
        raise RuntimeError("accepted direct-history validation is incomplete")
    manifest = read_json(MANIFEST)
    if manifest.get("status") != "complete" or manifest.get("source_head") != EXPECTED_HEAD:
        raise RuntimeError("checkpoint manifest is invalid")
    fieldnames, accepted_rows = read_csv(DIRECT)

    run_ids = {
        f"t2b-{representation.lower()}-{suffix}"
        for representation in "ABC"
        for suffix in ["m1y", "h1", "h2", "h5"]
    }
    selected_existing: list[dict[str, Any]] = []
    copied_groups = []
    for run_id in sorted(run_ids):
        final_entry = manifest["runs"][run_id]["final_entry"]
        representation = final_entry["representation"]
        label = final_entry["model_label"]
        for support in ["TRAINING TRUTH SUPPORT", "HELD-OUT TRUTH SUPPORT"]:
            rows = [
                row
                for row in accepted_rows
                if row["run_id"] == run_id
                and row["evaluation_state"] == "Y"
                and row["support"] == support
                and int(row["checkpoint_iteration"]) == int(final_entry["iteration"])
            ]
            should_exist = support == "TRAINING TRUTH SUPPORT" or label == "M1-Y"
            if should_exist:
                if len(rows) != EXPECTED_METRIC_ROWS[representation]:
                    raise RuntimeError(
                        f"accepted Y* metric coverage changed: {run_id} {support} {len(rows)}"
                    )
                selected_existing.extend(rows)
                copied_groups.append(
                    {
                        "run_id": run_id,
                        "support": support,
                        "row_count": len(rows),
                        "source": str(DIRECT.resolve()),
                        "copied_without_recomputation": True,
                    }
                )
            elif rows:
                raise RuntimeError(f"unexpected pre-existing held-out Y* rows: {run_id}")

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(M1Y))
    from reconstruct_direct_histories import (  # noqa: PLC0415
        evaluate_checkpoint,
        flatten_metric_rows,
        load_support,
    )
    from dimswe.test2b_rain_learning import build_model, load_parameters  # noqa: PLC0415

    data = load_support("Y", "heldout")
    if data["state_first"] != 81 or data["state_last"] != 160:
        raise RuntimeError("held-out Y* state alignment changed")
    models = {representation: jax.jit(build_model(representation)) for representation in "ABC"}
    computed_rows: list[dict[str, Any]] = []
    computed_groups = []
    for representation in "ABC":
        for suffix in ["h1", "h2", "h5"]:
            run_id = f"t2b-{representation.lower()}-{suffix}"
            entry = manifest["runs"][run_id]["final_entry"]
            checkpoint = Path(entry["path"])
            if sha256_file(checkpoint) != entry["npz_sha256"]:
                raise RuntimeError(f"checkpoint bytes changed: {run_id}")
            parameters, parameter_sidecar = load_parameters(checkpoint, representation)
            if parameter_sidecar["parameter_pytree_sha256"] != entry["parameter_sha256"]:
                raise RuntimeError(f"checkpoint parameter hash changed: {run_id}")
            print(f"evaluating held-out Y* {run_id}", flush=True)
            metrics = evaluate_checkpoint(
                representation, parameters, models[representation], data
            )
            rows = flatten_metric_rows(entry, "Y", "heldout", data, metrics)
            if len(rows) != EXPECTED_METRIC_ROWS[representation]:
                raise RuntimeError(f"unexpected metric row count: {run_id} {len(rows)}")
            computed_rows.extend(rows)
            computed_groups.append(
                {
                    "run_id": run_id,
                    "support": "HELD-OUT TRUTH SUPPORT",
                    "row_count": len(rows),
                    "checkpoint_iteration": entry["iteration"],
                    "checkpoint_npz_sha256": entry["npz_sha256"],
                    "checkpoint_parameter_pytree_sha256": entry["parameter_sha256"],
                    "computed_by_fixed_array_inference": True,
                }
            )

    combined = selected_existing + computed_rows
    combined.sort(
        key=lambda row: (
            row["representation"],
            EXPECTED_MODELS.index(row["model_label"]),
            0 if row["support"] == "TRAINING TRUTH SUPPORT" else 1,
            row["quantity"],
            row["metric"],
        )
    )
    expected_total = sum(EXPECTED_METRIC_ROWS[rep] * 4 * 2 for rep in "ABC")
    if len(combined) != expected_total:
        raise RuntimeError(f"final Y* metric coverage mismatch: {len(combined)}")
    if any(row["evaluation_state"] != "Y" for row in combined):
        raise RuntimeError("non-Y metric entered call-site output")
    if {row["model_label"] for row in combined} != set(EXPECTED_MODELS):
        raise RuntimeError("main-model coverage changed")

    write_csv(DESTINATION, fieldnames, combined)
    sidecar = {
        "status": "complete",
        "source_head": EXPECTED_HEAD,
        "record_count": len(combined),
        "model_count": 12,
        "models": EXPECTED_MODELS,
        "representations": ["A", "B", "C"],
        "evaluation_state": "truth-derived pre-moist state Y_n*=P(X_n*)",
        "training_states": [0, 80],
        "evaluation_states": [81, 160],
        "feature_order": ["h", "S", "Qv", "Qc", "B"],
        "normalization_provenance_sha256": training_cache[
            "normalization_provenance_sha256"
        ],
        "normalization_refitted": False,
        "carrier_weighting": "frozen Test-2B carrier-mass weights",
        "representation_targets": {
            "A": ["A"],
            "B": ["A", "R"],
            "C": ["S source", "Qv source", "Qc source", "Qr source"],
        },
        "copied_accepted_groups": copied_groups,
        "new_fixed_array_groups": computed_groups,
        "overlap_validation": {
            "accepted_training_Y_groups_reused": 12,
            "accepted_M1_Y_evaluation_groups_reused": 3,
            "new_H1_H2_H5_evaluation_groups": 9,
            "source_direct_history_validation_status": validation["status"],
            "no_accepted_value_recomputed_or_changed": True,
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in [DIRECT, DIRECT_VALIDATION, MANIFEST]
        ]
        + [training_cache, evaluation_cache],
        "output": {
            "path": str(DESTINATION.resolve()),
            "sha256": sha256_file(DESTINATION),
        },
        "operations": {
            "fixed_network_inference": True,
            "new_checkpoint_evaluations": 9,
            "training_or_optimization": False,
            "prefix_constructed": False,
            "timestep_integrated": False,
            "recursive_history": False,
            "autonomous_or_deployed_rollout": False,
            "truth_generated": False,
        },
        "wall_seconds": perf_counter() - started,
        "command": [sys.executable, str(Path(__file__).resolve())],
    }
    SIDECAR.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "records": len(combined),
                "copied_groups": len(copied_groups),
                "new_fixed_array_groups": len(computed_groups),
                "wall_seconds": sidecar["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
