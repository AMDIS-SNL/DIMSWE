#!/usr/bin/env python3
"""Read-only Test-2B X-support extraction for ML-results postprocessing.

This script performs no timestep advance, prefix integration, rollout, or
optimization.  It uses the accepted production interpolation and analytical
moist-rate routines on immutable saved truth boundary states, then writes a
new cache only beneath the isolated postprocessing directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from portable_paths import M1Y_REPOSITORY as M1Y_ROOT, PACKAGE_ROOT

CONFIG = M1Y_ROOT / "dimswe/configs/test2b_m1y_20260828.json"
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"
EXPECTED_FEATURE_ORDER = ("h", "S", "Qv", "Qc", "B")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def maximum_difference(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {reference.shape}")
    delta = np.abs(actual - reference)
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return {
        "bitwise_equal": bool(np.array_equal(actual, reference)),
        "maximum_absolute": float(np.max(delta)),
        "maximum_relative": float(np.max(delta / scale)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scope",
        choices=("heldout", "training-carriers"),
        default="heldout",
    )
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite held-out X cache")
    expected_parent = PACKAGE_ROOT / "data"
    if destination.parent != expected_parent.resolve():
        raise ValueError(f"output must be directly beneath {expected_parent}")

    # Import the frozen production implementation from the isolated M1-Y tree.
    sys.path.insert(0, str(M1Y_ROOT))
    from dimswe.test2b_m1y_campaign import (  # noqa: PLC0415
        _load_historical,
        load_m1y_configuration,
    )
    from dimswe.test2b_rain_learning import load_parameters  # noqa: PLC0415
    from dimswe.test2b_rain_learning_campaign import (  # noqa: PLC0415
        build_neural_case,
    )
    from dimswe.test2b_representation_b_postprocess import (  # noqa: PLC0415
        _truth_rate_arrays,
    )
    from dimswe.test2b_representation_c_postprocess import (  # noqa: PLC0415
        _truth_source,
    )

    started = perf_counter()
    configuration_source, campaign = load_m1y_configuration(CONFIG)
    if campaign["reference"]["head"] != EXPECTED_HEAD:
        raise ValueError("source HEAD contract changed")
    historical = _load_historical(configuration_source, campaign)
    normalization = historical["normalization"]
    if tuple(historical["metadata"]["normalization"]["feature_order"]) != EXPECTED_FEATURE_ORDER:
        raise ValueError("feature order changed")

    truth_root = (
        M1Y_ROOT / historical["configuration"]["truth"]["run_directory"]
    ).resolve()
    manifest_path = (
        M1Y_ROOT / campaign["historical"]["truth_manifest"]
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["inventories"]["restart_state_arrays"]
    if len(inventory) != 161:
        raise ValueError("truth restart inventory changed")
    heldout_restart_records = []
    for step in range(81, 161):
        item = inventory[step]
        expected_relative = f"restart/step_{step:08d}.npy"
        if item["path"] != expected_relative:
            raise ValueError(f"truth inventory order changed at state {step}")
        path = truth_root / expected_relative
        actual_sha = file_sha256(path)
        if path.stat().st_size != int(item["bytes"]) or actual_sha != item["sha256"]:
            raise ValueError(f"immutable truth state changed: {path}")
        heldout_restart_records.append({
            "state": step,
            "path": str(path),
            "bytes": int(item["bytes"]),
            "sha256": actual_sha,
        })

    checkpoint = (
        M1Y_ROOT
        / "external-results/test2b-rain-active-learning/production/"
        "representation-A/m1-seed0-m20-10k/final_parameters.npz"
    )
    parameters, checkpoint_record = load_parameters(checkpoint, "A")
    configuration = deepcopy(historical["configuration"])
    configuration["truth"] = dict(configuration["truth"])
    configuration["truth"]["run_directory"] = str(truth_root)
    case, truth, _ = build_neural_case(
        configuration, normalization, "A", parameters, 160
    )
    arrays = _truth_rate_arrays(case, truth, normalization)
    arrays["source"] = _truth_source(arrays["A"], arrays["R"], arrays["h"])

    # Independent extraction must reproduce the frozen X training cache before
    # the held-out portion is trusted.
    frozen = historical["data"]
    training_parity = {
        "x_features": maximum_difference(arrays["features"][:81], frozen["x_features"]),
        "x_A": maximum_difference(arrays["A"][:81], frozen["x_A"]),
        "x_R": maximum_difference(arrays["R"][:81], frozen["x_R"]),
    }
    if any(item["maximum_absolute"] > 2.0e-13 for item in training_parity.values()):
        raise RuntimeError(f"production extraction parity failed: {training_parity}")
    packed_b = np.asarray(arrays["features"][:, :, 4], dtype=np.float64)
    flat_b_audit = {
        "bitwise_zero": bool(np.array_equal(packed_b, 0.0)),
        "minimum": float(np.min(packed_b)),
        "maximum": float(np.max(packed_b)),
        "maximum_absolute": float(np.max(np.abs(packed_b))),
        "nonzero_count": int(np.count_nonzero(packed_b)),
    }
    # The independently interpolated held-out topography can contain only
    # roundoff-level values even though the flat physical field is B=0.  Keep
    # those production-packed values unchanged and fail on any material value.
    if flat_b_audit["maximum_absolute"] > 2.0e-13:
        raise RuntimeError(f"flat-case B channel is materially nonzero: {flat_b_audit}")

    if arguments.scope == "heldout":
        output_arrays = {
            "carrier_weights": np.asarray(frozen["carrier_weights"], dtype=np.float64),
            "heldout_x_features": np.asarray(arrays["features"][81:161], dtype=np.float64),
            "heldout_x_A": np.asarray(arrays["A"][81:161], dtype=np.float64),
            "heldout_x_R": np.asarray(arrays["R"][81:161], dtype=np.float64),
            "heldout_x_h": np.asarray(arrays["h"][81:161], dtype=np.float64),
            "heldout_x_Qr": np.asarray(arrays["Qr"][81:161], dtype=np.float64),
            "heldout_x_source": np.asarray(arrays["source"][81:161], dtype=np.float64),
        }
        expected_shapes = {
            "carrier_weights": (65536,),
            "heldout_x_features": (80, 65536, 5),
            "heldout_x_A": (80, 65536),
            "heldout_x_R": (80, 65536),
            "heldout_x_h": (80, 65536),
            "heldout_x_Qr": (80, 65536),
            "heldout_x_source": (80, 65536, 4),
        }
        state_indices = [81, 160]
        support_classification = "HELD-OUT TRUTH SUPPORT; post-hoc evaluation only"
    else:
        output_arrays = {
            "carrier_weights": np.asarray(frozen["carrier_weights"], dtype=np.float64),
            "training_x_h": np.asarray(arrays["h"][:81], dtype=np.float64),
            "training_x_Qr": np.asarray(arrays["Qr"][:81], dtype=np.float64),
        }
        expected_shapes = {
            "carrier_weights": (65536,),
            "training_x_h": (81, 65536),
            "training_x_Qr": (81, 65536),
        }
        state_indices = [0, 80]
        support_classification = "TRAINING TRUTH SUPPORT; carrier diagnostics only"
    for name, value in output_arrays.items():
        if value.shape != expected_shapes[name] or value.dtype != np.float64:
            raise ValueError(f"invalid {name}: {value.shape} {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite values in {name}")

    temporary = destination.with_name(destination.name + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **output_arrays)
    temporary.replace(destination)

    metadata_path = truth_root / "metadata.json"
    rain_audit_path = truth_root / "rain_activity_audit.json"
    record = {
        "status": "complete",
        "task": (
            "Test-2B held-out X fixed-array extraction"
            if arguments.scope == "heldout"
            else "Test-2B training-X carrier-array extraction"
        ),
        "evaluation_only": True,
        "optimizer_instantiated": False,
        "truth_generated": False,
        "timestep_advanced": False,
        "prefix_applied": False,
        "hybrid_rollout": False,
        "source_head": EXPECTED_HEAD,
        "state_indices": state_indices,
        "state_count": int(state_indices[1] - state_indices[0] + 1),
        "samples_per_state": 65536,
        "support_classification": support_classification,
        "evaluation_state": "X_n* saved timestep-boundary truth state",
        "target_state": "analytical A*/R*/source evaluated at X_n*",
        "feature_order": list(EXPECTED_FEATURE_ORDER),
        "features_are_normalized": True,
        "normalization": historical["metadata"]["normalization"],
        "normalization_refitted": False,
        "flat_B_production_packing_audit": flat_b_audit,
        "carrier_weighting": "frozen production carrier-mass weights",
        "source_order": ["S", "Qv", "Qc", "Qr"],
        "source_formula": "h*[beta2*A, A, -(A+R), R], beta2=98.0616",
        "training_cache_parity": training_parity,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
            }
            for name, value in output_arrays.items()
        },
        "npz_path": str(destination),
        "npz_sha256": file_sha256(destination),
        "truth_root": str(truth_root),
        "truth_manifest": str(manifest_path),
        "truth_manifest_sha256": file_sha256(manifest_path),
        "truth_metadata_sha256": file_sha256(metadata_path),
        "rain_activity_audit_sha256": file_sha256(rain_audit_path),
        "heldout_restart_files": heldout_restart_records,
        "production_extractor": {
            "module": "dimswe.test2b_representation_b_postprocess",
            "function": "_truth_rate_arrays",
            "checkpoint_used_only_to_construct_frozen neural case": str(checkpoint),
            "checkpoint_npz_sha256": file_sha256(checkpoint),
            "checkpoint_parameter_pytree_sha256": checkpoint_record[
                "parameter_pytree_sha256"
            ],
        },
        "configuration": str(configuration_source),
        "configuration_sha256": file_sha256(configuration_source),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "wall_seconds": float(perf_counter() - started),
    }
    sidecar.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": record["status"],
        "npz": str(destination),
        "npz_sha256": record["npz_sha256"],
        "training_cache_parity": training_parity,
        "wall_seconds": record["wall_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
