"""Frozen Test-2B M1-Y preparation, validation, and production training.

M1-Y is the historical Test-2B M1 direct-regression objective with one and
only one scientific change: features and analytical targets are evaluated at
the post-prefix truth state ``Y_n*=P(X_n*)``.  The prefix is precomputed with
the analytical model.  This module never differentiates through the prefix
and its operator objective never advances a model state.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_discrete_training import CompactCheckpointObjective
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_pyrol import build_test2a_lbfgs_parameters
from .test2b_rain_learning import (
    FEATURE_ORDER,
    RainMLPConfiguration,
    SOURCE_ORDER,
    build_model,
    canonical_sha256,
    initial_parameters,
    load_parameters,
    save_parameters,
)
from .test2b_rain_learning_campaign import (
    OperatorObjective,
    _analytical_case,
    load_configuration,
    load_preparation,
)
from .test2b_rain_learning_prepare import file_sha256
from .test2b_representation_a_postprocess import _weighted_metrics
from .test2b_representation_b_postprocess import _r_metrics
from .test2b_representation_c_postprocess import _source_diagnostics


CAMPAIGN_ID = "m1y_test2b_20260828"
EXPECTED_STAGE = "M1-Y"
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"
EXPECTED_BRANCH = "dev/dimswe-learned-physics-framework"
TRAINING_STEPS = tuple(range(81))
FEATURES = ("h", "S", "Qv", "Qc", "B")
BETA2 = 98.0616
EXPECTED_NORMALIZATION = {
    "input_offset": np.asarray(
        (749.6487720807651, 7376.434989735685, 1.4193153609575624,
         0.06015957787413514, 0.0),
        dtype=np.float64,
    ),
    "input_scale": np.asarray(
        (16.913638066122523, 133.5602198531373, 0.21326095651874272,
         0.012412402653357142, 1.0),
        dtype=np.float64,
    ),
    "sigma_a": 9.052258655848717e-8,
    "sigma_r": 1.9902871261559996e-11,
    "source_scales": np.asarray(
        (0.006671477765500949, 6.803353979030477e-5,
         6.80335397581467e-5, 1.5076498196845062e-8),
        dtype=np.float64,
    ),
    "provenance_sha256": (
        "794e074b2d3149f58025a7e6a74856374d86adab1e3ee518a64fe6f30ff0dd79"
    ),
}
EXPECTED_SEED_SHA = {
    "A": "6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975",
    "B": "cfadd9f3ee02a78c5b3a946b88c039d9f7ed34e719325ff22c92e1fe4afac056",
    "C": "e52dd73e3f97d44adf4d55354b1c8d9a9b252186a17cae4ad09410270b86df1e",
}
EXPECTED_HISTORICAL_PARAMETER_SHA = {
    "A": "471f3ac8a9b84f68bbe14bdc7dee62e3a025ac5cf61503db6644d5a1fa1bb506",
    "B": "cfc9d3da6a8d07d74ae17e3d9a5beabe434e63b8005e95df4a1925c3a63c609c",
    "C": "8ad9c8017c9304827d8e9e73392c3a9503f7de05a38b86baf52f4351c35615e4",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolved(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _maximum_differences(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"array shapes differ: {actual.shape} != {reference.shape}")
    delta = np.abs(actual - reference)
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return {
        "maximum_absolute": float(np.max(delta)),
        "maximum_relative": float(np.max(delta / scale)),
        "bitwise_equal": bool(np.array_equal(actual, reference)),
        "shape": list(actual.shape),
    }


def _validate_normalization(normalization, configured):
    checks = {
        "input_offset": np.array_equal(
            normalization.input_offset, EXPECTED_NORMALIZATION["input_offset"]
        ),
        "input_scale": np.array_equal(
            normalization.input_scale, EXPECTED_NORMALIZATION["input_scale"]
        ),
        "sigma_A": float(normalization.sigma_a)
        == EXPECTED_NORMALIZATION["sigma_a"],
        "sigma_R": float(normalization.sigma_r_active)
        == EXPECTED_NORMALIZATION["sigma_r"],
        "source_scales": np.array_equal(
            normalization.source_scales,
            EXPECTED_NORMALIZATION["source_scales"],
        ),
        "provenance": normalization.provenance_sha256
        == EXPECTED_NORMALIZATION["provenance_sha256"],
        "configured_offset": np.array_equal(
            np.asarray(configured["input_offset"], dtype=np.float64),
            EXPECTED_NORMALIZATION["input_offset"],
        ),
        "configured_scale": np.array_equal(
            np.asarray(configured["input_scale"], dtype=np.float64),
            EXPECTED_NORMALIZATION["input_scale"],
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"frozen Test2B normalization changed: {checks}")
    return checks


def load_m1y_configuration(path):
    source = Path(path).resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if (
        record.get("benchmark_stage")
        != "Test2B M1-Y direct-regression campaign 20260828"
        or record.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("not the frozen Test2B M1-Y campaign configuration")
    if record["reference"]["head"] != EXPECTED_HEAD:
        raise ValueError("reference HEAD changed")
    if record["reference"]["branch"] != EXPECTED_BRANCH:
        raise ValueError("reference branch changed")
    m1y = record["M1_Y"]
    if (
        m1y["training_states"] != [0, 80]
        or tuple(m1y["features"]) != FEATURES
        or m1y["features_evaluated_at"] != "Y_n*"
        or m1y["targets_evaluated_at"] != "Y_n*"
        or not m1y["offline_precomputed"]
        or m1y["recursive_rollout"]
        or m1y["differentiate_through_prefix"]
    ):
        raise ValueError("M1-Y scientific contract changed")
    training = record["training"]
    optimizer = training["optimizer"]
    if (
        training["representations"] != ["A", "B", "C"]
        or training["activation"] != "tanh"
        or training["output_activation"] != "linear"
        or training["dtype"] != "float64"
        or optimizer != {
            "library": "PyROL/ROL",
            "method": "line-search L-BFGS",
            "maximum_secant_storage": 20,
            "gradient_tolerance": 1.0e-8,
            "step_tolerance": 1.0e-12,
            "accepted_iteration_limit": 10000,
            "production_HVP": False,
        }
    ):
        raise ValueError("M1-Y optimizer/training contract changed")
    for representation in "ABC":
        architecture = RainMLPConfiguration(representation)
        configured = training["architectures"][representation]
        if (
            configured["layers"] != list(architecture.layer_dimensions)
            or configured["parameter_count"] != architecture.parameter_count
            or configured["seed0_parameter_sha256"]
            != EXPECTED_SEED_SHA[representation]
            or parameter_pytree_sha256(initial_parameters(representation))
            != EXPECTED_SEED_SHA[representation]
        ):
            raise ValueError(f"Representation {representation} contract changed")
    return source, record


def _load_historical(configuration_source, campaign):
    root = repository_root()
    historical_configuration_path = _resolved(
        root, campaign["historical"]["configuration"]
    )
    preparation_path = _resolved(root, campaign["historical"]["preparation"])
    if file_sha256(preparation_path) != campaign["historical"]["preparation_sha256"]:
        raise ValueError("historical Test2B preparation hash changed")
    if file_sha256(preparation_path.with_suffix(".json")) != campaign["historical"]["preparation_sidecar_sha256"]:
        raise ValueError("historical Test2B preparation sidecar hash changed")
    historical_configuration = load_configuration(historical_configuration_path)
    metadata, normalization, data, matrices = load_preparation(preparation_path)
    _validate_normalization(normalization, campaign["normalization"])
    if tuple(historical_configuration["model"]["features"]) != FEATURES:
        raise ValueError("historical feature packing changed")
    return {
        "campaign_configuration_path": configuration_source,
        "historical_configuration_path": historical_configuration_path,
        "historical_preparation_path": preparation_path,
        "configuration": historical_configuration,
        "metadata": metadata,
        "normalization": normalization,
        "data": data,
        "matrices": matrices,
    }


def verify_immutable_truth(configuration_path, output_path):
    configuration_source, campaign = load_m1y_configuration(configuration_path)
    historical = _load_historical(configuration_source, campaign)
    root = repository_root()
    manifest_path = _resolved(root, campaign["historical"]["truth_manifest"])
    if file_sha256(manifest_path) != campaign["historical"]["truth_manifest_sha256"]:
        raise ValueError("frozen truth manifest hash changed")
    manifest = read_json_record(manifest_path)
    truth_root = _resolved(root, historical["configuration"]["truth"]["run_directory"])
    metadata_path = truth_root / "metadata.json"
    rain_audit_path = truth_root / "rain_activity_audit.json"
    if file_sha256(metadata_path) != campaign["historical"]["truth_metadata_sha256"]:
        raise ValueError("truth metadata hash changed")
    if file_sha256(rain_audit_path) != campaign["historical"]["rain_audit_sha256"]:
        raise ValueError("truth rain audit hash changed")
    inventory = manifest["inventories"]["restart_state_arrays"]
    if len(inventory) != 161:
        raise ValueError("frozen truth restart inventory must contain states 0..160")
    verified = []
    for index, item in enumerate(inventory):
        expected = f"restart/step_{index:08d}.npy"
        if item["path"] != expected:
            raise ValueError(f"truth restart order changed at index {index}")
        path = truth_root / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise FileNotFoundError(f"missing/incomplete immutable truth input {path}")
        actual = file_sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"truth restart hash changed: {path}")
        verified.append({"step": index, "path": str(path), **item})
    payload = {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "source_head": EXPECTED_HEAD,
        "truth_root": str(truth_root),
        "truth_manifest": str(manifest_path),
        "truth_manifest_sha256": file_sha256(manifest_path),
        "truth_metadata_sha256": file_sha256(metadata_path),
        "rain_activity_audit_sha256": file_sha256(rain_audit_path),
        "restart_count": len(verified),
        "restart_total_bytes": int(sum(item["bytes"] for item in inventory)),
        "restart_inventory_sha256": canonical_sha256(inventory),
        "restart_files": verified,
        "historical_preparation": str(historical["historical_preparation_path"]),
        "historical_preparation_sha256": file_sha256(historical["historical_preparation_path"]),
        "historical_preparation_sidecar_sha256": file_sha256(historical["historical_preparation_path"].with_suffix(".json")),
        "configuration": str(configuration_source),
        "configuration_sha256": file_sha256(configuration_source),
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite immutable-input manifest")
    write_json_record(destination, payload)
    return payload


def independent_numpy_rates(physical_features, moist_parameters):
    """Independent NumPy transcription of the deployed local A/R law."""
    features = np.asarray(physical_features, dtype=np.float64)
    if features.shape[-1] != 5:
        raise ValueError("physical feature array must end in (h,S,Qv,Qc,B)")
    h, entropy, qv_total, qc_total, topography = np.moveaxis(features, -1, 0)
    scalar = lambda name: float(np.asarray(moist_parameters[name]))
    gravity = scalar("g")
    q0 = scalar("q0")
    reference_depth = scalar("H0")
    gamma_r = scalar("gamma_r")
    qprecip = scalar("qprecip")
    latent_ratio = scalar("L")
    configured_dt = scalar("configured_dt")
    qv = qv_total / h
    qc = qc_total / h
    b = entropy / h
    qsat = (
        q0 * reference_depth / (h + topography)
        * np.exp(20.0 * (1.0 - b / gravity))
    )
    beta2 = gravity * latent_ratio
    gamma_v = 1.0 / (1.0 + 20.0 * qsat * beta2 / gravity)
    condensation = np.maximum(0.0, gamma_v * (qv - qsat) / configured_dt)
    evaporation_positive = np.maximum(
        0.0, gamma_v * (qsat - qv) / configured_dt
    )
    evaporation = np.minimum(qc / configured_dt, evaporation_positive)
    rain = np.maximum(0.0, gamma_r * (qc - qprecip) / configured_dt)
    return {
        "A": evaporation - condensation,
        "R": rain,
        "qv": qv,
        "qc": qc,
        "b": b,
        "qsat": qsat,
        "gamma_v": gamma_v,
        "condensation": condensation,
        "evaporation": evaporation,
    }


def representation_target(representation, normalized_features, target_a, target_r, normalization):
    features = np.asarray(normalized_features, dtype=np.float64)
    a = np.asarray(target_a, dtype=np.float64)
    r = np.asarray(target_r, dtype=np.float64)
    h = features[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
    if representation == "A":
        return a[..., None]
    if representation == "B":
        return np.stack((a, r), axis=-1)
    if representation == "C":
        return np.stack((h * BETA2 * a, h * a, -h * (a + r), h * r), axis=-1)
    raise ValueError("M1-Y supports only A, B, and C")


def _pack_evaluation(result):
    return np.stack(
        (
            np.asarray(result.packed_state["h"]).reshape(-1),
            np.asarray(result.packed_state["S"]).reshape(-1),
            np.asarray(result.packed_state["Qv"]).reshape(-1),
            np.asarray(result.packed_state["Qc"]).reshape(-1),
            np.asarray(result.packed_fields["B"]).reshape(-1),
        ),
        axis=-1,
    )


def _postprefix(case, state, step):
    replay = case.helper.take_forward_step_cached(
        state, float(step) * case.dt, case.dt
    )
    if len(replay.boundary_states) != 7:
        raise RuntimeError("accepted six-child timestep boundary count changed")
    return replay.boundary_states[-2]


def prepare_m1y(configuration_path, immutable_manifest_path, output_path):
    configuration_source, campaign = load_m1y_configuration(configuration_path)
    historical = _load_historical(configuration_source, campaign)
    immutable = read_json_record(immutable_manifest_path)
    if (
        immutable.get("status") != "complete"
        or immutable.get("campaign_id") != CAMPAIGN_ID
        or immutable.get("restart_count") != 161
        or immutable.get("truth_manifest_sha256")
        != campaign["historical"]["truth_manifest_sha256"]
    ):
        raise ValueError("immutable-input gate has not passed")
    destination = Path(output_path)
    sidecar = destination.with_suffix(".json")
    progress_path = destination.parent / "preparation_progress.json"
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite M1-Y preparation")
    destination.parent.mkdir(parents=True, exist_ok=True)

    normalization = historical["normalization"]
    frozen = historical["data"]
    case, truth, adapter = _analytical_case(historical["configuration"])
    features, target_a, target_r, qr_values = [], [], [], []
    started = perf_counter()
    for step in TRAINING_STEPS:
        y_state = _postprefix(case, truth[step], step)
        evaluated = adapter.evaluate(y_state, case.dt)
        physical = _pack_evaluation(evaluated)
        _, qr = adapter.interpolate_and_pack(
            y_state.sub(5), f"m1y_test2b_Qr_{step}"
        )
        features.append(np.asarray(normalization.normalize_features(physical)))
        target_a.append(np.asarray(evaluated.rates["A"]).reshape(-1))
        target_r.append(np.asarray(evaluated.rates["R"]).reshape(-1))
        qr_values.append(np.asarray(qr).reshape(-1))
        write_json_record(progress_path, {
            "status": "in_progress",
            "campaign_id": CAMPAIGN_ID,
            "completed_state": int(step),
            "completed_state_count": int(step + 1),
            "required_state_count": 81,
            "elapsed_wall_seconds": float(perf_counter() - started),
            "state_definition": "Y_n*=boundary_states[-2] after accepted full analytical timestep replay",
        })

    features = np.stack(features).astype(np.float64, copy=False)
    target_a = np.stack(target_a).astype(np.float64, copy=False)
    target_r = np.stack(target_r).astype(np.float64, copy=False)
    qr_values = np.stack(qr_values).astype(np.float64, copy=False)
    weights = np.asarray(frozen["carrier_weights"], dtype=np.float64)
    if features.shape != (81, 65536, 5):
        raise ValueError(f"M1-Y feature shape changed: {features.shape}")
    if target_a.shape != (81, 65536) or target_r.shape != target_a.shape:
        raise ValueError("M1-Y target shape changed")
    if qr_values.shape != target_a.shape or weights.shape != (65536,):
        raise ValueError("M1-Y diagnostic/support shape changed")
    if not all(np.all(np.isfinite(value)) for value in (features, target_a, target_r, qr_values, weights)):
        raise FloatingPointError("M1-Y preparation contains nonfinite values")
    if not np.array_equal(features[..., 4], np.zeros_like(features[..., 4])):
        raise ValueError("flat DoubleVortex B feature changed from exact zero")

    parity = {
        "features_states_0_79": _maximum_differences(features[:80], frozen["y_features"]),
        "A_states_0_79": _maximum_differences(target_a[:80], frozen["y_A"]),
        "R_states_0_79": _maximum_differences(target_r[:80], frozen["y_R"]),
    }
    if parity["features_states_0_79"]["maximum_absolute"] > 1.0e-12:
        raise RuntimeError(f"regenerated post-prefix feature cache lacks parity: {parity}")
    if parity["A_states_0_79"]["maximum_absolute"] > 1.0e-18:
        raise RuntimeError(f"regenerated post-prefix A cache lacks parity: {parity}")
    if parity["R_states_0_79"]["maximum_absolute"] > 1.0e-22:
        raise RuntimeError(f"regenerated post-prefix R cache lacks parity: {parity}")
    if not np.array_equal(weights, frozen["carrier_weights"]):
        raise RuntimeError("historical carrier weights changed")

    broadcast_weights = np.broadcast_to(weights, target_a.shape)
    denominators = {
        representation: OperatorObjective(
            representation, features, target_a, target_r,
            broadcast_weights, normalization,
        ).denominator
        for representation in "ABC"
    }
    arrays = {
        "carrier_weights": weights,
        "m1y_features": features,
        "m1y_A": target_a,
        "m1y_R": target_r,
        "m1y_Qr": qr_values,
    }
    incomplete = destination.with_name(destination.name + ".incomplete")
    with incomplete.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    incomplete.replace(destination)
    prefix_source = inspect.getsource(_postprefix)
    metadata = {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "source_head": EXPECTED_HEAD,
        "training_state_indices": [0, 80],
        "training_state_count": 81,
        "samples_per_state": 65536,
        "total_sample_count": int(target_a.size),
        "historical_M1_X_support": [0, 80],
        "historical_H1_postprefix_support": [0, 79],
        "final_state_80_added_for_M1_Y": True,
        "features": list(FEATURES),
        "feature_order": list(FEATURE_ORDER),
        "Qr_is_network_input": False,
        "B_exactly_zero": True,
        "feature_state": "Y_n*=P(X_n*)",
        "target_state": "Y_n*=P(X_n*)",
        "prefix": {
            "implementation": "case.helper.take_forward_step_cached(X_n*, n*dt, dt).boundary_states[-2]",
            "complete_child_order": [
                "dry_rk4_0", "dry_rk4_1", "hyperviscosity_euler",
                "dg_ssprk43_0", "dg_ssprk43_1", "moist_euler",
            ],
            "selected_boundary_index": -2,
            "offline_precomputed": True,
            "theta_argument_present": False,
            "recursive_rollout": False,
            "differentiate_through_prefix": False,
            "local_wrapper_source_sha256": sha256(prefix_source.encode("utf-8")).hexdigest(),
        },
        "normalization": normalization.to_record(),
        "normalization_refitted_on_Y": False,
        "output_scales_refitted_on_Y": False,
        "carrier_weights_reused_bitwise": True,
        "historical_H1_cache_parity": parity,
        "operator_denominators": denominators,
        "architecture": {
            representation: RainMLPConfiguration(representation).to_record()
            for representation in "ABC"
        },
        "seed0_parameter_sha256": dict(EXPECTED_SEED_SHA),
        "arrays": {
            name: {
                "shape": list(np.shape(value)),
                "dtype": str(np.asarray(value).dtype),
                "sha256": _array_sha256(value),
            }
            for name, value in arrays.items()
        },
        "historical_configuration": str(historical["historical_configuration_path"]),
        "historical_configuration_sha256": file_sha256(historical["historical_configuration_path"]),
        "historical_preparation": str(historical["historical_preparation_path"]),
        "historical_preparation_sha256": file_sha256(historical["historical_preparation_path"]),
        "immutable_input_manifest": str(Path(immutable_manifest_path).resolve()),
        "immutable_input_manifest_sha256": file_sha256(immutable_manifest_path),
        "campaign_configuration": str(configuration_source),
        "campaign_configuration_sha256": file_sha256(configuration_source),
        "preparation_npz_sha256": file_sha256(destination),
        "wall_seconds": float(perf_counter() - started),
        "command": [sys.executable, "-m", "dimswe.test2b_m1y_campaign", *sys.argv[1:]],
    }
    write_json_record(sidecar, metadata)
    write_json_record(progress_path, {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "completed_state": 80,
        "completed_state_count": 81,
        "required_state_count": 81,
        "elapsed_wall_seconds": metadata["wall_seconds"],
        "preparation_npz": str(destination.resolve()),
        "preparation_npz_sha256": metadata["preparation_npz_sha256"],
    })
    return metadata


def load_m1y_preparation(path):
    source = Path(path).resolve()
    metadata = read_json_record(source.with_suffix(".json"))
    if (
        metadata.get("status") != "complete"
        or metadata.get("campaign_id") != CAMPAIGN_ID
        or metadata.get("training_state_indices") != [0, 80]
        or metadata.get("feature_state") != "Y_n*=P(X_n*)"
        or metadata.get("target_state") != "Y_n*=P(X_n*)"
        or metadata.get("normalization_refitted_on_Y")
        or metadata.get("output_scales_refitted_on_Y")
    ):
        raise ValueError("invalid M1-Y preparation metadata")
    if file_sha256(source) != metadata["preparation_npz_sha256"]:
        raise ValueError("M1-Y preparation hash mismatch")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in (
                "carrier_weights", "m1y_features", "m1y_A", "m1y_R",
                "m1y_Qr",
            )
        }
    for name, value in arrays.items():
        expected = metadata["arrays"][name]
        if (
            list(value.shape) != expected["shape"]
            or str(value.dtype) != expected["dtype"]
            or _array_sha256(value) != expected["sha256"]
        ):
            raise ValueError(f"M1-Y prepared array {name} changed")
    return metadata, arrays


def _select_validation_samples(arrays):
    a = arrays["m1y_A"]
    r = arrays["m1y_R"]
    def location(flat):
        step, sample = np.unravel_index(int(flat), a.shape)
        return {"step": int(step), "sample": int(sample)}
    inactive = r == 0.0
    if not np.any(a < 0.0) or not np.any(a > 0.0) or not np.any(r > 0.0):
        raise RuntimeError("M1-Y support lacks required moist regimes")
    masked_abs = np.where(inactive, np.abs(a), np.inf)
    return {
        "condensation_active": location(np.argmin(a)),
        "evaporation_active": location(np.argmax(a)),
        "rain_active": location(np.argmax(r)),
        "near_inactive": location(np.argmin(masked_abs)),
    }


def validate_m1y(configuration_path, preparation_path, output_path):
    configuration_source, campaign = load_m1y_configuration(configuration_path)
    historical = _load_historical(configuration_source, campaign)
    metadata, arrays = load_m1y_preparation(preparation_path)
    normalization = historical["normalization"]
    case, truth, adapter = _analytical_case(historical["configuration"])
    moist_parameters = adapter._parameters(None)
    selected = _select_validation_samples(arrays)
    records = {}
    for label, selection in selected.items():
        step = selection["step"]
        sample = selection["sample"]
        x_evaluation = adapter.evaluate(truth[step], case.dt)
        x_physical = _pack_evaluation(x_evaluation)
        x_normalized = np.asarray(normalization.normalize_features(x_physical))
        y_state = _postprefix(case, truth[step], step)
        y_evaluation = adapter.evaluate(y_state, case.dt)
        y_physical = _pack_evaluation(y_evaluation)
        y_normalized = np.asarray(normalization.normalize_features(y_physical))
        independent = independent_numpy_rates(y_physical, moist_parameters)
        expected_c = representation_target(
            "C", y_normalized, independent["A"], independent["R"],
            normalization,
        )
        actual_c = representation_target(
            "C", arrays["m1y_features"][step], arrays["m1y_A"][step],
            arrays["m1y_R"][step], normalization,
        )
        records[label] = {
            **selection,
            "X_feature_actual": historical["data"]["x_features"][step, sample].tolist(),
            "X_feature_independent_from_X_state": x_normalized[sample].tolist(),
            "Y_feature_actual": arrays["m1y_features"][step, sample].tolist(),
            "Y_feature_independent_from_prefix_state": y_normalized[sample].tolist(),
            "X_feature_maximum_absolute_discrepancy": float(np.max(np.abs(
                historical["data"]["x_features"][step, sample]
                - x_normalized[sample]
            ))),
            "Y_feature_maximum_absolute_discrepancy": float(np.max(np.abs(
                arrays["m1y_features"][step, sample] - y_normalized[sample]
            ))),
            "X_vs_Y_feature_maximum_absolute_difference": float(np.max(np.abs(
                x_normalized[sample] - y_normalized[sample]
            ))),
            "M1Y_A_actual": float(arrays["m1y_A"][step, sample]),
            "M1Y_A_independent_numpy": float(independent["A"][sample]),
            "M1Y_A_absolute_discrepancy": float(abs(
                arrays["m1y_A"][step, sample] - independent["A"][sample]
            )),
            "M1Y_R_actual": float(arrays["m1y_R"][step, sample]),
            "M1Y_R_independent_numpy": float(independent["R"][sample]),
            "M1Y_R_absolute_discrepancy": float(abs(
                arrays["m1y_R"][step, sample] - independent["R"][sample]
            )),
            "C_target_actual": actual_c[sample].tolist(),
            "C_target_independent": expected_c[sample].tolist(),
            "C_target_maximum_absolute_discrepancy": float(np.max(np.abs(
                actual_c[sample] - expected_c[sample]
            ))),
            "A_sign": "positive_evaporation" if independent["A"][sample] > 0.0 else (
                "negative_condensation" if independent["A"][sample] < 0.0 else "inactive"
            ),
            "rain_active": bool(independent["R"][sample] > 0.0),
            "physical_feature_order_h_S_Qv_Qc_B": y_physical[sample].tolist(),
            "specific_qv": float(independent["qv"][sample]),
            "specific_qc": float(independent["qc"][sample]),
            "specific_b": float(independent["b"][sample]),
            "q_sat": float(independent["qsat"][sample]),
        }
    feature_error = max(
        max(row["X_feature_maximum_absolute_discrepancy"], row["Y_feature_maximum_absolute_discrepancy"])
        for row in records.values()
    )
    a_error = max(row["M1Y_A_absolute_discrepancy"] for row in records.values())
    r_error = max(row["M1Y_R_absolute_discrepancy"] for row in records.values())
    c_error = max(row["C_target_maximum_absolute_discrepancy"] for row in records.values())
    source_method = inspect.getsource(type(case.helper).take_forward_step_cached)
    gates = {
        "support_is_exactly_0_through_80": metadata["training_state_indices"] == [0, 80]
        and metadata["training_state_count"] == 81,
        "historical_carrier_weights_bitwise_reused": metadata["carrier_weights_reused_bitwise"],
        "feature_order_is_h_S_Qv_Qc_B": tuple(metadata["feature_order"]) == FEATURES,
        "Qr_absent_from_network_input": not metadata["Qr_is_network_input"],
        "normalization_is_historical_X_fitted": (
            not metadata["normalization_refitted_on_Y"]
            and metadata["normalization"]["fitted_truth_state_indices"] == [0, 80]
            and metadata["normalization"]["provenance_sha256"]
            == EXPECTED_NORMALIZATION["provenance_sha256"]
        ),
        "H1_prefix_parity_passed": all(
            row["maximum_absolute"] <= tolerance
            for row, tolerance in zip(
                metadata["historical_H1_cache_parity"].values(),
                (1.0e-12, 1.0e-18, 1.0e-22),
            )
        ),
        "features_reproduced_independently": feature_error <= 1.0e-12,
        "A_targets_reproduced_independently": a_error <= 1.0e-18,
        "R_targets_reproduced_independently": r_error <= 1.0e-22,
        "C_source_order_reproduced_independently": c_error <= 1.0e-15,
        "offline_theta_independent_cache": (
            metadata["prefix"]["offline_precomputed"]
            and not metadata["prefix"]["theta_argument_present"]
            and not metadata["prefix"]["recursive_rollout"]
            and not metadata["prefix"]["differentiate_through_prefix"]
        ),
        "all_prepared_values_finite": all(
            np.all(np.isfinite(value)) for value in arrays.values()
        ),
        "seed0_initializations_match": all(
            parameter_pytree_sha256(initial_parameters(r)) == EXPECTED_SEED_SHA[r]
            for r in "ABC"
        ),
    }
    result = {
        "status": "passed" if all(gates.values()) else "failed",
        "campaign_id": CAMPAIGN_ID,
        "pretraining_gate": True,
        "gates": gates,
        "maximum_discrepancies": {
            "feature": feature_error,
            "A": a_error,
            "R": r_error,
            "C_source": c_error,
        },
        "spot_checks": records,
        "offline_semantics": {
            "Y_construction": "case.helper.take_forward_step_cached(X_n*, n*dt, dt).boundary_states[-2]",
            "prefix_source_path": inspect.getsourcefile(type(case.helper).take_forward_step_cached),
            "prefix_source_sha256": sha256(source_method.encode("utf-8")).hexdigest(),
            "features_and_targets_materialized_before_optimizer": True,
            "objective_inputs": ["m1y_features", "m1y_A", "m1y_R", "carrier_weights"],
            "theta_in_preparation": False,
            "recursive_rollout": False,
            "differentiation_through_P": False,
        },
        "configuration": str(configuration_source),
        "configuration_sha256": file_sha256(configuration_source),
        "preparation": str(Path(preparation_path).resolve()),
        "preparation_sha256": file_sha256(preparation_path),
        "historical_preparation": str(historical["historical_preparation_path"]),
        "historical_preparation_sha256": file_sha256(historical["historical_preparation_path"]),
        "command": [sys.executable, "-m", "dimswe.test2b_m1y_campaign", *sys.argv[1:]],
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite M1-Y validation")
    write_json_record(destination, result)
    if result["status"] != "passed":
        raise RuntimeError(f"M1-Y pretraining validation failed: {gates}")
    return result


def m1y_objective(configuration_path, preparation_path, representation):
    configuration_source, campaign = load_m1y_configuration(configuration_path)
    historical = _load_historical(configuration_source, campaign)
    metadata, arrays = load_m1y_preparation(preparation_path)
    if representation not in "ABC":
        raise ValueError("M1-Y trains only A, B, or C")
    weights = np.broadcast_to(arrays["carrier_weights"], arrays["m1y_A"].shape)
    objective = OperatorObjective(
        representation, arrays["m1y_features"], arrays["m1y_A"],
        arrays["m1y_R"], weights, historical["normalization"],
    )
    expected = metadata["operator_denominators"][representation]
    if objective.denominator != expected:
        raise ValueError("M1-Y operator denominator changed")
    return campaign, historical, metadata, arrays, objective


def certify_objectives(configuration_path, preparation_path, validation_path, output_path):
    validation = read_json_record(validation_path)
    if validation.get("status") != "passed" or not all(validation["gates"].values()):
        raise RuntimeError("state/target validation must pass before objective certification")
    if validation["preparation_sha256"] != file_sha256(preparation_path):
        raise ValueError("validated M1-Y preparation changed")
    records = {}
    for representation in "ABC":
        _, _, metadata, _, objective = m1y_objective(
            configuration_path, preparation_path, representation
        )
        parameters = initial_parameters(representation)
        value, gradient = objective.value_and_gradient(parameters)
        flat, unravel = ravel_pytree(parameters)
        flat_gradient, _ = ravel_pytree(gradient)
        direction = np.linspace(-0.7, 0.9, flat.size, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        direction = jnp.asarray(direction, dtype=jnp.float64)
        epsilon = 1.0e-6
        plus = unravel(flat + epsilon * direction)
        minus = unravel(flat - epsilon * direction)
        centered = (objective.value(plus) - objective.value(minus)) / (2.0 * epsilon)
        adjoint = float(jnp.vdot(flat_gradient, direction))
        discrepancy = abs(centered - adjoint)
        relative = discrepancy / max(
            abs(centered), abs(adjoint), np.finfo(np.float64).tiny
        )
        record = {
            "architecture": RainMLPConfiguration(representation).to_record(),
            "seed0_parameter_pytree_sha256": parameter_pytree_sha256(parameters),
            "objective": float(value),
            "objective_finite": bool(np.isfinite(value)),
            "gradient_norm": float(jnp.linalg.norm(flat_gradient)),
            "gradient_all_finite": bool(np.all(np.isfinite(np.asarray(flat_gradient)))),
            "directional_derivative": {
                "epsilon": epsilon,
                "adjoint": adjoint,
                "centered_finite_difference": centered,
                "absolute_discrepancy": discrepancy,
                "relative_discrepancy": relative,
            },
            "operator_denominator": float(objective.denominator),
            "prepared_operator_denominator": float(
                metadata["operator_denominators"][representation]
            ),
        }
        record["passed"] = bool(
            record["objective_finite"]
            and record["gradient_all_finite"]
            and record["seed0_parameter_pytree_sha256"]
            == EXPECTED_SEED_SHA[representation]
            and record["operator_denominator"]
            == record["prepared_operator_denominator"]
            and relative <= 1.0e-6
        )
        records[representation] = record
    result = {
        "status": "passed" if all(row["passed"] for row in records.values()) else "failed",
        "campaign_id": CAMPAIGN_ID,
        "objective": "M1-Y deterministic full-batch normalized direct regression",
        "representations": records,
        "preparation": str(Path(preparation_path).resolve()),
        "preparation_sha256": file_sha256(preparation_path),
        "validation": str(Path(validation_path).resolve()),
        "validation_sha256": file_sha256(validation_path),
        "command": [sys.executable, "-m", "dimswe.test2b_m1y_campaign", *sys.argv[1:]],
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite M1-Y objective certification")
    write_json_record(destination, result)
    if result["status"] != "passed":
        raise RuntimeError(f"M1-Y objective certification failed: {records}")
    return result


def _training_support_metrics(representation, parameters, arrays, normalization, comparison_scale):
    features = arrays["m1y_features"]
    weights = np.broadcast_to(arrays["carrier_weights"], arrays["m1y_A"].shape)
    raw = np.asarray(build_model(representation)(parameters, jnp.asarray(features)))
    physical = raw * normalization.output_scales(representation)
    if representation == "A":
        return {
            "A": _weighted_metrics(
                physical[..., 0], arrays["m1y_A"], weights,
                normalization.sigma_a,
            )
        }
    if representation == "B":
        h = features[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
        return {
            "A": _weighted_metrics(
                physical[..., 0], arrays["m1y_A"], weights,
                normalization.sigma_a,
            ),
            "R": _r_metrics(
                physical[..., 1], arrays["m1y_R"], weights, h,
                arrays["m1y_Qr"], comparison_scale,
            ),
        }
    target = representation_target(
        "C", features, arrays["m1y_A"], arrays["m1y_R"], normalization
    )
    h = features[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
    return _source_diagnostics(
        physical, target, arrays["m1y_A"], arrays["m1y_R"], h,
        arrays["m1y_Qr"], arrays["carrier_weights"], normalization,
        comparison_scale,
    )


def train_m1y(configuration_path, preparation_path, validation_path, representation, output_directory):
    from pyrol import Problem, Solver

    campaign, historical, preparation_metadata, arrays, objective = m1y_objective(
        configuration_path, preparation_path, representation
    )
    validation = read_json_record(validation_path)
    if validation.get("status") != "passed" or not all(validation["gates"].values()):
        raise RuntimeError("M1-Y pretraining validation gate has not passed")
    if validation["preparation_sha256"] != file_sha256(preparation_path):
        raise ValueError("validated M1-Y preparation changed")
    iteration_limit = int(campaign["training"]["optimizer"]["accepted_iteration_limit"])
    if iteration_limit != 10000:
        raise ValueError("production M1-Y iteration cap must remain 10000")
    parameters = initial_parameters(representation)
    seed_sha = parameter_pytree_sha256(parameters)
    if seed_sha != EXPECTED_SEED_SHA[representation]:
        raise ValueError("M1-Y seed-0 initialization changed")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("refusing to overwrite M1-Y training stage")
    output.mkdir(parents=True)
    checkpoint_schedule = (0, 1, 5, 10, 20, 100, 500, 1000, 5000, 10000)
    configuration_source = Path(configuration_path).resolve()
    preparation_source = Path(preparation_path).resolve()

    def parameter_metadata(iteration):
        return {
            "campaign_id": CAMPAIGN_ID,
            "stage": EXPECTED_STAGE,
            "evaluation_state": "Y_n*=P(X_n*)",
            "accepted_iteration": int(iteration),
            "training_states": [0, 80],
            "feature_order": list(FEATURES),
            "normalization_provenance_sha256": EXPECTED_NORMALIZATION["provenance_sha256"],
            "preparation_npz_sha256": file_sha256(preparation_source),
            "campaign_configuration_sha256": file_sha256(configuration_source),
        }

    initial_record = save_parameters(
        output / "checkpoint_000000000_parameters.npz", representation,
        parameters, metadata=parameter_metadata(0),
    )
    started = perf_counter()

    def accepted_callback(control, local_index, adapter):
        if local_index == 0 or (
            local_index not in checkpoint_schedule and local_index % 100 != 0
        ):
            return
        current = adapter.pytree_from_vector(control)
        current_objective = float(objective.value(current))
        if not np.isfinite(current_objective):
            raise FloatingPointError("nonfinite M1-Y objective at accepted iterate")
        if local_index in checkpoint_schedule:
            record = save_parameters(
                output / f"checkpoint_{local_index:09d}_parameters.npz",
                representation, current,
                metadata=parameter_metadata(local_index),
            ) if local_index != 0 else initial_record
            parameter_sha = record["parameter_pytree_sha256"]
        else:
            parameter_sha = parameter_pytree_sha256(current)
        progress = {
            "status": "in_progress",
            "campaign_id": CAMPAIGN_ID,
            "representation": representation,
            "stage": EXPECTED_STAGE,
            "evaluation_state": "Y_n*=P(X_n*)",
            "accepted_iteration": int(local_index),
            "objective": current_objective,
            "elapsed_wall_seconds": float(perf_counter() - started),
            "objective_evaluations": int(adapter.value_evaluations),
            "gradient_evaluations": int(adapter.gradient_evaluations),
            "parameter_pytree_sha256": parameter_sha,
            "source_secant_history_reused": False,
            "parameter_only_restart_restores_secant_history": False,
            "normalization_refitted_on_Y": False,
            "recursive_rollout": False,
            "differentiate_through_prefix": False,
        }
        write_json_record(output / "fit_progress.json", progress)

    adapter = CompactCheckpointObjective(
        objective.jax_value, parameters, use_jit=True,
        accepted_callback=accepted_callback,
    )
    control = adapter.vector_from_pytree(parameters)
    solver_parameters = build_test2a_lbfgs_parameters({
        "gradient_tolerance": 1.0e-8,
        "step_tolerance": 1.0e-12,
        "iteration_limit": iteration_limit,
        "maximum_secant_storage": 20,
    })
    solver = Solver(Problem(adapter, control), solver_parameters)
    solver.solve()
    final = adapter.pytree_from_vector(control)
    algorithm_state = solver.getAlgorithmState()
    final_objective = float(objective.value(final))
    if not np.isfinite(final_objective):
        raise FloatingPointError("nonfinite final M1-Y objective")
    if not all(np.isfinite(adapter.value_history)):
        raise FloatingPointError("nonfinite objective occurred during M1-Y line search")
    if not all(np.isfinite(adapter.gradient_norm_history)):
        raise FloatingPointError("nonfinite gradient norm occurred during M1-Y training")
    final_record = save_parameters(
        output / "final_parameters.npz", representation, final,
        metadata=parameter_metadata(algorithm_state.iter),
    )
    loaded, loaded_record = load_parameters(output / "final_parameters.npz", representation)
    if parameter_pytree_sha256(loaded) != final_record["parameter_pytree_sha256"]:
        raise RuntimeError("saved M1-Y checkpoint is not readable with identical parameters")
    rain_audit = read_json_record(
        _resolved(repository_root(), historical["configuration"]["truth"]["run_directory"])
        / "rain_activity_audit.json"
    )
    comparison_scale = float(
        rain_audit["activity_tolerance"]["comparison_rate_scale"]
    )
    metrics = _training_support_metrics(
        representation, loaded, arrays, historical["normalization"],
        comparison_scale,
    )
    fit_result = {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "representation": representation,
        "stage": EXPECTED_STAGE,
        "evaluation_state": "Y_n*=P(X_n*)",
        "training_states": [0, 80],
        "training_state_count": 81,
        "total_sample_count": int(arrays["m1y_A"].size),
        "architecture": RainMLPConfiguration(representation).to_record(),
        "initial_parameter_pytree_sha256": seed_sha,
        "initialization_convention": "Problem-A Glorot-uniform per layer, zero bias",
        "accepted_iterations": int(algorithm_state.iter),
        "termination_reason": str(algorithm_state.statusFlag),
        "final_objective": final_objective,
        "objective_denominator": float(objective.denominator),
        "objective_evaluations": int(adapter.value_evaluations),
        "gradient_evaluations": int(adapter.gradient_evaluations),
        "objective_history_all_finite": True,
        "gradient_norm_history_all_finite": True,
        "minimum_objective_evaluated": float(np.min(adapter.value_history)),
        "maximum_objective_evaluated": float(np.max(adapter.value_history)),
        "wall_seconds": float(perf_counter() - started),
        "final_parameter_file": str((output / "final_parameters.npz").resolve()),
        "final_parameter_npz_sha256": file_sha256(output / "final_parameters.npz"),
        "final_parameter_sidecar_sha256": file_sha256(output / "final_parameters.json"),
        "final_parameter_pytree_sha256": final_record["parameter_pytree_sha256"],
        "checkpoint_readable": loaded_record == final_record,
        "direct_training_support_metrics": metrics,
        "feature_order": list(FEATURES),
        "normalization": historical["normalization"].to_record(),
        "normalization_refitted_on_Y": False,
        "output_scales_refitted_on_Y": False,
        "pretraining_validation": str(Path(validation_path).resolve()),
        "pretraining_validation_sha256": file_sha256(validation_path),
        "preparation": str(preparation_source),
        "preparation_sha256": file_sha256(preparation_source),
        "campaign_configuration": str(configuration_source),
        "campaign_configuration_sha256": file_sha256(configuration_source),
        "source_secant_history_reused": False,
        "parameter_only_restart_restores_secant_history": False,
        "recursive_rollout": False,
        "differentiate_through_prefix": False,
        "command": [sys.executable, "-m", "dimswe.test2b_m1y_campaign", *sys.argv[1:]],
    }
    write_json_record(output / "fit_result.json", fit_result)
    write_json_record(output / "fit_progress.json", fit_result)
    return fit_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    immutable = subparsers.add_parser("verify-immutable-inputs")
    immutable.add_argument("--configuration", required=True)
    immutable.add_argument("--output", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--configuration", required=True)
    prepare.add_argument("--immutable-manifest", required=True)
    prepare.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--configuration", required=True)
    validate.add_argument("--preparation", required=True)
    validate.add_argument("--output", required=True)
    certify = subparsers.add_parser("certify-objectives")
    certify.add_argument("--configuration", required=True)
    certify.add_argument("--preparation", required=True)
    certify.add_argument("--validation", required=True)
    certify.add_argument("--output", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--configuration", required=True)
    train.add_argument("--preparation", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--representation", choices=("A", "B", "C"), required=True)
    train.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-immutable-inputs":
        verify_immutable_truth(args.configuration, args.output)
    elif args.command == "prepare":
        prepare_m1y(args.configuration, args.immutable_manifest, args.output)
    elif args.command == "validate":
        validate_m1y(args.configuration, args.preparation, args.output)
    elif args.command == "certify-objectives":
        certify_objectives(
            args.configuration, args.preparation, args.validation, args.output
        )
    else:
        train_m1y(
            args.configuration, args.preparation, args.validation,
            args.representation, args.output_directory,
        )


if __name__ == "__main__":
    main()
