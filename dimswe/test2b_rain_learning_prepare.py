"""Read-only freeze and support audit for the completed rain-active truth.

Neither command advances or writes a truth state.  ``freeze-truth`` hashes the
existing run.  ``audit-support`` reconstructs saved states and evaluates the
accepted analytical JAX law at boundary and post-prefix GLL states.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .test2b_rain_case_design import proposed_sustained_interval
from .test2b_rain_learning import FEATURE_ORDER, SOURCE_ORDER, canonical_sha256


FORMAT_VERSION = 1


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, record):
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


def _inventory(root, relative_paths):
    records = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(relative),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return records


def freeze_truth(run_directory, output):
    started = perf_counter()
    root = Path(run_directory).resolve()
    metadata_path = root / "metadata.json"
    audit_path = root / "rain_activity_audit.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or audit.get("status") != "complete":
        raise RuntimeError("truth metadata and rain audit must both be complete")
    steps = tuple(range(161))
    if tuple(metadata.get("completed_output_steps", ())) != steps:
        raise RuntimeError("completed truth steps are not exactly 0..160")
    if tuple(audit["state_access"]["saved_steps"]) != steps:
        raise RuntimeError("rain audit did not read exactly states 0..160")
    configuration = metadata["configuration"]
    expected = {
        "nx": 64, "ny": 64, "dt": 100.0, "nsteps": 160,
        "initial_moisture_zeta": -0.06, "moist_backend": "ufl",
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise RuntimeError(f"truth configuration {key} changed")
    moist = audit["moist_parameters"]
    if moist["qprecip"] != 1.0e-4 or moist["gamma_r"] != 0.001:
        raise RuntimeError("accepted rain-law constants changed")
    sustained = proposed_sustained_interval(audit["records"])
    if sustained is None or sustained["start_step"] != 51 or sustained["certification_step"] != 61:
        raise RuntimeError("completed run does not satisfy the approved rain criterion")
    restart = _inventory(root, [Path("restart") / f"step_{step:08d}.npy" for step in steps])
    checkpoints = _inventory(root, [Path("checkpoints") / f"step_{step:08d}.h5" for step in steps])
    diagnostics = _inventory(root, [Path("diagnostics") / f"step_{step:08d}.json" for step in steps])
    spectra = _inventory(root, [Path("spectra") / f"step_{step:08d}.npz" for step in steps])
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "identity": "frozen Test2B rain-active double-vortex truth",
        "source_run": str(root),
        "source_run_modified": False,
        "configuration": configuration,
        "configuration_sha256": metadata["configuration_sha256"],
        "metadata_sha256": file_sha256(metadata_path),
        "rain_activity_audit_sha256": file_sha256(audit_path),
        "mesh": metadata["mesh"],
        "state_convention": metadata["state_convention"],
        "time": metadata["time"],
        "initial_condition": metadata["initial_condition"],
        "physical_parameters": metadata["physical_parameters"],
        "moist_parameters": moist,
        "analytical_laws": {
            "source": "dimswe/jax_moist.py::_moist_algebra and UFL ThreeWayPhysics",
            "A": "E-C, with C=max(0,gamma_v(qv-qsat)/dt), E=min(qc/dt,max(0,gamma_v(qsat-qv)/dt))",
            "R": "max(0,gamma_r(qc-qprecip)/dt)",
            "source_map": {"S": "h*g*L*A", "Qv": "h*A", "Qc": "-h*(A+R)", "Qr": "h*R"},
        },
        "six_child_order": metadata["solver"]["six_child_order"],
        "rain_regimes": {
            "criterion": {
                "minimum_continuous_duration": 1000.0,
                "meaningful_positive_integrated_rain_at_every_saved_state": True,
                "minimum_mean_active_GLL_fraction": 1.0e-4,
                "positive_rain_mass_gain_above_float64_floor": True,
            },
            "PRE_RAIN": {"steps": [0, 50], "times": [0.0, 5000.0]},
            "first_exact_R": {"step": 51, "time": 5100.0},
            "first_meaningful_R": {"step": 51, "time": 5100.0},
            "ONSET": {"steps": [51, 60], "times": [5100.0, 6000.0]},
            "first_sustained_certification": sustained,
            "SUSTAINED_RAIN_ACTIVE": {"steps": [61, 160], "times": [6100.0, 16000.0]},
        },
        "conservation_certificate": {
            "relative_maximum_total_water_drift": audit["summary"]["relative_maximum_total_water_drift"],
            "maximum_absolute_total_water_drift": audit["summary"]["maximum_absolute_total_water_drift"],
            "maximum_source_invariant_residuals": audit["maximum_source_invariant_residuals"],
        },
        "rain_certificate": audit["summary"],
        "inventories": {
            "restart_state_arrays": restart,
            "firedrake_checkpoints": checkpoints,
            "diagnostics": diagnostics,
            "spectra": spectra,
        },
    }
    payload = dict(manifest)
    manifest["manifest_payload_sha256"] = canonical_sha256(payload)
    manifest["inventory_bytes"] = {
        name: sum(item["bytes"] for item in values)
        for name, values in manifest["inventories"].items()
    }
    manifest["wall_seconds"] = float(perf_counter() - started)
    write_json(output, manifest)
    return manifest


def _weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.sum(weights * values * values) / np.sum(weights)))


def _distribution(values, *, active=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = values if active is None else values[np.asarray(active, dtype=bool).reshape(-1)]
    result = {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "negative_count": int(np.count_nonzero(values < 0.0)),
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "positive_count": int(np.count_nonzero(values > 0.0)),
    }
    if selected.size:
        result["conditional_count"] = int(selected.size)
        result["conditional_quantiles"] = {
            str(q): float(np.quantile(selected, q)) for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        }
    return result


def _regime(step):
    if step <= 50:
        return "PRE_RAIN"
    if step <= 60:
        return "ONSET"
    return "SUSTAINED_RAIN_ACTIVE"


def audit_support(run_directory, output):
    """Evaluate analytical labels at all boundary and post-prefix saved states."""
    from firedrake import TestFunction, TrialFunction, assemble, inner
    from .hidden_c0 import STATE_FIELDS, _serial_solver_parameters
    from .jax_moist_adapter import JAXMoistEulerPrimal
    from .resolved_hidden_c0 import ResolvedPilotConfiguration
    from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
    from .test2a_discrete_training import _scipy_csr_from_petsc

    started = perf_counter()
    root = Path(run_directory).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "rain_activity_audit.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or audit.get("status") != "complete":
        raise RuntimeError("support audit requires a complete frozen truth")
    configuration = ResolvedPilotConfiguration(**{**metadata["configuration"], "output_directory": str(root)})
    case = build_resolved_hidden_c0_case(configuration)
    adapter = JAXMoistEulerPrimal(case.model, _serial_solver_parameters(), use_jit=True)
    carrier = adapter.carrier_space
    mass = assemble(inner(TestFunction(carrier), TrialFunction(carrier)) * case.model.spaces.dx, mat_type="aij")
    csr = _scipy_csr_from_petsc(mass).tocsr()
    packed = np.asarray(adapter.layout.cell_nodes, dtype=np.int64).reshape(-1)
    weights = np.asarray(csr.diagonal()[packed], dtype=np.float64)
    if np.any(weights <= 0.0):
        raise RuntimeError("GLL carrier weights are not positive")
    locations = {"boundary": [], "postprefix": []}
    records = []
    field_index = {name: index for index, name in enumerate(STATE_FIELDS)}
    for step in range(161):
        values = np.load(root / "restart" / f"step_{step:08d}.npy", allow_pickle=False)
        state = case.state_from_values(values, f"test2b_learning_support_{step}")
        time = float(step * case.dt)
        complete = case.helper.take_forward_step_cached(state, time, case.dt)
        states = {"boundary": state, "postprefix": complete.boundary_states[-2]}
        step_record = {"step": step, "time": time, "regime": _regime(step)}
        for location, active_state in states.items():
            moist = adapter.evaluate(active_state, case.dt)
            h = np.asarray(moist.packed_state["h"], dtype=np.float64).reshape(-1)
            features = np.stack(
                tuple(np.asarray(moist.packed_state[name]).reshape(-1) for name in ("h", "S", "Qv", "Qc"))
                + (np.asarray(moist.packed_fields["B"]).reshape(-1),), axis=-1,
            )
            _, qr = adapter.interpolate_and_pack(active_state.sub(field_index["Qr"]), f"test2b_support_Qr_{location}_{step}")
            a = np.asarray(moist.rates["A"], dtype=np.float64).reshape(-1)
            r = np.asarray(moist.rates["R"], dtype=np.float64).reshape(-1)
            source = np.stack(tuple(np.asarray(moist.source_density[name]).reshape(-1) for name in SOURCE_ORDER), axis=-1)
            qsat = float(moist.parameters["q0"] * moist.parameters["H0"]) / (features[:, 0] + features[:, 4]) * np.exp(20.0 * (1.0 - features[:, 1] / features[:, 0] / float(moist.parameters["g"])))
            locations[location].append({"features": features, "Qr": np.asarray(qr).reshape(-1), "A": a, "R": r, "source": source, "qsat": qsat})
            step_record[location] = {
                "A": _distribution(a),
                "R": _distribution(r, active=r > 0.0),
                "R_active_fraction": float(np.mean(r > 0.0)),
            }
        records.append(step_record)

    summary = {}
    for location, rows in locations.items():
        summary[location] = {}
        for regime in ("PRE_RAIN", "ONSET", "SUSTAINED_RAIN_ACTIVE"):
            selected = [rows[i] for i in range(161) if _regime(i) == regime]
            a = np.concatenate([row["A"] for row in selected])
            r = np.concatenate([row["R"] for row in selected])
            summary[location][regime] = {"saved_states": len(selected), "A": _distribution(a), "R": _distribution(r, active=r > 0.0), "R_active_fraction": float(np.mean(r > 0.0))}
        all_a = np.concatenate([row["A"] for row in rows])
        all_r = np.concatenate([row["R"] for row in rows])
        summary[location]["ALL"] = {"saved_states": 161, "A": _distribution(all_a), "R": _distribution(all_r, active=all_r > 0.0), "R_active_fraction": float(np.mean(all_r > 0.0))}

    training = locations["boundary"][:81]
    feature_values = np.stack([row["features"] for row in training])
    a_values = np.stack([row["A"] for row in training])
    r_values = np.stack([row["R"] for row in training])
    source_values = np.stack([row["source"] for row in training])
    tiled_weights = np.broadcast_to(weights, a_values.shape)
    total_weight = float(np.sum(tiled_weights))
    input_offset = np.sum(tiled_weights[..., None] * feature_values, axis=(0, 1)) / total_weight
    input_scale = np.sqrt(np.sum(tiled_weights[..., None] * (feature_values - input_offset) ** 2, axis=(0, 1)) / total_weight)
    maximum_absolute = np.max(np.abs(feature_values), axis=(0, 1))
    degenerate = input_scale <= 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, maximum_absolute)
    input_scale = np.where(degenerate, 1.0, input_scale)
    active = r_values > 0.0
    if not np.any(active):
        raise RuntimeError("training support contains no active analytical R")
    sigma_r = _weighted_rms(r_values[active], tiled_weights[active])
    source_scales = []
    for index in range(4):
        if index == 3:
            source_scales.append(_weighted_rms(source_values[..., index][active], tiled_weights[active]))
        else:
            source_scales.append(_weighted_rms(source_values[..., index], tiled_weights))
    scale_record = {
        "feature_order": list(FEATURE_ORDER),
        "input_offset": input_offset.tolist(),
        "input_scale": input_scale.tolist(),
        "input_zero_or_degenerate_scale": [bool(value) for value in degenerate],
        "sigma_A": _weighted_rms(a_values, tiled_weights),
        "sigma_R_active": sigma_r,
        "source_order": list(SOURCE_ORDER),
        "source_scales": source_scales,
        "training_state_indices": [0, 80],
        "R_active_sample_count": int(np.count_nonzero(active)),
        "R_total_sample_count": int(active.size),
        "R_active_fraction": float(np.mean(active)),
        "carrier_weight_sum": float(np.sum(weights)),
    }
    result = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "diagnostic": "Test2B rain-active boundary/post-prefix learning-support audit",
        "source_run": str(root),
        "source_metadata_sha256": file_sha256(root / "metadata.json"),
        "source_rain_audit_sha256": file_sha256(root / "rain_activity_audit.json"),
        "states_evaluated": [0, 160],
        "truth_states_modified": False,
        "deployed_GLL_samples_per_state": int(weights.size),
        "summary": summary,
        "per_state_rate_activity": records,
        "training_normalization": {**scale_record, "provenance_sha256": canonical_sha256(scale_record)},
        "training_regime_counts": {
            "boundary_states": {"PRE_RAIN": 51, "ONSET": 10, "SUSTAINED_RAIN_ACTIVE": 20},
            "H1_targets": {"PRE_RAIN": 50, "ONSET": 10, "SUSTAINED_RAIN_ACTIVE": 20},
            "H2_windows": {"PRE_RAIN": 25, "ONSET": 5, "SUSTAINED_RAIN_ACTIVE": 10},
            "H5_windows": {"PRE_RAIN": 10, "ONSET": 2, "SUSTAINED_RAIN_ACTIVE": 4},
        },
        "wall_seconds": float(perf_counter() - started),
    }
    write_json(output, result)
    return result


def repair_degenerate_feature_scales(input_path, output):
    """Apply the documented unit-scale rule to an audit made before that guard."""
    result = json.loads(Path(input_path).read_text(encoding="utf-8"))
    normalization = result["training_normalization"]
    scales = np.asarray(normalization["input_scale"], dtype=np.float64)
    degenerate = ~np.isfinite(scales) | (scales <= 0.0)
    scales[degenerate] = 1.0
    normalization["input_scale"] = scales.tolist()
    normalization["input_zero_or_degenerate_scale"] = [bool(value) for value in degenerate]
    provenance = dict(normalization)
    provenance.pop("provenance_sha256", None)
    normalization["provenance_sha256"] = canonical_sha256(provenance)
    result["normalization_repair"] = (
        "unit scale for zero/degenerate constant features, matching the accepted "
        "Test2A normalization convention; rate/source statistics unchanged"
    )
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-truth")
    freeze.add_argument("--run-directory", required=True)
    freeze.add_argument("--output", required=True)
    support = commands.add_parser("audit-support")
    support.add_argument("--run-directory", required=True)
    support.add_argument("--output", required=True)
    repair = commands.add_parser("repair-degenerate-scales")
    repair.add_argument("--input", required=True)
    repair.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-truth":
        freeze_truth(args.run_directory, args.output)
    elif args.command == "audit-support":
        audit_support(args.run_directory, args.output)
    else:
        repair_degenerate_feature_scales(args.input, args.output)


if __name__ == "__main__":
    main()
