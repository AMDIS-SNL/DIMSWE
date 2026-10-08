#!/usr/bin/env python3
"""Extract deterministic plotting maps from immutable DIMSWE truth states.

Saved-state panels are sampled on the two interior GLL points per coordinate in
each cell.  A and R are evaluated after exact replay of split children 1--5,
at the input to the deployed moist Euler child, matching the truth audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from firedrake import SpatialCoordinate

from dimswe.hidden_c0 import STATE_FIELDS, _serial_solver_parameters
from dimswe.jax_moist_adapter import JAXMoistEulerPrimal
from dimswe.resolved_hidden_c0 import ResolvedPilotConfiguration
from dimswe.resolved_hidden_c0_driver import build_resolved_hidden_c0_case
from dimswe.ufl_helpers import curl2D


INTERIOR_GLL = np.asarray((5, 6, 9, 10), dtype=np.int64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configuration_from_metadata(metadata, root: Path):
    values = dict(metadata["configuration"])
    values["output_directory"] = str(root)
    return ResolvedPilotConfiguration(**values)


def sustained_interval(records, duration=1000.0, minimum_mean_fraction=1.0e-4):
    floor = 128.0 * np.finfo(np.float64).eps * abs(
        float(records[0]["total_water_mass"])
    )
    for start in range(len(records)):
        fractions = []
        for stop in range(start, len(records)):
            row = records[stop]
            if not (
                float(row["physically_meaningful_R_fraction"]) > 0.0
                and float(row.get("rain_source_mass_rate", 0.0)) > 0.0
            ):
                break
            fractions.append(float(row["physically_meaningful_R_fraction"]))
            elapsed = float(row["time"]) - float(records[start]["time"])
            rain_gain = float(row["rain_water_mass"]) - float(
                records[start]["rain_water_mass"]
            )
            if (
                elapsed >= duration
                and float(np.mean(fractions)) >= minimum_mean_fraction
                and rain_gain > floor
            ):
                return {
                    "start_step": int(records[start]["step"]),
                    "start_time": float(records[start]["time"]),
                    "certification_step": int(row["step"]),
                    "certification_time": float(row["time"]),
                    "duration": elapsed,
                    "saved_states": int(stop - start + 1),
                    "mean_physically_meaningful_R_fraction": float(
                        np.mean(fractions)
                    ),
                    "rain_mass_gain": rain_gain,
                    "rain_mass_float64_floor": floor,
                }
    return None


def event_summary(records):
    def first_index(predicate):
        return next((i for i, row in enumerate(records) if predicate(row)), None)

    def peak_index(name):
        return int(np.argmax([float(row[name]) for row in records]))

    first = first_index(
        lambda row: float(row["physically_meaningful_R_fraction"]) > 0.0
    )
    indices = {
        "initial": 0,
        "last_clearly_pre_rain": (
            len(records) - 1 if first is None else max(0, first - 1)
        ),
        "first_certifiable_rain": first,
        "peak_integrated_cloud_water": peak_index("Qc_mass"),
        "peak_integrated_rain_production_rate": (
            None if first is None else peak_index("rain_source_mass_rate")
        ),
        "peak_local_specific_cloud_water": peak_index("specific_Qc_maximum"),
        "peak_local_rain_rate": (
            None if first is None else peak_index("R_maximum_absolute")
        ),
        "final": len(records) - 1,
    }
    result = {}
    for name, index in indices.items():
        if index is None:
            result[name] = None
            continue
        row = records[index]
        result[name] = {
            "step": int(row["step"]),
            "time_s": float(row["time"]),
        }
    result["sustained_rain"] = sustained_interval(records)
    result["mature_rain_state"] = result[
        "peak_integrated_rain_production_rate"
    ]
    return result


def extract(run: Path, audit_path: Path, cache_path: Path, summary_path: Path):
    started = perf_counter()
    run = run.resolve()
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or audit.get("status") != "complete":
        raise RuntimeError("source truth and audit must both be complete")
    configuration = configuration_from_metadata(metadata, run)
    steps = tuple(int(value) for value in metadata["time"]["output_steps"])
    records = tuple(audit["records"])
    if tuple(int(row["step"]) for row in records) != steps:
        raise RuntimeError("audit and truth output steps differ")

    case = build_resolved_hidden_c0_case(configuration)
    adapter = JAXMoistEulerPrimal(
        case.model, _serial_solver_parameters(), use_jit=True
    )
    field_index = {name: index for index, name in enumerate(STATE_FIELDS)}

    coordinate = SpatialCoordinate(case.model.mesh)
    _, packed_x = adapter.interpolate_and_pack(coordinate[0], "figure_coordinate_x")
    _, packed_y = adapter.interpolate_and_pack(coordinate[1], "figure_coordinate_y")
    packed_x = np.asarray(packed_x, dtype=np.float64)
    packed_y = np.asarray(packed_y, dtype=np.float64)
    x_flat = packed_x[:, INTERIOR_GLL].reshape(-1)
    y_flat = packed_y[:, INTERIOR_GLL].reshape(-1)
    # Periodic mesh coordinate interpolation can differ by roundoff at nominally
    # identical tensor-product levels.  Micrometre-rounded keys recover the
    # exact structured ordering without changing any sampled field value.
    x_key = np.round(x_flat, decimals=6)
    y_key = np.round(y_flat, decimals=6)
    order = np.lexsort((x_key, y_key))
    nx_plot = int(configuration.nx) * 2
    ny_plot = int(configuration.ny) * 2
    if order.size != nx_plot * ny_plot:
        raise RuntimeError("interior-GLL plotting grid has the wrong size")
    x_grid = x_key[order].reshape(ny_plot, nx_plot)
    y_grid = y_key[order].reshape(ny_plot, nx_plot)
    if not (
        np.all(np.diff(x_grid, axis=1) > 0.0)
        and np.all(np.diff(y_grid, axis=0) > 0.0)
    ):
        raise RuntimeError("plotting coordinates are not strictly ordered")

    def grid(packed):
        values = np.asarray(packed, dtype=np.float64)
        return values[:, INTERIOR_GLL].reshape(-1)[order].reshape(
            ny_plot, nx_plot
        )

    shape = (len(steps), ny_plot, nx_plot)
    maps = {
        name: np.empty(shape, dtype=np.float32)
        for name in (
            "height_anomaly_m",
            "relative_vorticity_1e5_s-1",
            "supersaturation_percent",
            "specific_cloud_g_kg-1",
            "specific_rain_ug_kg-1",
            "A_g_kg-1_h-1",
            "R_ug_kg-1_h-1",
        )
    }
    exact = {
        name: np.empty(len(steps), dtype=np.float64)
        for name in (
            "A_min_s-1",
            "A_max_s-1",
            "A_negative_fraction",
            "A_positive_fraction",
            "R_max_s-1",
            "R_positive_fraction",
            "postprefix_saturation_min",
            "postprefix_saturation_max",
        )
    }

    for frame, step in enumerate(steps):
        values = np.load(
            run / "restart" / f"step_{step:08d}.npy", allow_pickle=False
        )
        state = case.state_from_values(values, f"figure_state_{step}")

        packed = {}
        for name in ("h", "S", "Qv", "Qc", "Qr"):
            _, packed[name] = adapter.interpolate_and_pack(
                state.sub(field_index[name]), f"figure_saved_{name}_{step}"
            )
            packed[name] = np.asarray(packed[name], dtype=np.float64)
        _, packed_curl = adapter.interpolate_and_pack(
            curl2D(state.sub(field_index["v"])), f"figure_curl_{step}"
        )

        h = grid(packed["h"])
        specific_qv = grid(packed["Qv"]) / h
        specific_qc = grid(packed["Qc"]) / h
        specific_qr = grid(packed["Qr"]) / h
        specific_s = grid(packed["S"]) / h
        qsat = (
            0.002
            * 750.0
            / h
            * np.exp(20.0 * (1.0 - specific_s / 9.80616))
        )
        maps["height_anomaly_m"][frame] = h - 750.0
        maps["relative_vorticity_1e5_s-1"][frame] = grid(packed_curl) * 1.0e5
        maps["supersaturation_percent"][frame] = 100.0 * (
            specific_qv / qsat - 1.0
        )
        maps["specific_cloud_g_kg-1"][frame] = specific_qc * 1.0e3
        maps["specific_rain_ug_kg-1"][frame] = specific_qr * 1.0e9

        complete = case.helper.take_forward_step_cached(
            state, float(step * configuration.dt), float(configuration.dt)
        )
        post_prefix = complete.boundary_states[-2]
        moist = adapter.evaluate(post_prefix, float(configuration.dt))
        a_rate = np.asarray(moist.rates["A"], dtype=np.float64)
        r_rate = np.asarray(moist.rates["R"], dtype=np.float64)
        maps["A_g_kg-1_h-1"][frame] = grid(a_rate) * 3.6e6
        maps["R_ug_kg-1_h-1"][frame] = grid(r_rate) * 3.6e12
        post_h = np.asarray(moist.packed_state["h"], dtype=np.float64)
        post_s = np.asarray(moist.packed_state["S"], dtype=np.float64) / post_h
        post_qv = np.asarray(moist.packed_state["Qv"], dtype=np.float64) / post_h
        post_qsat = (
            float(np.asarray(moist.parameters["q0"]))
            * float(np.asarray(moist.parameters["H0"]))
            / (
                post_h
                + np.asarray(moist.packed_fields["B"], dtype=np.float64)
            )
            * np.exp(
                20.0
                * (
                    1.0
                    - post_s / float(np.asarray(moist.parameters["g"]))
                )
            )
        )
        exact["A_min_s-1"][frame] = float(np.min(a_rate))
        exact["A_max_s-1"][frame] = float(np.max(a_rate))
        exact["A_negative_fraction"][frame] = float(np.mean(a_rate < 0.0))
        exact["A_positive_fraction"][frame] = float(np.mean(a_rate > 0.0))
        exact["R_max_s-1"][frame] = float(np.max(r_rate))
        exact["R_positive_fraction"][frame] = float(np.mean(r_rate > 0.0))
        exact["postprefix_saturation_min"][frame] = float(
            np.min(post_qv / post_qsat)
        )
        exact["postprefix_saturation_max"][frame] = float(
            np.max(post_qv / post_qsat)
        )
        if frame % 10 == 0 or frame == len(steps) - 1:
            print(f"extracted {frame + 1}/{len(steps)} states (step {step})", flush=True)

    series_names = (
        "Qc_mass",
        "rain_water_mass",
        "rain_source_mass_rate",
        "specific_Qc_maximum",
        "saturation_ratio_minimum",
        "saturation_ratio_maximum",
        "physically_meaningful_R_fraction",
        "total_water_mass",
    )
    series = {
        name: np.asarray([float(row[name]) for row in records], dtype=np.float64)
        for name in series_names
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            step=np.asarray(steps, dtype=np.int64),
            time_s=np.asarray([float(row["time"]) for row in records]),
            x_km=x_grid[0] / 1000.0,
            y_km=y_grid[:, 0] / 1000.0,
            **maps,
            **exact,
            **series,
        )
    temporary.replace(cache_path)

    events = event_summary(records)
    summary = {
        "format_version": 1,
        "status": "complete",
        "source_truth": str(run),
        "source_truth_modified": False,
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": file_sha256(metadata_path),
        "source_audit": str(audit_path.resolve()),
        "source_audit_sha256": file_sha256(audit_path.resolve()),
        "source_configuration_sha256": metadata["configuration_sha256"],
        "truth_git_checkpoint": metadata.get("git", {}).get("checkpoint"),
        "steps": [int(steps[0]), int(steps[-1])],
        "frame_count": len(steps),
        "truth_cadence_s": float(configuration.dt),
        "plot_grid": {
            "shape_yx": [ny_plot, nx_plot],
            "definition": (
                "two interior GLL nodes per coordinate per quadrilateral cell; "
                "cell-boundary duplicates excluded"
            ),
            "state_source": "saved restart arrays",
            "rate_source": (
                "exact split replay through children 1--5 followed by the "
                "deployed analytical moist law at child-6 input"
            ),
        },
        "variables": {
            "height_anomaly_m": "h - H0 at saved state",
            "relative_vorticity_1e5_s-1": "repository curl2D(v) times 1e5",
            "supersaturation_percent": "100*(qv/qsat - 1) at saved state",
            "specific_cloud_g_kg-1": "1000*Qc/h at saved state",
            "specific_rain_ug_kg-1": "1e9*Qr/h at saved state",
            "A_g_kg-1_h-1": "3.6e6*A at exact post-prefix state",
            "R_ug_kg-1_h-1": "3.6e12*R at exact post-prefix state",
        },
        "events": events,
        "conservation": {
            "initial_total_water_mass": audit["summary"][
                "initial_total_water_mass"
            ],
            "maximum_absolute_total_water_drift": audit["summary"][
                "maximum_absolute_total_water_drift"
            ],
            "relative_maximum_total_water_drift": audit["summary"][
                "relative_maximum_total_water_drift"
            ],
            "maximum_source_invariant_residuals": audit[
                "maximum_source_invariant_residuals"
            ],
        },
        "cache": str(cache_path.resolve()),
        "cache_sha256": file_sha256(cache_path),
        "wall_seconds": float(perf_counter() - started),
    }
    write_json(summary_path, summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", required=True, type=Path)
    result.add_argument("--audit", required=True, type=Path)
    result.add_argument("--cache", required=True, type=Path)
    result.add_argument("--summary", required=True, type=Path)
    return result


def main():
    arguments = parser().parse_args()
    summary = extract(
        arguments.run, arguments.audit, arguments.cache, arguments.summary
    )
    print(json.dumps(summary["events"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
