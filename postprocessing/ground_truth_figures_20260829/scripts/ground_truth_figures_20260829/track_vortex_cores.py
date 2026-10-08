#!/usr/bin/env python3
"""Track the two DoubleVortex positive-vorticity cores from an extracted map cache."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DOMAIN_KM = 5000.0
VORTEX_WIDTH_KM = 375.0
NOMINAL_CENTERS_KM = np.array(((2000.0, 2000.0), (3000.0, 3000.0)))
ASSOCIATION_RADIUS_KM = 750.0
CENTROID_RADIUS_KM = 600.0
THRESHOLD_FRACTION = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def min_image(delta):
    return (np.asarray(delta) + 0.5 * DOMAIN_KM) % DOMAIN_KM - 0.5 * DOMAIN_KM


def periodic_distance(x, y, center):
    return np.hypot(min_image(x - center[0]), min_image(y - center[1]))


def weighted_centroid(
    field, x_grid, y_grid, center, radius_km, threshold_fraction=None
):
    dx = min_image(x_grid - center[0])
    dy = min_image(y_grid - center[1])
    neighborhood = dx * dx + dy * dy <= radius_km**2
    local_maximum = float(field[neighborhood].max())
    if threshold_fraction is not None:
        neighborhood &= field >= threshold_fraction * local_maximum
    weights = np.where(neighborhood, np.maximum(field, 0.0), 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("positive-vorticity centroid neighborhood has zero weight")
    centroid = np.array(
        (
            (center[0] + float((weights * dx).sum()) / total) % DOMAIN_KM,
            (center[1] + float((weights * dy).sum()) / total) % DOMAIN_KM,
        )
    )
    weighted_zeta = float((weights * field).sum() / total)
    return centroid, weighted_zeta, int(np.count_nonzero(weights)), local_maximum


def unwrap_track(wrapped):
    result = np.empty_like(wrapped)
    result[0] = wrapped[0]
    for index in range(1, len(wrapped)):
        result[index] = result[index - 1] + min_image(wrapped[index] - wrapped[index - 1])
    return result


def derived_track(xy):
    periodic_delta = min_image(xy - xy[0])
    displacement = np.linalg.norm(periodic_delta, axis=2)
    separation_vector = min_image(xy[:, 1] - xy[:, 0])
    separation = np.linalg.norm(separation_vector, axis=1)
    orientation = np.rad2deg(
        np.unwrap(np.arctan2(separation_vector[:, 1], separation_vector[:, 0]))
    )
    midpoint = (xy[:, 0] + 0.5 * separation_vector) % DOMAIN_KM
    midpoint_delta = min_image(midpoint - midpoint[0])
    midpoint_displacement = np.linalg.norm(midpoint_delta, axis=1)
    unwrapped = np.stack((unwrap_track(xy[:, 0]), unwrap_track(xy[:, 1])), axis=1)
    return {
        "periodic_delta": periodic_delta,
        "displacement": displacement,
        "separation_vector": separation_vector,
        "separation": separation,
        "orientation": orientation,
        "midpoint": midpoint,
        "midpoint_delta": midpoint_delta,
        "midpoint_displacement": midpoint_displacement,
        "unwrapped": unwrapped,
    }


def continuous_centroid_track(vorticity, x_grid, y_grid, radius_km, threshold_fraction):
    count = vorticity.shape[0]
    xy = np.empty((count, 2, 2), dtype=np.float64)
    weighted_zeta = np.empty((count, 2), dtype=np.float64)
    point_count = np.empty((count, 2), dtype=np.int64)
    local_maximum = np.empty((count, 2), dtype=np.float64)
    previous = NOMINAL_CENTERS_KM.copy()
    for frame in range(count):
        field = np.asarray(vorticity[frame], dtype=np.float64)
        for vortex in range(2):
            (
                xy[frame, vortex],
                weighted_zeta[frame, vortex],
                point_count[frame, vortex],
                local_maximum[frame, vortex],
            ) = weighted_centroid(
                field,
                x_grid,
                y_grid,
                previous[vortex],
                radius_km,
                threshold_fraction,
            )
        previous = xy[frame]
    return xy, weighted_zeta, point_count, local_maximum


def track(vorticity, x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    count = vorticity.shape[0]
    raw_xy = np.empty((count, 2, 2), dtype=np.float64)
    zeta_max = np.empty((count, 2), dtype=np.float64)
    (
        centroid_xy,
        centroid_weighted_zeta,
        centroid_points,
        centroid_local_maximum,
    ) = continuous_centroid_track(
        vorticity, x_grid, y_grid, CENTROID_RADIUS_KM, None
    )
    (
        threshold_xy,
        threshold_weighted_zeta,
        threshold_points,
        threshold_local_maximum,
    ) = continuous_centroid_track(
        vorticity, x_grid, y_grid, CENTROID_RADIUS_KM, THRESHOLD_FRACTION
    )

    for frame in range(count):
        field = np.asarray(vorticity[frame], dtype=np.float64)
        references = NOMINAL_CENTERS_KM if frame == 0 else centroid_xy[frame - 1]
        chosen_flat = []
        for vortex in range(2):
            support = periodic_distance(x_grid, y_grid, references[vortex]) <= ASSOCIATION_RADIUS_KM
            if not np.any(support):
                raise RuntimeError(f"empty association support at frame {frame}, vortex {vortex + 1}")
            restricted = np.where(support, field, -np.inf)
            flat = int(np.argmax(restricted))
            if flat in chosen_flat:
                raise RuntimeError(f"vortex association collision at frame {frame}")
            chosen_flat.append(flat)
            iy, ix = np.unravel_index(flat, field.shape)
            raw_xy[frame, vortex] = (x[ix], y[iy])
            zeta_max[frame, vortex] = field[iy, ix]

    return {
        "raw_xy": raw_xy,
        "centroid_xy": centroid_xy,
        "zeta_max": zeta_max,
        "centroid_weighted_zeta": centroid_weighted_zeta,
        "centroid_points": centroid_points,
        "centroid_local_maximum": centroid_local_maximum,
        "threshold_centroid_xy": threshold_xy,
        "threshold_centroid_weighted_zeta": threshold_weighted_zeta,
        "threshold_centroid_points": threshold_points,
        "threshold_centroid_local_maximum": threshold_local_maximum,
        "raw": derived_track(raw_xy),
        "centroid": derived_track(centroid_xy),
        "threshold_centroid": derived_track(threshold_xy),
    }


def table_columns(result):
    columns = {"step": result["step"], "time_s": result["time_s"]}
    for prefix in ("raw", "centroid", "threshold_centroid"):
        xy = result[f"{prefix}_xy"]
        derived = result[prefix]
        for vortex in range(2):
            number = vortex + 1
            columns[f"{prefix}_x{number}_km"] = xy[:, vortex, 0]
            columns[f"{prefix}_y{number}_km"] = xy[:, vortex, 1]
            columns[f"{prefix}_unwrapped_x{number}_km"] = derived["unwrapped"][:, vortex, 0]
            columns[f"{prefix}_unwrapped_y{number}_km"] = derived["unwrapped"][:, vortex, 1]
            columns[f"{prefix}_dx{number}_km"] = derived["periodic_delta"][:, vortex, 0]
            columns[f"{prefix}_dy{number}_km"] = derived["periodic_delta"][:, vortex, 1]
            columns[f"{prefix}_displacement{number}_km"] = derived["displacement"][:, vortex]
        columns[f"{prefix}_separation_km"] = derived["separation"]
        columns[f"{prefix}_orientation_deg_unwrapped"] = derived["orientation"]
        columns[f"{prefix}_midpoint_x_km"] = derived["midpoint"][:, 0]
        columns[f"{prefix}_midpoint_y_km"] = derived["midpoint"][:, 1]
        columns[f"{prefix}_midpoint_dx_km"] = derived["midpoint_delta"][:, 0]
        columns[f"{prefix}_midpoint_dy_km"] = derived["midpoint_delta"][:, 1]
        columns[f"{prefix}_midpoint_displacement_km"] = derived["midpoint_displacement"]
    for vortex in range(2):
        number = vortex + 1
        columns[f"zeta_max{number}_1e-5_s-1"] = result["zeta_max"][:, vortex]
        columns[f"centroid_weighted_zeta{number}_1e-5_s-1"] = result[
            "centroid_weighted_zeta"
        ][:, vortex]
        columns[f"centroid_point_count{number}"] = result["centroid_points"][:, vortex]
        columns[f"threshold_centroid_weighted_zeta{number}_1e-5_s-1"] = result[
            "threshold_centroid_weighted_zeta"
        ][:, vortex]
        columns[f"threshold_centroid_point_count{number}"] = result[
            "threshold_centroid_points"
        ][:, vortex]
    return columns


def write_csv(path, columns):
    names = list(columns)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names)
        for row in range(len(columns["step"])):
            writer.writerow(
                int(columns[name][row])
                if name == "step" or "centroid_point_count" in name
                else format(float(columns[name][row]), ".17g")
                for name in names
            )


def make_figure(result, output):
    time_h = result["time_s"] / 3600.0
    smooth = result["centroid"]
    colors = ("#167D9A", "#C44E52")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), constrained_layout=True)
    local_delta = smooth["periodic_delta"]
    for vortex, color in enumerate(colors):
        axes[0].plot(
            local_delta[:, vortex, 0],
            local_delta[:, vortex, 1],
            "-",
            lw=2.3,
            color=color,
            label=f"core {vortex + 1}",
        )
        axes[0].plot(
            local_delta[0, vortex, 0],
            local_delta[0, vortex, 1],
            marker="o" if vortex == 0 else "D",
            ms=4.5,
            mfc="white",
            mec=color,
            mew=1.3,
            ls="none",
            zorder=5,
        )
        axes[0].plot(
            local_delta[-1, vortex, 0],
            local_delta[-1, vortex, 1],
            marker="s" if vortex == 0 else "^",
            ms=4.8,
            mfc=color,
            mec=color,
            ls="none",
            zorder=5,
        )
        axes[1].plot(
            time_h,
            smooth["displacement"][:, vortex],
            "-",
            lw=3.2 if vortex == 0 else 1.7,
            color=color,
            label=f"core {vortex + 1}",
            zorder=2 + vortex,
        )
    local_limit = max(10.0, 1.12 * float(np.max(np.abs(local_delta))))
    axes[0].set(
        xlabel=r"$\Delta x_i$ (km)",
        ylabel=r"$\Delta y_i$ (km)",
        xlim=(-local_limit, local_limit),
        ylim=(-local_limit, local_limit),
        title="Local centroid trajectories",
    )
    axes[0].set_aspect("equal")
    axes[0].axhline(0.0, color="0.75", lw=0.8, zorder=0)
    axes[0].axvline(0.0, color="0.75", lw=0.8, zorder=0)
    axes[0].legend(fontsize=11, loc="best")
    axes[1].set(
        xlabel="time (h)",
        ylabel="displacement from initial (km)",
        title="Core displacement",
    )
    axes[1].legend(fontsize=11, loc="upper left")
    axes[1].text(
        0.96,
        0.08,
        "core curves overlap by symmetry",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="0.3",
    )
    separation = axes[2].plot(
        time_h,
        smooth["separation"],
        color="#6A3D9A",
        lw=2.3,
        label="pair separation",
    )[0]
    axes[2].set(
        xlabel="time (h)",
        ylabel="pair separation (km)",
        title="Pair geometry",
    )
    angle_axis = axes[2].twinx()
    angle = angle_axis.plot(
        time_h,
        smooth["orientation"],
        color="#E07A2D",
        lw=2.2,
        label="orientation",
    )[0]
    angle_axis.set_ylabel("orientation (deg, unwrapped)", color="#E07A2D", fontsize=13)
    angle_axis.tick_params(labelsize=11, colors="#E07A2D")
    axes[2].legend(
        (separation, angle),
        ("pair separation", "orientation"),
        fontsize=10.5,
        loc="upper left",
    )
    for index, axis in enumerate(axes):
        axis.grid(alpha=0.25)
        axis.set_title(axis.get_title(), fontsize=14)
        axis.set_xlabel(axis.get_xlabel(), fontsize=13)
        axis.set_ylabel(axis.get_ylabel(), fontsize=13)
        axis.tick_params(labelsize=11)
        axis.text(
            0.015,
            0.985,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            zorder=10,
        )
    png = output / "figure5_test2b_vortex_core_motion.png"
    pdf = output / "figure5_test2b_vortex_core_motion.pdf"
    fixed_time = datetime(2026, 8, 29, tzinfo=timezone.utc)
    fixed = {"CreationDate": fixed_time, "ModDate": fixed_time}
    fig.savefig(png, dpi=300, metadata={"Software": "DIMSWE cached-map vortex tracker"})
    fig.savefig(pdf, metadata=fixed)
    plt.close(fig)
    return png, pdf


def summarize_geometry(xy, derived):
    step_motion = np.linalg.norm(min_image(np.diff(xy, axis=0)), axis=2)
    summary = {
        "core1_maximum_displacement_km": float(derived["displacement"][:, 0].max()),
        "core2_maximum_displacement_km": float(derived["displacement"][:, 1].max()),
        "core1_maximum_single_step_motion_km": float(step_motion[:, 0].max()),
        "core2_maximum_single_step_motion_km": float(step_motion[:, 1].max()),
        "pair_midpoint_maximum_displacement_km": float(
            derived["midpoint_displacement"].max()
        ),
        "separation_initial_km": float(derived["separation"][0]),
        "separation_final_km": float(derived["separation"][-1]),
        "separation_range_km": [
            float(derived["separation"].min()),
            float(derived["separation"].max()),
        ],
        "separation_maximum_absolute_change_from_initial_km": float(
            np.max(np.abs(derived["separation"] - derived["separation"][0]))
        ),
        "orientation_initial_deg": float(derived["orientation"][0]),
        "orientation_final_deg": float(derived["orientation"][-1]),
        "orientation_range_deg_unwrapped": [
            float(derived["orientation"].min()),
            float(derived["orientation"].max()),
        ],
        "orientation_maximum_absolute_change_from_initial_deg": float(
            np.max(np.abs(derived["orientation"] - derived["orientation"][0]))
        ),
    }
    for vortex in (1, 2):
        distance = summary[f"core{vortex}_maximum_displacement_km"]
        summary[f"core{vortex}_maximum_displacement_domain_fraction"] = (
            distance / DOMAIN_KM
        )
        summary[f"core{vortex}_maximum_displacement_width_fraction"] = (
            distance / VORTEX_WIDTH_KM
        )
    return summary


def raw_switch_audit(result, vorticity, x, y):
    def is_eight_neighbor_maximum(field, iy, ix):
        center_value = field[iy, ix]
        neighbors = [
            field[(iy + dy) % field.shape[0], (ix + dx) % field.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dx or dy
        ]
        return bool(center_value >= max(neighbors))

    raw = result["raw_xy"]
    motion = np.linalg.norm(min_image(np.diff(raw, axis=0)), axis=2)
    transitions = np.flatnonzero(np.any(motion > 1.0, axis=1)) + 1
    records = []
    for after in transitions:
        before = after - 1
        record = {
            "before_step": int(result["step"][before]),
            "before_time_s": float(result["time_s"][before]),
            "after_step": int(result["step"][after]),
            "after_time_s": float(result["time_s"][after]),
            "vortices": [],
        }
        for vortex in range(2):
            old_xy = raw[before, vortex]
            new_xy = raw[after, vortex]
            old_ix = int(np.argmin(np.abs(x - old_xy[0])))
            old_iy = int(np.argmin(np.abs(y - old_xy[1])))
            new_ix = int(np.argmin(np.abs(x - new_xy[0])))
            new_iy = int(np.argmin(np.abs(y - new_xy[1])))
            before_field = vorticity[before]
            after_field = vorticity[after]
            grid_delta = (
                min(abs(new_ix - old_ix), len(x) - abs(new_ix - old_ix)),
                min(abs(new_iy - old_iy), len(y) - abs(new_iy - old_iy)),
            )
            values = {
                "old_location_before": float(before_field[old_iy, old_ix]),
                "new_location_before": float(before_field[new_iy, new_ix]),
                "old_location_after": float(after_field[old_iy, old_ix]),
                "new_location_after": float(after_field[new_iy, new_ix]),
            }
            record["vortices"].append(
                {
                    "vortex": vortex + 1,
                    "old_raw_xy_km": old_xy.tolist(),
                    "new_raw_xy_km": new_xy.tolist(),
                    "raw_step_motion_km": float(motion[before, vortex]),
                    "periodic_grid_index_delta_xy": list(grid_delta),
                    "adjacent_grid_locations": bool(max(grid_delta) == 1),
                    "eight_neighbor_local_maximum_status": {
                        "old_before": is_eight_neighbor_maximum(
                            before_field, old_iy, old_ix
                        ),
                        "new_before": is_eight_neighbor_maximum(
                            before_field, new_iy, new_ix
                        ),
                        "old_after": is_eight_neighbor_maximum(
                            after_field, old_iy, old_ix
                        ),
                        "new_after": is_eight_neighbor_maximum(
                            after_field, new_iy, new_ix
                        ),
                    },
                    "competing_vorticity_values_1e-5_s-1": values,
                    "relative_winner_gap_before": abs(
                        values["old_location_before"] - values["new_location_before"]
                    )
                    / max(values["old_location_before"], values["new_location_before"]),
                    "relative_winner_gap_after": abs(
                        values["old_location_after"] - values["new_location_after"]
                    )
                    / max(values["old_location_after"], values["new_location_after"]),
                    "continuous_centroid_step_motion_km": float(
                        np.linalg.norm(
                            min_image(
                                result["centroid_xy"][after, vortex]
                                - result["centroid_xy"][before, vortex]
                            )
                        )
                    ),
                    "threshold_centroid_step_motion_km": float(
                        np.linalg.norm(
                            min_image(
                                result["threshold_centroid_xy"][after, vortex]
                                - result["threshold_centroid_xy"][before, vortex]
                            )
                        )
                    ),
                }
            )
        records.append(record)
    return records


def sensitivity_audit(vorticity, x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    values = {}
    for radius in (375.0, 500.0, 600.0):
        xy, _, _, _ = continuous_centroid_track(
            vorticity, x_grid, y_grid, radius, None
        )
        values[f"all_positive_radius_{int(radius)}km"] = summarize_geometry(
            xy, derived_track(xy)
        )
    for fraction in (0.3, 0.5, 0.7):
        xy, _, _, _ = continuous_centroid_track(
            vorticity, x_grid, y_grid, CENTROID_RADIUS_KM, fraction
        )
        values[f"radius_{int(CENTROID_RADIUS_KM)}km_threshold_{fraction:.1f}"] = (
            summarize_geometry(xy, derived_track(xy))
        )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--data-output", required=True, type=Path)
    parser.add_argument("--figure-output", required=True, type=Path)
    args = parser.parse_args()
    args.data_output.mkdir(parents=True, exist_ok=True)
    args.figure_output.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache) as loaded:
        step = loaded["step"].astype(np.int64)
        time_s = loaded["time_s"].astype(np.float64)
        x = loaded["x_km"].astype(np.float64)
        y = loaded["y_km"].astype(np.float64)
        vorticity = loaded["relative_vorticity_1e5_s-1"]
    tracked = track(vorticity, x, y)
    result = {"step": step, "time_s": time_s, **tracked}
    columns = table_columns(result)

    csv_path = args.data_output / "test2b_vortex_core_tracks.csv"
    npz_path = args.data_output / "test2b_vortex_core_tracks.npz"
    write_csv(csv_path, columns)
    np.savez_compressed(npz_path, **columns)
    png, pdf = make_figure(result, args.figure_output)

    smooth = result["centroid"]
    raw = result["raw"]
    maxima = {
        "raw": summarize_geometry(result["raw_xy"], raw),
        "centroid": summarize_geometry(result["centroid_xy"], smooth),
        "threshold_centroid": summarize_geometry(
            result["threshold_centroid_xy"], result["threshold_centroid"]
        ),
    }
    switch_audit = raw_switch_audit(result, vorticity, x, y)
    sensitivity = sensitivity_audit(vorticity, x, y)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    script = Path(__file__).resolve()
    metadata = {
        "format_version": 1,
        "case": "test2b",
        "method": {
            "input": "cached relative-vorticity maps only; no Firedrake replay",
            "identity": "maximum inside a 750 km periodic association disk centered on the previous continuous positive-vorticity centroid",
            "centroid": "positive-vorticity-weighted periodic centroid within a 600 km disk centered on the preceding centroid; the two disks remain disjoint and the definition never recenters on a gridpoint winner",
            "threshold_centroid": "independent sensitivity track using the same preceding-center 600 km disk but retaining points at or above 0.5 times that vortex's local maximum",
            "domain_km": DOMAIN_KM,
            "vortex_width_km": VORTEX_WIDTH_KM,
            "nominal_initial_centers_km": NOMINAL_CENTERS_KM.tolist(),
            "association_radius_km": ASSOCIATION_RADIUS_KM,
            "centroid_radius_km": CENTROID_RADIUS_KM,
            "threshold_fraction": THRESHOLD_FRACTION,
        },
        "source_truth": summary["source_truth"],
        "source_cache": {"path": str(args.cache.resolve()), "sha256": sha256(args.cache)},
        "states": {"count": len(step), "first": int(step[0]), "last": int(step[-1]), "cadence_s": float(time_s[1] - time_s[0])},
        "outputs": [
            {"path": str(csv_path.resolve()), "sha256": sha256(csv_path)},
            {"path": str(npz_path.resolve()), "sha256": sha256(npz_path)},
            {"path": str(png.resolve()), "sha256": sha256(png)},
            {"path": str(pdf.resolve()), "sha256": sha256(pdf)},
        ],
        "columns": list(columns),
        "units": {
            "coordinates_displacements_separation": "km",
            "orientation": "degrees, temporally unwrapped",
            "zeta": "1e-5 s-1",
            "fractions": "1",
        },
        "maxima": maxima,
        "raw_maximum_switch_audit": switch_audit,
        "sensitivity": sensitivity,
        "endpoints": {
            "raw_initial_xy_km": result["raw_xy"][0].tolist(),
            "raw_final_xy_km": result["raw_xy"][-1].tolist(),
            "centroid_initial_xy_km": result["centroid_xy"][0].tolist(),
            "centroid_final_xy_km": result["centroid_xy"][-1].tolist(),
            "threshold_centroid_initial_xy_km": result["threshold_centroid_xy"][0].tolist(),
            "threshold_centroid_final_xy_km": result["threshold_centroid_xy"][-1].tolist(),
            "zeta_maximum_ranges_1e-5_s-1": [
                [float(result["zeta_max"][:, vortex].min()), float(result["zeta_max"][:, vortex].max())]
                for vortex in range(2)
            ],
        },
        "script": str(script),
        "script_sha256": sha256(script),
    }
    metadata_path = args.data_output / "test2b_vortex_core_tracks.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure_sidecar = {
        "format_version": 1,
        "figure_id": "figure5",
        "outputs": metadata["outputs"][2:],
        "source_track_metadata": str(metadata_path.resolve()),
        "source_truth": summary["source_truth"],
        "source_cache": metadata["source_cache"],
        "state_indices": [int(value) for value in step],
        "variables_units": {
            "continuous_centroid_local_dx_dy": "km",
            "periodic_displacement": "km",
            "pair_separation": "km",
            "pair_orientation": "degrees, temporally unwrapped",
        },
        "script": str(script),
        "script_sha256": sha256(script),
    }
    (args.figure_output / "figure5.json").write_text(
        json.dumps(figure_sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(maxima["centroid"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
