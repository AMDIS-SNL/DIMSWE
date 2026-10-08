#!/usr/bin/env python3
"""Render all-state Test 2A and Test 2B truth frames and animated GIFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def regime_2b(step: int) -> str:
    special = {
        0: "initial supersaturation",
        50: "last pre-rain state",
        51: "first certifiable rain production",
        61: "sustained-rain certification",
        89: "peak local cloud/rain rate",
        120: "mature rain; peak integrated production",
        160: "final state",
    }
    if step in special:
        return special[step]
    if step <= 50:
        return "condensation and cloud accumulation (pre-rain)"
    if step <= 60:
        return "rain onset"
    return "sustained raining state"


def regime_2a(step: int) -> str:
    if step == 0:
        return "analytically saturated; discrete projection imbalance"
    if step == 85:
        return "peak local cloud water"
    if step == 160:
        return "final non-raining state"
    return "reversible condensation/evaporation; no rain"


def configure_axes(axes):
    for axis in np.asarray(axes).reshape(-1):
        axis.set_xlim(0, 5000)
        axis.set_ylim(0, 5000)
        axis.set_aspect("equal")
        axis.set_xticks((0, 2500, 5000))
        axis.set_yticks((0, 2500, 5000))
        axis.tick_params(labelsize=7)


def render_test2b(data, frame_dir: Path, fps: int):
    frame_dir.mkdir(parents=True, exist_ok=True)
    extent = (0, 5000, 0, 5000)
    specs = (
        ("relative_vorticity_1e5_s-1", r"relative vorticity ($10^{-5}$ s$^{-1}$)", "RdBu_r", TwoSlopeNorm(vmin=-17, vcenter=0, vmax=17), False),
        ("supersaturation_percent", r"$100(q_v/q_{sat}-1)$ (%)", "RdBu_r", TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=6.1), False),
        ("specific_cloud_g_kg-1", r"$q_c$ (g kg$^{-1}$)", "Blues", None, True),
        ("specific_rain_ug_kg-1", r"$q_r$ ($\mu$g kg$^{-1}$)", "PuRd", None, True),
        ("A_g_kg-1_h-1", r"$A$ (g kg$^{-1}$ h$^{-1}$)", "RdBu_r", SymLogNorm(linthresh=0.002, linscale=0.7, vmin=-3.1, vmax=0.1, base=10), False),
        ("R_ug_kg-1_h-1", r"$R$ ($\mu$g kg$^{-1}$ h$^{-1}$)", "magma", None, False),
    )
    limits = {
        "relative_vorticity_1e5_s-1": [-17.0, 17.0],
        "supersaturation_percent": [-2.5, 6.1],
        "specific_cloud_g_kg-1": [0.0, 0.11],
        "specific_rain_ug_kg-1": [0.0, 130.0],
        "A_g_kg-1_h-1": [-3.1, 0.1],
        "R_ug_kg-1_h-1": [0.0, 180.0],
    }
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.45), constrained_layout=True)
    configure_axes(axes)
    images = []
    for axis, (name, label, cmap, norm, clip) in zip(axes.reshape(-1), specs):
        values = data[name][0]
        if clip:
            values = np.maximum(values, 0.0)
        kwargs = {"norm": norm} if norm is not None else {
            "vmin": limits[name][0], "vmax": limits[name][1]
        }
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap, **kwargs)
        fig.colorbar(image, ax=axis, label=label, shrink=0.76, pad=0.02)
        axis.set_title(label, fontsize=8)
        images.append(image)
    for axis in axes[:, 0]:
        axis.set_ylabel("y (km)", fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("x (km)", fontsize=8)
    title = fig.suptitle("", fontsize=11)
    note = fig.text(
        0.5, 0.002,
        r"state panels: saved at $t$; $A,R$: exact post-prefix moist tendency over the following 100 s child",
        ha="center", va="bottom", fontsize=7,
    )
    del note
    frame_paths = []
    for frame, step in enumerate(data["step"]):
        for image, (name, _, _, _, clip) in zip(images, specs):
            values = data[name][frame]
            image.set_data(np.maximum(values, 0.0) if clip else values)
        title.set_text(
            f"DIMSWE Test 2B truth — t = {data['time_s'][frame] / 3600:.2f} h "
            f"(step {int(step)}) — {regime_2b(int(step))}"
        )
        path = frame_dir / f"frame_{int(step):08d}.png"
        fig.savefig(path, dpi=110, metadata={"Software": "DIMSWE truth movie renderer"})
        frame_paths.append(path)
        if frame % 20 == 0 or frame == len(data["step"]) - 1:
            print(f"Test 2B frames {frame + 1}/{len(data['step'])}", flush=True)
    plt.close(fig)
    return frame_paths, limits, [item[0] for item in specs]


def render_test2a(data, frame_dir: Path, fps: int):
    frame_dir.mkdir(parents=True, exist_ok=True)
    extent = (0, 5000, 0, 5000)
    specs = (
        ("relative_vorticity_1e5_s-1", r"relative vorticity ($10^{-5}$ s$^{-1}$)", "RdBu_r", TwoSlopeNorm(vmin=-17, vcenter=0, vmax=17), False),
        ("supersaturation_percent", r"$100(q_v/q_{sat}-1)$ (%)", "RdBu_r", TwoSlopeNorm(vmin=-7, vcenter=0, vmax=1), False),
        ("specific_cloud_g_kg-1", r"$q_c$ (g kg$^{-1}$)", "Blues", None, True),
        ("A_g_kg-1_h-1", r"$A$ (g kg$^{-1}$ h$^{-1}$)", "RdBu_r", TwoSlopeNorm(vmin=-0.24, vcenter=0, vmax=0.10), False),
    )
    limits = {
        "relative_vorticity_1e5_s-1": [-17.0, 17.0],
        "supersaturation_percent": [-7.0, 1.0],
        "specific_cloud_g_kg-1": [0.0, 0.032],
        "A_g_kg-1_h-1": [-0.24, 0.10],
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.55), constrained_layout=True)
    configure_axes(axes)
    images = []
    for axis, (name, label, cmap, norm, clip) in zip(axes.reshape(-1), specs):
        values = data[name][0]
        if clip:
            values = np.maximum(values, 0.0)
        kwargs = {"norm": norm} if norm is not None else {
            "vmin": limits[name][0], "vmax": limits[name][1]
        }
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap, **kwargs)
        fig.colorbar(image, ax=axis, label=label, shrink=0.78, pad=0.02)
        axis.set_title(label, fontsize=8)
        images.append(image)
    for axis in axes[:, 0]:
        axis.set_ylabel("y (km)", fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("x (km)", fontsize=8)
    title = fig.suptitle("", fontsize=10)
    fig.text(
        0.5, 0.002,
        r"state panels: saved at $t$; $A$: exact post-prefix moist tendency over the following 100 s child",
        ha="center", va="bottom", fontsize=7,
    )
    frame_paths = []
    for frame, step in enumerate(data["step"]):
        for image, (name, _, _, _, clip) in zip(images, specs):
            values = data[name][frame]
            image.set_data(np.maximum(values, 0.0) if clip else values)
        title.set_text(
            f"DIMSWE Test 2A truth — t = {data['time_s'][frame] / 3600:.2f} h "
            f"(step {int(step)}) — {regime_2a(int(step))}"
        )
        path = frame_dir / f"frame_{int(step):08d}.png"
        fig.savefig(path, dpi=110, metadata={"Software": "DIMSWE truth movie renderer"})
        frame_paths.append(path)
        if frame % 20 == 0 or frame == len(data["step"]) - 1:
            print(f"Test 2A frames {frame + 1}/{len(data['step'])}", flush=True)
    plt.close(fig)
    return frame_paths, limits, [item[0] for item in specs]


def make_gif(frame_paths, destination: Path, fps: int):
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for path in frame_paths]
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )
    for image in images:
        image.close()


def render_one(case, cache_path, summary_path, frame_dir, movie_path, metadata_path, fps):
    with np.load(cache_path) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if case == "test2b":
        frames, limits, variables = render_test2b(data, frame_dir, fps)
        behavior = (
            "Initial domain-wide condensation rapidly creates cloud water. The vortices "
            "then wrap saturation and cloud anomalies into two spiral arms; rain first "
            "appears on narrow cloud-threshold arcs, expands through the sustained regime, "
            "and accumulates downstream along the rotating arms."
        )
    else:
        frames, limits, variables = render_test2a(data, frame_dir, fps)
        behavior = (
            "The coarse saturated case develops paired condensation and evaporation "
            "patterns as the vortices reorganize moisture. Cloud water remains below the "
            "rain threshold throughout, so no rain-water or rain-production panels are needed."
        )
    make_gif(frames, movie_path, fps)
    script = Path(__file__).resolve()
    frame_inventory = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in frames
    ]
    metadata = {
        "format_version": 1,
        "case": case,
        "movie": {"path": str(movie_path.resolve()), "format": "GIF", "sha256": sha256(movie_path)},
        "frame_count": len(frames),
        "state_indices": [int(value) for value in data["step"]],
        "times_s": [float(value) for value in data["time_s"]],
        "truth_cadence_s": float(data["time_s"][1] - data["time_s"][0]),
        "playback_fps": fps,
        "playback_duration_s": len(frames) / fps,
        "source_truth": summary["source_truth"],
        "source_cache": {"path": str(cache_path.resolve()), "sha256": sha256(cache_path)},
        "variables": variables,
        "color_limits": limits,
        "coordinate_units": "km",
        "state_rate_timing": "state maps at saved t; A/R at exact post-prefix input to the following moist Euler child",
        "display_clipping": "negative numerical Qc/Qr undershoots clipped to zero; raw cache retained",
        "frames": frame_inventory,
        "script": str(script),
        "script_sha256": sha256(script),
        "behavior_visible": behavior,
        "optional_mp4_command": (
            f"ffmpeg -framerate {fps} -i {frame_dir.resolve()}/frame_%08d.png "
            f"-c:v libx264 -pix_fmt yuv420p {movie_path.with_suffix('.mp4').resolve()}"
        ),
        "renderer_note": "Pillow was available; ffmpeg was not present, so GIF is the native rendered movie.",
    }
    write_json(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test2a-cache", required=True, type=Path)
    parser.add_argument("--test2b-cache", required=True, type=Path)
    parser.add_argument("--test2a-summary", required=True, type=Path)
    parser.add_argument("--test2b-summary", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--movies-root", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    for case, cache, summary in (
        ("test2b", args.test2b_cache, args.test2b_summary),
        ("test2a", args.test2a_cache, args.test2a_summary),
    ):
        render_one(
            case,
            cache,
            summary,
            args.frames_root / case,
            args.movies_root / f"{case}_truth_evolution.gif",
            args.movies_root / f"{case}_truth_evolution.json",
            args.fps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
