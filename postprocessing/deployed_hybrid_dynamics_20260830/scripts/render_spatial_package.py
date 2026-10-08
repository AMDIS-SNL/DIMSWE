#!/usr/bin/env python3
"""Render common-scale event galleries and cache-only deployed movies."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dimswe-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import numpy as np
from PIL import Image

from portable_paths import PACKAGE_ROOT as ROOT, TRUTH_MAP_CACHE

TRUTH_CACHE = TRUTH_MAP_CACHE
LIMITS_PATH = ROOT / "COMMON_VISUAL_LIMITS.json"
EVENT_STEPS = (0, 50, 61, 120, 160)
EVENT_ROLES = (
    "initial",
    "truth pre-rain reference",
    "truth sustained-rain reference",
    "truth peak integrated rain-production reference",
    "final",
)
METHOD_LABELS = {"M1Y": "M1-Y", "H1": "H1", "H2": "H2", "H5": "H5"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_limits():
    return json.loads(LIMITS_PATH.read_text(encoding="utf-8"))


def make_norm(definition):
    if definition["normalization"] == "TwoSlopeNorm":
        return TwoSlopeNorm(
            vmin=definition["vmin"],
            vcenter=definition["vcenter"],
            vmax=definition["vmax"],
        )
    if definition["normalization"] == "SymLogNorm":
        return SymLogNorm(
            linthresh=definition["linthresh"],
            linscale=definition["linscale"],
            vmin=definition["vmin"],
            vmax=definition["vmax"],
            base=definition["base"],
        )
    raise ValueError(f"unknown normalization {definition['normalization']}")


def ticks(variable):
    return {
        "relative_vorticity_1e5_s-1": [-16, -8, 0, 8, 16],
        "supersaturation_percent": [-6, -3, 0, 3, 6],
        "specific_cloud_g_kg-1": [-0.03, 0, 0.05, 0.10],
        "specific_rain_ug_kg-1": [-75, 0, 50, 100, 140],
        "A_g_kg-1_h-1": [-3, -1, -0.1, -0.01, 0, 0.01, 0.1, 0.4],
        "R_ug_kg-1_h-1": [-20, -10, -1, -0.1, 0, 0.1, 1, 10, 100],
    }[variable]


def source_for(representation=None, method=None):
    if representation is None:
        return TRUTH_CACHE, "truth", "DoubleVortex truth"
    source = ROOT / "data" / f"rep{representation}_{method}_maps.npz"
    return (
        source,
        f"rep{representation}_{method}",
        f"Representation {representation} — {METHOD_LABELS[method]}",
    )


def load_data(source: Path):
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def render_gallery(representation=None, method=None):
    source, stem_identity, identifier = source_for(representation, method)
    limits = load_limits()
    data = load_data(source)
    if not np.array_equal(data["step"], np.arange(161)):
        raise RuntimeError("gallery source step axis changed")

    variables = (
        "supersaturation_percent",
        "specific_cloud_g_kg-1",
        "specific_rain_ug_kg-1",
        "R_ug_kg-1_h-1",
    )
    indices = [int(np.flatnonzero(data["step"] == step)[0]) for step in EVENT_STEPS]
    if representation is None:
        output_directory = ROOT / "figures/truth"
        output_stem = "gallery_truth_reference_common_scale"
    else:
        output_directory = ROOT / f"figures/representation_{representation}"
        output_stem = f"gallery_rep{representation}_{method}"
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf = output_directory / f"{output_stem}.pdf"
    png = output_directory / f"{output_stem}.png"
    sidecar = output_directory / f"{output_stem}.json"
    if pdf.exists() or png.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite gallery {output_stem}")

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        4,
        5,
        figsize=(11.35, 8.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    extent = (0.0, 5000.0, 0.0, 5000.0)
    contour_levels = limits["relative_vorticity_contour_levels_1e5_s-1"]
    row_images = []
    for row, variable in enumerate(variables):
        definition = limits["variables"][variable]
        norm = make_norm(definition)
        images = []
        for column, index in enumerate(indices):
            axis = axes[row, column]
            image = axis.imshow(
                data[variable][index],
                origin="lower",
                extent=extent,
                cmap=definition["cmap"],
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
            images.append(image)
            axis.contour(
                data["x_km"],
                data["y_km"],
                data["relative_vorticity_1e5_s-1"][index],
                levels=contour_levels,
                colors="0.2",
                linewidths=0.28,
                alpha=0.42,
            )
            axis.set_aspect("equal")
            if row == 0:
                axis.set_title(f"{EVENT_STEPS[column] * 100:d} s")
            if column == 0:
                axis.set_ylabel("y (km)")
            if row == 3:
                axis.set_xlabel("x (km)")
            axis.set_xticks((0, 2500, 5000))
            axis.set_yticks((0, 2500, 5000))
        row_images.append(images[-1])
        colorbar = fig.colorbar(
            images[-1], ax=axes[row, :], pad=0.010, shrink=0.92, aspect=26
        )
        colorbar.set_label(definition["label"])
        colorbar.set_ticks(ticks(variable))
        colorbar.ax.tick_params(labelsize=8.5)
    fig.suptitle(identifier, fontsize=12.2, y=1.015)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "complete",
        "figure": identifier,
        "source_cache": str(source),
        "source_cache_sha256": digest(source),
        "pdf": str(pdf),
        "pdf_sha256": digest(pdf),
        "png": str(png),
        "png_sha256": digest(png),
        "event_steps": list(EVENT_STEPS),
        "event_times_s": [100.0 * step for step in EVENT_STEPS],
        "event_roles_are_truth_defined": True,
        "event_roles": list(EVENT_ROLES),
        "variables": list(variables),
        "representation_c_rate_label": (
            "effective R from accepted physical two-rate projection"
            if representation == "C" else None
        ),
        "vorticity_overlay": {
            "source": "same deployed model boundary state"
            if representation is not None else "truth boundary state",
            "variable": "relative_vorticity_1e5_s-1",
            "levels": contour_levels,
        },
        "common_visual_limits": str(LIMITS_PATH),
        "common_visual_limits_sha256": digest(LIMITS_PATH),
        "model_specific_autoscaling": False,
        "negative_values_clipped": False,
        "rendering": "cache only",
    }
    write_json(sidecar, metadata)
    print(f"rendered {output_stem}", flush=True)


def render_movie(representation: str, method: str, fps: int = 10):
    source, stem_identity, identifier = source_for(representation, method)
    limits = load_limits()
    data = load_data(source)
    output_directory = ROOT / f"movies/representation_{representation}"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = f"movie_rep{representation}_{method}"
    gif = output_directory / f"{output_stem}.gif"
    sidecar = output_directory / f"{output_stem}.json"
    if gif.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite movie {output_stem}")

    variables = (
        "relative_vorticity_1e5_s-1",
        "supersaturation_percent",
        "specific_cloud_g_kg-1",
        "specific_rain_ug_kg-1",
        "A_g_kg-1_h-1",
        "R_ug_kg-1_h-1",
    )
    panel_titles = (
        "relative vorticity",
        "saturation departure",
        "specific cloud water",
        "specific rain water",
        "effective A" if representation == "C" else "phase-change rate A",
        "effective R" if representation == "C" else "rain-production rate R",
    )
    plt.rcParams.update(
        {
            "font.size": 9.3,
            "axes.titlesize": 10.4,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.7), constrained_layout=True)
    extent = (0.0, 5000.0, 0.0, 5000.0)
    images = []
    for axis, variable, title in zip(axes.reshape(-1), variables, panel_titles):
        definition = limits["variables"][variable]
        image = axis.imshow(
            data[variable][0],
            origin="lower",
            extent=extent,
            cmap=definition["cmap"],
            norm=make_norm(definition),
            interpolation="nearest",
            animated=False,
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")
        axis.set_xticks((0, 2500, 5000))
        axis.set_yticks((0, 2500, 5000))
        axis.set_aspect("equal")
        colorbar = fig.colorbar(image, ax=axis, pad=0.018, shrink=0.86, aspect=22)
        colorbar.set_label(definition["label"], fontsize=8.2)
        colorbar.set_ticks(ticks(variable))
        colorbar.ax.tick_params(labelsize=7.0)
    time_text = fig.text(
        0.98,
        0.995,
        "t = 0 s",
        ha="right",
        va="top",
        fontsize=10.4,
    )
    fig.suptitle(identifier, fontsize=11.5, x=0.02, ha="left", y=0.995)

    frames = []
    for frame in range(161):
        for image, variable in zip(images, variables):
            image.set_data(data[variable][frame])
        time_text.set_text(f"t = {int(data['time_s'][frame]):d} s")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(
            Image.fromarray(rgba).convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=256
            )
        )
        if frame % 20 == 0 or frame == 160:
            print(
                f"{output_stem}: rendered frame {frame}/160",
                flush=True,
            )
    plt.close(fig)
    temporary = gif.with_suffix(".gif.tmp")
    frames[0].save(
        temporary,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=False,
        disposal=2,
    )
    temporary.replace(gif)
    del frames

    with Image.open(gif) as image:
        actual_frames = int(getattr(image, "n_frames", 1))
        dimensions = [int(image.width), int(image.height)]
        duration_ms = int(image.info.get("duration", 0))
    if actual_frames != 161 or duration_ms != int(round(1000 / fps)):
        raise RuntimeError(f"invalid rendered movie {gif}")
    metadata = {
        "status": "complete",
        "figure": identifier,
        "format": "GIF",
        "mp4_not_rendered_reason": "ffmpeg unavailable in execution environment",
        "source_cache": str(source),
        "source_cache_sha256": digest(source),
        "movie": str(gif),
        "movie_sha256": digest(gif),
        "frame_count": actual_frames,
        "fps": fps,
        "frame_duration_ms": duration_ms,
        "pixel_dimensions": dimensions,
        "steps": [0, 160],
        "times_s": [0.0, 16000.0],
        "variables": list(variables),
        "panel_titles": list(panel_titles),
        "representation_c_rate_labels": (
            {"A": "effective A", "R": "effective R"}
            if representation == "C" else None
        ),
        "common_visual_limits": str(LIMITS_PATH),
        "common_visual_limits_sha256": digest(LIMITS_PATH),
        "model_specific_autoscaling": False,
        "negative_values_clipped": False,
        "rendering": "cache only",
    }
    write_json(sidecar, metadata)
    print(f"rendered {output_stem}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("galleries", "movie"), required=True)
    parser.add_argument("--representation", choices=tuple("ABC"))
    parser.add_argument("--method", choices=("M1Y", "H1", "H2", "H5"))
    args = parser.parse_args(argv)
    if args.kind == "galleries":
        if args.representation is not None or args.method is not None:
            parser.error("galleries renders the complete truth/model set")
        render_gallery()
        for representation in "ABC":
            for method in ("M1Y", "H1", "H2", "H5"):
                render_gallery(representation, method)
        return
    if args.representation is None or args.method is None:
        parser.error("movie requires --representation and --method")
    render_movie(args.representation, args.method)


if __name__ == "__main__":
    main()
