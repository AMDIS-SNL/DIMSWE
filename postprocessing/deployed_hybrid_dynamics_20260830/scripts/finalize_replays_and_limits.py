#!/usr/bin/env python3
"""Validate all replay caches and freeze cross-model visualization limits."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from portable_paths import PACKAGE_ROOT as ROOT, TRUTH_MAP_CACHE

TRUTH_CACHE = TRUTH_MAP_CACHE
VARIABLES = (
    "relative_vorticity_1e5_s-1",
    "supersaturation_percent",
    "specific_cloud_g_kg-1",
    "specific_rain_ug_kg-1",
    "A_g_kg-1_h-1",
    "R_ug_kg-1_h-1",
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    models = [
        (representation, method)
        for representation in "ABC"
        for method in ("M1Y", "H1", "H2", "H5")
    ]
    sources = [("truth", TRUTH_CACHE)]
    validation = []
    for representation, method in models:
        cache = ROOT / "data" / f"rep{representation}_{method}_maps.npz"
        sidecar = cache.with_suffix(".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or not metadata.get(
            "parity_passed"
        ):
            raise RuntimeError(f"unvalidated cache: {cache}")
        cache_hash = digest(cache)
        if cache_hash != metadata["cache_sha256"]:
            raise RuntimeError(f"cache hash mismatch: {cache}")
        with np.load(cache, allow_pickle=False) as archive:
            if not np.array_equal(archive["step"], np.arange(161)):
                raise RuntimeError(f"step axis mismatch: {cache}")
            if not np.array_equal(
                archive["time_s"], np.arange(161, dtype=np.float64) * 100.0
            ):
                raise RuntimeError(f"time axis mismatch: {cache}")
            for variable in VARIABLES:
                values = archive[variable]
                if values.shape != (161, 128, 128):
                    raise RuntimeError(f"shape mismatch {variable}: {cache}")
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"nonfinite {variable}: {cache}")
            if representation == "C":
                residual = archive["source_manifold_residual_normalized"]
                if residual.shape != (161, 128, 128) or not np.all(
                    np.isfinite(residual)
                ):
                    raise RuntimeError(f"invalid C residual: {cache}")
        parity_rows = metadata["parity"]
        validation.append(
            {
                "representation": representation,
                "method": method,
                "cache": str(cache),
                "cache_sha256": cache_hash,
                "frame_count": 161,
                "all_scalar_parity_exact": all(
                    row["exact_array_equal"] for row in parity_rows.values()
                ),
                "maximum_parity_absolute_difference": max(
                    row["maximum_absolute_difference"]
                    for row in parity_rows.values()
                ),
                "maximum_parity_relative_difference": max(
                    row["maximum_relative_difference"]
                    for row in parity_rows.values()
                ),
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            }
        )
        sources.append((f"rep{representation}_{method}", cache))

    extrema = {}
    for variable in VARIABLES:
        minimum = (np.inf, None)
        maximum = (-np.inf, None)
        negative_count = 0
        zero_count = 0
        total_count = 0
        per_source = {}
        for label, source in sources:
            with np.load(source, allow_pickle=False) as archive:
                values = np.asarray(archive[variable])
                local_min = float(np.min(values))
                local_max = float(np.max(values))
                per_source[label] = {
                    "minimum": local_min,
                    "maximum": local_max,
                    "negative_fraction": float(np.mean(values < 0.0)),
                    "zero_fraction": float(np.mean(values == 0.0)),
                }
                if local_min < minimum[0]:
                    minimum = (local_min, label)
                if local_max > maximum[0]:
                    maximum = (local_max, label)
                negative_count += int(np.count_nonzero(values < 0.0))
                zero_count += int(np.count_nonzero(values == 0.0))
                total_count += int(values.size)
        extrema[variable] = {
            "global_minimum": minimum[0],
            "minimum_source": minimum[1],
            "global_maximum": maximum[0],
            "maximum_source": maximum[1],
            "negative_fraction": negative_count / total_count,
            "zero_fraction": zero_count / total_count,
            "sample_count": total_count,
            "per_source": per_source,
        }

    definitions = {
        "relative_vorticity_1e5_s-1": {
            "label": r"relative vorticity ($10^{-5}$ s$^{-1}$)",
            "normalization": "TwoSlopeNorm",
            "vmin": -17.0,
            "vcenter": 0.0,
            "vmax": 17.0,
            "cmap": "RdBu_r",
        },
        "supersaturation_percent": {
            "label": r"$100(q_v/q_{sat}-1)$ (%)",
            "normalization": "TwoSlopeNorm",
            "vmin": -6.6,
            "vcenter": 0.0,
            "vmax": 6.6,
            "cmap": "RdBu_r",
        },
        "specific_cloud_g_kg-1": {
            "label": r"$q_c$ (g kg$^{-1}$)",
            "normalization": "TwoSlopeNorm",
            "vmin": -0.035,
            "vcenter": 0.0,
            "vmax": 0.11,
            "cmap": "BrBG",
        },
        "specific_rain_ug_kg-1": {
            "label": r"$q_r$ ($\mu$g kg$^{-1}$)",
            "normalization": "TwoSlopeNorm",
            "vmin": -80.0,
            "vcenter": 0.0,
            "vmax": 140.0,
            "cmap": "PuOr_r",
        },
        "A_g_kg-1_h-1": {
            "label": r"$A$ (g kg$^{-1}$ h$^{-1}$)",
            "normalization": "SymLogNorm",
            "vmin": -3.1,
            "vmax": 0.42,
            "linthresh": 0.002,
            "linscale": 1.0,
            "base": 10.0,
            "cmap": "RdBu_r",
        },
        "R_ug_kg-1_h-1": {
            "label": r"$R$ ($\mu$g kg$^{-1}$ h$^{-1}$)",
            "normalization": "SymLogNorm",
            "vmin": -22.0,
            "vmax": 190.0,
            "linthresh": 0.1,
            "linscale": 1.0,
            "base": 10.0,
            "cmap": "PuOr_r",
        },
    }
    for variable, definition in definitions.items():
        observed = extrema[variable]
        if not (
            definition["vmin"] <= observed["global_minimum"]
            and definition["vmax"] >= observed["global_maximum"]
        ):
            raise RuntimeError(f"visual bounds clip {variable}")

    c_residual = {}
    for representation, method in models:
        if representation != "C":
            continue
        source = ROOT / "data" / f"repC_{method}_maps.npz"
        with np.load(source, allow_pickle=False) as archive:
            values = archive["source_manifold_residual_normalized"]
            c_residual[method] = {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "RMS": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
            }

    limits = {
        "status": "complete",
        "scope": "truth plus all 12 deployed models at all 161 times",
        "source_count": len(sources),
        "frame_count_per_source": 161,
        "model_specific_autoscaling": False,
        "negative_values_clipped": False,
        "variables": definitions,
        "observed_extrema": extrema,
        "relative_vorticity_contour_levels_1e5_s-1": [
            -8.0,
            -4.0,
            4.0,
            8.0,
            12.0,
        ],
        "representation_c_source_manifold_residual": c_residual,
        "rationale": {
            "linear_two_slope": (
                "vorticity, saturation departure, cloud water, and rain "
                "water fit readable common finite ranges; zero is explicit "
                "and signed undershoots/failures are retained"
            ),
            "symmetric_log": (
                "A and R span both signs and multiple orders near zero; "
                "common SymLogNorm preserves zeros, weak activity, and full "
                "outlier ranges without model-specific rescaling"
            ),
        },
    }
    write_json(ROOT / "COMMON_VISUAL_LIMITS.json", limits)
    write_json(ROOT / "data/REPLAY_VALIDATION.json", {
        "status": "complete",
        "model_count": 12,
        "all_models_passed": all(
            row["all_scalar_parity_exact"] for row in validation
        ),
        "models": validation,
    })
    print("validated 12 caches and froze common visual limits")


if __name__ == "__main__":
    main()
