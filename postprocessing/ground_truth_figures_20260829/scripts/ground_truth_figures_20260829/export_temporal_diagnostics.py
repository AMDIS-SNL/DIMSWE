#!/usr/bin/env python3
"""Export compact, machine-readable temporal diagnostics from truth-map caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COLUMNS = (
    "step",
    "time_s",
    "Qc_mass",
    "rain_water_mass",
    "rain_source_mass_rate",
    "specific_Qc_maximum",
    "A_min_s-1",
    "A_max_s-1",
    "A_negative_fraction",
    "A_positive_fraction",
    "R_max_s-1",
    "R_positive_fraction",
    "postprefix_saturation_min",
    "postprefix_saturation_max",
    "saturation_ratio_minimum",
    "saturation_ratio_maximum",
    "physically_meaningful_R_fraction",
    "total_water_mass",
)

UNITS = {
    "step": "1",
    "time_s": "s",
    "Qc_mass": "m3",
    "rain_water_mass": "m3",
    "rain_source_mass_rate": "m3 s-1",
    "specific_Qc_maximum": "1",
    "A_min_s-1": "s-1",
    "A_max_s-1": "s-1",
    "A_negative_fraction": "1",
    "A_positive_fraction": "1",
    "R_max_s-1": "s-1",
    "R_positive_fraction": "1",
    "postprefix_saturation_min": "1",
    "postprefix_saturation_max": "1",
    "saturation_ratio_minimum": "1",
    "saturation_ratio_maximum": "1",
    "physically_meaningful_R_fraction": "1",
    "total_water_mass": "m3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(case: str, cache_path: Path, summary_path: Path, output: Path) -> None:
    with np.load(cache_path) as loaded:
        arrays = {name: loaded[name] for name in COLUMNS}
    row_count = len(arrays["step"])
    csv_path = output / f"{case}_temporal_diagnostics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in range(row_count):
            writer.writerow(
                int(arrays[name][row]) if name == "step" else format(float(arrays[name][row]), ".17g")
                for name in COLUMNS
            )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    script = Path(__file__).resolve()
    metadata = {
        "format_version": 1,
        "case": case,
        "csv": {"path": str(csv_path.resolve()), "sha256": sha256(csv_path)},
        "columns": list(COLUMNS),
        "units": UNITS,
        "row_count": row_count,
        "state_indices": [int(value) for value in arrays["step"]],
        "time_range_s": [float(arrays["time_s"][0]), float(arrays["time_s"][-1])],
        "truth_cadence_s": float(arrays["time_s"][1] - arrays["time_s"][0]),
        "source_truth": summary["source_truth"],
        "source_cache": {"path": str(cache_path.resolve()), "sha256": sha256(cache_path)},
        "script": str(script),
        "script_sha256": sha256(script),
        "timing_note": "A/R extrema and fractions use the exact post-prefix state entering the following 100 s moist Euler child.",
    }
    metadata_path = output / f"{case}_temporal_diagnostics.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test2a-cache", required=True, type=Path)
    parser.add_argument("--test2b-cache", required=True, type=Path)
    parser.add_argument("--test2a-summary", required=True, type=Path)
    parser.add_argument("--test2b-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export("test2a", args.test2a_cache, args.test2a_summary, args.output)
    export("test2b", args.test2b_cache, args.test2b_summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
