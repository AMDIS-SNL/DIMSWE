#!/usr/bin/env python3
"""Copy accepted CSV field text exactly into filtered main-figure sidecars."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "figures/main"
SUPPLEMENT = ROOT / "figures/supplement"
OUTPUT = ROOT / "FILTERED_CSV_TEXT_PARITY.json"
MODELS = {"M1-Y", "H1", "H2", "H5"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def exact_filter(source: Path, destination: Path) -> int:
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames or "model_label" not in fieldnames:
            raise RuntimeError(f"missing model_label: {source}")
        rows = [row for row in reader if row["model_label"] in MODELS]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def update_sidecar(stem: str, row_count: int) -> None:
    path = MAIN / f"{stem}.json"
    payload = json.loads(path.read_text())
    csv_record = record(MAIN / f"{stem}.csv")
    payload["files"]["csv"].update(csv_record)
    payload["files"]["csv"]["rows"] = row_count
    payload["rendering"]["csv_field_text_copied_exactly_from_accepted_source"] = True
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    tasks = [
        (
            "ML4_main_deployed_physical_diagnostics",
            SUPPLEMENT / "ML5_deployed_physical_diagnostics_test2b.csv",
        )
    ] + [
        (
            f"ML5{representation}_main_global_trajectories",
            SUPPLEMENT / f"ML6_global_trajectories_representation_{representation}_all_models.csv",
        )
        for representation in "ABC"
    ]
    records = []
    for stem, source in tasks:
        destination = MAIN / f"{stem}.csv"
        before = record(destination)
        row_count = exact_filter(source, destination)
        update_sidecar(stem, row_count)
        records.append(
            {
                "figure": stem,
                "source": record(source),
                "before": before,
                "after": record(destination),
                "rows": row_count,
                "operation": "string-preserving row filter; no numeric parsing",
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "remove pandas float-text round-trip drift from plotted CSV sidecars",
                "figures": records,
                "rendered_PDF_PNG_changed": False,
                "scientific_values_changed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"status": "complete", "figures": len(records)}, indent=2))


if __name__ == "__main__":
    main()
