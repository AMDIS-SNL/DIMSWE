#!/usr/bin/env python3
"""Build the Track 1 file-disposition table from the frozen evidence manifest.

This script is deliberately path-based: it does not inspect or mutate evidence
files.  The rules are conservative.  Ambiguous scientific outputs remain in
place as local archaeological evidence instead of being deleted.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CANONICAL_RESULT_ROOTS = (
    "external-results/test2a/fair-longfit/",
    "external-results/test2a/fiml-sparse-endpoint-h2-h5/",
    "external-results/test2a/horizon-curriculum-h1-h2-h5/",
    "external-results/test2a/m1-to-m2-finetune/",
    "external-results/test2a/problem-b/",
    "external-results/test2b-rain-active-learning/production/representation-A/",
    "external-results/test2b-rain-active-learning/production/representation-B/",
    "external-results/test2b-rain-active-learning/production/representation-C/",
)

ARCHAEOLOGY_ROOTS = (
    "external-results/test1b0/",
    "external-results/test1b0-stability/",
    "external-results/test1b-production/",
    "external-results/test2-prep/",
    "external-results/test2a-2/",
    "external-results/test2b-preparation/",
    "external-results/test2b-rain-active-case-design/",
)

VERSIONED_RESULT_NAMES = {
    "campaign_manifest.json",
    "fit_result.json",
    "final_parameters.json",
    "final_parameters.npz",
    "representation_a_final_comparison.json",
    "representation_b_final_comparison.json",
    "representation_c_final_comparison.json",
    "fair_longfit_comparison.json",
    "problem_b_comparison.json",
}

VERSIONED_RESULT_SUFFIXES = (
    "_comparison.json",
    "_report.json",
    "_certification.json",
    "_manifest.json",
    "_output_map.json",
    "_summary.json",
)


def is_versioned_result(path: str, size: int) -> bool:
    """Select compact final/reproduction evidence, not every checkpoint."""
    name = Path(path).name
    if name in VERSIONED_RESULT_NAMES or name.endswith(VERSIONED_RESULT_SUFFIXES):
        return True
    if "/comparison/" in path or "/postprocess/" in path:
        return name.endswith((".json", ".md")) and size <= 20_000_000
    return False


def result_class(path: str) -> str:
    name = Path(path).name
    if name.startswith("final_parameters") or "parameter" in name.lower():
        return "CANONICAL_PARAMETER_ARTIFACT"
    return "CANONICAL_RESULT"


def classify(path: str, size: int) -> tuple[str, str, str, str]:
    """Return classification, destination, disposition, and reason."""
    destination = path

    if path == ".DS_Store" or "/.DS_Store" in path or "/__pycache__/" in path or path.endswith(".pyc"):
        return (
            "OBSOLETE",
            "(excluded; frozen copy remains in the authoritative checkout and manifest)",
            "EXCLUDED_GENERATED",
            "OS/Python generated clutter with no scientific role",
        )

    if path.startswith("dimswe/configs/"):
        return "CANONICAL_CONFIG", destination, "RETAINED_VERSIONED", "scientific configuration"
    if path.startswith("tests/"):
        return "CANONICAL_TEST", destination, "RETAINED_VERSIONED", "scientific/regression test"
    if path.startswith("docs/"):
        return "CANONICAL_DOC", destination, "RETAINED_VERSIONED", "scientific report or design record"
    if path.startswith("dimswe/") and path.endswith(".py"):
        return "CANONICAL_SOURCE", destination, "RETAINED_VERSIONED", "scientific implementation"
    if path.startswith("scripts/"):
        return "CANONICAL_SOURCE", destination, "RETAINED_VERSIONED", "driver, postprocessor, or audit utility"

    if path.startswith("external-results/"):
        lower = path.lower()
        name = Path(path).name

        if "superseded" in lower:
            return "SUPERSEDED", destination, "RETAINED_LOCAL_NOT_VERSIONED", "explicitly marked superseded; retained for provenance"
        if "nonscientific-smoke" in lower or "smokes/" in lower:
            return "ARCHAEOLOGY", destination, "RETAINED_LOCAL_NOT_VERSIONED", "smoke/prelaunch evidence, not a canonical result"
        if "/representation-btp/" in lower or "/representation-btpl/" in lower:
            return "UNKNOWN", destination, "RETAINED_LOCAL_NOT_VERSIONED", "partial learned-rain output not established as the B+ campaign"
        if path.startswith(CANONICAL_RESULT_ROOTS):
            klass = result_class(path)
            if is_versioned_result(path, size):
                return klass, destination, "RETAINED_VERSIONED", "compact canonical result/metadata"
            return klass, destination, "RETAINED_LOCAL_NOT_VERSIONED", "canonical evidence retained locally; too bulky or low-level for ordinary Git"
        if path.startswith("external-results/test2b-rain-active-learning/preparation/"):
            if name.endswith(".json"):
                return "CANONICAL_RESULT", destination, "RETAINED_VERSIONED", "compact preparation/certification metadata"
            return "CANONICAL_RESULT", destination, "RETAINED_LOCAL_NOT_VERSIONED", "canonical fixed data/cache retained locally and hash-addressed"
        if path.startswith("external-results/test2b-rain-active-truth/"):
            return "GENERATED_REPRODUCIBLE_RESULT", destination, "RETAINED_LOCAL_NOT_VERSIONED", "truth trajectory needed for reproduction but unsuitable for ordinary Git"
        if path.startswith(ARCHAEOLOGY_ROOTS):
            return "ARCHAEOLOGY", destination, "RETAINED_LOCAL_NOT_VERSIONED", "preparatory or earlier campaign evidence"
        if path.startswith("external-results/test2a/"):
            return "ARCHAEOLOGY", destination, "RETAINED_LOCAL_NOT_VERSIONED", "non-final Test 2A campaign evidence retained for provenance"
        return "UNKNOWN", destination, "RETAINED_LOCAL_NOT_VERSIONED", "scientific relevance not resolved; retained conservatively"

    if path.endswith((".cfg", ".json", ".yaml", ".yml", ".toml")):
        return "CANONICAL_CONFIG", destination, "RETAINED_VERSIONED", "configuration outside standard config directory"
    if path.endswith(".py"):
        return "CANONICAL_SOURCE", destination, "RETAINED_VERSIONED", "source outside standard source directory"
    return "UNKNOWN", destination, "RETAINED_VERSIONED", "unclassified evidence retained conservatively"


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

with args.input.open(newline="", encoding="utf-8") as source, args.output.open(
    "w", newline="", encoding="utf-8"
) as target:
    reader = csv.DictReader(source, delimiter="\t")
    fieldnames = [
        "original_path",
        "original_status",
        "bytes",
        "sha256",
        "classification",
        "collaborator_destination",
        "disposition",
        "reason",
    ]
    writer = csv.DictWriter(
        target, delimiter="\t", fieldnames=fieldnames, lineterminator="\n"
    )
    writer.writeheader()
    for row in reader:
        classification, destination, disposition, reason = classify(
            row["path"], int(row["bytes"])
        )
        writer.writerow(
            {
                "original_path": row["path"],
                "original_status": row["status"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "classification": classification,
                "collaborator_destination": destination,
                "disposition": disposition,
                "reason": reason,
            }
        )
