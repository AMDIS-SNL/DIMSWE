#!/usr/bin/env python3
"""Render the objective-consistent Test-2B training/evaluation history figure.

This script reads the completed fixed-array/fixed-map history artifact.  It
does not import DIMSWE, inspect truth states, evaluate checkpoints, or alter
any numerical value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAIN = ROOT / "figures/main"
SOURCE = DATA / "objective_training_evaluation_histories.csv"
SOURCE_SIDECAR = SOURCE.with_suffix(".json")
STEM = "ML2_objective_training_evaluation_history_test2b"
DESTINATION = MAIN / f"{STEM}.csv"
SIDECAR = MAIN / f"{STEM}.json"
BASELINE = ROOT / "OBJECTIVE_HISTORY_FIGURE_BASELINE.json"
OLD_STEM = "ML2_posthoc_direct_history_test2b"

MODELS = ["M1-X", "M1-Y", "M2-X-independent", "warm M2-X", "H1"]
DIRECT_MODELS = MODELS[:2]
DISCRETE_MODELS = MODELS[2:]
MODEL_LABELS = {
    "M1-X": "M1-X",
    "M1-Y": "M1-Y",
    "M2-X-independent": "M2-X (ind.)",
    "warm M2-X": "M2-X (warm)",
    "H1": "H1",
}
MODEL_COLORS = {
    "M1-X": "#1f77b4",
    "M1-Y": "#d62728",
    "M2-X-independent": "#9467bd",
    "warm M2-X": "#8c564b",
    "H1": "#2ca02c",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1,
        "legend.fontsize": 8.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.55,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def transformed_iteration(values) -> np.ndarray:
    return np.log10(np.asarray(values, dtype=float) + 1.0)


def format_iteration_axis(axis) -> None:
    ticks = [0, 1, 10, 100, 1_000, 10_000]
    labels = ["0", "1", "10", "100", "1k", "10k"]
    axis.set_xticks(transformed_iteration(ticks), labels)
    axis.set_xlim(0.0, float(np.log10(10_001.0)))
    axis.margins(x=0)
    axis.tick_params(axis="x", labelbottom=True)
    axis.set_xlabel("iteration")


def line(axis, frame: pd.DataFrame, model: str, role: str) -> None:
    subset = frame[
        frame["model_label"].eq(model) & frame["curve_role"].eq(role)
    ].sort_values("checkpoint_iteration")
    if subset.empty:
        raise RuntimeError(f"missing plotted curve: {model} {role}")
    evaluation = role == "evaluation"
    axis.plot(
        transformed_iteration(subset["checkpoint_iteration"]),
        subset["normalized_objective"],
        color=MODEL_COLORS[model],
        linestyle="--" if evaluation else "-",
        marker="o",
        markerfacecolor="white" if evaluation else MODEL_COLORS[model],
        markeredgewidth=0.9,
        zorder=3 if evaluation else 2,
    )


def establish_baseline() -> None:
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
        for record in baseline["old_main_bundle"].values():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError("common-X baseline changed before archival")
        return
    old_files = {
        suffix: file_record(MAIN / f"{OLD_STEM}.{suffix}")
        for suffix in ["csv", "json", "pdf", "png"]
    }
    payload = {
        "purpose": "Freeze common-X ML-2 bundle before supplementary archival",
        "old_main_bundle": old_files,
        "objective_history_source": file_record(SOURCE),
        "objective_history_source_sidecar": file_record(SOURCE_SIDECAR),
    }
    BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    source_sidecar = json.loads(SOURCE_SIDECAR.read_text())
    if source_sidecar.get("status") != "complete":
        raise RuntimeError("objective histories are not complete")
    if not all(source_sidecar.get("validation_gates", {}).values()):
        raise RuntimeError("objective-history validation gate failed")
    if sha256(SOURCE) != source_sidecar["output"]["sha256"]:
        raise RuntimeError("objective-history CSV hash mismatch")

    establish_baseline()
    shutil.copyfile(SOURCE, DESTINATION)
    if sha256(DESTINATION) != sha256(SOURCE):
        raise RuntimeError("plotted CSV differs from the accepted source")
    data = pd.read_csv(DESTINATION)
    if len(data) != 288 or set(data["model_label"]) != set(MODELS):
        raise RuntimeError("unexpected plotted-data coverage")

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.05), sharex=True)
    for row, representation in enumerate("ABC"):
        panel = data[data["representation"].eq(representation)]
        for column, models in enumerate([DIRECT_MODELS, DISCRETE_MODELS]):
            axis = axes[row, column]
            for model in models:
                line(axis, panel, model, "training")
                line(axis, panel, model, "evaluation")
            axis.set_yscale("log")
            format_iteration_axis(axis)

    axes[0, 0].set_title("Direct regression", pad=10)
    axes[0, 1].set_title("Discrete nonrecursive", pad=10)

    direct_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], marker="o", label=MODEL_LABELS[model])
        for model in DIRECT_MODELS
    ]
    discrete_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], marker="o", label=MODEL_LABELS[model])
        for model in DISCRETE_MODELS
    ]
    style_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", marker="o", label="training"),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            label="evaluation",
        ),
    ]
    fig.legend(
        handles=direct_handles + discrete_handles + style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handletextpad=0.55,
    )
    fig.supylabel("normalized objective", x=0.052)
    for label, y in zip(["Rep. A", "Rep. B", "Rep. C"], [0.805, 0.526, 0.247]):
        fig.text(
            0.012,
            y,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontweight="semibold",
        )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.145, hspace=0.34, wspace=0.30)

    pdf = MAIN / f"{STEM}.pdf"
    png = MAIN / f"{STEM}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": STEM,
            "Subject": "Training and evaluation histories of objective-consistent Test-2B losses",
        },
    )
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Objective-consistent histories for the nonrecursive Test 2B objectives. "
        "Solid curves use the training portion and dashed curves with open markers use "
        "the later evaluation portion; the evaluation curves were calculated from saved "
        "networks after optimization and did not influence training. M1-X and both M2-X "
        "runs are evaluated on timestep-boundary states X*, whereas M1-Y is evaluated on "
        "Y*=P(X*) and H1 uses fixed one-step pairs from Y* to the corresponding next truth "
        "state. Each evaluation curve uses the same normalization, output scaling, weighting, "
        "and normalized-target-energy formula as its fitted objective. Objective "
        "normalizations differ across representations. H2 and H5 are omitted because their "
        "recursive evaluation histories were not computed."
    )
    payload = {
        "figure_id": STEM,
        "draft_caption": caption,
        "scientific_question": (
            "For each nonrecursive training objective, how does the same objective behave "
            "on the training and later evaluation portions of the Test-2B trajectory?"
        ),
        "quantity_kind": "OBJECTIVE-CONSISTENT SAVED-NETWORK EVALUATION",
        "support_classification": (
            "Test 2B training states/windows and temporally later evaluation states/windows; "
            "evaluation values did not influence optimization"
        ),
        "state_contracts": source_sidecar["state_contracts"],
        "denominator_convention": source_sidecar["denominator_convention"],
        "normalization": source_sidecar["normalization"],
        "weighting": source_sidecar["weighting"],
        "validation_gates": source_sidecar["validation_gates"],
        "model_labels": MODELS,
        "representations": ["A", "B", "C"],
        "excluded_models": ["H2", "H5"],
        "excluded_reason": "recursive evaluation histories were not computed",
        "units": {
            "checkpoint_iteration": "iteration",
            "normalized_objective": "dimensionless objective",
        },
        "metric_definitions": {
            "M1-X": "fitted carrier-weighted direct-regression objective at X*",
            "M1-Y": "fitted carrier-weighted direct-regression objective at Y*=P(X*)",
            "M2-X": "fitted fixed source-to-tendency mixed-state objective at X*",
            "H1": "fitted fixed one-step mixed-state objective from Y* to the next truth state",
        },
        "notes": [
            "Only saved iterations are plotted; no interpolation is used.",
            "Training is solid; evaluation is dashed with open markers.",
            "Raw objective values should not be compared across representations without accounting for their distinct target scalings and normalizations.",
        ],
        "source_artifacts": source_sidecar["inputs"]
        + [
            {
                "path": str(SOURCE.resolve()),
                "sha256": sha256(SOURCE),
            },
            {
                "path": str(SOURCE_SIDECAR.resolve()),
                "sha256": sha256(SOURCE_SIDECAR),
            },
        ],
        "files": {
            "csv": {**file_record(DESTINATION), "rows": len(data)},
            "pdf": file_record(pdf),
            "png_300dpi": file_record(png),
        },
        "rendering": {
            "iteration_transform": "log10(iteration+1)",
            "iteration_ticks": [0, 1, 10, 100, 1000, 10000],
            "y_scale": "logarithmic within each panel",
            "date": "2026-08-29",
        },
        "operations": source_sidecar["operations"],
    }
    SIDECAR.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "records": len(data),
                "pdf": str(pdf),
                "png": str(png),
                "csv_sha256_matches_source": sha256(DESTINATION) == sha256(SOURCE),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
