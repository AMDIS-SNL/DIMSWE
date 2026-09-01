#!/usr/bin/env python3
"""Generate the final main-paper ML figure/table subset from accepted data.

Only rendering, filtering, table formatting, and parsing of already stored
deployment diagnostics occur here.  No DIMSWE module or checkpoint is loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from portable_paths import (
    M1Y_REPOSITORY as M1Y,
    PACKAGE_ROOT,
    REFERENCE_REPOSITORY as AUTH,
)

ROOT = PACKAGE_ROOT
DATA = ROOT / "data"
MAIN = ROOT / "figures/main"
SUPPLEMENT = ROOT / "figures/supplement"
TABLE_ROOT = ROOT / "tables"
TABLE_MAIN = TABLE_ROOT / "main"
TABLE_SUPPLEMENT = TABLE_ROOT / "supplement"
BASELINE = ROOT / "FINAL_MAIN_RESET_BASELINE.json"
EXPECTED_HEAD = "d2f5d66ecb5500aad24eca37280f8a52e22a250f"

MODELS = ["M1-Y", "H1", "H2", "H5"]
MODEL_COLORS = {
    "M1-Y": "#d62728",
    "H1": "#2ca02c",
    "H2": "#ff7f0e",
    "H5": "#17becf",
    "Truth": "#111111",
}
MODEL_LABELS = {"M1-Y": "M1-Y", "H1": "H1", "H2": "H2", "H5": "H5"}
REP_COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
REP_MARKERS = {"A": "o", "B": "s", "C": "^"}

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
        "lines.markersize": 4.2,
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


def establish_baseline() -> None:
    if BASELINE.exists():
        raise FileExistsError(f"refusing to overwrite {BASELINE}")
    old_main = {
        path.name: file_record(path)
        for path in sorted(MAIN.iterdir())
        if path.is_file()
    }
    sources = [
        MAIN / "ML1_optimization_progress_test2b.csv",
        MAIN / "ML2_objective_training_evaluation_history_test2b.csv",
        MAIN / "ML5_deployed_physical_diagnostics_test2b.csv",
        DATA / "final_callsite_y_metrics.csv",
        DATA / "final_callsite_y_metrics.json",
    ] + [
        SUPPLEMENT / f"ML6_global_trajectories_representation_{rep}_all_models.csv"
        for rep in "ABC"
    ]
    tables = {
        path.name: file_record(path)
        for path in sorted(TABLE_ROOT.iterdir())
        if path.is_file()
    }
    payload = {
        "purpose": "Freeze accepted inputs before the final main-paper subset reset",
        "source_head": EXPECTED_HEAD,
        "old_main_figure_bundle": old_main,
        "numerical_sources": [file_record(path) for path in sources],
        "complete_root_tables": tables,
    }
    BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def transformed_iteration(values: Iterable[float]) -> np.ndarray:
    return np.log10(np.asarray(values, dtype=float) + 1.0)


def iteration_axis(axis, maximum: int) -> None:
    ticks = (
        [0, 1, 10, 100, 1000, 10000]
        if maximum == 10000
        else [0, 1, 10, 100, 1000, maximum]
    )
    labels = [f"{value // 1000}k" if value >= 1000 else str(value) for value in ticks]
    axis.set_xticks(transformed_iteration(ticks), labels)
    axis.set_xlim(0.0, float(np.log10(maximum + 1.0)))
    axis.margins(x=0)
    axis.set_xlabel("iteration")


def write_figure_csv(
    stem: str, frame: pd.DataFrame, *, overwrite_task_output: bool = False
) -> Path:
    path = MAIN / f"{stem}.csv"
    if path.exists() and not overwrite_task_output:
        raise FileExistsError(f"refusing to overwrite {path}")
    frame.to_csv(path, index=False)
    return path


def save_figure(
    figure,
    stem: str,
    frame: pd.DataFrame,
    *,
    caption: str,
    question: str,
    quantity_kind: str,
    support: str,
    sources: list[Path],
    notes: list[str],
    representations: list[str] | None = None,
    models: list[str] | None = None,
    state_contracts: dict[str, Any] | None = None,
    overwrite_task_output: bool = False,
) -> None:
    csv_path = write_figure_csv(
        stem, frame, overwrite_task_output=overwrite_task_output
    )
    pdf_path = MAIN / f"{stem}.pdf"
    png_path = MAIN / f"{stem}.png"
    json_path = MAIN / f"{stem}.json"
    for path in [pdf_path, png_path, json_path]:
        if path.exists() and not overwrite_task_output:
            raise FileExistsError(f"refusing to overwrite {path}")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Title": stem, "Subject": question},
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    payload = {
        "figure_id": stem,
        "draft_caption": caption,
        "scientific_question": question,
        "quantity_kind": quantity_kind,
        "support_classification": support,
        "representations": representations or ["A", "B", "C"],
        "model_labels": models or MODELS,
        "state_contracts": state_contracts or {},
        "notes": notes,
        "source_artifacts": [file_record(path) for path in sources],
        "files": {
            "csv": {**file_record(csv_path), "rows": len(frame)},
            "pdf": file_record(pdf_path),
            "png_300dpi": file_record(png_path),
        },
        "rendering": {
            "date": "2026-08-29",
            "numerical_values_recomputed": False,
            "source_values_filtered_without_modification": True,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def add_row_labels(figure) -> None:
    for label, y in zip(["Rep. A", "Rep. B", "Rep. C"], [0.805, 0.526, 0.247]):
        figure.text(
            0.012,
            y,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontweight="semibold",
        )


def figure1() -> None:
    source = SUPPLEMENT / "ML1_optimization_progress_test2b.csv"
    data = pd.read_csv(source)
    plotted = data[data["model_label"].isin(MODELS)].copy()
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 7.05))
    for row, representation in enumerate("ABC"):
        rep = plotted[plotted["representation"].eq(representation)]
        for column, (model, maximum) in enumerate([("M1-Y", 10000), ("H1", 5000)]):
            axis = axes[row, column]
            curve = rep[rep["model_label"].eq(model)].sort_values("checkpoint_iteration")
            axis.plot(
                transformed_iteration(curve["checkpoint_iteration"]),
                curve["value"],
                color=MODEL_COLORS[model],
                marker="o",
            )
            axis.set_yscale("log")
            iteration_axis(axis, maximum)
        axis = axes[row, 2]
        for model, offset in [("H2", -0.055), ("H5", 0.055)]:
            endpoints = rep[rep["model_label"].eq(model)].sort_values("checkpoint_iteration")
            if len(endpoints) != 2:
                raise RuntimeError(f"expected two recursive endpoints: {representation} {model}")
            x = np.asarray([0.0, 1.0]) + offset
            axis.plot(x, endpoints["value"], color=MODEL_COLORS[model], marker="o")
        axis.set_xticks([0, 1], ["initial", "final"])
        axis.set_xlim(-0.25, 1.25)
        axis.set_yscale("log")
        axis.set_xlabel("stage endpoint")

    for axis, title in zip(axes[0], ["M1-Y", "H1 / M2-Y", "H2 and H5"]):
        axis.set_title(title, pad=10)
    figure.supylabel("objective", x=0.052)
    add_row_labels(figure)
    figure.legend(
        handles=[
            Line2D([0], [0], color=MODEL_COLORS[model], marker="o", label=MODEL_LABELS[model])
            for model in MODELS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.125, hspace=0.38, wspace=0.36)
    save_figure(
        figure,
        "ML1_main_optimization",
        plotted,
        caption=(
            "Optimization of the Test 2B training objectives for the four main-paper methods. "
            "M1-Y and H1/M2-Y are shown at saved iterations. H2 and H5 are shown only at "
            "their initial and final stage endpoints; intermediate recursive objective values "
            "were not reconstructed. Objective normalizations differ across representations."
        ),
        question="Did each main-paper training objective decrease under its fixed budget?",
        quantity_kind="TRAINING OBJECTIVE HISTORY OR STORED ENDPOINT",
        support="Test 2B objective-specific training states or windows",
        sources=[source],
        notes=[
            "H2/H5 final endpoints correspond to their 20-iteration stages.",
            "Objective magnitudes are not compared across representations.",
        ],
        overwrite_task_output=True,
    )


def figure2() -> None:
    source = DATA / "objective_training_evaluation_histories.csv"
    data = pd.read_csv(source)
    plotted = data[data["model_label"].isin(["M1-Y", "H1"])].copy()
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 7.05), sharex=False)
    for row, representation in enumerate("ABC"):
        rep = plotted[plotted["representation"].eq(representation)]
        for column, (model, maximum) in enumerate([("M1-Y", 10000), ("H1", 5000)]):
            axis = axes[row, column]
            for role, linestyle, face in [
                ("training", "-", MODEL_COLORS[model]),
                ("evaluation", "--", "white"),
            ]:
                curve = rep[
                    rep["model_label"].eq(model) & rep["curve_role"].eq(role)
                ].sort_values("checkpoint_iteration")
                axis.plot(
                    transformed_iteration(curve["checkpoint_iteration"]),
                    curve["normalized_objective"],
                    color=MODEL_COLORS[model],
                    linestyle=linestyle,
                    marker="o",
                    markerfacecolor=face,
                    markeredgewidth=0.9,
                )
            axis.set_yscale("log")
            iteration_axis(axis, maximum)
    axes[0, 0].set_title("M1-Y", pad=10)
    axes[0, 1].set_title("H1 / M2-Y", pad=10)
    figure.supylabel("objective", x=0.052)
    add_row_labels(figure)
    figure.legend(
        handles=[
            Line2D([0], [0], color="0.25", marker="o", linestyle="-", label="training"),
            Line2D(
                [0], [0], color="0.25", marker="o", markerfacecolor="white",
                linestyle="--", label="evaluation",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.125, hspace=0.38, wspace=0.30)
    save_figure(
        figure,
        "ML2_main_training_evaluation",
        plotted,
        caption=(
            "Training and evaluation histories of the fitted nonrecursive objectives. Solid "
            "curves show training states or windows; dashed curves with open markers show the "
            "later evaluation states or windows. M1-Y is evaluated on truth-derived pre-moist "
            "states Y*=P(X*). H1/M2-Y uses fixed one-step pairs from Y* to the corresponding "
            "next truth state. Evaluation values were calculated from saved networks and did "
            "not influence training."
        ),
        question="Does each fitted objective remain accurate over the later evaluation interval?",
        quantity_kind="OBJECTIVE-CONSISTENT SAVED-NETWORK EVALUATION",
        support="Test 2B training and evaluation states/windows",
        sources=[source, source.with_suffix(".json")],
        notes=[
            "M1-Y: Y*_0..80 and Y*_81..160.",
            "H1: Y*_0..79 to X*_1..80 and Y*_81..159 to X*_82..160.",
            "H2/H5 are excluded because recursive evaluation histories were not computed.",
        ],
        models=["M1-Y", "H1"],
        state_contracts={
            "M1-Y": "Y* at states 0..80 and 81..160",
            "H1": "Y* starts 0..79/81..159 with next truth targets 1..80/82..160",
        },
        overwrite_task_output=True,
    )


def metric_value(data: pd.DataFrame, representation: str, model: str, support: str, quantity: str, metric: str) -> float:
    selected = data[
        data["representation"].eq(representation)
        & data["model_label"].eq(model)
        & data["support"].eq(support)
        & data["quantity"].eq(quantity)
        & data["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise RuntimeError((representation, model, support, quantity, metric, len(selected)))
    return float(selected.iloc[0]["value"])


def grouped_support_points(axis, data, representation: str, quantity: str, metric: str) -> None:
    x = np.arange(len(MODELS), dtype=float)
    for model_index, model in enumerate(MODELS):
        for support, offset, face in [
            ("TRAINING TRUTH SUPPORT", -0.10, MODEL_COLORS[model]),
            ("HELD-OUT TRUTH SUPPORT", 0.10, "white"),
        ]:
            value = metric_value(data, representation, model, support, quantity, metric)
            axis.scatter(
                model_index + offset,
                value,
                color=MODEL_COLORS[model],
                facecolor=face,
                edgecolor=MODEL_COLORS[model],
                marker="o",
                s=38,
                linewidth=1.1,
                zorder=3,
            )
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    axis.set_yscale("log")
    axis.set_ylabel("relative RMS error")


def figure3() -> None:
    source = DATA / "final_callsite_y_metrics.csv"
    data = pd.read_csv(source)
    masks = (
        (data["representation"].eq("A") & data["quantity"].eq("A") & data["metric"].eq("relative_RMS_error"))
        | (
            data["representation"].eq("B")
            & data["quantity"].isin(["A", "R_all", "R_truth_active"])
            & data["metric"].eq("relative_RMS_error")
        )
        | (
            data["representation"].eq("C")
            & data["quantity"].eq("source_vector")
            & data["metric"].eq("normalized_vector_relative_RMS_error")
        )
    )
    plotted = data[masks].copy()
    figure = plt.figure(figsize=(7.2, 7.2))
    grid = figure.add_gridspec(3, 2, hspace=0.58, wspace=0.32)
    axes = [
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, :]),
    ]
    grouped_support_points(axes[0], data, "A", "A", "relative_RMS_error")
    grouped_support_points(axes[1], data, "B", "A", "relative_RMS_error")

    axis = axes[2]
    x = np.arange(len(MODELS), dtype=float)
    positions = {
        ("R_all", "TRAINING TRUTH SUPPORT"): -0.24,
        ("R_all", "HELD-OUT TRUTH SUPPORT"): -0.08,
        ("R_truth_active", "TRAINING TRUTH SUPPORT"): 0.08,
        ("R_truth_active", "HELD-OUT TRUTH SUPPORT"): 0.24,
    }
    for model_index, model in enumerate(MODELS):
        for quantity, marker in [("R_all", "o"), ("R_truth_active", "s")]:
            for support, face in [
                ("TRAINING TRUTH SUPPORT", MODEL_COLORS[model]),
                ("HELD-OUT TRUTH SUPPORT", "white"),
            ]:
                value = metric_value(data, "B", model, support, quantity, "relative_RMS_error")
                axis.scatter(
                    model_index + positions[(quantity, support)],
                    value,
                    color=MODEL_COLORS[model],
                    facecolor=face,
                    edgecolor=MODEL_COLORS[model],
                    marker=marker,
                    s=38,
                    linewidth=1.1,
                    zorder=3,
                )
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    axis.set_yscale("log")
    axis.set_ylabel("relative RMS error")
    grouped_support_points(
        axes[3], data, "C", "source_vector", "normalized_vector_relative_RMS_error"
    )

    for axis, title in zip(
        axes,
        [r"Rep. A: $A$", r"Rep. B: $A$", r"Rep. B: $R$", "Rep. C: source vector"],
    ):
        axis.set_title(title)
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.25", markerfacecolor="0.25", linestyle="none", label="training"),
            Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white", linestyle="none", label="evaluation"),
            Line2D([0], [0], marker="o", color="0.25", linestyle="none", label=r"all $R$ samples"),
            Line2D([0], [0], marker="s", color="0.25", linestyle="none", label=r"active $R$ samples"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.12)
    save_figure(
        figure,
        "ML3_main_callsite_physical_accuracy",
        plotted,
        caption=(
            "Final-model direct physical-law errors at truth-derived pre-moist states "
            "Y*=P(X*). Filled points show training states 0--80 and open points show "
            "evaluation states 81--160. Representation A reports A; Representation B "
            "reports A and both all-sample and truth-active R errors; Representation C "
            "reports the normalized source-vector error. Detailed C source-component and "
            "B activation metrics are retained in the supplementary data and main accuracy table."
        ),
        question="How accurately does each final model reproduce local moist physics at its truth-derived call site?",
        quantity_kind="FINAL FROZEN-NETWORK DIRECT PHYSICAL-LAW ERROR",
        support="Test 2B training states 0--80 and evaluation states 81--160 at Y*=P(X*)",
        sources=[source, source.with_suffix(".json")],
        notes=[
            "All features and analytical targets are evaluated at Y*.",
            "Representation-B active R uses the accepted meaningful-activity mask.",
            "Evaluation states did not influence training.",
        ],
        state_contracts={"all panels": "truth-derived Y*=P(X*)"},
    )
    figure3_c_supplement(data, source)


def figure3_c_supplement(data: pd.DataFrame, source: Path) -> None:
    components = ["S", "Qv", "Qc", "Qr"]
    rows = data[
        data["representation"].eq("C")
        & data["quantity"].isin([f"source_components.{name}" for name in components])
        & data["metric"].eq("relative_RMS_error")
    ].copy()
    stem = "ML3_supplement_callsite_source_components"
    csv_path = SUPPLEMENT / f"{stem}.csv"
    json_path = SUPPLEMENT / f"{stem}.json"
    pdf_path = SUPPLEMENT / f"{stem}.pdf"
    png_path = SUPPLEMENT / f"{stem}.png"
    for path in [csv_path, json_path, pdf_path, png_path]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    rows.to_csv(csv_path, index=False)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for axis, component in zip(axes.flat, components):
        grouped_support_points(
            axis, data, "C", f"source_components.{component}", "relative_RMS_error"
        )
        axis.set_title(f"{component} source")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.25", markerfacecolor="0.25", linestyle="none", label="training"),
            Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white", linestyle="none", label="evaluation"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.savefig(pdf_path, bbox_inches="tight", metadata={"Title": stem})
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    payload = {
        "figure_id": stem,
        "draft_caption": (
            "Representation C source-component errors at truth-derived pre-moist states Y*. "
            "Filled points show training states 0--80 and open points show evaluation states "
            "81--160. All values use the frozen carrier weighting and output scales."
        ),
        "scientific_question": "Which Representation C source components dominate the final Y* error?",
        "quantity_kind": "FINAL FROZEN-NETWORK DIRECT PHYSICAL-LAW ERROR",
        "support_classification": "Test 2B training/evaluation states at Y*=P(X*)",
        "representations": ["C"],
        "model_labels": MODELS,
        "source_artifacts": [file_record(source), file_record(source.with_suffix(".json"))],
        "files": {
            "csv": {**file_record(csv_path), "rows": len(rows)},
            "pdf": file_record(pdf_path),
            "png_300dpi": file_record(png_path),
        },
        "notes": ["Evaluation states did not influence training."],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def figure4() -> None:
    source = SUPPLEMENT / "ML5_deployed_physical_diagnostics_test2b.csv"
    data = pd.read_csv(source)
    plotted = data[data["model_label"].isin(MODELS)].copy()
    specs = [
        ("relative_maximum_total_water_drift", "maximum relative total-water drift", "log"),
        ("final_mixed_state_error", "final state error", "log"),
        ("minimum_moisture_coefficient", "minimum moisture coefficient", "symlog"),
        ("absolute_relative_final_Qr_error", "relative final rain-mass error", "log"),
        ("absolute_relative_final_Qc_error", "relative final cloud-mass error", "log"),
        ("rain_onset_error_s", "rain-onset timing error", "linear"),
    ]
    x = np.arange(len(MODELS), dtype=float)
    offsets = {"A": -0.20, "B": 0.0, "C": 0.20}
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.35), constrained_layout=True)
    for axis, (column, title, scale) in zip(axes.flat, specs):
        for representation in "ABC":
            panel = plotted[plotted["representation"].eq(representation)].set_index("model_label")
            values = pd.to_numeric(panel.reindex(MODELS)[column], errors="coerce").to_numpy(float)
            axis.scatter(
                x + offsets[representation],
                values,
                s=32,
                color=REP_COLORS[representation],
                marker=REP_MARKERS[representation],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        if scale == "log":
            axis.set_yscale("log")
        elif scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-8)
            axis.axhline(0.0, color="0.25", linewidth=0.9)
        else:
            axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set_title(title)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS], rotation=24, ha="right")
        axis.set_ylabel(
            "time difference (s)" if column == "rain_onset_error_s"
            else "coefficient value" if column == "minimum_moisture_coefficient"
            else "relative value"
        )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker=REP_MARKERS[rep], color="none",
                markerfacecolor=REP_COLORS[rep], markeredgecolor="white",
                label=f"Rep. {rep}", markersize=7,
            )
            for rep in "ABC"
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    save_figure(
        figure,
        "ML4_main_deployed_physical_diagnostics",
        plotted,
        caption=(
            "Physical diagnostics after deploying the four main-paper models in Test 2B. "
            "Points denote distinct frozen models and are not connected. Representations A "
            "and B impose the water and thermodynamic source identities algebraically; "
            "Representation C must approximate them. Moisture minima are finite-element "
            "coefficient minima rather than a pointwise positivity proof."
        ),
        question="Which physical properties remain accurate after deployment?",
        quantity_kind="FINAL STORED DEPLOYMENT DIAGNOSTIC",
        support="Test 2B deployed trajectories, states 0--160",
        sources=[source],
        notes=[
            "A/B source identities are structural.",
            "No trajectories were rerun.",
        ],
    )


def metric_series(data: pd.DataFrame, model: str, metric: str) -> pd.DataFrame:
    return data[
        data["model_label"].eq(model)
        & data["metric"].eq(metric)
        & data["step"].notna()
    ].sort_values("time_s")


def endpoint_value(data: pd.DataFrame, model: str, metric: str) -> float:
    selected = data[data["model_label"].eq(model) & data["metric"].eq(metric)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one endpoint: {model} {metric}")
    return float(selected.iloc[0]["value"])


def figure5(representation: str) -> None:
    source = SUPPLEMENT / f"ML6_global_trajectories_representation_{representation}_all_models.csv"
    data = pd.read_csv(source)
    plotted = data[data["model_label"].isin(MODELS)].copy()
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    truth_cloud = metric_series(plotted, "M1-Y", "truth_Qc_mass")
    truth_rain = metric_series(plotted, "M1-Y", "truth_Qr_mass")
    axes[0, 0].plot(truth_cloud["time_s"], truth_cloud["value"] / 1e12, color=MODEL_COLORS["Truth"], linewidth=1.8, label="Truth")
    axes[0, 1].plot(truth_rain["time_s"], truth_rain["value"] / 1e8, color=MODEL_COLORS["Truth"], linewidth=1.8, label="Truth")
    for model in MODELS:
        for axis, metric, divisor in [
            (axes[0, 0], "model_Qc_mass", 1e12),
            (axes[0, 1], "model_Qr_mass", 1e8),
            (axes[0, 2], "model_relative_total_water_drift", 1.0),
            (axes[1, 0], "kinetic_energy_relative_error", 1.0),
            (axes[1, 1], "projected_relative_vorticity_squared_relative_error", 1.0),
        ]:
            curve = metric_series(plotted, model, metric)
            axis.plot(curve["time_s"], curve["value"] / divisor, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    axes[0, 0].set_title(r"Integrated cloud water $M_c(t)$")
    axes[0, 0].set_ylabel(r"$M_c$ ($10^{12}$ m$^3$)")
    axes[0, 1].set_title(r"Integrated rain water $M_r(t)$")
    axes[0, 1].set_ylabel(r"$M_r$ ($10^8$ m$^3$)")
    axes[0, 2].set_title("Relative total-water drift")
    axes[0, 2].set_ylabel("relative drift")
    if representation in "AB":
        axes[0, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        axes[0, 2].set_yscale("symlog", linthresh=1e-5)
        axes[0, 2].set_yticks([-1e-2, -1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4])
    axes[0, 2].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 0].set_title("Kinetic-energy difference from truth")
    axes[1, 0].set_ylabel("relative difference")
    axes[1, 0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 1].set_title("Projected relative-vorticity-squared\ndifference from truth")
    axes[1, 1].set_ylabel("relative difference")
    axes[1, 1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0, 1].axhline(0.0, color="0.20" if representation == "C" else "0.55", linewidth=1.0 if representation == "C" else 0.7)
    x = np.arange(len(MODELS), dtype=float)
    width = 0.36
    final = [endpoint_value(plotted, model, "final_mixed_state_error") for model in MODELS]
    maximum = [endpoint_value(plotted, model, "maximum_mixed_state_error") for model in MODELS]
    axes[1, 2].bar(x - width / 2, final, width, color="#4C78A8", label="final")
    axes[1, 2].bar(x + width / 2, maximum, width, color="#F58518", label="maximum")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xticks(x, [MODEL_LABELS[model] for model in MODELS], rotation=24, ha="right")
    axes[1, 2].set_title("Final and maximum state error")
    axes[1, 2].set_ylabel("relative error")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat[:5]:
        axis.set_xlabel("time (s)")
    for axis in [axes[0, 0], axes[0, 1]]:
        axis.axvline(5100, color="0.55", linestyle=":", linewidth=0.75)
        axis.axvline(12000, color="0.55", linestyle="--", linewidth=0.75)
    figure.legend(
        handles=[Line2D([0], [0], color=MODEL_COLORS["Truth"], linewidth=1.8, label="Truth")]
        + [Line2D([0], [0], color=MODEL_COLORS[model], marker="o", label=model) for model in MODELS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.17, hspace=0.56, wspace=0.40)
    stem = f"ML5{representation}_main_global_trajectories"
    save_figure(
        figure,
        stem,
        plotted,
        caption=(
            f"Global Test 2B trajectories for Representation {representation} and the four "
            "main-paper models. Integrated cloud and rain water use m^3 under the unit-"
            "reference-density convention. Kinetic energy is not total energy, and projected "
            "relative-vorticity squared is not potential enstrophy. Vertical guides mark first "
            "truth rain at 5100 s and peak integrated truth rain production at 12000 s."
        ),
        question="Do the deployed main-paper models reproduce the global physical evolution?",
        quantity_kind="FINAL STORED DEPLOYMENT DIAGNOSTIC TIME SERIES",
        support="Test 2B deployed trajectories, states 0--160",
        sources=[source],
        notes=[
            "Truth and model time indices match.",
            "Representation C rain water has an explicit zero line.",
            "No solver rerun was performed.",
        ],
        representations=[representation],
    )


def write_table_bundle(
    stem: str,
    frame: pd.DataFrame,
    title: str,
    caption: str,
    *,
    directory: Path = TABLE_MAIN,
    overwrite_task_output: bool = False,
) -> None:
    csv_path = directory / f"{stem}.csv"
    md_path = directory / f"{stem}.md"
    tex_path = directory / f"{stem}.tex"
    for path in [csv_path, md_path, tex_path]:
        if path.exists() and not overwrite_task_output:
            raise FileExistsError(f"refusing to overwrite {path}")
    frame.to_csv(csv_path, index=False)
    display = frame.copy()

    def markdown_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.4g}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    headers = [markdown_cell(column) for column in display.columns]
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(markdown_cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    md_path.write_text(
        f"# {title}\n\n{caption}\n\n" + "\n".join(markdown_rows) + "\n"
    )
    latex_escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    def latex_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        rendered = (
            f"{float(value):.4g}"
            if isinstance(value, (float, np.floating))
            else str(value)
        )
        return "".join(latex_escapes.get(character, character) for character in rendered)

    latex_rows = [
        r"\begin{tabular}{" + "l" * len(headers) + "}",
        r"\hline",
        " & ".join(latex_cell(column) for column in display.columns) + r" \\",
        r"\hline",
    ]
    latex_rows.extend(
        " & ".join(latex_cell(value) for value in row) + r" \\"
        for row in display.itertuples(index=False, name=None)
    )
    latex_rows.extend([r"\hline", r"\end{tabular}"])
    tex_path.write_text("% " + caption + "\n" + "\n".join(latex_rows) + "\n")


def extract_yhat_metrics() -> None:
    destination = DATA / "deployed_callsite_yhat_metrics.csv"
    sidecar_path = destination.with_suffix(".json")
    if destination.exists() or sidecar_path.exists():
        raise FileExistsError("refusing to overwrite deployed call-site metrics")
    rows = []
    sources = []
    for representation in "ABC":
        historical_path = AUTH / (
            f"external-results/test2b-rain-active-learning/production/representation-{representation}/"
            f"representation_{representation.lower()}_final_comparison.json"
        )
        m1y_path = M1Y / (
            f"external-results/m1y-test2b-20260828/evaluation/representation_{representation}_matched.json"
        )
        historical = json.loads(historical_path.read_text())
        m1y = json.loads(m1y_path.read_text())
        sources.extend([historical_path, m1y_path])
        for model in MODELS:
            record = (
                m1y["standard_M1_Y"]["autonomous"]
                if model == "M1-Y"
                else historical["autonomous"][model]
            )
            source_path = m1y_path if model == "M1-Y" else historical_path
            if representation == "A":
                metrics = {"A": record["A_error_on_model_postprefix_states"]["ALL"]}
            elif representation == "B":
                rain = record["R_error_on_model_postprefix_states"]["ALL"]
                metrics = {
                    "A": record["A_error_on_model_postprefix_states"]["ALL"],
                    "R_all": rain["all_samples"],
                    "R_truth_active": rain["truth_active_samples"],
                }
            else:
                components = record["source_diagnostics_on_model_postprefix_states"]["ALL"]["component_errors"]
                metrics = {f"{name}_source": components[name] for name in ["S", "Qv", "Qc", "Qr"]}
            for quantity, metric in metrics.items():
                rows.append(
                    {
                        "physical_case": "Test 2B",
                        "representation": representation,
                        "model_label": model,
                        "state": "model-generated pre-moist state Yhat=P(Xhat)",
                        "regime": "ALL deployed calls",
                        "quantity": quantity,
                        "physical_RMS_error": metric.get("physical_RMS_error"),
                        "normalized_RMS_error": metric.get("normalized_RMS_error"),
                        "relative_RMS_error": metric.get("relative_RMS_error"),
                        "target_RMS": metric.get("target_RMS"),
                        "sample_count": metric.get("sample_count"),
                        "source_path": str(source_path.resolve()),
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 32 or frame.groupby(["representation", "model_label"]).ngroups != 12:
        raise RuntimeError("deployed call-site coverage is not uniform")
    frame.to_csv(destination, index=False)
    sidecar = {
        "status": "complete",
        "coverage_uniform_for_all_12_main_models": True,
        "state": "model-generated pre-moist states Yhat_n=P(Xhat_n)",
        "scope": "ALL 160 deployed moist-physics calls; 10,485,760 spatial samples per metric",
        "representation_metrics": {
            "A": ["A"],
            "B": ["A", "R all", "R truth-active"],
            "C": ["S", "Qv", "Qc", "Qr source components"],
        },
        "common_scalar_across_representations": False,
        "note": "Coverage is uniform, but the learned quantity differs by representation; no cross-representation scalar is manufactured.",
        "sources": [file_record(path) for path in sorted(set(sources))],
        "output": file_record(destination),
        "operations": {"stored_JSON_parsing_only": True, "rollout_performed": False},
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    write_table_bundle(
        "tableS_deployed_callsite_yhat_accuracy",
        frame.drop(columns=["source_path"]),
        "Supplementary table. Local moist-physics error on deployed pre-moist states",
        "Stored local errors evaluated on model-generated pre-moist states Yhat=P(Xhat). Coverage exists for all four main models and all representations, but each representation learns a different quantity.",
        directory=TABLE_SUPPLEMENT,
        overwrite_task_output=True,
    )


def tables() -> None:
    TABLE_MAIN.mkdir(parents=True, exist_ok=True)
    TABLE_SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    for source in sorted(TABLE_ROOT.iterdir()):
        if source.is_file() and source.suffix in {".csv", ".md", ".tex"}:
            destination = TABLE_SUPPLEMENT / source.name
            if destination.exists():
                if sha256(source) != sha256(destination):
                    raise RuntimeError(f"existing supplementary copy differs: {destination}")
            else:
                shutil.copyfile(source, destination)

    for suffix in ["csv", "md", "tex"]:
        source = TABLE_ROOT / f"table1_data_supports.{suffix}"
        destination = TABLE_MAIN / f"table1_data_supports.{suffix}"
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise RuntimeError(f"existing main-table copy differs: {destination}")
        else:
            shutil.copyfile(source, destination)

    contracts = pd.read_csv(TABLE_ROOT / "table2_training_contracts.csv")
    contracts = contracts[
        contracts["physical_case"].eq("Test 2B") & contracts["model_label"].isin(MODELS)
    ].copy()
    write_table_bundle(
        "table2_main_training_contracts",
        contracts,
        "Table 2. Main Test 2B training contracts",
        "Frozen contracts for M1-Y, H1/M2-Y, H2, and H5. H1/H2/H5 are sequential continuations; objective and optimization history therefore change together.",
        overwrite_task_output=True,
    )

    direct = pd.read_csv(DATA / "final_callsite_y_metrics.csv")
    records = []
    for representation in "ABC":
        for model in MODELS:
            for support, label in [
                ("TRAINING TRUTH SUPPORT", "training"),
                ("HELD-OUT TRUTH SUPPORT", "evaluation"),
            ]:
                def value(quantity: str, metric: str):
                    selected = direct[
                        direct["representation"].eq(representation)
                        & direct["model_label"].eq(model)
                        & direct["support"].eq(support)
                        & direct["quantity"].eq(quantity)
                        & direct["metric"].eq(metric)
                    ]
                    return np.nan if selected.empty else float(selected.iloc[0]["value"])
                records.append(
                    {
                        "representation": representation,
                        "model": model,
                        "states": label,
                        "A_rel_RMS": value("A", "relative_RMS_error"),
                        "R_all_rel_RMS": value("R_all", "relative_RMS_error"),
                        "R_active_rel_RMS": value("R_truth_active", "relative_RMS_error"),
                        "R_false_positive_rate": value("R_activation", "false_positive_rate_given_truth_inactive"),
                        "R_false_negative_rate": value("R_activation", "false_negative_rate_given_truth_active"),
                        "S_source_rel_RMS": value("source_components.S", "relative_RMS_error"),
                        "Qv_source_rel_RMS": value("source_components.Qv", "relative_RMS_error"),
                        "Qc_source_rel_RMS": value("source_components.Qc", "relative_RMS_error"),
                        "Qr_source_rel_RMS": value("source_components.Qr", "relative_RMS_error"),
                        "source_vector_rel_RMS": value("source_vector", "normalized_vector_relative_RMS_error"),
                    }
                )
    direct_table = pd.DataFrame(records)
    write_table_bundle(
        "table3_main_callsite_accuracy",
        direct_table,
        "Table 3. Final direct physical-law accuracy at Y*",
        "Final frozen-network errors at truth-derived pre-moist states Y*=P(X*) for training states 0--80 and evaluation states 81--160. Evaluation states did not influence training.",
        overwrite_task_output=True,
    )

    rain = pd.read_csv(TABLE_ROOT / "table5_rain_events.csv")
    rain = rain[rain["model_label"].isin(["Truth"] + MODELS)].copy()
    write_table_bundle(
        "table5_main_rain_events",
        rain,
        "Table 5. Main rain-event and water-partition diagnostics",
        "Stored Test 2B rain-event and water-partition diagnostics for truth and the four main-paper methods.",
        overwrite_task_output=True,
    )
    extract_yhat_metrics()


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    establish_baseline()
    figure1()
    figure2()
    figure3()
    figure4()
    for representation in "ABC":
        figure5(representation)
    tables()
    outputs = sorted(MAIN.glob("ML*_main_*.json"))
    print(
        json.dumps(
            {
                "status": "complete",
                "new_main_figure_count": len(outputs),
                "new_main_figures": [path.stem for path in outputs],
                "main_table_count": len(list(TABLE_MAIN.glob("*.csv"))),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
