#!/usr/bin/env python3
"""Generate the frozen DIMSWE ML-results figure candidates.

This script performs plotting and JSON/CSV parsing only.  It does not import
the DIMSWE solver, instantiate an optimizer, advance a timestep, or write to
the authoritative repository.  Every figure is accompanied by the exact
plotted rows and a machine-readable provenance/caption sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from portable_paths import REFERENCE_REPOSITORY as AUTH

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAIN = ROOT / "figures" / "main"
SUPP = ROOT / "figures" / "supplement"
MODEL_ORDER = [
    "M1-X",
    "M1-Y",
    "M2-X-independent",
    "warm M2-X",
    "H1",
    "H2",
    "H5",
]
T2A_MODEL_ORDER = ["M1-X", "M2-X-independent", "warm M2-X", "H1", "H2", "H5"]
MODEL_COLORS = {
    "M1-X": "#1f77b4",
    "M1-Y": "#d62728",
    "M2-X-independent": "#9467bd",
    "warm M2-X": "#8c564b",
    "H1": "#2ca02c",
    "H2": "#ff7f0e",
    "H5": "#17becf",
    "Truth": "#111111",
}
REP_COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
SUPPORT_STYLES = {
    "TRAINING TRUTH SUPPORT": "-",
    "HELD-OUT TRUTH SUPPORT": "--",
}
CONTEXT_COLORS = {
    "X training": "#1b9e77",
    "Y training": "#d95f02",
    "X held-out": "#7570b3",
    "Y held-out": "#e7298a",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.55,
        "lines.linewidth": 1.25,
        "lines.markersize": 3.5,
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    return value


def save_bundle(
    fig: plt.Figure,
    destination: Path,
    plotted: pd.DataFrame,
    *,
    caption: str,
    scientific_question: str,
    quantity_kind: str,
    source_artifacts: Iterable[str | Path],
    support: str,
    metric_definitions: dict[str, str],
    units: dict[str, str],
    notes: Iterable[str] = (),
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = destination.with_suffix(".pdf")
    png = destination.with_suffix(".png")
    csv = destination.with_suffix(".csv")
    sidecar = destination.with_suffix(".json")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Title": destination.name, "Subject": scientific_question},
    )
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plotted.to_csv(csv, index=False)
    payload = {
        "figure_id": destination.name,
        "scientific_question": scientific_question,
        "draft_caption": caption,
        "quantity_kind": quantity_kind,
        "source_artifacts": sorted({str(Path(p)) for p in source_artifacts}),
        "support_classification": support,
        "metric_definitions": metric_definitions,
        "units": units,
        "model_labels": [
            str(v)
            for v in pd.unique(plotted["model_label"])
            if "model_label" in plotted.columns and pd.notna(v)
        ],
        "representations": (
            [str(v) for v in pd.unique(plotted["representation"]) if pd.notna(v)]
            if "representation" in plotted.columns
            else []
        ),
        "state_intervals": (
            sorted(
                {
                    f"{int(a)}--{int(b)}"
                    for a, b in zip(plotted["state_first"], plotted["state_last"])
                    if pd.notna(a) and pd.notna(b)
                }
            )
            if {"state_first", "state_last"}.issubset(plotted.columns)
            else []
        ),
        "notes": list(notes),
        "files": {
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
            "png_300dpi": {"path": str(png), "sha256": sha256(png)},
            "csv": {"path": str(csv), "sha256": sha256(csv), "rows": len(plotted)},
        },
    }
    sidecar.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def model_handles(models: Iterable[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=MODEL_COLORS[m], marker="o", label=m)
        for m in models
    ]


def sparse_x_axis(ax: plt.Axes, xmax: float) -> None:
    ax.set_xscale("symlog", linthresh=1.0)
    if xmax >= 100_000:
        candidates = [0, 10, 1_000, 10_000, 200_000]
    elif xmax > 10_000:
        candidates = [0, 10, 1_000, 10_000, 50_000]
    else:
        candidates = [0, 1, 10, 100, 1_000, 10_000]
    ticks = [v for v in candidates if v <= xmax]
    ax.set_xticks(ticks)
    def compact(value: float, _position: float) -> str:
        if value >= 1_000:
            return f"{value / 1_000:g}k"
        return f"{value:g}"
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(compact))


def figure_ml1_test2b() -> None:
    source = DATA / "checkpoint_training_objectives.csv"
    raw = pd.read_csv(source)
    data = raw[raw["physical_case"].eq("Test 2B")].copy()
    families = [
        ("Direct physical-law objectives", ["M1-X", "M1-Y"]),
        ("One-step objectives", ["M2-X-independent", "warm M2-X", "H1"]),
        ("Recursive objectives: stored endpoints", ["H2", "H5"]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.25), constrained_layout=True)
    plotted_rows: list[pd.DataFrame] = []
    for ri, rep in enumerate("ABC"):
        for ci, (family, models) in enumerate(families):
            ax = axes[ri, ci]
            panel = data[data["representation"].eq(rep) & data["model_label"].isin(models)]
            plotted_rows.append(panel)
            for model in models:
                q = panel[panel["model_label"].eq(model)].sort_values("checkpoint_iteration")
                if q.empty:
                    continue
                if ci == 2:
                    ax.scatter(
                        q["checkpoint_iteration"],
                        q["value"],
                        s=22,
                        color=MODEL_COLORS[model],
                        marker="D" if model == "H5" else "o",
                        label=model,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        q["checkpoint_iteration"],
                        q["value"],
                        color=MODEL_COLORS[model],
                        marker="o",
                        label=model,
                    )
            ax.set_yscale("log")
            sparse_x_axis(ax, max(20, float(panel["checkpoint_iteration"].max())))
            if ri == 0:
                ax.set_title(family)
            if ci == 0:
                ax.set_ylabel(f"Representation {rep}\nnormalized objective")
            if ri == 2:
                ax.set_xlabel("accepted iteration")
            ax.legend(loc="best", frameon=False)
            if ci == 2:
                ax.text(
                    0.03,
                    0.04,
                    "no recursive checkpoint\nhistory reconstructed",
                    transform=ax.transAxes,
                    va="bottom",
                    fontsize=6.8,
                    color="0.35",
                )
    plotted = pd.concat(plotted_rows, ignore_index=True)
    caption = (
        "Sparse Test 2B fitted-objective progress at actual saved checkpoints. "
        "M1-X/M1-Y are fixed-array direct objectives; M2-X and H1 were reconstructed "
        "from fixed prepared maps. H2/H5 show stored initial/final objective endpoints "
        "only; no recursive checkpoint history was run. Curves are not compared across "
        "representations because their normalizations differ."
    )
    save_bundle(
        fig,
        MAIN / "ML1_optimization_progress_test2b",
        plotted,
        caption=caption,
        scientific_question="Did optimization make progress under the frozen budgets?",
        quantity_kind="FITTED OBJECTIVE HISTORY or STORED OBJECTIVE ENDPOINT",
        source_artifacts=[source],
        support="Test 2B training truth support 0--80; objective-specific X, Y, or recursive evaluation state",
        metric_definitions={"value": "Normalized objective actually fitted by the named run."},
        units={"value": "dimensionless normalized objective", "checkpoint_iteration": "accepted iterations"},
        notes=[
            "Markers occur only at actual checkpoint iterations.",
            "H2/H5 connecting histories are deliberately absent under the strict cost gate.",
            "MAXITER endpoint status is retained in Table 2 and the history CSV.",
        ],
    )


def figure_ml1_test2a() -> None:
    source = DATA / "checkpoint_training_objectives.csv"
    raw = pd.read_csv(source)
    data = raw[raw["physical_case"].eq("Test 2A")].copy()
    families = [
        ("M1-X", ["M1-X"]),
        ("M2-X and H1", ["M2-X-independent", "warm M2-X", "H1"]),
        ("H2 and H5", ["H2", "H5"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), constrained_layout=True)
    plotted_rows: list[pd.DataFrame] = []
    for ri, rep in enumerate(["A", "C"]):
        for ci, (family, models) in enumerate(families):
            ax = axes[ri, ci]
            panel = data[data["representation"].eq(rep) & data["model_label"].isin(models)]
            plotted_rows.append(panel)
            for model in models:
                q = panel[panel["model_label"].eq(model)].sort_values("checkpoint_iteration")
                if q.empty:
                    continue
                ax.plot(
                    q["checkpoint_iteration"],
                    q["value"],
                    color=MODEL_COLORS[model],
                    marker="o",
                    label=model,
                )
            ax.set_yscale("log")
            sparse_x_axis(ax, float(panel["checkpoint_iteration"].max()))
            if ri == 0:
                ax.set_title(family)
            if ci == 0:
                ax.set_ylabel(f"Representation {rep}\nnormalized objective")
            if ri == 1:
                ax.set_xlabel("accepted iteration")
            ax.legend(frameon=False, loc="best")
    plotted = pd.concat(plotted_rows, ignore_index=True)
    save_bundle(
        fig,
        SUPP / "ML1_test2a_recorded_optimization_progress",
        plotted,
        caption=(
            "Historically recorded sparse Test 2A training-objective histories for the "
            "accepted Representation A and C ladders. All curves concern training support "
            "0--80; no Test 2A held-out or validation protocol is introduced."
        ),
        scientific_question="What optimization progress is historically recorded for the Test 2A precursor?",
        quantity_kind="RECORDED SPARSE TRAINING OBJECTIVE HISTORY",
        source_artifacts=[source],
        support="Test 2A training support 0--80 only",
        metric_definitions={"value": "Historically recorded normalized fitted objective."},
        units={"value": "dimensionless normalized objective", "checkpoint_iteration": "accepted iterations"},
        notes=["No states 81--160 were evaluated for this figure."],
    )


DIRECT_PANELS = [
    ("A", "A", "relative_RMS_error", "Representation A: relative A RMS error"),
    ("B", "A", "relative_RMS_error", "Representation B: relative A RMS error"),
    ("B", "R_truth_active", "relative_RMS_error", "Representation B: active-R relative RMS error"),
    (
        "C",
        "source_vector",
        "normalized_vector_relative_RMS_error",
        "Representation C: normalized source-vector relative RMS error",
    ),
]


def figure_ml2(models: list[str], destination: Path, variant: str) -> None:
    source = DATA / "checkpoint_direct_histories.csv"
    raw = pd.read_csv(source)
    data = raw[
        raw["physical_case"].eq("Test 2B")
        & raw["evaluation_state"].eq("X")
        & raw["model_label"].isin(models)
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), constrained_layout=True)
    plotted_rows: list[pd.DataFrame] = []
    for ax, (rep, quantity, metric, title) in zip(axes.flat, DIRECT_PANELS):
        panel = data[
            data["representation"].eq(rep)
            & data["quantity"].eq(quantity)
            & data["metric"].eq(metric)
        ]
        plotted_rows.append(panel)
        for model in models:
            for support, style in SUPPORT_STYLES.items():
                q = panel[
                    panel["model_label"].eq(model) & panel["support"].eq(support)
                ].sort_values("checkpoint_iteration")
                if q.empty:
                    continue
                ax.plot(
                    q["checkpoint_iteration"],
                    q["value"],
                    color=MODEL_COLORS[model],
                    linestyle=style,
                    marker="o",
                    markerfacecolor="white" if support.startswith("HELD") else MODEL_COLORS[model],
                )
        ax.set_title(title)
        ax.set_yscale("log")
        sparse_x_axis(ax, 10_000)
        ax.set_xlabel("accepted iteration")
        ax.set_ylabel("post-hoc relative error")
    handles = model_handles(models)
    handles += [
        Line2D([0], [0], color="0.25", linestyle="-", label="training truth support 0--80"),
        Line2D([0], [0], color="0.25", linestyle="--", marker="o", markerfacecolor="white", label="held-out truth support 81--160"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    plotted = pd.concat(plotted_rows, ignore_index=True)
    save_bundle(
        fig,
        destination,
        plotted,
        caption=(
            f"{variant} Test 2B post-hoc direct prediction errors at saved checkpoints. "
            "Solid curves use training truth support 0--80 and dashed/open-marker curves "
            "use temporally held-out truth support 81--160. These are not validation losses "
            "recorded during training and are not the fitted objective for M2-X/H1/H2/H5."
        ),
        scientific_question="How does direct physical-quantity prediction evolve on training and held-out truth states?",
        quantity_kind="POST-HOC CHECKPOINT EVALUATION",
        source_artifacts=[source, DATA / "heldout_x_test2b.npz", DATA / "training_x_carriers_test2b.npz"],
        support="Test 2B training truth support 0--80 and held-out truth support 81--160, evaluated at X",
        metric_definitions={
            "relative_RMS_error": "Carrier-mass-weighted physical RMS error divided by carrier-mass-weighted target RMS.",
            "normalized_vector_relative_RMS_error": "Carrier-mass-weighted RMS in frozen output-scaled source coordinates divided by the corresponding target RMS.",
        },
        units={"value": "dimensionless relative RMS", "checkpoint_iteration": "accepted iterations"},
        notes=[
            "Only actual saved checkpoints are marked; no interpolation is used.",
            "Representation C retains componentwise histories in checkpoint_direct_histories.csv.",
        ],
    )


def context_label(row: pd.Series) -> str:
    state = str(row["evaluation_state"])
    support = "training" if str(row["support"]).startswith("TRAINING") else "held-out"
    return f"{state} {support}"


def figure_ml3_final() -> None:
    source = DATA / "m1_cross_state_final_metrics.csv"
    raw = pd.read_csv(source)
    raw["context"] = raw.apply(context_label, axis=1)
    panels = [
        ("A", "A", "relative_RMS_error", "A: relative A RMS error"),
        ("B", "A", "relative_RMS_error", "B: relative A RMS error"),
        ("B", "R_truth_active", "relative_RMS_error", "B: active-R relative RMS error"),
        (
            "C",
            "source_vector",
            "normalized_vector_relative_RMS_error",
            "C: normalized source-vector relative RMS error",
        ),
    ]
    contexts = ["X training", "Y training", "X held-out", "Y held-out"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    plotted_rows: list[pd.DataFrame] = []
    x = np.arange(len(contexts))
    width = 0.36
    for ax, (rep, quantity, metric, title) in zip(axes.flat, panels):
        panel = raw[
            raw["representation"].eq(rep)
            & raw["quantity"].eq(quantity)
            & raw["metric"].eq(metric)
        ].copy()
        plotted_rows.append(panel)
        for offset, model in [(-width / 2, "M1-X"), (width / 2, "M1-Y")]:
            q = panel[panel["model_label"].eq(model)].set_index("context")
            values = [float(q.loc[c, "value"]) for c in contexts]
            ax.bar(
                x + offset,
                values,
                width,
                color=MODEL_COLORS[model],
                label=f"trained {model}",
            )
        ax.set_yscale("log")
        ax.set_xticks(x, [c.replace(" ", "\n") for c in contexts])
        ax.set_ylabel("post-hoc relative error")
        ax.set_title(title)
        ax.legend(frameon=False)
    plotted = pd.concat(plotted_rows, ignore_index=True)
    save_bundle(
        fig,
        MAIN / "ML3_m1x_m1y_cross_state_final",
        plotted,
        caption=(
            "Matched final-checkpoint cross-state evaluation for Test 2B M1-X and M1-Y. "
            "Each frozen model is evaluated on X and Y at training truth indices 0--80 and "
            "held-out truth indices 81--160. M1-X and M1-Y share architecture, seed, frozen "
            "X-fitted normalization, weighting, optimizer, and 10,000-iteration budget; "
            "the intended training change is X versus Y evaluation location."
        ),
        scientific_question="How strongly does timestep evaluation location affect direct-law recovery?",
        quantity_kind="POST-HOC FINAL CHECKPOINT EVALUATION",
        source_artifacts=[source, DATA / "heldout_x_test2b.npz", DATA / "checkpoint_hash_manifest.json"],
        support="Separate Test 2B X/Y training truth support 0--80 and X/Y held-out truth support 81--160",
        metric_definitions={
            "A/R relative_RMS_error": "Carrier-mass-weighted physical RMS error divided by target RMS.",
            "source-vector relative_RMS_error": "Carrier-mass-weighted relative RMS in frozen output-scaled source coordinates.",
        },
        units={"value": "dimensionless relative RMS"},
        notes=["Training and held-out supports are not pooled.", "M1-Y is not claimed to improve every metric."],
    )


def figure_ml3_history() -> None:
    source = DATA / "checkpoint_direct_histories.csv"
    raw = pd.read_csv(source)
    raw = raw[
        raw["model_label"].isin(["M1-X", "M1-Y"])
        & raw["evaluation_state"].isin(["X", "Y"])
    ].copy()
    raw["context"] = raw.apply(context_label, axis=1)
    panels = [
        ("A", "A", "relative_RMS_error", "relative A RMS"),
        ("B", "A", "relative_RMS_error", "relative A RMS"),
        ("B", "R_truth_active", "relative_RMS_error", "active-R relative RMS"),
        ("C", "source_vector", "normalized_vector_relative_RMS_error", "source-vector relative RMS"),
    ]
    contexts = ["X training", "Y training", "X held-out", "Y held-out"]
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.25), constrained_layout=True)
    plotted_rows: list[pd.DataFrame] = []
    for ri, (rep, quantity, metric, row_title) in enumerate(panels):
        for ci, model in enumerate(["M1-X", "M1-Y"]):
            ax = axes[ri, ci]
            panel = raw[
                raw["representation"].eq(rep)
                & raw["model_label"].eq(model)
                & raw["quantity"].eq(quantity)
                & raw["metric"].eq(metric)
            ]
            plotted_rows.append(panel)
            for context in contexts:
                q = panel[panel["context"].eq(context)].sort_values("checkpoint_iteration")
                ax.plot(
                    q["checkpoint_iteration"],
                    q["value"],
                    color=CONTEXT_COLORS[context],
                    marker="o",
                    label=context,
                )
            ax.set_yscale("log")
            sparse_x_axis(ax, 10_000)
            if ri == 0:
                ax.set_title(f"trained {model}")
            if ci == 0:
                ax.set_ylabel(f"{rep}: {row_title}")
            if ri == len(panels) - 1:
                ax.set_xlabel("accepted iteration")
    fig.legend(
        handles=[Line2D([0], [0], color=c, marker="o", label=k) for k, c in CONTEXT_COLORS.items()],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    plotted = pd.concat(plotted_rows, ignore_index=True)
    save_bundle(
        fig,
        SUPP / "ML3_m1x_m1y_cross_state_checkpoint_history",
        plotted,
        caption=(
            "Checkpoint-resolved matched M1-X/M1-Y cross-state evaluation. Every marker is "
            "a saved checkpoint evaluated post hoc on one of four distinct X/Y and training/"
            "held-out supports; none is a validation loss used during optimization."
        ),
        scientific_question="When during optimization does X/Y state specialization emerge?",
        quantity_kind="POST-HOC CHECKPOINT EVALUATION",
        source_artifacts=[source, DATA / "heldout_x_test2b.npz"],
        support="Separate Test 2B X/Y training truth support 0--80 and X/Y held-out truth support 81--160",
        metric_definitions={"value": "Representation-specific relative RMS direct-prediction metric named by the panel."},
        units={"value": "dimensionless relative RMS", "checkpoint_iteration": "accepted iterations"},
        notes=["No checkpoint interpolation is used."],
    )


def objective_value_frame(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    out = np.full((len(data), len(columns)), np.nan)
    for i, (_, row) in enumerate(data.iterrows()):
        for j, col in enumerate(columns):
            try:
                out[i, j] = float(row[col])
            except (TypeError, ValueError):
                pass
    return out


def draw_objective_heatmap(
    ax: plt.Axes, panel: pd.DataFrame, columns: list[str], title: str
) -> None:
    values = objective_value_frame(panel, columns)
    positive = values[np.isfinite(values) & (values > 0)]
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap="viridis_r",
        norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
        aspect="auto",
    )
    ax.set_xticks(range(len(columns)), [c.replace("J_", "$J_{") + "}$" for c in columns], rotation=35, ha="right")
    ax.set_yticks(range(len(panel)), panel["model_label"])
    ax.set_title(title)
    for i, (_, row) in enumerate(panel.iterrows()):
        for j, col in enumerate(columns):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.4, color="0.35")
                continue
            fitted = row.get("fitted_column") == col
            ax.text(
                j,
                i,
                f"{value:.1e}" + ("*" if fitted else ""),
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if image.norm(value) > 0.55 else "black",
                fontweight="bold" if fitted else "normal",
            )
            if fitted:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#d62728", lw=1.7))
    plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025, label="objective value")


def figure_ml4() -> None:
    source = DATA / "completed_objective_matrix.csv"
    raw = pd.read_csv(source)
    columns = ["J_M1_X", "J_M1_Y", "J_M2_X", "J_H1", "J_H2", "J_H5"]
    t2b = raw[raw["physical_case"].eq("Test 2B")].copy()
    t2b["order"] = t2b["model_label"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.1), constrained_layout=True)
    panels: list[pd.DataFrame] = []
    for ax, rep in zip(axes, "ABC"):
        panel = t2b[t2b["representation"].eq(rep)].sort_values("order")
        panels.append(panel)
        draw_objective_heatmap(ax, panel, columns, f"Representation {rep}")
    plotted = pd.concat(panels, ignore_index=True)
    save_bundle(
        fig,
        MAIN / "ML4_frozen_model_objective_matrix_test2b",
        plotted,
        caption=(
            "Test 2B frozen-model objective matrix. A red outline and asterisk identify the "
            "objective fitted by each row; every other populated cell is a post-hoc diagnostic "
            "evaluation. Color normalization is independent in A/B/C and therefore must not "
            "be compared between panels."
        ),
        scientific_question="How does each frozen model score under the other diagnostic objectives?",
        quantity_kind="FITTED FINAL OBJECTIVE and POST-HOC DIAGNOSTIC OBJECTIVE",
        source_artifacts=[source, DATA / "j_m1y_diagnostics.csv"],
        support="Test 2B objective-specific training support 0--80",
        metric_definitions={c: f"Frozen production definition of {c}; see objective-matrix source paths." for c in columns},
        units={c: "dimensionless normalized objective" for c in columns},
        notes=["No common color scale is used across representations."],
    )

    t2a = raw[raw["physical_case"].eq("Test 2A")].copy()
    t2a["order"] = t2a["model_label"].map({m: i for i, m in enumerate(T2A_MODEL_ORDER)})
    cols2a = ["J_M1_X", "J_M2_X", "J_H1", "J_H2", "J_H5"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), constrained_layout=True)
    panels = []
    for ax, rep in zip(axes, ["A", "C"]):
        panel = t2a[t2a["representation"].eq(rep)].sort_values("order")
        panels.append(panel)
        draw_objective_heatmap(ax, panel, cols2a, f"Test 2A, Representation {rep}")
    plotted = pd.concat(panels, ignore_index=True)
    save_bundle(
        fig,
        SUPP / "ML4_frozen_model_objective_matrix_test2a",
        plotted,
        caption=(
            "Historically available Test 2A frozen-model objective matrix on training support. "
            "An asterisk/red outline marks the fitted objective; other values are diagnostics. "
            "M1-Y was not part of Test 2A and no such column is introduced."
        ),
        scientific_question="Which cross-objective endpoint evaluations exist for Test 2A?",
        quantity_kind="FITTED FINAL OBJECTIVE and POST-HOC DIAGNOSTIC OBJECTIVE",
        source_artifacts=[source],
        support="Test 2A training support 0--80 only",
        metric_definitions={c: f"Frozen production definition of {c}." for c in cols2a},
        units={c: "dimensionless normalized objective" for c in cols2a},
        notes=["Blank cells were not evaluated; Test 2A has no M1-Y column."],
    )


def figure_ml5() -> None:
    source = DATA / "deployed_diagnostics.csv"
    data = pd.read_csv(source)
    data["model_order"] = data["model_label"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    data = data.sort_values(["representation", "model_order"])
    truth_qr = 230733006.980403
    truth_qc = 1551511316353.2312
    data["absolute_relative_final_Qr_error"] = pd.to_numeric(data["final_Qr_mass_error"], errors="coerce").abs() / truth_qr
    data["absolute_relative_final_Qc_error"] = pd.to_numeric(data["final_Qc_mass_error"], errors="coerce").abs() / truth_qc
    specs = [
        ("relative_maximum_total_water_drift", "maximum relative total-water drift", "log", True),
        ("final_mixed_state_error", "final mixed-state relative error", "log", True),
        ("minimum_moisture_coefficient", "minimum moisture coefficient", "symlog", False),
        ("absolute_relative_final_Qr_error", "final rain-mass error\n(absolute / truth)", "log", True),
        ("absolute_relative_final_Qc_error", "final cloud-mass error\n(absolute / truth)", "log", True),
        ("rain_onset_error_s", "rain-onset timing error (s)", "linear", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    for ax, (column, title, scale, absolute) in zip(axes.flat, specs):
        for rep in "ABC":
            panel = data[data["representation"].eq(rep)].set_index("model_label")
            values = pd.to_numeric(panel.reindex(MODEL_ORDER)[column], errors="coerce").to_numpy(float)
            if absolute:
                values = np.abs(values)
            ax.plot(x, values, marker="o", color=REP_COLORS[rep], label=f"Representation {rep}")
        if scale == "log":
            ax.set_yscale("log")
        elif scale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.axhline(0.0, color="0.4", lw=0.7)
        else:
            ax.axhline(0.0, color="0.4", lw=0.7)
        ax.set_title(title)
        short_labels = ["M1-X", "M1-Y", "M2-indep.", "M2-warm", "H1", "H2", "H5"]
        ax.set_xticks(x, short_labels, rotation=28, ha="right")
    axes[0, 0].legend(frameon=False)
    save_bundle(
        fig,
        MAIN / "ML5_deployed_physical_diagnostics_test2b",
        data,
        caption=(
            "Stored Test 2B deployment diagnostics for all 21 main frozen models. A/B impose "
            "the moist-source water and thermodynamic identities structurally; their small "
            "water drift is a deployment diagnostic, not learned conservation. C does not "
            "enforce those identities. Minima are finite-element coefficient minima, not a "
            "pointwise positivity proof."
        ),
        scientific_question="Which physical properties remain accurate after embedding each frozen network?",
        quantity_kind="FINAL STORED DEPLOYMENT DIAGNOSTIC",
        source_artifacts=[source] + data["source_path"].dropna().tolist(),
        support="Test 2B deployed autonomous trajectories, states 0--160",
        metric_definitions={
            "relative_maximum_total_water_drift": "Maximum absolute drift of domain-integrated Qv+Qc+Qr divided by its initial value.",
            "final_mixed_state_error": "Stored final mixed state relative mass-norm error.",
            "minimum_moisture_coefficient": "Minimum saved finite-element coefficient among Qv, Qc, and Qr.",
            "final partition errors": "Absolute final model-minus-truth integral mass error divided by the corresponding truth final mass.",
            "rain_onset_error_s": "First meaningful model rain time minus truth first-rain time (5100 s).",
        },
        units={
            "drift/errors": "dimensionless",
            "minimum_moisture_coefficient": "production state coefficient units",
            "rain_onset_error_s": "s",
        },
        notes=[
            "Representation A uses analytical R on the model-generated state.",
            "Representation B learns R directly; Representation C uses an effective Qr-source rain diagnostic.",
        ],
    )

    c = data[data["representation"].eq("C")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "water_source_defect_RMS", "C: water-source identity defect RMS"),
        (axes[1], "S_minus_beta2_Qv_source_defect_RMS", "C: thermodynamic-source identity defect RMS"),
    ]:
        values = pd.to_numeric(c.set_index("model_label").reindex(MODEL_ORDER)[col], errors="coerce")
        ax.bar(x, values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
        ax.set_yscale("log")
        ax.set_xticks(x, [m.replace(" ", "\n") for m in MODEL_ORDER], rotation=35, ha="right")
        ax.set_title(title)
        ax.text(
            0.02,
            0.97,
            "A/B: STRUCTURALLY ENFORCED\n(not a learned metric)",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color="0.30",
        )
    save_bundle(
        fig,
        SUPP / "ML5_representation_c_source_structure",
        c,
        caption=(
            "Representation C deployed source-manifold defects for Test 2B. In A/B the same "
            "water and thermodynamic identities are structurally enforced and are therefore "
            "not plotted as evidence of learning."
        ),
        scientific_question="How far do unconstrained C sources depart from the analytical source identities after deployment?",
        quantity_kind="LEARNED / NOT ENFORCED DEPLOYMENT DIAGNOSTIC",
        source_artifacts=[source] + c["source_path"].dropna().tolist(),
        support="Test 2B deployed autonomous trajectories, states 0--160",
        metric_definitions={
            "water_source_defect_RMS": "RMS of source_Qv+source_Qc+source_Qr.",
            "S_minus_beta2_Qv_source_defect_RMS": "RMS of source_S-beta2*source_Qv.",
        },
        units={"defects": "physical source units"},
        notes=["A/B values are algebraic identities up to roundoff and are not learned degrees of freedom."],
    )


def global_plot_data(rep: str, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories = pd.read_csv(DATA / "global_trajectories" / "test2b_global_trajectories.csv")
    trajectories = trajectories[
        trajectories["representation"].eq(rep) & trajectories["model_label"].isin(models)
    ].copy()
    endpoints = pd.read_csv(DATA / "deployed_diagnostics.csv")
    endpoints = endpoints[
        endpoints["representation"].eq(rep) & endpoints["model_label"].isin(models)
    ].copy()
    return trajectories, endpoints


def figure_ml6(rep: str, models: list[str], destination: Path, variant: str) -> None:
    source = DATA / "global_trajectories" / "test2b_global_trajectories.csv"
    trajectories, endpoints = global_plot_data(rep, models)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25), constrained_layout=True)
    truth_once = trajectories[trajectories["model_label"].eq(models[0])]
    axes[0, 0].plot(truth_once["time_s"], truth_once["truth_Qc_mass"] / 1e12, color="black", lw=1.7, label="Truth")
    axes[0, 1].plot(truth_once["time_s"], truth_once["truth_Qr_mass"] / 1e8, color="black", lw=1.7, label="Truth")
    for model in models:
        q = trajectories[trajectories["model_label"].eq(model)].sort_values("time_s")
        c = MODEL_COLORS[model]
        axes[0, 0].plot(q["time_s"], q["model_Qc_mass"] / 1e12, color=c, label=model)
        axes[0, 1].plot(q["time_s"], q["model_Qr_mass"] / 1e8, color=c, label=model)
        axes[0, 2].plot(q["time_s"], q["model_relative_total_water_drift"], color=c, label=model)
        axes[1, 0].plot(q["time_s"], q["kinetic_energy_relative_error"], color=c, label=model)
        axes[1, 1].plot(q["time_s"], q["projected_relative_vorticity_squared_relative_error"], color=c, label=model)
    axes[0, 0].set_title(r"integrated cloud water $M_c(t)$")
    axes[0, 0].set_ylabel(r"$10^{12}$ integral-mass units")
    axes[0, 1].set_title(r"integrated rain water $M_r(t)$")
    axes[0, 1].set_ylabel(r"$10^{8}$ integral-mass units")
    axes[0, 2].set_title("relative total-water drift")
    axes[0, 2].set_ylabel("relative drift")
    axes[0, 2].set_yscale("symlog", linthresh=1e-14)
    axes[1, 0].set_title("kinetic energy relative to truth")
    axes[1, 0].set_ylabel("relative mismatch")
    axes[1, 0].axhline(0.0, color="0.4", lw=0.7)
    axes[1, 1].set_title("projected relative-vorticity squared")
    axes[1, 1].set_ylabel("relative mismatch")
    axes[1, 1].axhline(0.0, color="0.4", lw=0.7)
    x = np.arange(len(models))
    ep = endpoints.set_index("model_label").reindex(models)
    width = 0.36
    axes[1, 2].bar(x - width / 2, ep["final_mixed_state_error"], width, color="#4C78A8", label="final")
    axes[1, 2].bar(x + width / 2, ep["maximum_mixed_state_error"], width, color="#F58518", label="maximum")
    axes[1, 2].set_yscale("log")
    short = {
        "M1-X": "M1-X", "M1-Y": "M1-Y", "M2-X-independent": "M2-indep.",
        "warm M2-X": "M2-warm", "H1": "H1", "H2": "H2", "H5": "H5",
    }
    axes[1, 2].set_xticks(x, [short[m] for m in models], rotation=30, ha="right")
    axes[1, 2].set_title("stored mixed-state error endpoints")
    axes[1, 2].set_ylabel("relative mass-norm error")
    axes[1, 2].legend(frameon=False)
    for ax in axes.flat[:5]:
        ax.set_xlabel("time (s)")
    for ax in [axes[0, 0], axes[0, 1]]:
        for t, style in [(5100, ":"), (12000, "--")]:
            ax.axvline(t, color="0.55", linestyle=style, lw=0.7, zorder=0)
    if len(models) > 4:
        handles = [Line2D([0], [0], color="black", lw=1.7, label="Truth")] + model_handles(models)
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    else:
        axes[0, 0].legend(frameon=False, ncol=2)

    # The plotted CSV is tidy and includes endpoint values without pretending
    # that the historical Test-2B records contain a mixed-error time series.
    rows: list[dict[str, Any]] = []
    for _, row in trajectories.iterrows():
        for metric, value in [
            ("model_Qc_mass", row["model_Qc_mass"]),
            ("truth_Qc_mass", row["truth_Qc_mass"]),
            ("model_Qr_mass", row["model_Qr_mass"]),
            ("truth_Qr_mass", row["truth_Qr_mass"]),
            ("model_relative_total_water_drift", row["model_relative_total_water_drift"]),
            ("kinetic_energy_relative_error", row["kinetic_energy_relative_error"]),
            (
                "projected_relative_vorticity_squared_relative_error",
                row["projected_relative_vorticity_squared_relative_error"],
            ),
        ]:
            rows.append(
                {
                    "representation": rep,
                    "model_label": row["model_label"],
                    "step": row["step"],
                    "time_s": row["time_s"],
                    "metric": metric,
                    "value": value,
                    "record_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC time series",
                    "source_path": row["source_path"],
                }
            )
    for _, row in endpoints.iterrows():
        for metric in ["final_mixed_state_error", "maximum_mixed_state_error"]:
            rows.append(
                {
                    "representation": rep,
                    "model_label": row["model_label"],
                    "step": np.nan,
                    "time_s": np.nan,
                    "metric": metric,
                    "value": row[metric],
                    "record_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC endpoint",
                    "source_path": row["source_path"],
                }
            )
    plotted = pd.DataFrame(rows)
    save_bundle(
        fig,
        destination,
        plotted,
        caption=(
            f"{variant} Test 2B Representation {rep} global deployment diagnostics. Cloud "
            "and rain masses are compared with truth; kinetic energy means "
            "0.5 integral h|v|^2 and is not total energy; projected relative-vorticity "
            "squared means the stored CG(3)-projection diagnostic and is not potential "
            "enstrophy. Vertical guides mark first truth rain (5100 s) and peak integrated "
            "truth rain production (12000 s). The final panel uses stored final/maximum "
            "mixed-state endpoints because the accepted Test 2B JSONs do not contain a "
            "mixed-state-error time series; no rollout was rerun."
        ),
        scientific_question="Do frozen deployed models reproduce long-time global physical evolution?",
        quantity_kind="FINAL STORED DEPLOYMENT DIAGNOSTIC",
        source_artifacts=[source, DATA / "deployed_diagnostics.csv"] + trajectories["source_path"].dropna().tolist(),
        support="Test 2B deployed trajectories, states 0--160; truth events at 5100 and 12000 s",
        metric_definitions={
            "M_c/M_r": "Domain integrals of conservative cloud/rain water variables.",
            "total-water drift": "Relative change of integral(Qv+Qc+Qr) from its initial value.",
            "kinetic energy": "0.5*integral(h*|v|^2) dA; not total Hamiltonian energy.",
            "projected relative-vorticity squared": "0.5*integral(zeta_h^2) dA after CG(3) L2 projection; not potential enstrophy.",
            "mixed-state endpoints": "Stored final and maximum relative mixed mass-norm errors; no time series was present.",
        },
        units={
            "time_s": "s",
            "M_c/M_r": "production integral-mass units",
            "relative diagnostics": "dimensionless",
        },
        notes=[
            "Truth and model time grids were verified identical at steps 0--160.",
            "A mixed-state-error trajectory would require a deployment rerun and is deferred.",
        ],
    )


def extract_test2a_deployment() -> pd.DataFrame:
    fair_path = AUTH / "external-results/test2a/fair-longfit/comparison/fair_longfit_comparison.json"
    warm_path = AUTH / "external-results/test2a/m1-to-m2-finetune/postprocess/autonomous/iter-50000/trajectory_metrics.json"
    horizon_path = AUTH / "external-results/test2a/horizon-curriculum-h1-h2-h5/postprocess/horizon_curriculum_report.json"
    c_path = AUTH / "external-results/test2a/problem-b/production/problem_b_comparison.json"
    fair = json.loads(fair_path.read_text())
    warm = json.loads(warm_path.read_text())
    horizon = json.loads(horizon_path.read_text())
    cdata = json.loads(c_path.read_text())

    records: list[dict[str, Any]] = []

    def add(rep: str, model: str, payload: dict[str, Any], source: Path, flat_mixed: bool = False) -> None:
        if flat_mixed:
            final = payload["final_mixed_state_relative_error"]
            maximum = payload["maximum_mixed_state_relative_error"]
        else:
            final = payload["mixed_state_error"]["final"]
            maximum = payload["mixed_state_error"]["maximum"]
        for metric, value in [("final_mixed_state_error", final), ("maximum_mixed_state_error", maximum)]:
            records.append(
                {
                    "representation": rep,
                    "model_label": model,
                    "metric": metric,
                    "step": 80,
                    "time_s": 8000.0,
                    "value": value,
                    "source_path": str(source),
                    "quantity_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC endpoint",
                }
            )
        for field, metric in [
            ("kinetic_energy", "kinetic_energy_relative_error"),
            ("projected_enstrophy", "projected_relative_vorticity_squared_relative_error"),
        ]:
            block = payload[field]
            for step, time, value in zip(block["steps"], block["times"], block["relative_mismatch"]):
                records.append(
                    {
                        "representation": rep,
                        "model_label": model,
                        "metric": metric,
                        "step": step,
                        "time_s": time,
                        "value": value,
                        "source_path": str(source),
                        "quantity_kind": "FINAL STORED DEPLOYMENT DIAGNOSTIC time series",
                    }
                )

    add("A", "M1-X", fair["autonomous_training_support"]["theta_op_long"], fair_path, True)
    add("A", "M2-X-independent", fair["autonomous_training_support"]["theta_disc_long"], fair_path, True)
    add("A", "warm M2-X", warm, warm_path)
    horizon_labels = {"H1-final": "H1", "H2-final": "H2", "H5-final": "H5"}
    for entry in horizon["entries"]:
        if entry["label"] in horizon_labels:
            add("A", horizon_labels[entry["label"]], entry["autonomous_training_support"], horizon_path)
    cmap = {
        "M1": "M1-X",
        "M2-X-independent": "M2-X-independent",
        "M1-to-M2-X": "warm M2-X",
        "H1": "H1",
        "H2": "H2",
        "H5": "H5",
    }
    for key, model in cmap.items():
        add("C", model, cdata["artifacts"][key]["autonomous_training_support_posthoc"], c_path)
    out = pd.DataFrame(records)
    expected = {(rep, model) for rep in ["A", "C"] for model in T2A_MODEL_ORDER}
    actual = set(zip(out["representation"], out["model_label"]))
    if actual != expected:
        raise RuntimeError(f"Test 2A deployment coverage mismatch: {actual ^ expected}")
    if out["time_s"].max() != 8000.0:
        raise RuntimeError("Test 2A deployment support does not end at 8000 s/state 80")
    target = DATA / "global_trajectories" / "test2a_training_support_deployment.csv"
    out.to_csv(target, index=False)
    meta = {
        "support": "Test 2A training interval deployment only, states 0--80",
        "terminology": "No held-out/test set is defined or created.",
        "sources": [str(fair_path), str(warm_path), str(horizon_path), str(c_path)],
        "source_sha256": {str(p): sha256(p) for p in [fair_path, warm_path, horizon_path, c_path]},
        "csv": {"path": str(target), "sha256": sha256(target), "rows": len(out)},
        "no_model_execution": True,
    }
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out


def figure_test2a_deployment() -> None:
    data = extract_test2a_deployment()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.85), constrained_layout=True)
    x = np.arange(len(T2A_MODEL_ORDER))
    width = 0.36
    for ri, rep in enumerate(["A", "C"]):
        panel = data[data["representation"].eq(rep)]
        final = panel[panel["metric"].eq("final_mixed_state_error")].set_index("model_label").reindex(T2A_MODEL_ORDER)
        maximum = panel[panel["metric"].eq("maximum_mixed_state_error")].set_index("model_label").reindex(T2A_MODEL_ORDER)
        axes[ri, 0].bar(x - width / 2, final["value"], width, color="#4C78A8", label="final")
        axes[ri, 0].bar(x + width / 2, maximum["value"], width, color="#F58518", label="maximum")
        axes[ri, 0].set_yscale("log")
        short = ["M1-X", "M2-indep.", "M2-warm", "H1", "H2", "H5"]
        axes[ri, 0].set_xticks(x, short, rotation=30, ha="right")
        axes[ri, 0].set_ylabel(f"Representation {rep}\nrelative error")
        axes[ri, 0].legend(frameon=False)
        for ci, metric in [
            (1, "kinetic_energy_relative_error"),
            (2, "projected_relative_vorticity_squared_relative_error"),
        ]:
            ax = axes[ri, ci]
            for model in T2A_MODEL_ORDER:
                q = panel[panel["metric"].eq(metric) & panel["model_label"].eq(model)].sort_values("time_s")
                ax.plot(q["time_s"], q["value"], color=MODEL_COLORS[model], label=model)
            ax.axhline(0.0, color="0.4", lw=0.7)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("relative mismatch")
    axes[0, 0].set_title("mixed-state error endpoints")
    axes[0, 1].set_title("kinetic energy relative to truth")
    axes[0, 2].set_title("projected relative-vorticity squared")
    fig.legend(handles=model_handles(T2A_MODEL_ORDER), loc="outside lower center", ncol=3, frameon=False)
    sources = data["source_path"].unique().tolist()
    save_bundle(
        fig,
        SUPP / "ML_test2a_training_interval_deployment",
        data,
        caption=(
            "Test 2A deployment diagnostics over the training interval 0--80 only. The "
            "figure uses historically stored mixed-state endpoints, kinetic energy, and "
            "projected relative-vorticity-squared diagnostics for the accepted A/C ladders. "
            "No states 81--160 are evaluated and no held-out/test interpretation is made."
        ),
        scientific_question="How do the accepted Test 2A precursor models behave over their recorded training interval?",
        quantity_kind="FINAL STORED DEPLOYMENT DIAGNOSTIC",
        source_artifacts=sources,
        support="Test 2A training-interval deployment, states 0--80 only",
        metric_definitions={
            "mixed-state endpoints": "Stored final and maximum mixed-state relative mass-norm errors.",
            "kinetic energy": "0.5*integral(h*|v|^2) dA relative mismatch; not total energy.",
            "projected relative-vorticity squared": "Stored projected-vorticity-squared relative mismatch; not potential enstrophy.",
        },
        units={"time_s": "s", "relative diagnostics": "dimensionless"},
        notes=["This figure does not create a Test 2A validation or test protocol."],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite-accepted-assets",
        action="store_true",
        help="allow this script to recreate its accepted figure bundles",
    )
    return parser


def main(argv=None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.overwrite_accepted_assets:
        parser.error(
            "refusing to write accepted figure bundles without "
            "--overwrite-accepted-assets"
        )
    MAIN.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    figure_ml1_test2b()
    figure_ml1_test2a()
    figure_ml2(
        ["M1-X", "M1-Y", "warm M2-X", "H1", "H5"],
        MAIN / "ML2_posthoc_direct_history_test2b",
        "Main-text subset",
    )
    figure_ml2(MODEL_ORDER, SUPP / "ML2_posthoc_direct_history_test2b_all_models", "Complete accepted-model")
    figure_ml3_final()
    figure_ml3_history()
    figure_ml4()
    figure_ml5()
    for rep in "ABC":
        figure_ml6(
            rep,
            ["M1-X", "M1-Y", "H1", "H5"],
            MAIN / f"ML6_global_trajectories_representation_{rep}",
            "Main-text subset",
        )
        figure_ml6(
            rep,
            MODEL_ORDER,
            SUPP / f"ML6_global_trajectories_representation_{rep}_all_models",
            "Complete accepted-model",
        )
    figure_test2a_deployment()
    generated = sorted([p for p in (ROOT / "figures").rglob("*.png")])
    print(json.dumps({"status": "success", "png_figures": len(generated), "files": [str(p) for p in generated]}, indent=2))


if __name__ == "__main__":
    main()
