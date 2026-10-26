"""Plot optimization and validation results: truth vs optimized runs, errors, and parameter recovery.

For each optimization job directory this writes to <job-dir>/plots/:

- <field>__frame-NNN.png   truth | optimal | initial guess | optimal - truth, on a shared scale
- <field>__compare.webp    a movie of the same panels over all outputs (with --movie)
- errors__<field>.png      relative L2 and Linf errors over time, training window shaded
- statistics__<name>.png   statistic time series for truth, optimal and initial guess
- history.png              loss and gradient norm per optimizer iteration
- parameters.png           parameter values per iteration, with the true value marked

With --sweep, it also writes to <optimization-root>/plots/:

- recovery__<parameter>.png   recovered vs true parameter values, one colour per loss
- errors-by-loss__<field>.png final relative L2 error per truth run and loss

Run it with one MPI rank (no mpiexec): plotting reads whole checkpoints on a single process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import find_job_dirs, is_vector_quantity, load_yaml, read_csv

# Categorical slots validated together (all pairs) for colour-vision deficiency separation.
# Slot 3 sits below 3:1 contrast on the light surface, so every chart carries a legend and
# direct labels rather than relying on colour alone.
SERIES_COLORS = {"truth": "#2a78d6", "optimal": "#eb6834", "initial-guess": "#1baf7a"}
SERIES_LABELS = {"truth": "Truth", "optimal": "Optimized", "initial-guess": "Initial guess"}
LOSS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
WINDOW_FILL = "#f0efec"
# sequential (one hue, light to dark) and diverging (two poles, neutral gray midpoint) maps
SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_STEPS = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#8f2526"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for validation plotting."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", help="Optimization job directories, or optimization roots containing them")
    parser.add_argument("--fields", nargs="*", help="Field quantities to plot (defaults to the x variables)")
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="Output indices to render (defaults to first, end of training window, last)")
    parser.add_argument("--movie", action="store_true", help="Also render WEBP comparison movies")
    parser.add_argument("--statistics", nargs="*", default=None, help="Statistics to plot (defaults to all available)")
    parser.add_argument("--sweep", action="store_true", help="Also render sweep-level recovery and error-by-loss plots")
    return parser.parse_args()


def apply_style() -> None:
    """Set recessive axes, thin marks and the chart typeface for every figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["system-ui", "Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_PRIMARY,
    })


def sequential_cmap() -> Any:
    """Return the single-hue sequential colormap used for field values."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dimswe-sequential", SEQUENTIAL_STEPS)


def diverging_cmap() -> Any:
    """Return the blue-gray-red diverging colormap used for signed differences."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dimswe-diverging", DIVERGING_STEPS)


def validation_cases(job_dir: Path) -> dict[str, Path]:
    """Return the validated case directories of a job, keyed by case name."""
    cases = {}
    for case in ("optimal", "initial-guess"):
        case_dir = job_dir / "validation" / case
        if (case_dir / "sim.h5").exists():
            cases[case] = case_dir
    return cases


def label_line_end(axis: Any, x: list[float], y: list[float], text: str, color: str) -> None:
    """Directly label the end of a line in ink colour, with a small coloured marker beside it."""
    if not x:
        return
    axis.plot([x[-1]], [y[-1]], marker="o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    axis.annotate(
        text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
        va="center", ha="left", color=TEXT_SECONDARY, fontsize=9,
    )


def shade_training_window(axis: Any, window_time: list[float]) -> None:
    """Shade the training window behind the data and label it."""
    axis.axvspan(window_time[0], window_time[1], color=WINDOW_FILL, zorder=0, linewidth=0)
    axis.text(
        window_time[0], 1.0, " training window", transform=axis.get_xaxis_transform(),
        va="top", ha="left", color=TEXT_MUTED, fontsize=8,
    )


def scalar_view(function: Any) -> Any:
    """Return a scalar Function to draw: the field itself, or its magnitude for vectors."""
    from firedrake import Function, FunctionSpace, inner, sqrt

    if not is_vector_quantity(function.name()) and len(function.ufl_shape) == 0:
        return function
    mesh = function.function_space().mesh()
    element = function.function_space().ufl_element()
    degree = getattr(element, "embedded_superdegree", None)
    if not isinstance(degree, int):
        degree = element.degree() if isinstance(element.degree(), int) else max(element.degree())
    magnitude = Function(FunctionSpace(mesh, "CG", max(degree, 1)), name=f"|{function.name()}|")
    magnitude.interpolate(sqrt(inner(function, function)))
    return magnitude


def load_frame(checkpoint: Path, name: str, index: int) -> Any:
    """Load a scalar view of one field at one output index from a checkpoint."""
    from firedrake import CheckpointFile

    with CheckpointFile(str(checkpoint), "r") as checkpoint_file:
        mesh = checkpoint_file.load_mesh()
        return scalar_view(checkpoint_file.load_function(mesh, name, idx=index))


def field_panels(job_dir: Path, result: dict[str, Any], cases: dict[str, Path], name: str, index: int) -> list[dict[str, Any]]:
    """Load the panels for one field at one output: truth, cases, then optimal minus truth."""
    truth_dir = Path(result["truth_run_dirs"][0])
    panels = [{"key": "truth", "title": SERIES_LABELS["truth"], "function": load_frame(truth_dir / "sim.h5", name, index)}]
    for case in ("optimal", "initial-guess"):
        if case in cases:
            panels.append({"key": case, "title": SERIES_LABELS[case], "function": load_frame(cases[case] / "sim.h5", name, index)})
    if "optimal" in cases and (cases["optimal"] / "errors.h5").exists():
        from firedrake import CheckpointFile

        with CheckpointFile(str(cases["optimal"] / "errors.h5"), "r") as checkpoint_file:
            mesh = checkpoint_file.load_mesh()
            difference = checkpoint_file.load_function(mesh, f"diff_{name}", idx=index)
        if len(difference.ufl_shape) == 0:
            panels.append({"key": "difference", "title": "Optimized − truth", "function": difference})
        else:
            panels.append({"key": "difference-magnitude", "title": "|Optimized − truth|", "function": scalar_view(difference)})
    return panels


def value_limits(panels: list[dict[str, Any]]) -> tuple[tuple[float, float], float]:
    """Return the shared value range of the field panels and the symmetric difference limit."""
    import numpy as np

    values = [panel["function"].dat.data_ro for panel in panels if panel["key"] in SERIES_COLORS]
    low = min(float(np.min(data)) for data in values)
    high = max(float(np.max(data)) for data in values)
    differences = [panel["function"].dat.data_ro for panel in panels if panel["key"].startswith("difference")]
    limit = max((float(np.max(np.abs(data))) for data in differences), default=0.0)
    return (low, high if high > low else low + 1.0), (limit if limit > 0 else 1.0)


def draw_field_panels(figure: Any, axes: Any, panels: list[dict[str, Any]], frame_label: str) -> tuple[Any, Any]:
    """Draw field panels with one shared value colorbar and one difference colorbar."""
    from firedrake.pyplot import tripcolor

    (low, high), limit = value_limits(panels)
    value_artist = None
    difference_artist = None
    for axis, panel in zip(axes, panels, strict=True):
        axis.clear()
        axis.grid(False)
        if panel["key"] == "difference":
            artist = tripcolor(panel["function"], axes=axis, cmap=diverging_cmap(), vmin=-limit, vmax=limit)
            difference_artist = artist
        elif panel["key"] == "difference-magnitude":
            artist = tripcolor(panel["function"], axes=axis, cmap=sequential_cmap(), vmin=0.0, vmax=limit)
            difference_artist = artist
        else:
            artist = tripcolor(panel["function"], axes=axis, cmap=sequential_cmap(), vmin=low, vmax=high)
            value_artist = artist
        axis.set_title(f"{panel['title']} — {frame_label}", fontsize=10)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    return value_artist, difference_artist


def render_field_frame(job_dir: Path, result: dict[str, Any], cases: dict[str, Path], name: str, index: int, outdir: Path) -> None:
    """Render one comparison frame for a field."""
    import matplotlib.pyplot as plt

    panels = field_panels(job_dir, result, cases, name, index)
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), squeeze=False)
    value_artist, difference_artist = draw_field_panels(figure, axes[0], panels, f"output {index}")
    value_axes = [axis for axis, panel in zip(axes[0], panels, strict=True) if not panel["key"].startswith("difference")]
    difference_axes = [axis for axis, panel in zip(axes[0], panels, strict=True) if panel["key"].startswith("difference")]
    if value_artist is not None:
        bar = figure.colorbar(value_artist, ax=value_axes, orientation="horizontal", fraction=0.05, pad=0.04)
        bar.set_label(name, color=TEXT_SECONDARY)
        bar.outline.set_visible(False)
    if difference_artist is not None:
        bar = figure.colorbar(difference_artist, ax=difference_axes, orientation="horizontal", fraction=0.05, pad=0.04)
        bar.set_label(f"{name} difference", color=TEXT_SECONDARY)
        bar.outline.set_visible(False)
    figure.savefig(outdir / f"{name}__frame-{index:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_field_movie(job_dir: Path, result: dict[str, Any], cases: dict[str, Path], name: str, noutput: int, outdir: Path) -> None:
    """Render a WEBP movie of the comparison panels for a field."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    panels = field_panels(job_dir, result, cases, name, 0)
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), squeeze=False)

    def update(index: int) -> list[Any]:
        frame_panels = field_panels(job_dir, result, cases, name, index)
        draw_field_panels(figure, axes[0], frame_panels, f"output {index}")
        return list(axes[0])

    update(0)
    figure.tight_layout()
    movie = animation.FuncAnimation(figure, update, frames=range(noutput), interval=150, blit=False, repeat=False)
    movie.save(filename=outdir / f"{name}__compare.webp", writer="pillow")
    plt.close(figure)


def render_error_series(cases: dict[str, Path], name: str, window_time: list[float], outdir: Path) -> None:
    """Plot relative L2 and Linf errors over time as two small multiples (one measure each)."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for axis, metric, title in zip(axes, ("rel_l2", "rel_linf"), ("Relative L2 error", "Relative Linf error"), strict=True):
        shade_training_window(axis, window_time)
        plotted = False
        for case in ("optimal", "initial-guess"):
            if case not in cases:
                continue
            rows = read_csv(cases[case] / "errors.csv")
            column = f"{name}__{metric}"
            points = [(row["time"], row[column]) for row in rows if isinstance(row.get(column), float) and row[column] > 0]
            if not points:
                continue
            times, values = zip(*points)
            axis.plot(times, values, color=SERIES_COLORS[case], label=SERIES_LABELS[case])
            label_line_end(axis, list(times), list(values), SERIES_LABELS[case], SERIES_COLORS[case])
            plotted = True
        if plotted:
            axis.set_yscale("log")
        axis.set_title(f"{title} — {name}")
        axis.set_xlabel("time (s)")
        axis.margins(x=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) >= 2:
        figure.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04))
    figure.tight_layout()
    figure.savefig(outdir / f"errors__{name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def start_time(result: dict[str, Any], parameters: dict[str, Any], window_time: list[float]) -> float:
    """Return the run's initial time, recovered from the training-window start time."""
    window_offset = result["window"]["start_index"] * parameters["output"]["output_freq"] * parameters["timestepping"]["dt"]
    return window_time[0] - window_offset


def render_statistic(result: dict[str, Any], cases: dict[str, Path], name: str, window_time: list[float], outdir: Path) -> None:
    """Plot one statistic over time for truth, optimal and initial guess."""
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    truth_dir = Path(result["truth_run_dirs"][0])
    sources = {"truth": truth_dir}
    sources.update(cases)
    figure, axis = plt.subplots(figsize=(7, 3.8))
    shade_training_window(axis, window_time)
    plotted = 0
    for key, run_dir in sources.items():
        parameters = load_yaml(run_dir / "run.cfg")
        with h5py.File(run_dir / "sim.h5", "r") as h5file:
            if name not in h5file:
                continue
            series = np.array(h5file[name])
        stat_dt = parameters["timestepping"]["dt"] * parameters["output"]["stat_freq"]
        times = start_time(result, parameters, window_time) + np.arange(series.size) * stat_dt
        axis.plot(times, series, color=SERIES_COLORS[key], label=SERIES_LABELS[key],
                  linestyle="--" if key == "truth" else "-", zorder=3 if key == "truth" else 2)
        label_line_end(axis, list(times), list(series), SERIES_LABELS[key], SERIES_COLORS[key])
        plotted += 1
    if plotted == 0:
        plt.close(figure)
        return
    axis.set_title(name.replace("_", " "))
    axis.set_xlabel("time (s)")
    axis.margins(x=0.15)
    if plotted >= 2:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(outdir / f"statistics__{name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_history(job_dir: Path, result: dict[str, Any], outdir: Path) -> None:
    """Plot loss and gradient norm per iteration, and each parameter's path to its optimum."""
    import math

    import matplotlib.pyplot as plt

    rows = read_csv(job_dir / "history.csv")
    if not rows:
        return
    iterations = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for axis, column, title in zip(axes, ("loss", "grad_norm"), ("Loss", "Gradient norm (scaled variables)"), strict=True):
        points = [(i, row[column]) for i, row in zip(iterations, rows, strict=True)
                  if isinstance(row[column], float) and not math.isnan(row[column]) and row[column] > 0]
        if points:
            xs, ys = zip(*points)
            axis.plot(xs, ys, color=SERIES_COLORS["optimal"], marker="o", markersize=4)
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("iteration")
    figure.suptitle(f"{result['job_name']} — {result['optimizer']['message']}", color=TEXT_SECONDARY, fontsize=9)
    figure.tight_layout()
    figure.savefig(outdir / "history.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    parameters = [name for name, entry in (result.get("parameters") or {}).items() if entry.get("free", True)]
    if not parameters:
        return
    figure, axes = plt.subplots(1, len(parameters), figsize=(4.2 * len(parameters), 3.4), squeeze=False)
    for axis, name in zip(axes[0], parameters, strict=True):
        values = [row[name] for row in rows]
        axis.plot(iterations, values, color=SERIES_COLORS["optimal"], marker="o", markersize=4, label=SERIES_LABELS["optimal"])
        true_value = result["parameters"][name].get("true")
        if true_value is not None:
            axis.axhline(true_value, color=SERIES_COLORS["truth"], linestyle="--", linewidth=1.5, label=SERIES_LABELS["truth"])
            axis.annotate("true", (iterations[-1], true_value), xytext=(4, 4), textcoords="offset points", color=TEXT_SECONDARY, fontsize=8)
        axis.set_title(name)
        axis.set_xlabel("iteration")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(handles) >= 2:
        figure.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    figure.tight_layout()
    figure.savefig(outdir / "parameters.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


FIXED_FIELDS = ("coriolis", "bottom_topography")


def default_fields(summary: dict[str, Any], available: list[str]) -> list[str]:
    """Return the fields plotted by default: prognostic variables, then moisture diagnostics.

    Uses the field groups recorded by validate.py, so moist models include Qv, Qc, Qr and the
    moisture diagnostics (Q*_l, qsat, rh). Older validations without groups fall back to every
    recorded field except the fixed ones (coriolis, bottom topography).
    """
    groups = summary.get("fields")
    if groups:
        ordered = groups.get("prognostic", []) + groups.get("moisture", [])
        return [name for name in ordered if name in available]
    return [name for name in available if name not in FIXED_FIELDS]


def default_frames(result: dict[str, Any], noutput: int, output_freq: int) -> list[int]:
    """Return the first output, the end of the training window, and the last output."""
    window_end = result["window"]["start_index"] + result["window"]["num_steps"] // output_freq
    return sorted({0, min(window_end, noutput - 1), noutput - 1})


def plot_job(job_dir: Path, args: argparse.Namespace) -> None:
    """Render all per-job plots."""
    result = load_yaml(job_dir / "result.yaml")
    cases = validation_cases(job_dir)
    outdir = job_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    render_history(job_dir, result, outdir)
    if not cases:
        print(f"No validation runs for {job_dir}; run workflows/validate.sh first for field and error plots")
        return

    summary = load_yaml(cases["optimal"] / "errors-summary.yaml") if "optimal" in cases else {}
    window_time = summary.get("training_window_time", [0.0, 0.0])
    reference = load_yaml(next(iter(cases.values())) / "run.cfg")
    truth_parameters = load_yaml(Path(result["truth_run_dirs"][0]) / "run.cfg")
    output_freq = reference["output"]["output_freq"]
    noutput = min(
        reference["timestepping"]["num_steps"] // output_freq + 1,
        truth_parameters["timestepping"]["num_steps"] // output_freq + 1,
    )
    rows = read_csv(next(iter(cases.values())) / "errors.csv")
    available = sorted({key.split("__")[0] for key in rows[0] if key.endswith("__rel_l2")}) if rows else []
    fields = args.fields if args.fields else default_fields(summary, available)
    missing = [name for name in fields if name not in available]
    if missing:
        raise ValueError(f"No validation errors recorded for {missing}; available: {available}")
    frames = args.frames if args.frames is not None else default_frames(result, noutput, output_freq)

    for name in fields:
        render_error_series(cases, name, window_time, outdir)
        for index in frames:
            render_field_frame(job_dir, result, cases, name, index, outdir)
        if args.movie:
            render_field_movie(job_dir, result, cases, name, noutput, outdir)

    import h5py

    with h5py.File(Path(result["truth_run_dirs"][0]) / "sim.h5", "r") as h5file:
        statistics = [name for name in h5file.keys() if name.startswith("total_")]
    if args.statistics is not None:
        statistics = args.statistics
    for name in statistics:
        render_statistic(result, cases, name, window_time, outdir)
    print(f"Plots: {outdir}")


def render_sweep(root: Path) -> None:
    """Render recovered-vs-true and error-by-loss plots from a summary.csv."""
    import math

    import matplotlib.pyplot as plt

    summary_path = root / "summary.csv"
    if not summary_path.exists():
        print(f"No {summary_path}; run workflows/validate.sh on the optimization root first")
        return
    rows = [row for row in read_csv(summary_path) if row["case"] == "optimal"]
    losses = sorted({row["loss"] for row in rows})
    if len(losses) > len(LOSS_COLORS):
        raise ValueError(f"At most {len(LOSS_COLORS)} losses can be compared in one chart, got {losses}")
    colors = dict(zip(losses, LOSS_COLORS))
    markers = dict(zip(losses, ["o", "s", "^"]))
    outdir = root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    parameters = sorted({key[: -len("__true")] for row in rows for key in row if key.endswith("__true")})
    for name in parameters:
        points = [row for row in rows
                  if isinstance(row.get(f"{name}__true"), float) and math.isfinite(row[f"{name}__true"])
                  and isinstance(row.get(f"{name}__value"), float) and math.isfinite(row[f"{name}__value"])]
        if not points:
            continue
        figure, axis = plt.subplots(figsize=(4.8, 4.6))
        values = [row[f"{name}__true"] for row in points] + [row[f"{name}__value"] for row in points]
        low, high = min(values), max(values)
        pad = 0.05 * (high - low or abs(high) or 1.0)
        axis.plot([low - pad, high + pad], [low - pad, high + pad], color=BASELINE, linewidth=1.0, zorder=1)
        axis.text(0.97, 0.03, "on the line = exact recovery", transform=axis.transAxes, ha="right", va="bottom",
                  color=TEXT_MUTED, fontsize=8)
        for loss in losses:
            subset = [row for row in points if row["loss"] == loss]
            # hollow markers of different shapes stay visible when losses recover the same value
            axis.scatter([row[f"{name}__true"] for row in subset], [row[f"{name}__value"] for row in subset],
                         facecolors="none", edgecolors=colors[loss], marker=markers[loss],
                         s=[90, 60, 34][losses.index(loss)], linewidths=2, label=loss, zorder=3)
        axis.set_xlabel(f"true {name}")
        axis.set_ylabel(f"recovered {name}")
        axis.set_title(f"Parameter recovery — {name}")
        axis.set_aspect("equal", adjustable="datalim")
        if len(losses) >= 2:
            axis.legend(loc="upper left")
        figure.tight_layout()
        figure.savefig(outdir / f"recovery__{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

    fields = sorted({key.split("__")[0] for row in rows for key in row if key.endswith("__rel_l2__final")})
    truth_runs = sorted({row["truth_run"] for row in rows})
    offsets = {loss: (k - (len(losses) - 1) / 2) * 0.18 for k, loss in enumerate(losses)}
    skipped = []
    for field in fields:
        column = f"{field}__rel_l2__final"
        # errors are drawn on a log scale, so only positive finite values can be shown; zero or
        # NaN errors (for example a relative error of a field that is zero in the truth) are left out
        points = {loss: ([], []) for loss in losses}
        for loss in losses:
            for position, truth_run in enumerate(truth_runs):
                for row in rows:
                    value = row.get(column)
                    if (row["truth_run"] == truth_run and row["loss"] == loss and isinstance(value, float)
                            and math.isfinite(value) and value > 0):
                        points[loss][0].append(position + offsets[loss])
                        points[loss][1].append(value)
        if not any(ys for _, ys in points.values()):
            skipped.append(field)
            continue
        figure, axis = plt.subplots(figsize=(max(5.0, 0.9 * len(truth_runs) + 2.5), 3.8))
        for loss in losses:
            xs, ys = points[loss]
            if ys:
                axis.scatter(xs, ys, facecolors="none", edgecolors=colors[loss], marker=markers[loss], s=60, linewidths=2, label=loss, zorder=3)
        axis.set_yscale("log")
        axis.set_xticks(range(len(truth_runs)))
        axis.set_xticklabels(truth_runs, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel("relative L2 error, final time")
        axis.set_title(f"Validation error by loss — {field}")
        if len(losses) >= 2:
            axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(outdir / f"errors-by-loss__{field}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    if skipped:
        print(f"Skipped errors-by-loss plots for {skipped}: no positive, finite final relative L2 errors "
              "(zero errors, or NaN relative errors of fields that are zero in the truth)")
    print(f"Sweep plots: {outdir}")


def main() -> None:
    """Render plots for every requested job, and sweep plots when asked."""
    args = parse_args()
    apply_style()
    job_dirs = find_job_dirs(args.targets)
    for job_dir in job_dirs:
        plot_job(job_dir, args)
    if args.sweep:
        for root in sorted({job_dir.parent for job_dir in job_dirs}):
            render_sweep(root)


if __name__ == "__main__":
    main()
