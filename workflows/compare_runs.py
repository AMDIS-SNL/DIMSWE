"""Create multipanel comparison plots and movies for saved runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import is_vector_quantity, load_yaml, quantity_categories, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for comparison plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compare_cfg", help="Path to the comparison YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without rendering")
    return parser.parse_args()


def resolve_run_entry(entry: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve file system paths for a run entry."""
    run_dir = resolve_path(entry["run_dir"], base_dir)
    cfg_path = run_dir / "run.cfg"
    return {
        "label": entry.get("label", run_dir.name),
        "run_dir": run_dir,
        "cfg_path": cfg_path,
        "checkpoint": run_dir / "sim.h5",
    }


def load_run_context(entry: dict[str, Any]) -> dict[str, Any]:
    """Load run metadata needed for comparison plotting."""
    from dimswe.logger import EmptyLogger
    from dimswe.models import get_model

    parameters = load_yaml(entry["cfg_path"])
    model = get_model(parameters, EmptyLogger(), has_dynamics_statistics=True)
    categories = quantity_categories(model, parameters["output"]["output_aux_vars"])
    noutput = parameters["timestepping"]["num_steps"] // parameters["output"]["output_freq"] + 1
    return {
        "label": entry["label"],
        "run_dir": entry["run_dir"],
        "cfg_path": entry["cfg_path"],
        "checkpoint": entry["checkpoint"],
        "parameters": parameters,
        "categories": categories,
        "noutput": noutput,
    }


def classify_quantity(quantity: str, run_context: dict[str, Any]) -> str:
    """Determine which category a quantity belongs to."""
    for category, names in run_context["categories"].items():
        if quantity in names:
            return category
    raise ValueError(f"Quantity '{quantity}' is not available for run {run_context['label']}")


def load_statistic_series(run_context: dict[str, Any], quantity: str) -> Any:
    """Load a statistic dataset from a run checkpoint."""
    import numpy as np
    from firedrake import CheckpointFile

    with CheckpointFile(str(run_context["checkpoint"]), "r") as checkpoint:
        return np.array(checkpoint.h5pyfile[quantity])


def render_statistic_plot(run_contexts: list[dict[str, Any]], quantity: str, outdir: Path, fractional: bool) -> None:
    """Render an overlaid comparison plot for a statistic."""
    import matplotlib.pyplot as plt

    plt.figure()
    for run_context in run_contexts:
        data = load_statistic_series(run_context, quantity)
        if fractional:
            if data[0] == 0:
                series = data - data[0]
            else:
                series = (data - data[0]) / data[0] * 100.0
        else:
            series = data
        plt.plot(series, label=run_context["label"])
    plt.legend()
    plt.title(quantity if not fractional else f"{quantity} fractional change")
    suffix = "fractional-change" if fractional else "timeseries"
    plt.savefig(outdir / f"{quantity}__{suffix}.png")
    plt.close()


def draw_quantity_panel(axes: Any, checkpoint: Any, mesh: Any, quantity: str, output_index: int) -> Any:
    """Draw one quantity on an axes and return the artist."""
    from firedrake.pyplot import quiver, tripcolor

    field = checkpoint.load_function(mesh, quantity, idx=output_index)
    if is_vector_quantity(quantity):
        return quiver(field, axes=axes)
    return tripcolor(field, axes=axes, cmap="inferno")


def render_field_frame(run_contexts: list[dict[str, Any]], quantity: str, output_index: int, outdir: Path) -> None:
    """Render a side-by-side frame for a field quantity across runs."""
    import matplotlib.pyplot as plt
    from firedrake import CheckpointFile

    use_colorbar = not is_vector_quantity(quantity)
    ncols = len(run_contexts) + (1 if use_colorbar else 0)
    width_ratios = [1.0] * len(run_contexts) + ([0.05] if use_colorbar else [])
    figure, axes = plt.subplots(
        1,
        ncols,
        squeeze=False,
        figsize=(6 * len(run_contexts) + (0.8 if use_colorbar else 0.0), 4),
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes_row = axes[0, : len(run_contexts)]
    colorbar_axis = axes[0, -1] if use_colorbar else None
    color_artist = None
    for axis, run_context in zip(axes_row, run_contexts, strict=True):
        with CheckpointFile(str(run_context["checkpoint"]), "r") as checkpoint:
            mesh = checkpoint.load_mesh()
            color_artist = draw_quantity_panel(axis, checkpoint, mesh, quantity, output_index)
        axis.set_title(f"{run_context['label']} — {quantity} — frame {output_index}")
        axis.set_aspect("equal")
    if use_colorbar and color_artist is not None and colorbar_axis is not None:
        figure.colorbar(color_artist, cax=colorbar_axis)
    figure.tight_layout()
    figure.savefig(outdir / f"{quantity}__frame-{output_index:03d}.png", bbox_inches="tight")
    plt.close(figure)


def render_field_movie(run_contexts: list[dict[str, Any]], quantity: str, outdir: Path) -> None:
    """Render a WEBP movie comparing one quantity across runs."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from firedrake import CheckpointFile

    noutput = min(run_context["noutput"] for run_context in run_contexts)
    if noutput <= 0:
        raise ValueError(f"No output frames are available for quantity '{quantity}'")

    figure, axes = plt.subplots(1, len(run_contexts), squeeze=False, figsize=(6 * len(run_contexts), 4))
    axes_row = axes[0, :]

    def update(output_index: int) -> list[Any]:
        frame_artists: list[Any] = []
        for axis, run_context in zip(axes_row, run_contexts, strict=True):
            axis.clear()
            with CheckpointFile(str(run_context["checkpoint"]), "r") as checkpoint:
                mesh = checkpoint.load_mesh()
                artist = draw_quantity_panel(axis, checkpoint, mesh, quantity, output_index)
            axis.set_title(f"{run_context['label']} — {quantity} — frame {output_index}")
            axis.set_aspect("equal")
            text = axis.text(
                0.02,
                0.98,
                f"frame={output_index}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                color="w",
                bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=2),
            )
            if isinstance(artist, tuple):
                frame_artists.extend(list(artist))
            else:
                frame_artists.append(artist)
            frame_artists.append(text)
        return frame_artists

    update(0)
    figure.tight_layout(rect=(0.0, 0.0, 0.96, 1.0))
    movie = animation.FuncAnimation(figure, update, frames=range(noutput), interval=100, blit=False, repeat=False)
    movie.save(filename=outdir / f"{quantity}__compare.webp", writer="pillow")
    plt.close(figure)


def main() -> None:
    """Validate a comparison config and render requested outputs."""
    args = parse_args()
    compare_cfg_path = resolve_path(args.compare_cfg)
    compare_cfg = load_yaml(compare_cfg_path)
    base_dir = compare_cfg_path.parent
    run_contexts = [
        load_run_context(resolve_run_entry(entry, base_dir))
        for entry in compare_cfg.get("runs", [])
    ]
    if not run_contexts:
        raise ValueError("Comparison config must include at least one run")

    output_cfg = compare_cfg.get("output", {})
    outdir = resolve_path(output_cfg.get("outdir", "workflows/output"), base_dir)
    quantities_cfg = compare_cfg.get("quantities", {})
    field_quantities = quantities_cfg.get("fields", [])
    statistic_quantities = quantities_cfg.get("statistics", [])
    time_selection = compare_cfg.get("time_selection", {})
    frame_indices = time_selection.get("frames", [0])

    for quantity in field_quantities + statistic_quantities:
        classify_quantity(quantity, run_contexts[0])

    if args.dry_run:
        print(f"Validated comparison config: {compare_cfg_path}")
        print(f"Runs: {[run['label'] for run in run_contexts]}")
        print(f"Field quantities: {field_quantities}")
        print(f"Statistic quantities: {statistic_quantities}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    for quantity in statistic_quantities:
        render_statistic_plot(run_contexts, quantity, outdir, fractional=False)
        render_statistic_plot(run_contexts, quantity, outdir, fractional=True)
    for quantity in field_quantities:
        for output_index in frame_indices:
            render_field_frame(run_contexts, quantity, output_index, outdir)
        render_field_movie(run_contexts, quantity, outdir)


if __name__ == "__main__":
    main()
