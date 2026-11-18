"""Create multipanel comparison plots and movies for saved runs.

Field panels of all runs share one colour scale (fixed over the whole movie), and difference
panels <run> - <reference> are drawn on a symmetric diverging scale, with the reference set by the
optional `reference` key (default: the first run) and disabled with `differences: false`.
Differences are exact finite element differences when the runs' meshes are numbered
identically; otherwise both fields are point-evaluated on a common grid (`samples_per_cell`).
"""

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
    parser.add_argument(
        "--no-movies",
        action="store_true",
        help="Skip the WEBP movies (overrides output.movies in the config); PNG frames are still rendered",
    )
    return parser.parse_args()


def resolve_render_movies(output_cfg: dict[str, Any], no_movies_flag: bool) -> bool:
    """Return whether to render movies: output.movies (default true), unless --no-movies is given."""
    movies = output_cfg.get("movies", True)
    if not isinstance(movies, bool):
        raise ValueError(f"'output.movies' must be true or false, got {movies!r}")
    return movies and not no_movies_flag


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


def resolve_comparison_options(compare_cfg: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    """Return the reference run label and whether to draw difference panels.

    The optional top-level keys are `reference` (a run label, defaulting to the first run),
    `differences` (default true) and `samples_per_cell` (default 4, the sampling density per cell
    and direction for differences between differently numbered meshes). Differences are drawn as
    <run> - <reference> for every other run, so they need at least two runs.
    """
    if len(set(labels)) != len(labels):
        raise ValueError(f"Run labels must be unique to compute differences, got {labels}")
    reference = compare_cfg.get("reference", labels[0])
    if reference not in labels:
        raise ValueError(f"Reference run '{reference}' is not one of the run labels {labels}")
    differences = compare_cfg.get("differences", True)
    if not isinstance(differences, bool):
        raise ValueError(f"'differences' must be true or false, got {differences!r}")
    samples_per_cell = compare_cfg.get("samples_per_cell", 4)
    if isinstance(samples_per_cell, bool) or not isinstance(samples_per_cell, int) or samples_per_cell < 1:
        raise ValueError(f"'samples_per_cell' must be a positive integer, got {samples_per_cell!r}")
    return {
        "reference": reference,
        "differences": differences and len(labels) > 1,
        "samples_per_cell": samples_per_cell,
    }


def combine_limits(value_arrays: list[Any], difference_arrays: list[Any]) -> tuple[tuple[float, float], float]:
    """Return the shared (low, high) range of the value arrays and a symmetric difference limit.

    Degenerate ranges (constant fields, identical runs) are widened so colour normalization stays
    well defined. NaN entries (sample points outside the mesh) are ignored.
    """
    import numpy as np

    if value_arrays:
        low = min(float(np.nanmin(data)) for data in value_arrays)
        high = max(float(np.nanmax(data)) for data in value_arrays)
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + max(abs(low), 1.0) * 1e-12 + 1e-300
    limit = max((float(np.nanmax(np.abs(data))) for data in difference_arrays), default=0.0)
    return (low, high), (limit if limit > 0 else 1.0)


def merge_limits(
    first: tuple[tuple[float, float], float] | None, second: tuple[tuple[float, float], float]
) -> tuple[tuple[float, float], float]:
    """Return limits covering both sets of (value range, difference limit)."""
    if first is None:
        return second
    (low_a, high_a), limit_a = first
    (low_b, high_b), limit_b = second
    return (min(low_a, low_b), max(high_a, high_b)), max(limit_a, limit_b)


def transfer_to_reference(field: Any, reference_field: Any) -> Any:
    """Return `field` as a Function on the function space of `reference_field`.

    Each checkpoint is loaded with its own mesh object, so fields from different runs cannot be
    combined directly. This exact transfer needs identical coordinate dofs, ie the same mesh
    *and* the same cell/dof numbering, which is checked with meshes_match. Identical mesh cfgs
    are not enough: a mesh reloaded from a checkpoint is numbered according to how it was
    partitioned when written, which can differ between runs. The dofs are then copied onto the
    reference mesh in the field's own element, and interpolated onto the reference element if
    the two runs use different elements (for example H1 vs CF). Raises ValueError otherwise;
    callers then fall back to sampled_difference.
    """
    from firedrake import Function, FunctionSpace

    from workflows.opt_helpers import meshes_match

    reference_space = reference_field.function_space()
    source_space = field.function_space()
    if not meshes_match(source_space.mesh(), reference_space.mesh()):
        raise ValueError("runs do not share the same mesh numbering")
    same_mesh_copy = Function(FunctionSpace(reference_space.mesh(), source_space.ufl_element()))
    if same_mesh_copy.dat.data_ro.shape != field.dat.data_ro.shape:
        raise ValueError("dof layouts differ")
    same_mesh_copy.dat.data[:] = field.dat.data_ro
    if source_space.ufl_element() == reference_space.ufl_element():
        return same_mesh_copy
    return Function(reference_space).interpolate(same_mesh_copy)


_PRINTED_NOTES: set[str] = set()


def note_once(message: str) -> None:
    """Print a message only the first time it occurs (panels are reloaded for every frame)."""
    if message not in _PRINTED_NOTES:
        _PRINTED_NOTES.add(message)
        print(message)


def grid_axis_points(low: float, high: float, ncells: int, samples_per_cell: int) -> Any:
    """Return sample coordinates along one axis at the midpoints of a uniform sub-grid.

    With ncells uniform mesh cells per direction, the points (i + 0.5) / samples_per_cell inside
    every cell never lie on a cell edge, so point evaluation of discontinuous fields is unambiguous.
    """
    import numpy as np

    nsamples = ncells * samples_per_cell
    return low + (np.arange(nsamples) + 0.5) * (high - low) / nsamples


def sampling_grid(reference_field: Any, mesh_parameters: dict[str, Any], samples_per_cell: int) -> dict[str, Any]:
    """Return a regular 2D grid of sample points covering the reference mesh.

    The grid uses the mesh's bounding box and its nx/ny cell counts (from the run cfg, or the
    square root of the cell count if they are missing). Returns a dict with the flattened
    `points`, the grid `shape` (ny, nx) and the imshow `extent`.
    """
    import numpy as np

    mesh = reference_field.function_space().mesh()
    from dimswe.bernstein import ufl_attribute

    if ufl_attribute(mesh, 'geometric_dimension') != 2:
        raise ValueError("sampled differences are only implemented for 2D meshes")
    coordinates = mesh.coordinates.dat.data_ro
    (xlow, ylow), (xhigh, yhigh) = coordinates.min(axis=0), coordinates.max(axis=0)
    default_cells = max(int(round(np.sqrt(mesh.num_cells()))), 1)
    xs = grid_axis_points(xlow, xhigh, int(mesh_parameters.get("nx", default_cells)), samples_per_cell)
    ys = grid_axis_points(ylow, yhigh, int(mesh_parameters.get("ny", default_cells)), samples_per_cell)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return {
        "points": np.column_stack([grid_x.ravel(), grid_y.ravel()]),
        "shape": grid_x.shape,
        "extent": (float(xlow), float(xhigh), float(ylow), float(yhigh)),
    }


def sample_field(field: Any, points: Any) -> Any:
    """Evaluate a Function at the given points, with NaN for points outside its mesh."""
    import numpy as np

    values = field.at(points, dont_raise=True)
    samples = np.full((len(points),) + tuple(field.ufl_shape), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            samples[i] = value
    return samples


def sampled_difference(field: Any, reference_field: Any, grid: dict[str, Any]) -> tuple[Any, bool]:
    """Return (difference on the sampling grid, signed) for two fields on different meshes.

    Both fields are point-evaluated on the same grid, so this works regardless of mesh numbering
    or element. Scalar differences are signed; vector differences are returned as magnitudes.
    """
    import numpy as np

    difference = sample_field(field, grid["points"]) - sample_field(reference_field, grid["points"])
    signed = difference.ndim == 1
    if not signed:
        difference = np.linalg.norm(difference, axis=-1)
    return difference.reshape(grid["shape"]), signed


def load_field_panels(
    run_contexts: list[dict[str, Any]], quantity: str, output_index: int, options: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load the value panels of every run and the difference panels against the reference run.

    Each panel is a dict with `kind` ("value" or "difference"), `title`, and either `function`
    or, for sampled differences, `grid` and `extent`. Scalar differences are signed; vector
    differences are drawn as |run - reference|. Differences are exact finite element differences
    when the meshes are numbered identically, and are sampled on a regular grid otherwise.
    """
    from firedrake import CheckpointFile, Function

    from workflows.plot_validation import scalar_view

    fields = {}
    for run_context in run_contexts:
        with CheckpointFile(str(run_context["checkpoint"]), "r") as checkpoint:
            mesh = checkpoint.load_mesh()
            fields[run_context["label"]] = checkpoint.load_function(mesh, quantity, idx=output_index)

    panels = [
        {"kind": "value", "title": run_context["label"], "function": fields[run_context["label"]]}
        for run_context in run_contexts
    ]
    if not options["differences"]:
        return panels

    reference = options["reference"]
    reference_field = fields[reference]
    reference_context = next(run for run in run_contexts if run["label"] == reference)
    grid = None
    for run_context in run_contexts:
        label = run_context["label"]
        if label == reference:
            continue
        signed = len(reference_field.ufl_shape) == 0
        title = f"{label} − {reference}" if signed else f"|{label} − {reference}|"
        try:
            # exact finite element difference, when the meshes are numbered identically
            transferred = transfer_to_reference(fields[label], reference_field)
        except ValueError as error:
            # fall back to point-evaluating both fields on a common grid
            note_once(
                f"Note: {label} - {reference} difference for {quantity} is sampled on a grid "
                f"({options['samples_per_cell']} points per cell per direction), not exact: {error}"
            )
            if grid is None:
                grid = sampling_grid(
                    reference_field, reference_context["parameters"].get("mesh", {}), options["samples_per_cell"]
                )
            values, signed = sampled_difference(fields[label], reference_field, grid)
            panels.append({
                "kind": "difference", "title": title, "grid": values, "extent": grid["extent"], "signed": signed,
            })
            continue
        difference = Function(reference_field.function_space(), name=f"diff_{quantity}")
        difference.interpolate(transferred - reference_field)
        panels.append({
            "kind": "difference",
            "title": title,
            "function": difference if signed else scalar_view(difference),
            "signed": signed,
        })
    return panels


def panel_data(panel: dict[str, Any]) -> Any:
    """Return the array of values drawn in a panel (dofs of a Function, or a sampled grid)."""
    if "grid" in panel:
        return panel["grid"]
    return panel["function"].dat.data_ro


def panel_limits(panels: list[dict[str, Any]], quantity: str) -> tuple[tuple[float, float], float]:
    """Return the shared value range and difference limit of a set of panels.

    Vector value panels are drawn with quiver, so they do not contribute to the value range.
    """
    vector_values = is_vector_quantity(quantity)
    value_arrays = [panel_data(panel) for panel in panels if panel["kind"] == "value" and not vector_values]
    difference_arrays = [panel_data(panel) for panel in panels if panel["kind"] == "difference"]
    return combine_limits(value_arrays, difference_arrays)


def create_panel_figure(nruns: int, ndifferences: int, quantity: str) -> tuple[Any, list[Any], Any, Any]:
    """Create a one-row figure: run panels, value colorbar, difference panels, difference colorbar.

    Returns (figure, panel axes in panel order, value colorbar axis or None, difference colorbar
    axis or None).
    """
    import matplotlib.pyplot as plt

    use_value_colorbar = not is_vector_quantity(quantity)
    width_ratios = [1.0] * nruns + ([0.05] if use_value_colorbar else [])
    if ndifferences > 0:
        width_ratios += [1.0] * ndifferences + [0.05]
    figure, axes = plt.subplots(
        1,
        len(width_ratios),
        squeeze=False,
        figsize=(6 * (nruns + ndifferences) + 0.8 * (use_value_colorbar + (ndifferences > 0)), 4),
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes_row = list(axes[0])
    panel_axes = axes_row[:nruns]
    colorbar_axis = axes_row[nruns] if use_value_colorbar else None
    difference_colorbar_axis = None
    if ndifferences > 0:
        offset = nruns + (1 if use_value_colorbar else 0)
        panel_axes += axes_row[offset : offset + ndifferences]
        difference_colorbar_axis = axes_row[-1]
    return figure, panel_axes, colorbar_axis, difference_colorbar_axis


def panel_norms(limits: tuple[tuple[float, float], float], signed: bool) -> tuple[Any, Any]:
    """Return the matplotlib norms for the value panels and the difference panels."""
    from matplotlib.colors import Normalize

    (low, high), limit = limits
    difference_norm = Normalize(vmin=-limit, vmax=limit) if signed else Normalize(vmin=0.0, vmax=limit)
    return Normalize(vmin=low, vmax=high), difference_norm


def draw_field_panels(
    panel_axes: list[Any], panels: list[dict[str, Any]], quantity: str, output_index: int,
    limits: tuple[tuple[float, float], float],
) -> list[Any]:
    """Draw the panels with a shared value colour scale and a shared difference colour scale.

    Returns the drawn artists (used by the movie writer).
    """
    from firedrake.pyplot import quiver, tripcolor

    from workflows.plot_validation import diverging_cmap, sequential_cmap

    signed = all(panel.get("signed", True) for panel in panels if panel["kind"] == "difference")
    value_norm, difference_norm = panel_norms(limits, signed)
    artists: list[Any] = []
    for axis, panel in zip(panel_axes, panels, strict=True):
        axis.clear()
        if panel["kind"] == "value" and is_vector_quantity(quantity):
            artist = quiver(panel["function"], axes=axis)
        elif panel["kind"] == "value":
            artist = tripcolor(
                panel["function"], axes=axis, cmap=sequential_cmap(), vmin=value_norm.vmin, vmax=value_norm.vmax
            )
        elif "grid" in panel:
            cmap = diverging_cmap() if signed else sequential_cmap()
            artist = axis.imshow(
                panel["grid"], origin="lower", extent=panel["extent"], interpolation="nearest",
                cmap=cmap, vmin=difference_norm.vmin, vmax=difference_norm.vmax,
            )
        else:
            cmap = diverging_cmap() if signed else sequential_cmap()
            artist = tripcolor(
                panel["function"], axes=axis, cmap=cmap, vmin=difference_norm.vmin, vmax=difference_norm.vmax
            )
        axis.set_title(f"{panel['title']} — {quantity} — frame {output_index}")
        axis.set_aspect("equal")
        artists.extend(list(artist) if isinstance(artist, tuple) else [artist])
    return artists


def draw_colorbars(
    figure: Any, colorbar_axis: Any, difference_colorbar_axis: Any, panels: list[dict[str, Any]], quantity: str,
    limits: tuple[tuple[float, float], float],
) -> None:
    """Draw the value and difference colorbars from the shared norms (not from a single panel)."""
    from matplotlib.cm import ScalarMappable

    from workflows.plot_validation import diverging_cmap, sequential_cmap

    signed = all(panel.get("signed", True) for panel in panels if panel["kind"] == "difference")
    value_norm, difference_norm = panel_norms(limits, signed)
    if colorbar_axis is not None:
        figure.colorbar(ScalarMappable(norm=value_norm, cmap=sequential_cmap()), cax=colorbar_axis, label=quantity)
    if difference_colorbar_axis is not None:
        cmap = diverging_cmap() if signed else sequential_cmap()
        figure.colorbar(
            ScalarMappable(norm=difference_norm, cmap=cmap), cax=difference_colorbar_axis, label=f"{quantity} difference"
        )


def count_differences(panels: list[dict[str, Any]]) -> int:
    """Return the number of difference panels."""
    return sum(1 for panel in panels if panel["kind"] == "difference")


def render_field_frame(
    run_contexts: list[dict[str, Any]], quantity: str, output_index: int, outdir: Path, options: dict[str, Any]
) -> None:
    """Render a side-by-side frame for a field across runs, with differences against the reference."""
    import matplotlib.pyplot as plt

    panels = load_field_panels(run_contexts, quantity, output_index, options)
    limits = panel_limits(panels, quantity)
    figure, panel_axes, colorbar_axis, difference_colorbar_axis = create_panel_figure(
        len(run_contexts), count_differences(panels), quantity
    )
    draw_field_panels(panel_axes, panels, quantity, output_index, limits)
    draw_colorbars(figure, colorbar_axis, difference_colorbar_axis, panels, quantity, limits)
    figure.tight_layout()
    figure.savefig(outdir / f"{quantity}__frame-{output_index:03d}.png", bbox_inches="tight")
    plt.close(figure)


def render_field_movie(run_contexts: list[dict[str, Any]], quantity: str, outdir: Path, options: dict[str, Any]) -> None:
    """Render a WEBP movie comparing one quantity across runs.

    The colour scales are fixed over the whole movie, so a first pass over every output computes
    the value range and difference limit before any frame is drawn.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    noutput = min(run_context["noutput"] for run_context in run_contexts)
    if noutput <= 0:
        raise ValueError(f"No output frames are available for quantity '{quantity}'")

    limits = None
    difference_counts = set()
    for output_index in range(noutput):
        panels = load_field_panels(run_contexts, quantity, output_index, options)
        limits = merge_limits(limits, panel_limits(panels, quantity))
        difference_counts.add(count_differences(panels))
    if len(difference_counts) != 1:
        raise ValueError(f"Difference panels for '{quantity}' are not available at every output")
    ndifferences = difference_counts.pop()

    figure, panel_axes, colorbar_axis, difference_colorbar_axis = create_panel_figure(
        len(run_contexts), ndifferences, quantity
    )

    def update(output_index: int) -> list[Any]:
        frame_panels = load_field_panels(run_contexts, quantity, output_index, options)
        return draw_field_panels(panel_axes, frame_panels, quantity, output_index, limits)

    draw_colorbars(figure, colorbar_axis, difference_colorbar_axis, panels, quantity, limits)
    update(0)
    figure.tight_layout()
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
    render_movies = resolve_render_movies(output_cfg, args.no_movies)
    quantities_cfg = compare_cfg.get("quantities", {})
    field_quantities = quantities_cfg.get("fields", [])
    statistic_quantities = quantities_cfg.get("statistics", [])
    time_selection = compare_cfg.get("time_selection", {})
    frame_indices = time_selection.get("frames", [0])

    for quantity in field_quantities + statistic_quantities:
        for run_context in run_contexts:
            classify_quantity(quantity, run_context)
    options = resolve_comparison_options(compare_cfg, [run["label"] for run in run_contexts])

    if args.dry_run:
        print(f"Validated comparison config: {compare_cfg_path}")
        print(f"Runs: {[run['label'] for run in run_contexts]}")
        print(f"Field quantities: {field_quantities}")
        print(f"Statistic quantities: {statistic_quantities}")
        print(f"Reference run: {options['reference']} (differences: {options['differences']})")
        print(f"Movies: {render_movies}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    for quantity in statistic_quantities:
        render_statistic_plot(run_contexts, quantity, outdir, fractional=False)
        render_statistic_plot(run_contexts, quantity, outdir, fractional=True)
    for quantity in field_quantities:
        for output_index in frame_indices:
            render_field_frame(run_contexts, quantity, output_index, outdir, options)
        if render_movies:
            render_field_movie(run_contexts, quantity, outdir, options)


if __name__ == "__main__":
    main()
