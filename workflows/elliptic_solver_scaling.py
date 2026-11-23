"""Solver scaling study for elliptic problems (TODO-elliptic §4.12).

Runs a template elliptic cfg over a grid of resolutions, orders, solvers and coefficient
contrasts, and records KSP iterations, converged reasons, solve times and problem sizes from each
run's statistics. Writes runs/<name>/<case>/sim.h5 (+ run.cfg) per case, runs/<name>/summary.yaml,
runs/<name>/summary.csv and runs/<name>/iterations.png (iterations vs dofs, one series per
solver/order/contrast). For MPI-rank scaling, run the whole sweep once per rank count (the rank
count is recorded in every row) with --name suffixes, e.g.:

    bash workflows/elliptic_solver_scaling.sh 1 elliptic.cfg --nx 8 16 32 --solver riesz-amg riesz-gmg
    bash workflows/elliptic_solver_scaling.sh 4 elliptic.cfg --nx 16 32 64 --solver riesz-amg --name scaling-np4

--contrast c sets a kappa field that jumps by the factor c between the left and right halves of
the domain (needs a problem providing that coefficient; the default `piecewise` problem flag uses
the built-in two-material kappa of dimswe.elliptic_problems.TwoMaterialSineProblem).
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import (build_output_basename, deep_copy_parameters, dump_yaml, ensure_output_sections,
                              is_mpi_root, load_yaml, resolve_path, resolve_runs_root, root_print, set_by_dotted_path,
                              timestamp_utc, write_csv)
from workflows.elliptic_convergence import parse_overrides

RECORDED = ("ksp_iterations", "ksp_converged_reason", "solve_time", "num_dofs", "u_error")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cfg", help="Template elliptic cfg (model.type: elliptic)")
    parser.add_argument("--nx", type=int, nargs="+", required=True, help="Cells per direction (mesh.nx/ny/nz)")
    parser.add_argument("--order", type=int, nargs="+", default=None, help="Polynomial orders (default: cfg)")
    parser.add_argument("--solver", nargs="+", default=None, help="elliptic.solver values (default: cfg)")
    parser.add_argument("--contrast", type=float, nargs="+", default=None,
                        help="kappa jump factors (two-material problem; default: none)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                        help="Dotted-path override for every run (YAML value); may be repeated")
    parser.add_argument("--name", help="Sweep name (default: <cfg stem>-scaling)")
    parser.add_argument("--runs-root", help="Runs root directory (default: runs/)")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse runs whose sim.h5 exists")
    parser.add_argument("--no-plot", action="store_true", help="Do not write iterations.png")
    return parser.parse_args(argv)


def case_name(nx: int, order: int, solver: str, contrast: float | None) -> str:
    """Directory name of one case."""
    name = f"{solver}_p{order}_nx{nx}"
    return name if contrast is None else f"{name}_c{contrast:g}"


def case_parameters(template: dict[str, Any], overrides: dict[str, Any], nx: int, order: int, solver: str,
                    contrast: float | None, run_directory: Path) -> dict[str, Any]:
    """Parameters of one case (overrides first, then the swept values and the output path)."""
    parameters = deep_copy_parameters(template)
    for path, value in overrides.items():
        set_by_dotted_path(parameters, path, value)
    for key in ("nx", "ny", "nz"):
        parameters["mesh"][key] = int(nx)
    parameters["spatial-discretization"]["order"] = int(order)
    parameters.setdefault("elliptic", {})["solver"] = solver
    if contrast is not None:
        parameters.setdefault("elliptic-problem", {})["name"] = "two-material-sine"
        parameters["elliptic-problem"]["contrast"] = float(contrast)
        parameters["elliptic"].setdefault("coefficients", {})["kappa"] = {"kind": "problem"}
    ensure_output_sections(parameters)
    parameters["output"]["outfile_name"] = str(build_output_basename(run_directory))
    return parameters


def read_statistics(h5_path: str | Path) -> dict[str, float]:
    """The recorded scalar statistics of one run (missing ones are None)."""
    import h5py

    values: dict[str, float] = {}
    with h5py.File(h5_path, "r") as handle:
        for name in RECORDED:
            values[name] = float(handle[name][()].reshape(-1)[0]) if name in handle else None
    return values


def plot_iterations(rows: list[dict[str, Any]], path: Path) -> Path | None:
    """KSP iterations vs number of dofs (log x), one series per (solver, order, contrast)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dimswe.plotting.plot_elliptic import (AXIS, ERROR_SERIES, GRIDLINE, SURFACE, TEXT_SECONDARY,
                                               _style_axes)

    series: dict[tuple, list] = {}
    for row in rows:
        if row.get("num_dofs") and row.get("ksp_iterations") is not None:
            series.setdefault((row["solver"], row["order"], row["contrast"]), []).append(
                (row["num_dofs"], row["ksp_iterations"]))
    if not series:
        return None
    # Fixed categorical colors/markers (reference palette slots in order; folded after five).
    styles = list(ERROR_SERIES.values())
    fig, axes = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
    for index, (key, points) in enumerate(sorted(series.items(), key=lambda item: str(item[0]))):
        points.sort()
        _, color, marker = styles[min(index, len(styles) - 1)]
        solver, order, contrast = key
        label = f"{solver}, p={order}" + ("" if contrast is None else f", contrast {contrast:g}")
        axes.semilogx([p[0] for p in points], [p[1] for p in points], color=color, linewidth=2, marker=marker,
                      markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    axes.set_xlabel("degrees of freedom")
    axes.set_ylabel("KSP iterations")
    axes.set_ylim(bottom=0)
    _style_axes(axes)
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def run_scaling(template: dict[str, Any], args: argparse.Namespace, sweep_directory: Path) -> list[dict[str, Any]]:
    """Run (or reuse) every case and write the summary; returns the rows."""
    from mpi4py import MPI
    from run_model import run_model

    comm = MPI.COMM_WORLD
    overrides = parse_overrides(args.overrides)
    orders = args.order or [int(template["spatial-discretization"]["order"])]
    solvers = args.solver or [template.get("elliptic", {}).get("solver", "elliptic-direct")]
    contrasts = args.contrast or [None]
    rows = []
    for solver, order, contrast, nx in product(solvers, orders, contrasts, args.nx):
        run_directory = sweep_directory / case_name(nx, order, solver, contrast)
        parameters = case_parameters(template, overrides, nx, order, solver, contrast, run_directory)
        h5_path = Path(parameters["output"]["outfile_name"] + ".h5")
        exists = comm.bcast(h5_path.exists() if is_mpi_root() else None, root=0)
        if not (args.skip_existing and exists):
            if is_mpi_root():
                run_directory.mkdir(parents=True, exist_ok=True)
                dump_yaml(parameters, run_directory / "run.cfg")
            comm.Barrier()
            root_print(f"running {run_directory.name}")
            run_model(parameters)
        comm.Barrier()
        stats = comm.bcast(read_statistics(h5_path) if is_mpi_root() else None, root=0)
        rows.append({"solver": solver, "order": order, "contrast": contrast, "nx": nx,
                     "mpi_ranks": comm.Get_size(), **stats})
    if is_mpi_root():
        dump_yaml({"created": timestamp_utc(), "overrides": overrides, "rows": rows}, sweep_directory / "summary.yaml")
        write_csv(sweep_directory / "summary.csv", rows)
        if not args.no_plot:
            plot_iterations(rows, sweep_directory / "iterations.png")
    comm.Barrier()
    return rows


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    cfg_path = resolve_path(args.cfg)
    template = load_yaml(cfg_path)
    if template.get("model", {}).get("type") != "elliptic":
        raise ValueError("elliptic_solver_scaling needs a cfg with model.type: elliptic")
    sweep_directory = resolve_runs_root(args.runs_root) / (args.name or f"{cfg_path.stem}-scaling")
    rows = run_scaling(template, args, sweep_directory)
    for row in rows:
        root_print(f"{row['solver']:>12} p={row['order']} nx={row['nx']:>4} dofs={row['num_dofs']} "
                   f"its={row['ksp_iterations']} reason={row['ksp_converged_reason']} time={row['solve_time']}")
    root_print(f"summary written to {sweep_directory / 'summary.yaml'}")


if __name__ == "__main__":
    main()
