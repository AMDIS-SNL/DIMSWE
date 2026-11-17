"""Run an nx refinement sweep of an elliptic cfg and summarize the observed convergence rates.

For each nx the template cfg (with optional dotted overrides) is run through run_model into
runs/<name>/nx<nx>/sim.h5 (with run.cfg provenance). The error statistics ('*_error') are read
back from each sim.h5, observed rates log(e_coarse / e_fine) / log(nx_fine / nx_coarse) are
computed between consecutive resolutions, and summary.yaml / summary.csv are written to the
sweep directory, together with a log-log convergence plot (convergence.png, unless --no-plot).

Example:
    bash workflows/elliptic_convergence.sh 1 elliptic.cfg --nx 8 16 32 \\
        --set elliptic.form_degree=2 --set elliptic.complex=grad-curl
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from workflows.common import (
    build_output_basename,
    deep_copy_parameters,
    dump_yaml,
    ensure_output_sections,
    is_mpi_root,
    load_yaml,
    resolve_path,
    resolve_runs_root,
    root_print,
    set_by_dotted_path,
    timestamp_utc,
    write_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cfg", help="Template elliptic cfg (model.type: elliptic)")
    parser.add_argument("--nx", type=int, nargs="+", required=True,
                        help="Cells per direction for each run (sets mesh.nx, mesh.ny and mesh.nz)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                        help="Dotted-path override applied to every run, value parsed as YAML "
                             "(e.g. elliptic.form_degree=2); may be repeated")
    parser.add_argument("--name", help="Sweep name (default: the cfg file stem)")
    parser.add_argument("--runs-root", help="Runs root directory (default: runs/)")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse runs whose sim.h5 already exists")
    parser.add_argument("--no-plot", action="store_true", help="Do not write convergence.png")
    return parser.parse_args(argv)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ['a.b=1', 'c=[x]'] into {'a.b': 1, 'c': ['x']} (values parsed as YAML)."""
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override '{item}' must have the form PATH=VALUE")
        path, value = item.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"override '{item}' has an empty path")
        overrides[path] = yaml.safe_load(value)
    return overrides


def refinement_parameters(template: dict[str, Any], nx: int, overrides: dict[str, Any],
                          run_directory: Path) -> dict[str, Any]:
    """Parameters of one refinement run: overrides, then mesh size, then the output path."""
    parameters = deep_copy_parameters(template)
    for path, value in overrides.items():
        set_by_dotted_path(parameters, path, value)
    for key in ("nx", "ny", "nz"):
        parameters["mesh"][key] = int(nx)
    ensure_output_sections(parameters)
    parameters["output"]["outfile_name"] = str(build_output_basename(run_directory))
    return parameters


def read_error_statistics(h5_path: str | Path) -> dict[str, float]:
    """Read the scalar '*_error' statistics (first entry of each dataset) from a sim.h5."""
    import h5py

    errors: dict[str, float] = {}
    with h5py.File(h5_path, "r") as handle:
        for name, dataset in handle.items():
            if name.endswith("_error") and isinstance(dataset, h5py.Dataset) and dataset.size > 0:
                errors[name] = float(dataset[()].reshape(-1)[0])
    return errors


def observed_rate(e_coarse: float, e_fine: float, nx_coarse: int, nx_fine: int) -> float | None:
    """Observed order log(e_coarse / e_fine) / log(nx_fine / nx_coarse), or None if undefined."""
    if not (e_coarse > 0.0 and e_fine > 0.0) or nx_fine == nx_coarse:
        return None
    if not (math.isfinite(e_coarse) and math.isfinite(e_fine)):
        return None
    return math.log(e_coarse / e_fine) / math.log(nx_fine / nx_coarse)


def compute_observed_rates(nxs: list[int], errors: list[dict[str, float]]) -> list[dict[str, Any]]:
    """One row per resolution: nx, each error, and the rate from the previous resolution.

    Rates are 'rate_<name>' entries (None for the first row, or when an error is zero/missing).
    Resolutions are sorted by nx first.
    """
    if len(nxs) != len(errors):
        raise ValueError("need one error dictionary per resolution")
    order = sorted(range(len(nxs)), key=lambda i: nxs[i])
    names = sorted({name for e in errors for name in e})
    rows: list[dict[str, Any]] = []
    previous = None
    for i in order:
        row: dict[str, Any] = {"nx": int(nxs[i])}
        for name in names:
            row[name] = errors[i].get(name)
        for name in names:
            rate = None
            if previous is not None:
                e0, e1 = errors[previous].get(name), errors[i].get(name)
                if e0 is not None and e1 is not None:
                    rate = observed_rate(e0, e1, nxs[previous], nxs[i])
            row["rate_" + name] = rate
        rows.append(row)
        previous = i
    return rows


def run_sweep(template: dict[str, Any], nxs: list[int], overrides: dict[str, Any], sweep_directory: Path,
              skip_existing: bool = False, plot: bool = True) -> list[dict[str, Any]]:
    """Run (or reuse) every resolution, then write and return the summary rows.

    All MPI ranks run the simulations together; rank 0 writes provenance and summary files and,
    if plot is True, convergence.png (dimswe.plotting.plot_elliptic.plot_convergence).
    """
    from mpi4py import MPI
    from run_model import run_model

    comm = MPI.COMM_WORLD
    errors = []
    for nx in nxs:
        run_directory = sweep_directory / f"nx{nx}"
        parameters = refinement_parameters(template, nx, overrides, run_directory)
        h5_path = Path(parameters["output"]["outfile_name"] + ".h5")
        exists = comm.bcast(h5_path.exists() if is_mpi_root() else None, root=0)
        if skip_existing and exists:
            root_print(f"reusing {h5_path}")
        else:
            if is_mpi_root():
                run_directory.mkdir(parents=True, exist_ok=True)
                dump_yaml(parameters, run_directory / "run.cfg")
            comm.Barrier()
            root_print(f"running nx = {nx} -> {h5_path}")
            run_model(parameters)
        comm.Barrier()
        errors.append(comm.bcast(read_error_statistics(h5_path) if is_mpi_root() else None, root=0))

    rows = compute_observed_rates(nxs, errors)
    if is_mpi_root():
        # The effective elliptic/spatial settings after the overrides (identical for every nx).
        effective = refinement_parameters(template, nxs[0], overrides, sweep_directory)
        summary = {
            "created": timestamp_utc(),
            "overrides": overrides,
            "nx": [int(n) for n in sorted(nxs)],
            "elliptic": effective.get("elliptic", {}),
            "spatial-discretization": effective.get("spatial-discretization", {}),
            "rows": rows,
        }
        dump_yaml(summary, sweep_directory / "summary.yaml")
        write_csv(sweep_directory / "summary.csv", rows)
        if plot:
            from dimswe.plotting.plot_elliptic import plot_convergence
            plot_convergence(summary, sweep_directory / "convergence.png")
    comm.Barrier()
    return rows


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    cfg_path = resolve_path(args.cfg)
    template = load_yaml(cfg_path)
    if template.get("model", {}).get("type") != "elliptic":
        raise ValueError("elliptic_convergence needs a cfg with model.type: elliptic")
    overrides = parse_overrides(args.overrides)
    name = args.name or cfg_path.stem
    sweep_directory = resolve_runs_root(args.runs_root) / name
    rows = run_sweep(template, args.nx, overrides, sweep_directory, skip_existing=args.skip_existing,
                     plot=not args.no_plot)
    for row in rows:
        rates = {k[5:]: (None if v is None else round(v, 3)) for k, v in row.items() if k.startswith("rate_")}
        root_print(f"nx = {row['nx']}: rates {rates}")
    root_print(f"summary written to {sweep_directory / 'summary.yaml'}")


if __name__ == "__main__":
    main()
