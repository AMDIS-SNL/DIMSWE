"""Validate optimized parameters by rerunning the model and measuring errors against the truth.

Step 3 of the optimization workflow. For each optimization job directory (one containing
result.yaml), this runs the model forward from the truth's initial time with the optimal
parameters and, optionally, with the initial guess as a baseline. At every output it compares
fields with the truth run and records:

- absolute and relative L2 errors: ||a - b||_L2 and ||a - b||_L2 / ||b||_L2
- absolute and relative Linf errors over the dofs (pointwise magnitude for vector fields)
- differences of scalar statistics (energy, mass, ...)

Outputs, written to <job-dir>/validation/<case>/ with case 'optimal' or 'initial-guess':

    run.cfg, sim.h5, sim.pvd   the validation run itself
    errors.csv                 per output time, per quantity errors
    errors-summary.yaml        final, max, mean, training-window and forecast maxima
    errors.h5                  difference fields (run minus truth) per output, for plotting

and a summary.csv across all validated jobs at the optimization root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import (
    deep_copy_parameters,
    find_job_dirs,
    dump_yaml,
    load_yaml,
    relative_error,
    root_print,
    summarize_series,
    timestamp_utc,
    to_builtin,
    write_csv,
)

CASES = ("optimal", "initial-guess")
# fields that never change during a run: comparing them only adds zero (or NaN) error columns
FIXED_FIELDS = ("coriolis", "bottom_topography")
# per-unit-mass forms of the moisture densities, and moisture diagnostics
MOISTURE_PROGNOSTICS = ("Qv", "Qc", "Qr")
ERROR_METRICS = ("l2", "rel_l2", "linf", "rel_linf")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for validation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", help="Optimization job directories, or optimization roots containing them")
    parser.add_argument("--num-steps", type=int, help="Steps to run (defaults to the truth run length, up to which errors are computed)")
    parser.add_argument("--no-initial-guess", action="store_true", help="Skip the initial-guess baseline run")
    parser.add_argument("--fields", nargs="*", help="Field quantities to compare (defaults to all x variables and diagnostics)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases whose errors-summary.yaml exists")
    parser.add_argument("--dry-run", action="store_true", help="List the validation runs without executing")
    return parser.parse_args()


def plan_validation(job_dir: Path, result: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the validation cases of a job: each truth run against optimal/initial-guess parameters."""
    cases = ["optimal"] if args.no_initial_guess else list(CASES)
    plans = []
    for truth_dir in result["truth_run_dirs"]:
        truth_name = Path(truth_dir).name
        for case in cases:
            suffix = case if len(result["truth_run_dirs"]) == 1 else f"{case}__{truth_name}"
            plans.append(
                {
                    "case": case,
                    "truth_dir": Path(truth_dir),
                    "truth_name": truth_name,
                    "out_dir": job_dir / "validation" / suffix,
                }
            )
    return plans


def case_parameters(result: dict[str, Any], case: str, truth_parameters: dict[str, Any]) -> dict[str, Any]:
    """Build the parameters of a validation run: the truth cfg with the case's parameter values.

    Using the truth cfg as the base (rather than optimal.cfg alone) keeps each truth run's own
    non-optimized settings when an optimization was done jointly over several truth runs.
    """
    from workflows.common import set_by_dotted_path

    parameters = deep_copy_parameters(truth_parameters)
    if result["target"] in ("coeffs", "ic-params"):
        for name, entry in result["parameters"].items():
            value = entry["optimal"] if case == "optimal" else entry["initial"]
            set_by_dotted_path(parameters, name, value)
            section = name.split(".")[0]
            if result["target"] == "coeffs":
                # validation runs use plain parameters, not coefficients
                parameters[section]["treat_as_coeffs"] = False
    elif case == "initial-guess":
        for name, value in result["initial_guess_parameters"].items():
            set_by_dotted_path(parameters, name, value)
    return parameters


class ErrorRecorder:
    """Compute field and statistic errors against the truth at every output step of a run."""

    def __init__(self, truth_checkpoint: Path, fields: list[str] | None, out_dir: Path, truth_noutput: int):
        self.truth_checkpoint = truth_checkpoint
        self.requested_fields = fields
        self.out_dir = out_dir
        self.truth_noutput = truth_noutput
        self.rows: list[dict[str, Any]] = []
        self.truth_file = None
        self.truth_mesh = None
        self.diff_file = None
        # per-quantity (truth, difference) work Functions, created once so every output of a
        # quantity is checkpointed on the same function space
        self.work: dict[str, tuple[Any, Any]] = {}

    def open(self) -> Any:
        """Open the truth checkpoint and the difference-field checkpoint, and return the truth mesh.

        The validation run is built on this mesh, so model fields and truth fields loaded through
        the same file handle share one mesh object.
        """
        from firedrake import CheckpointFile

        self.truth_file = CheckpointFile(str(self.truth_checkpoint), "r")
        self.truth_mesh = self.truth_file.load_mesh()
        self.diff_file = CheckpointFile(str(self.out_dir / "errors.h5"), "w")
        self.diff_file.save_mesh(self.truth_mesh)
        return self.truth_mesh

    def close(self) -> None:
        """Close any open checkpoint files."""
        if self.truth_file is not None:
            self.truth_file.close()
        if self.diff_file is not None:
            self.diff_file.close()

    def quantities(self, model: Any, xn_sub: dict[str, Any]) -> dict[str, Any]:
        """Return the field quantities to compare, by name."""
        available = {var: xn_sub[var] for var in model.get_x_var_list()}
        available.update(model.get_diagnostics())
        if self.requested_fields is None:
            return {name: field for name, field in available.items() if name not in FIXED_FIELDS}
        missing = [name for name in self.requested_fields if name not in available]
        if missing:
            raise ValueError(f"Unknown field quantities {missing}; available: {sorted(available)}")
        return {name: available[name] for name in self.requested_fields}

    def __call__(self, model: Any, xn: Any, xn_sub: dict[str, Any], t: Any, step: int, output_step: int) -> None:
        """Record errors for one output (run_model output_callback signature)."""
        from firedrake import Function

        from workflows.opt_helpers import copy_function_data

        if output_step >= self.truth_noutput:
            return
        row: dict[str, Any] = {"output_index": output_step, "step": step, "time": float(t)}
        for name, field in self.quantities(model, xn_sub).items():
            truth_loaded = self.truth_file.load_function(self.truth_mesh, name, idx=output_step)
            if name not in self.work:
                space = collapsed_space(field)
                self.work[name] = (Function(space, name=f"truth_{name}"), Function(space, name=f"diff_{name}"))
            truth, difference = self.work[name]
            copy_function_data(truth, truth_loaded)
            # x variables are components of a mixed Function, so subtract dof arrays directly
            difference.dat.data[:] = field.dat.data_ro - truth.dat.data_ro
            metrics = field_errors(model, difference, truth)
            for metric, value in metrics.items():
                row[f"{name}__{metric}"] = value
            self.diff_file.save_function(difference, idx=output_step)
        self.rows.append(row)

    def add_statistics(self, run_checkpoint: Path, statistics: list[str], stat_per_output: int) -> None:
        """Add statistic differences (at output times) by reading both checkpoints' datasets."""
        import h5py
        import numpy as np

        with h5py.File(self.truth_checkpoint, "r") as truth_h5, h5py.File(run_checkpoint, "r") as run_h5:
            for name in statistics:
                if name not in truth_h5 or name not in run_h5:
                    continue
                truth_series = np.array(truth_h5[name])
                run_series = np.array(run_h5[name])
                for row in self.rows:
                    index = row["output_index"] * stat_per_output
                    if index < len(truth_series) and index < len(run_series):
                        difference = float(run_series[index] - truth_series[index])
                        row[f"{name}__diff"] = difference
                        row[f"{name}__rel_diff"] = relative_error(abs(difference), abs(float(truth_series[index])))


def collapsed_space(field: Any) -> Any:
    """Return a standalone function space for a field that may be a mixed-Function component."""
    space = field.function_space()
    if getattr(space, "index", None) is not None or getattr(space, "parent", None) is not None:
        return space.collapse()
    return space


def field_errors(model: Any, difference: Any, truth: Any) -> dict[str, float]:
    """Return absolute and relative L2 and Linf norms of a difference field.

    L2 norms are integrated with the model's quadrature; Linf norms are the maximum dof magnitude
    (the pointwise vector magnitude for vector fields), reduced over MPI ranks.
    """
    from firedrake import assemble, inner

    l2 = float(assemble(inner(difference, difference) * model.spaces.dx)) ** 0.5
    truth_l2 = float(assemble(inner(truth, truth) * model.spaces.dx)) ** 0.5
    linf = dof_max_norm(difference)
    truth_linf = dof_max_norm(truth)
    return {
        "l2": l2,
        "rel_l2": relative_error(l2, truth_l2),
        "linf": linf,
        "rel_linf": relative_error(linf, truth_linf),
    }


def dof_max_norm(function: Any) -> float:
    """Return the maximum dof magnitude of a Function across MPI ranks."""
    import numpy as np
    from mpi4py import MPI

    data = function.dat.data_ro
    if data.size == 0:
        local = 0.0
    elif data.ndim > 1:
        local = float(np.max(np.linalg.norm(data, axis=-1)))
    else:
        local = float(np.max(np.abs(data)))
    return function.function_space().mesh().comm.allreduce(local, op=MPI.MAX)


def field_groups(model: Any, compared: list[str]) -> dict[str, list[str]]:
    """Group the compared fields so plotting can choose sensible defaults.

    prognostic: the model's x variables (v, h, S, and Qv, Qc, Qr for moist models)
    moisture: moisture diagnostics (the per-unit-mass Q*_l, qsat and relative humidity rh)
    diagnostic: every other compared diagnostic
    """
    prognostic = [name for name in model.get_x_var_list() if name in compared]
    moisture_names = {f"{name}_l" for name in MOISTURE_PROGNOSTICS} | {"qsat", "rh"}
    moisture = [name for name in compared if name in moisture_names]
    diagnostic = [name for name in compared if name not in prognostic and name not in moisture]
    return {"prognostic": prognostic, "moisture": moisture, "diagnostic": diagnostic}


def initial_state_setter(result: dict[str, Any], case: str, job_dir: Path) -> Any:
    """Return a run_model initial_state callback for ic-field jobs, or None."""
    if result["target"] != "ic-field":
        return None
    from workflows.opt_helpers import OPTIMAL_IC_NAME, load_state_into

    path = job_dir / ("optimal_ic.h5" if case == "optimal" else "initial_guess_ic.h5")

    def set_initial_state(model: Any, xn: Any, t: Any) -> None:
        load_state_into(path, OPTIMAL_IC_NAME, xn[0])

    return set_initial_state


def summarize_errors(rows: list[dict[str, Any]], train_end_time: float) -> dict[str, Any]:
    """Summarize every error column of an errors table."""
    times = [row["time"] for row in rows]
    columns = [key for key in rows[0] if key not in ("output_index", "step", "time")] if rows else []
    return {
        column: summarize_series(times, [row.get(column, float("nan")) for row in rows], train_end_time)
        for column in columns
    }


def run_case(job_dir: Path, result: dict[str, Any], plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run one validation case and write its error tables."""
    from mpi4py import MPI

    from run_model import run_model

    comm = MPI.COMM_WORLD
    out_dir = plan["out_dir"]
    truth_parameters = load_yaml(plan["truth_dir"] / "run.cfg")
    parameters = case_parameters(result, plan["case"], truth_parameters)
    if args.num_steps is not None:
        parameters["timestepping"]["num_steps"] = args.num_steps
    parameters["output"]["outfile_name"] = str(out_dir / "sim")
    truth_noutput = truth_parameters["timestepping"]["num_steps"] // truth_parameters["output"]["output_freq"] + 1
    if parameters["output"]["output_freq"] != truth_parameters["output"]["output_freq"]:
        raise ValueError("Validation output_freq must match the truth run")

    if comm.Get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(to_builtin(parameters), out_dir / "run.cfg")
    comm.barrier()

    recorder = ErrorRecorder(plan["truth_dir"] / "sim.h5", args.fields, out_dir, truth_noutput)
    mesh = recorder.open()
    try:
        model = run_model(
            parameters,
            mesh=mesh,
            initial_state=initial_state_setter(result, plan["case"], job_dir),
            output_callback=recorder,
        )
    finally:
        recorder.close()
    stat_per_output = parameters["output"]["output_freq"] // parameters["output"]["stat_freq"]
    if parameters["output"]["output_freq"] % parameters["output"]["stat_freq"] == 0:
        recorder.add_statistics(out_dir / "sim.h5", list(model.get_statistics_list()), stat_per_output)

    window = result["window"]
    t0 = float(model.initcond.get_t0())
    dt = parameters["timestepping"]["dt"]
    train_start = t0 + window["start_index"] * parameters["output"]["output_freq"] * dt
    train_end = train_start + window["num_steps"] * dt
    summary = {
        "created_at": timestamp_utc(),
        "job_name": result["job_name"],
        "case": plan["case"],
        "truth_run": plan["truth_name"],
        "target": result["target"],
        "loss_label": result["loss_label"],
        "training_window_time": [train_start, train_end],
        "fields": field_groups(model, [name for name in recorder.work]),
        "errors": summarize_errors(recorder.rows, train_end),
    }
    if comm.Get_rank() == 0:
        write_csv(out_dir / "errors.csv", recorder.rows)
        dump_yaml(to_builtin(summary), out_dir / "errors-summary.yaml")
    comm.barrier()
    return summary


def summary_row(result: dict[str, Any], plan: dict[str, Any], summary: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Build one row of the sweep-level summary table."""
    row: dict[str, Any] = {
        "job": result["job_name"],
        "case": plan["case"],
        "truth_run": plan["truth_name"],
        "target": result["target"],
        "loss": result["loss_label"],
        "initial_loss": result.get("initial_loss"),
        "final_loss": result.get("final_loss"),
        "optimizer_success": result["optimizer"]["success"],
        "nit": result["optimizer"]["nit"],
        "nfev": result["optimizer"]["nfev"],
    }
    for name, entry in (result.get("parameters") or {}).items():
        row[f"{name}__true"] = entry.get("true", float("nan"))
        row[f"{name}__value"] = entry["optimal"] if plan["case"] == "optimal" else entry["initial"]
        if "true" in entry:
            value = row[f"{name}__value"]
            row[f"{name}__rel_error"] = relative_error(abs(value - entry["true"]), abs(entry["true"]))
    for field in fields:
        for metric in ("rel_l2", "rel_linf"):
            column = f"{field}__{metric}"
            if column in summary["errors"]:
                row[f"{column}__final"] = summary["errors"][column]["final"]
                row[f"{column}__max"] = summary["errors"][column]["max"]
    return row


def main() -> None:
    """Validate every requested optimization job."""
    args = parse_args()
    job_dirs = find_job_dirs(args.targets)
    planned = []
    for job_dir in job_dirs:
        result = load_yaml(job_dir / "result.yaml")
        for plan in plan_validation(job_dir, result, args):
            planned.append((job_dir, result, plan))
    for _, result, plan in planned:
        root_print(f"Validation: {result['job_name']} case={plan['case']} truth={plan['truth_name']} -> {plan['out_dir']}")
    if args.dry_run:
        return

    from mpi4py import MPI

    rows_by_root: dict[Path, list[dict[str, Any]]] = {}
    for job_dir, result, plan in planned:
        summary_path = plan["out_dir"] / "errors-summary.yaml"
        if args.skip_existing and summary_path.exists():
            root_print(f"Skipping existing validation {plan['out_dir']}")
            summary = load_yaml(summary_path)
        else:
            summary = run_case(job_dir, result, plan, args)
        fields = sorted({column.split("__")[0] for column in summary["errors"] if column.endswith("__rel_l2")} - set(FIXED_FIELDS))
        rows_by_root.setdefault(job_dir.parent, []).append(summary_row(result, plan, summary, fields))

    if MPI.COMM_WORLD.Get_rank() == 0:
        for root, rows in rows_by_root.items():
            write_csv(root / "summary.csv", rows)
            print(f"Summary: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
