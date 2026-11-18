"""Compare optimization jobs across optimizer methods using their recorded metrics.

Reads the result.yaml files written by optimize_params.py and builds one comparison row per job.
Nothing is parsed from optimizer logs, and the model is never run. A job is located either
directly, as a directory containing result.yaml, or through an optimization root, whose
subdirectories are jobs.

Jobs solve the same problem when they share target and job name (<truth-run>__<loss-label>).
This holds for the paired first- and second-order examples in workflows/examples/, which differ
only in optimization_name and optimizer settings. Each row records:

    optimization, job, problem          where the job came from and which problem it solves
    target, loss, truth_runs
    method, use_hessp, hessp_used       optimizer method and whether HVPs reached scipy
    success, status, message            scipy's outcome
    nit, nfev, njev, nhvp               optimizer iterations, and actual objective, gradient and
                                        Hessian-vector-product evaluations
    initial_loss, final_loss
    loss_reduction                      final_loss / initial_loss
    optimize_seconds                    optimizer wall time (NaN for results written before it
                                        was recorded)
    final_loss_vs_best                  final_loss / the lowest final_loss of the same problem
    nmethods                            number of jobs solving the same problem
    <parameter>__rel_error              relative parameter error, when the truth is known

Results written before HVP support have no method metadata or nhvp; defaults are used for them
(see optimize_params.optimizer_summary).

write_comparison saves the rows as optimizer-comparison.csv. It also saves
optimizer-comparison.yaml, which groups the rows by problem and records, per problem, the
methods with the lowest final loss, the fewest gradients and the shortest optimizer time.
format_comparison_table renders the rows as a plain-text table for printing.

Command line (no Firedrake or MPI needed):

    bash workflows/compare_optimizers.sh <job-dir-or-root> [<job-dir-or-root> ...]
        [--output-dir DIR] [--no-save] [--only SUBSTRING ...] [--columns COLUMN ...]
        [--save-baseline FILE] [--check-baseline FILE [--count-rtol R] [--loss-factor F]
         [--time-factor T]]

prints the comparison table and saves both files to --output-dir. --output-dir defaults to the
common parent directory of the jobs' optimization roots: the root itself for a single root, and
for example runs/optimizations for the paired example roots.

Regression baselines (step 19). --save-baseline FILE records, per job (keyed by
'<optimization>/<job>'), the metrics in BASELINE_METRICS plus optimize_seconds. --check-baseline
FILE compares the current jobs with a saved baseline (see check_against_baseline). It exits with
status 1 if any job regressed: it is missing, it no longer converges, it needs more evaluations
than the baseline allows, or its final loss is worse by more than --loss-factor. Wall time is only
checked with --time-factor, since it depends on the machine. workflows/optimizer_baseline.sh runs
the representative problem and saves or checks workflows/baselines/doublevortex-params.yaml.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import dump_yaml, find_job_dirs, load_yaml, timestamp_utc, to_builtin, write_csv

BASELINE_VERSION = 1

COMPARISON_CSV = "optimizer-comparison.csv"
COMPARISON_YAML = "optimizer-comparison.yaml"

# leading columns of every comparison row, in order; per-parameter error columns follow
COMPARISON_COLUMNS = (
    "optimization", "job", "problem", "target", "loss", "truth_runs",
    "method", "use_hessp", "hessp_used", "success", "status", "message",
    "nit", "nfev", "njev", "nhvp",
    "initial_loss", "final_loss", "loss_reduction", "optimize_seconds",
    "final_loss_vs_best", "nmethods",
)

# columns of the printed table: the metrics the comparison is about, without the long message
TABLE_COLUMNS = (
    "problem", "optimization", "method", "success", "nit", "nfev", "njev", "nhvp",
    "final_loss", "final_loss_vs_best", "optimize_seconds",
)

# rankings recorded per problem in the YAML summary: (name, column); lower values are better
RANKINGS = (
    ("lowest_final_loss", "final_loss"),
    ("fewest_gradients", "njev"),
    ("shortest_time", "optimize_seconds"),
)


def as_float(value: Any) -> float:
    """Return value as a float, or NaN if it is missing or not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def problem_key(target: str, job_name: str) -> str:
    """Return the key identifying the problem a job solves.

    Jobs from different optimization roots solve the same problem when they share target and job
    name, since the job name encodes the truth run (or joint group) and the loss label.
    """
    return f"{target}:{job_name}"


def comparison_row(job_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Build the comparison row of one job from its result.yaml contents.

    job_dir is the job directory; its parent's name is used as the optimization name. Missing
    entries (older results) default to empty strings, zero counts or NaN.
    """
    optimizer = result.get("optimizer") or {}
    initial_loss = as_float(result.get("initial_loss"))
    final_loss = as_float(result.get("final_loss"))
    job_name = result.get("job_name", job_dir.name)
    target = result.get("target", "")
    row: dict[str, Any] = {
        "optimization": job_dir.parent.name,
        "job": job_name,
        "problem": problem_key(target, job_name),
        "target": target,
        "loss": result.get("loss_label", ""),
        "truth_runs": " ".join(str(name) for name in result.get("truth_runs") or []),
        "method": optimizer.get("method", ""),
        "use_hessp": optimizer.get("use_hessp"),
        "hessp_used": bool(optimizer.get("hessp_used", False)),
        "success": optimizer.get("success"),
        "status": optimizer.get("status"),
        "message": optimizer.get("message", ""),
        "nit": int(optimizer.get("nit", -1)),
        "nfev": int(optimizer.get("nfev", 0)),
        "njev": int(optimizer.get("njev", 0)),
        "nhvp": int(optimizer.get("nhvp", 0)),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        # a ratio is only meaningful for a positive initial loss
        "loss_reduction": final_loss / initial_loss if initial_loss > 0 else float("nan"),
        "optimize_seconds": as_float(result.get("optimize_seconds")),
        # filled in by add_problem_comparisons, once all jobs are known
        "final_loss_vs_best": float("nan"),
        "nmethods": 1,
    }
    for name, entry in (result.get("parameters") or {}).items():
        if "rel_error" in entry:
            row[f"{name}__rel_error"] = as_float(entry["rel_error"])
    return row


def add_problem_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in final_loss_vs_best and nmethods by comparing the rows of each problem.

    final_loss_vs_best is the final loss divided by the lowest finite final loss among the jobs
    of the same problem. It is 1 for the best job. It is NaN when the job's loss is not finite,
    or when the best loss is not positive (a ratio would be meaningless). Rows are modified in
    place and also returned.
    """
    for problem_rows in group_by_problem(rows).values():
        losses = [row["final_loss"] for row in problem_rows if math.isfinite(row["final_loss"])]
        best = min(losses) if losses else float("nan")
        for row in problem_rows:
            row["nmethods"] = len(problem_rows)
            if math.isfinite(row["final_loss"]) and best > 0:
                row["final_loss_vs_best"] = row["final_loss"] / best
            else:
                row["final_loss_vs_best"] = float("nan")
    return rows


def group_by_problem(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the rows grouped by problem key, in first-seen order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["problem"], []).append(row)
    return groups


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the rows sorted by problem, then optimization name and method, so pairs are adjacent."""
    return sorted(rows, key=lambda row: (row["problem"], row["optimization"], str(row["method"])))


def collect_comparison_rows(targets: list[str]) -> list[dict[str, Any]]:
    """Return sorted comparison rows for every job under the given job directories or roots.

    targets are job directories or optimization roots, as accepted by
    workflows.common.find_job_dirs. A job listed twice (directly and through its root) is read
    once.
    """
    return rows_for_job_dirs(resolve_job_dirs(targets))


def resolve_job_dirs(targets: list[str]) -> list[Path]:
    """Return the resolved job directories under targets, without duplicates, in first-seen order."""
    return list(dict.fromkeys(path.resolve() for path in find_job_dirs(targets)))


def rows_for_job_dirs(job_dirs: list[Path]) -> list[dict[str, Any]]:
    """Return sorted, compared rows for the given job directories (each containing result.yaml)."""
    rows = [comparison_row(job_dir, load_yaml(job_dir / "result.yaml")) for job_dir in job_dirs]
    return sort_rows(add_problem_comparisons(rows))


def best_by(problem_rows: list[dict[str, Any]], column: str) -> list[str]:
    """Return the labels of the jobs with the lowest finite value of column (ties included).

    Labels are '<optimization>/<method>', so jobs with the same method from different roots
    stay distinguishable. Returns an empty list when no job has a finite value.
    """
    values = [(as_float(row[column]), row) for row in problem_rows]
    finite = [value for value, _ in values if math.isfinite(value)]
    if not finite:
        return []
    lowest = min(finite)
    return [f"{row['optimization']}/{row['method']}" for value, row in values if value == lowest]


def comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the YAML summary: per problem, the jobs' metrics and the best jobs per ranking."""
    problems: dict[str, Any] = {}
    for key, problem_rows in group_by_problem(rows).items():
        problems[key] = {
            "jobs": [{column: row[column] for column in COMPARISON_COLUMNS if column != "problem"}
                     for row in problem_rows],
            "best": {name: best_by(problem_rows, column) for name, column in RANKINGS},
        }
    return {"created_at": timestamp_utc(), "columns": list(COMPARISON_COLUMNS), "problems": problems}


def write_comparison(rows: list[dict[str, Any]], out_dir: str | Path) -> tuple[Path, Path]:
    """Write optimizer-comparison.csv and optimizer-comparison.yaml to out_dir and return both paths.

    The CSV has the COMPARISON_COLUMNS first, then the per-parameter error columns of any job.
    Jobs without a column leave it empty.
    """
    out_dir = Path(out_dir)
    columns = list(COMPARISON_COLUMNS) + sorted({key for row in rows for key in row} - set(COMPARISON_COLUMNS))
    # write_csv takes the union of keys in first-seen order, so give every row every column in order
    ordered = [{column: row.get(column, "") for column in columns} for row in rows]
    csv_path = out_dir / COMPARISON_CSV
    yaml_path = out_dir / COMPARISON_YAML
    write_csv(csv_path, ordered)
    dump_yaml(to_builtin(comparison_summary(rows)), yaml_path)
    return csv_path, yaml_path


def format_value(value: Any) -> str:
    """Format one table cell: floats in compact scientific or fixed notation, others as text."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def format_comparison_table(rows: list[dict[str, Any]], columns: tuple[str, ...] = TABLE_COLUMNS) -> str:
    """Return the rows as a plain-text table with aligned columns and a header line."""
    cells = [[format_value(row.get(column, "")) for column in columns] for row in rows]
    widths = [max([len(column)] + [len(line[index]) for line in cells]) for index, column in enumerate(columns)]
    lines = ["  ".join(column.ljust(width) for column, width in zip(columns, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(line, widths)) for line in cells)
    return "\n".join(lines)


# metrics recorded per job in a baseline; optimize_seconds is recorded too, but only checked on request
BASELINE_METRICS = ("method", "target", "success", "status", "nit", "nfev", "njev", "nhvp", "initial_loss", "final_loss")
# evaluation counts checked against the baseline with --count-rtol
BASELINE_COUNTS = ("nit", "nfev", "njev", "nhvp")


def baseline_key(row: dict[str, Any]) -> str:
    """Return the key of a job in a baseline: '<optimization>/<job>', unique per job directory."""
    return f"{row['optimization']}/{row['job']}"


def baseline_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the baseline mapping of the given comparison rows, ready to be written as YAML."""
    jobs = {}
    for row in rows:
        entry = {metric: row[metric] for metric in BASELINE_METRICS}
        entry["optimize_seconds"] = row["optimize_seconds"]
        jobs[baseline_key(row)] = entry
    return {"version": BASELINE_VERSION, "created_at": timestamp_utc(), "jobs": jobs}


def save_baseline(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write the baseline of the given rows to path (creating its directory) and return the path."""
    path = Path(path).resolve()
    dump_yaml(to_builtin(baseline_from_rows(rows)), path)
    return path


def check_against_baseline(rows: list[dict[str, Any]], baseline: dict[str, Any], count_rtol: float = 0.25,
                           loss_factor: float = 10.0, time_factor: float | None = None) -> tuple[list[str], list[str]]:
    """Compare comparison rows with a saved baseline and return (regressions, notes).

    Each job of the baseline is looked up by '<optimization>/<job>'. A job regresses if
      - it is missing from rows,
      - its method or target changed (the comparison would be meaningless),
      - it converged (success true) in the baseline but not now,
      - any of nit, nfev, njev, nhvp exceeds the baseline value by more than count_rtol
        (relatively) plus one evaluation, which absorbs round-off changes in the line searches,
      - its final loss exceeds loss_factor times the baseline final loss (final losses near
        convergence are dominated by round-off, so a factor is used instead of a tolerance),
      - with time_factor given, optimize_seconds exceeds time_factor times the baseline.
    Notes record jobs that are new, that use fewer evaluations, that reach a lower loss, and the
    wall-time ratio, none of which fail the check.
    """
    regressions: list[str] = []
    notes: list[str] = []
    current = {baseline_key(row): row for row in rows}
    for key, expected in (baseline.get("jobs") or {}).items():
        row = current.get(key)
        if row is None:
            regressions.append(f"{key}: missing from the compared jobs")
            continue
        changed = [name for name in ("method", "target") if row[name] != expected.get(name)]
        if changed:
            regressions.append(f"{key}: {', '.join(changed)} changed from the baseline "
                               f"({[expected.get(name) for name in changed]} -> {[row[name] for name in changed]})")
            continue
        if expected.get("success") and not row["success"]:
            regressions.append(f"{key}: converged in the baseline but not now (status {row['status']}: {row['message']})")
        for count in BASELINE_COUNTS:
            reference = int(expected.get(count, 0))
            allowed = reference * (1.0 + count_rtol) + 1
            if row[count] > allowed:
                regressions.append(f"{key}: {count} {row[count]} exceeds the baseline {reference} "
                                   f"(allowed up to {allowed:.0f} with --count-rtol {count_rtol})")
            elif row[count] < reference:
                notes.append(f"{key}: {count} {row[count]} is below the baseline {reference}")
        reference_loss = as_float(expected.get("final_loss"))
        if math.isfinite(reference_loss):
            if not math.isfinite(row["final_loss"]) or row["final_loss"] > loss_factor * max(reference_loss, 0.0):
                regressions.append(f"{key}: final_loss {row['final_loss']:.3e} is worse than {loss_factor:g} x the "
                                   f"baseline {reference_loss:.3e}")
            elif row["final_loss"] < reference_loss:
                notes.append(f"{key}: final_loss {row['final_loss']:.3e} is below the baseline {reference_loss:.3e}")
        reference_time = as_float(expected.get("optimize_seconds"))
        if math.isfinite(reference_time) and reference_time > 0 and math.isfinite(row["optimize_seconds"]):
            ratio = row["optimize_seconds"] / reference_time
            if time_factor is not None and ratio > time_factor:
                regressions.append(f"{key}: optimize_seconds {row['optimize_seconds']:.1f} is {ratio:.2f} x the baseline "
                                   f"{reference_time:.1f} (allowed {time_factor:g} x)")
            else:
                notes.append(f"{key}: optimize_seconds {ratio:.2f} x the baseline")
    for key in sorted(set(current) - set(baseline.get("jobs") or {})):
        notes.append(f"{key}: not in the baseline")
    return regressions, notes


def default_output_dir(job_dirs: list[Path]) -> Path:
    """Return the common parent directory of the jobs' optimization roots.

    A single optimization root gives the root itself. Sibling roots, such as
    runs/optimizations/doublevortex-params and runs/optimizations/doublevortex-params-newton-cg,
    give their parent.
    """
    roots = sorted({str(job_dir.parent) for job_dir in job_dirs})
    return Path(os.path.commonpath(roots))


def filter_job_dirs(job_dirs: list[Path], only: list[str] | None) -> list[Path]:
    """Keep the job directories whose '<optimization>/<job>' path contains any of the substrings.

    With only None or empty, every job is kept.
    """
    if not only:
        return job_dirs
    return [job_dir for job_dir in job_dirs if any(text in f"{job_dir.parent.name}/{job_dir.name}" for text in only)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the optimizer comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", help="Optimization job directories, or optimization roots containing them")
    parser.add_argument("--output-dir", help="Directory for optimizer-comparison.csv/.yaml "
                        "(default: the common parent of the jobs' optimization roots)")
    parser.add_argument("--no-save", action="store_true", help="Only print the table; do not write files")
    parser.add_argument("--only", nargs="+", metavar="SUBSTRING",
                        help="Compare only jobs whose '<optimization>/<job>' contains one of these substrings")
    parser.add_argument("--columns", nargs="+", metavar="COLUMN",
                        help=f"Columns of the printed table (default: {' '.join(TABLE_COLUMNS)}); "
                        "any CSV column can be used. The saved files always hold every column")
    parser.add_argument("--save-baseline", metavar="FILE", help="Save the compared jobs' metrics as a regression baseline")
    parser.add_argument("--check-baseline", metavar="FILE",
                        help="Check the compared jobs against a saved baseline; exit with status 1 on a regression")
    parser.add_argument("--count-rtol", type=float, default=0.25,
                        help="Allowed relative increase of nit/nfev/njev/nhvp over the baseline (default 0.25)")
    parser.add_argument("--loss-factor", type=float, default=10.0,
                        help="Allowed factor of the final loss over the baseline (default 10)")
    parser.add_argument("--time-factor", type=float, default=None,
                        help="Allowed factor of optimize_seconds over the baseline (default: report only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[Path, Path] | None:
    """Compare the requested jobs: print the table and save the CSV and YAML files.

    Returns the (csv, yaml) paths written, or None with --no-save. Unknown --columns raise
    ValueError, and so does an --only filter that matches no job, since a comparison of nothing
    is almost certainly a typo. --save-baseline writes a baseline file; --check-baseline compares
    with one and raises SystemExit(1) on a regression (after printing every finding).
    """
    args = parse_args(argv)
    job_dirs = filter_job_dirs(resolve_job_dirs(args.targets), args.only)
    if not job_dirs:
        raise ValueError(f"No jobs match --only {args.only}")
    rows = rows_for_job_dirs(job_dirs)

    columns = tuple(args.columns) if args.columns else TABLE_COLUMNS
    known = set(COMPARISON_COLUMNS) | {key for row in rows for key in row}
    unknown = [column for column in columns if column not in known]
    if unknown:
        raise ValueError(f"Unknown --columns {unknown}; available: {sorted(known)}")

    nproblems = len(group_by_problem(rows))
    print(f"Optimizer comparison: {len(rows)} job(s), {nproblems} problem(s)")
    print(format_comparison_table(rows, columns))
    paths = None
    if not args.no_save:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(job_dirs)
        paths = write_comparison(rows, output_dir)
        print(f"Comparison: {paths[0]}")
        print(f"Summary:    {paths[1]}")
    if args.save_baseline:
        print(f"Baseline:   {save_baseline(rows, args.save_baseline)}")
    if args.check_baseline:
        report_baseline_check(rows, load_yaml(args.check_baseline), args)
    return paths


def report_baseline_check(rows: list[dict[str, Any]], baseline: dict[str, Any], args: argparse.Namespace) -> None:
    """Print the result of check_against_baseline, and exit with status 1 on any regression."""
    regressions, notes = check_against_baseline(rows, baseline, count_rtol=args.count_rtol,
                                                loss_factor=args.loss_factor, time_factor=args.time_factor)
    print(f"Baseline check against {args.check_baseline} (created {baseline.get('created_at', 'unknown')}):")
    for note in notes:
        print(f"  note: {note}")
    for regression in regressions:
        print(f"  REGRESSION: {regression}")
    if regressions:
        raise SystemExit(1)
    print(f"  passed: {len(baseline.get('jobs') or {})} baseline job(s) within tolerances")


if __name__ == "__main__":
    main()
