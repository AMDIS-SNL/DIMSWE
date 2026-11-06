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

prints the comparison table and saves both files to --output-dir. --output-dir defaults to the
common parent directory of the jobs' optimization roots: the root itself for a single root, and
for example runs/optimizations for the paired example roots.
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[Path, Path] | None:
    """Compare the requested jobs: print the table and save the CSV and YAML files.

    Returns the (csv, yaml) paths written, or None with --no-save. Unknown --columns raise
    ValueError, and so does an --only filter that matches no job, since a comparison of nothing
    is almost certainly a typo.
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
    if args.no_save:
        return None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(job_dirs)
    csv_path, yaml_path = write_comparison(rows, output_dir)
    print(f"Comparison: {csv_path}")
    print(f"Summary:    {yaml_path}")
    return csv_path, yaml_path


if __name__ == "__main__":
    main()
