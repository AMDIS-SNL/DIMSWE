"""Measure where objective, gradient and Hessian-vector-product evaluations spend their time.

Step 20, part 2 of TODO-hvp.md (docs/hvp_performance_plan.md, section 2). For one job of an
optimization spec (the same model, truth data and objective optimize_params.py would build),
this evaluates at the starting point, in this order,

    obj-1, obj-2 ..        objective; the first call includes TSFC compilation of the forward forms
    jac-1, jac-2 ..        gradient; the first call compiles the first-order adjoint forms
    hessp-1, hessp-2 ..    HVP in direction p1; the first call includes _setup_second_order and the
                           compilation of the tangent-linear and second-order adjoint forms
    hessp-<last>           HVP in a second direction p2, at the same point

(--repeats sets how many steady-state calls follow each first call). Every call is timed as a
whole and through the profiling regions of dimswe/profiling.py, which split it into the forward
pass, the substep recomputes, the forward/tangent/adjoint/SOA stage solves, the increment
assemblies, the observation sources and the initial-condition map, per sub-integrator.

The report (printed, and written to <output-dir>/profile.yaml and profile-tree.txt) contains
- first-call and steady-state seconds per evaluation type, and their difference (setup cost);
- the steady-state ratios jac/obj and hessp/jac, next to the cost model of the plan;
- the number of stage passes per time step and substep, checked against the cost model;
- the time per region category, and the remainder outside every instrumented solve or assembly
  (Python, UFL expression building, numpy copies);
- the cost of one stage pass per sub-integrator and stage kind, relative to a forward pass, which
  shows how much more expensive the second-order forms are;
- if the job already has a result.yaml (or --results is given), the optimizer's evaluation counts
  combined with the measured costs, compared with the recorded optimize_seconds.

PETSc options on the command line (for example -log_view :flamegraph.txt:ascii_flamegraph) are
passed through to PETSc; the profiling regions then also appear in the PETSc flamegraph. See
workflows/profile_hvp.sh.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import (
    PROJECT_ROOT,
    dump_yaml,
    is_mpi_root,
    load_yaml,
    resolve_path,
    resolve_runs_root,
    root_print,
    timestamp_utc,
    to_builtin,
)

PROFILE_YAML = "profile.yaml"
PROFILE_TREE = "profile-tree.txt"
DEFAULT_OUTPUT_ROOT = "runs/profiles"

EVALUATION_KINDS = ("obj", "jac", "hessp")

# stage-solve regions of GeneralRK; one call is one pass over all stages of one substep
STAGE_KINDS = ("forward stages", "tangent stages", "adjoint stages", "soa stages")

# Cost model of docs/hvp_performance_plan.md, section 1: stage passes per substep and time step.
# The forward stages are solved in the forward pass and again when the adjoint sweep recomputes
# the step (LieSplit substep recompute, or the GeneralRK adjoint step of a single RK
# timestepper); the HVP does the same for the tangent stages. A LieSplit with stage reuse (plan
# item A3, timestepping.reuse_stages, the default) restores the recomputed stages in its substep
# adjoint steps instead of solving them a third time. A change that stores more stage values
# (plan item A2) must update this table.
COST_MODEL_STAGE_PASSES = {
    "obj": {"forward stages": 1},
    "jac": {"forward stages": 2, "adjoint stages": 1},
    "hessp": {"forward stages": 2, "tangent stages": 2, "adjoint stages": 1, "soa stages": 1},
}
# LieSplit with reuse_stages false: the substep adjoint steps re-solve the stages once more
COST_MODEL_STAGE_PASSES_RECOMPUTE = {
    "obj": {"forward stages": 1},
    "jac": {"forward stages": 3, "adjoint stages": 1},
    "hessp": {"forward stages": 3, "tangent stages": 3, "adjoint stages": 1, "soa stages": 1},
}
# the same models in stage solves, assuming every stage pass costs the same: obj S, jac 3S,
# hessp 6S (with recompute: jac 4S, hessp 8S)
COST_MODEL_RATIOS = {"jac/obj": 3.0, "hessp/jac": 2.0, "hessp/obj": 6.0}
COST_MODEL_RATIOS_RECOMPUTE = {"jac/obj": 4.0, "hessp/jac": 2.0, "hessp/obj": 8.0}

# leaf regions that do the actual numerical work; time outside them is overhead
WORK_REGIONS = STAGE_KINDS + (
    "restore stages",
    "update",
    "gradient increments",
    "hessian increments",
    "lambda increment solve",
    "lambda_dot increment solve",
    "loss",
    "observation sources",
    "setup hvp",
    "setup second order",
    "ic map forward",
    "ic map tangents",
    "ic map second tangents",
)


def profiling_module():
    """Return dimswe.profiling without importing the dimswe package, which needs Firedrake.

    The module is registered under its real name, so a later ``import dimswe`` (whose
    timesteppers do ``from .profiling import region``) finds and reuses this module object, and
    the driver and the instrumented code share one timing table.
    """
    module = sys.modules.get("dimswe.profiling")
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location("dimswe.profiling", PROJECT_ROOT / "dimswe" / "profiling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dimswe.profiling"] = module
    spec.loader.exec_module(module)
    return module


def mpi_world() -> tuple[int, Callable[[], None]]:
    """Return (world size, barrier), falling back to (1, no-op) without mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover - serial fallback
        return 1, lambda: None
    return MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.barrier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line. Unknown options (PETSc options such as -log_view) are ignored."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("optimize_cfg", help="Optimization YAML spec whose job is profiled")
    parser.add_argument("--job", default=None,
                        help="Profile the first job whose name contains this string (default: the first planned job)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Steady-state calls after the first call of each evaluation type (default 1)")
    parser.add_argument("--seed", type=int, default=2020, help="Seed of the random HVP directions")
    parser.add_argument("--output-dir", default=None,
                        help=f"Where profile.yaml is written (default {DEFAULT_OUTPUT_ROOT}/<optimization>/<job>)")
    parser.add_argument("--results", nargs="*", default=None,
                        help="result.yaml files whose optimizer counts are combined with the measured costs "
                             "(default: the profiled job's own result.yaml, if it exists)")
    parser.add_argument("--min-share", type=float, default=0.005,
                        help="Hide regions below this share of their evaluation in the printed tree")
    parser.add_argument("--recompute-stages", action="store_true",
                        help="Switch LieSplit stage reuse off (reuse_stages false) for an A/B comparison with the "
                             "default; the default output directory gets a __recompute suffix")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected job and evaluation plan only")
    parser.add_argument("--print-output-dir", action="store_true",
                        help="Print only the output directory and exit (used by profile_hvp.sh --flamegraph)")
    args, unknown = parser.parse_known_args(argv)
    # PETSc options are "-name" or "-name value"; anything else that argparse did not consume is
    # a mistake (for example a second spec path)
    stray = [arg for k, arg in enumerate(unknown)
             if not arg.startswith("-") and (k == 0 or not unknown[k - 1].startswith("-"))]
    if stray:
        parser.error(f"unrecognized arguments: {' '.join(stray)}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.petsc_options = unknown
    return args


def evaluation_plan(repeats: int) -> list[tuple[str, str, int]]:
    """Return the evaluation sequence as (label, kind, direction index) triples.

    The direction index is 0 for obj and jac, 1 for the HVPs in p1 and 2 for the last HVP, in p2.
    """
    plan = []
    for kind in ("obj", "jac"):
        plan += [(f"{kind}-{k + 1}", kind, 0) for k in range(repeats + 1)]
    plan += [(f"hessp-{k + 1}", "hessp", 1) for k in range(repeats + 1)]
    plan.append((f"hessp-{repeats + 2}", "hessp", 2))
    return plan


def random_directions(size: int, seed: int) -> tuple[Any, Any]:
    """Return two unit directions in the reduced space, identical on every rank for a given seed."""
    import numpy as np

    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(2):
        p = rng.standard_normal(size)
        directions.append(p / np.linalg.norm(p))
    return directions[0], directions[1]


def profile_evaluations(problem: Any, repeats: int = 1, seed: int = 2020) -> list[dict[str, Any]]:
    """Evaluate obj, jac and hessp at the problem's starting point and profile every call.

    problem is a ReducedProblem (or anything with z0, to_full, direction_to_full, full_fun,
    full_jac and full_hessp). The full-space functions are called directly, bypassing the
    ReducedProblem memo, so repeated calls at the same point really recompute. Each call runs
    inside a top-level profiling region named by its label, with the timing table reset before
    it, so the returned records hold one region table per call:

        {"label": "hessp-2", "kind": "hessp", "direction": 1, "seconds": s, "regions": {...}}

    Calls are timed with the clock of the timing table (profiling.now), so the call times and the
    region times are consistent.
    """
    profiling = profiling_module()
    x = problem.to_full(problem.z0)
    p1, p2 = random_directions(problem.z0.size, seed)
    directions = {1: problem.direction_to_full(p1), 2: problem.direction_to_full(p2)}
    calls = {
        "obj": lambda direction: problem.full_fun(x),
        "jac": lambda direction: problem.full_jac(x),
        "hessp": lambda direction: problem.full_hessp(x, directions[direction]),
    }

    was_enabled = profiling.is_enabled()
    profiling.enable()
    records = []
    try:
        for label, kind, direction in evaluation_plan(repeats):
            profiling.reset()
            start = profiling.now()
            with profiling.region(label):
                calls[kind](direction)
            seconds = profiling.now() - start
            records.append({"label": label, "kind": kind, "direction": direction, "seconds": seconds,
                            "regions": profiling.snapshot()})
            root_print(f"  {label:<10} {seconds:10.3f} s")
    finally:
        profiling.enable(was_enabled)
    return records


def strip_top(regions: dict[str, dict[str, float]], label: str) -> dict[str, dict[str, float]]:
    """Return a record's regions without its top-level label, so calls of one kind compare directly."""
    prefix = label + "/"
    return {path[len(prefix):]: entry for path, entry in regions.items() if path.startswith(prefix)}


def evaluation_seconds(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return first-call, steady-state (median of the later calls) and setup seconds per kind."""
    result = {}
    for kind in EVALUATION_KINDS:
        times = [record["seconds"] for record in records if record["kind"] == kind]
        if not times:
            continue
        steady = statistics.median(times[1:]) if len(times) > 1 else times[0]
        result[kind] = {"first": times[0], "steady": steady, "setup": times[0] - steady, "calls": len(times)}
    return result


def steady_record(records: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    """Return the last call of a kind, a steady-state call (None if the kind was not evaluated)."""
    matching = [record for record in records if record["kind"] == kind]
    return matching[-1] if matching else None


def cost_ratios(seconds: dict[str, dict[str, float]]) -> dict[str, float | None]:
    """Return the measured steady-state ratios jac/obj, hessp/jac and hessp/obj."""
    def ratio(numerator: str, denominator: str) -> float | None:
        if numerator not in seconds or denominator not in seconds or seconds[denominator]["steady"] <= 0:
            return None
        return seconds[numerator]["steady"] / seconds[denominator]["steady"]

    return {"jac/obj": ratio("jac", "obj"), "hessp/jac": ratio("hessp", "jac"), "hessp/obj": ratio("hessp", "obj")}


def stage_passes(record: dict[str, Any], nsteps: int, nsubsteps: int,
                 recompute_stages: bool = False) -> dict[str, dict[str, float | None]]:
    """Return stage passes per time step and substep for one call, next to the cost model.

    recompute_stages selects COST_MODEL_STAGE_PASSES_RECOMPUTE (LieSplit with reuse_stages off)
    instead of COST_MODEL_STAGE_PASSES (see timestepper_layout).
    """
    totals = profiling_module().totals_by_name(strip_top(record["regions"], record["label"]))
    model = (COST_MODEL_STAGE_PASSES_RECOMPUTE if recompute_stages else COST_MODEL_STAGE_PASSES)[record["kind"]]
    result = {}
    for kind in STAGE_KINDS:
        calls = totals.get(kind, {"calls": 0})["calls"]
        result[kind] = {"measured": calls / (nsteps * nsubsteps), "model": float(model.get(kind, 0))}
    return result


def breakdown(record: dict[str, Any]) -> dict[str, Any]:
    """Return seconds and shares of the work regions of one call, plus the uninstrumented rest.

    other is the call's time outside every work region: Python and UFL overhead between solves
    (expression building in assign, numpy copies, resets) and the substep bookkeeping.
    """
    totals = profiling_module().totals_by_name(strip_top(record["regions"], record["label"]))
    total = record["seconds"]
    categories = {}
    accounted = 0.0
    for name in WORK_REGIONS:
        if name in totals:
            seconds = totals[name]["seconds"]
            categories[name] = {"seconds": seconds, "share": seconds / total if total > 0 else 0.0,
                                "calls": totals[name]["calls"]}
            accounted += seconds
    other = total - accounted
    categories["other"] = {"seconds": other, "share": other / total if total > 0 else 0.0, "calls": 0}
    return dict(sorted(categories.items(), key=lambda item: -item[1]["seconds"]))


def stage_costs(record: dict[str, Any]) -> dict[str, dict[str, dict[str, float | None]]]:
    """Return the cost of one stage pass per sub-integrator and stage kind, for one call.

    Stage regions are attributed to the nearest enclosing sub-integrator region (named like
    "0:RK4(model)"), or to "rk" for a plain GeneralRK timestepper. vs_forward is the cost of a
    pass relative to a forward-stage pass of the same sub-integrator in the same call, which
    measures how much larger the tangent, adjoint and second-order adjoint forms are.
    """
    regions = strip_top(record["regions"], record["label"])
    table: dict[str, dict[str, dict[str, float]]] = {}
    for path, entry in regions.items():
        parts = path.split("/")
        if parts[-1] not in STAGE_KINDS:
            continue
        label = next((part for part in reversed(parts[:-1]) if ":" in part), "rk")
        cell = table.setdefault(label, {}).setdefault(parts[-1], {"calls": 0, "seconds": 0.0})
        cell["calls"] += entry["calls"]
        cell["seconds"] += entry["seconds"]
    result = {}
    for label in sorted(table):
        forward = table[label].get("forward stages")
        forward_per_pass = forward["seconds"] / forward["calls"] if forward and forward["calls"] else None
        result[label] = {}
        for kind in STAGE_KINDS:
            if kind not in table[label]:
                continue
            cell = table[label][kind]
            per_pass = cell["seconds"] / cell["calls"] if cell["calls"] else None
            result[label][kind] = {
                "calls": cell["calls"], "seconds": cell["seconds"], "seconds_per_pass": per_pass,
                "vs_forward": per_pass / forward_per_pass if per_pass is not None and forward_per_pass else None,
            }
    return result


def optimizer_estimate(result: dict[str, Any], seconds: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Combine a result.yaml's evaluation counts with the measured costs.

    optimize_params.run_job evaluates the loss and gradient once before it starts the optimizer
    clock (initial_gradient_report), so those first calls, and their compilation, are not part
    of optimize_seconds; the one-off HVP setup is, if HVPs were used. The estimate is
        (nfev - 1) obj + (njev - 1) jac + nhvp hessp + hessp setup (if nhvp > 0),
    with steady-state costs. A large gap to optimize_seconds points at costs outside the
    evaluations, or at a machine that was loaded differently.
    """
    optimizer = result.get("optimizer", {})
    nfev = int(optimizer.get("nfev", 0))
    njev = int(optimizer.get("njev", 0))
    nhvp = int(optimizer.get("nhvp", 0) or 0)
    steady = {kind: seconds.get(kind, {}).get("steady", 0.0) for kind in EVALUATION_KINDS}
    parts = {
        "obj": max(nfev - 1, 0) * steady["obj"],
        "jac": max(njev - 1, 0) * steady["jac"],
        "hessp": nhvp * steady["hessp"],
        "hessp setup": seconds.get("hessp", {}).get("setup", 0.0) if nhvp > 0 else 0.0,
    }
    estimate = sum(parts.values())
    recorded = result.get("optimize_seconds")
    return {
        "job": result.get("job_name"), "method": optimizer.get("method"), "nit": optimizer.get("nit"),
        "nfev": nfev, "njev": njev, "nhvp": nhvp,
        "estimate_seconds": estimate, "estimate_parts": parts,
        "hvp_share": (parts["hessp"] + parts["hessp setup"]) / estimate if estimate > 0 else 0.0,
        "optimize_seconds": recorded,
        "estimate_vs_recorded": estimate / float(recorded) if recorded else None,
        "hvps_per_iteration": nhvp / int(optimizer["nit"]) if optimizer.get("nit") else None,
    }


def summarize(records: list[dict[str, Any]], nsteps: int, nsubsteps: int,
              results: list[dict[str, Any]] | None = None, recompute_stages: bool = False) -> dict[str, Any]:
    """Build the summary section of profile.yaml from the profiled calls (see the module doc).

    nsteps and nsubsteps are the time steps per data block and substeps per time step,
    recompute_stages selects the cost model (see timestepper_layout), and results holds the
    result.yaml contents to combine with the measured costs.
    """
    seconds = evaluation_seconds(records)
    summary: dict[str, Any] = {
        "seconds": seconds,
        "ratios": cost_ratios(seconds),
        "cost_model_ratios": dict(COST_MODEL_RATIOS_RECOMPUTE if recompute_stages else COST_MODEL_RATIOS),
        "stage_passes_per_substep": {},
        "breakdown": {},
        "stage_costs": {},
    }
    for kind in EVALUATION_KINDS:
        record = steady_record(records, kind)
        if record is None:
            continue
        summary["stage_passes_per_substep"][kind] = stage_passes(record, nsteps, nsubsteps, recompute_stages)
        summary["breakdown"][kind] = breakdown(record)
        summary["stage_costs"][kind] = stage_costs(record)
    first_hessp = next((record for record in records if record["kind"] == "hessp"), None)
    if first_hessp is not None:
        # form/solver construction only; the rest of the setup cost is compilation in the first solves
        totals = profiling_module().totals_by_name(strip_top(first_hessp["regions"], first_hessp["label"]))
        summary["hessp_setup_construction_seconds"] = sum(
            totals.get(name, {"seconds": 0.0})["seconds"] for name in ("setup hvp", "setup second order"))
    summary["optimizer"] = [optimizer_estimate(result, seconds) for result in (results or [])]
    return summary


def format_number(value: Any, digits: int = 3) -> str:
    """Format a float for the printed report, or '-' for None."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_summary(summary: dict[str, Any]) -> str:
    """Format the summary as the printed text report."""
    lines = ["", "Seconds per evaluation (first call, steady state, setup = first - steady):"]
    for kind, entry in summary["seconds"].items():
        lines.append(f"  {kind:<6} first {entry['first']:10.3f}   steady {entry['steady']:10.3f}   setup {entry['setup']:10.3f}")
    if "hessp_setup_construction_seconds" in summary:
        lines.append(f"  of the hessp setup, form and solver construction: {summary['hessp_setup_construction_seconds']:.3f} s "
                     "(the rest is mostly compilation)")
    lines.append("Steady-state ratios (measured vs cost model):")
    for name, value in summary["ratios"].items():
        lines.append(f"  {name:<10} {format_number(value, 2):>8}   model {summary['cost_model_ratios'][name]:.1f}")
    lines.append("Stage passes per time step and substep (measured / model):")
    for kind, passes in summary["stage_passes_per_substep"].items():
        cells = ", ".join(f"{name.split()[0]} {entry['measured']:g}/{entry['model']:g}" for name, entry in passes.items()
                          if entry["measured"] or entry["model"])
        lines.append(f"  {kind:<6} {cells}")
    for kind, categories in summary["breakdown"].items():
        lines.append(f"Steady-state {kind} by region:")
        for name, entry in categories.items():
            if entry["share"] >= 0.005:
                lines.append(f"  {name:<28} {entry['seconds']:10.3f} s {entry['share']:7.1%}")
    hessp_costs = summary["stage_costs"].get("hessp", {})
    if hessp_costs:
        lines.append("Cost of one stage pass in hessp, relative to a forward pass of the same sub-integrator:")
        for label, kinds in hessp_costs.items():
            cells = ", ".join(f"{kind.split()[0]} {format_number(entry['vs_forward'], 2)}" for kind, entry in kinds.items())
            forward = kinds.get("forward stages", {}).get("seconds_per_pass")
            lines.append(f"  {label:<28} forward pass {format_number(forward, 4)} s; {cells}")
    for estimate in summary["optimizer"]:
        lines.append(
            f"Optimizer run {estimate['job']} ({estimate['method']}, nit {estimate['nit']}, nfev {estimate['nfev']}, "
            f"njev {estimate['njev']}, nhvp {estimate['nhvp']}): estimated {estimate['estimate_seconds']:.1f} s "
            f"({estimate['hvp_share']:.0%} in HVPs), recorded {format_number(estimate['optimize_seconds'], 1)} s")
    return "\n".join(lines)


def format_trees(records: list[dict[str, Any]], min_share: float) -> str:
    """Format the region tree of every profiled call, for profile-tree.txt and the console."""
    profiling = profiling_module()
    blocks = []
    for record in records:
        blocks.append(f"== {record['label']} ({record['seconds']:.3f} s)\n" + profiling.format_tree(record["regions"], min_share))
    return "\n\n".join(blocks)


def select_job(spec_path: Path, job_filter: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the spec and return (spec, job) for the first planned job matching job_filter."""
    from workflows.common import resolve_truth_runs
    from workflows.optimize_params import load_spec, plan_jobs

    spec = load_spec(spec_path)
    truth_runs = resolve_truth_runs(spec["truth"], spec_path.parent)
    output_dir = resolve_runs_root(spec.get("output_root", "runs/optimizations")) / spec["optimization_name"]
    jobs = plan_jobs(spec, truth_runs, output_dir)
    if job_filter is not None:
        jobs = [job for job in jobs if job_filter in job["name"]]
    if not jobs:
        raise ValueError(f"no job of {spec_path} matches --job {job_filter!r}")
    return spec, jobs[0]


def timestepper_layout(timestepper: Any) -> tuple[int, bool]:
    """Return (substeps per time step, recompute_stages) for a timestepper.

    A LieSplit timestepper takes sum(subcycle_list) substeps per step; a single RK one takes one.
    recompute_stages is True for a LieSplit with reuse_stages off, whose substep adjoint steps
    re-solve the stages (COST_MODEL_STAGE_PASSES_RECOMPUTE).
    """
    subcycles = getattr(timestepper, "subcycle_list", None)
    if subcycles is None:
        return 1, False
    return int(sum(subcycles)), not getattr(timestepper, "reuse_stages", False)


def build_job_problem(spec: dict[str, Any], job: dict[str, Any],
                      recompute_stages: bool = False) -> tuple[Any, int, int, bool]:
    """Build the job's ReducedProblem exactly as optimize_params.run_job does.

    With recompute_stages, a LieSplit timestepper's stage reuse is switched off afterwards, so the
    same build can be profiled on the old recompute path (--recompute-stages). Returns (problem,
    nsteps, nsubsteps, recompute_stages): the time steps per data block, the substeps per time
    step and whether the stages are re-solved in the adjoint substeps (see timestepper_layout).
    """
    from workflows.optimize_params import Job

    state = Job(spec, job)
    problem, _, _ = state.build_problem()
    if recompute_stages and hasattr(state.timestepper, "reuse_stages"):
        state.timestepper.reuse_stages = False
    nsubsteps, recompute_stages = timestepper_layout(state.timestepper)
    return problem, int(state.objective.nsteps), nsubsteps, recompute_stages


def load_results(paths: list[str] | None, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the result.yaml files given with --results, or the job's own result.yaml if it exists."""
    if paths is None:
        own = Path(job["job_dir"]) / "result.yaml"
        return [load_yaml(own)] if own.exists() else []
    return [load_yaml(resolve_path(path)) for path in paths]


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    """Profile one job and write the report; returns the profile (None for --dry-run and --print-output-dir)."""
    args = parse_args(argv)
    spec_path = resolve_path(args.optimize_cfg)
    spec, job = select_job(spec_path, args.job)
    output_dir = (resolve_path(args.output_dir) if args.output_dir
                  else resolve_runs_root(DEFAULT_OUTPUT_ROOT) / spec["optimization_name"]
                  / (job["name"] + ("__recompute" if args.recompute_stages else "")))
    if args.print_output_dir:
        root_print(str(output_dir))
        return None
    root_print(f"Profiling job {job['name']} of {spec['optimization_name']} (target {spec['target']}) -> {output_dir}")
    root_print("  evaluations: " + ", ".join(label for label, _, _ in evaluation_plan(args.repeats)))
    if args.petsc_options:
        root_print("  PETSc options: " + " ".join(args.petsc_options))
    if args.dry_run:
        return None

    start = time.perf_counter()
    problem, nsteps, nsubsteps, recompute_stages = build_job_problem(spec, job, recompute_stages=args.recompute_stages)
    build_seconds = time.perf_counter() - start
    root_print(f"  built the model and problem in {build_seconds:.3f} s ({nsteps} steps x {nsubsteps} substeps, "
               f"{problem.z0.size} optimization variables)")

    records = profile_evaluations(problem, repeats=args.repeats, seed=args.seed)
    results = load_results(args.results, job)
    summary = summarize(records, nsteps, nsubsteps, results, recompute_stages=recompute_stages)

    world_size, barrier = mpi_world()
    profile = {
        "created_at": timestamp_utc(),
        "optimize_cfg": str(spec_path),
        "optimization_name": spec["optimization_name"],
        "job_name": job["name"],
        "target": spec["target"],
        "method": spec["optimizer"]["method"],
        "mpi_world_size": world_size,
        "petsc_options": args.petsc_options,
        "nsteps": nsteps,
        "nsubsteps": nsubsteps,
        "recompute_stages": recompute_stages,
        "nvariables": int(problem.z0.size),
        "repeats": args.repeats,
        "seed": args.seed,
        "build_seconds": build_seconds,
        "summary": summary,
        "evaluations": records,
    }
    trees = format_trees(records, args.min_share)
    root_print("\n" + trees)
    root_print(format_summary(summary))
    if is_mpi_root():
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(to_builtin(profile), output_dir / PROFILE_YAML)
        (output_dir / PROFILE_TREE).write_text(trees + "\n" + format_summary(summary) + "\n", encoding="utf-8")
        root_print(f"\nWrote {output_dir / PROFILE_YAML} and {output_dir / PROFILE_TREE}")
    barrier()
    return profile


if __name__ == "__main__":
    main()
