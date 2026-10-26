"""Find optimal model parameters against truth runs with adjoint-based optimization.

Step 2 of the optimization workflow. For every truth run (or for all truth runs jointly) and
every listed loss, this builds the model on the truth mesh, loads the truth states the loss
needs, and minimizes the loss with dimswe.optimize. Each optimization writes its own directory:

    <output_root>/<optimization_name>/<truth-run>__<loss-label>/
        optimize.cfg       resolved optimization spec for this job
        result.yaml        initial, optimal and true values, final loss, optimizer status
        history.csv        loss, gradient norm and parameters per optimizer iteration
        optimal.cfg        truth cfg with the optimal parameters written back
        optimal_ic.h5      optimized initial state (ic-field target only)
        taylor-test.yaml   gradient check results (when optimizer.check_gradient is enabled)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import (
    build_observation_schedule,
    check_paths_exist,
    classify_parameter,
    deep_copy_parameters,
    dump_yaml,
    format_value_for_name,
    load_yaml,
    loss_label,
    resolve_path,
    resolve_runs_root,
    resolve_truth_runs,
    root_print,
    set_by_dotted_path,
    timestamp_utc,
    to_builtin,
    validate_target_and_losses,
    write_csv,
)

DEFAULT_OPTIMIZER = {
    "method": "L-BFGS-B",
    "gtol": 1.0e-6,
    "ftol": 1.0e-10,
    "maxiter": 100,
    "check_gradient": False,
    # relative: gtol is a fraction of the initial gradient's largest component, so stopping does
    # not depend on the overall size of the loss; absolute: gtol is passed to scipy unchanged
    "tolerance": "relative",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for parameter optimization."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("optimize_cfg", help="Path to the optimization YAML spec")
    parser.add_argument("--dry-run", action="store_true", help="Validate the spec and print planned jobs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs whose result.yaml already exists")
    parser.add_argument("--only", nargs="*", default=None, help="Only run jobs whose name contains one of these strings")
    parser.add_argument("--check-gradient", action="store_true", help="Run a Taylor test before optimizing")
    return parser.parse_args()


def normalize_parameter_specs(target: str, parameter_specs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize the parameters section of a spec into {path: {initial, bounds, scale, fixed}}.

    Entries may be a mapping or a bare number (the initial value). ``initial`` may be omitted, in
    which case the value from the truth cfg is used, which is mainly useful for fixed entries.
    """
    normalized = {}
    for path, spec in (parameter_specs or {}).items():
        if not isinstance(spec, dict):
            spec = {"initial": spec}
        unknown = set(spec) - {"initial", "bounds", "scale", "fixed"}
        if unknown:
            raise ValueError(f"Parameter '{path}' has unknown keys {sorted(unknown)}")
        entry = {
            "initial": spec.get("initial"),
            "bounds": spec.get("bounds"),
            "scale": spec.get("scale"),
            "fixed": spec.get("fixed", False),
        }
        if entry["fixed"] is not False and entry["fixed"] is not True:
            # `fixed: 3.2` is shorthand for a fixed parameter with that value
            entry["initial"] = entry["fixed"]
            entry["fixed"] = True
        if entry["bounds"] is not None:
            bounds = entry["bounds"]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"Parameter '{path}' bounds must be [low, high]")
            low = float("-inf") if bounds[0] is None else float(bounds[0])
            high = float("inf") if bounds[1] is None else float(bounds[1])
            if not low <= high:
                raise ValueError(f"Parameter '{path}' bounds must satisfy low <= high")
            entry["bounds"] = [low, high]
        if entry["scale"] is not None and float(entry["scale"]) <= 0:
            raise ValueError(f"Parameter '{path}' scale must be positive")
        if target == "ic-params" and entry["bounds"] is not None:
            raise ValueError("Bounds are not supported for the ic-params target; use 'scale' instead")
        normalized[path] = entry
    return normalized


def load_spec(path: Path) -> dict[str, Any]:
    """Load, default and validate an optimization spec."""
    spec = load_yaml(path)
    spec["_source_path"] = str(path)
    spec.setdefault("optimization_name", path.stem)
    spec.setdefault("normalize", True)
    spec.setdefault("overrides", {})
    spec["optimizer"] = {**DEFAULT_OPTIMIZER, **(spec.get("optimizer") or {})}
    truth = spec.get("truth") or {}
    truth.setdefault("mode", "per-run")
    spec["truth"] = truth
    target = spec.get("target")
    validate_target_and_losses(target, spec.get("losses", []), truth["mode"])
    if "window" not in spec or "num_steps" not in spec["window"]:
        raise ValueError("window.num_steps is required")
    spec["window"].setdefault("start_index", 0)
    if target in ("ic-field", "ic-params") and spec["window"]["start_index"] != 0:
        raise ValueError("Initial-condition targets must use window.start_index 0")
    spec["parameters"] = normalize_parameter_specs(target, spec.get("parameters"))

    if target in ("coeffs", "ic-params") and not spec["parameters"]:
        raise ValueError(f"Target '{target}' needs a parameters section")
    if target in ("coeffs", "ic-params") and all(entry["fixed"] for entry in spec["parameters"].values()):
        raise ValueError("At least one parameter must be free")
    # parameter kinds are checked against each truth cfg in check_spec_against_truth
    for parameter_path in spec["parameters"]:
        if not parameter_path.count(".") == 1:
            raise ValueError(f"Parameter '{parameter_path}' must be a two-level dotted path")
        if target == "coeffs" and parameter_path.startswith("initial-conditions."):
            raise ValueError("Initial-condition parameters cannot be optimized with the coeffs target")
        if target != "coeffs" and not parameter_path.startswith("initial-conditions."):
            raise ValueError(f"Target '{target}' parameters must be initial-condition parameters")
    if target == "ic-field":
        for entry in spec["parameters"].values():
            if entry["bounds"] is not None or entry["scale"] is not None or entry["fixed"]:
                raise ValueError("ic-field parameters only set the initial guess; bounds/scale/fixed are not used")
    return spec


def check_spec_against_truth(spec: dict[str, Any], truth_parameters: dict[str, Any]) -> None:
    """Check that the optimized parameters exist in a truth cfg and have the expected kind."""
    paths = list(spec["parameters"])
    check_paths_exist(truth_parameters, paths, f"truth cfg for '{spec['optimization_name']}'")
    check_paths_exist(
        truth_parameters,
        [path for path in spec["overrides"] if not path.startswith("output.")],
        "optimization overrides",
    )
    expected = "coeff" if spec["target"] == "coeffs" else "ic"
    for path in paths:
        kind = classify_parameter(path, truth_parameters)
        if kind != expected:
            raise ValueError(f"'{path}' is a '{kind}' parameter, but target '{spec['target']}' needs '{expected}'")
        if spec["target"] == "coeffs":
            section = path.split(".")[0]
            # check the model actually used for optimization, i.e. after overrides
            forcing_terms = spec["overrides"].get("model.forcing_terms", truth_parameters["model"].get("forcing_terms", []))
            if section not in forcing_terms:
                raise ValueError(f"Coefficient '{path}' belongs to inactive forcing term '{section}'")


def plan_jobs(spec: dict[str, Any], truth_runs: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Expand a spec into one job per (truth group, loss) with its observation schedule."""
    window = spec["window"]
    if spec["truth"]["mode"] == "joint":
        groups = [{"name": spec["truth"].get("label", "joint"), "runs": truth_runs}]
    else:
        groups = [{"name": run["run_name"], "runs": [run]} for run in truth_runs]

    jobs = []
    for group in groups:
        reference = group["runs"][0]["parameters"]
        for run in group["runs"]:
            check_spec_against_truth(spec, run["parameters"])
            for key in ("dt", "num_steps"):
                if run["parameters"]["timestepping"][key] != reference["timestepping"][key]:
                    raise ValueError("Jointly optimized truth runs must share timestepping settings")
            if run["parameters"]["output"]["output_freq"] != reference["output"]["output_freq"]:
                raise ValueError("Jointly optimized truth runs must share output.output_freq")
        output_freq = reference["output"]["output_freq"]
        available = reference["timestepping"]["num_steps"] // output_freq + 1
        for loss in spec["losses"]:
            schedule = build_observation_schedule(
                loss, window["start_index"], window["num_steps"], output_freq, available
            )
            label = loss_label(loss)
            job_name = f"{format_value_for_name(group['name'])}__{label}"
            jobs.append(
                {
                    "name": job_name,
                    "group": group["name"],
                    "truth_runs": group["runs"],
                    "loss": loss,
                    "loss_label": label,
                    "schedule": schedule,
                    "job_dir": output_dir / job_name,
                }
            )
    return jobs


def describe_job(job: dict[str, Any]) -> str:
    """Return a human-readable summary of a planned job."""
    schedule = job["schedule"]
    return (
        f"{job['name']}: truth={[run['run_name'] for run in job['truth_runs']]} "
        f"loss={job['loss_label']} blocks={len(schedule['block_start_indices'])}x{schedule['block_steps']} steps "
        f"observations={schedule['obs_steps']}"
    )


def resolve_parameter_values(spec: dict[str, Any], truth_parameters: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Return initial values, bounds, scales and free flags for the given dotted paths.

    Paths that are not listed in the spec keep their truth value and are held fixed.
    """
    from workflows.common import get_by_dotted_path

    resolved = {}
    for path in names:
        entry = spec["parameters"].get(path)
        truth_value = float(get_by_dotted_path(truth_parameters, path))
        if entry is None:
            resolved[path] = {"initial": truth_value, "bounds": None, "scale": None, "free": False}
            continue
        initial = truth_value if entry["initial"] is None else float(entry["initial"])
        resolved[path] = {
            "initial": initial,
            "bounds": entry["bounds"],
            "scale": entry["scale"],
            "free": not entry["fixed"],
        }
    return resolved


class History:
    """Collect optimizer iterations and write them as CSV."""

    def __init__(self, names: list[str] | None):
        self.names = names
        self.rows: list[dict[str, Any]] = []

    def __call__(self, iteration: int, x_full: Any, fun_value: float | None, grad_norm: float | None) -> None:
        row: dict[str, Any] = {
            "iteration": iteration,
            "loss": float("nan") if fun_value is None else fun_value,
            "grad_norm": float("nan") if grad_norm is None else grad_norm,
        }
        if self.names is not None:
            for name, value in zip(self.names, x_full, strict=True):
                row[name] = float(value)
        self.rows.append(row)
        root_print(f"  iteration {iteration}: loss={row['loss']:.6e} |grad|={row['grad_norm']:.3e}")


class Job:
    """One optimization: a model on the truth mesh, its truth data, objective and optimizer."""

    def __init__(self, spec: dict[str, Any], job: dict[str, Any]):
        from dimswe.logger import EmptyLogger
        from dimswe.optimize import Lagrangian_ODEConstrainedOptimization
        from dimswe.timestepping import get_timestepper
        from workflows.losses import build_data_blocks, build_objective, needed_output_indices
        from workflows.opt_helpers import (
            build_model,
            enable_coefficients,
            load_truth_run,
            optimization_solver_parameters,
            output_time,
            truth_states_on_model,
        )

        self.spec = spec
        self.job = job
        self.target = spec["target"]
        self.schedule = job["schedule"]
        truth_runs = job["truth_runs"]

        # the model setup comes from the (first) truth run, so dt and discretization match the truth
        self.truth_parameters = truth_runs[0]["parameters"]
        parameters = deep_copy_parameters(self.truth_parameters)
        for path, value in spec["overrides"].items():
            set_by_dotted_path(parameters, path, value)
        if self.target == "coeffs":
            enable_coefficients(parameters, list(spec["parameters"]))
        self.parameters = parameters

        indices = needed_output_indices(self.schedule)
        loaded_runs = [load_truth_run(run["checkpoint"], indices) for run in truth_runs]
        self.mesh = loaded_runs[0][0]
        self.model = build_model(parameters, self.mesh)
        self.truth_states = [
            truth_states_on_model(self.model, mesh, loaded["xn0"], prefix=f"truth{r}")
            for r, (mesh, loaded) in enumerate(loaded_runs)
        ]
        start_times = [
            [output_time(run["parameters"], self.model, index) for index in self.schedule["block_start_indices"]]
            for run in truth_runs
        ]
        data_blocks, t_blocks = build_data_blocks(self.schedule, self.truth_states, start_times)
        self.objective = build_objective(self.model, self.schedule, data_blocks, t_blocks, spec["normalize"])

        self.timestepper = get_timestepper(parameters, self.model, EmptyLogger(), optimization_solver_parameters())
        self.dt = parameters["timestepping"]["dt"]
        self.coeff, self.coeff_sub, _ = self.model.get_coeff_var("coeff")
        self.model.set_coeffs(parameters, self.coeff_sub)
        self.timestepper.set_coeff(self.coeff)
        self.optimizer = Lagrangian_ODEConstrainedOptimization(self.model, self.timestepper, self.objective, self.dt)
        self.optimizer.verbose = False

    def build_problem(self) -> tuple[Any, list[str] | None, dict[str, Any]]:
        """Build the ReducedProblem, the parameter names, and the resolved parameter values."""
        import numpy as np

        if self.target == "coeffs":
            from workflows.opt_helpers import coefficient_paths

            names = coefficient_paths(self.model)
            resolved = resolve_parameter_values(self.spec, self.parameters, names)
            initial = np.array([resolved[name]["initial"] for name in names])
            model_lb, model_ub = self.model.get_coeff_bounds()
            lb = np.array(model_lb, dtype=float)
            ub = np.array(model_ub, dtype=float)
            for k, name in enumerate(names):
                if resolved[name]["bounds"] is not None:
                    lb[k], ub[k] = resolved[name]["bounds"]
            free = np.array([resolved[name]["free"] for name in names])
            outside = [name for k, name in enumerate(names) if free[k] and not lb[k] <= initial[k] <= ub[k]]
            if outside:
                raise ValueError(f"Initial values of {outside} lie outside their bounds {list(zip(lb, ub))}")
            # L-BFGS-B projects the gradient onto the bounds, so a parameter starting on a bound with
            # an outward-pointing gradient contributes zero and can make the optimizer stop at once
            on_bound = [name for k, name in enumerate(names) if free[k] and initial[k] in (lb[k], ub[k])]
            if on_bound:
                root_print(f"  warning: {on_bound} start exactly on a bound and may be unable to move; start strictly inside the bounds")
            # fixed coefficients are excluded by free_mask, so their bounds must just contain them
            lb[~free] = np.minimum(lb[~free], initial[~free])
            ub[~free] = np.maximum(ub[~free], initial[~free])
            scale = np.array([
                resolved[name]["scale"] if resolved[name]["scale"] is not None else (abs(resolved[name]["initial"]) or 1.0)
                for name in names
            ], dtype=float)
            problem = self.optimizer.reduced_problem(
                initial, opt_type="coeffs", bounds=(lb, ub), scaling_factors=scale, free_mask=free
            )
            return problem, names, resolved

        if self.target == "ic-params":
            from dimswe.optimize import ParametricInitialCondition

            names = list(self.spec["parameters"])
            ic_names = [name.split(".", 1)[1] for name in names]
            resolved = resolve_parameter_values(self.spec, self.parameters, names)
            initial = np.array([resolved[name]["initial"] for name in names])
            self.ic_map = ParametricInitialCondition(self.model, self.parameters, ic_names)
            free = np.array([resolved[name]["free"] for name in names])
            scale = np.array([
                resolved[name]["scale"] if resolved[name]["scale"] is not None else (abs(resolved[name]["initial"]) or 1.0)
                for name in names
            ], dtype=float)
            problem = self.optimizer.reduced_problem(
                initial, opt_type="ic-params", coeffs0=self.coeff, ic_map=self.ic_map,
                scaling_factors=scale, free_mask=free,
            )
            return problem, names, resolved

        # ic-field: the initial guess is the analytic IC with perturbed parameter values
        from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function
        from workflows.opt_helpers import parameters_with_values, per_variable_scaling, project_initial_state

        names = list(self.spec["parameters"])
        resolved = resolve_parameter_values(self.spec, self.parameters, names)
        guess_parameters = parameters_with_values(
            self.parameters, {name: resolved[name]["initial"] for name in names}
        )
        self.initial_guess_state = project_initial_state(self.model, guess_parameters, "x0_guess")
        initial = create_flattened_numpy_arr_from_mixed_function(self.initial_guess_state).copy()
        scale = per_variable_scaling(self.model, self.truth_states[0][self.schedule["block_start_indices"][0]][0])
        problem = self.optimizer.reduced_problem(initial, opt_type="ics", coeffs0=self.coeff, scaling_factors=scale)
        return problem, None, resolved


def run_taylor_test(problem: Any, job_dir: Path) -> dict[str, Any]:
    """Run a Taylor test of the reduced gradient in a fixed random direction."""
    import numpy as np
    from mpi4py import MPI

    from dimswe.optimize import taylor_test

    # a fixed seed gives the same direction on every rank, as replicated parameters require
    # (the ic-field target, whose vector is rank-local, only runs in serial)
    direction = np.random.default_rng(1234).standard_normal(problem.z0.size)
    eps, remainders, rates = taylor_test(problem, direction)
    result = {"eps": eps, "remainders": remainders, "rates": rates, "expected_rate": 2.0}
    root_print(f"  Taylor test rates (expect ~2): {np.round(rates, 3).tolist()}")
    if MPI.COMM_WORLD.Get_rank() == 0:
        dump_yaml(to_builtin(result), job_dir / "taylor-test.yaml")
    return result


def initial_gradient_report(problem: Any, names: list[str] | None) -> dict[str, Any]:
    """Evaluate the loss and gradient at the starting point and describe them.

    The evaluation is memoized by the ReducedProblem, so the optimizer reuses it at no extra cost.
    Gradient components are in the scaled optimizer variables (value / scale).
    """
    import numpy as np

    loss = problem.fun(problem.z0)
    gradient = np.asarray(problem.jac(problem.z0), dtype=float)
    report: dict[str, Any] = {
        "loss": float(loss),
        "grad_norm": float(np.linalg.norm(gradient)),
        "grad_max": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
    }
    if names is not None:
        free_names = [name for name, free in zip(names, problem.free_mask, strict=True) if free]
        report["gradient"] = {name: float(value) for name, value in zip(free_names, gradient, strict=True)}
    root_print(f"  initial loss {report['loss']:.6e}, |grad| {report['grad_norm']:.3e}, max |grad_i| {report['grad_max']:.3e}")
    for name, value in report.get("gradient", {}).items():
        root_print(f"    dJ/d({name}/scale) = {value:.3e}")
    return report


ZERO_GRADIENT_MESSAGE = (
    "The loss gradient is exactly zero at the initial guess, although the loss is {loss:.6e}, so "
    "the optimizer cannot move. The loss does not depend on the free parameters there. For "
    "threewayphysics this usually means a threshold is never crossed in the training window: rain "
    "only forms where cloud water exceeds qprecip, so if it never does, gamma_r and qprecip have no "
    "effect. Use a longer training window (window.num_steps), an initial guess closer to the truth "
    "(for example a smaller initial qprecip), or check that the parameters are active in the model."
)


def optimizer_gtol(optimizer_spec: dict[str, Any], report: dict[str, Any]) -> float:
    """Return the gtol passed to scipy, scaled by the initial gradient for relative tolerances."""
    gtol = float(optimizer_spec["gtol"])
    tolerance = optimizer_spec.get("tolerance", "relative")
    if tolerance == "absolute":
        return gtol
    if tolerance != "relative":
        raise ValueError("optimizer.tolerance must be 'relative' or 'absolute'")
    return gtol * report["grad_max"]


def true_values(job: dict[str, Any], names: list[str]) -> dict[str, float] | None:
    """Return the truth parameter values for a per-run job (None for joint jobs)."""
    from workflows.common import get_by_dotted_path

    if len(job["truth_runs"]) != 1:
        values = {}
        for name in names:
            run_values = {float(get_by_dotted_path(run["parameters"], name)) for run in job["truth_runs"]}
            if len(run_values) != 1:
                return None
            values[name] = run_values.pop()
        return values
    return {name: float(get_by_dotted_path(job["truth_runs"][0]["parameters"], name)) for name in names}


def run_job(spec: dict[str, Any], job: dict[str, Any], check_gradient: bool) -> None:
    """Run one optimization job and write its outputs."""
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    if spec["target"] == "ic-field" and MPI.COMM_WORLD.Get_size() > 1:
        raise RuntimeError(
            "The ic-field target is serial only: scipy treats the rank-local initial state as the "
            "whole optimization vector. Run it with 1 MPI rank, or use the ic-params target."
        )
    job_dir = job["job_dir"]
    if rank == 0:
        job_dir.mkdir(parents=True, exist_ok=True)
        job_spec = {key: value for key, value in spec.items() if not key.startswith("_")}
        job_spec.update({
            "source_cfg": spec["_source_path"],
            "job_name": job["name"],
            "truth_run_dirs": [str(run["run_dir"]) for run in job["truth_runs"]],
            "loss": job["loss"],
            "schedule": job["schedule"],
        })
        dump_yaml(to_builtin(job_spec), job_dir / "optimize.cfg")
    MPI.COMM_WORLD.barrier()

    root_print(f"Job: {describe_job(job)}")
    state = Job(spec, job)
    problem, names, resolved = state.build_problem()

    if check_gradient or spec["optimizer"]["check_gradient"]:
        run_taylor_test(problem, job_dir)

    report = initial_gradient_report(problem, names)
    if report["grad_max"] == 0.0:
        raise RuntimeError(ZERO_GRADIENT_MESSAGE.format(loss=report["loss"]))

    history = History(names)
    optimizer_spec = spec["optimizer"]
    gtol = optimizer_gtol(optimizer_spec, report)
    x_opt, result = state.optimizer.optimize(
        None,
        method=optimizer_spec["method"],
        gtol=gtol,
        ftol=float(optimizer_spec["ftol"]),
        maxiter=int(optimizer_spec["maxiter"]),
        callback=history,
        return_result=True,
        problem=problem,
    )
    initial_loss = history.rows[0]["loss"] if history.rows else float("nan")
    final_loss = float(result.fun)
    root_print(f"  finished: {result.message} loss {initial_loss:.6e} -> {final_loss:.6e} after {int(getattr(result, 'nit', -1))} iteration(s)")
    if int(getattr(result, "nit", 0)) == 0:
        root_print(
            "  warning: the optimizer stopped without taking a step; see initial_gradient in "
            "result.yaml, and consider a tighter gtol or optimizer.tolerance: relative"
        )

    summary: dict[str, Any] = {
        "created_at": timestamp_utc(),
        "job_name": job["name"],
        "target": spec["target"],
        "loss": job["loss"],
        "loss_label": job["loss_label"],
        "truth_runs": [run["run_name"] for run in job["truth_runs"]],
        "truth_run_dirs": [str(run["run_dir"]) for run in job["truth_runs"]],
        "window": spec["window"],
        "schedule": job["schedule"],
        "normalize": spec["normalize"],
        "optimizer": {
            **optimizer_spec,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "gtol_used": gtol,
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(problem.nfev),
            "njev": int(problem.njev),
        },
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_gradient": report,
        "mpi_world_size": MPI.COMM_WORLD.Get_size(),
    }

    optimal_parameters = deep_copy_parameters(state.parameters)
    if names is not None:
        initial_values = {name: resolved[name]["initial"] for name in names}
        optimal_values = {name: float(value) for name, value in zip(names, x_opt, strict=True)}
        summary["parameters"] = {
            name: {
                "initial": initial_values[name],
                "optimal": optimal_values[name],
                "free": resolved[name]["free"],
            }
            for name in names
        }
        truths = true_values(job, names)
        if truths is not None:
            for name in names:
                entry = summary["parameters"][name]
                entry["true"] = truths[name]
                entry["abs_error"] = abs(entry["optimal"] - truths[name])
                entry["rel_error"] = entry["abs_error"] / abs(truths[name]) if truths[name] != 0 else float("nan")
                entry["initial_abs_error"] = abs(entry["initial"] - truths[name])
        for name, value in optimal_values.items():
            set_by_dotted_path(optimal_parameters, name, value)
        initial_parameters = deep_copy_parameters(state.parameters)
        for name, value in initial_values.items():
            set_by_dotted_path(initial_parameters, name, value)
    else:
        # ic-field: write the optimized initial state, and record which IC parameters seeded the guess
        from dimswe.numpy_helpers import set_mixed_function_from_flattened_array
        from workflows.opt_helpers import OPTIMAL_IC_NAME, save_state

        summary["initial_guess_parameters"] = {name: resolved[name]["initial"] for name in resolved}
        x0_opt, _, _ = state.model.get_x_var(OPTIMAL_IC_NAME)
        set_mixed_function_from_flattened_array(x0_opt, x_opt)
        save_state(job_dir / "optimal_ic.h5", state.mesh, x0_opt, OPTIMAL_IC_NAME)
        save_state(job_dir / "initial_guess_ic.h5", state.mesh, state.initial_guess_state, OPTIMAL_IC_NAME)
        summary["optimal_ic"] = str(job_dir / "optimal_ic.h5")
        summary["initial_guess_ic"] = str(job_dir / "initial_guess_ic.h5")
        initial_parameters = deep_copy_parameters(state.parameters)
        for name in resolved:
            set_by_dotted_path(initial_parameters, name, resolved[name]["initial"])

    if rank == 0:
        write_csv(job_dir / "history.csv", history.rows)
        dump_yaml(to_builtin(optimal_parameters), job_dir / "optimal.cfg")
        dump_yaml(to_builtin(initial_parameters), job_dir / "initial_guess.cfg")
        dump_yaml(to_builtin(summary), job_dir / "result.yaml")
    MPI.COMM_WORLD.barrier()


def main() -> None:
    """Validate the spec, plan jobs, and run each optimization."""
    args = parse_args()
    spec_path = resolve_path(args.optimize_cfg)
    spec = load_spec(spec_path)
    truth_runs = resolve_truth_runs(spec["truth"], spec_path.parent)
    output_root = resolve_runs_root(spec.get("output_root", "runs/optimizations"))
    output_dir = output_root / spec["optimization_name"]
    jobs = plan_jobs(spec, truth_runs, output_dir)
    if args.only:
        jobs = [job for job in jobs if any(token in job["name"] for token in args.only)]

    root_print(f"Optimization '{spec['optimization_name']}' target={spec['target']} -> {output_dir}")
    for job in jobs:
        root_print(f"  planned {describe_job(job)}")
    if args.dry_run:
        return

    from mpi4py import MPI

    failures = []
    for job in jobs:
        if args.skip_existing and (job["job_dir"] / "result.yaml").exists():
            root_print(f"Skipping existing job {job['name']}")
            continue
        try:
            run_job(spec, job, args.check_gradient)
        except Exception as exc:  # pragma: no cover - defensive error reporting
            error_text = str(exc)
        else:
            error_text = None
        errors = [text for text in MPI.COMM_WORLD.allgather(error_text) if text]
        if errors:
            failures.append((job["name"], errors[0]))
            root_print(f"Job failed: {job['name']}: {errors[0]}")
    if failures:
        raise RuntimeError(f"{len(failures)} optimization job(s) failed: {failures}")


if __name__ == "__main__":
    main()
