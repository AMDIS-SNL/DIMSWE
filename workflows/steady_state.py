"""Steady-state (time-invariant) coefficient optimization for optimize_params.py (TODO-elliptic §4.10).

Used when the truth runs are elliptic models (model.type: elliptic). One forward solve per loss
evaluation and one adjoint solve per gradient, via dimswe.optimize.SteadyStateConstrainedOptimization;
there is no time window. Spec (same file format as the time-dependent optimizations):

    optimization_name: kappa-recovery
    truth: {runs: [runs/elliptic-truth], mode: per-run}
    target: coeffs
    parameters:
      elliptic.coefficients.kappa: {initial: 1.0, bounds: [0.1, 10.0], scale: 1.0, log: true}
      elliptic.coefficients.mu:    {initial: 0.5, field: DG0}     # a DG0 field, uniform initial value
    losses:
      - {type: steady, fields: [u]}                               # L2 misfit of the solution fields
      - {type: steady, fields: [sigma, u], label: sigma-and-u, regularization: 1.0e-8}
    optimizer: {method: L-BFGS-B, gtol: 1.0e-8, ftol: 1.0e-12, maxiter: 100}

Parameters are elliptic coefficients ('mu', 'kappa', 'c'); scalars live in R, `field: DG0` (or
another scalar space) optimizes a field. `log: true` optimizes theta = log(coefficient); initial
values and bounds are always given in coefficient units, and results are reported in them. Only
'per-run' truth mode is supported (each truth run is its own steady-state inverse problem).

The model is built from the truth cfg (so the source and boundary data, which come from the
coefficients' cfg values, are those of the truth) with the listed coefficients marked as
parameters, and the optimization starts from the given initial values.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from workflows.common import (deep_copy_parameters, dump_yaml, format_value_for_name, get_by_dotted_path,
                              has_dotted_path, root_print, set_by_dotted_path, timestamp_utc, to_builtin,
                              write_csv)

COEFFICIENT_PREFIX = "elliptic.coefficients."
STEADY_LOSS_KEYS = {"type", "fields", "label", "weights", "regularization"}
STEADY_PARAMETER_KEYS = {"initial", "bounds", "scale", "log", "field", "fixed"}


def is_time_invariant_cfg(parameters: dict[str, Any]) -> bool:
    """True for elliptic truth runs (the only time-invariant models with coefficients)."""
    return parameters.get("model", {}).get("type") == "elliptic"


def coefficient_name(path: str) -> str:
    """'elliptic.coefficients.kappa' -> 'kappa' (ValueError for other paths)."""
    if not path.startswith(COEFFICIENT_PREFIX) or path.count(".") != 2:
        raise ValueError(f"steady-state parameters must be elliptic coefficients like "
                         f"'{COEFFICIENT_PREFIX}kappa', got '{path}'")
    return path[len(COEFFICIENT_PREFIX):]


def steady_loss_label(loss: dict[str, Any]) -> str:
    """Filesystem-friendly label of a steady loss ('steady-u', 'steady-sigma-u', or its label)."""
    if "label" in loss:
        return format_value_for_name(loss["label"])
    return "steady-" + "-".join(loss.get("fields", ["u"]))


def validate_steady_spec(spec: dict[str, Any]) -> None:
    """Check a steady-state spec (losses, parameters, truth mode); raises ValueError."""
    if spec.get("target") != "coeffs":
        raise ValueError("steady-state optimizations only support target: coeffs")
    if spec.get("truth", {}).get("mode", "per-run") != "per-run":
        raise ValueError("steady-state optimizations only support truth.mode: per-run")
    losses = spec.get("losses") or []
    if not losses:
        raise ValueError("at least one loss must be listed")
    for loss in losses:
        unknown = set(loss) - STEADY_LOSS_KEYS
        if unknown:
            raise ValueError(f"steady loss has unknown keys {sorted(unknown)}")
        if loss.get("type") != "steady":
            raise ValueError(f"steady-state optimizations need losses of type 'steady', got {loss.get('type')!r}")
        fields = loss.setdefault("fields", ["u"])
        if not fields or any(f not in ("sigma", "u") for f in fields):
            raise ValueError(f"steady loss fields must be a non-empty subset of [sigma, u], got {fields}")
        if float(loss.get("regularization", 0.0)) < 0.0:
            raise ValueError("steady loss regularization must be non-negative")
    labels = [steady_loss_label(loss) for loss in losses]
    if len(set(labels)) != len(labels):
        raise ValueError(f"loss labels must be unique, got {labels}; add a 'label'")
    parameters = spec.get("parameters") or {}
    if not parameters:
        raise ValueError("target 'coeffs' needs a parameters section")
    free = 0
    for path, entry in parameters.items():
        coefficient_name(path)
        entry = entry if isinstance(entry, dict) else {"initial": entry}
        unknown = set(entry) - STEADY_PARAMETER_KEYS
        if unknown:
            raise ValueError(f"parameter '{path}' has unknown keys {sorted(unknown)}")
        bounds = entry.get("bounds")
        if bounds is not None and (len(bounds) != 2 or not float(bounds[0]) < float(bounds[1])):
            raise ValueError(f"parameter '{path}' bounds must be [low, high] with low < high")
        if entry.get("log") and bounds is not None and float(bounds[0]) <= 0.0:
            raise ValueError(f"parameter '{path}' with log: true needs positive bounds")
        if entry.get("scale") is not None and float(entry["scale"]) <= 0.0:
            raise ValueError(f"parameter '{path}' scale must be positive")
        free += 0 if entry.get("fixed", False) else 1
    if free == 0:
        raise ValueError("at least one parameter must be free")


def truth_coefficient_value(truth_parameters: dict[str, Any], name: str) -> float | None:
    """The truth run's constant scalar value of a coefficient (None if not a constant scalar)."""
    path = COEFFICIENT_PREFIX + name
    entry = get_by_dotted_path(truth_parameters, path) if has_dotted_path(truth_parameters, path) else {}
    entry = entry or {}
    if entry.get("kind", "constant") != "constant" or entry.get("rank", "scalar") != "scalar":
        return None
    return float(entry.get("value", {"mu": 1.0, "kappa": 1.0, "c": 0.0}[name]))


def model_parameters(truth_parameters: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Truth cfg with the listed coefficients turned into model coefficients (spec overrides applied).

    The coefficients keep the truth's kind/value (so the problem data are the truth's); only
    treat_as_coeff, the parameter space, log parametrization, bounds and scaling change. Field
    parameters (field: DG0) are initialized from the truth value, or from the problem when the
    truth coefficient is a problem expression.
    """
    parameters = deep_copy_parameters(truth_parameters)
    for path, value in (spec.get("overrides") or {}).items():
        set_by_dotted_path(parameters, path, value)
    parameters.setdefault("elliptic", {}).setdefault("coefficients", {})
    coefficients = parameters["elliptic"]["coefficients"] = dict(parameters["elliptic"]["coefficients"] or {})
    for path, entry in spec["parameters"].items():
        entry = entry if isinstance(entry, dict) else {"initial": entry}
        name = coefficient_name(path)
        base = dict(coefficients.get(name) or {})
        if base.get("rank", "scalar") != "scalar":
            raise ValueError(f"coefficient '{name}' is a tensor in the truth cfg; only scalars can be optimized")
        if entry.get("field"):
            initial_from_problem = base.get("kind") == "problem" or base.get("init") == "problem"
            base = {"kind": "field", "space": entry["field"],
                    "init": "problem" if initial_from_problem else "value",
                    "value": base.get("value", {"mu": 1.0, "kappa": 1.0, "c": 0.0}[name])}
        elif base.get("kind", "constant") != "constant":
            raise ValueError(f"coefficient '{name}' is not a constant in the truth cfg; optimize it as a field "
                             f"(field: DG0)")
        base["treat_as_coeff"] = True
        base["log_parametrization"] = bool(entry.get("log", False))
        if entry.get("bounds") is not None:
            base["bounds"] = [float(b) for b in entry["bounds"]]
        if entry.get("scale") is not None:
            base["scaling"] = float(entry["scale"])
        coefficients[name] = base
    return parameters


def to_parameter_value(value: float, log: bool) -> float:
    """Coefficient units -> optimization variable (theta = log(value) for log)."""
    if log:
        if value <= 0.0:
            raise ValueError("log-parametrized coefficients need positive values")
        return math.log(value)
    return float(value)


def from_parameter_value(theta: float, log: bool) -> float:
    """Optimization variable -> coefficient units."""
    return math.exp(theta) if log else float(theta)


def check_loss_fields(truth_parameters: dict[str, Any], fields: list[str]) -> None:
    """Reject loss fields the truth run does not have: sigma does not exist for k = 0.

    Checked from the truth cfg before any checkpoint is opened, so the error is explicit instead of
    a missing-function error from the checkpoint loader.
    """
    k = (truth_parameters.get("elliptic") or {}).get("form_degree")
    if k == 0 and "sigma" in fields:
        raise ValueError("loss field 'sigma' does not exist for elliptic.form_degree 0 (the primal problem has no "
                         "sigma); use fields: [u]")


class SteadyStateJob:
    """One steady-state optimization: truth data, model, optimizer and reduced problem."""

    def __init__(self, spec: dict[str, Any], job: dict[str, Any]):
        import numpy as np
        from firedrake import Function

        from dimswe.elliptic import EllipticProblem
        from dimswe.optimize import SteadyStateConstrainedOptimization
        from workflows.opt_helpers import load_truth_run

        self.spec, self.job = spec, job
        self.loss = job["loss"]
        run = job["truth_runs"][0]
        self.truth_parameters = run["parameters"]
        fields = list(self.loss["fields"])
        check_loss_fields(self.truth_parameters, fields)
        mesh, loaded = load_truth_run(run["checkpoint"], [0], names=fields)
        self.parameters = model_parameters(self.truth_parameters, spec)
        self.model = EllipticProblem(self.parameters, mesh=mesh)
        spaces = {"u": self.model.Vk, "sigma": self.model.Vsigma}
        data = {}
        for name in fields:
            if spaces[name] is None:
                raise ValueError(f"loss field '{name}' does not exist for k = {self.model.k}")
            data[name] = Function(spaces[name], name=f"{name}_truth")
            data[name].dat.data[:] = loaded[name][0].dat.data_ro
        self.optimizer = SteadyStateConstrainedOptimization(
            self.model, data, fields=fields, var_weights=self.loss.get("weights"),
            regularization=float(self.loss.get("regularization", 0.0)))
        self.optimizer.verbose = False
        self.names = self.model.get_coeff_list()
        self.entries = {coefficient_name(path): (entry if isinstance(entry, dict) else {"initial": entry})
                        for path, entry in spec["parameters"].items()}
        self.np = np

    def _per_dof(self, values):
        """Expand per-coefficient values to the flattened parameter dofs (coefficient-list order)."""
        return self.model.coefficients._per_dof(values)

    def build_problem(self):
        """ReducedProblem starting from the spec's initial values; free mask, bounds and scaling per dof."""
        np = self.np
        coefficients = self.model.config.coefficients
        initial, free = [], []
        for name in self.names:
            entry = self.entries[name]
            log = coefficients[name].log_parametrization
            value = entry.get("initial")
            if value is None:
                value = truth_coefficient_value(self.truth_parameters, name)
                if value is None:
                    raise ValueError(f"parameter '{name}' needs an initial value (the truth has no constant value)")
            initial.append(to_parameter_value(float(value), log))
            free.append(not entry.get("fixed", False))
        x0 = self._per_dof(initial)
        free_mask = self._per_dof([1.0 if f else 0.0 for f in free]).astype(bool)
        lb, ub = self.model.get_coeff_bounds()
        lb, ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
        # fixed parameters are excluded by free_mask, so their bounds just need to contain them
        lb[~free_mask] = np.minimum(lb[~free_mask], x0[~free_mask])
        ub[~free_mask] = np.maximum(ub[~free_mask], x0[~free_mask])
        if np.any(free_mask & ((x0 < lb) | (x0 > ub))):
            raise ValueError("initial values lie outside their bounds")
        bounds = None if (np.all(np.isinf(lb)) and np.all(np.isinf(ub))) else (lb, ub)
        self.x0 = x0
        return self.optimizer.reduced_problem(x0, bounds=bounds, scaling_factors=self.model.get_coeff_scaling_factors(),
                                              free_mask=free_mask, use_model_bounds=False)

    def summarize_values(self, x):
        """{name: value in coefficient units} for scalar R parameters, {name: [min, mean, max]} for fields."""
        np = self.np
        coefficients = self.model.config.coefficients
        values, offset = {}, 0
        for name, block in zip(self.names, self.model.coefficients._blocks(self.model.coefficients.coeff)):
            n = int(np.asarray(block.dat.data_ro).size)
            part = np.array([from_parameter_value(t, coefficients[name].log_parametrization) for t in x[offset:offset + n]])
            offset += n
            if coefficients[name].kind == "constant":
                values[name] = float(part[0])
            else:
                values[name] = {"min": float(part.min()), "mean": float(part.mean()), "max": float(part.max())}
        return values


def run_steady_job(spec: dict[str, Any], job: dict[str, Any], check_gradient: bool) -> None:
    """Run one steady-state optimization job and write optimize.cfg, history.csv, result.yaml, optimal.cfg."""
    from mpi4py import MPI

    from workflows.optimize_params import History, initial_gradient_report, optimizer_gtol, run_taylor_test

    if MPI.COMM_WORLD.Get_size() > 1:
        has_fields = any(isinstance(e, dict) and e.get("field") for e in spec["parameters"].values())
        if has_fields:
            raise RuntimeError("field parameters are serial only: scipy treats the rank-local field as the whole "
                               "optimization vector; run with 1 MPI rank or optimize scalar coefficients")
    rank = MPI.COMM_WORLD.Get_rank()
    job_dir = job["job_dir"]
    if rank == 0:
        job_dir.mkdir(parents=True, exist_ok=True)
        job_spec = {key: value for key, value in spec.items() if not key.startswith("_")}
        job_spec.update({"source_cfg": spec["_source_path"], "job_name": job["name"],
                         "truth_run_dirs": [str(run["run_dir"]) for run in job["truth_runs"]], "loss": job["loss"]})
        dump_yaml(to_builtin(job_spec), job_dir / "optimize.cfg")
    MPI.COMM_WORLD.barrier()

    root_print(f"Job: {job['name']} (steady state, fields {job['loss']['fields']})")
    state = SteadyStateJob(spec, job)
    problem = state.build_problem()
    if check_gradient or spec["optimizer"].get("check_gradient", False):
        run_taylor_test(problem, job_dir)
    report = initial_gradient_report(problem, None)
    if report["grad_max"] == 0.0:
        raise RuntimeError("the loss gradient is exactly zero at the initial guess; check that the parameters affect "
                           "the solution fields in the loss")
    history = History(None)
    optimizer_spec = spec["optimizer"]
    gtol = optimizer_gtol(optimizer_spec, report)
    x_opt, result = state.optimizer.optimize(method=optimizer_spec["method"], gtol=gtol,
                                             ftol=float(optimizer_spec["ftol"]), maxiter=int(optimizer_spec["maxiter"]),
                                             callback=history, return_result=True, problem=problem)
    initial_loss = history.rows[0]["loss"] if history.rows else float("nan")
    root_print(f"  finished: {result.message} loss {initial_loss:.6e} -> {float(result.fun):.6e}")

    initial_values = state.summarize_values(state.x0)
    optimal_values = state.summarize_values(x_opt)
    parameters_summary = {}
    for name in state.names:
        entry = {"initial": initial_values[name], "optimal": optimal_values[name],
                 "free": not state.entries[name].get("fixed", False)}
        truth = truth_coefficient_value(state.truth_parameters, name)
        if truth is not None and isinstance(entry["optimal"], float):
            entry["true"] = truth
            entry["abs_error"] = abs(entry["optimal"] - truth)
            entry["rel_error"] = entry["abs_error"] / abs(truth) if truth != 0 else float("nan")
        parameters_summary[name] = entry

    summary = {
        "created_at": timestamp_utc(), "job_name": job["name"], "target": "coeffs", "steady_state": True,
        "loss": job["loss"], "loss_label": job["loss_label"],
        "truth_runs": [run["run_name"] for run in job["truth_runs"]],
        "truth_run_dirs": [str(run["run_dir"]) for run in job["truth_runs"]],
        "optimizer": {**optimizer_spec, "success": bool(result.success), "status": int(result.status),
                      "message": str(result.message), "gtol_used": gtol, "nit": int(getattr(result, "nit", -1)),
                      "nfev": int(problem.nfev), "njev": int(problem.njev),
                      "forward_solves": int(state.optimizer.nsolves)},
        "initial_loss": initial_loss, "final_loss": float(result.fun), "initial_gradient": report,
        "parameters": parameters_summary, "mpi_world_size": MPI.COMM_WORLD.Get_size(),
    }
    # optimal.cfg: constants written back as values; fields keep their cfg (the field is saved separately).
    optimal_parameters = deep_copy_parameters(state.truth_parameters)
    for name, value in optimal_values.items():
        if isinstance(value, float):
            set_by_dotted_path(optimal_parameters, COEFFICIENT_PREFIX + name + ".value", value)
    if any(not isinstance(v, float) for v in optimal_values.values()):
        from workflows.opt_helpers import save_state
        state.model.set_numpy_coeff(x_opt)
        save_state(job_dir / "optimal_coefficients.h5", state.model.mesh, state.model.coefficients.coeff,
                   "optimal_coefficients")
        summary["optimal_coefficients"] = str(job_dir / "optimal_coefficients.h5")
    if rank == 0:
        write_csv(job_dir / "history.csv", history.rows)
        dump_yaml(to_builtin(optimal_parameters), job_dir / "optimal.cfg")
        dump_yaml(to_builtin(summary), job_dir / "result.yaml")
    MPI.COMM_WORLD.barrier()


def plan_steady_jobs(spec: dict[str, Any], truth_runs: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """One job per (truth run, loss)."""
    jobs = []
    for run in truth_runs:
        if not is_time_invariant_cfg(run["parameters"]):
            raise ValueError("all truth runs of a steady-state optimization must be elliptic runs")
        for loss in spec["losses"]:
            label = steady_loss_label(loss)
            name = f"{format_value_for_name(run['run_name'])}__{label}"
            jobs.append({"name": name, "group": run["run_name"], "truth_runs": [run], "loss": loss,
                         "loss_label": label, "schedule": None, "job_dir": output_dir / name})
    return jobs
