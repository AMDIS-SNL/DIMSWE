"""Controlled ROL optimizer study for the frozen Test 2A-1 objective.

Every method starts from the same seed-0 parameter pytree and uses the same
training-only normalized, deterministic full-batch objective.  This module
changes optimizer dispatch only; it does not construct data, alter the selected
Test 2A configuration, or access any truth trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from pyrol import ParameterList, Problem, Solver
from pyrol.vectors import NumPyVector

from .resolved_hidden_c0 import write_json_record
from .test2a_operator import (
    DenseMLP,
    initialize_mlp,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    normalized_operator_objective,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters,
)
from .test2a_pyrol import JAXPytreeObjective, build_test2a_lbfgs_parameters


SUPPORTED_METHOD_KINDS = (
    "line_search_lbfgs",
    "trust_region_truncated_cg_exact_hvp",
)


def load_optimizer_study_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("benchmark_stage") != "Test 2A-1 controlled optimizer study":
        raise ValueError("not a Test 2A optimizer-study configuration")
    names = []
    for method in record["methods"]:
        if method["kind"] not in SUPPORTED_METHOD_KINDS:
            raise ValueError(f"unsupported optimizer kind {method['kind']}")
        if method["name"] in names:
            raise ValueError("optimizer-study method names must be unique")
        names.append(method["name"])
    if len(names) < 3:
        raise ValueError("study must include two L-BFGS methods and exact-HVP ROL")
    return record


def parameter_fingerprint(parameters):
    flat, _ = ravel_pytree(parameters)
    values = np.ascontiguousarray(flat, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def build_study_rol_parameters(method):
    """Dispatch only exact installed ROL method names and interfaces."""
    kind = method["kind"]
    common = {
        "gradient_tolerance": float(method["gradient_tolerance"]),
        "step_tolerance": float(method["step_tolerance"]),
        "iteration_limit": int(method["iteration_limit"]),
    }
    if kind == "line_search_lbfgs":
        return build_test2a_lbfgs_parameters(
            {
                **common,
                "maximum_secant_storage": int(method["maximum_secant_storage"]),
            }
        )
    if kind == "trust_region_truncated_cg_exact_hvp":
        parameters = build_test2a_lbfgs_parameters(
            {**common, "maximum_secant_storage": 1}
        )
        step = parameters.sublist("Step")
        step.set("Type", "Trust Region")
        trust_region = ParameterList("Trust Region")
        trust_region.set("Subproblem Solver", "Truncated CG")
        trust_region.set("Initial Radius", float(method["initial_radius"]))
        trust_region.set("Maximum Radius", float(method["maximum_radius"]))
        step.set("Trust Region", trust_region)
        general = parameters.sublist("General")
        secant = general.sublist("Secant")
        secant.set("Use as Hessian", False)
        secant.set("Use as Preconditioner", False)
        krylov = ParameterList("Krylov")
        krylov.set("Type", "Conjugate Gradients")
        krylov.set("Absolute Tolerance", float(method["krylov_absolute_tolerance"]))
        krylov.set("Relative Tolerance", float(method["krylov_relative_tolerance"]))
        krylov.set("Iteration Limit", int(method["krylov_iteration_limit"]))
        general.set("Krylov", krylov)
        return parameters
    raise ValueError(f"unsupported optimizer kind {kind}")


def _warm_adapter(adapter, control, method):
    """Compile method-required callbacks outside the measured ROL solve."""
    started = perf_counter()
    initial_value = adapter.value(control, 0.0)
    gradient = NumPyVector(np.empty(adapter.dimension, dtype=np.float64))
    adapter.gradient(gradient, control, 0.0)
    if method["kind"] == "trust_region_truncated_cg_exact_hvp":
        direction_values = np.linspace(
            -1.0, 1.0, adapter.dimension, dtype=np.float64
        )
        direction_values /= np.linalg.norm(direction_values)
        direction = NumPyVector(direction_values)
        action = NumPyVector(np.empty(adapter.dimension, dtype=np.float64))
        adapter.hessVec(action, direction, control, 0.0)
    elapsed = float(perf_counter() - started)
    initial_gradient_norm = float(np.linalg.norm(gradient.array))
    adapter.reset_accounting()
    return initial_value, initial_gradient_norm, elapsed


def run_rol_method(objective, initial_parameters, method):
    """Run one fair common-start ROL method and preserve exact termination."""
    adapter = JAXPytreeObjective(objective, initial_parameters, use_jit=True)
    control = adapter.vector_from_pytree(initial_parameters)
    start_fingerprint = parameter_fingerprint(initial_parameters)
    initial_value, initial_gradient_norm, compile_wall = _warm_adapter(
        adapter, control, method
    )
    problem = Problem(adapter, control)
    parameters = build_study_rol_parameters(method)
    solver = Solver(problem, parameters)
    started = perf_counter()
    solver.solve()
    solver_wall = float(perf_counter() - started)
    state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    accepted = []
    for index, record in enumerate(adapter.accepted_iteration_history):
        accepted.append(
            {
                "accepted_index": index,
                "rol_update_type": record["rol_update_type"],
                "rol_iteration_argument": record["rol_iteration_argument"],
                "objective": record["objective"],
                "gradient_norm": record["gradient_norm"],
            }
        )
    result = {
        "name": method["name"],
        "kind": method["kind"],
        "configuration": dict(method),
        "starting_parameter_sha256": start_fingerprint,
        "initial_objective": initial_value,
        "final_objective": float(state.value),
        "initial_gradient_norm": initial_gradient_norm,
        "final_gradient_norm": float(state.gnorm),
        "accepted_iterations": int(state.iter),
        "termination_reason": str(state.statusFlag),
        "objective_evaluations": int(adapter.value_evaluations),
        "gradient_evaluations": int(adapter.gradient_evaluations),
        "HVP_evaluations": int(adapter.hvp_evaluations),
        "rol_reported_objective_evaluations": int(state.nfval),
        "rol_reported_gradient_evaluations": int(state.ngrad),
        "jit_warmup_wall_time_seconds": compile_wall,
        "solver_wall_time_seconds": solver_wall,
        "total_method_wall_time_seconds": compile_wall + solver_wall,
        "accepted_iteration_history": accepted,
        "line_search_trial_objective_history": [
            float(value) for value in adapter.value_history
        ],
        "gradient_callback_norm_history": [
            float(value) for value in adapter.gradient_norm_history
        ],
        "exact_JAX_gradients": True,
        "exact_JAX_HVP_available": True,
        "exact_JAX_HVP_requested": adapter.hvp_evaluations > 0,
    }
    return final_parameters, result


def _conservative_classification(records, reference):
    """Return a transparent diagnostic classification, not a universal gate."""
    completed = [record for record in records if record.get("status") == "complete"]
    if not completed:
        return "OPTIMIZATION_AND_MODEL_BOTH_UNRESOLVED", {
            "reason": "no optimizer completed"
        }
    best = min(completed, key=lambda record: record["final_objective"])
    credible_stationarity = best["final_gradient_norm"] <= max(
        1.0e-6, 1.0e-4 * best["initial_gradient_norm"]
    )
    metrics = best["metrics_complete_training_support"]
    strong_accuracy = (
        metrics["relative_rms_error"] <= 0.25
        and metrics["sign_accuracy"] is not None
        and metrics["sign_accuracy"] >= 0.90
        and metrics["magnitude_strata"]["abs_A_gt_1e-03_max_abs_A"][
            "relative_rms_error"
        ]
        <= 0.30
    )
    substantially_better_than_100 = (
        best["final_objective"] <= 0.75 * float(reference["final_objective"])
        or metrics["relative_rms_error"]
        <= float(reference["relative_rms_error"]) - 0.10
    )
    if not credible_stationarity:
        classification = "OPTIMIZATION_AND_MODEL_BOTH_UNRESOLVED"
    elif strong_accuracy:
        classification = "READY_FOR_EMBEDDING"
    elif substantially_better_than_100:
        classification = "OPTIMIZER_LIMITED_PREVIOUS_RUN"
    else:
        classification = "OPTIMIZATION_CONVERGED_MODEL_INADEQUATE"
    return classification, {
        "best_method": best["name"],
        "credible_stationarity": credible_stationarity,
        "credible_stationarity_rule": "||g|| <= max(1e-6, 1e-4 ||g_initial||)",
        "strong_accuracy": strong_accuracy,
        "strong_accuracy_screen": (
            "relative RMS <= 0.25, sign accuracy >= 0.90, and active-1e-3 "
            "relative RMS <= 0.30; conservative study screen, not universal"
        ),
        "substantially_better_than_100_iteration_reference": substantially_better_than_100,
    }


def _write_study_plots(summary, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    completed = [
        record for record in summary["methods"] if record.get("status") == "complete"
    ]
    figure, axis = plt.subplots()
    for record in completed:
        history = record["accepted_iteration_history"]
        iterations = range(len(history))
        values = [item["objective"] for item in history]
        valid = [(index, value) for index, value in zip(iterations, values) if value is not None]
        if valid:
            axis.semilogy(
                [item[0] for item in valid],
                [item[1] for item in valid],
                label=record["name"],
            )
    axis.set(xlabel="accepted ROL iterate", ylabel="normalized operator objective")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "objective_vs_iteration.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    names = [record["name"] for record in completed]
    errors = [
        record["metrics_complete_training_support"]["relative_rms_error"]
        for record in completed
    ]
    axis.bar(names, errors)
    axis.axhline(
        summary["reference_100_iteration_lbfgs"]["relative_rms_error"],
        color="black",
        linestyle="--",
        label="100-iteration L-BFGS",
    )
    axis.set(ylabel="relative RMS(A) error")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "relative_rms_comparison.png", dpi=150)
    plt.close(figure)


def _write_compact_summaries(summary, output_directory):
    completed = [
        record for record in summary["methods"] if record.get("status") == "complete"
    ]
    cost = []
    accuracy = []
    for record in completed:
        cost.append(
            {
                key: record[key]
                for key in (
                    "name",
                    "kind",
                    "accepted_iterations",
                    "termination_reason",
                    "initial_objective",
                    "final_objective",
                    "initial_gradient_norm",
                    "final_gradient_norm",
                    "objective_evaluations",
                    "gradient_evaluations",
                    "HVP_evaluations",
                    "jit_warmup_wall_time_seconds",
                    "solver_wall_time_seconds",
                    "total_method_wall_time_seconds",
                )
            }
        )
        metrics = record["metrics_complete_training_support"]
        accuracy.append(
            {
                "name": record["name"],
                **{
                    key: metrics[key]
                    for key in (
                        "normalized_mse",
                        "physical_rmse_A",
                        "physical_mae_A",
                        "relative_rms_error",
                        "maximum_absolute_error",
                        "correlation",
                        "sign_accuracy",
                    )
                },
                "active_strata": metrics["magnitude_strata"],
            }
        )
    destination = Path(output_directory)
    write_json_record(
        destination / "convergence_cost_summary.json",
        {
            "classification": summary["classification"],
            "classification_evidence": summary["classification_evidence"],
            "methods": cost,
        },
    )
    write_json_record(
        destination / "physical_accuracy_summary.json",
        {
            "methods": accuracy,
            "diagnostic_baselines": summary["diagnostic_baselines"],
        },
    )

def run_optimizer_study(
    configuration_path,
    study_configuration_path,
    dataset_path,
    output_directory,
):
    selected = load_selected_configuration(configuration_path)
    study = load_optimizer_study_configuration(study_configuration_path)
    dataset, metadata = load_operator_dataset(dataset_path)
    if dataset.sample_count != int(selected["data"]["sample_count"]):
        raise ValueError("optimizer-study dataset does not match selected sample count")
    output_root = Path(output_directory)
    comparison_path = output_root / "optimizer_comparison.json"
    if comparison_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing optimizer study {comparison_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    normalization = normalization_from_record(metadata["normalization"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    model = DenseMLP(model_configuration)
    initial_parameters = initialize_mlp(model_configuration)
    common_fingerprint = parameter_fingerprint(initial_parameters)
    normalized_features = jnp.asarray(
        normalization.normalize_features(dataset.features), dtype=jnp.float64
    )
    normalized_targets = jnp.asarray(
        normalization.normalize_a(dataset.targets), dtype=jnp.float64
    ).reshape(-1, 1)

    def objective(parameters):
        return normalized_operator_objective(
            parameters, model, normalized_features, normalized_targets
        )

    summary = {
        "status": "in_progress",
        "benchmark_stage": "Test 2A-1 controlled optimizer study",
        "selected_configuration": str(Path(configuration_path).resolve()),
        "study_configuration": str(Path(study_configuration_path).resolve()),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_sha256_float64_content": metadata["sha256_float64_content"],
        "frozen_problem": study["frozen_problem"],
        "starting_parameter_sha256": common_fingerprint,
        "reference_100_iteration_lbfgs": study[
            "reference_100_iteration_lbfgs"
        ],
        "full_bfgs_availability_audit": study["full_bfgs_availability_audit"],
        "diagnostic_baselines": metadata["diagnostic_baselines"],
        "truth_state_access": {
            "dataset_only": True,
            "truth_states_opened": False,
            "states_after_80_accessed": False,
        },
        "methods": [],
    }
    write_json_record(comparison_path, summary)
    for method in study["methods"]:
        try:
            parameters, result = run_rol_method(
                objective, initial_parameters, method
            )
            if parameter_fingerprint(initial_parameters) != common_fingerprint:
                raise AssertionError("an optimizer mutated the shared initialization")
            if result["starting_parameter_sha256"] != common_fingerprint:
                raise AssertionError("optimizer did not receive common initial values")
            predictions = physical_predictions(
                parameters, model, normalization, dataset.features
            )
            result["metrics_complete_training_support"] = operator_metrics(
                predictions, dataset.targets
            )
            result["status"] = "complete"
            parameter_path = output_root / f"{method['name']}_parameters.npz"
            save_mlp_parameters(parameter_path, parameters, model_configuration)
            result["trained_parameter_file"] = str(parameter_path.resolve())
            write_json_record(output_root / f"{method['name']}.json", result)
        except Exception as exc:
            result = {
                "name": method["name"],
                "kind": method["kind"],
                "configuration": dict(method),
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "starting_parameter_sha256": common_fingerprint,
            }
            write_json_record(output_root / f"{method['name']}.json", result)
        summary["methods"].append(result)
        write_json_record(comparison_path, summary)
    classification, evidence = _conservative_classification(
        summary["methods"], summary["reference_100_iteration_lbfgs"]
    )
    summary["classification"] = classification
    summary["classification_evidence"] = evidence
    summary["status"] = "complete"
    write_json_record(comparison_path, summary)
    _write_study_plots(summary, output_root)
    _write_compact_summaries(summary, output_root)
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--study-configuration", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    run_optimizer_study(
        arguments.configuration,
        arguments.study_configuration,
        arguments.dataset,
        arguments.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "SUPPORTED_METHOD_KINDS",
    "build_study_rol_parameters",
    "load_optimizer_study_configuration",
    "parameter_fingerprint",
    "run_optimizer_study",
    "run_rol_method",
)
