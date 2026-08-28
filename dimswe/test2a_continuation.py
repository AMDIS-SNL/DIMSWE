"""Checkpointed continuation of the accepted Test 2A memory-20 L-BFGS fit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from pyrol import Problem, Solver
from pyrol.vectors import NumPyVector

from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_operator import (
    DenseMLP,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    normalized_operator_objective,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters,
)
from .test2a_optimizer_study import parameter_fingerprint
from .test2a_pyrol import JAXPytreeObjective, build_test2a_lbfgs_parameters


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_continuation_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    accepted_stages = {
        "Test 2A-1 memory-20 L-BFGS continuation",
        "Test 2A-1 memory-20 L-BFGS convergence-to-plateau continuation",
    }
    if record.get("benchmark_stage") not in accepted_stages:
        raise ValueError("not the selected Test 2A continuation configuration")
    optimizer = record["optimizer"]
    if (
        optimizer["kind"] != "line_search_lbfgs"
        or int(optimizer["maximum_secant_storage"]) != 20
        or optimizer["production_HVP"] is not False
    ):
        raise ValueError("continuation optimizer must remain gradient-only L-BFGS m=20")
    checkpoints = tuple(int(value) for value in record["checkpoint_additional_iterations"])
    limit = int(record["additional_accepted_iteration_limit"])
    if (
        not checkpoints
        or checkpoints != tuple(sorted(set(checkpoints)))
        or checkpoints[0] <= 0
        or checkpoints[-1] != limit
    ):
        raise ValueError("invalid continuation checkpoint schedule")
    return record


def _verify_source_result(source, source_result):
    contract = source.get("result_contract", "optimizer_study_method")
    tolerance = 32.0 * np.finfo(np.float64).eps
    if contract == "optimizer_study_method":
        result_objective = source_result.get("final_objective")
    elif contract == "completed_continuation":
        result_objective = source_result.get("final_metrics", {}).get("normalized_mse")
    else:
        raise ValueError(f"unsupported continuation source contract {contract!r}")
    common = (
        source_result.get("status") == "complete"
        and result_objective is not None
        and np.isclose(
            float(result_objective),
            float(source["final_objective"]),
            rtol=0.0,
            atol=tolerance,
        )
    )
    if contract == "optimizer_study_method":
        valid = common and source_result.get("name") == source.get(
            "expected_result_name", "rol_lbfgs_m20_i500"
        )
    elif contract == "completed_continuation":
        optimizer = source_result.get("optimizer", {})
        valid = (
            common
            and source_result.get("benchmark_stage")
            == source["expected_benchmark_stage"]
            and source_result.get("decision") == source["expected_decision"]
            and int(optimizer.get("additional_accepted_iterations", -1))
            == int(source["accepted_iterations"])
            and int(optimizer.get("maximum_secant_storage", -1)) == 20
            and int(optimizer.get("HVP_evaluations", -1)) == 0
        )
    if not valid:
        raise ValueError("source result does not satisfy the selected continuation contract")


def _physical_metric_stability(periodic_metrics, criteria):
    if len(periodic_metrics) < 2:
        return {"stable": False, "reason": "fewer than two physical metric records"}
    previous = periodic_metrics[-2]
    final = periodic_metrics[-1]
    left = previous["metrics"]
    right = final["metrics"]
    relative_rms_change = abs(
        right["relative_rms_error"] - left["relative_rms_error"]
    ) / max(abs(left["relative_rms_error"]), np.finfo(np.float64).tiny)
    correlation_change = abs(right["correlation"] - left["correlation"])
    sign_change = abs(right["sign_accuracy"] - left["sign_accuracy"])
    activity_changes = {}
    for label in (
        "abs_A_gt_1e-03_max_abs_A",
        "abs_A_gt_1e-02_max_abs_A",
    ):
        previous_accuracy = left["sign_accuracy_strata"][label]["accuracy"]
        final_accuracy = right["sign_accuracy_strata"][label]["accuracy"]
        activity_changes[label] = (
            None
            if previous_accuracy is None or final_accuracy is None
            else abs(final_accuracy - previous_accuracy)
        )
    finite_activity_changes = [
        value for value in activity_changes.values() if value is not None
    ]
    stable = (
        relative_rms_change
        <= float(criteria["plateau_physical_relative_rms_relative_change_maximum"])
        and correlation_change
        <= float(criteria["plateau_physical_correlation_change_maximum"])
        and sign_change
        <= float(criteria["plateau_physical_sign_accuracy_change_maximum"])
        and all(
            value
            <= float(criteria["plateau_physical_active_sign_change_maximum"])
            for value in finite_activity_changes
        )
    )
    return {
        "stable": stable,
        "starting_additional_iteration": int(
            previous["additional_accepted_iteration"]
        ),
        "ending_additional_iteration": int(final["additional_accepted_iteration"]),
        "relative_rms_relative_change": relative_rms_change,
        "correlation_absolute_change": correlation_change,
        "sign_accuracy_absolute_change": sign_change,
        "active_sign_accuracy_absolute_changes": activity_changes,
    }


def verify_parameter_artifact(path, expected_npz_sha256, expected_pytree_sha256):
    source = Path(path)
    actual_file = _file_sha256(source)
    if actual_file != expected_npz_sha256:
        raise ValueError(
            f"parameter NPZ fingerprint mismatch: {actual_file} != {expected_npz_sha256}"
        )
    parameters, configuration = load_mlp_parameters(source)
    actual_tree = parameter_fingerprint(parameters)
    if actual_tree != expected_pytree_sha256:
        raise ValueError(
            f"parameter pytree fingerprint mismatch: {actual_tree} != {expected_pytree_sha256}"
        )
    return parameters, configuration


class CheckpointingPytreeObjective(JAXPytreeObjective):
    """Invoke an external monitor after each initial/accepted ROL update."""

    def __init__(self, *args, accepted_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted_callback = accepted_callback

    def update(self, control, *args):
        previous = len(self.accepted_iteration_history)
        super().update(control, *args)
        if (
            self.accepted_callback is not None
            and len(self.accepted_iteration_history) == previous + 1
        ):
            local_accepted = len(self.accepted_iteration_history) - 1
            self.accepted_callback(control, local_accepted, self)


def _periodic_metric_record(
    parameters,
    model,
    normalization,
    dataset,
    additional_iteration,
):
    prediction = physical_predictions(
        parameters, model, normalization, dataset.features
    )
    return {
        "additional_accepted_iteration": int(additional_iteration),
        "metrics": operator_metrics(prediction, dataset.targets),
        "parameter_pytree_sha256": parameter_fingerprint(parameters),
    }


def _relative_decrease(history, window):
    values = [record["objective"] for record in history if record["objective"] is not None]
    if len(values) <= window:
        return None
    old = float(values[-window - 1])
    new = float(values[-1])
    return {
        "window": int(window),
        "starting_objective": old,
        "ending_objective": new,
        "absolute_decrease": old - new,
        "relative_decrease": (old - new) / max(abs(old), np.finfo(np.float64).tiny),
    }


def _continuation_decision(metrics, stationarity, criteria):
    labels = criteria.get(
        "classification_labels",
        {
            "improving": "CONTINUING_OPTIMIZER_LIMITED",
            "plateau_inadequate": "PRACTICALLY_CONVERGED_MODEL_INADEQUATE",
            "adequate": "READY_FOR_EMBEDDING",
        },
    )
    sign_strata = metrics.get("sign_accuracy_strata", {})
    active_sign_1e3 = sign_strata.get("abs_A_gt_1e-03_max_abs_A", {}).get("accuracy")
    active_sign_1e2 = sign_strata.get("abs_A_gt_1e-02_max_abs_A", {}).get("accuracy")
    active_sign_1e3_ok = (
        "ready_active_1e-3_sign_accuracy_minimum" not in criteria
        or (
            active_sign_1e3 is not None
            and active_sign_1e3
            >= float(criteria["ready_active_1e-3_sign_accuracy_minimum"])
        )
    )
    active_sign_1e2_ok = (
        "ready_active_1e-2_sign_accuracy_minimum" not in criteria
        or (
            active_sign_1e2 is not None
            and active_sign_1e2
            >= float(criteria["ready_active_1e-2_sign_accuracy_minimum"])
        )
    )
    strong_accuracy = (
        metrics["relative_rms_error"] <= float(criteria["ready_relative_rms_maximum"])
        and metrics["magnitude_strata"]["abs_A_gt_1e-03_max_abs_A"][
            "relative_rms_error"
        ]
        <= float(criteria["ready_active_1e-3_relative_rms_maximum"])
        and (
            metrics["sign_accuracy"] is not None
            and metrics["sign_accuracy"]
            >= float(criteria["ready_sign_accuracy_minimum"])
        )
        and active_sign_1e3_ok
        and active_sign_1e2_ok
    )
    recent = stationarity["recent_100_objective_decrease"]
    plateau = (
        recent is not None
        and recent["relative_decrease"]
        <= float(criteria["plateau_recent_100_relative_objective_decrease_maximum"])
        and stationarity["final_gradient_norm_relative_to_start"]
        <= float(criteria["plateau_final_gradient_ratio_maximum"])
        and stationarity["final_step_norm_relative_to_parameter_norm"]
        <= float(criteria["plateau_final_relative_step_maximum"])
        and stationarity.get("physical_metrics_stability", {}).get("stable", True)
    )
    if plateau and strong_accuracy:
        decision = labels["adequate"]
    elif plateau:
        decision = labels["plateau_inadequate"]
    else:
        decision = labels["improving"]
    return decision, {
        "strong_accuracy": strong_accuracy,
        "practical_plateau": plateau,
        "plateau_requires_objective_gradient_and_step_evidence": True,
        "criteria": criteria,
    }


def _plot_continuation(result, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    history = result["accepted_iteration_history"]
    iterations = [record["additional_accepted_iteration"] for record in history]
    objectives = [record["objective"] for record in history]
    gradients = [record["gradient_norm"] for record in history]
    figure, axes = plt.subplots(2, 1, sharex=True)
    axes[0].semilogy(iterations, objectives)
    axes[0].set(ylabel="normalized objective")
    axes[1].semilogy(iterations, gradients)
    axes[1].set(xlabel="additional accepted iteration", ylabel="ROL gradient norm")
    figure.tight_layout()
    figure.savefig(destination / "continuation_objective_gradient.png", dpi=150)
    plt.close(figure)

    periodic = result["periodic_physical_metrics"]
    figure, axis = plt.subplots()
    axis.plot(
        [record["additional_accepted_iteration"] for record in periodic],
        [record["metrics"]["relative_rms_error"] for record in periodic],
        marker="o",
        label="relative RMS(A)",
    )
    axis.plot(
        [record["additional_accepted_iteration"] for record in periodic],
        [record["metrics"]["sign_accuracy"] for record in periodic],
        marker="o",
        label="sign accuracy (existing >1e-6 max)",
    )
    for threshold, label in (
        ("abs_A_gt_1e-03_max_abs_A", "sign accuracy, |A| > 1e-3 max"),
        ("abs_A_gt_1e-02_max_abs_A", "sign accuracy, |A| > 1e-2 max"),
        ("abs_A_gt_1e-01_max_abs_A", "sign accuracy, |A| > 1e-1 max"),
    ):
        axis.plot(
            [record["additional_accepted_iteration"] for record in periodic],
            [
                record["metrics"]["sign_accuracy_strata"][threshold]["accuracy"]
                for record in periodic
            ],
            marker="o",
            label=label,
        )
    axis.set(xlabel="additional accepted iteration", ylabel="metric")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "continuation_physical_metrics.png", dpi=150)
    plt.close(figure)


def run_continuation(
    selected_configuration_path,
    continuation_configuration_path,
    dataset_path,
    output_directory,
    *,
    resume=False,
):
    selected = load_selected_configuration(selected_configuration_path)
    continuation = load_continuation_configuration(continuation_configuration_path)
    dataset, metadata = load_operator_dataset(dataset_path)
    if dataset.sample_count != int(continuation["frozen_problem"]["sample_count"]):
        raise ValueError("continuation dataset sample count changed")
    output_root = Path(output_directory)
    result_path = output_root / "continuation_result.json"
    progress_path = output_root / "continuation_progress.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite complete continuation {result_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    source = continuation["source"]
    source_path = Path(source["parameter_file"])
    source_result = read_json_record(source["result_file"])
    _verify_source_result(source, source_result)
    start_parameters, model_configuration = verify_parameter_artifact(
        source_path,
        source["parameter_npz_sha256"],
        source["parameter_pytree_sha256"],
    )
    if model_configuration.to_record() != mlp_configuration_from_record(
        selected["model"]
    ).to_record():
        raise ValueError("continuation architecture differs from selected Test 2A")

    offset = 0
    previous_history = []
    previous_periodic = []
    previous_runs = []
    cumulative_counts = {"objective": 0, "gradient": 0, "HVP": 0}
    cumulative_wall = 0.0
    secant_history_restored = True
    if resume:
        if not progress_path.exists():
            raise FileNotFoundError("no continuation progress record to resume")
        progress = read_json_record(progress_path)
        offset = int(progress["last_checkpoint_additional_iteration"])
        checkpoint = progress["last_checkpoint_parameter_file"]
        start_parameters, checkpoint_configuration = verify_parameter_artifact(
            checkpoint,
            progress["last_checkpoint_npz_sha256"],
            progress["last_checkpoint_pytree_sha256"],
        )
        if checkpoint_configuration != model_configuration:
            raise ValueError("resume checkpoint architecture changed")
        previous_history = list(progress.get("accepted_iteration_history", []))
        previous_periodic = list(progress.get("periodic_physical_metrics", []))
        previous_runs = list(progress.get("runs", []))
        cumulative_counts = dict(progress.get("cumulative_evaluations", cumulative_counts))
        cumulative_wall = float(progress.get("cumulative_solver_wall_time_seconds", 0.0))
        secant_history_restored = False
    elif progress_path.exists():
        raise FileExistsError(
            f"incomplete progress exists at {progress_path}; use --resume"
        )

    limit = int(continuation["additional_accepted_iteration_limit"])
    remaining = limit - offset
    if remaining <= 0:
        raise ValueError("continuation checkpoint already reached the configured limit")
    normalization = normalization_from_record(metadata["normalization"])
    model = DenseMLP(model_configuration)
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

    checkpoint_set = set(continuation["checkpoint_additional_iterations"])
    periodic_by_iteration = {
        int(record["additional_accepted_iteration"]): record
        for record in previous_periodic
    }
    if offset not in periodic_by_iteration:
        periodic_by_iteration[offset] = _periodic_metric_record(
            start_parameters, model, normalization, dataset, offset
        )
    run_metadata = {
        "starting_additional_iteration": offset,
        "requested_additional_iterations": remaining,
        "secant_history_restored": secant_history_restored,
    }
    def accepted_callback(control, local_accepted, adapter):
        if local_accepted == 0:
            return
        global_accepted = offset + local_accepted
        if global_accepted not in checkpoint_set:
            return
        parameters = adapter.pytree_from_vector(control)
        parameter_path = output_root / f"parameters_plus_{global_accepted:04d}.npz"
        save_mlp_parameters(parameter_path, parameters, model_configuration)
        file_fingerprint = _file_sha256(parameter_path)
        tree_fingerprint = parameter_fingerprint(parameters)
        periodic_by_iteration[global_accepted] = _periodic_metric_record(
            parameters,
            model,
            normalization,
            dataset,
            global_accepted,
        )
        local_history = []
        for local_index, record in enumerate(adapter.accepted_iteration_history[:-1]):
            local_history.append(
                {
                    "additional_accepted_iteration": offset + local_index,
                    "objective": record["objective"],
                    "gradient_norm": record["gradient_norm"],
                }
            )
        progress = {
            "status": "in_progress",
            "last_checkpoint_additional_iteration": global_accepted,
            "last_checkpoint_parameter_file": str(parameter_path.resolve()),
            "last_checkpoint_npz_sha256": file_fingerprint,
            "last_checkpoint_pytree_sha256": tree_fingerprint,
            "accepted_iteration_history": previous_history + local_history,
            "periodic_physical_metrics": [
                periodic_by_iteration[key] for key in sorted(periodic_by_iteration)
            ],
            "runs": previous_runs + [run_metadata],
            "cumulative_evaluations": {
                "objective": cumulative_counts["objective"] + adapter.value_evaluations,
                "gradient": cumulative_counts["gradient"] + adapter.gradient_evaluations,
                "HVP": cumulative_counts["HVP"] + adapter.hvp_evaluations,
            },
            "cumulative_solver_wall_time_seconds": cumulative_wall,
            "resume_contract": (
                "parameters resume exactly from the latest checkpoint; ROL secant "
                "history is process-local and resets after interruption"
            ),
        }
        write_json_record(progress_path, progress)
        write_json_record(
            output_root / f"checkpoint_plus_{global_accepted:04d}.json",
            {
                "additional_accepted_iteration": global_accepted,
                "parameter_file": str(parameter_path.resolve()),
                "parameter_npz_sha256": file_fingerprint,
                "parameter_pytree_sha256": tree_fingerprint,
                "physical_metrics": periodic_by_iteration[global_accepted]["metrics"],
            },
        )

    adapter = CheckpointingPytreeObjective(
        objective,
        start_parameters,
        use_jit=True,
        accepted_callback=accepted_callback,
    )
    control = adapter.vector_from_pytree(start_parameters)
    warm_started = perf_counter()
    initial_value = adapter.value(control, 0.0)
    initial_gradient = NumPyVector(np.full(adapter.dimension, np.nan, dtype=np.float64))
    adapter.gradient(initial_gradient, control, 0.0)
    jit_warmup_wall = float(perf_counter() - warm_started)
    starting_gradient_norm = float(np.linalg.norm(initial_gradient.array))
    adapter.reset_accounting()

    optimizer = continuation["optimizer"]
    rol_parameters = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": optimizer["gradient_tolerance"],
            "step_tolerance": optimizer["step_tolerance"],
            "iteration_limit": remaining,
            "maximum_secant_storage": 20,
        }
    )
    problem = Problem(adapter, control)
    solver = Solver(problem, rol_parameters)
    started = perf_counter()
    solver.solve()
    solver_wall = float(perf_counter() - started)
    state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    final_additional = offset + int(state.iter)
    final_parameter_path = output_root / "continuation_final_parameters.npz"
    save_mlp_parameters(final_parameter_path, final_parameters, model_configuration)

    local_history = []
    local_controls = []
    for local_index, record in enumerate(adapter.accepted_iteration_history):
        local_history.append(
            {
                "additional_accepted_iteration": offset + local_index,
                "objective": record["objective"],
                "gradient_norm": record["gradient_norm"],
            }
        )
        local_controls.append(np.asarray(record["control"], dtype=np.float64))
    if previous_history and local_history:
        combined_history = previous_history + local_history[1:]
    else:
        combined_history = previous_history + local_history
    if final_additional not in periodic_by_iteration:
        periodic_by_iteration[final_additional] = _periodic_metric_record(
            final_parameters, model, normalization, dataset, final_additional
        )
    final_metrics = periodic_by_iteration[final_additional]["metrics"]
    start_metrics = periodic_by_iteration[0]["metrics"]
    if len(local_controls) >= 2:
        final_step = local_controls[-1] - local_controls[-2]
        final_parameter = local_controls[-1]
        final_relative_step = float(
            np.linalg.norm(final_step) / max(1.0, np.linalg.norm(final_parameter))
        )
    else:
        final_relative_step = None
    final_gradient_norm = float(state.gnorm)
    original_continuation_gradient = float(source["final_gradient_norm"])
    ordered_periodic = [
        periodic_by_iteration[key] for key in sorted(periodic_by_iteration)
    ]
    stationarity = {
        "actual_ROL_gradient_norm": final_gradient_norm,
        "continuation_start_gradient_norm_recomputed": starting_gradient_norm
        if offset == 0
        else None,
        "source_continuation_start_gradient_norm": original_continuation_gradient,
        "final_gradient_norm_relative_to_start": final_gradient_norm
        / original_continuation_gradient,
        "final_step_norm_relative_to_parameter_norm": final_relative_step,
        "recent_50_objective_decrease": _relative_decrease(combined_history, 50),
        "recent_100_objective_decrease": _relative_decrease(combined_history, 100),
        "stationarity_not_inferred_from_objective_alone": True,
    }
    if "plateau_physical_relative_rms_relative_change_maximum" in continuation[
        "decision_diagnostics"
    ]:
        stationarity["physical_metrics_stability"] = _physical_metric_stability(
            ordered_periodic, continuation["decision_diagnostics"]
        )
    decision, decision_evidence = _continuation_decision(
        final_metrics, stationarity, continuation["decision_diagnostics"]
    )
    counts = {
        "objective": cumulative_counts["objective"] + adapter.value_evaluations,
        "gradient": cumulative_counts["gradient"] + adapter.gradient_evaluations,
        "HVP": cumulative_counts["HVP"] + adapter.hvp_evaluations,
    }
    result = {
        "status": "complete",
        "benchmark_stage": continuation["benchmark_stage"],
        "selected_configuration": str(Path(selected_configuration_path).resolve()),
        "continuation_configuration": str(
            Path(continuation_configuration_path).resolve()
        ),
        "dataset": str(Path(dataset_path).resolve()),
        "source_parameter_verification": {
            "parameter_file": str(source_path.resolve()),
            "parameter_npz_sha256": source["parameter_npz_sha256"],
            "parameter_pytree_sha256": source["parameter_pytree_sha256"],
            "verified": True,
        },
        "optimizer": {
            **optimizer,
            "additional_accepted_iterations": final_additional,
            "actual_ROL_termination_reason": str(state.statusFlag),
            "objective_evaluations": counts["objective"],
            "gradient_evaluations": counts["gradient"],
            "HVP_evaluations": counts["HVP"],
            "jit_warmup_wall_time_seconds_last_process": jit_warmup_wall,
            "solver_wall_time_seconds": cumulative_wall + solver_wall,
            "runs": previous_runs + [run_metadata],
        },
        "accepted_iteration_history": combined_history,
        "periodic_physical_metrics": ordered_periodic,
        "stationarity_diagnostics": stationarity,
        "final_metrics": final_metrics,
        "comparison": {
            source.get("comparison_label", "m20_500"): {
                "objective": float(source["final_objective"]),
                "metrics": start_metrics,
            },
            "affine_baseline": metadata["diagnostic_baselines"][
                "affine_normalized_five_input"
            ]["metrics"],
            "objective_decrease_from_continuation_start": float(source["final_objective"])
            - float(state.value),
            "relative_RMS_error_reduction_from_continuation_start": (
                start_metrics["relative_rms_error"]
                - final_metrics["relative_rms_error"]
            )
            / start_metrics["relative_rms_error"],
            "correlation_change_from_continuation_start": final_metrics["correlation"]
            - start_metrics["correlation"],
            "sign_accuracy_change_from_continuation_start": final_metrics["sign_accuracy"]
            - start_metrics["sign_accuracy"],
        },
        "decision": decision,
        "decision_evidence": decision_evidence,
        "final_parameter_file": str(final_parameter_path.resolve()),
        "truth_state_access": {
            "prepared_dataset_only": True,
            "truth_snapshots_opened": False,
            "states_after_80_accessed": False,
        },
    }
    write_json_record(result_path, result)
    write_json_record(
        progress_path,
        {
            "status": "complete",
            "last_checkpoint_additional_iteration": final_additional,
            "last_checkpoint_parameter_file": str(final_parameter_path.resolve()),
            "last_checkpoint_npz_sha256": _file_sha256(final_parameter_path),
            "last_checkpoint_pytree_sha256": parameter_fingerprint(final_parameters),
            "accepted_iteration_history": combined_history,
            "periodic_physical_metrics": result["periodic_physical_metrics"],
            "runs": result["optimizer"]["runs"],
            "cumulative_evaluations": counts,
            "cumulative_solver_wall_time_seconds": cumulative_wall + solver_wall,
        },
    )
    _plot_continuation(result, output_root)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--continuation-configuration", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    run_continuation(
        arguments.configuration,
        arguments.continuation_configuration,
        arguments.dataset,
        arguments.output_directory,
        resume=arguments.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CheckpointingPytreeObjective",
    "load_continuation_configuration",
    "run_continuation",
    "verify_parameter_artifact",
)
