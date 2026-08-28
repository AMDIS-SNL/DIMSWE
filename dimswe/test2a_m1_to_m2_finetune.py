"""Test-2A diagnostic: operator-pretrained deployed-discrete fine-tuning.

This module does not redefine either offline objective.  It evaluates the
existing exact JAX operator objective and the production-oracle-certified
fixed deployed-discrete cache, and provides diagnostics around a new PyROL/ROL
process initialized from the matched 200k Method-1 parameter artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
from jax.flatten_util import ravel_pytree
import numpy as np

from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_discrete_training import (
    CompactCheckpointObjective,
    FastFixedDiscreteObjective,
    load_discrete_training_configuration,
    load_fixed_cache,
    load_training_initial_parameters,
)
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import (
    DenseMLP,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters_atomic,
)
from .test2a_pyrol import build_test2a_lbfgs_parameters


OPERATOR_200K_SHA256 = (
    "f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8"
)
DISCRETE_200K_SHA256 = (
    "94bb112961bc2f2e05cbca459bc50d64513110a077e2b15cded39fe8427de6f8"
)
CACHE_SHA256 = (
    "baee2dd3ae8a5e3f9ec16f6883e3583d4ac61281d777c3079b002e611504bacf"
)

IMMUTABLE_REFERENCES = {
    "historical_practical_method1": {
        "J_op": 0.004285912836972889,
        "J_disc": 0.00794193542678781,
    },
    "historical_practical_method2": {
        "J_op": 0.0020819762080123453,
        "J_disc": 0.0017427829635521567,
    },
    "matched_200k_method1": {
        "J_op": 0.000373006108792648,
        "J_disc": 0.0008346864309047664,
        "parameter_pytree_sha256": OPERATOR_200K_SHA256,
    },
    "matched_200k_method2": {
        "J_op": 0.002489117530253537,
        "J_disc": 0.001721966994676836,
        "parameter_pytree_sha256": DISCRETE_200K_SHA256,
    },
}


def gradient_geometry(operator_value, operator_gradient, discrete_value, discrete_gradient):
    """Return scale-aware geometry for two arbitrary float64 pytrees."""

    operator_flat = np.asarray(ravel_pytree(operator_gradient)[0], dtype=np.float64)
    discrete_flat = np.asarray(ravel_pytree(discrete_gradient)[0], dtype=np.float64)
    if operator_flat.shape != discrete_flat.shape:
        raise ValueError("operator and deployed-discrete gradients differ in shape")
    operator_norm = float(np.linalg.norm(operator_flat))
    discrete_norm = float(np.linalg.norm(discrete_flat))
    dot = float(operator_flat @ discrete_flat)
    cosine = (
        None
        if operator_norm == 0.0 or discrete_norm == 0.0
        else dot / (operator_norm * discrete_norm)
    )
    alpha = None if operator_norm == 0.0 else dot / (operator_norm * operator_norm)
    relative_residual = None
    if alpha is not None and discrete_norm > 0.0:
        relative_residual = float(
            np.linalg.norm(discrete_flat - alpha * operator_flat) / discrete_norm
        )
    return {
        "J_op": float(operator_value),
        "J_disc": float(discrete_value),
        "operator_gradient_norm": operator_norm,
        "deployed_discrete_gradient_norm": discrete_norm,
        "gradient_dot_product": dot,
        "gradient_cosine_similarity": cosine,
        "best_scalar_alpha_for_g_disc_minus_alpha_g_op": alpha,
        "relative_orthogonal_component_of_g_disc": relative_residual,
        "relative_nonproportional_gradient_residual": relative_residual,
        "parameter_dimension": int(operator_flat.size),
    }


def _load_context(configuration_path, cache_path):
    training = load_discrete_training_configuration(configuration_path)
    cache = load_fixed_cache(cache_path)
    if cache.metadata["cache_npz_sha256"] != CACHE_SHA256:
        raise ValueError("accepted deployed-discrete cache fingerprint changed")
    compatible = training["compatible_cache_training_configuration_sha256"]
    if cache.metadata["training_configuration_sha256"] != compatible:
        raise ValueError("fine-tune configuration is incompatible with fixed cache")
    selected = load_selected_configuration(training["selected_operator_configuration"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    dataset, dataset_metadata = load_operator_dataset(training["operator_dataset"])
    normalization = normalization_from_record(dataset_metadata["normalization"])
    fast = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    return training, cache, model_configuration, dataset, normalization, fast


def _verified_parameters(path, expected_sha, model_configuration):
    parameters, artifact_configuration = load_mlp_parameters(path)
    if artifact_configuration != model_configuration:
        raise ValueError(f"parameter architecture changed for {path}")
    actual = parameter_pytree_sha256(parameters)
    if actual != expected_sha:
        raise ValueError(f"parameter pytree fingerprint changed for {path}")
    return parameters


def run_gradient_geometry(configuration_path, cache_path, output_path):
    """Evaluate both exact gradients at the matched 200k solutions."""

    training, cache, model_configuration, _, _, fast = _load_context(
        configuration_path, cache_path
    )
    probes = {
        "theta_op_200k": (
            training["initialization"]["source_parameter_file"],
            OPERATOR_200K_SHA256,
        ),
        "theta_disc_200k": (
            training["matched_seed0_method2_reference"]["parameter_file"],
            DISCRETE_200K_SHA256,
        ),
    }
    operator_value_gradient = jax.jit(
        jax.value_and_grad(lambda parameters: fast._objectives(parameters)[1])
    )
    results = {}
    for name, (path, expected_sha) in probes.items():
        parameters = _verified_parameters(path, expected_sha, model_configuration)
        discrete_value, discrete_gradient = fast.value_and_gradient(parameters)
        operator_value, operator_gradient = operator_value_gradient(parameters)
        results[name] = {
            **gradient_geometry(
                operator_value,
                operator_gradient,
                discrete_value,
                discrete_gradient,
            ),
            "parameter_file": str(Path(path).resolve()),
            "parameter_pytree_sha256": expected_sha,
        }
    result = {
        "status": "complete",
        "diagnostic": "M1_TO_M2_GRADIENT_GEOMETRY",
        "probes": results,
        "cache": {
            "path": str(Path(cache_path).resolve()),
            "npz_sha256": cache.metadata["cache_npz_sha256"],
            "production_oracle_certified": bool(
                cache.metadata["production_oracle_certified"]
            ),
        },
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
        },
    }
    write_json_record(output_path, result)
    return result


def run_warm_start_smoke(
    configuration_path, cache_path, output_directory, *, iterations=20
):
    """Run a bounded NONSCIENTIFIC new-history PyROL plumbing smoke."""

    if int(iterations) < 1 or int(iterations) > 20:
        raise ValueError("warm-start smoke requires 1..20 accepted iterations")
    from pyrol import Problem, Solver
    from pyrol.vectors import NumPyVector

    training, cache, model_configuration, _, _, fast = _load_context(
        configuration_path, cache_path
    )
    initial = load_training_initial_parameters(training, model_configuration)
    output_root = Path(output_directory)
    if output_root.exists():
        raise FileExistsError("refusing to overwrite warm-start smoke output")
    output_root.mkdir(parents=True)
    adapter = CompactCheckpointObjective(fast.jax_value, initial, use_jit=True)
    control = adapter.vector_from_pytree(initial)
    initial_value = adapter.value(control, 0.0)
    warm_gradient = NumPyVector(np.full(adapter.dimension, np.nan, dtype=np.float64))
    adapter.gradient(warm_gradient, control, 0.0)
    adapter.reset_accounting()
    optimizer = training["optimizer"]
    rol = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": optimizer["gradient_tolerance"],
            "step_tolerance": optimizer["step_tolerance"],
            "iteration_limit": int(iterations),
            "maximum_secant_storage": optimizer["maximum_secant_storage"],
        }
    )
    started = perf_counter()
    solver = Solver(Problem(adapter, control), rol)
    solver.solve()
    wall = perf_counter() - started
    state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    final_value, final_gradient = fast.value_and_gradient(final_parameters)
    parameter_path = output_root / "smoke_final_parameters.npz"
    save_mlp_parameters_atomic(parameter_path, final_parameters, model_configuration)
    result = {
        "status": "complete",
        "interpretation": "NONSCIENTIFIC M1_TO_M2 FINE_TUNE SMOKE",
        "initialization_kind": "operator_200k_warm_start",
        "initial_parameter_pytree_sha256": parameter_pytree_sha256(initial),
        "source_optimizer_secant_history_reused": False,
        "new_LBFGS_history_started_empty": True,
        "initial_J_disc": float(initial_value),
        "final_J_disc": float(final_value),
        "final_gradient_norm": float(
            np.linalg.norm(np.asarray(ravel_pytree(final_gradient)[0]))
        ),
        "objective_decreased": bool(final_value < initial_value),
        "accepted_iterations": int(state.iter),
        "actual_ROL_termination_reason": str(state.statusFlag),
        "objective_evaluations": int(adapter.value_evaluations),
        "gradient_evaluations": int(adapter.gradient_evaluations),
        "HVP_evaluations": int(adapter.hvp_evaluations),
        "wall_time_seconds": float(wall),
        "final_parameter_file": str(parameter_path.resolve()),
        "final_parameter_pytree_sha256": parameter_pytree_sha256(final_parameters),
        "estimated_50k_wall_seconds_from_smoke_linear": float(
            wall * 50000.0 / max(int(state.iter), 1)
        ),
        "cache_npz_sha256": cache.metadata["cache_npz_sha256"],
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
            "recursive_model_state_propagation": False,
        },
    }
    if not result["objective_decreased"] or result["HVP_evaluations"] != 0:
        raise RuntimeError("warm-start fine-tune smoke failed")
    write_json_record(output_root / "smoke_result.json", result)
    return result


def prepare_checkpoint_postprocessing(
    configuration_path, cache_path, fit_result_path, output_directory
):
    """Evaluate cross-objectives/direct-A metrics and write rollout manifest."""

    training, cache, model_configuration, dataset, normalization, fast = _load_context(
        configuration_path, cache_path
    )
    fit = read_json_record(fit_result_path)
    if fit.get("status") != "complete":
        raise ValueError("fine-tune fit is not complete")
    if fit["initialization"]["parameter_pytree_sha256"] != OPERATOR_200K_SHA256:
        raise ValueError("fine-tune did not begin from matched Method-1 parameters")
    entries = list(fit.get("checkpoint_diagnostics", []))
    final = fit["final_diagnostics"]
    if not any(
        item["parameter_pytree_sha256"] == final["parameter_pytree_sha256"]
        for item in entries
    ):
        entries.append(
            {
                **final,
                "parameter_file": fit["final_parameter_file"],
                "parameter_npz_sha256": None,
            }
        )
    entries.sort(key=lambda item: int(item["accepted_iteration"]))
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    rollout_entries = []
    seen = set()
    model = DenseMLP(model_configuration)
    for entry in entries:
        iteration = int(entry["accepted_iteration"])
        path = entry["parameter_file"]
        expected_sha = entry["parameter_pytree_sha256"]
        parameters = _verified_parameters(path, expected_sha, model_configuration)
        J_disc, J_op = fast.objectives(parameters)
        predictions = physical_predictions(
            parameters, model, normalization, dataset.features
        )
        record = {
            "accepted_iteration": iteration,
            "parameter_file": str(Path(path).resolve()),
            "parameter_pytree_sha256": expected_sha,
            "J_op": J_op,
            "J_disc": J_disc,
            "direct_A_metrics": operator_metrics(predictions, dataset.targets),
        }
        diagnostics.append(record)
        if expected_sha not in seen:
            label = f"iter-{iteration:05d}"
            rollout_entries.append(
                {
                    "label": label,
                    "accepted_iteration": iteration,
                    "parameter_file": str(Path(path).resolve()),
                    "parameter_pytree_sha256": expected_sha,
                    "output_directory": str(
                        (output_root / "autonomous" / label).resolve()
                    ),
                }
            )
            seen.add(expected_sha)
    checkpoint_record = {
        "status": "complete",
        "diagnostic": "M1_TO_M2_CHECKPOINT_CROSS_OBJECTIVES_AND_DIRECT_A",
        "checkpoints": diagnostics,
        "immutable_reference_results": IMMUTABLE_REFERENCES,
        "cache_npz_sha256": cache.metadata["cache_npz_sha256"],
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
        },
    }
    manifest = {
        "status": "ready",
        "configuration": str(Path(configuration_path).resolve()),
        "entries": rollout_entries,
        "deployment_contract": {
            "trusted_initial_state": 0,
            "complete_steps": 80,
            "truth_resets_after_initialization": 0,
            "states_after_80_accessed": False,
            "checkpoint_selection_uses_autonomous_error": False,
        },
    }
    write_json_record(output_root / "checkpoint_metrics.json", checkpoint_record)
    write_json_record(output_root / "autonomous_manifest.json", manifest)
    return checkpoint_record, manifest


def _rollout_compact(record):
    return {
        "mixed_state_relative_error": record["mixed_state_error"],
        "off_manifold_A": record["aggregate_off_manifold_A_diagnostic"],
        "kinetic_energy": record["kinetic_energy"],
        "projected_enstrophy": record["projected_enstrophy"],
        "fieldwise_errors": record["fieldwise_errors"],
        "rain_activity": record["rain_activity_summary"],
        "source_structural_invariants": record["source_structural_invariants"],
        "all_states_finite": record["all_states_finite"],
    }


def write_finetune_report(postprocess_directory, output_json, output_markdown):
    """Combine completed checkpoint metrics and training-support rollouts."""

    root = Path(postprocess_directory)
    checkpoint_record = read_json_record(root / "checkpoint_metrics.json")
    manifest = read_json_record(root / "autonomous_manifest.json")
    rollouts = {}
    for entry in manifest["entries"]:
        summary = read_json_record(Path(entry["output_directory"]) / "rollout_summary.json")
        if summary.get("status") != "complete":
            raise ValueError(f"autonomous result {entry['label']} is incomplete")
        contract = summary["deployment_contract"]
        if contract.get("states_after_80_accessed", True) or contract[
            "truth_states_accessed"
        ] != [0, 80]:
            raise ValueError("autonomous postprocessing accessed held-out truth")
        rollouts[entry["label"]] = _rollout_compact(summary)
    initial = checkpoint_record["checkpoints"][0]
    final = checkpoint_record["checkpoints"][-1]
    result = {
        "status": "complete",
        "diagnostic": "OPERATOR_PRETRAINED_DISCRETE_FINETUNE",
        "checkpoint_metrics": checkpoint_record,
        "autonomous_training_support": rollouts,
        "immutable_reference_results": IMMUTABLE_REFERENCES,
        "objective_change_from_initial": {
            "delta_J_op": final["J_op"] - initial["J_op"],
            "delta_J_disc": final["J_disc"] - initial["J_disc"],
            "Pareto_better_in_both_offline_objectives": bool(
                final["J_op"] < initial["J_op"]
                and final["J_disc"] < initial["J_disc"]
            ),
        },
        "interpretation_contract": {
            "matched_seed0_comparison_replaced": False,
            "sequential_workflow_diagnostic_only": True,
            "heldout_states_accessed": False,
            "autonomous_error_used_for_optimizer_stopping": False,
        },
    }
    write_json_record(output_json, result)
    lines = [
        "# Test 2A M1 to M2 fine-tuning diagnostic",
        "",
        "This is a sequential workflow diagnostic, not a replacement for the matched seed-zero comparison.",
        "",
        "| iteration | J_op | J_disc | relative RMS(A) | correlation |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in checkpoint_record["checkpoints"]:
        metrics = item["direct_A_metrics"]
        lines.append(
            f"| {item['accepted_iteration']} | {item['J_op']:.12g} | "
            f"{item['J_disc']:.12g} | {metrics['relative_rms_error']:.12g} | "
            f"{metrics['correlation']:.12g} |"
        )
    lines.extend(
        [
            "",
            "All autonomous evaluations use only truth states 0..80 and do not influence optimization stopping.",
            "Method-1 secant history is not transferred into the Method-2 optimizer.",
        ]
    )
    Path(output_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    geometry = commands.add_parser("gradient-geometry")
    geometry.add_argument("--configuration", required=True)
    geometry.add_argument("--cache", required=True)
    geometry.add_argument("--output", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--configuration", required=True)
    smoke.add_argument("--cache", required=True)
    smoke.add_argument("--output-directory", required=True)
    smoke.add_argument("--iterations", type=int, default=20)
    postprocess = commands.add_parser("prepare-postprocess")
    postprocess.add_argument("--configuration", required=True)
    postprocess.add_argument("--cache", required=True)
    postprocess.add_argument("--fit-result", required=True)
    postprocess.add_argument("--output-directory", required=True)
    report = commands.add_parser("report")
    report.add_argument("--postprocess-directory", required=True)
    report.add_argument("--output-json", required=True)
    report.add_argument("--output-markdown", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "gradient-geometry":
        run_gradient_geometry(
            arguments.configuration, arguments.cache, arguments.output
        )
    elif arguments.command == "smoke":
        run_warm_start_smoke(
            arguments.configuration,
            arguments.cache,
            arguments.output_directory,
            iterations=arguments.iterations,
        )
    elif arguments.command == "prepare-postprocess":
        prepare_checkpoint_postprocessing(
            arguments.configuration,
            arguments.cache,
            arguments.fit_result,
            arguments.output_directory,
        )
    elif arguments.command == "report":
        write_finetune_report(
            arguments.postprocess_directory,
            arguments.output_json,
            arguments.output_markdown,
        )
    else:
        raise AssertionError("unreachable M1-to-M2 command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CACHE_SHA256",
    "DISCRETE_200K_SHA256",
    "IMMUTABLE_REFERENCES",
    "OPERATOR_200K_SHA256",
    "gradient_geometry",
    "prepare_checkpoint_postprocessing",
    "run_gradient_geometry",
    "run_warm_start_smoke",
    "write_finetune_report",
)
