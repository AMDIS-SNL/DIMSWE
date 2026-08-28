"""Matched seed-0 Method-1/Method-2 long-fit operations and reporting."""

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
from .test2a_discrete_training import (
    FastFixedDiscreteObjective,
    load_fixed_cache,
)
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import (
    DenseMLP,
    initialize_mlp,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    normalized_operator_objective,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters_atomic,
)
from .test2a_pyrol import JAXPytreeObjective, build_test2a_lbfgs_parameters


CANONICAL_SEED_SHA256 = (
    "6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975"
)


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_sha256(record):
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def load_fair_operator_configuration(path):
    record = read_json_record(path)
    if record.get("benchmark_stage") != "Test 2A fair operator seed-0 m20 long fit":
        raise ValueError("not a fair Method-1 long-fit configuration")
    if record["truth_state_indices"] != [0, 80] or not record[
        "states_after_80_forbidden"
    ]:
        raise ValueError("fair Method-1 fit may use only states 0..80")
    optimizer = record["optimizer"]
    if (
        optimizer["library"] != "PyROL/ROL"
        or optimizer["method"] != "line-search L-BFGS"
        or int(optimizer["maximum_secant_storage"]) != 20
        or float(optimizer["gradient_tolerance"]) != 1.0e-8
        or float(optimizer["step_tolerance"]) != 1.0e-12
        or int(optimizer["accepted_iteration_limit"]) != 200000
        or optimizer["production_HVP"] is not False
    ):
        raise ValueError("fair Method-1 optimizer contract changed")
    checkpoints = tuple(int(value) for value in record["checkpoint_accepted_iterations"])
    if checkpoints != (25000, 50000, 75000, 100000, 150000, 200000):
        raise ValueError("fair Method-1 checkpoint schedule changed")
    if record["initialization"]["parameter_pytree_sha256"] != CANONICAL_SEED_SHA256:
        raise ValueError("fair Method-1 canonical seed changed")
    return record


class LongRunJAXObjective(JAXPytreeObjective):
    """Bounded-memory ROL callback for long full-batch fits."""

    def __init__(self, *args, accepted_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted_callback = accepted_callback
        self.accepted_update_count = 0
        self.previous_accepted_control = None
        self.current_accepted_control = None

    def _pending_record(self, flat):
        del flat
        return None

    def update(self, control, *args):
        update_type = str(args[0]) if args else "unspecified"
        if "Initial" not in update_type and "Accept" not in update_type:
            return
        values = np.asarray(
            self._flat_from_vector(control, "control"), dtype=np.float64
        ).copy()
        self.previous_accepted_control = self.current_accepted_control
        self.current_accepted_control = values
        local_index = self.accepted_update_count
        self.accepted_update_count += 1
        if self.accepted_callback is not None:
            self.accepted_callback(control, local_index, self)

    def hessVec(self, output, direction, control, tolerance):
        del output, direction, control, tolerance
        self.hvp_evaluations += 1
        raise RuntimeError("fair Method-1 L-BFGS must not request an HVP")


def train_operator_long(configuration_path, output_directory, *, resume=False):
    configuration = load_fair_operator_configuration(configuration_path)
    config_sha = _record_sha256(configuration)
    selected = load_selected_configuration(configuration["selected_configuration"])
    dataset, metadata = load_operator_dataset(configuration["operator_dataset"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    normalization = normalization_from_record(metadata["normalization"])
    model = DenseMLP(model_configuration)
    features = jnp.asarray(
        normalization.normalize_features(dataset.features), dtype=jnp.float64
    )
    targets = jnp.asarray(
        normalization.normalize_a(dataset.targets), dtype=jnp.float64
    ).reshape(-1, 1)

    def objective(parameters):
        return normalized_operator_objective(parameters, model, features, targets)

    initial = initialize_mlp(model_configuration)
    if parameter_pytree_sha256(initial) != CANONICAL_SEED_SHA256:
        raise ValueError("canonical seed-0 initialization fingerprint changed")
    output_root = Path(output_directory)
    result_path = output_root / "fit_result.json"
    progress_path = output_root / "fit_progress.json"
    if result_path.exists():
        raise FileExistsError("refusing to overwrite completed fair Method-1 fit")
    output_root.mkdir(parents=True, exist_ok=True)
    offset = 0
    start_parameters = initial
    previous_checkpoints = []
    cumulative = {"objective": 0, "gradient": 0, "HVP": 0, "wall_seconds": 0.0}
    secant_restored = True
    if resume:
        if not progress_path.exists():
            raise FileNotFoundError("no fair Method-1 checkpoint to resume")
        progress = read_json_record(progress_path)
        if progress["configuration_sha256"] != config_sha:
            raise ValueError("Method-1 resume configuration changed")
        offset = int(progress["last_checkpoint_accepted_iteration"])
        start_parameters, resumed_configuration = load_mlp_parameters(
            progress["last_checkpoint_parameter_file"]
        )
        if resumed_configuration != model_configuration:
            raise ValueError("Method-1 resume architecture changed")
        if parameter_pytree_sha256(start_parameters) != progress[
            "last_checkpoint_parameter_pytree_sha256"
        ]:
            raise ValueError("Method-1 resume parameter fingerprint changed")
        previous_checkpoints = list(progress.get("checkpoint_diagnostics", []))
        cumulative = dict(progress.get("cumulative_accounting", cumulative))
        secant_restored = False
    elif progress_path.exists():
        raise FileExistsError("incomplete Method-1 fit exists; use --resume manually")
    total_limit = int(configuration["optimizer"]["accepted_iteration_limit"])
    remaining = total_limit - offset
    if remaining <= 0:
        raise ValueError("Method-1 checkpoint already reached the configured cap")
    checkpoint_set = set(configuration["checkpoint_accepted_iterations"])
    checkpoints = {int(value["accepted_iteration"]): value for value in previous_checkpoints}
    progress_stride = int(configuration["progress_accepted_iteration_stride"])
    run_started = None

    def monitor(parameters, iteration):
        value, gradient = jax.value_and_grad(objective)(parameters)
        prediction = physical_predictions(
            parameters, model, normalization, dataset.features
        )
        return {
            "accepted_iteration": int(iteration),
            "J_op": float(value),
            "gradient_norm_J_op": float(
                np.sqrt(
                    sum(
                        np.vdot(np.asarray(leaf), np.asarray(leaf)).real
                        for leaf in jax.tree_util.tree_leaves(gradient)
                    )
                )
            ),
            "physical_A_metrics": operator_metrics(prediction, dataset.targets),
            "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
        }

    def accepted_callback(control, local_index, adapter):
        if local_index == 0:
            return
        iteration = offset + local_index
        is_checkpoint = iteration in checkpoint_set
        is_progress = iteration % progress_stride == 0
        if not is_checkpoint and not is_progress:
            return
        parameters = adapter.pytree_from_vector(control)
        record = monitor(parameters, iteration)
        elapsed = 0.0 if run_started is None else perf_counter() - run_started
        print(
            json.dumps(
                {
                    "event": "progress",
                    "method": "operator",
                    "accepted_iteration": iteration,
                    "J_op": record["J_op"],
                    "gradient_norm": record["gradient_norm_J_op"],
                    "elapsed_wall_seconds_this_process": elapsed,
                    "objective_evaluations_this_process": adapter.value_evaluations,
                    "gradient_evaluations_this_process": adapter.gradient_evaluations,
                    "parameter_pytree_sha256": record["parameter_pytree_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not is_checkpoint:
            return
        parameter_path = output_root / f"parameters_iter_{iteration:06d}.npz"
        save_mlp_parameters_atomic(parameter_path, parameters, model_configuration)
        record.update(
            {
                "parameter_file": str(parameter_path.resolve()),
                "parameter_npz_sha256": _file_sha256(parameter_path),
                "parameter_step_norm_relative_to_parameter_norm": (
                    None
                    if adapter.previous_accepted_control is None
                    else float(
                        np.linalg.norm(
                            adapter.current_accepted_control
                            - adapter.previous_accepted_control
                        )
                        / max(
                            np.linalg.norm(adapter.current_accepted_control),
                            np.finfo(np.float64).tiny,
                        )
                    )
                ),
            }
        )
        checkpoints[iteration] = record
        progress = {
            "status": "in_progress",
            "configuration_sha256": config_sha,
            "last_checkpoint_accepted_iteration": iteration,
            "last_checkpoint_parameter_file": str(parameter_path.resolve()),
            "last_checkpoint_parameter_npz_sha256": record["parameter_npz_sha256"],
            "last_checkpoint_parameter_pytree_sha256": record[
                "parameter_pytree_sha256"
            ],
            "checkpoint_diagnostics": [checkpoints[key] for key in sorted(checkpoints)],
            "cumulative_accounting": {
                "objective": int(cumulative["objective"]) + adapter.value_evaluations,
                "gradient": int(cumulative["gradient"]) + adapter.gradient_evaluations,
                "HVP": int(cumulative["HVP"]) + adapter.hvp_evaluations,
                "wall_seconds": float(cumulative["wall_seconds"]) + elapsed,
            },
            "resume_contract": configuration["resume_contract"],
        }
        write_json_record(progress_path, progress)
        write_json_record(output_root / f"checkpoint_iter_{iteration:06d}.json", record)

    adapter = LongRunJAXObjective(
        objective,
        start_parameters,
        use_jit=True,
        accepted_callback=accepted_callback,
    )
    control = adapter.vector_from_pytree(start_parameters)
    adapter.value(control, 0.0)
    warm_gradient = NumPyVector(np.full(adapter.dimension, np.nan, dtype=np.float64))
    adapter.gradient(warm_gradient, control, 0.0)
    adapter.reset_accounting()
    adapter.accepted_update_count = 0
    adapter.previous_accepted_control = None
    adapter.current_accepted_control = None
    optimizer = configuration["optimizer"]
    rol_parameters = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": optimizer["gradient_tolerance"],
            "step_tolerance": optimizer["step_tolerance"],
            "iteration_limit": remaining,
            "maximum_secant_storage": optimizer["maximum_secant_storage"],
        }
    )
    solver = Solver(Problem(adapter, control), rol_parameters)
    print(
        json.dumps(
            {
                "event": "fair_operator_fit_start",
                "accepted_iteration_offset": offset,
                "accepted_iteration_limit": total_limit,
                "remaining_iteration_budget": remaining,
                "initial_parameter_pytree_sha256": parameter_pytree_sha256(
                    start_parameters
                ),
                "secant_history_restored": secant_restored,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run_started = perf_counter()
    solver.solve()
    wall = perf_counter() - run_started
    state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    final_iteration = offset + int(state.iter)
    final_path = output_root / "final_parameters.npz"
    save_mlp_parameters_atomic(final_path, final_parameters, model_configuration)
    final_record = monitor(final_parameters, final_iteration)
    counts = {
        "objective": int(cumulative["objective"]) + adapter.value_evaluations,
        "gradient": int(cumulative["gradient"]) + adapter.gradient_evaluations,
        "HVP": int(cumulative["HVP"]) + adapter.hvp_evaluations,
    }
    if counts["HVP"] != 0:
        raise RuntimeError("fair Method-1 L-BFGS unexpectedly requested HVP")
    result = {
        "status": "complete",
        "benchmark_stage": configuration["benchmark_stage"],
        "configuration_sha256": config_sha,
        "initialization": {
            **configuration["initialization"],
            "verified": True,
        },
        "optimizer": {
            **optimizer,
            "accepted_iterations": final_iteration,
            "actual_ROL_termination_reason": str(state.statusFlag),
            "objective_evaluations": counts["objective"],
            "gradient_evaluations": counts["gradient"],
            "HVP_evaluations": counts["HVP"],
            "wall_time_seconds": float(cumulative["wall_seconds"]) + wall,
            "secant_history_restored_on_resume": secant_restored,
        },
        "final_diagnostics": final_record,
        "checkpoint_diagnostics": [checkpoints[key] for key in sorted(checkpoints)],
        "final_parameter_file": str(final_path.resolve()),
        "final_parameter_npz_sha256": _file_sha256(final_path),
        "final_parameter_pytree_sha256": parameter_pytree_sha256(final_parameters),
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
        },
        "resume_disclosure": (
            "parameter checkpoints are exact; a new process does not restore "
            "process-local ROL L-BFGS secant history"
        ),
    }
    write_json_record(result_path, result)
    write_json_record(
        progress_path,
        {
            "status": "complete",
            "configuration_sha256": config_sha,
            "last_checkpoint_accepted_iteration": final_iteration,
            "last_checkpoint_parameter_file": str(final_path.resolve()),
            "last_checkpoint_parameter_npz_sha256": result[
                "final_parameter_npz_sha256"
            ],
            "last_checkpoint_parameter_pytree_sha256": result[
                "final_parameter_pytree_sha256"
            ],
            "checkpoint_diagnostics": result["checkpoint_diagnostics"],
            "cumulative_accounting": {
                **counts,
                "wall_seconds": result["optimizer"]["wall_time_seconds"],
            },
            "resume_contract": configuration["resume_contract"],
        },
    )
    return result


def cross_objective_postprocess(
    operator_result_path,
    discrete_result_path,
    cache_path,
    selected_configuration,
    dataset_path,
    output_path,
):
    operator_result = read_json_record(operator_result_path)
    discrete_result = read_json_record(discrete_result_path)
    for name, result in (("operator", operator_result), ("discrete", discrete_result)):
        if result.get("status") != "complete":
            raise ValueError(f"{name} fit is not complete")
    selected = load_selected_configuration(selected_configuration)
    dataset, metadata = load_operator_dataset(dataset_path)
    normalization = normalization_from_record(metadata["normalization"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    model = DenseMLP(model_configuration)
    cache = load_fixed_cache(cache_path)
    if not cache.metadata.get("production_oracle_certified", False):
        raise ValueError("Method-2 fixed cache is not production-oracle certified")
    fast = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    rows = {}
    for name, result in (("theta_op_long", operator_result), ("theta_disc_long", discrete_result)):
        parameters, configuration = load_mlp_parameters(result["final_parameter_file"])
        if configuration != model_configuration:
            raise ValueError(f"{name} architecture mismatch")
        expected = result["final_parameter_pytree_sha256"]
        if parameter_pytree_sha256(parameters) != expected:
            raise ValueError(f"{name} parameter fingerprint mismatch")
        discrete, operator = fast.objectives(parameters)
        predictions = physical_predictions(
            parameters, model, normalization, dataset.features
        )
        rows[name] = {
            "J_op": operator,
            "J_disc": discrete,
            "direct_A_metrics": operator_metrics(predictions, dataset.targets),
            "parameter_file": result["final_parameter_file"],
            "parameter_pytree_sha256": expected,
        }
    result = {
        "status": "complete",
        "comparison": "matched seed-0 m20 200k-cap fair long fits",
        "cross_objective_table": rows,
        "historical_practical_fits": {
            "theta_op": {
                "J_op": 0.004285912836972889,
                "J_disc": 0.00794193542678781,
                "parameter_pytree_sha256": "f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56",
            },
            "theta_disc": {
                "J_op": 0.0020819762080123453,
                "J_disc": 0.0017427829635521567,
                "parameter_pytree_sha256": "4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a",
            },
        },
        "cache": {
            "path": str(Path(cache_path).resolve()),
            "npz_sha256": cache.metadata["cache_npz_sha256"],
            "production_oracle_certified": True,
        },
        "truth_state_access": {"state_indices": [0, 80], "states_after_80_accessed": False},
    }
    write_json_record(output_path, result)
    return result


def run_fairness_smoke(
    operator_configuration,
    discrete_configuration,
    cache_path,
    output_directory,
    *,
    iterations=20,
):
    """Run matched NONSCIENTIFIC short optimizer plumbing checks."""
    if int(iterations) < 1 or int(iterations) > 20:
        raise ValueError("fairness smoke requires 1..20 accepted iterations")
    operator_record = load_fair_operator_configuration(operator_configuration)
    from .test2a_discrete_training import load_discrete_training_configuration

    discrete_record = load_discrete_training_configuration(discrete_configuration)
    if operator_record["optimizer"] != discrete_record["optimizer"]:
        raise ValueError("fair Method-1/Method-2 optimizer policies differ")
    selected = load_selected_configuration(operator_record["selected_configuration"])
    dataset, metadata = load_operator_dataset(operator_record["operator_dataset"])
    normalization = normalization_from_record(metadata["normalization"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    model = DenseMLP(model_configuration)
    initial = initialize_mlp(model_configuration)
    initial_sha = parameter_pytree_sha256(initial)
    if initial_sha != CANONICAL_SEED_SHA256:
        raise ValueError("fairness smoke seed fingerprint changed")
    features = jnp.asarray(
        normalization.normalize_features(dataset.features), dtype=jnp.float64
    )
    targets = jnp.asarray(
        normalization.normalize_a(dataset.targets), dtype=jnp.float64
    ).reshape(-1, 1)
    operator_objective = lambda parameters: normalized_operator_objective(
        parameters, model, features, targets
    )
    cache = load_fixed_cache(cache_path)
    fast = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    objective_specs = (
        ("operator", operator_objective, 0.9135568693989472),
        ("deployed_discrete", fast.jax_value, 1.2027413730332317),
    )
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError("refusing to overwrite fairness smoke output")
    destination.mkdir(parents=True)
    results = {}
    for name, objective, expected_initial in objective_specs:
        adapter = LongRunJAXObjective(objective, initial, use_jit=True)
        control = adapter.vector_from_pytree(initial)
        initial_value = adapter.value(control, 0.0)
        warm_gradient = NumPyVector(
            np.full(adapter.dimension, np.nan, dtype=np.float64)
        )
        adapter.gradient(warm_gradient, control, 0.0)
        np.testing.assert_allclose(
            initial_value, expected_initial, rtol=2.0e-13, atol=2.0e-13
        )
        adapter.reset_accounting()
        parameters = operator_record["optimizer"]
        rol = build_test2a_lbfgs_parameters(
            {
                "gradient_tolerance": parameters["gradient_tolerance"],
                "step_tolerance": parameters["step_tolerance"],
                "iteration_limit": int(iterations),
                "maximum_secant_storage": parameters["maximum_secant_storage"],
            }
        )
        started = perf_counter()
        solver = Solver(Problem(adapter, control), rol)
        solver.solve()
        wall = perf_counter() - started
        state = solver.getAlgorithmState()
        final_parameters = adapter.pytree_from_vector(control)
        final_value = float(objective(final_parameters))
        checkpoint = destination / f"{name}_parameters.npz"
        save_mlp_parameters_atomic(checkpoint, final_parameters, model_configuration)
        results[name] = {
            "interpretation": "NONSCIENTIFIC IMPLEMENTATION SMOKE",
            "initial_parameter_pytree_sha256": initial_sha,
            "initial_objective": initial_value,
            "final_objective": final_value,
            "objective_decreased": final_value < initial_value,
            "accepted_iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": adapter.value_evaluations,
            "gradient_evaluations": adapter.gradient_evaluations,
            "HVP_evaluations": adapter.hvp_evaluations,
            "wall_time_seconds": wall,
            "checkpoint_file": str(checkpoint.resolve()),
            "checkpoint_parameter_pytree_sha256": parameter_pytree_sha256(
                final_parameters
            ),
        }
        if not results[name]["objective_decreased"] or adapter.hvp_evaluations != 0:
            raise RuntimeError(f"{name} fairness smoke failed")
    result = {
        "status": "complete",
        "interpretation": "NONSCIENTIFIC IMPLEMENTATION SMOKE",
        "matched_optimizer": operator_record["optimizer"],
        "primary_cap": 200000,
        "smoke_iteration_cap": int(iterations),
        "canonical_seed_pytree_sha256": initial_sha,
        "methods": results,
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
        },
    }
    write_json_record(destination / "fairness_smoke.json", result)
    return result


def write_final_comparison_report(
    cross_path,
    operator_rollout_path,
    discrete_rollout_path,
    output_json,
    output_markdown,
):
    cross = read_json_record(cross_path)
    rollouts = {
        "theta_op_long": read_json_record(operator_rollout_path),
        "theta_disc_long": read_json_record(discrete_rollout_path),
    }
    for name, rollout in rollouts.items():
        if rollout.get("status") != "complete":
            raise ValueError(f"{name} autonomous evaluation is not complete")
        if rollout["deployment_contract"].get("states_after_80_accessed", True):
            raise ValueError(f"{name} autonomous evaluation accessed held-out truth")
    summary = {}
    for name, rollout in rollouts.items():
        summary[name] = {
            "final_mixed_state_relative_error": rollout["mixed_state_error"]["final"],
            "maximum_mixed_state_relative_error": rollout["mixed_state_error"]["maximum"],
            "accumulated_mixed_state_relative_error": rollout["mixed_state_error"]["accumulated"],
            "aggregate_off_manifold_A": rollout["aggregate_off_manifold_A_diagnostic"],
            "kinetic_energy": rollout["kinetic_energy"],
            "projected_enstrophy": rollout["projected_enstrophy"],
            "fieldwise_errors": rollout["fieldwise_errors"],
            "rain_activity": rollout["rain_activity_summary"],
            "source_structural_invariants": rollout["source_structural_invariants"],
            "parameter_provenance": rollout["parameter_provenance"],
        }
    result = {
        "status": "complete",
        "cross_objectives_and_direct_A": cross,
        "autonomous_training_support": summary,
        "interpretation_contract": {
            "matched_long_fits": True,
            "training_support_only": True,
            "heldout_states_accessed": False,
            "method2_superiority_not_assumed": True,
        },
    }
    write_json_record(output_json, result)
    lines = [
        "# Test 2A matched Method-1/Method-2 long-fit comparison",
        "",
        "Both fits begin from the canonical seed-0 pytree and use matched PyROL/ROL line-search L-BFGS settings. This is training-support evidence only.",
        "",
        "| network | J_op | J_disc | final mixed error | max mixed error | accumulated mixed error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("theta_op_long", "theta_disc_long"):
        row = cross["cross_objective_table"][name]
        deployment = summary[name]
        lines.append(
            f"| {name} | {row['J_op']:.12g} | {row['J_disc']:.12g} | "
            f"{deployment['final_mixed_state_relative_error']:.12g} | "
            f"{deployment['maximum_mixed_state_relative_error']:.12g} | "
            f"{deployment['accumulated_mixed_state_relative_error']:.12g} |"
        )
    lines.extend(
        [
            "",
            "Historical practical fits remain separate comparison evidence in the JSON report.",
            "No truth state after index 80 was accessed.",
        ]
    )
    Path(output_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-operator")
    train.add_argument("--configuration", required=True)
    train.add_argument("--output-directory", required=True)
    train.add_argument("--resume", action="store_true")
    cross = commands.add_parser("cross-objectives")
    cross.add_argument("--operator-result", required=True)
    cross.add_argument("--discrete-result", required=True)
    cross.add_argument("--cache", required=True)
    cross.add_argument("--selected-configuration", required=True)
    cross.add_argument("--dataset", required=True)
    cross.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("--cross-objectives", required=True)
    report.add_argument("--operator-rollout", required=True)
    report.add_argument("--discrete-rollout", required=True)
    report.add_argument("--output-json", required=True)
    report.add_argument("--output-markdown", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--operator-configuration", required=True)
    smoke.add_argument("--discrete-configuration", required=True)
    smoke.add_argument("--cache", required=True)
    smoke.add_argument("--output-directory", required=True)
    smoke.add_argument("--iterations", type=int, default=20)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "train-operator":
        train_operator_long(args.configuration, args.output_directory, resume=args.resume)
    elif args.command == "cross-objectives":
        cross_objective_postprocess(
            args.operator_result,
            args.discrete_result,
            args.cache,
            args.selected_configuration,
            args.dataset,
            args.output,
        )
    elif args.command == "report":
        write_final_comparison_report(
            args.cross_objectives,
            args.operator_rollout,
            args.discrete_rollout,
            args.output_json,
            args.output_markdown,
        )
    elif args.command == "smoke":
        run_fairness_smoke(
            args.operator_configuration,
            args.discrete_configuration,
            args.cache,
            args.output_directory,
            iterations=args.iterations,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
