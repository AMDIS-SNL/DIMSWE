"""Short external/local certification for shared Test-2A trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from pyrol import Problem, Solver

from .hidden_c0 import _copy_function, _flat_values
from .learned_physics.parameters import tree_axpy, tree_dot, tree_norm, tree_zeros
from .resolved_hidden_c0 import ResolvedPilotConfiguration, write_json_record
from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
from .resolved_hidden_c0_inference import load_resolved_truth
from .selected_test1b import load_selected_test1b_plan
from .test2a_apriori_autonomous import load_compatible_neural_physics
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import load_mlp_parameters, save_mlp_parameters_atomic
from .test2a_pyrol import build_test2a_lbfgs_parameters
from .test2a_trajectory import (
    NeuralTrajectoryObjective,
    TrajectoryLossMode,
    TrajectoryPyROLObjective,
    continuous_rollout,
    reset_windows,
)


def load_trajectory_preparation_configuration(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("benchmark_stage") != (
        "Test 2A Method-3/Method-4 shared trajectory preparation"
    ):
        raise ValueError("not a Test-2A trajectory preparation configuration")
    if record["truth"]["allowed_state_indices"] != [0, 80] or not record[
        "truth"
    ]["states_after_80_forbidden"]:
        raise ValueError("trajectory preparation may access only states 0..80")
    if record["performance_horizons"] != [1, 2, 5, 10]:
        raise ValueError("required trajectory performance horizons changed")
    if record["loss"]["production_metric_scientifically_frozen"] is not False:
        raise ValueError("Method-3/4 production metric must remain unfrozen")
    return record


def _build_case(configuration_path, *, maximum_truth_step=10):
    selected = load_trajectory_preparation_configuration(configuration_path)
    _, plan = load_selected_test1b_plan(selected["truth"]["selected_plan"])
    inference = plan.inference_configuration(
        Path(selected["truth"]["run_directory"]).resolve()
    )
    _, loaded = load_resolved_truth(inference, include_heldout=False)
    if tuple(loaded.states) != tuple(range(81)):
        raise ValueError("trajectory certification requires truth states 0..80 only")
    if int(maximum_truth_step) > 80:
        raise ValueError("trajectory certification cannot access a state after 80")
    model_record = selected["model"]
    physics = load_compatible_neural_physics(
        model_record["embedding_configuration"],
        model_record["certification_parameter_file"],
    )
    parameters, model_configuration = load_mlp_parameters(
        model_record["certification_parameter_file"]
    )
    if parameter_pytree_sha256(parameters) != parameter_pytree_sha256(
        physics.parameters
    ):
        raise ValueError("certification parameter artifact changed while loading")
    pilot = ResolvedPilotConfiguration.from_dict(loaded.metadata["configuration"])
    neural_pilot = replace(
        pilot,
        moist_backend="jax",
        output_directory="/tmp/test2a-trajectory-certification-no-output",
    )
    case = build_resolved_hidden_c0_case(
        neural_pilot, jax_moist_local_physics=physics
    )
    truth = {
        step: case.state_from_values(
            _flat_values(loaded.states[step]), f"test2a_trajectory_truth_{step}"
        )
        for step in range(int(maximum_truth_step) + 1)
    }
    return selected, case, truth, parameters, model_configuration


def _fieldwise_maximum_difference(left, right):
    result = {}
    for name, actual, expected in zip(
        ("v", "h", "S", "Qv", "Qc", "Qr"),
        left.subfunctions,
        right.subfunctions,
    ):
        result[name] = float(
            np.max(
                np.abs(
                    np.asarray(actual.dat.data_ro, dtype=np.float64)
                    - np.asarray(expected.dat.data_ro, dtype=np.float64)
                )
            )
        )
    return result


def _tree_comparison(actual, expected):
    absolute = float(tree_norm(jax.tree.map(lambda x, y: x - y, actual, expected)))
    reference = float(tree_norm(expected))
    dot = float(tree_dot(actual, expected))
    actual_norm = float(tree_norm(actual))
    return {
        "absolute_error": absolute,
        "relative_error": absolute / max(reference, np.finfo(np.float64).tiny),
        "cosine": dot
        / max(actual_norm * reference, np.finfo(np.float64).tiny),
    }


def _deterministic_direction(parameters):
    direction = jax.tree.map(
        lambda leaf: jnp.linspace(-0.5, 0.5, leaf.size, dtype=jnp.float64).reshape(
            leaf.shape
        ),
        parameters,
    )
    return jax.tree.map(lambda leaf: leaf / tree_norm(direction), direction)


def certify_trajectory_framework(configuration_path, output_path):
    selected, case, truth, parameters, _ = _build_case(
        configuration_path, maximum_truth_step=10
    )
    helper = case.helper
    direction = _deterministic_direction(parameters)
    result = {
        "status": "in_progress",
        "benchmark_stage": selected["benchmark_stage"],
        "interpretation": "short derivative/performance certification; not Method-3/4 training",
        "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
        "truth_state_access": {
            "loaded_state_indices": [0, 80],
            "numerically_materialized_state_indices": [0, 10],
            "states_after_80_accessed": False,
        },
    }

    # H=2 primal parity against ordinary repeated complete steps.
    objective = NeuralTrajectoryObjective(
        case,
        truth,
        continuous_rollout(2, "accumulated", (0.5, 0.5)),
        c0=0.14,
        use_fixed_prefix=True,
    )
    tape = objective._tape(parameters)
    manual = [_copy_function(truth[0], "test2a_manual_0")]
    with case.physical_c0(0.14):
        for step in range(2):
            cache = helper.take_forward_step_cached(
                manual[-1],
                case.t0 + step * case.dt,
                case.dt,
                neural_parameters=parameters,
            )
            manual.append(_copy_function(cache.state_out, f"test2a_manual_{step + 1}"))
    result["primal_parity"] = {
        f"step_{step}": _fieldwise_maximum_difference(
            tape.windows[0].states[step], manual[step]
        )
        for step in (1, 2)
    }

    no_prefix = NeuralTrajectoryObjective(
        case,
        truth,
        continuous_rollout(2, "accumulated", (0.5, 0.5)),
        c0=0.14,
        use_fixed_prefix=False,
    )
    fixed_value, fixed_gradient = objective.value_and_gradient(parameters)
    ordinary_value, ordinary_gradient = no_prefix.value_and_gradient(parameters)
    result["fixed_prefix_parity"] = {
        "objective_absolute_difference": abs(fixed_value - ordinary_value),
        "gradient": _tree_comparison(fixed_gradient, ordinary_gradient),
        "state_step_1_fieldwise_maximum_difference": _fieldwise_maximum_difference(
            objective._last_tape.windows[0].states[1],
            no_prefix._last_tape.windows[0].states[1],
        ),
        "cached_child_order": list(
            objective._prefixes[0].forward_child_order
        ),
    }

    # Full 1281-vector adjoint gradients and centered directional checks.
    derivative_records = {}
    for horizon in (1, 2):
        local = NeuralTrajectoryObjective(
            case,
            truth,
            continuous_rollout(
                horizon, "accumulated", tuple(1.0 / horizon for _ in range(horizon))
            ),
            c0=0.14,
            use_fixed_prefix=True,
        )
        value, gradient = local.value_and_gradient(parameters)
        epsilon = 2.0e-5
        plus = local.value(tree_axpy(parameters, epsilon, direction))
        minus = local.value(tree_axpy(parameters, -epsilon, direction))
        centered = (plus - minus) / (2.0 * epsilon)
        adjoint = float(tree_dot(gradient, direction))
        derivative_records[f"H{horizon}"] = {
            "objective": value,
            "gradient_norm": float(tree_norm(gradient)),
            "gradient_parameter_count": sum(
                int(leaf.size) for leaf in jax.tree_util.tree_leaves(gradient)
            ),
            "directional_centered_fd": centered,
            "directional_adjoint": adjoint,
            "absolute_discrepancy": abs(centered - adjoint),
            "scale_aware_relative_discrepancy": abs(centered - adjoint)
            / max(abs(centered), abs(adjoint), np.finfo(np.float64).tiny),
        }
    result["parameter_gradient_certification"] = derivative_records

    # State tangent/adjoint consistency across two recursive steps.
    window = tape.windows[0]
    state_direction = _copy_function(truth[1], "test2a_state_direction")
    with state_direction.dat.vec as vector:
        vector.scale(1.0e-6)
    current_direction = state_direction
    zero_parameters = tree_zeros(parameters)
    for cache in window.step_caches:
        tangent = helper.take_neural_parameter_tangent_step(
            cache, current_direction, zero_parameters
        )
        current_direction = tangent.state_direction_out
    probe_state = _copy_function(truth[2], "test2a_state_adjoint_probe")
    with probe_state.dat.vec as vector:
        vector.scale(1.0e-6)
    probe = helper.state_mass_map(probe_state, "test2a_state_adjoint_probe_dual")
    current_adjoint = probe
    for cache in reversed(window.step_caches):
        reverse = helper.take_neural_parameter_adjoint_step(
            cache, current_adjoint, stop_at_fixed_prefix=False
        )
        current_adjoint = reverse.state_adjoint_in
    left = helper.dual_pairing(probe, current_direction)
    right = helper.dual_pairing(current_adjoint, state_direction)
    result["state_tangent_adjoint"] = {
        "H": 2,
        "tangent_pairing": left,
        "adjoint_pairing": right,
        "absolute_discrepancy": abs(left - right),
        "relative_discrepancy": abs(left - right)
        / max(abs(left), abs(right), np.finfo(np.float64).tiny),
    }

    # Same-theta tape hit and strict changed-theta invalidation.
    cache_test = NeuralTrajectoryObjective(
        case,
        truth,
        continuous_rollout(2, "endpoint", (1.0,)),
        c0=0.14,
        use_fixed_prefix=True,
    )
    cached_value = cache_test.value(parameters)
    cached_value_gradient, cached_gradient = cache_test.value_and_gradient(parameters)
    hit_counts = cache_test.work_counts()
    perturbed = tree_axpy(parameters, 1.0e-8, direction)
    cache_test.value(perturbed)
    invalidated_counts = cache_test.work_counts()
    clean = NeuralTrajectoryObjective(
        case,
        truth,
        continuous_rollout(2, "endpoint", (1.0,)),
        c0=0.14,
        use_fixed_prefix=True,
    )
    _, clean_gradient = clean.value_and_gradient(parameters)
    result["same_theta_tape"] = {
        "value_bitwise_equal": cached_value == cached_value_gradient,
        "cached_vs_clean_gradient": _tree_comparison(cached_gradient, clean_gradient),
        "same_theta_hits_after_value_then_gradient": hit_counts.same_theta_tape_hits,
        "forward_steps_after_value_then_gradient": hit_counts.forward_complete_steps,
        "changed_theta_invalidations": invalidated_counts.tape_invalidations,
        "changed_theta_forward_steps": invalidated_counts.forward_complete_steps,
        "cache_key_parameter_fingerprint": "exact float64 pytree SHA256",
    }

    # Recursive H=2 gradient must differ from independently truth-reset H=1 terms.
    full = NeuralTrajectoryObjective(
        case,
        truth,
        continuous_rollout(2, "accumulated", (0.5, 0.5)),
        c0=0.14,
    )
    reset = NeuralTrajectoryObjective(
        case,
        truth,
        reset_windows((0, 1), 1, "endpoint", (0.5,)),
        c0=0.14,
    )
    _, full_gradient = full.value_and_gradient(parameters)
    _, reset_gradient = reset.value_and_gradient(parameters)
    result["cross_time_sensitivity"] = {
        "continuous_H2_vs_two_truth_reset_H1_gradients": _tree_comparison(
            full_gradient, reset_gradient
        ),
        "not_reduced_to_independent_truth_steps": (
            float(tree_norm(jax.tree.map(lambda x, y: x - y, full_gradient, reset_gradient)))
            > 0.0
        ),
    }

    timings = {}
    for horizon in selected["performance_horizons"]:
        weights = tuple(1.0 / horizon for _ in range(horizon))
        started = perf_counter()
        timed = NeuralTrajectoryObjective(
            case,
            truth,
            continuous_rollout(horizon, "accumulated", weights),
            c0=0.14,
            use_fixed_prefix=True,
        )
        prefix_setup = perf_counter() - started
        timed.clear_parameter_tape()
        started = perf_counter()
        timed.value(parameters)
        value_wall = perf_counter() - started
        started = perf_counter()
        timed.value_and_gradient(parameters)
        cached_gradient_wall = perf_counter() - started
        tape_bytes = timed._last_tape.estimated_owned_bytes
        timed.clear_parameter_tape()
        started = perf_counter()
        timed.value_and_gradient(parameters)
        fresh_value_gradient_wall = perf_counter() - started
        timings[f"H{horizon}"] = {
            "fixed_prefix_setup_wall_seconds": prefix_setup,
            "fresh_value_wall_seconds": value_wall,
            "gradient_after_same_theta_value_wall_seconds": cached_gradient_wall,
            "fresh_value_and_gradient_wall_seconds": fresh_value_gradient_wall,
            "same_theta_value_plus_gradient_wall_seconds": value_wall
            + cached_gradient_wall,
            "same_theta_reuse_speedup_vs_separate_fresh_value_and_gradient": (
                value_wall + fresh_value_gradient_wall
            )
            / max(value_wall + cached_gradient_wall, np.finfo(np.float64).tiny),
            "complete_forward_steps_per_fresh_evaluation": horizon,
            "complete_reverse_steps_per_gradient": horizon,
            "estimated_owned_tape_bytes": tape_bytes,
            "firedrake_PETSc_solve_instrumentation": (
                "existing certified child solves retained; no new solve counter added"
            ),
        }
    # Direct H=1 value comparison isolates fixed-prefix first-step savings.
    prefix_timing = {}
    for enabled in (False, True):
        local = NeuralTrajectoryObjective(
            case,
            truth,
            continuous_rollout(1, "endpoint", (1.0,)),
            c0=0.14,
            use_fixed_prefix=enabled,
        )
        local.clear_parameter_tape()
        started = perf_counter()
        value = local.value(parameters)
        prefix_timing[str(enabled).lower()] = {
            "objective": value,
            "wall_seconds": perf_counter() - started,
        }
    prefix_timing["speedup"] = prefix_timing["false"]["wall_seconds"] / max(
        prefix_timing["true"]["wall_seconds"], np.finfo(np.float64).tiny
    )
    result["performance"] = {
        "horizons": timings,
        "fixed_prefix_H1": prefix_timing,
        "serial_reference": True,
        "method3_parallelism": (
            "independent per-window value/gradient map followed by deterministic "
            "serial tree reduction; process-level parallel execution remains future opt-in"
        ),
        "parallel_prototype_run": False,
        "method4_parallelism_boundary": (
            "ordinary forward and reverse timesteps are sequential; no Parareal/MGRIT"
        ),
    }
    result["reuse_classification"] = {
        "fixed_across_all_theta": [
            "mesh/function spaces/quadrature/topography",
            "assembled fixed mass/weak operators and solvers",
            "truth states/time metadata/normalization/loss weights",
            "children 1-5 first-step prefix at each truth reset origin",
        ],
        "reusable_only_at_identical_theta": [
            "complete timestep-boundary states",
            "all split-child primal/RK/DG/moist caches needed by reverse",
            "local loss values",
        ],
        "must_recompute_after_theta_changes": [
            "neural moist outputs",
            "all states after the first neural moist child",
            "all later split/RK/DG/moist tapes and adjoints",
        ],
    }
    result["scientific_choices_unfrozen"] = [
        "Method-3 reset schedule and horizon",
        "endpoint versus accumulated loss",
        "accumulated weights",
        "final differentiable state metric and field weighting",
        "Method-4 production horizon",
    ]
    result["HVP"] = {
        "production_LBFGS_requires_HVP": False,
        "trajectory_HVP_optimized_tonight": False,
        "existing complete-child differentiated-VJP_retained": True,
    }
    result["status"] = "complete"
    write_json_record(output_path, result)
    return result


def run_nonscientific_smokes(configuration_path, output_directory, *, iterations=2):
    if int(iterations) < 1 or int(iterations) > 20:
        raise ValueError("trajectory smoke requires 1..20 accepted iterations")
    _, case, truth, parameters, model_configuration = _build_case(
        configuration_path, maximum_truth_step=2
    )
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("refusing to overwrite trajectory smoke outputs")
    output.mkdir(parents=True)
    problems = {
        "method3_truth_reset": reset_windows(
            (0, 1), 1, TrajectoryLossMode.ENDPOINT, (5.0e11,)
        ),
        "method4_continuous": continuous_rollout(
            2, TrajectoryLossMode.ACCUMULATED, (5.0e11, 5.0e11)
        ),
    }
    results = {}
    for name, windows in problems.items():
        objective = NeuralTrajectoryObjective(case, truth, windows, c0=0.14)
        adapter = TrajectoryPyROLObjective(objective, parameters)
        control = adapter.vector_from_pytree(parameters)
        initial = adapter.value(control, 0.0)
        rol = build_test2a_lbfgs_parameters(
            {
                "gradient_tolerance": 1.0e-8,
                "step_tolerance": 1.0e-12,
                "iteration_limit": int(iterations),
                "maximum_secant_storage": 20,
            }
        )
        started = perf_counter()
        solver = Solver(Problem(adapter, control), rol)
        solver.solve()
        wall = perf_counter() - started
        state = solver.getAlgorithmState()
        final_parameters = adapter.pytree_from_vector(control)
        final = objective.value(final_parameters)
        artifact = output / f"{name}_parameters.npz"
        save_mlp_parameters_atomic(artifact, final_parameters, model_configuration)
        results[name] = {
            "interpretation": "NONSCIENTIFIC IMPLEMENTATION SMOKE",
            "certification_only_uniform_loss_scale": 1.0e12,
            "initial_objective": initial,
            "final_objective": final,
            "objective_decreased": final < initial,
            "accepted_iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": adapter.value_evaluations,
            "gradient_evaluations": adapter.gradient_evaluations,
            "HVP_evaluations": adapter.hvp_evaluations,
            "wall_time_seconds": wall,
            "parameter_checkpoint": str(artifact.resolve()),
            "parameter_pytree_sha256": parameter_pytree_sha256(final_parameters),
            "work_counts": objective.work_counts().__dict__,
        }
        if adapter.hvp_evaluations != 0 or not results[name]["objective_decreased"]:
            raise RuntimeError(f"{name} nonscientific smoke failed")
    record = {
        "status": "complete",
        "interpretation": "NONSCIENTIFIC IMPLEMENTATION SMOKE",
        "methods": results,
        "truth_state_access": {
            "loaded_state_indices": [0, 80],
            "numerically_materialized_state_indices": [0, 2],
            "states_after_80_accessed": False,
        },
    }
    write_json_record(output / "trajectory_smokes.json", record)
    return record


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    certify = commands.add_parser("certify")
    certify.add_argument("--configuration", required=True)
    certify.add_argument("--output", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--configuration", required=True)
    smoke.add_argument("--output-directory", required=True)
    smoke.add_argument("--iterations", type=int, default=2)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "certify":
        certify_trajectory_framework(args.configuration, args.output)
    else:
        run_nonscientific_smokes(
            args.configuration, args.output_directory, iterations=args.iterations
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
