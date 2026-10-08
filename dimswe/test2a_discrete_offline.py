"""Test 2A-3A deployed-discrete offline neural-A objective.

The generic objective and comparison helpers are Firedrake-free. Production
construction imports Firedrake-backed adapters lazily, reads truth states
0..80 only, and never advances a model-generated state recursively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .learned_physics.parameters import (
    tree_axpy,
    tree_copy,
    tree_dot,
    tree_norm,
    tree_zeros,
    validate_float64_tree,
)
from .resolved_hidden_c0 import write_json_record
from .test2a_embedded_moist import (
    load_frozen_neural_a_physics,
    parameter_pytree_sha256 as parameter_fingerprint,
)
from .test2a_operator import (
    DenseMLP,
    initialize_mlp,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    normalized_operator_objective,
    save_mlp_parameters,
)


TRAINING_STEPS = tuple(range(81))
OBJECTIVE_SEMANTICS = (
    "global normalized squared mixed-mass norm of the difference between "
    "analytical-A/original-R and neural-A/original-R mass-solved moist "
    "tendencies at fixed truth states"
)


def require_training_steps(steps):
    actual = tuple(int(step) for step in steps)
    if actual != TRAINING_STEPS:
        raise ValueError(
            "Test 2A-3A requires exactly fixed truth states 0..80; "
            "states after 80 are forbidden"
        )
    return actual


def load_discrete_offline_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("benchmark_stage") != (
        "Test 2A-3A deployed-discrete offline objective"
    ):
        raise ValueError("not a selected Test 2A-3A configuration")
    if record["truth"] != {
        "run_directory": "external-results/test1b-production/truth_c0_0.14",
        "selected_plan": "dimswe/configs/test1b_selected_plan.json",
        "state_indices": [0, 80],
        "states_after_80_forbidden": True,
    }:
        raise ValueError("Test 2A-3A truth support changed")
    objective = record["objective"]
    if (
        objective["quantity"] != "mass-solved moist tendency"
        or objective["metric"] != "squared production mixed L2/mass norm"
        or objective["state_indices"] != [0, 80]
        or objective["per_state_inverse_activity_weighting"] is not False
    ):
        raise ValueError("Test 2A-3A objective contract changed")
    if record["physics"]["recursive_model_state_propagation"] is not False:
        raise ValueError("deployed-discrete offline cannot propagate predictions")
    optimizer = record["optimizer"]
    if (
        optimizer["library"] != "PyROL/ROL"
        or optimizer["method"] != "line-search L-BFGS"
        or int(optimizer["maximum_secant_storage"]) != 20
        or optimizer["initialization"] != "seed0_initial"
    ):
        raise ValueError("Test 2A-3A optimizer contract changed")
    return record


@dataclass(frozen=True)
class DiscreteOfflineObservation:
    """One fixed-state target and its analytical-A normalization response."""

    step: int
    payload: object
    target_tendency: object
    analytical_a_tendency: object
    semantics: str = OBJECTIVE_SEMANTICS


@dataclass(frozen=True)
class DiscretePredictionCache:
    tendency: object
    auxiliary: object = None


@dataclass(frozen=True)
class _ForwardEvaluation:
    flat_parameters: np.ndarray
    value: float
    entries: tuple


class DeployedDiscreteOfflineObjective:
    """Globally normalized fixed-state deployed-discrete least squares.

    The operations object owns the concrete state representation. It supplies
    prediction, subtraction, mixed-mass norm, parameter pullback, and optional
    exact HVP contribution actions. No objective method can replace one
    observation's fixed state with a prediction from another observation.
    """

    def __init__(self, observations, operations, *, require_canonical_steps=False):
        self.observations = tuple(observations)
        if not self.observations:
            raise ValueError("deployed-discrete objective requires observations")
        steps = tuple(observation.step for observation in self.observations)
        if len(set(steps)) != len(steps):
            raise ValueError("deployed-discrete observation steps must be unique")
        if require_canonical_steps:
            require_training_steps(steps)
        self.operations = operations
        terms = tuple(
            float(operations.squared_mass_norm(observation.analytical_a_tendency))
            for observation in self.observations
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in terms):
            raise ValueError("normalization terms must be finite and nonnegative")
        self.normalization_terms = terms
        self.normalizer = float(sum(terms))
        if self.normalizer <= np.finfo(np.float64).tiny:
            raise ValueError(
                "global analytical-A tendency normalization must be positive"
            )
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.prediction_evaluations = 0
        self._forward_cache = None

    @staticmethod
    def _flat_parameters(parameters):
        owned = validate_float64_tree(parameters, name="neural parameters")
        flat, _ = ravel_pytree(owned)
        result = np.asarray(flat, dtype=np.float64).copy()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("neural parameters must be finite")
        return owned, result

    def _forward(self, parameters):
        owned, flat = self._flat_parameters(parameters)
        cached = self._forward_cache
        if cached is not None and np.array_equal(flat, cached.flat_parameters):
            return owned, cached
        numerator = 0.0
        entries = []
        for observation in self.observations:
            prediction = self.operations.predict(owned, observation)
            residual = self.operations.subtract(
                prediction.tendency,
                observation.target_tendency,
                f"test2a3a_residual_{observation.step}",
            )
            squared = float(self.operations.squared_mass_norm(residual))
            if not np.isfinite(squared) or squared < 0.0:
                raise FloatingPointError("nonfinite deployed-discrete residual")
            numerator += squared
            entries.append((observation, prediction, residual))
            self.prediction_evaluations += 1
        cached = _ForwardEvaluation(
            flat_parameters=flat,
            value=float(numerator / self.normalizer),
            entries=tuple(entries),
        )
        self._forward_cache = cached
        return owned, cached

    def value(self, parameters):
        _, evaluation = self._forward(parameters)
        self.value_evaluations += 1
        return evaluation.value

    def gradient(self, parameters):
        owned, evaluation = self._forward(parameters)
        result = tree_zeros(owned)
        for observation, prediction, residual in evaluation.entries:
            contribution = self.operations.gradient_contribution(
                owned, observation, prediction, residual
            )
            result = tree_axpy(result, 1.0 / self.normalizer, contribution)
        self.gradient_evaluations += 1
        return result

    def value_and_gradient(self, parameters):
        value = self.value(parameters)
        return value, self.gradient(parameters)

    def hess_vec(self, parameters, direction):
        if not hasattr(self.operations, "hvp_contribution"):
            raise NotImplementedError("operations do not expose an exact HVP")
        owned, evaluation = self._forward(parameters)
        direction = validate_float64_tree(direction, name="parameter direction")
        result = tree_zeros(owned)
        for observation, prediction, residual in evaluation.entries:
            contribution = self.operations.hvp_contribution(
                owned, direction, observation, prediction, residual
            )
            result = tree_axpy(result, 1.0 / self.normalizer, contribution)
        self.hvp_evaluations += 1
        return result


def objective_gradient_comparison(
    operator_value, operator_gradient, discrete_value, discrete_gradient
):
    """Return scale-aware evidence of parameter-space objective distinction."""
    operator_gradient = validate_float64_tree(
        operator_gradient, name="operator gradient"
    )
    discrete_gradient = validate_float64_tree(
        discrete_gradient, name="discrete gradient"
    )
    operator_norm = float(tree_norm(operator_gradient))
    discrete_norm = float(tree_norm(discrete_gradient))
    dot = float(tree_dot(operator_gradient, discrete_gradient))
    if operator_norm == 0.0 or discrete_norm == 0.0:
        cosine = None
    else:
        cosine = dot / (operator_norm * discrete_norm)
    operator_squared = float(tree_dot(operator_gradient, operator_gradient))
    scale = None if operator_squared == 0.0 else dot / operator_squared
    proportional_residual = None
    if scale is not None and discrete_norm > 0.0:
        difference = tree_axpy(discrete_gradient, -scale, operator_gradient)
        proportional_residual = float(tree_norm(difference)) / discrete_norm
    return {
        "operator_value": float(operator_value),
        "deployed_discrete_value": float(discrete_value),
        "operator_gradient_norm": operator_norm,
        "deployed_discrete_gradient_norm": discrete_norm,
        "gradient_dot_product": dot,
        "gradient_cosine_similarity": cosine,
        "best_positive_scaling_discrete_over_operator": scale,
        "relative_nonproportional_gradient_residual": proportional_residual,
    }


@dataclass(frozen=True)
class ProductionObservationPayload:
    packed_state: object
    packed_fields: object
    moist_parameters: object
    target_original_r: np.ndarray

    def __post_init__(self):
        original_r = np.array(
            self.target_original_r, dtype=np.float64, copy=True
        )
        original_r.setflags(write=False)
        object.__setattr__(self, "target_original_r", original_r)


class ProductionDiscreteOfflineOperations:
    """Certified JAX local physics plus existing Firedrake W, M^-1, and W*."""

    def __init__(self, helper, neural_physics):
        self.helper = helper
        self.neural_physics = neural_physics

        def source(parameters, state, fields, moist_parameters):
            return neural_physics.combined_with_parameters(
                state, fields, moist_parameters, parameters
            )["source"]

        def combined(parameters, state, fields, moist_parameters):
            return neural_physics.combined_with_parameters(
                state, fields, moist_parameters, parameters
            )

        def parameter_vjp(parameters, state, fields, moist_parameters, covector):
            local_source = lambda active: source(
                active, state, fields, moist_parameters
            )
            _, pullback = jax.vjp(local_source, parameters)
            return pullback(covector)[0]

        def parameter_jvp(parameters, direction, state, fields, moist_parameters):
            local_source = lambda active: source(
                active, state, fields, moist_parameters
            )
            return jax.jvp(local_source, (parameters,), (direction,))

        def differentiated_vjp(
            parameters,
            covector,
            direction,
            covector_direction,
            state,
            fields,
            moist_parameters,
        ):
            def vjp_map(active_parameters, active_covector):
                local_source = lambda active: source(
                    active, state, fields, moist_parameters
                )
                _, pullback = jax.vjp(local_source, active_parameters)
                return pullback(active_covector)[0]

            return jax.jvp(
                vjp_map,
                (parameters, covector),
                (direction, covector_direction),
            )

        self._source_kernel = jax.jit(source)
        self._combined_kernel = jax.jit(combined)
        self._parameter_vjp_kernel = jax.jit(parameter_vjp)
        self._parameter_jvp_kernel = jax.jit(parameter_jvp)
        self._differentiated_vjp_kernel = jax.jit(differentiated_vjp)

    @staticmethod
    def _copy_state(value, name):
        result = value.copy(deepcopy=True)
        result.rename(name)
        return result

    def predict(self, parameters, observation):
        payload = observation.payload
        combined = self._combined_kernel(
            parameters,
            payload.packed_state,
            payload.packed_fields,
            payload.moist_parameters,
        )
        source = self.helper._from_device_tree(combined["source"])
        rates = self.helper._from_device_tree(combined["rates"])
        original_r_scale = max(
            float(np.max(np.abs(rates["R"]))),
            float(np.max(np.abs(payload.target_original_r))),
        )
        maximum_difference = float(
            np.max(np.abs(rates["R"] - payload.target_original_r))
        )
        original_r_tolerance = (
            0.0
            if original_r_scale == 0.0
            else 8.0 * np.finfo(np.float64).eps * original_r_scale
        )
        if maximum_difference > original_r_tolerance:
            raise AssertionError(
                "analytical target and neural prediction did not evaluate "
                "the same original R to float64 precision at the fixed truth "
                f"state; maximum absolute difference={maximum_difference:.17e}, "
                f"tolerance={original_r_tolerance:.17e}"
            )
        source_dual = self.helper.source_assembly(source)
        tendency = self.helper.state_riesz_representative(
            source_dual, f"test2a3a_prediction_tendency_{observation.step}"
        )
        return DiscretePredictionCache(
            tendency=tendency,
            auxiliary={"source": source, "rates": rates},
        )

    def subtract(self, left, right, name):
        result = self._copy_state(left, name)
        with result.dat.vec as output, right.dat.vec_ro as reference:
            output.axpy(-1.0, reference)
        return result

    def squared_mass_norm(self, value):
        dual = self.helper.state_mass_map(value, "test2a3a_mass_norm_dual")
        return float(self.helper.dual_pairing(dual, value))

    def gradient_contribution(
        self, parameters, observation, prediction, residual
    ):
        del prediction
        payload = observation.payload
        covector = self.helper.source_assembly_transpose(residual)
        pullback = self._parameter_vjp_kernel(
            parameters,
            payload.packed_state,
            payload.packed_fields,
            payload.moist_parameters,
            self.helper._to_device_tree(covector),
        )
        return jax.tree.map(lambda value: 2.0 * value, pullback)

    def hvp_contribution(
        self, parameters, direction, observation, prediction, residual
    ):
        del prediction
        payload = observation.payload
        _, source_direction_device = self._parameter_jvp_kernel(
            parameters,
            direction,
            payload.packed_state,
            payload.packed_fields,
            payload.moist_parameters,
        )
        source_direction = self.helper._from_device_tree(source_direction_device)
        tendency_direction = self.helper.state_riesz_representative(
            self.helper.source_assembly(source_direction),
            f"test2a3a_parameter_tendency_direction_{observation.step}",
        )
        covector = self.helper.source_assembly_transpose(residual)
        covector_direction = self.helper.source_assembly_transpose(
            tendency_direction
        )
        _, incremental = self._differentiated_vjp_kernel(
            parameters,
            self.helper._to_device_tree(covector),
            direction,
            self.helper._to_device_tree(covector_direction),
            payload.packed_state,
            payload.packed_fields,
            payload.moist_parameters,
        )
        return jax.tree.map(lambda value: 2.0 * value, incremental)


@dataclass(frozen=True)
class PreparedDiscreteOfflineProblem:
    objective: DeployedDiscreteOfflineObjective
    initial_parameters: object
    frozen_operator_parameters: object
    operator_objective: object
    model_configuration: object
    configuration: dict
    truth_metadata: dict


def _a_sensitive_source(primal_cache):
    h = np.asarray(primal_cache.packed_state["h"], dtype=np.float64)
    a_rate = np.asarray(primal_cache.rates["A"], dtype=np.float64)
    beta2 = float(primal_cache.parameters["g"] * primal_cache.parameters["L"])
    return {
        "S": h * beta2 * a_rate,
        "Qv": h * a_rate,
        "Qc": -h * a_rate,
        "Qr": np.zeros_like(a_rate),
    }


def prepare_production_problem(
    configuration_path,
    *,
    truth_run=None,
    selected_plan=None,
    operator_dataset=None,
):
    """Build fixed states/targets only; never execute a recursive trajectory."""
    configuration = load_discrete_offline_configuration(configuration_path)
    truth_run = truth_run or configuration["truth"]["run_directory"]
    selected_plan = selected_plan or configuration["truth"]["selected_plan"]
    operator_dataset = operator_dataset or configuration["model"][
        "operator_dataset"
    ]

    from .hidden_c0 import _serial_solver_parameters
    from .jax_moist_hvp import JAXMoistEulerHVP
    from .logger import EmptyLogger
    from .resolved_hidden_c0_inference import load_resolved_truth
    from .selected_test1b import load_selected_test1b_plan
    from .timestepping import Euler

    _, selected = load_selected_test1b_plan(selected_plan)
    inference = selected.inference_configuration(Path(truth_run).resolve())
    if (inference.training_start_step, inference.training_stop_step) != (0, 80):
        raise ValueError("selected Test 1B plan no longer exposes states 0..80")
    case, trajectory = load_resolved_truth(inference, include_heldout=False)
    require_training_steps(trajectory.states.keys())

    solver_parameters = _serial_solver_parameters()
    analytical_euler = Euler(
        case.model,
        EmptyLogger(),
        solver_parameters,
        terms=["threewayphysics"],
    )
    helper = JAXMoistEulerHVP(analytical_euler, use_jit=True)
    neural_physics = load_frozen_neural_a_physics(
        configuration["model"]["embedding_configuration"], use_jit=True
    )
    operations = ProductionDiscreteOfflineOperations(helper, neural_physics)
    observations = []
    for step in TRAINING_STEPS:
        target = helper.take_forward_step_cached(
            trajectory.states[step], case.t0 + step * case.dt, case.dt
        )
        analytical_a_source = _a_sensitive_source(target)
        analytical_a_tendency = helper.state_riesz_representative(
            helper.source_assembly(analytical_a_source),
            f"test2a3a_analytical_a_tendency_{step}",
        )
        payload = ProductionObservationPayload(
            packed_state=helper._to_device_tree(target.packed_state),
            packed_fields=helper._to_device_tree(target.packed_fields),
            moist_parameters=helper._to_device_tree(target.parameters),
            target_original_r=np.asarray(
                target.rates["R"], dtype=np.float64
            ).copy(),
        )
        observations.append(
            DiscreteOfflineObservation(
                step=step,
                payload=payload,
                target_tendency=target.tendency.copy(deepcopy=True),
                analytical_a_tendency=analytical_a_tendency,
            )
        )
    objective = DeployedDiscreteOfflineObjective(
        observations, operations, require_canonical_steps=True
    )

    selected_operator = load_selected_configuration(
        configuration["model"]["selected_operator_configuration"]
    )
    model_configuration = mlp_configuration_from_record(selected_operator["model"])
    initial_parameters = initialize_mlp(model_configuration)
    expected_fingerprint = configuration["model"][
        "canonical_initial_parameter_sha256"
    ]
    if parameter_fingerprint(initial_parameters) != expected_fingerprint:
        raise ValueError("Test 2A-3A seed-0 initialization changed")

    dataset, metadata = load_operator_dataset(operator_dataset)
    if tuple(dataset.steps) != TRAINING_STEPS:
        raise ValueError("operator comparison dataset is not states 0..80")
    normalization = normalization_from_record(metadata["normalization"])
    model = DenseMLP(model_configuration)
    features = jnp.asarray(
        normalization.normalize_features(dataset.features), dtype=jnp.float64
    )
    targets = jnp.asarray(
        normalization.normalize_a(dataset.targets), dtype=jnp.float64
    ).reshape(-1, 1)

    def operator_objective(parameters):
        return normalized_operator_objective(
            parameters, model, features, targets
        )

    return PreparedDiscreteOfflineProblem(
        objective=objective,
        initial_parameters=tree_copy(initial_parameters),
        frozen_operator_parameters=neural_physics.parameters,
        operator_objective=operator_objective,
        model_configuration=model_configuration,
        configuration=configuration,
        truth_metadata=dict(trajectory.metadata),
    )


def _deterministic_parameter_vectors(prepared):
    initial = prepared.initial_parameters
    flat, unravel = ravel_pytree(initial)
    values = np.linspace(-1.0, 1.0, int(flat.size), dtype=np.float64)
    values /= np.linalg.norm(values)
    relative = float(
        prepared.configuration["comparison"][
            "deterministic_relative_perturbation"
        ]
    )
    amplitude = relative * max(1.0, float(np.linalg.norm(np.asarray(flat))))
    direction = jnp.asarray(values, dtype=jnp.float64)
    return {
        "seed0_initial": tree_copy(initial),
        "frozen_operator_trained": tree_copy(prepared.frozen_operator_parameters),
        "seed0_plus_deterministic_relative_1e-3": unravel(flat + amplitude * direction),
        "seed0_minus_deterministic_relative_1e-3": unravel(flat - amplitude * direction),
    }, unravel(direction)


def compare_operator_and_discrete(
    configuration_path,
    output,
    *,
    truth_run=None,
    selected_plan=None,
    operator_dataset=None,
):
    prepared = prepare_production_problem(
        configuration_path,
        truth_run=truth_run,
        selected_plan=selected_plan,
        operator_dataset=operator_dataset,
    )
    vectors, direction = _deterministic_parameter_vectors(prepared)
    operator_value_gradient = jax.jit(
        jax.value_and_grad(prepared.operator_objective)
    )
    records = []
    for name, parameters in vectors.items():
        started = perf_counter()
        operator_value, operator_gradient = operator_value_gradient(parameters)
        discrete_value, discrete_gradient = prepared.objective.value_and_gradient(
            parameters
        )
        comparison = objective_gradient_comparison(
            operator_value,
            operator_gradient,
            discrete_value,
            discrete_gradient,
        )
        comparison.update(
            {
                "name": name,
                "parameter_sha256": parameter_fingerprint(parameters),
                "operator_directional_derivative": float(
                    tree_dot(operator_gradient, direction)
                ),
                "deployed_discrete_directional_derivative": float(
                    tree_dot(discrete_gradient, direction)
                ),
                "wall_time_seconds": float(perf_counter() - started),
            }
        )
        records.append(comparison)
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A-3A offline objective comparison",
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
            "recursive_model_state_propagation": False,
        },
        "objective": {
            "semantics": OBJECTIVE_SEMANTICS,
            "normalizer": prepared.objective.normalizer,
            "normalization_terms": list(prepared.objective.normalization_terms),
            "target_count": len(prepared.objective.observations),
        },
        "canonical_initial_parameter_sha256": parameter_fingerprint(
            prepared.initial_parameters
        ),
        "comparisons": records,
        "accounting": {
            "discrete_value_evaluations": prepared.objective.value_evaluations,
            "discrete_gradient_evaluations": prepared.objective.gradient_evaluations,
            "discrete_prediction_evaluations": (
                prepared.objective.prediction_evaluations
            ),
            "complete_solver_steps": 0,
        },
    }
    write_json_record(output, result)
    return result


def train_deployed_discrete(
    configuration_path,
    output,
    parameter_output,
    *,
    truth_run=None,
    selected_plan=None,
    operator_dataset=None,
):
    """Prepared canonical ROL path; execution requires a later explicit gate."""
    from pyrol import Problem, Solver

    from .test2a_pyrol import (
        CallbackPytreeObjective,
        build_test2a_lbfgs_parameters,
    )

    parameter_path = Path(parameter_output)
    if (
        Path(output).exists()
        or parameter_path.exists()
        or parameter_path.with_suffix(".json").exists()
    ):
        raise FileExistsError("refusing to overwrite deployed-discrete fit output")
    prepared = prepare_production_problem(
        configuration_path,
        truth_run=truth_run,
        selected_plan=selected_plan,
        operator_dataset=operator_dataset,
    )
    adapter = CallbackPytreeObjective(
        prepared.objective.value,
        prepared.objective.gradient,
        prepared.initial_parameters,
    )
    control = adapter.vector_from_pytree(prepared.initial_parameters)
    optimizer = prepared.configuration["optimizer"]
    problem = Problem(adapter, control)
    solver = Solver(problem, build_test2a_lbfgs_parameters(optimizer))
    started = perf_counter()
    solver.solve()
    wall_time = float(perf_counter() - started)
    state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    save_mlp_parameters(
        parameter_output, final_parameters, prepared.model_configuration
    )
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A-3A deployed-discrete offline fit",
        "initialization": {
            "kind": "same Test 2A-1 seed-0 initial pytree",
            "parameter_sha256": parameter_fingerprint(
                prepared.initial_parameters
            ),
            "operator_trained_warm_start": False,
        },
        "optimizer": {
            **optimizer,
            "accepted_iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": adapter.value_evaluations,
            "gradient_evaluations": adapter.gradient_evaluations,
            "HVP_evaluations": adapter.hvp_evaluations,
            "wall_time_seconds": wall_time,
        },
        "final_objective": float(state.value),
        "final_parameter_sha256": parameter_fingerprint(final_parameters),
        "parameter_output": str(Path(parameter_output).resolve()),
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
            "recursive_model_state_propagation": False,
        },
    }
    write_json_record(output, result)
    return result


def _add_common_arguments(parser):
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--truth-run")
    parser.add_argument("--selected-plan")
    parser.add_argument("--operator-dataset")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    _add_common_arguments(compare)
    compare.add_argument("--output", required=True)
    train = subparsers.add_parser("train")
    _add_common_arguments(train)
    train.add_argument("--output", required=True)
    train.add_argument("--parameter-output", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    keyword = {
        "truth_run": arguments.truth_run,
        "selected_plan": arguments.selected_plan,
        "operator_dataset": arguments.operator_dataset,
    }
    if arguments.command == "compare":
        compare_operator_and_discrete(
            arguments.configuration, arguments.output, **keyword
        )
        return 0
    if arguments.command == "train":
        train_deployed_discrete(
            arguments.configuration,
            arguments.output,
            arguments.parameter_output,
            **keyword,
        )
        return 0
    raise AssertionError("unreachable Test 2A-3A command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DeployedDiscreteOfflineObjective",
    "DiscreteOfflineObservation",
    "DiscretePredictionCache",
    "OBJECTIVE_SEMANTICS",
    "ProductionDiscreteOfflineOperations",
    "compare_operator_and_discrete",
    "load_discrete_offline_configuration",
    "objective_gradient_comparison",
    "prepare_production_problem",
    "require_training_steps",
    "train_deployed_discrete",
)
