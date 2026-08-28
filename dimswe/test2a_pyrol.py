"""PyROL/ROL line-search L-BFGS driver for the Test 2A-1 JAX pytree.

The adapter is the deliberately small missing bridge between J4A arbitrary
JAX pytrees and PyROL's serial ``NumPyVector``.  Test 2A optimization uses
exact JAX gradients.  The exact JAX HVP callback remains available for local
derivative certification, but canonical L-BFGS does not request it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from pyrol import Objective, Problem, Solver
from pyrol.vectors import NumPyVector

from .rol_adapter import bound_constrained_lbfgs_parameters
from .resolved_hidden_c0 import write_json_record
from .test2a_operator import (
    DenseMLP,
    diagnostic_baselines,
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


class PytreeVectorCodec:
    """One immutable float64 pytree/serial-NumPyVector convention."""

    def __init__(self, initial_parameters):
        if not bool(jax.config.read("jax_enable_x64")):
            raise RuntimeError("Test 2A PyROL requires JAX_ENABLE_X64=True")
        flat, unravel = ravel_pytree(initial_parameters)
        if flat.dtype != jnp.float64:
            raise TypeError("neural parameter pytree must flatten to float64")
        self._unravel = unravel
        self._tree_definition = jax.tree_util.tree_structure(initial_parameters)
        self._leaf_shapes = tuple(
            tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(initial_parameters)
        )
        self._dimension = int(flat.size)

    @property
    def dimension(self):
        return self._dimension

    def flat_from_vector(self, vector, name, *, require_finite=True):
        if not isinstance(vector, NumPyVector):
            raise TypeError(f"{name} must be a PyROL NumPyVector")
        if vector.dimension() != self.dimension:
            raise ValueError(
                f"{name} dimension {vector.dimension()} != {self.dimension}"
            )
        values = np.asarray(vector.array)
        if values.dtype != np.float64:
            raise TypeError(f"{name} must contain float64 values")
        if require_finite and not np.all(np.isfinite(values)):
            raise TypeError(f"{name} must contain finite float64 values")
        return jnp.asarray(values, dtype=jnp.float64)

    def vector_from_pytree(self, parameters):
        if jax.tree_util.tree_structure(parameters) != self._tree_definition:
            raise ValueError("parameter pytree structure differs")
        shapes = tuple(
            tuple(np.shape(leaf)) for leaf in jax.tree_util.tree_leaves(parameters)
        )
        if shapes != self._leaf_shapes:
            raise ValueError("parameter pytree leaf shapes differ")
        flat, _ = ravel_pytree(parameters)
        if flat.size != self.dimension or flat.dtype != jnp.float64:
            raise ValueError("parameter pytree has incompatible size or dtype")
        return NumPyVector(np.asarray(flat, dtype=np.float64).copy())

    def pytree_from_vector(self, vector):
        return self._unravel(self.flat_from_vector(vector, "control"))


class JAXPytreeObjective(Objective):
    """Expose an arbitrary float64 JAX pytree objective to serial PyROL."""

    def __init__(self, objective, initial_parameters, *, use_jit=True):
        super().__init__()
        self._codec = PytreeVectorCodec(initial_parameters)
        self._objective = objective

        def flat_value(values):
            result = jnp.asarray(
                objective(self._codec._unravel(values)), dtype=jnp.float64
            )
            if result.shape != ():
                raise ValueError("JAX objective must return one scalar")
            return result

        value_and_gradient = jax.value_and_grad(flat_value)
        gradient = jax.grad(flat_value)
        hvp = lambda values, direction: jax.jvp(
            gradient, (values,), (direction,)
        )[1]
        if use_jit:
            self._flat_value = jax.jit(flat_value)
            self._flat_value_and_gradient = jax.jit(value_and_gradient)
            self._flat_hvp = jax.jit(hvp)
        else:
            self._flat_value = flat_value
            self._flat_value_and_gradient = value_and_gradient
            self._flat_hvp = hvp
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.value_history = []
        self.gradient_norm_history = []
        self.accepted_iteration_history = []
        self._pending_accepted_index = None

    @property
    def dimension(self):
        return self._codec.dimension

    def _flat_from_vector(self, vector, name, *, require_finite=True):
        return self._codec.flat_from_vector(
            vector, name, require_finite=require_finite
        )

    def vector_from_pytree(self, parameters):
        return self._codec.vector_from_pytree(parameters)

    def pytree_from_vector(self, vector):
        return self._codec.pytree_from_vector(vector)

    def update(self, control, *args):
        """Record only initial/accepted ROL iterates, never line-search trials."""
        flat = self._flat_from_vector(control, "control")
        update_type = str(args[0]) if args else "unspecified"
        iteration = int(args[1]) if len(args) > 1 else -1
        if "Initial" in update_type or "Accept" in update_type:
            self.accepted_iteration_history.append(
                {
                    "rol_update_type": update_type,
                    "rol_iteration_argument": iteration,
                    "objective": None,
                    "gradient_norm": None,
                    "control": np.asarray(flat, dtype=np.float64).copy(),
                }
            )
            self._pending_accepted_index = len(self.accepted_iteration_history) - 1

    def reset_accounting(self):
        """Reset callback counters after optional untimed JIT warm-up."""
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.value_history.clear()
        self.gradient_norm_history.clear()
        self.accepted_iteration_history.clear()
        self._pending_accepted_index = None

    def _pending_record(self, flat):
        if self._pending_accepted_index is None:
            return None
        record = self.accepted_iteration_history[self._pending_accepted_index]
        if np.array_equal(record["control"], np.asarray(flat, dtype=np.float64)):
            return record
        return None

    def value(self, control, tolerance):
        flat = self._flat_from_vector(control, "control")
        result = float(self._flat_value(flat))
        self.value_evaluations += 1
        self.value_history.append(result)
        record = self._pending_record(flat)
        if record is not None and record["objective"] is None:
            record["objective"] = result
        return result

    def gradient(self, output, control, tolerance):
        flat = self._flat_from_vector(control, "control")
        self._flat_from_vector(
            output, "gradient output", require_finite=False
        )
        value, gradient = self._flat_value_and_gradient(flat)
        output.array[:] = np.asarray(gradient, dtype=np.float64)
        self.gradient_evaluations += 1
        gradient_norm = float(jnp.linalg.norm(gradient))
        self.gradient_norm_history.append(gradient_norm)
        record = self._pending_record(flat)
        if record is not None and record["gradient_norm"] is None:
            record["gradient_norm"] = gradient_norm
            self._pending_accepted_index = None
        if not np.isfinite(float(value)):
            raise FloatingPointError("nonfinite objective during gradient evaluation")

    def hessVec(self, output, direction, control, tolerance):
        flat = self._flat_from_vector(control, "control")
        tangent = self._flat_from_vector(direction, "HVP direction")
        self._flat_from_vector(output, "HVP output", require_finite=False)
        action = self._flat_hvp(flat, tangent)
        output.array[:] = np.asarray(action, dtype=np.float64)
        self.hvp_evaluations += 1


class CallbackPytreeObjective(Objective):
    """Expose externally differentiated pytree value/gradient callbacks to ROL."""

    def __init__(self, value_callback, gradient_callback, initial_parameters):
        super().__init__()
        if not callable(value_callback) or not callable(gradient_callback):
            raise TypeError("value_callback and gradient_callback must be callable")
        self._codec = PytreeVectorCodec(initial_parameters)
        self._value_callback = value_callback
        self._gradient_callback = gradient_callback
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.value_history = []
        self.gradient_norm_history = []
        self.accepted_iteration_history = []

    @property
    def dimension(self):
        return self._codec.dimension

    def vector_from_pytree(self, parameters):
        return self._codec.vector_from_pytree(parameters)

    def pytree_from_vector(self, vector):
        return self._codec.pytree_from_vector(vector)

    def update(self, control, *args):
        update_type = str(args[0]) if args else "unspecified"
        iteration = int(args[1]) if len(args) > 1 else -1
        if "Initial" in update_type or "Accept" in update_type:
            self.accepted_iteration_history.append(
                {
                    "rol_update_type": update_type,
                    "rol_iteration_argument": iteration,
                    "control": np.asarray(
                        self._codec.flat_from_vector(control, "control"),
                        dtype=np.float64,
                    ).copy(),
                }
            )

    def value(self, control, tolerance):
        del tolerance
        parameters = self.pytree_from_vector(control)
        value = float(self._value_callback(parameters))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite externally differentiated objective")
        self.value_evaluations += 1
        self.value_history.append(value)
        return value

    def gradient(self, output, control, tolerance):
        del tolerance
        self._codec.flat_from_vector(
            output, "gradient output", require_finite=False
        )
        parameters = self.pytree_from_vector(control)
        gradient = self._gradient_callback(parameters)
        encoded = self.vector_from_pytree(gradient)
        output.array[:] = encoded.array
        self.gradient_evaluations += 1
        self.gradient_norm_history.append(float(np.linalg.norm(encoded.array)))

    def hessVec(self, output, direction, control, tolerance):
        del output, direction, control, tolerance
        self.hvp_evaluations += 1
        raise NotImplementedError(
            "canonical deployed-discrete L-BFGS requests exact gradients only"
        )


def _best_baseline_relative_rms(baselines):
    values = [
        record["metrics"]["relative_rms_error"]
        for record in baselines.values()
        if record["metrics"]["relative_rms_error"] is not None
    ]
    return min(values) if values else None


def build_test2a_lbfgs_parameters(optimizer):
    """Build the selected unbounded ROL L-BFGS policy with explicit memory."""
    parameters = bound_constrained_lbfgs_parameters(
        gradient_tolerance=float(optimizer["gradient_tolerance"]),
        step_tolerance=float(optimizer["step_tolerance"]),
        iteration_limit=int(optimizer["iteration_limit"]),
    )
    parameters.sublist("General").sublist("Secant").set(
        "Maximum Storage", int(optimizer["maximum_secant_storage"])
    )
    return parameters


def train_operator(configuration_path, dataset_path, output, parameter_output, plot_directory=None):
    """Run canonical deterministic full-batch ROL line-search L-BFGS."""
    selected = load_selected_configuration(configuration_path)
    dataset, metadata = load_operator_dataset(dataset_path)
    if dataset.sample_count != int(selected["data"]["sample_count"]):
        raise ValueError("dataset does not match the selected full training sample count")
    normalization = normalization_from_record(metadata["normalization"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    model = DenseMLP(model_configuration)
    initial_parameters = initialize_mlp(model_configuration)
    features = jnp.asarray(
        normalization.normalize_features(dataset.features), dtype=jnp.float64
    )
    targets = jnp.asarray(
        normalization.normalize_a(dataset.targets), dtype=jnp.float64
    ).reshape(-1, 1)

    def objective(parameters):
        return normalized_operator_objective(parameters, model, features, targets)

    adapter = JAXPytreeObjective(objective, initial_parameters, use_jit=True)
    control = adapter.vector_from_pytree(initial_parameters)
    initial_value = adapter.value(control, 0.0)
    optimizer = selected["optimizer"]
    if optimizer["library"] != "PyROL/ROL" or optimizer["method"] != "line-search L-BFGS":
        raise ValueError("selected Test 2A optimizer must be PyROL/ROL L-BFGS")
    parameters = build_test2a_lbfgs_parameters(optimizer)
    problem = Problem(adapter, control)
    solver = Solver(problem, parameters)
    started = perf_counter()
    solver.solve()
    wall_time = float(perf_counter() - started)
    final_value = adapter.value(control, 0.0)
    final_parameters = adapter.pytree_from_vector(control)
    state = solver.getAlgorithmState()
    predictions = physical_predictions(
        final_parameters, model, normalization, dataset.features
    )
    fit_metrics = operator_metrics(predictions, dataset.targets)
    baselines = metadata.get("diagnostic_baselines")
    if baselines is None:
        baselines = diagnostic_baselines(
            dataset.features, dataset.targets, normalization
        )
    best_baseline = _best_baseline_relative_rms(baselines)
    model_error = fit_metrics["relative_rms_error"]
    minimum_fraction = float(
        selected["embedding_readiness"]["minimum_relative_rms_improvement_fraction"]
    )
    material_improvement = (
        best_baseline is not None
        and model_error is not None
        and model_error <= (1.0 - minimum_fraction) * best_baseline
    )
    save_mlp_parameters(parameter_output, final_parameters, model_configuration)
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A-1 local operator learning",
        "selected_configuration": str(Path(configuration_path).resolve()),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_sha256_float64_content": metadata["sha256_float64_content"],
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
        },
        "architecture": model_configuration.to_record(),
        "normalization": normalization.to_record(),
        "objective": {
            "definition": "mean(((A_theta - A_truth) / RMS_training(A_truth))**2)",
            "batch_policy": "deterministic full batch",
            "sample_count": dataset.sample_count,
            "initial_value": initial_value,
            "final_value": final_value,
        },
        "optimizer": {
            **optimizer,
            "iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": adapter.value_evaluations,
            "gradient_evaluations": adapter.gradient_evaluations,
            "HVP_evaluations": adapter.hvp_evaluations,
            "exact_JAX_gradients": True,
            "HVP_used_by_canonical_fit": adapter.hvp_evaluations > 0,
            "wall_time_seconds": wall_time,
            "objective_history": [float(value) for value in adapter.value_history],
            "gradient_norm_history": [
                float(value) for value in adapter.gradient_norm_history
            ],
        },
        "metrics_complete_training_support": fit_metrics,
        "diagnostic_baselines": baselines,
        "embedding_readiness": {
            "best_baseline_relative_rms_error": best_baseline,
            "selected_model_relative_rms_error": model_error,
            "minimum_improvement_fraction": minimum_fraction,
            "materially_outperforms_trivial_baselines": material_improvement,
            "interpretation": (
                "operational pre-embedding screen, not a claim of future-state "
                "generalization"
            ),
        },
        "trained_parameter_file": str(Path(parameter_output).resolve()),
    }
    write_json_record(output, result)
    if plot_directory is not None:
        plot_operator_result(
            predictions,
            dataset.targets.reshape(-1),
            adapter.value_history,
            plot_directory,
        )
    return result


def plot_operator_result(prediction, target, history, destination):
    """Write lightweight deterministic fit/history diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    stride = max(1, target.size // 5000)
    figure, axis = plt.subplots()
    axis.scatter(target[::stride], prediction[::stride], s=3, alpha=0.4)
    lower = min(float(np.min(target)), float(np.min(prediction)))
    upper = max(float(np.max(target)), float(np.max(prediction)))
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.set(xlabel="truth A", ylabel="learned A")
    figure.tight_layout()
    figure.savefig(directory / "operator_prediction_vs_truth.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.semilogy(range(len(history)), history)
    axis.set(xlabel="objective callback evaluation", ylabel="normalized MSE")
    figure.tight_layout()
    figure.savefig(directory / "operator_training_history.png", dpi=150)
    plt.close(figure)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--configuration", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--parameter-output", required=True)
    train.add_argument("--plot-directory")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "train":
        train_operator(
            arguments.configuration,
            arguments.dataset,
            arguments.output,
            arguments.parameter_output,
            arguments.plot_directory,
        )
        return 0
    raise AssertionError("unreachable Test 2A PyROL command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CallbackPytreeObjective",
    "JAXPytreeObjective",
    "PytreeVectorCodec",
    "plot_operator_result",
    "build_test2a_lbfgs_parameters",
    "train_operator",
)
