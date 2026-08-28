"""Exact shared window trajectories for Test-2A Methods 3 and 4.

This module composes the already-certified complete six-child DIMSWE step.
It does not select a production training protocol: endpoint versus accumulated
losses, weights, reset origins, and horizons remain explicit configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from numbers import Integral, Real
from types import MappingProxyType

import jax
from jax.flatten_util import ravel_pytree
import numpy as np
from pyrol import Objective

from .hidden_c0 import _copy_function
from .hyperviscosity_hvp import _copy_cofunction
from .learned_physics.parameters import (
    tree_axpy,
    tree_copy,
    tree_zeros,
    validate_float64_tree,
)
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_pyrol import PytreeVectorCodec


class TrajectoryLossMode(str, Enum):
    ENDPOINT = "endpoint"
    ACCUMULATED = "accumulated"


@dataclass(frozen=True)
class NeuralWindowSpec:
    """One reset window, or the single continuous Method-4 rollout."""

    start_step: int
    horizon: int
    loss_mode: TrajectoryLossMode
    weights: tuple[float, ...]
    label: str = "window"

    def __post_init__(self):
        for name in ("start_step", "horizon"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if int(self.start_step) < 0 or int(self.horizon) < 1:
            raise ValueError("window start must be nonnegative and horizon positive")
        if int(self.start_step) + int(self.horizon) > 80:
            raise ValueError("Test-2A trajectories may not access a state after 80")
        mode = TrajectoryLossMode(self.loss_mode)
        expected = 1 if mode is TrajectoryLossMode.ENDPOINT else int(self.horizon)
        if len(tuple(self.weights)) != expected:
            raise ValueError(f"{mode.value} loss requires {expected} explicit weights")
        for weight in self.weights:
            if not isinstance(weight, Real) or not np.isfinite(float(weight)):
                raise TypeError("trajectory weights must be finite real scalars")
            if float(weight) < 0.0:
                raise ValueError("trajectory weights must be nonnegative")
        if not any(float(value) > 0.0 for value in self.weights):
            raise ValueError("at least one trajectory weight must be positive")

    @property
    def target_offsets(self):
        if TrajectoryLossMode(self.loss_mode) is TrajectoryLossMode.ENDPOINT:
            return (int(self.horizon),)
        return tuple(range(1, int(self.horizon) + 1))

    @property
    def target_steps(self):
        return tuple(int(self.start_step) + value for value in self.target_offsets)

    def to_record(self):
        return {
            "start_step": int(self.start_step),
            "horizon": int(self.horizon),
            "loss_mode": TrajectoryLossMode(self.loss_mode).value,
            "weights": [float(value) for value in self.weights],
            "target_steps": list(self.target_steps),
            "label": str(self.label),
        }


@dataclass(frozen=True)
class NeuralWindowTape:
    spec: NeuralWindowSpec
    states: tuple[object, ...]
    step_caches: tuple[object, ...]
    used_fixed_prefix: bool
    local_values: MappingProxyType


@dataclass(frozen=True)
class NeuralTrajectoryTape:
    parameter_sha256: str
    problem_sha256: str
    windows: tuple[NeuralWindowTape, ...]
    objective_value: float
    estimated_owned_bytes: int


@dataclass(frozen=True)
class NeuralTrajectoryWorkCounts:
    value_evaluations: int
    gradient_evaluations: int
    forward_complete_steps: int
    reverse_complete_steps: int
    fixed_prefix_builds: int
    fixed_prefix_uses: int
    full_first_prefix_recomputations: int
    same_theta_tape_hits: int
    same_theta_tape_misses: int
    tape_invalidations: int


def reset_windows(starts, horizon, loss_mode, weights):
    """Build explicit independent Method-3 windows without hidden overlap rules."""
    return tuple(
        NeuralWindowSpec(
            start_step=int(start),
            horizon=int(horizon),
            loss_mode=TrajectoryLossMode(loss_mode),
            weights=tuple(float(value) for value in weights),
            label=f"truth_reset_{int(start)}_H{int(horizon)}",
        )
        for start in tuple(starts)
    )


def continuous_rollout(horizon, loss_mode, weights):
    """Build one Method-4 trajectory beginning once from truth state zero."""
    return (
        NeuralWindowSpec(
            start_step=0,
            horizon=int(horizon),
            loss_mode=TrajectoryLossMode(loss_mode),
            weights=tuple(float(value) for value in weights),
            label=f"continuous_rollout_H{int(horizon)}",
        ),
    )


def _state_bytes(value):
    return sum(
        int(np.asarray(field.dat.data_ro).nbytes) for field in value.subfunctions
    )


def estimate_owned_bytes(value):
    """Conservative unique-array/Firedrake-data byte count for an owned tape."""
    seen = set()

    def visit(item):
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, np.ndarray):
            return int(item.nbytes)
        if hasattr(item, "subfunctions") and hasattr(item, "dat"):
            try:
                return _state_bytes(item)
            except (AttributeError, TypeError):
                return int(np.asarray(item.dat.data_ro).nbytes)
        if isinstance(item, dict) or isinstance(item, MappingProxyType):
            return sum(visit(entry) for entry in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(entry) for entry in item)
        if is_dataclass(item):
            return sum(visit(getattr(item, field.name)) for field in fields(item))
        try:
            leaves = jax.tree_util.tree_leaves(item)
        except TypeError:
            return 0
        if len(leaves) == 1 and leaves[0] is item:
            return 0
        return sum(int(np.asarray(leaf).nbytes) for leaf in leaves)

    return int(visit(value))


class CertificationMixedMassMetric:
    """Replaceable certification-only mixed-mass least-squares metric.

    Each target is normalized by its own positive squared mixed mass norm.
    This convention is already certified for Test 1B plumbing, but it is not
    frozen here as the final scientific Method-3/4 training metric.
    """

    name = "certification_only_per_target_relative_mixed_mass_L2"
    scientifically_frozen = False

    def __init__(self, helper, targets):
        self.helper = helper
        self._normalizers = {}
        for step, target in targets.items():
            dual = helper.state_mass_map(
                target, f"test2a_trajectory_target_mass_{int(step)}"
            )
            squared = float(helper.dual_pairing(dual, target))
            if not np.isfinite(squared) or squared <= 0.0:
                raise ValueError(f"target {step} has no positive mixed mass norm")
            self._normalizers[int(step)] = squared

    def record(self):
        return {
            "name": self.name,
            "scientifically_frozen": False,
            "loss": "0.5*w_j*||Xhat-Xtruth||_M^2/||Xtruth||_M^2",
            "target_normalization": "per-target truth mixed mass norm",
            "component_weights": "none beyond the production mixed mass metric",
        }

    def value_and_dual(self, state, target, target_step, weight, name):
        residual = _copy_function(state, f"{name}_residual")
        with residual.dat.vec as output, target.dat.vec_ro as truth:
            output.axpy(-1.0, truth)
        dual = self.helper.state_mass_map(residual, f"{name}_mass_dual")
        scale = float(weight) / self._normalizers[int(target_step)]
        value = 0.5 * scale * float(self.helper.dual_pairing(dual, residual))
        with dual.dat.vec as vector:
            vector.scale(scale)
        return float(value), dual

    def value(self, state, target, target_step, weight, name):
        return self.value_and_dual(
            state, target, target_step, weight, name
        )[0]


class GlobalMixedMassMetric:
    """Frozen globally normalized mixed-mass trajectory metric.

    Unlike the earlier certification-only metric, this metric applies one
    positive, parameter-independent denominator to every target and contains
    no hidden factor of one half.  Consequently its derivative is
    ``2*w/D * M(state-target)``.
    """

    name = "global_analytical_A_increment_normalized_mixed_mass_L2"
    scientifically_frozen = True

    def __init__(self, helper, denominator, *, denominator_sha256):
        self.helper = helper
        self.denominator = float(denominator)
        if not np.isfinite(self.denominator) or self.denominator <= 0.0:
            raise ValueError("global trajectory denominator must be positive")
        fingerprint = str(denominator_sha256)
        if len(fingerprint) != 64:
            raise ValueError("global trajectory denominator requires a SHA256")
        self.denominator_sha256 = fingerprint

    def record(self):
        return {
            "name": self.name,
            "scientifically_frozen": True,
            "loss": "w_j*||Xhat-Xtruth||_M^2/D",
            "target_normalization": "one common analytical-A increment denominator",
            "component_weights": "none beyond the production mixed mass metric",
            "denominator": self.denominator,
            "denominator_sha256": self.denominator_sha256,
        }

    def value_and_dual(self, state, target, target_step, weight, name):
        del target_step
        residual = _copy_function(state, f"{name}_residual")
        with residual.dat.vec as output, target.dat.vec_ro as truth:
            output.axpy(-1.0, truth)
        dual = self.helper.state_mass_map(residual, f"{name}_mass_dual")
        scale = float(weight) / self.denominator
        value = scale * float(self.helper.dual_pairing(dual, residual))
        with dual.dat.vec as vector:
            vector.scale(2.0 * scale)
        return float(value), dual

    def value(self, state, target, target_step, weight, name):
        return self.value_and_dual(
            state, target, target_step, weight, name
        )[0]


class NeuralTrajectoryObjective:
    """Serial exact map/reduce objective shared by Methods 3 and 4."""

    def __init__(
        self,
        case,
        truth_states,
        windows,
        *,
        metric=None,
        c0=0.14,
        use_fixed_prefix=True,
    ):
        self.case = case
        self.helper = case.helper
        self.c0 = float(c0)
        if not np.isfinite(self.c0) or self.c0 <= 0.0:
            raise ValueError("c0 must be positive and finite")
        self.windows = tuple(windows)
        if not self.windows:
            raise ValueError("trajectory objective requires at least one window")
        for window in self.windows:
            if not isinstance(window, NeuralWindowSpec):
                raise TypeError("windows must contain NeuralWindowSpec values")
        required = {
            int(window.start_step) for window in self.windows
        } | {
            step for window in self.windows for step in window.target_steps
        }
        if max(required) > 80:
            raise ValueError("states after 80 are locked")
        missing = sorted(required.difference(truth_states))
        if missing:
            raise KeyError(f"required training truth states are missing: {missing}")
        self.truth_states = {
            step: _copy_function(
                truth_states[step], f"test2a_trajectory_truth_{step}"
            )
            for step in sorted(required)
        }
        targets = {
            step: self.truth_states[step]
            for window in self.windows
            for step in window.target_steps
        }
        self.metric = metric or CertificationMixedMassMetric(self.helper, targets)
        self.use_fixed_prefix = bool(use_fixed_prefix)
        self._prefixes = {}
        if self.use_fixed_prefix:
            with self.case.physical_c0(self.c0):
                for start in sorted({int(window.start_step) for window in self.windows}):
                    self._prefixes[start] = self.helper.take_fixed_prefix_cached(
                        self.truth_states[start],
                        self.case.t0 + start * self.case.dt,
                        self.case.dt,
                    )
        problem_record = {
            "windows": [window.to_record() for window in self.windows],
            "metric": self.metric.record(),
            "c0": self.c0,
            "dt": float(self.case.dt),
            "truth_steps": sorted(self.truth_states),
            "fixed_prefix": self.use_fixed_prefix,
        }
        digest = sha256(json.dumps(problem_record, sort_keys=True).encode("utf-8"))
        for step in sorted(self.truth_states):
            digest.update(np.asarray([step], dtype=np.int64).tobytes())
            for field in self.truth_states[step].subfunctions:
                digest.update(
                    np.ascontiguousarray(field.dat.data_ro, dtype=np.float64).tobytes()
                )
        self.problem_sha256 = digest.hexdigest()
        self._last_tape = None
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.forward_complete_steps = 0
        self.reverse_complete_steps = 0
        self.fixed_prefix_builds = len(self._prefixes)
        self.fixed_prefix_uses = 0
        self.full_first_prefix_recomputations = 0
        self.same_theta_tape_hits = 0
        self.same_theta_tape_misses = 0
        self.tape_invalidations = 0

    def _forward_window(self, parameters, spec):
        states = [
            _copy_function(
                self.truth_states[int(spec.start_step)],
                f"test2a_{spec.label}_state_0",
            )
        ]
        caches = []
        with self.case.physical_c0(self.c0):
            if self.use_fixed_prefix:
                first = self.helper.take_forward_step_from_prefix(
                    self._prefixes[int(spec.start_step)], parameters
                )
                self.fixed_prefix_uses += 1
            else:
                first = self.helper.take_forward_step_cached(
                    states[0],
                    self.case.t0 + int(spec.start_step) * self.case.dt,
                    self.case.dt,
                    neural_parameters=parameters,
                )
                self.full_first_prefix_recomputations += 1
            caches.append(first)
            states.append(
                _copy_function(first.state_out, f"test2a_{spec.label}_state_1")
            )
            for offset in range(1, int(spec.horizon)):
                cache = self.helper.take_forward_step_cached(
                    states[-1],
                    self.case.t0
                    + (int(spec.start_step) + offset) * self.case.dt,
                    self.case.dt,
                    neural_parameters=parameters,
                )
                caches.append(cache)
                states.append(
                    _copy_function(
                        cache.state_out,
                        f"test2a_{spec.label}_state_{offset + 1}",
                    )
                )
        self.forward_complete_steps += int(spec.horizon)
        local_values = {}
        for offset, target_step, weight in zip(
            spec.target_offsets, spec.target_steps, spec.weights
        ):
            local_values[offset] = self.metric.value(
                states[offset],
                self.truth_states[target_step],
                target_step,
                weight,
                f"test2a_{spec.label}_target_{target_step}",
            )
        return NeuralWindowTape(
            spec=spec,
            states=tuple(states),
            step_caches=tuple(caches),
            used_fixed_prefix=self.use_fixed_prefix,
            local_values=MappingProxyType(local_values),
        )

    def _tape(self, parameters):
        owned = validate_float64_tree(parameters, name="parameters")
        fingerprint = parameter_pytree_sha256(owned)
        if (
            self._last_tape is not None
            and self._last_tape.parameter_sha256 == fingerprint
            and self._last_tape.problem_sha256 == self.problem_sha256
        ):
            self.same_theta_tape_hits += 1
            return self._last_tape
        if self._last_tape is not None:
            self.tape_invalidations += 1
        self.same_theta_tape_misses += 1
        windows = tuple(
            self._forward_window(owned, window) for window in self.windows
        )
        objective = float(
            sum(sum(tape.local_values.values()) for tape in windows)
        )
        tape = NeuralTrajectoryTape(
            parameter_sha256=fingerprint,
            problem_sha256=self.problem_sha256,
            windows=windows,
            objective_value=objective,
            estimated_owned_bytes=0,
        )
        tape = NeuralTrajectoryTape(
            parameter_sha256=tape.parameter_sha256,
            problem_sha256=tape.problem_sha256,
            windows=tape.windows,
            objective_value=tape.objective_value,
            estimated_owned_bytes=estimate_owned_bytes(tape.windows),
        )
        self._last_tape = tape
        return tape

    def value(self, parameters):
        self.value_evaluations += 1
        return float(self._tape(parameters).objective_value)

    @staticmethod
    def _add_duals(left, right, name):
        result = _copy_cofunction(left, name)
        with result.dat.vec as output, right.dat.vec_ro as increment:
            output.axpy(1.0, increment)
        return result

    def _zero_dual(self, name):
        zero = self.case.new_state(f"{name}_zero_state")
        zero.assign(0.0)
        return self.helper.state_mass_map(zero, f"{name}_zero_dual")

    def _gradient_window(self, parameters, tape):
        local_duals = {}
        for offset, target_step, weight in zip(
            tape.spec.target_offsets,
            tape.spec.target_steps,
            tape.spec.weights,
        ):
            _, local_duals[offset] = self.metric.value_and_dual(
                tape.states[offset],
                self.truth_states[target_step],
                target_step,
                weight,
                f"test2a_{tape.spec.label}_gradient_target_{target_step}",
            )
        current = self._zero_dual(f"test2a_{tape.spec.label}_reverse")
        gradient = tree_zeros(parameters)
        for step in range(int(tape.spec.horizon) - 1, -1, -1):
            target_offset = step + 1
            if target_offset in local_duals:
                current = self._add_duals(
                    current,
                    local_duals[target_offset],
                    f"test2a_{tape.spec.label}_adjoint_{target_offset}",
                )
            reverse = self.helper.take_neural_parameter_adjoint_step(
                tape.step_caches[step],
                current,
                stop_at_fixed_prefix=(step == 0 and tape.used_fixed_prefix),
            )
            gradient = tree_axpy(gradient, 1.0, reverse.parameter_adjoint)
            current = reverse.state_adjoint_in
            self.reverse_complete_steps += 1
        return gradient

    def value_and_gradient(self, parameters):
        self.gradient_evaluations += 1
        owned = validate_float64_tree(parameters, name="parameters")
        tape = self._tape(owned)
        gradient = tree_zeros(owned)
        # Deterministic serial map/reduce reference.  Each window owns its
        # truth reset and contributes independently at fixed theta.
        for window in tape.windows:
            gradient = tree_axpy(
                gradient, 1.0, self._gradient_window(owned, window)
            )
        return float(tape.objective_value), tree_copy(gradient)

    def gradient(self, parameters):
        return self.value_and_gradient(parameters)[1]

    def clear_parameter_tape(self):
        self._last_tape = None

    def work_counts(self):
        return NeuralTrajectoryWorkCounts(
            value_evaluations=self.value_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            forward_complete_steps=self.forward_complete_steps,
            reverse_complete_steps=self.reverse_complete_steps,
            fixed_prefix_builds=self.fixed_prefix_builds,
            fixed_prefix_uses=self.fixed_prefix_uses,
            full_first_prefix_recomputations=self.full_first_prefix_recomputations,
            same_theta_tape_hits=self.same_theta_tape_hits,
            same_theta_tape_misses=self.same_theta_tape_misses,
            tape_invalidations=self.tape_invalidations,
        )


class TrajectoryPyROLObjective(Objective):
    """PyROL-compatible exact-gradient adapter with same-theta tape reuse."""

    def __init__(self, trajectory_objective, initial_parameters, *, accepted_callback=None):
        super().__init__()
        self.trajectory_objective = trajectory_objective
        self.codec = PytreeVectorCodec(initial_parameters)
        self.accepted_callback = accepted_callback
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.accepted_updates = 0
        self.previous_accepted_control = None
        self.current_accepted_control = None

    def vector_from_pytree(self, parameters):
        return self.codec.vector_from_pytree(parameters)

    def pytree_from_vector(self, vector):
        return self.codec.pytree_from_vector(vector)

    def update(self, control, *args):
        update_type = str(args[0]) if args else "unspecified"
        if "Initial" not in update_type and "Accept" not in update_type:
            return
        values = np.asarray(
            self.codec.flat_from_vector(control, "control"), dtype=np.float64
        ).copy()
        self.previous_accepted_control = self.current_accepted_control
        self.current_accepted_control = values
        local_index = self.accepted_updates
        self.accepted_updates += 1
        if self.accepted_callback is not None:
            self.accepted_callback(control, local_index, self)

    def value(self, control, tolerance):
        del tolerance
        parameters = self.pytree_from_vector(control)
        result = self.trajectory_objective.value(parameters)
        self.value_evaluations += 1
        return float(result)

    def gradient(self, output, control, tolerance):
        del tolerance
        parameters = self.pytree_from_vector(control)
        _, gradient = self.trajectory_objective.value_and_gradient(parameters)
        flat, _ = ravel_pytree(gradient)
        output.array[:] = np.asarray(flat, dtype=np.float64)
        self.gradient_evaluations += 1

    def hessVec(self, output, direction, control, tolerance):
        del output, direction, control, tolerance
        self.hvp_evaluations += 1
        raise RuntimeError("trajectory L-BFGS plumbing must not request an HVP")


__all__ = (
    "CertificationMixedMassMetric",
    "GlobalMixedMassMetric",
    "NeuralTrajectoryObjective",
    "NeuralTrajectoryTape",
    "NeuralTrajectoryWorkCounts",
    "NeuralWindowSpec",
    "NeuralWindowTape",
    "TrajectoryLossMode",
    "TrajectoryPyROLObjective",
    "continuous_rollout",
    "estimate_owned_bytes",
    "reset_windows",
)
