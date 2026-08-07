"""Cached accumulated solver objectives for resolved hidden-``c0`` studies.

The production adapter in this module composes the certified complete-step
forward, tangent, adjoint, and incremental-adjoint maps.  It changes neither
the deployed split nor its discrete derivatives.  The generic operation
boundary also permits tiny algebraic equivalence tests without a resolved run.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .hidden_c0 import (
    HiddenC0Objective,
    _copy_function,
    _finite_float,
    _state_squared_difference,
)
from .hyperviscosity_hvp import _copy_cofunction
from .learned_physics.objectives import TrainingMode


@dataclass(frozen=True)
class AccumulatedTrajectoryTarget:
    """One fixed target attached to a model-step offset in a window."""

    offset: int
    target: object
    normalizer: float
    target_step: int


@dataclass(frozen=True)
class AccumulatedTrajectoryWindow:
    """One truth-started trajectory with one or more accumulated targets."""

    initial_state: object
    start_time: float
    targets: tuple[AccumulatedTrajectoryTarget, ...]


@dataclass(frozen=True)
class TrajectoryWorkCounts:
    """Intrinsic complete-step traversals, separate from API evaluations."""

    forward_steps: int
    reverse_steps: int
    tangent_steps: int
    incremental_reverse_steps: int


class ProductionAccumulatedTrajectoryOperations:
    """Dual-native operations backed by the certified production split helper."""

    def __init__(self, case):
        self.case = case
        self.helper = case.helper

    def parameter_context(self, physical_c0):
        return self.case.physical_c0(physical_c0)

    @staticmethod
    def copy_state(value, name):
        return _copy_function(value, name)

    @staticmethod
    def copy_dual(value, name):
        return _copy_cofunction(value, name)

    def zero_state(self, name):
        result = self.case.new_state(name)
        result.assign(0)
        return result

    def forward_step(self, state, time, dt):
        cache = self.helper.take_forward_step_cached(state, time, dt)
        return cache, cache.state_out

    def tangent_step(self, primal_cache, state_direction, physical_direction):
        cache = self.helper.take_tangent_step(
            primal_cache, state_direction, physical_direction
        )
        return cache, cache.state_direction_out

    def reverse_step(self, primal_cache, state_adjoint_out):
        result = self.helper.take_adjoint_step_cached(
            primal_cache, state_adjoint_out
        )
        return (
            result.state_adjoint_in,
            float(result.physical_c0_gradient),
        )

    def incremental_reverse_step(
        self, tangent_cache, state_adjoint_out, incremental_adjoint_out
    ):
        result = self.helper.take_incremental_adjoint_step(
            tangent_cache, state_adjoint_out, incremental_adjoint_out
        )
        return (
            result.ordinary.state_adjoint_in,
            result.incremental_state_adjoint_in,
            float(result.ordinary.physical_c0_gradient),
            float(result.physical_c0_hvp),
        )

    @staticmethod
    def _scale_dual(value, scale, name):
        result = _copy_cofunction(value, name)
        with result.dat.vec as vector:
            vector.scale(float(scale))
        return result

    @staticmethod
    def add_duals(left, right, name):
        result = _copy_cofunction(left, name)
        with result.dat.vec as result_vector, right.dat.vec_ro as right_vector:
            result_vector.axpy(1.0, right_vector)
        return result

    def local_loss(self, state, target, scale, name):
        squared = _state_squared_difference(
            self.case, state, target, f"{name}_residual_norm"
        )
        residual = _copy_function(state, f"{name}_residual")
        with residual.dat.vec as residual_vector, target.dat.vec_ro as target_vector:
            residual_vector.axpy(-1.0, target_vector)
        dual = self.helper.state_mass_map(residual, f"{name}_state_dual")
        return (
            0.5 * float(scale) * squared,
            self._scale_dual(dual, scale, f"{name}_scaled_state_dual"),
        )

    def local_loss_value(self, state, target, scale, name):
        squared = _state_squared_difference(
            self.case, state, target, f"{name}_residual_norm"
        )
        return 0.5 * float(scale) * squared

    def local_loss_dual(self, state, target, scale, name):
        residual = _copy_function(state, f"{name}_residual")
        with residual.dat.vec as residual_vector, target.dat.vec_ro as target_vector:
            residual_vector.axpy(-1.0, target_vector)
        dual = self.helper.state_mass_map(residual, f"{name}_state_dual")
        return self._scale_dual(dual, scale, f"{name}_scaled_state_dual")

    def local_loss_hessian_action(self, state_direction, scale, name):
        dual = self.helper.state_mass_map(
            state_direction, f"{name}_incremental_state_dual"
        )
        return self._scale_dual(
            dual, scale, f"{name}_scaled_incremental_state_dual"
        )


class CachedAccumulatedC0Objective(HiddenC0Objective):
    """One-forward/one-reverse accumulated objective over explicit windows.

    The operations object acts on primal states and state duals.  Production
    use supplies :class:`ProductionAccumulatedTrajectoryOperations`; tests may
    supply exact scalar operations implementing the same contract.
    """

    def __init__(self, mode, case, windows, *, operations=None):
        if mode not in (TrainingMode.TRUTH_RESET, TrainingMode.ROLLOUT):
            raise ValueError("cached trajectory objective requires reset or rollout")
        super().__init__(mode, case.c0_scale)
        self.case = case
        self.operations = operations or ProductionAccumulatedTrajectoryOperations(
            case
        )
        self.windows = self._own_windows(windows)
        if not self.windows:
            raise ValueError("cached trajectory objective requires windows")
        self.target_count = sum(len(window.targets) for window in self.windows)
        if self.target_count < 1:
            raise ValueError("cached trajectory objective requires targets")
        self.forward_model_steps = 0
        self.reverse_model_steps = 0
        self.tangent_model_steps = 0
        self.incremental_reverse_model_steps = 0

    def _own_windows(self, windows):
        owned = []
        for window_index, window in enumerate(tuple(windows)):
            if not isinstance(window, AccumulatedTrajectoryWindow):
                raise TypeError("windows must contain AccumulatedTrajectoryWindow")
            targets = []
            seen_offsets = set()
            previous = 0
            for target_index, target in enumerate(tuple(window.targets)):
                if not isinstance(target, AccumulatedTrajectoryTarget):
                    raise TypeError(
                        "window targets must be AccumulatedTrajectoryTarget"
                    )
                if not isinstance(target.offset, Integral) or isinstance(
                    target.offset, bool
                ):
                    raise TypeError("target offset must be an integer")
                offset = int(target.offset)
                if offset < 1 or offset <= previous or offset in seen_offsets:
                    raise ValueError(
                        "target offsets must be positive, unique, and increasing"
                    )
                normalizer = _finite_float(
                    "normalizer", target.normalizer, positive=True
                )
                if not isinstance(target.target_step, Integral) or isinstance(
                    target.target_step, bool
                ):
                    raise TypeError("target_step must be an integer")
                targets.append(
                    AccumulatedTrajectoryTarget(
                        offset=offset,
                        target=self.operations.copy_state(
                            target.target,
                            f"cached_{self.mode.value}_target_"
                            f"{window_index}_{target_index}",
                        ),
                        normalizer=normalizer,
                        target_step=int(target.target_step),
                    )
                )
                previous = offset
                seen_offsets.add(offset)
            if not targets:
                raise ValueError("each cached trajectory window needs a target")
            if not isinstance(window.start_time, Real):
                raise TypeError("window start_time must be real")
            start_time = float(window.start_time)
            if not np.isfinite(start_time):
                raise ValueError("window start_time must be finite")
            owned.append(
                AccumulatedTrajectoryWindow(
                    initial_state=self.operations.copy_state(
                        window.initial_state,
                        f"cached_{self.mode.value}_initial_{window_index}",
                    ),
                    start_time=start_time,
                    targets=tuple(targets),
                )
            )
        return tuple(owned)

    @property
    def target_steps(self):
        return tuple(
            target.target_step
            for window in self.windows
            for target in window.targets
        )

    def work_counts(self):
        return TrajectoryWorkCounts(
            forward_steps=self.forward_model_steps,
            reverse_steps=self.reverse_model_steps,
            tangent_steps=self.tangent_model_steps,
            incremental_reverse_steps=self.incremental_reverse_model_steps,
        )

    def _physical_c0(self, normalized_z):
        return self.c0_scale * _finite_float("normalized_z", normalized_z)

    def _target_scale(self, target):
        return 1.0 / (self.target_count * target.normalizer)

    def _forward_window(self, window):
        nsteps = window.targets[-1].offset
        states = [
            self.operations.copy_state(
                window.initial_state, f"cached_{self.mode.value}_state_0"
            )
        ]
        caches = []
        current = states[0]
        for step in range(nsteps):
            cache, state_out = self.operations.forward_step(
                current, window.start_time + step * self.case.dt, self.case.dt
            )
            caches.append(cache)
            current = self.operations.copy_state(
                state_out, f"cached_{self.mode.value}_state_{step + 1}"
            )
            states.append(current)
        self.forward_model_steps += nsteps
        self.solver_calls += nsteps
        return tuple(states), tuple(caches)

    def _local_losses(self, window, states):
        values = {}
        duals = {}
        for target in window.targets:
            value, dual = self.operations.local_loss(
                states[target.offset],
                target.target,
                self._target_scale(target),
                f"cached_{self.mode.value}_local_{target.target_step}",
            )
            values[target.offset] = float(value)
            duals[target.offset] = dual
        return values, duals

    def _local_loss_duals(self, window, states):
        return {
            target.offset: self.operations.local_loss_dual(
                states[target.offset],
                target.target,
                self._target_scale(target),
                f"cached_{self.mode.value}_local_dual_{target.target_step}",
            )
            for target in window.targets
        }

    def _value_window(self, window):
        states, _ = self._forward_window(window)
        return float(
            sum(
                self.operations.local_loss_value(
                    states[target.offset],
                    target.target,
                    self._target_scale(target),
                    f"cached_{self.mode.value}_value_{target.target_step}",
                )
                for target in window.targets
            )
        )

    def _gradient_window(self, window):
        states, caches = self._forward_window(window)
        values, local_duals = self._local_losses(window, states)
        current = self.operations.copy_dual(
            local_duals[len(caches)],
            f"cached_{self.mode.value}_terminal_adjoint",
        )
        physical_gradient = 0.0
        for step in range(len(caches) - 1, -1, -1):
            current, contribution = self.operations.reverse_step(
                caches[step], current
            )
            self.reverse_model_steps += 1
            physical_gradient += float(contribution)
            if step in local_duals:
                current = self.operations.add_duals(
                    current,
                    local_duals[step],
                    f"cached_{self.mode.value}_adjoint_with_local_{step}",
                )
        return float(sum(values.values())), physical_gradient

    def _hvp_window(self, window, physical_direction):
        nsteps = window.targets[-1].offset
        states = [
            self.operations.copy_state(
                window.initial_state, f"cached_{self.mode.value}_hvp_state_0"
            )
        ]
        directions = [
            self.operations.zero_state(
                f"cached_{self.mode.value}_hvp_direction_0"
            )
        ]
        tangents = []
        current_state = states[0]
        current_direction = directions[0]
        for step in range(nsteps):
            primal, state_out = self.operations.forward_step(
                current_state,
                window.start_time + step * self.case.dt,
                self.case.dt,
            )
            tangent, direction_out = self.operations.tangent_step(
                primal, current_direction, physical_direction
            )
            current_state = self.operations.copy_state(
                state_out, f"cached_{self.mode.value}_hvp_state_{step + 1}"
            )
            current_direction = self.operations.copy_state(
                direction_out,
                f"cached_{self.mode.value}_hvp_direction_{step + 1}",
            )
            states.append(current_state)
            directions.append(current_direction)
            tangents.append(tangent)
        self.forward_model_steps += nsteps
        self.tangent_model_steps += nsteps
        self.solver_calls += nsteps

        local_duals = self._local_loss_duals(window, states)
        incremental_duals = {
            target.offset: self.operations.local_loss_hessian_action(
                directions[target.offset],
                self._target_scale(target),
                f"cached_{self.mode.value}_incremental_local_"
                f"{target.target_step}",
            )
            for target in window.targets
        }
        current = self.operations.copy_dual(
            local_duals[nsteps],
            f"cached_{self.mode.value}_hvp_terminal_adjoint",
        )
        current_incremental = self.operations.copy_dual(
            incremental_duals[nsteps],
            f"cached_{self.mode.value}_hvp_terminal_incremental_adjoint",
        )
        physical_hvp = 0.0
        for step in range(nsteps - 1, -1, -1):
            (
                current,
                current_incremental,
                _,
                contribution,
            ) = self.operations.incremental_reverse_step(
                tangents[step], current, current_incremental
            )
            self.incremental_reverse_model_steps += 1
            physical_hvp += float(contribution)
            if step in local_duals:
                current = self.operations.add_duals(
                    current,
                    local_duals[step],
                    f"cached_{self.mode.value}_hvp_adjoint_with_local_{step}",
                )
                current_incremental = self.operations.add_duals(
                    current_incremental,
                    incremental_duals[step],
                    f"cached_{self.mode.value}_hvp_incremental_with_local_{step}",
                )
        return physical_hvp

    def value(self, normalized_z):
        physical_c0 = self._physical_c0(normalized_z)
        self.objective_evaluations += 1
        with self.operations.parameter_context(physical_c0):
            return float(sum(self._value_window(window) for window in self.windows))

    def value_and_gradient(self, normalized_z):
        physical_c0 = self._physical_c0(normalized_z)
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        with self.operations.parameter_context(physical_c0):
            pairs = tuple(self._gradient_window(window) for window in self.windows)
        return (
            float(sum(value for value, _ in pairs)),
            self.c0_scale * float(sum(gradient for _, gradient in pairs)),
        )

    def gradient(self, normalized_z):
        physical_c0 = self._physical_c0(normalized_z)
        self.gradient_evaluations += 1
        with self.operations.parameter_context(physical_c0):
            gradients = tuple(
                self._gradient_window(window)[1] for window in self.windows
            )
        return self.c0_scale * float(sum(gradients))

    def hess_vec(self, normalized_z, direction):
        physical_c0 = self._physical_c0(normalized_z)
        normalized_direction = _finite_float("direction", direction)
        physical_direction = self.c0_scale * normalized_direction
        self.hvp_evaluations += 1
        with self.operations.parameter_context(physical_c0):
            values = tuple(
                self._hvp_window(window, physical_direction)
                for window in self.windows
            )
        return self.c0_scale * float(sum(values))


class PrefixAccumulatedC0ObjectiveOracle(CachedAccumulatedC0Objective):
    """Redundant prefix definition retained only for tiny equivalence tests."""

    def __init__(self, mode, case, windows, *, operations=None):
        prefix_windows = tuple(
            AccumulatedTrajectoryWindow(
                initial_state=window.initial_state,
                start_time=window.start_time,
                targets=(target,),
            )
            for window in tuple(windows)
            for target in window.targets
        )
        super().__init__(mode, case, prefix_windows, operations=operations)


__all__ = (
    "AccumulatedTrajectoryTarget",
    "AccumulatedTrajectoryWindow",
    "CachedAccumulatedC0Objective",
    "PrefixAccumulatedC0ObjectiveOracle",
    "ProductionAccumulatedTrajectoryOperations",
    "TrajectoryWorkCounts",
)
