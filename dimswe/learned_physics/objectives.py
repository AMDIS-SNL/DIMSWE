"""Explicit, semantically distinct learned-physics training objectives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import jax.numpy as jnp

from .parameters import tree_copy, tree_dot, validate_float64_tree


class TrainingMode(str, Enum):
    """The four J4A training semantics; values are stable record keys."""

    APRIORI_OFFLINE = "apriori_offline"
    DISCRETE_OFFLINE = "discrete_offline"
    TRUTH_RESET = "truth_reset"
    ROLLOUT = "rollout"


class LossAccumulation(str, Enum):
    """Whether a solver-window loss observes only its end or every step."""

    TERMINAL = "terminal"
    ACCUMULATED = "accumulated"


def squared_l2_loss(prediction, target):
    """Half squared Euclidean distance between matching float64 pytrees."""
    prediction_owned = validate_float64_tree(
        prediction, name="prediction"
    )
    target_owned = validate_float64_tree(target, name="target")
    difference = jax_tree_subtract(prediction_owned, target_owned)
    return jnp.float64(0.5) * tree_dot(difference, difference)


def jax_tree_subtract(left, right):
    """Subtract matching float64 trees while preserving their structure."""
    import jax

    left_owned = validate_float64_tree(left, name="left")
    right_owned = validate_float64_tree(right, name="right")
    if jax.tree_util.tree_structure(left_owned) != jax.tree_util.tree_structure(
        right_owned
    ):
        raise ValueError("pytree structures differ")
    return jax.tree_util.tree_map(
        lambda x, y: jnp.array(x - y, copy=True), left_owned, right_owned
    )


@dataclass(frozen=True)
class LocalOfflineExample:
    """Truth state plus an instantaneous/local physics target."""

    state: Any
    context: Any
    physics_target: Any

    def __post_init__(self):
        object.__setattr__(self, "state", tree_copy(self.state))
        object.__setattr__(self, "context", tree_copy(self.context))
        object.__setattr__(
            self, "physics_target", tree_copy(self.physics_target)
        )


@dataclass(frozen=True)
class DiscreteOfflineExample:
    """Truth state plus a target after a fixed deployed discrete map."""

    state: Any
    context: Any
    discrete_target: Any

    def __post_init__(self):
        object.__setattr__(self, "state", tree_copy(self.state))
        object.__setattr__(self, "context", tree_copy(self.context))
        object.__setattr__(
            self, "discrete_target", tree_copy(self.discrete_target)
        )


@dataclass(frozen=True)
class TruthResetWindow:
    """One trusted reset state and the targets inside its solver window."""

    initial_state: Any
    contexts: tuple[Any, ...]
    targets: tuple[Any, ...]

    def __post_init__(self):
        if not self.contexts or len(self.contexts) != len(self.targets):
            raise ValueError(
                "a truth-reset window needs one target per nonempty context"
            )
        object.__setattr__(
            self, "initial_state", tree_copy(self.initial_state)
        )
        object.__setattr__(
            self, "contexts", tuple(tree_copy(x) for x in self.contexts)
        )
        object.__setattr__(
            self, "targets", tuple(tree_copy(x) for x in self.targets)
        )


@dataclass(frozen=True)
class RolloutExample:
    """One trusted initial state and an autonomous target trajectory."""

    initial_state: Any
    contexts: tuple[Any, ...]
    targets: tuple[Any, ...]

    def __post_init__(self):
        if not self.contexts or len(self.contexts) != len(self.targets):
            raise ValueError(
                "a rollout needs one target per nonempty context"
            )
        object.__setattr__(
            self, "initial_state", tree_copy(self.initial_state)
        )
        object.__setattr__(
            self, "contexts", tuple(tree_copy(x) for x in self.contexts)
        )
        object.__setattr__(
            self, "targets", tuple(tree_copy(x) for x in self.targets)
        )


def _mean(losses):
    if not losses:
        raise ValueError("an objective requires at least one loss term")
    return sum(losses, start=jnp.float64(0.0)) / jnp.float64(len(losses))


def apriori_offline(
    parameters,
    examples: tuple[LocalOfflineExample, ...],
    predict_physics: Callable[[Any, Any, Any], Any],
    loss: Callable[[Any, Any], Any] = squared_l2_loss,
):
    """Compare local physics without applying a deployed state transition."""
    if not examples:
        raise ValueError("apriori_offline requires examples")
    values = []
    for example in examples:
        prediction = predict_physics(
            tree_copy(parameters),
            tree_copy(example.state),
            tree_copy(example.context),
        )
        values.append(loss(prediction, tree_copy(example.physics_target)))
    return _mean(values)


def discrete_offline(
    parameters,
    examples: tuple[DiscreteOfflineExample, ...],
    predict_physics: Callable[[Any, Any, Any], Any],
    discrete_map: Callable[[Any, Any, Any], Any],
    loss: Callable[[Any, Any], Any] = squared_l2_loss,
):
    """Apply a fixed deployed discrete map independently at truth states.

    No predicted state is passed to the next example.  This is therefore
    offline even when ``discrete_map`` contains an assembly/mass/update map.
    """
    if not examples:
        raise ValueError("discrete_offline requires examples")
    values = []
    for example in examples:
        truth_state = tree_copy(example.state)
        physics = predict_physics(
            tree_copy(parameters),
            tree_copy(truth_state),
            tree_copy(example.context),
        )
        prediction = discrete_map(
            tree_copy(truth_state),
            tree_copy(example.context),
            tree_copy(physics),
        )
        values.append(loss(prediction, tree_copy(example.discrete_target)))
    return _mean(values)


def _window_losses(
    parameters,
    initial_state,
    contexts,
    targets,
    transition,
    loss,
    accumulation,
):
    current = tree_copy(initial_state)
    values = []
    for context, target in zip(contexts, targets):
        current = transition(
            tree_copy(parameters), tree_copy(current), tree_copy(context)
        )
        if accumulation is LossAccumulation.ACCUMULATED:
            values.append(loss(current, tree_copy(target)))
    if accumulation is LossAccumulation.TERMINAL:
        values.append(loss(current, tree_copy(targets[-1])))
    return values


def truth_reset(
    parameters,
    windows: tuple[TruthResetWindow, ...],
    transition: Callable[[Any, Any, Any], Any],
    loss: Callable[[Any, Any], Any] = squared_l2_loss,
    *,
    accumulation: LossAccumulation = LossAccumulation.TERMINAL,
):
    """Run recursively inside each window, then reset the next window to truth."""
    if not windows:
        raise ValueError("truth_reset requires windows")
    accumulation = LossAccumulation(accumulation)
    values = []
    for window in windows:
        # This assignment is the semantic reset; no previous prediction enters.
        values.extend(
            _window_losses(
                parameters,
                window.initial_state,
                window.contexts,
                window.targets,
                transition,
                loss,
                accumulation,
            )
        )
    return _mean(values)


def rollout(
    parameters,
    example: RolloutExample,
    transition: Callable[[Any, Any, Any], Any],
    loss: Callable[[Any, Any], Any] = squared_l2_loss,
    *,
    accumulation: LossAccumulation = LossAccumulation.ACCUMULATED,
):
    """Run one autonomous trajectory with no truth reset after its start."""
    accumulation = LossAccumulation(accumulation)
    values = _window_losses(
        parameters,
        example.initial_state,
        example.contexts,
        example.targets,
        transition,
        loss,
        accumulation,
    )
    return _mean(values)


def objective_for_mode(mode: TrainingMode, parameters, **kwargs):
    """Dispatch records by stable mode while retaining four explicit APIs."""
    selected = TrainingMode(mode)
    if selected is TrainingMode.APRIORI_OFFLINE:
        return apriori_offline(parameters, **kwargs)
    if selected is TrainingMode.DISCRETE_OFFLINE:
        return discrete_offline(parameters, **kwargs)
    if selected is TrainingMode.TRUTH_RESET:
        return truth_reset(parameters, **kwargs)
    if selected is TrainingMode.ROLLOUT:
        return rollout(parameters, **kwargs)
    raise AssertionError("unreachable training mode")


__all__ = (
    "DiscreteOfflineExample",
    "LocalOfflineExample",
    "LossAccumulation",
    "RolloutExample",
    "TrainingMode",
    "TruthResetWindow",
    "apriori_offline",
    "discrete_offline",
    "objective_for_mode",
    "rollout",
    "squared_l2_loss",
    "truth_reset",
)
