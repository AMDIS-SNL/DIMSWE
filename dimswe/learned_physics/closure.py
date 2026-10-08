"""Pure-JAX composition of features, parameters, and physics output maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .parameters import tree_copy, validate_float64_tree


@runtime_checkable
class FeatureMap(Protocol):
    """Map an owned numerical state and context to numerical features."""

    def __call__(self, state: Any, context: Any) -> Any: ...


@runtime_checkable
class ParameterizedModel(Protocol):
    """Map an arbitrary parameter pytree and features to raw output."""

    def __call__(self, parameters: Any, features: Any) -> Any: ...


@runtime_checkable
class OutputMap(Protocol):
    """Turn raw model output into the physics consumed by deployment."""

    def __call__(
        self,
        state: Any,
        context: Any,
        baseline_physics: Any,
        raw_output: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class LearnedPhysicsModel:
    """Model-agnostic ``Phi -> N_theta -> D`` composition.

    Every numerical input is validated and copied before it crosses a user
    callback.  Numerical outputs are copied as well.  Callbacks therefore own
    only disposable containers and cannot mutate caller-owned state, context,
    baseline physics, or parameters.  Firedrake objects are outside this
    pure-JAX boundary and must be handled by a deployment adapter.
    """

    feature_map: FeatureMap
    model: ParameterizedModel
    output_map: OutputMap
    name: str = "learned_physics"

    def __post_init__(self):
        for attribute in ("feature_map", "model", "output_map"):
            if not callable(getattr(self, attribute)):
                raise TypeError(f"{attribute} must be callable")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")

    def __call__(
        self,
        parameters,
        state,
        context=None,
        baseline_physics=None,
    ):
        owned_parameters = validate_float64_tree(
            parameters, name="parameters"
        )
        owned_state = validate_float64_tree(state, name="state")
        owned_context = validate_float64_tree(context, name="context")
        owned_baseline = validate_float64_tree(
            baseline_physics, name="baseline_physics"
        )
        features = self.feature_map(
            tree_copy(owned_state), tree_copy(owned_context)
        )
        features = validate_float64_tree(features, name="features")
        raw_output = self.model(
            tree_copy(owned_parameters), tree_copy(features)
        )
        raw_output = validate_float64_tree(raw_output, name="raw_output")
        physics = self.output_map(
            tree_copy(owned_state),
            tree_copy(owned_context),
            tree_copy(owned_baseline),
            tree_copy(raw_output),
        )
        return validate_float64_tree(physics, name="learned_physics")

    apply = __call__


__all__ = (
    "FeatureMap",
    "LearnedPhysicsModel",
    "OutputMap",
    "ParameterizedModel",
)
