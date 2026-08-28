"""Deterministic contracts and records for learned-physics experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .objectives import TrainingMode


TRUTH_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 1


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, Integral, Real)):
        return value
    if isinstance(value, np.ndarray):
        return tuple(_freeze_json(item) for item in value.tolist())
    raise TypeError(f"value of type {type(value).__name__} is not JSON data")


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, Integral, Real)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not serializable")


def canonical_json(value) -> str:
    """Return the stable JSON representation used by configs and metadata."""
    return json.dumps(
        _jsonable(value), sort_keys=True, indent=2, allow_nan=False
    ) + "\n"


class LearnedPhysicsRole(str, Enum):
    """How learned output relates to the deployed physical formulation."""

    PARAMETER_INFERENCE = "parameter_inference"
    REPLACEMENT = "replacement"
    CORRECTION = "correction"


@dataclass(frozen=True)
class BenchmarkContract:
    """Extension contract without selecting features or an architecture."""

    name: str
    truth_role: str
    baseline_role: str
    learned_role: LearnedPhysicsRole
    configurable_representations: tuple[str, ...]
    configurable_formulation_fields: tuple[str, ...]
    training_modes: tuple[TrainingMode, ...] = tuple(TrainingMode)

    def to_dict(self):
        return _jsonable(self)


BENCHMARK_CONTRACTS = MappingProxyType(
    {
        "hidden_c0": BenchmarkContract(
            name="hidden_c0",
            truth_role="production DIMSWE with hidden physical c0",
            baseline_role="identical production DIMSWE with unknown c0",
            learned_role=LearnedPhysicsRole.PARAMETER_INFERENCE,
            configurable_representations=("physical_c0", "normalized_c0"),
            configurable_formulation_fields=(),
        ),
        "learned_moist_replacement": BenchmarkContract(
            name="learned_moist_replacement",
            truth_role="PDE plus certified original moist physics",
            baseline_role="PDE with moist closure replaced through OutputMap",
            learned_role=LearnedPhysicsRole.REPLACEMENT,
            configurable_representations=(
                "direct_rates",
                "subprocess_rates",
                "invariant_null_space_sources",
                "other_output_map",
            ),
            configurable_formulation_fields=(
                "feature_map",
                "architecture",
                "normalization",
                "output_parameterization",
            ),
        ),
        "misspecified_moist_correction": BenchmarkContract(
            name="misspecified_moist_correction",
            truth_role="PDE plus reference moist formulation",
            baseline_role="PDE plus configurable nearby perturbed formulation",
            learned_role=LearnedPhysicsRole.CORRECTION,
            configurable_representations=(
                "residual_rates",
                "residual_subprocess_rates",
                "physical_parameter_correction",
                "other_correction_map",
            ),
            configurable_formulation_fields=(
                "gamma_r_multiplier",
                "qprecip_perturbation",
                "relaxation_times",
                "saturation_parameters",
                "rain_subprocess",
                "evaporation_cap",
            ),
        ),
    }
)


@dataclass(frozen=True)
class ExperimentDefinition:
    """Complete, immutable experiment intent separate from generated data."""

    benchmark: str
    truth_configuration: Mapping[str, Any]
    baseline_configuration: Mapping[str, Any]
    model_configuration: Mapping[str, Any]
    training_mode: TrainingMode
    observation_definition: Mapping[str, Any]
    rollout_horizon: int
    seed: int
    optimizer_configuration: Mapping[str, Any]
    evaluation_metrics: tuple[str, ...]

    def __post_init__(self):
        if self.benchmark not in BENCHMARK_CONTRACTS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}")
        object.__setattr__(self, "training_mode", TrainingMode(self.training_mode))
        if not isinstance(self.rollout_horizon, Integral) or isinstance(
            self.rollout_horizon, bool
        ) or int(self.rollout_horizon) < 1:
            raise ValueError("rollout_horizon must be a positive integer")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        for name in (
            "truth_configuration",
            "baseline_configuration",
            "model_configuration",
            "observation_definition",
            "optimizer_configuration",
        ):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))
        object.__setattr__(
            self, "evaluation_metrics", tuple(str(x) for x in self.evaluation_metrics)
        )

    def to_dict(self):
        return _jsonable(self)

    def to_json(self):
        return canonical_json(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))


@dataclass(frozen=True)
class TruthMetadata:
    """Required provenance for one stored DIMSWE truth trajectory."""

    benchmark: str
    solver_backend: str
    timestep: float
    num_steps: int
    initial_condition: Mapping[str, Any]
    physical_parameters: Mapping[str, Any]
    truth_c0: float
    moist_backend: str
    random_seed: int
    state_control_convention: Mapping[str, Any]
    solver_configuration: Mapping[str, Any]
    format_version: int = TRUTH_FORMAT_VERSION

    def __post_init__(self):
        if self.benchmark != "hidden_c0":
            raise ValueError("J4A TruthMetadata currently stores hidden_c0 data")
        if float(self.timestep) <= 0.0 or not np.isfinite(self.timestep):
            raise ValueError("timestep must be positive and finite")
        if int(self.num_steps) < 1:
            raise ValueError("num_steps must be positive")
        if not np.isfinite(self.truth_c0):
            raise ValueError("truth_c0 must be finite")
        for name in (
            "initial_condition",
            "physical_parameters",
            "state_control_convention",
            "solver_configuration",
        ):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))

    def to_dict(self):
        return _jsonable(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))


@dataclass(frozen=True)
class TruthDataset:
    """Owned, read-only dense snapshots plus deterministic provenance."""

    states: np.ndarray
    times: np.ndarray
    metadata: TruthMetadata

    def __post_init__(self):
        states = np.array(self.states, dtype=np.float64, copy=True)
        times = np.array(self.times, dtype=np.float64, copy=True)
        if states.ndim != 2:
            raise ValueError("states must have shape (num_steps + 1, state_size)")
        if times.shape != (states.shape[0],):
            raise ValueError("times must contain one entry per state snapshot")
        if states.shape[0] != self.metadata.num_steps + 1:
            raise ValueError("snapshot count disagrees with metadata num_steps")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(times)):
            raise FloatingPointError("truth data must be finite")
        states.setflags(write=False)
        times.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "times", times)


def _truth_paths(path):
    base = Path(path)
    if base.suffix in (".npz", ".json"):
        base = base.with_suffix("")
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_truth_dataset(dataset: TruthDataset, path):
    """Write one transparent NPZ payload and one canonical JSON sidecar."""
    if not isinstance(dataset, TruthDataset):
        raise TypeError("dataset must be a TruthDataset")
    npz_path, json_path = _truth_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, states=dataset.states, times=dataset.times)
    json_path.write_text(canonical_json(dataset.metadata), encoding="utf-8")
    return npz_path, json_path


def load_truth_dataset(path):
    """Load and validate a dataset written by :func:`save_truth_dataset`."""
    npz_path, json_path = _truth_paths(path)
    metadata = TruthMetadata.from_dict(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"states", "times"}:
            raise ValueError("truth NPZ must contain exactly states and times")
        states = np.array(payload["states"], copy=True)
        times = np.array(payload["times"], copy=True)
    return TruthDataset(states=states, times=times, metadata=metadata)


@dataclass(frozen=True)
class ExperimentResult:
    """Machine-readable outcome and cost accounting for one fit."""

    benchmark: str
    training_mode: TrainingMode
    seed: int
    truth_configuration: Mapping[str, Any]
    baseline_configuration: Mapping[str, Any]
    model_configuration: Mapping[str, Any]
    initial_parameters: Any
    final_parameters: Any
    objective_history: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    objective_evaluations: int
    gradient_evaluations: int
    hvp_evaluations: int
    solver_calls: int
    timing: Mapping[str, Any]
    deployment_evaluation_metrics: Mapping[str, Any]
    success: bool
    failure_reason: str | None = None
    format_version: int = RESULT_FORMAT_VERSION

    def __post_init__(self):
        object.__setattr__(self, "training_mode", TrainingMode(self.training_mode))
        for name in (
            "truth_configuration",
            "baseline_configuration",
            "model_configuration",
            "initial_parameters",
            "final_parameters",
            "timing",
            "deployment_evaluation_metrics",
        ):
            object.__setattr__(self, name, _freeze_json(_jsonable(getattr(self, name))))
        object.__setattr__(
            self, "objective_history", tuple(float(x) for x in self.objective_history)
        )
        object.__setattr__(
            self, "gradient_norms", tuple(float(x) for x in self.gradient_norms)
        )
        if self.success and self.failure_reason is not None:
            raise ValueError("successful results cannot have a failure reason")
        if not self.success and not self.failure_reason:
            raise ValueError("failed results require a failure reason")

    def to_dict(self):
        return _jsonable(self)

    def to_json(self):
        return canonical_json(self)


def save_experiment_result(result: ExperimentResult, path):
    """Write a result as canonical JSON; plotting remains a separate concern."""
    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_json(), encoding="utf-8")
    return target


def summarize_experiment_result(result: ExperimentResult):
    """Return a compact JSON-safe report without coupling optimization to plots."""
    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")
    history = result.objective_history
    return {
        "benchmark": result.benchmark,
        "training_mode": result.training_mode.value,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "initial_parameters": _jsonable(result.initial_parameters),
        "final_parameters": _jsonable(result.final_parameters),
        "initial_objective": history[0] if history else None,
        "final_objective": history[-1] if history else None,
        "objective_evaluations": result.objective_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "hvp_evaluations": result.hvp_evaluations,
        "solver_calls": result.solver_calls,
        "timing": _jsonable(result.timing),
    }


__all__ = (
    "BENCHMARK_CONTRACTS",
    "BenchmarkContract",
    "ExperimentDefinition",
    "ExperimentResult",
    "LearnedPhysicsRole",
    "TruthDataset",
    "TruthMetadata",
    "canonical_json",
    "load_truth_dataset",
    "save_experiment_result",
    "save_truth_dataset",
    "summarize_experiment_result",
)
