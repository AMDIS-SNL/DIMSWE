"""Test 2A-1 local neural replacement of the deployed moist rate ``A``.

This module is opt-in and keeps its numerical core Firedrake-free.  The
``prepare-data`` command imports Firedrake lazily, reads only existing truth
states 0 through 80, and reuses :class:`JAXMoistEulerPrimal` for the exact
cell-local 4x4 GLL representation and certified target rate.  It never
advances the PDE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .learned_physics import LearnedPhysicsModel
from .learned_physics.parameters import validate_float64_tree
from .resolved_hidden_c0 import write_json_record


FEATURE_ORDER = ("h", "S", "Qv", "Qc", "B")
TRAINING_STEPS = tuple(range(81))
POINTS_PER_CELL = 16
SELECTED_NX = 16
SELECTED_NY = 16
SELECTED_SAMPLE_COUNT = 331_776
_FLOAT64_EPSILON = np.finfo(np.float64).eps


def _require_x64():
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("Test 2A requires JAX_ENABLE_X64=True")


def _finite_float64(name, values, *, ndim=None):
    array = np.asarray(values)
    if array.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64, got {array.dtype}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.ndim}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def deployed_sample_count(states, nx, ny, points_per_cell=POINTS_PER_CELL):
    values = (states, nx, ny, points_per_cell)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in values
    ):
        raise ValueError("sample-count dimensions must be positive integers")
    return states * nx * ny * points_per_cell


def require_training_steps(steps):
    actual = tuple(int(step) for step in steps)
    if actual != TRAINING_STEPS:
        raise ValueError(
            "Test 2A-1 data must contain exactly states 0..80 in order; "
            "states after 80 are forbidden"
        )
    return actual


@dataclass(frozen=True)
class MLPConfiguration:
    """Configurable dense scalar-output MLP description."""

    input_dimension: int = 5
    hidden_layers: tuple[int, ...] = (32, 32)
    output_dimension: int = 1
    activation: str = "tanh"
    dtype: str = "float64"
    seed: int = 0

    def __post_init__(self):
        dimensions = (
            self.input_dimension,
            *self.hidden_layers,
            self.output_dimension,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in dimensions
        ):
            raise ValueError("all MLP layer dimensions must be positive integers")
        if self.input_dimension != len(FEATURE_ORDER):
            raise ValueError("Test 2A input dimension must match (h,S,Qv,Qc,B)")
        if self.output_dimension != 1:
            raise ValueError("Test 2A predicts exactly one scalar A")
        if self.activation not in ("tanh", "relu", "gelu"):
            raise ValueError("activation must be tanh, relu, or gelu")
        if self.dtype != "float64":
            raise TypeError("Test 2A MLP dtype must be float64")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("MLP seed must be an integer")

    @property
    def layer_dimensions(self):
        return (
            self.input_dimension,
            *self.hidden_layers,
            self.output_dimension,
        )

    @property
    def parameter_count(self):
        return int(
            sum(
                fan_in * fan_out + fan_out
                for fan_in, fan_out in zip(
                    self.layer_dimensions[:-1], self.layer_dimensions[1:]
                )
            )
        )

    def to_record(self):
        return {
            "input_dimension": self.input_dimension,
            "hidden_layers": list(self.hidden_layers),
            "output_dimension": self.output_dimension,
            "activation": self.activation,
            "dtype": self.dtype,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
        }


def mlp_configuration_from_record(record):
    return MLPConfiguration(
        input_dimension=int(record["input_dimension"]),
        hidden_layers=tuple(int(value) for value in record["hidden_layers"]),
        output_dimension=int(record["output_dimension"]),
        activation=str(record["activation"]),
        dtype=str(record["dtype"]),
        seed=int(record["seed"]),
    )


def initialize_mlp(configuration: MLPConfiguration):
    """Return deterministic x64 Glorot-uniform parameters without A encoding."""
    _require_x64()
    keys = jax.random.split(
        jax.random.PRNGKey(configuration.seed),
        len(configuration.layer_dimensions) - 1,
    )
    layers = []
    for key, fan_in, fan_out in zip(
        keys,
        configuration.layer_dimensions[:-1],
        configuration.layer_dimensions[1:],
    ):
        limit = np.sqrt(6.0 / float(fan_in + fan_out))
        layers.append(
            {
                "weight": jax.random.uniform(
                    key,
                    (fan_in, fan_out),
                    minval=-limit,
                    maxval=limit,
                    dtype=jnp.float64,
                ),
                "bias": jnp.zeros((fan_out,), dtype=jnp.float64),
            }
        )
    parameters = {"layers": tuple(layers)}
    owned = validate_float64_tree(parameters, name="MLP parameters")
    count = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(owned))
    if count != configuration.parameter_count:
        raise AssertionError("MLP parameter-count formula disagrees with pytree")
    return owned


@dataclass(frozen=True)
class DenseMLP:
    """J4A-compatible functional dense model with configurable depth/width."""

    configuration: MLPConfiguration

    def __call__(self, parameters, features):
        values = jnp.asarray(features, dtype=jnp.float64)
        activation = {
            "tanh": jnp.tanh,
            "relu": jax.nn.relu,
            "gelu": jax.nn.gelu,
        }[self.configuration.activation]
        layers = parameters["layers"]
        if len(layers) != len(self.configuration.layer_dimensions) - 1:
            raise ValueError("parameter pytree has the wrong number of layers")
        for index, layer in enumerate(layers):
            values = values @ layer["weight"] + layer["bias"]
            if index + 1 != len(layers):
                values = activation(values)
        return values


@dataclass(frozen=True)
class NormalizationMetadata:
    """Training-only affine inputs and uncentered positive output scale."""

    feature_order: tuple[str, ...]
    input_offset: np.ndarray
    input_scale: np.ndarray
    input_zero_scale: tuple[bool, ...]
    output_scale: float
    output_scale_method: str
    output_candidates: Mapping[str, float]
    output_zero_scale: bool
    fitted_state_first: int = 0
    fitted_state_last: int = 80

    def __post_init__(self):
        if tuple(self.feature_order) != FEATURE_ORDER:
            raise ValueError("normalization feature order must be h,S,Qv,Qc,B")
        offset = _finite_float64("input offset", self.input_offset, ndim=1)
        scale = _finite_float64("input scale", self.input_scale, ndim=1)
        if offset.shape != (5,) or scale.shape != (5,):
            raise ValueError("input normalization arrays must have shape (5,)")
        if np.any(scale <= 0.0):
            raise ValueError("input scales must be positive")
        if len(self.input_zero_scale) != 5:
            raise ValueError("input zero-scale flags must have length five")
        if not np.isfinite(self.output_scale) or self.output_scale <= 0.0:
            raise ValueError("output scale must be finite and positive")
        if (self.fitted_state_first, self.fitted_state_last) != (0, 80):
            raise ValueError("normalization may be fitted only from states 0..80")
        owned_offset = np.array(offset, dtype=np.float64, copy=True)
        owned_scale = np.array(scale, dtype=np.float64, copy=True)
        owned_offset.setflags(write=False)
        owned_scale.setflags(write=False)
        object.__setattr__(self, "input_offset", owned_offset)
        object.__setattr__(self, "input_scale", owned_scale)
        object.__setattr__(
            self,
            "output_candidates",
            MappingProxyType(
                {key: float(value) for key, value in self.output_candidates.items()}
            ),
        )

    def normalize_features(self, features):
        return (
            jnp.asarray(features, dtype=jnp.float64)
            - jnp.asarray(self.input_offset, dtype=jnp.float64)
        ) / jnp.asarray(self.input_scale, dtype=jnp.float64)

    def inverse_features(self, normalized):
        return (
            jnp.asarray(normalized, dtype=jnp.float64)
            * jnp.asarray(self.input_scale, dtype=jnp.float64)
            + jnp.asarray(self.input_offset, dtype=jnp.float64)
        )

    def normalize_a(self, values):
        return jnp.asarray(values, dtype=jnp.float64) / self.output_scale

    def inverse_a(self, normalized):
        return jnp.asarray(normalized, dtype=jnp.float64) * self.output_scale

    def to_record(self):
        return {
            "feature_order": list(self.feature_order),
            "input": {
                name: {
                    "offset": float(self.input_offset[index]),
                    "scale": float(self.input_scale[index]),
                    "zero_or_degenerate_scale": bool(
                        self.input_zero_scale[index]
                    ),
                    "method": "training mean and population standard deviation",
                }
                for index, name in enumerate(self.feature_order)
            },
            "output": {
                "scale": float(self.output_scale),
                "method": self.output_scale_method,
                "centered": False,
                "zero_or_degenerate_scale": bool(self.output_zero_scale),
                "candidate_scales": {
                    key: float(value)
                    for key, value in self.output_candidates.items()
                },
                "reason": (
                    "uncentered RMS conditions the mean-squared operator loss "
                    "at its typical target magnitude and maps normalized zero "
                    "exactly to physical A=0; max_abs would compress the bulk "
                    "under intermittent tails"
                ),
            },
            "fitted_truth_state_indices": [0, 80],
            "future_states_used": False,
        }


def fit_normalization(features, targets):
    """Fit all statistics from the complete training-only operator dataset."""
    feature_array = _finite_float64("features", features, ndim=2)
    target_array = _finite_float64("targets", targets)
    target_array = target_array.reshape(-1, 1)
    if feature_array.shape[1] != 5:
        raise ValueError("features must have columns h,S,Qv,Qc,B")
    if feature_array.shape[0] != target_array.shape[0]:
        raise ValueError("features and targets must have equal sample counts")
    offset = np.mean(feature_array, axis=0, dtype=np.float64)
    standard_deviation = np.std(feature_array, axis=0, dtype=np.float64)
    maximum_absolute = np.max(np.abs(feature_array), axis=0)
    threshold = 64.0 * _FLOAT64_EPSILON * np.maximum(1.0, maximum_absolute)
    degenerate = standard_deviation <= threshold
    scale = np.where(degenerate, 1.0, standard_deviation).astype(np.float64)

    flat_target = target_array.reshape(-1)
    rms = float(np.sqrt(np.mean(flat_target * flat_target)))
    std = float(np.std(flat_target))
    max_abs = float(np.max(np.abs(flat_target)))
    candidates = {"rms": rms, "standard_deviation": std, "max_abs": max_abs}
    target_threshold = 64.0 * _FLOAT64_EPSILON * max(1.0, max_abs)
    target_degenerate = rms <= target_threshold
    output_scale = 1.0 if target_degenerate else rms
    return NormalizationMetadata(
        feature_order=FEATURE_ORDER,
        input_offset=np.asarray(offset, dtype=np.float64),
        input_scale=scale,
        input_zero_scale=tuple(bool(value) for value in degenerate),
        output_scale=float(output_scale),
        output_scale_method="uncentered_training_rms",
        output_candidates=candidates,
        output_zero_scale=bool(target_degenerate),
    )


def normalization_from_record(record):
    inputs = record["input"]
    order = tuple(record["feature_order"])
    output = record["output"]
    return NormalizationMetadata(
        feature_order=order,
        input_offset=np.asarray(
            [inputs[name]["offset"] for name in order], dtype=np.float64
        ),
        input_scale=np.asarray(
            [inputs[name]["scale"] for name in order], dtype=np.float64
        ),
        input_zero_scale=tuple(
            bool(inputs[name]["zero_or_degenerate_scale"]) for name in order
        ),
        output_scale=float(output["scale"]),
        output_scale_method=str(output["method"]),
        output_candidates={
            key: float(value) for key, value in output["candidate_scales"].items()
        },
        output_zero_scale=bool(output["zero_or_degenerate_scale"]),
    )


@dataclass(frozen=True)
class LocalAFeatureMap:
    normalization: NormalizationMetadata

    def __call__(self, state, context):
        physical = jnp.stack(
            (
                state["h"],
                state["S"],
                state["Qv"],
                state["Qc"],
                context["B"],
            ),
            axis=-1,
        )
        return self.normalization.normalize_features(physical)


@dataclass(frozen=True)
class HybridAMoistOutputMap:
    """Replace A only while retaining the supplied original deployed R."""

    normalization: NormalizationMetadata

    def __call__(self, state, context, baseline_physics, raw_output):
        learned_a = self.normalization.inverse_a(jnp.squeeze(raw_output, axis=-1))
        original_r = jnp.asarray(baseline_physics["R"], dtype=jnp.float64)
        h = jnp.asarray(state["h"], dtype=jnp.float64)
        beta2 = jnp.asarray(context["beta2"], dtype=jnp.float64)
        return {
            "A": learned_a,
            "R": original_r,
            "source": {
                "Qv": h * learned_a,
                "Qc": -h * (learned_a + original_r),
                "Qr": h * original_r,
                "S": h * beta2 * learned_a,
            },
        }


def build_learned_a_model(configuration, normalization):
    """Compose Test 2A through the existing J4A learned-physics abstraction."""
    return LearnedPhysicsModel(
        feature_map=LocalAFeatureMap(normalization),
        model=DenseMLP(configuration),
        output_map=HybridAMoistOutputMap(normalization),
        name="test2a_hybrid_learned_A_original_R",
    )


@dataclass(frozen=True)
class OperatorDataset:
    features: np.ndarray
    targets: np.ndarray
    steps: tuple[int, ...]
    cells: int
    points_per_cell: int = POINTS_PER_CELL

    def __post_init__(self):
        features = _finite_float64("dataset features", self.features, ndim=2)
        targets = _finite_float64("dataset targets", self.targets)
        if features.shape[1] != 5 or targets.reshape(-1, 1).shape[1] != 1:
            raise ValueError("dataset must contain five inputs and scalar A")
        if features.shape[0] != targets.size:
            raise ValueError("dataset features and targets do not align")
        require_training_steps(self.steps)
        expected = len(self.steps) * self.cells * self.points_per_cell
        if features.shape[0] != expected:
            raise ValueError("dataset sample count disagrees with deployed layout")
        owned_features = np.array(features, dtype=np.float64, order="C", copy=True)
        owned_targets = np.array(
            targets.reshape(-1, 1), dtype=np.float64, order="C", copy=True
        )
        owned_features.setflags(write=False)
        owned_targets.setflags(write=False)
        object.__setattr__(self, "features", owned_features)
        object.__setattr__(self, "targets", owned_targets)

    @property
    def sample_count(self):
        return int(self.features.shape[0])


def assemble_operator_dataset(records, *, cells, points_per_cell=POINTS_PER_CELL):
    """Assemble state-major/cell-major/GLL-major samples without deduplication."""
    steps = require_training_steps(records.keys())
    feature_blocks = []
    target_blocks = []
    expected_shape = (int(cells), int(points_per_cell))
    for step in steps:
        record = records[step]
        columns = []
        for name in FEATURE_ORDER:
            value = _finite_float64(f"step {step} feature {name}", record[name])
            if value.shape != expected_shape:
                raise ValueError(f"step {step} feature {name} has wrong shape")
            columns.append(value.reshape(-1))
        target = _finite_float64(f"step {step} target A", record["A"])
        if target.shape != expected_shape:
            raise ValueError(f"step {step} target A has wrong shape")
        feature_blocks.append(np.stack(columns, axis=1))
        target_blocks.append(target.reshape(-1, 1))
    return OperatorDataset(
        features=np.asarray(np.concatenate(feature_blocks), dtype=np.float64),
        targets=np.asarray(np.concatenate(target_blocks), dtype=np.float64),
        steps=steps,
        cells=int(cells),
        points_per_cell=int(points_per_cell),
    )


def normalized_operator_objective(parameters, model, normalized_features, normalized_a):
    """Full-batch ``mean(((A_theta-A*)/A_rms)**2)`` objective."""
    prediction = model(parameters, normalized_features)
    difference = prediction - normalized_a
    return jnp.mean(difference * difference)


def physical_predictions(parameters, model, normalization, features):
    normalized_features = normalization.normalize_features(features)
    normalized_prediction = model(parameters, normalized_features)
    return np.asarray(
        normalization.inverse_a(normalized_prediction), dtype=np.float64
    ).reshape(-1)


def operator_metrics(prediction, target):
    prediction_array = _finite_float64("prediction", prediction).reshape(-1)
    target_array = _finite_float64("target", target).reshape(-1)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes differ")
    error = prediction_array - target_array
    target_rms = float(np.sqrt(np.mean(target_array * target_array)))
    rmse = float(np.sqrt(np.mean(error * error)))
    maximum = float(np.max(np.abs(target_array)))
    near_zero = 1.0e-6 * maximum
    sign_mask = np.abs(target_array) > near_zero
    if np.any(sign_mask):
        sign_accuracy = float(
            np.mean(np.sign(prediction_array[sign_mask]) == np.sign(target_array[sign_mask]))
        )
    else:
        sign_accuracy = None
    if np.std(prediction_array) == 0.0 or np.std(target_array) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(prediction_array, target_array)[0, 1])
    sign_strata = {}
    for threshold in (1.0e-3, 1.0e-2, 1.0e-1):
        mask = (
            np.abs(target_array) > threshold * maximum
            if maximum
            else np.zeros_like(target_array, dtype=bool)
        )
        label = f"abs_A_gt_{threshold:.0e}_max_abs_A"
        sign_strata[label] = {
            "sample_count": int(np.count_nonzero(mask)),
            "sample_fraction": float(np.mean(mask)),
            "threshold": float(threshold * maximum),
            "accuracy": (
                float(
                    np.mean(
                        np.sign(prediction_array[mask])
                        == np.sign(target_array[mask])
                    )
                )
                if np.any(mask)
                else None
            ),
        }
    strata = {}
    for threshold in (1.0e-3, 1.0e-6):
        mask = np.abs(target_array) > threshold * maximum if maximum else np.zeros_like(
            target_array, dtype=bool
        )
        label = f"abs_A_gt_{threshold:.0e}_max_abs_A"
        if np.any(mask):
            subset_error = error[mask]
            subset_target = target_array[mask]
            subset_rms = float(np.sqrt(np.mean(subset_target * subset_target)))
            strata[label] = {
                "sample_count": int(np.count_nonzero(mask)),
                "sample_fraction": float(np.mean(mask)),
                "rmse": float(np.sqrt(np.mean(subset_error * subset_error))),
                "mae": float(np.mean(np.abs(subset_error))),
                "relative_rms_error": float(
                    np.sqrt(np.mean(subset_error * subset_error)) / subset_rms
                ),
            }
        else:
            strata[label] = {"sample_count": 0, "sample_fraction": 0.0}
    return {
        "sample_count": int(target_array.size),
        "normalized_mse": None if target_rms == 0.0 else float((rmse / target_rms) ** 2),
        "physical_rmse_A": rmse,
        "physical_mae_A": float(np.mean(np.abs(error))),
        "relative_rms_error": None if target_rms == 0.0 else float(rmse / target_rms),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "correlation": correlation,
        "sign_accuracy": sign_accuracy,
        "sign_accuracy_threshold": near_zero,
        "sign_accuracy_strata": sign_strata,
        "magnitude_strata": strata,
    }


def diagnostic_baselines(features, targets, normalization):
    """Return zero, mean, and affine least-squares reference predictions."""
    feature_array = _finite_float64("features", features, ndim=2)
    target_array = _finite_float64("targets", targets).reshape(-1)
    normalized = np.asarray(
        normalization.normalize_features(feature_array), dtype=np.float64
    )
    design = np.column_stack(
        (normalized, np.ones((normalized.shape[0],), dtype=np.float64))
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, target_array, rcond=None)
    predictions = {
        "zero": np.zeros_like(target_array),
        "constant_training_mean": np.full_like(target_array, np.mean(target_array)),
        "affine_normalized_five_input": design @ coefficients,
    }
    return {
        name: {
            "metrics": operator_metrics(prediction, target_array),
            **(
                {"coefficients": coefficients.tolist(), "feature_order": list(FEATURE_ORDER)}
                if name == "affine_normalized_five_input"
                else {}
            ),
        }
        for name, prediction in predictions.items()
    }


def _array_sha256(features, targets):
    digest = hashlib.sha256()
    for array in (features, targets):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def save_operator_dataset(dataset, destination, metadata):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=np.asarray(dataset.features, dtype=np.float64),
        targets=np.asarray(dataset.targets, dtype=np.float64),
    )
    record = dict(metadata)
    record.update(
        {
            "format_version": 1,
            "dataset_file": destination.name,
            "feature_order": list(FEATURE_ORDER),
            "target": "A",
            "sample_count": dataset.sample_count,
            "truth_state_indices": [0, 80],
            "states_after_80_accessed": False,
            "cells": dataset.cells,
            "points_per_cell": dataset.points_per_cell,
            "shared_physical_cg_points_deduplicated": False,
            "sha256_float64_content": _array_sha256(
                dataset.features, dataset.targets
            ),
        }
    )
    metadata_path = destination.with_suffix(".json")
    write_json_record(metadata_path, record)
    return destination, metadata_path


def load_operator_dataset(path):
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        features = np.array(archive["features"], dtype=np.float64, copy=True)
        targets = np.array(archive["targets"], dtype=np.float64, copy=True)
    with source.with_suffix(".json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["truth_state_indices"] != [0, 80] or metadata.get(
        "states_after_80_accessed", True
    ):
        raise ValueError("dataset metadata violates the Test 2A training-only contract")
    expected_hash = metadata["sha256_float64_content"]
    if _array_sha256(features, targets) != expected_hash:
        raise ValueError("operator dataset content hash does not match metadata")
    dataset = OperatorDataset(
        features=features,
        targets=targets,
        steps=TRAINING_STEPS,
        cells=int(metadata["cells"]),
        points_per_cell=int(metadata["points_per_cell"]),
    )
    return dataset, metadata


def save_mlp_parameters(path, parameters, configuration):
    """Persist the known configurable dense-layer pytree without pickle."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    layers = parameters["layers"]
    for index, layer in enumerate(layers):
        arrays[f"layer_{index:03d}_weight"] = np.asarray(
            layer["weight"], dtype=np.float64
        )
        arrays[f"layer_{index:03d}_bias"] = np.asarray(
            layer["bias"], dtype=np.float64
        )
    np.savez_compressed(destination, **arrays)
    write_json_record(
        destination.with_suffix(".json"),
        {
            "format_version": 1,
            "architecture": configuration.to_record(),
            "parameter_file": destination.name,
            "analytical_A_formula_encoded": False,
        },
    )
    return destination


def save_mlp_parameters_atomic(path, parameters, configuration):
    """Atomically publish a matching NPZ/JSON parameter checkpoint pair."""
    destination = Path(path)
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite parameter artifact {destination}")
    temporary = destination.with_name(destination.stem + ".incomplete.npz")
    temporary_sidecar = temporary.with_suffix(".json")
    if temporary.exists() or temporary_sidecar.exists():
        raise FileExistsError(f"incomplete parameter artifact exists at {temporary}")
    try:
        save_mlp_parameters(temporary, parameters, configuration)
        temporary.replace(destination)
        temporary_sidecar.replace(sidecar)
    except BaseException:
        for candidate in (temporary, temporary_sidecar):
            if candidate.exists():
                candidate.unlink()
        raise
    return destination


def load_mlp_parameters(path):
    source = Path(path)
    with source.with_suffix(".json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    configuration = mlp_configuration_from_record(metadata["architecture"])
    layers = []
    with np.load(source, allow_pickle=False) as archive:
        for index in range(len(configuration.layer_dimensions) - 1):
            layers.append(
                {
                    "weight": jnp.asarray(
                        archive[f"layer_{index:03d}_weight"], dtype=jnp.float64
                    ),
                    "bias": jnp.asarray(
                        archive[f"layer_{index:03d}_bias"], dtype=jnp.float64
                    ),
                }
            )
    return validate_float64_tree({"layers": tuple(layers)}), configuration


def load_selected_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("benchmark_stage") != "Test 2A-1 local operator learning":
        raise ValueError("not a selected Test 2A-1 configuration")
    if record["data"]["truth_state_indices"] != [0, 80]:
        raise ValueError("selected Test 2A data must be restricted to states 0..80")
    if tuple(record["data"]["feature_order"]) != FEATURE_ORDER:
        raise ValueError("selected Test 2A feature order changed")
    model = mlp_configuration_from_record(record["model"])
    if int(record["model"]["parameter_count"]) != model.parameter_count:
        raise ValueError("selected Test 2A parameter count is inconsistent")
    if record["physics"].get("original_R_retained") is not True:
        raise ValueError("selected Test 2A must retain original deployed R")
    return record


def prepare_training_data(truth_run, selected_plan, configuration_path, output):
    """Read exact J1 GLL samples for states 0..80; never integrate the solver."""
    from .hidden_c0 import _serial_solver_parameters
    from .jax_moist_adapter import JAXMoistEulerPrimal
    from .resolved_hidden_c0_inference import load_resolved_truth
    from .selected_test1b import load_selected_test1b_plan

    selected_configuration = load_selected_configuration(configuration_path)
    _, selected = load_selected_test1b_plan(selected_plan)
    inference = selected.inference_configuration(Path(truth_run).resolve())
    if (inference.training_start_step, inference.training_stop_step) != (0, 80):
        raise ValueError("truth plan does not expose the required states 0..80")
    case, trajectory = load_resolved_truth(inference, include_heldout=False)
    require_training_steps(trajectory.states.keys())
    adapter = JAXMoistEulerPrimal(
        case.model,
        _serial_solver_parameters(),
        use_jit=True,
    )
    records = {}
    for step in TRAINING_STEPS:
        cache = adapter.evaluate(trajectory.states[step], case.dt)
        records[step] = {
            "h": np.asarray(cache.packed_state["h"], dtype=np.float64),
            "S": np.asarray(cache.packed_state["S"], dtype=np.float64),
            "Qv": np.asarray(cache.packed_state["Qv"], dtype=np.float64),
            "Qc": np.asarray(cache.packed_state["Qc"], dtype=np.float64),
            "B": np.asarray(cache.packed_fields["B"], dtype=np.float64),
            "A": np.asarray(cache.rates["A"], dtype=np.float64),
        }
    dataset = assemble_operator_dataset(
        records,
        cells=adapter.layout.owned_cell_count,
        points_per_cell=adapter.layout.points_per_cell,
    )
    if dataset.sample_count != int(selected_configuration["data"]["sample_count"]):
        raise ValueError("prepared dataset does not have the selected sample count")
    normalization = fit_normalization(dataset.features, dataset.targets)
    baselines = diagnostic_baselines(dataset.features, dataset.targets, normalization)
    metadata = {
        "status": "complete",
        "benchmark_stage": "Test 2A-1 operator dataset",
        "source_truth_run": str(Path(truth_run).resolve()),
        "selected_test1b_plan": str(Path(selected_plan).resolve()),
        "selected_test2a_configuration": str(Path(configuration_path).resolve()),
        "source_truth_provenance": selected_configuration["data"][
            "source_truth_provenance"
        ],
        "deployed_representation": {
            "adapter": "dimswe.jax_moist_adapter.JAXMoistEulerPrimal",
            "shape_per_state": [adapter.layout.owned_cell_count, 16],
            "ordering": "state-major, cell-major, local GLL point",
            "shared_points_repeated": True,
            "j1_certified_target": "cache.rates['A']",
        },
        "normalization": normalization.to_record(),
        "diagnostic_baselines": baselines,
        "training_policy": {
            "sample_policy": "all deployed samples from every state 0..80",
            "batch_policy": "deterministic full batch",
            "pointwise_validation_split": None,
            "purpose": "representability over training support, not generalization",
        },
    }
    return save_operator_dataset(dataset, output, metadata)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--truth-run", required=True)
    prepare.add_argument("--selected-plan", required=True)
    prepare.add_argument("--configuration", required=True)
    prepare.add_argument("--output", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare-data":
        prepare_training_data(
            arguments.truth_run,
            arguments.selected_plan,
            arguments.configuration,
            arguments.output,
        )
        return 0
    raise AssertionError("unreachable Test 2A command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DenseMLP",
    "FEATURE_ORDER",
    "HybridAMoistOutputMap",
    "LocalAFeatureMap",
    "MLPConfiguration",
    "NormalizationMetadata",
    "OperatorDataset",
    "POINTS_PER_CELL",
    "SELECTED_SAMPLE_COUNT",
    "TRAINING_STEPS",
    "assemble_operator_dataset",
    "build_learned_a_model",
    "deployed_sample_count",
    "diagnostic_baselines",
    "fit_normalization",
    "initialize_mlp",
    "load_mlp_parameters",
    "load_operator_dataset",
    "load_selected_configuration",
    "normalized_operator_objective",
    "normalization_from_record",
    "operator_metrics",
    "physical_predictions",
    "prepare_training_data",
    "require_training_steps",
    "save_mlp_parameters",
    "save_operator_dataset",
)
