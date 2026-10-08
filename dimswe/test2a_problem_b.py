"""Test-2A Problem-B independent four-tendency learned moist physics.

The local network changes only the representation of the learned moist law:
it maps the frozen Problem-A features ``(h,S,Qv,Qc,B)`` to four independent
physical source densities in the order ``(S,Qv,Qc,Qr)``.  No analytical rate,
conservation projection, or shared scalar is used by the source calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from .learned_physics.parameters import tree_copy, validate_float64_tree
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import DenseMLP, FEATURE_ORDER, initialize_mlp


PHYSICS_MODE_NEURAL_FOUR_TENDENCY = "neural_four_tendency"
SOURCE_ORDER = ("S", "Qv", "Qc", "Qr")
STATE_KEYS = ("h", "S", "Qv", "Qc")
PARAMETER_COUNT = 1380
TRAINING_STEPS = tuple(range(81))
_FLOAT64 = jnp.dtype(jnp.float64)


def _canonical_sha256(record):
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProblemBMLPConfiguration:
    input_dimension: int = 5
    hidden_layers: tuple[int, ...] = (32, 32)
    output_dimension: int = 4
    activation: str = "tanh"
    dtype: str = "float64"
    seed: int = 0

    def __post_init__(self):
        if self.input_dimension != 5 or self.output_dimension != 4:
            raise ValueError("Problem B architecture must be 5->...->4")
        if self.hidden_layers != (32, 32):
            raise ValueError("Problem B hidden layers must be (32,32)")
        if self.activation != "tanh" or self.dtype != "float64" or self.seed != 0:
            raise ValueError("Problem B requires tanh, float64, seed 0")
        if self.parameter_count != PARAMETER_COUNT:
            raise AssertionError("Problem B parameter-count contract changed")

    @property
    def layer_dimensions(self):
        return (self.input_dimension, *self.hidden_layers, self.output_dimension)

    @property
    def parameter_count(self):
        return int(
            sum(
                left * right + right
                for left, right in zip(
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


@dataclass(frozen=True)
class FourTendencyNormalization:
    """Frozen Problem-A input map and positive Problem-B output scales."""

    input_offset: np.ndarray
    input_scale: np.ndarray
    sigma_s: float
    sigma_q: float
    input_normalization_sha256: str
    scale_provenance_sha256: str

    def __post_init__(self):
        offset = np.asarray(self.input_offset, dtype=np.float64)
        scale = np.asarray(self.input_scale, dtype=np.float64)
        if offset.shape != (5,) or scale.shape != (5,) or np.any(scale <= 0.0):
            raise ValueError("Problem B input normalization must contain five scales")
        if not np.isfinite(self.sigma_s) or self.sigma_s <= 0.0:
            raise ValueError("sigma_S must be finite and positive")
        if not np.isfinite(self.sigma_q) or self.sigma_q <= 0.0:
            raise ValueError("sigma_Q must be finite and positive")
        offset = np.array(offset, copy=True)
        scale = np.array(scale, copy=True)
        offset.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "input_offset", offset)
        object.__setattr__(self, "input_scale", scale)

    @property
    def output_scales(self):
        return np.asarray(
            (self.sigma_s, self.sigma_q, self.sigma_q, self.sigma_q),
            dtype=np.float64,
        )

    def normalize_features(self, features):
        return (
            jnp.asarray(features, dtype=jnp.float64)
            - jnp.asarray(self.input_offset, dtype=jnp.float64)
        ) / jnp.asarray(self.input_scale, dtype=jnp.float64)

    def physical_tendencies(self, coordinates):
        return jnp.asarray(coordinates, dtype=jnp.float64) * jnp.asarray(
            self.output_scales, dtype=jnp.float64
        )

    def normalized_tendencies(self, physical):
        return jnp.asarray(physical, dtype=jnp.float64) / jnp.asarray(
            self.output_scales, dtype=jnp.float64
        )

    def to_record(self):
        return {
            "feature_order": list(FEATURE_ORDER),
            "input_offset": self.input_offset.tolist(),
            "input_scale": self.input_scale.tolist(),
            "input_normalization_sha256": self.input_normalization_sha256,
            "output_order": list(SOURCE_ORDER),
            "sigma_S": float(self.sigma_s),
            "sigma_Q": float(self.sigma_q),
            "output_scales": self.output_scales.tolist(),
            "definition": (
                "sigma_S=RMS_M(S_t*); sigma_Q=sqrt((RMS_M(Qv_t*)^2+"
                "RMS_M(Qc_t*)^2)/2); sigma_Q shared by Qv,Qc,Qr"
            ),
            "fitted_truth_state_indices": [0, 80],
            "states_after_80_accessed": False,
            "scale_provenance_sha256": self.scale_provenance_sha256,
        }


def normalization_from_record(record):
    if tuple(record["feature_order"]) != FEATURE_ORDER:
        raise ValueError("Problem B feature order changed")
    if tuple(record["output_order"]) != SOURCE_ORDER:
        raise ValueError("Problem B source order changed")
    if record.get("fitted_truth_state_indices") != [0, 80] or record.get(
        "states_after_80_accessed", True
    ):
        raise ValueError("Problem B normalization violates truth support")
    return FourTendencyNormalization(
        input_offset=np.asarray(record["input_offset"], dtype=np.float64),
        input_scale=np.asarray(record["input_scale"], dtype=np.float64),
        sigma_s=float(record["sigma_S"]),
        sigma_q=float(record["sigma_Q"]),
        input_normalization_sha256=str(record["input_normalization_sha256"]),
        scale_provenance_sha256=str(record["scale_provenance_sha256"]),
    )


def build_problem_b_model(configuration=None):
    configuration = configuration or ProblemBMLPConfiguration()
    return DenseMLP(configuration)


def initial_problem_b_parameters(configuration=None):
    configuration = configuration or ProblemBMLPConfiguration()
    parameters = initialize_mlp(configuration)
    count = sum(int(x.size) for x in jax.tree_util.tree_leaves(parameters))
    if count != PARAMETER_COUNT:
        raise AssertionError("Problem B initialized pytree does not contain 1380 values")
    return parameters


def save_problem_b_parameters(path, parameters, configuration=None, *, metadata=None):
    configuration = configuration or ProblemBMLPConfiguration()
    owned = validate_float64_tree(parameters, name="Problem B parameters")
    destination = Path(path)
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite Problem B parameters")
    destination.parent.mkdir(parents=True, exist_ok=True)
    leaves = {}
    for index, layer in enumerate(owned["layers"]):
        leaves[f"layer_{index:03d}_weight"] = np.asarray(layer["weight"])
        leaves[f"layer_{index:03d}_bias"] = np.asarray(layer["bias"])
    temporary = destination.with_name(destination.name + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **leaves)
    temporary.replace(destination)
    record = {
        "format_version": 1,
        "architecture": configuration.to_record(),
        "parameter_pytree_sha256": parameter_pytree_sha256(owned),
        "initialization_convention": "Problem-A Glorot-uniform per layer, zero bias",
        "metadata": dict(metadata or {}),
    }
    sidecar_temporary = sidecar.with_name(sidecar.name + ".incomplete")
    sidecar_temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar_temporary.replace(sidecar)
    return record


def load_problem_b_parameters(path):
    source = Path(path)
    record = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    configuration = ProblemBMLPConfiguration()
    if record["architecture"] != configuration.to_record():
        raise ValueError("Problem B parameter architecture changed")
    layers = []
    dimensions = configuration.layer_dimensions
    with np.load(source, allow_pickle=False) as archive:
        for index, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            weight = np.asarray(archive[f"layer_{index:03d}_weight"])
            bias = np.asarray(archive[f"layer_{index:03d}_bias"])
            if weight.shape != (fan_in, fan_out) or bias.shape != (fan_out,):
                raise ValueError("Problem B parameter leaf shape changed")
            if weight.dtype != np.float64 or bias.dtype != np.float64:
                raise TypeError("Problem B parameters must be float64")
            layers.append({"weight": jnp.asarray(weight), "bias": jnp.asarray(bias)})
    parameters = {"layers": tuple(layers)}
    if parameter_pytree_sha256(parameters) != record["parameter_pytree_sha256"]:
        raise ValueError("Problem B parameter pytree fingerprint mismatch")
    return parameters, configuration, record


def _require_tree(values, keys, name):
    result = {}
    for key in keys:
        value = jnp.asarray(values[key])
        if value.dtype != _FLOAT64:
            raise TypeError(f"{name}['{key}'] must be float64")
        result[key] = value
    return result


class NeuralFourTendencyMoistPhysics:
    """Pure-JAX independent four-source provider for the certified adapter."""

    physics_mode = PHYSICS_MODE_NEURAL_FOUR_TENDENCY

    def __init__(self, parameters, normalization, *, use_jit=True, provenance=None):
        self.configuration = ProblemBMLPConfiguration()
        self.model = build_problem_b_model(self.configuration)
        self.normalization = normalization
        self.use_jit = bool(use_jit)
        self.provenance = MappingProxyType(dict(provenance or {}))
        self._parameters = validate_float64_tree(parameters, name="Problem B parameters")
        count = sum(int(x.size) for x in jax.tree_util.tree_leaves(self._parameters))
        if count != PARAMETER_COUNT:
            raise ValueError("Problem B provider requires 1380 parameters")

        def combined(state, fields, moist_parameters, neural_parameters):
            del moist_parameters
            state = _require_tree(state, STATE_KEYS, "state_q")
            fields = _require_tree(fields, ("B",), "fields_q")
            neural_parameters = validate_float64_tree(
                neural_parameters, name="neural_parameters"
            )
            features = jnp.stack(
                (state["h"], state["S"], state["Qv"], state["Qc"], fields["B"]),
                axis=-1,
            )
            coordinates = self.model(
                neural_parameters, self.normalization.normalize_features(features)
            )
            physical = self.normalization.physical_tendencies(coordinates)
            source = {name: physical[..., index] for index, name in enumerate(SOURCE_ORDER)}
            return {"rates": {}, "source": source}

        self._combined_explicit = combined
        self.combined_parameterized_kernel = jax.jit(combined) if use_jit else combined
        frozen = lambda state, fields, parameters: combined(
            state, fields, parameters, self._parameters
        )
        self.combined_kernel = jax.jit(frozen) if use_jit else frozen

        def diagnostics(state, fields, parameters, neural_parameters):
            del fields, parameters, neural_parameters
            reference = jnp.asarray(state["h"], dtype=jnp.float64)
            false = jnp.zeros_like(reference, dtype=bool)
            one = jnp.ones_like(reference, dtype=jnp.float64)
            return {
                "condensation_mask": false,
                "evaporation_mask": false,
                "uncapped_evaporation_mask": false,
                "rain_mask": false,
                "condensation_margin": one,
                "evaporation_margin": one,
                "evaporation_cap_margin": one,
                "rain_margin": one,
                "depth_denominator_margin": one,
            }

        parameterized_diagnostics = jax.jit(diagnostics) if use_jit else diagnostics
        self.diagnostic_parameterized_kernel = parameterized_diagnostics
        frozen_diagnostics = lambda state, fields, parameters: diagnostics(
            state, fields, parameters, self._parameters
        )
        self.diagnostic_kernel = (
            jax.jit(frozen_diagnostics) if use_jit else frozen_diagnostics
        )

        def state_jvp(state, direction, fields, parameters):
            source = lambda active: combined(active, fields, parameters, self._parameters)[
                "source"
            ]
            return jax.jvp(
                source,
                (_require_tree(state, STATE_KEYS, "state_q"),),
                (_require_tree(direction, STATE_KEYS, "dstate_q"),),
            )

        def state_vjp(state, source_covector, fields, parameters):
            active = _require_tree(state, STATE_KEYS, "state_q")
            covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
            source = lambda value: combined(value, fields, parameters, self._parameters)[
                "source"
            ]
            _, pullback = jax.vjp(source, active)
            return pullback(covector)[0]

        def state_differentiated_vjp(
            state, source_covector, direction, source_covector_direction, fields, parameters
        ):
            active = _require_tree(state, STATE_KEYS, "state_q")
            covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
            dstate = _require_tree(direction, STATE_KEYS, "dstate_q")
            dcovector = _require_tree(
                source_covector_direction, SOURCE_ORDER, "dsource_covector_q"
            )

            def vjp_map(local_state, local_covector):
                source = lambda value: combined(
                    value, fields, parameters, self._parameters
                )["source"]
                _, pullback = jax.vjp(source, local_state)
                return pullback(local_covector)[0]

            return jax.jvp(vjp_map, (active, covector), (dstate, dcovector))

        self.state_jvp_kernel = jax.jit(state_jvp) if use_jit else state_jvp
        self.state_vjp_kernel = jax.jit(state_vjp) if use_jit else state_vjp
        self.state_differentiated_vjp_kernel = (
            jax.jit(state_differentiated_vjp) if use_jit else state_differentiated_vjp
        )

    @property
    def parameters(self):
        return tree_copy(self._parameters)

    def combined_with_parameters(self, state, fields, moist_parameters, neural_parameters):
        return self._combined_explicit(
            state, fields, moist_parameters, neural_parameters
        )

    def parameter_jvp(self, state, direction, fields, parameters, *, base_parameters=None):
        base = self._parameters if base_parameters is None else validate_float64_tree(
            base_parameters, name="base_parameters"
        )
        direction = validate_float64_tree(direction, name="parameter_direction")
        source = lambda active: self._combined_explicit(state, fields, parameters, active)[
            "source"
        ]
        return jax.jvp(source, (base,), (direction,))

    def parameter_vjp(
        self, state, source_covector, fields, parameters, *, base_parameters=None
    ):
        base = self._parameters if base_parameters is None else validate_float64_tree(
            base_parameters, name="base_parameters"
        )
        covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
        source = lambda active: self._combined_explicit(state, fields, parameters, active)[
            "source"
        ]
        _, pullback = jax.vjp(source, base)
        return pullback(covector)[0]

    def joint_differentiated_vjp(
        self,
        state,
        source_covector,
        state_direction,
        parameter_direction,
        source_covector_direction,
        fields,
        parameters,
        *,
        base_parameters=None,
    ):
        active_state = _require_tree(state, STATE_KEYS, "state_q")
        covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
        dstate = _require_tree(state_direction, STATE_KEYS, "dstate_q")
        dparameters = validate_float64_tree(
            parameter_direction, name="parameter_direction"
        )
        dcovector = _require_tree(
            source_covector_direction, SOURCE_ORDER, "dsource_covector_q"
        )
        base = self._parameters if base_parameters is None else validate_float64_tree(
            base_parameters, name="base_parameters"
        )

        def vjp_map(local_state, local_parameters, local_covector):
            source = lambda state_value, parameter_value: self._combined_explicit(
                state_value, fields, parameters, parameter_value
            )["source"]
            _, pullback = jax.vjp(source, local_state, local_parameters)
            return pullback(local_covector)

        return jax.jvp(
            vjp_map,
            (active_state, base, covector),
            (dstate, dparameters, dcovector),
        )


@dataclass(frozen=True)
class ProblemBOperatorDataset:
    normalized_features: np.ndarray
    physical_targets: np.ndarray
    spatial_weights: np.ndarray
    normalization: FourTendencyNormalization
    metadata: dict

    def __post_init__(self):
        features = np.asarray(self.normalized_features, dtype=np.float64)
        targets = np.asarray(self.physical_targets, dtype=np.float64)
        weights = np.asarray(self.spatial_weights, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError("Problem B features must have shape (samples,5)")
        if targets.shape != (features.shape[0], 4):
            raise ValueError("Problem B targets must have shape (samples,4)")
        if weights.shape != (features.shape[0],) or np.any(weights <= 0.0):
            raise ValueError("Problem B mass weights must be positive per sample")
        if self.metadata.get("truth_state_indices") != [0, 80] or self.metadata.get(
            "states_after_80_accessed", True
        ):
            raise ValueError("Problem B dataset violates training support")

    @property
    def sample_count(self):
        return int(self.normalized_features.shape[0])


class ProblemBOperatorObjective:
    """Pure-JAX direct four-tendency mass-weighted M1 objective."""

    def __init__(self, dataset, *, use_jit=True):
        self.dataset = dataset
        self.model = build_problem_b_model()
        features = jnp.asarray(dataset.normalized_features, dtype=jnp.float64)
        targets = dataset.normalization.normalized_tendencies(dataset.physical_targets)
        weights = jnp.asarray(dataset.spatial_weights, dtype=jnp.float64)
        denominator = jnp.sum(weights[:, None] * targets * targets)

        def objective(parameters):
            error = self.model(parameters, features) - targets
            return jnp.sum(weights[:, None] * error * error) / denominator

        self.denominator = float(denominator)
        self._value = jax.jit(objective) if use_jit else objective
        value_gradient = jax.value_and_grad(objective)
        self._value_gradient = jax.jit(value_gradient) if use_jit else value_gradient

    def value(self, parameters):
        return float(self._value(parameters))

    def jax_value(self, parameters):
        return self._value(parameters)

    def value_and_gradient(self, parameters):
        value, gradient = self._value_gradient(parameters)
        return float(value), gradient

    def gradient(self, parameters):
        return self.value_and_gradient(parameters)[1]


def structural_diagnostics(physical_prediction, physical_truth, beta2, normalization, weights):
    """Return diagnostics only; no projected value feeds the learned child."""
    prediction = np.asarray(physical_prediction, dtype=np.float64)
    truth = np.asarray(physical_truth, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape or prediction.shape != (weights.size, 4):
        raise ValueError("structural diagnostic arrays are inconsistent")
    scales = normalization.output_scales
    normalized_prediction = prediction / scales
    normalized_truth = truth / scales
    v = np.asarray((float(beta2), 1.0, -1.0, 0.0), dtype=np.float64)
    normalized_v = v / scales
    coefficient = (normalized_prediction @ normalized_v) / float(
        normalized_v @ normalized_v
    )
    residual = normalized_prediction - coefficient[:, None] * normalized_v[None, :]
    weighted = lambda values: float(
        np.sqrt(np.sum(weights * values * values) / np.sum(weights))
    )
    water = prediction[:, 1] + prediction[:, 2] + prediction[:, 3]
    beta_defect = prediction[:, 0] - float(beta2) * prediction[:, 1]
    dot = np.sum(weights[:, None] * normalized_prediction * normalized_truth)
    pred_norm = np.sqrt(np.sum(weights[:, None] * normalized_prediction**2))
    truth_norm = np.sqrt(np.sum(weights[:, None] * normalized_truth**2))
    return {
        "water_source_defect_rms": weighted(water),
        "beta_source_defect_rms": weighted(beta_defect),
        "spurious_Qr_t_rms": weighted(prediction[:, 3]),
        "normalized_manifold_residual_rms": float(
            np.sqrt(np.sum(weights[:, None] * residual**2) / np.sum(weights))
        ),
        "normalized_vector_cosine": float(
            dot / max(pred_norm * truth_norm, np.finfo(np.float64).tiny)
        ),
        "component_physical_rms_error": {
            name: weighted(prediction[:, i] - truth[:, i])
            for i, name in enumerate(SOURCE_ORDER)
        },
        "diagnostic_metric_sha256": _canonical_sha256(
            {
                "source_order": SOURCE_ORDER,
                "output_scales": scales.tolist(),
                "manifold_vector": v.tolist(),
            }
        ),
    }


__all__ = (
    "FourTendencyNormalization",
    "NeuralFourTendencyMoistPhysics",
    "PARAMETER_COUNT",
    "PHYSICS_MODE_NEURAL_FOUR_TENDENCY",
    "ProblemBMLPConfiguration",
    "ProblemBOperatorDataset",
    "ProblemBOperatorObjective",
    "SOURCE_ORDER",
    "build_problem_b_model",
    "initial_problem_b_parameters",
    "load_problem_b_parameters",
    "normalization_from_record",
    "parameter_pytree_sha256",
    "save_problem_b_parameters",
    "structural_diagnostics",
)
