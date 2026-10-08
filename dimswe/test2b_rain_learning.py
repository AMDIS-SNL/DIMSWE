"""Rain-active Test2B learned-moist representation contracts.

The frozen A/B/C representations and the BTPL/BTP follow-ups share the
deployed feature vector ``(h,S,Qv,Qc,B)`` and differ only in their
output/source map.  ``BPLUS`` remains a backward-compatible identifier for
the originally prepared linear-exceedance BTPL map.
This module contains no optimizer and cannot advance a production trajectory
on its own.
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

from .jax_moist import moist_diagnostics_jax, moist_rates_jax
from .learned_physics.parameters import tree_copy, validate_float64_tree
from .test2a_operator import DenseMLP, initialize_mlp
from .test2a_embedded_moist import parameter_pytree_sha256


FEATURE_ORDER = ("h", "S", "Qv", "Qc", "B")
SOURCE_ORDER = ("S", "Qv", "Qc", "Qr")
STATE_KEYS = ("h", "S", "Qv", "Qc")
STRUCTURED_RAIN_VARIANTS = ("BPLUS", "BTPL", "BTP")
LINEAR_EXCEEDANCE_VARIANTS = ("BPLUS", "BTPL")
REPRESENTATIONS = ("A", "B", "C", *STRUCTURED_RAIN_VARIANTS)
PHYSICS_MODES = {
    "A": "neural_A_original_R",
    "B": "neural_A_R",
    "C": "neural_four_tendency",
    "BPLUS": "neural_A_threshold_nonnegative_R",
    "BTPL": "neural_A_threshold_nonnegative_R",
    "BTP": "neural_A_threshold_positive_gate_R",
}
OUTPUT_ORDERS = {
    "A": ("A",),
    "B": ("A", "R"),
    "C": SOURCE_ORDER,
    "BPLUS": ("A", "R_TPL"),
    "BTPL": ("A", "R_TPL"),
    "BTP": ("A", "R_TP"),
}
PARAMETER_COUNTS = {
    "A": 1281, "B": 1314, "C": 1380,
    "BPLUS": 1314, "BTPL": 1314, "BTP": 1314,
}
_FLOAT64 = jnp.dtype(jnp.float64)


def canonical_sha256(record):
    return sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RainMLPConfiguration:
    representation: str
    input_dimension: int = 5
    hidden_layers: tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    dtype: str = "float64"
    seed: int = 0

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS:
            raise ValueError("unsupported Test2B representation")
        if self.input_dimension != 5 or self.hidden_layers != (32, 32):
            raise ValueError("Test2B architecture family is 5->32->32->d")
        if self.activation != "tanh" or self.dtype != "float64" or self.seed != 0:
            raise ValueError("Test2B requires tanh, float64, seed 0")
        if self.parameter_count != PARAMETER_COUNTS[self.representation]:
            raise AssertionError("Test2B parameter-count contract changed")

    @property
    def output_dimension(self):
        return len(OUTPUT_ORDERS[self.representation])

    @property
    def layer_dimensions(self):
        return (self.input_dimension, *self.hidden_layers, self.output_dimension)

    @property
    def parameter_count(self):
        return int(
            sum(a * b + b for a, b in zip(self.layer_dimensions[:-1], self.layer_dimensions[1:]))
        )

    def to_record(self):
        return {
            "representation": self.representation,
            "features": list(FEATURE_ORDER),
            "outputs": list(OUTPUT_ORDERS[self.representation]),
            "layers": list(self.layer_dimensions),
            "activation": self.activation,
            "dtype": self.dtype,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class RainLearningNormalization:
    """Training-only feature and output coordinates for all representations."""

    input_offset: np.ndarray
    input_scale: np.ndarray
    sigma_a: float
    sigma_r_active: float
    source_scales: np.ndarray
    provenance_sha256: str
    bplus_delta_q_scale: float | None = None
    bplus_q_precip: float | None = None
    bplus_provenance_sha256: str | None = None
    btp_q_precip: float | None = None
    btp_provenance_sha256: str | None = None

    def __post_init__(self):
        offset = np.asarray(self.input_offset, dtype=np.float64)
        scale = np.asarray(self.input_scale, dtype=np.float64)
        source = np.asarray(self.source_scales, dtype=np.float64)
        if offset.shape != (5,) or scale.shape != (5,) or np.any(scale <= 0.0):
            raise ValueError("five finite positive feature scales are required")
        if source.shape != (4,) or np.any(source <= 0.0):
            raise ValueError("four finite positive source scales are required")
        if self.sigma_a <= 0.0 or self.sigma_r_active <= 0.0:
            raise ValueError("A and active-R scales must be positive")
        bplus = (
            self.bplus_delta_q_scale,
            self.bplus_q_precip,
            self.bplus_provenance_sha256,
        )
        if any(value is not None for value in bplus):
            if any(value is None for value in bplus):
                raise ValueError("BPLUS output metadata must be complete")
            if self.bplus_delta_q_scale <= 0.0:
                raise ValueError("BPLUS delta_q_scale must be positive")
            if self.bplus_q_precip <= 0.0:
                raise ValueError("BPLUS q_precip must be positive")
            if len(self.bplus_provenance_sha256) != 64:
                raise ValueError("BPLUS provenance fingerprint must be SHA256")
        btp = (self.btp_q_precip, self.btp_provenance_sha256)
        if any(value is not None for value in btp):
            if any(value is None for value in btp):
                raise ValueError("BTP output metadata must be complete")
            if self.btp_q_precip <= 0.0:
                raise ValueError("BTP q_precip must be positive")
            if len(self.btp_provenance_sha256) != 64:
                raise ValueError("BTP provenance fingerprint must be SHA256")
        for name, value in (("input_offset", offset), ("input_scale", scale), ("source_scales", source)):
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

    def normalize_features(self, values):
        return (jnp.asarray(values, dtype=jnp.float64) - jnp.asarray(self.input_offset)) / jnp.asarray(self.input_scale)

    def output_scales(self, representation):
        if representation == "A":
            return np.asarray((self.sigma_a,), dtype=np.float64)
        if representation == "B" or representation in STRUCTURED_RAIN_VARIANTS:
            return np.asarray((self.sigma_a, self.sigma_r_active), dtype=np.float64)
        if representation == "C":
            return self.source_scales
        raise ValueError("unsupported Test2B representation")

    def to_record(self):
        record = {
            "feature_order": list(FEATURE_ORDER),
            "input_offset": self.input_offset.tolist(),
            "input_scale": self.input_scale.tolist(),
            "sigma_A_all_training_support": float(self.sigma_a),
            "sigma_R_active_training_support": float(self.sigma_r_active),
            "source_order": list(SOURCE_ORDER),
            "source_scales": self.source_scales.tolist(),
            "R_scale_definition": "mass-weighted RMS over meaningful R>0 samples; all zero samples remain in every loss numerator",
            "fitted_truth_state_indices": [0, 80],
            "heldout_state_indices": [81, 160],
            "provenance_sha256": self.provenance_sha256,
        }
        if self.bplus_delta_q_scale is not None:
            record["BTPL"] = {
                "delta_q_scale": float(self.bplus_delta_q_scale),
                "q_precip": float(self.bplus_q_precip),
                "provenance_sha256": self.bplus_provenance_sha256,
            }
        if self.btp_q_precip is not None:
            record["BTP"] = {
                "q_precip": float(self.btp_q_precip),
                "provenance_sha256": self.btp_provenance_sha256,
            }
        return record


def bplus_physical_rates(raw_output, h, qc_total, q_precip, normalization):
    """Legacy BPLUS/BTPL linear-exceedance physical rate map.

    The hard threshold is intentional.  It preserves an exactly dry region;
    callers must not replace it with a smooth approximation.
    """
    if normalization.bplus_delta_q_scale is None:
        raise ValueError("BPLUS output metadata is required")
    raw = jnp.asarray(raw_output, dtype=jnp.float64)
    if raw.shape[-1] != 2:
        raise ValueError("BPLUS requires exactly two raw outputs")
    h = jnp.asarray(h, dtype=jnp.float64)
    qc_total = jnp.asarray(qc_total, dtype=jnp.float64)
    q_precip = jnp.asarray(q_precip, dtype=jnp.float64)
    delta_factor = jnp.maximum(
        jnp.asarray(0.0, dtype=jnp.float64),
        (qc_total / h - q_precip) / normalization.bplus_delta_q_scale,
    )
    a = normalization.sigma_a * raw[..., 0]
    r = (
        normalization.sigma_r_active
        * delta_factor
        * jax.nn.softplus(raw[..., 1])
        / jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    )
    return a, r


btpl_physical_rates = bplus_physical_rates


def btp_physical_rates(raw_output, h, qc_total, q_precip, normalization):
    """Map raw coordinates through the threshold/positivity-only BTP gate."""
    if normalization.btp_q_precip is None:
        raise ValueError("BTP output metadata is required")
    raw = jnp.asarray(raw_output, dtype=jnp.float64)
    if raw.shape[-1] != 2:
        raise ValueError("BTP requires exactly two raw outputs")
    h = jnp.asarray(h, dtype=jnp.float64)
    qc_total = jnp.asarray(qc_total, dtype=jnp.float64)
    q_precip = jnp.asarray(q_precip, dtype=jnp.float64)
    a = normalization.sigma_a * raw[..., 0]
    active_r = (
        normalization.sigma_r_active
        * jax.nn.softplus(raw[..., 1])
        / jnp.log(jnp.asarray(2.0, dtype=jnp.float64))
    )
    r = jnp.where(qc_total / h > q_precip, active_r, 0.0)
    return a, r


def structured_rain_physical_rates(
    representation, raw_output, h, qc_total, q_precip, normalization
):
    if representation in LINEAR_EXCEEDANCE_VARIANTS:
        return btpl_physical_rates(
            raw_output, h, qc_total, q_precip, normalization
        )
    if representation == "BTP":
        return btp_physical_rates(
            raw_output, h, qc_total, q_precip, normalization
        )
    raise ValueError("representation is not a structured rain-output variant")


def build_model(representation):
    return DenseMLP(RainMLPConfiguration(representation))


def initial_parameters(representation):
    configuration = RainMLPConfiguration(representation)
    parameters = initialize_mlp(configuration)
    count = sum(int(x.size) for x in jax.tree_util.tree_leaves(parameters))
    if count != configuration.parameter_count:
        raise AssertionError("initialized parameter count changed")
    return parameters


def save_parameters(path, representation, parameters, *, metadata=None):
    configuration = RainMLPConfiguration(representation)
    owned = validate_float64_tree(parameters, name="Test2B parameters")
    destination = Path(path)
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite Test2B parameters")
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
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def load_parameters(path, representation):
    source = Path(path)
    record = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    configuration = RainMLPConfiguration(representation)
    if record["architecture"] != configuration.to_record():
        raise ValueError("Test2B parameter architecture changed")
    layers = []
    with np.load(source, allow_pickle=False) as archive:
        for index, (fan_in, fan_out) in enumerate(zip(configuration.layer_dimensions[:-1], configuration.layer_dimensions[1:])):
            weight = np.asarray(archive[f"layer_{index:03d}_weight"])
            bias = np.asarray(archive[f"layer_{index:03d}_bias"])
            if weight.shape != (fan_in, fan_out) or bias.shape != (fan_out,):
                raise ValueError("Test2B parameter leaf shape changed")
            if weight.dtype != np.float64 or bias.dtype != np.float64:
                raise TypeError("Test2B parameters must be float64")
            layers.append({"weight": jnp.asarray(weight), "bias": jnp.asarray(bias)})
    parameters = {"layers": tuple(layers)}
    if parameter_pytree_sha256(parameters) != record["parameter_pytree_sha256"]:
        raise ValueError("Test2B parameter fingerprint mismatch")
    return parameters, record


def _require_tree(values, keys, name):
    result = {}
    for key in keys:
        value = jnp.asarray(values[key])
        if value.dtype != _FLOAT64:
            raise TypeError(f"{name}['{key}'] must be float64")
        result[key] = value
    return result


class RainActiveNeuralMoistPhysics:
    """Exact local source provider for any frozen Test2B representation."""

    def __init__(self, representation, parameters, normalization, *, use_jit=True, provenance=None):
        self.representation = representation
        self.physics_mode = PHYSICS_MODES[representation]
        self.configuration = RainMLPConfiguration(representation)
        self.model = build_model(representation)
        self.normalization = normalization
        self.use_jit = bool(use_jit)
        self.provenance = MappingProxyType(dict(provenance or {}))
        self._parameters = validate_float64_tree(parameters, name="Test2B parameters")
        count = sum(int(x.size) for x in jax.tree_util.tree_leaves(self._parameters))
        if count != self.configuration.parameter_count:
            raise ValueError("Test2B provider parameter count changed")
        scales = jnp.asarray(normalization.output_scales(representation), dtype=jnp.float64)

        def combined(state, fields, moist_parameters, neural_parameters):
            state = _require_tree(state, STATE_KEYS, "state_q")
            fields = _require_tree(fields, ("B",), "fields_q")
            moist_parameters = _require_tree(
                moist_parameters,
                ("g", "q0", "H0", "gamma_r", "qprecip", "L", "configured_dt"),
                "moist_parameters",
            )
            neural_parameters = validate_float64_tree(neural_parameters, name="neural_parameters")
            features = jnp.stack((state["h"], state["S"], state["Qv"], state["Qc"], fields["B"]), axis=-1)
            raw = self.model(
                neural_parameters, normalization.normalize_features(features)
            )
            h = state["h"]
            beta2 = moist_parameters["g"] * moist_parameters["L"]
            if representation in STRUCTURED_RAIN_VARIANTS:
                a, r = structured_rain_physical_rates(
                    representation,
                    raw, h, state["Qc"], moist_parameters["qprecip"],
                    normalization,
                )
                source = {
                    "S": h * beta2 * a,
                    "Qv": h * a,
                    "Qc": -h * (a + r),
                    "Qr": h * r,
                }
                rates = {"A": a, "R": r}
            else:
                # Keep the accepted A/B/C scaling operation unchanged.
                physical = raw * scales
            if representation == "A":
                a = physical[..., 0]
                r = moist_rates_jax(state, fields, moist_parameters)["R"]
                source = {"S": h * beta2 * a, "Qv": h * a, "Qc": -h * (a + r), "Qr": h * r}
                rates = {"A": a, "R": r}
            elif representation == "B":
                a, r = physical[..., 0], physical[..., 1]
                source = {"S": h * beta2 * a, "Qv": h * a, "Qc": -h * (a + r), "Qr": h * r}
                rates = {"A": a, "R": r}
            elif representation == "C":
                source = {name: physical[..., index] for index, name in enumerate(SOURCE_ORDER)}
                rates = {}
            return {"rates": rates, "source": source}

        self._combined_explicit = combined
        self.combined_parameterized_kernel = jax.jit(combined) if use_jit else combined
        frozen = lambda state, fields, moist_parameters: combined(state, fields, moist_parameters, self._parameters)
        self.combined_kernel = jax.jit(frozen) if use_jit else frozen

        def diagnostics(state, fields, moist_parameters, neural_parameters):
            result = combined(state, fields, moist_parameters, neural_parameters)
            if representation == "A":
                values = dict(moist_diagnostics_jax(state, fields, moist_parameters))
                values.update(result["rates"])
                return values
            reference = jnp.asarray(state["h"], dtype=jnp.float64)
            false = jnp.zeros_like(reference, dtype=bool)
            one = jnp.ones_like(reference)
            values = {
                "condensation_mask": false, "evaporation_mask": false,
                "uncapped_evaporation_mask": false, "rain_mask": false,
                "condensation_margin": one, "evaporation_margin": one,
                "evaporation_cap_margin": one, "rain_margin": one,
                "depth_denominator_margin": one,
            }
            if representation in STRUCTURED_RAIN_VARIANTS:
                delta_q = state["Qc"] / state["h"] - moist_parameters["qprecip"]
                values.update(result["rates"])
                values["rain_mask"] = delta_q > 0.0
                values["rain_margin"] = delta_q
            return values

        self.diagnostic_parameterized_kernel = jax.jit(diagnostics) if use_jit else diagnostics
        frozen_diagnostics = lambda state, fields, moist_parameters: diagnostics(state, fields, moist_parameters, self._parameters)
        self.diagnostic_kernel = jax.jit(frozen_diagnostics) if use_jit else frozen_diagnostics

        def state_jvp(state, direction, fields, moist_parameters):
            source = lambda active: combined(active, fields, moist_parameters, self._parameters)["source"]
            return jax.jvp(source, (_require_tree(state, STATE_KEYS, "state_q"),), (_require_tree(direction, STATE_KEYS, "dstate_q"),))

        def state_vjp(state, source_covector, fields, moist_parameters):
            active = _require_tree(state, STATE_KEYS, "state_q")
            covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
            source = lambda value: combined(value, fields, moist_parameters, self._parameters)["source"]
            _, pullback = jax.vjp(source, active)
            return pullback(covector)[0]

        def state_differentiated_vjp(state, source_covector, direction, source_covector_direction, fields, moist_parameters):
            active = _require_tree(state, STATE_KEYS, "state_q")
            covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
            dstate = _require_tree(direction, STATE_KEYS, "dstate_q")
            dcovector = _require_tree(source_covector_direction, SOURCE_ORDER, "dsource_covector_q")
            def vjp_map(local_state, local_covector):
                source = lambda value: combined(value, fields, moist_parameters, self._parameters)["source"]
                _, pullback = jax.vjp(source, local_state)
                return pullback(local_covector)[0]
            return jax.jvp(vjp_map, (active, covector), (dstate, dcovector))

        self.state_jvp_kernel = jax.jit(state_jvp) if use_jit else state_jvp
        self.state_vjp_kernel = jax.jit(state_vjp) if use_jit else state_vjp
        self.state_differentiated_vjp_kernel = jax.jit(state_differentiated_vjp) if use_jit else state_differentiated_vjp

    @property
    def parameters(self):
        return tree_copy(self._parameters)

    def combined_with_parameters(self, state, fields, moist_parameters, neural_parameters):
        return self._combined_explicit(state, fields, moist_parameters, neural_parameters)

    def parameter_jvp(self, state, direction, fields, moist_parameters, *, base_parameters=None):
        base = self._parameters if base_parameters is None else validate_float64_tree(base_parameters, name="base_parameters")
        direction = validate_float64_tree(direction, name="parameter_direction")
        source = lambda active: self._combined_explicit(state, fields, moist_parameters, active)["source"]
        return jax.jvp(source, (base,), (direction,))

    def parameter_vjp(self, state, source_covector, fields, moist_parameters, *, base_parameters=None):
        base = self._parameters if base_parameters is None else validate_float64_tree(base_parameters, name="base_parameters")
        covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
        source = lambda active: self._combined_explicit(state, fields, moist_parameters, active)["source"]
        _, pullback = jax.vjp(source, base)
        return pullback(covector)[0]

    def joint_differentiated_vjp(self, state, source_covector, state_direction, parameter_direction, source_covector_direction, fields, moist_parameters, *, base_parameters=None):
        active = _require_tree(state, STATE_KEYS, "state_q")
        covector = _require_tree(source_covector, SOURCE_ORDER, "source_covector_q")
        dstate = _require_tree(state_direction, STATE_KEYS, "dstate_q")
        dparameters = validate_float64_tree(parameter_direction, name="parameter_direction")
        dcovector = _require_tree(source_covector_direction, SOURCE_ORDER, "dsource_covector_q")
        base = self._parameters if base_parameters is None else validate_float64_tree(base_parameters, name="base_parameters")
        def vjp_map(local_state, local_parameters, local_covector):
            source = lambda state_value, parameter_value: self._combined_explicit(state_value, fields, moist_parameters, parameter_value)["source"]
            _, pullback = jax.vjp(source, local_state, local_parameters)
            return pullback(local_covector)
        return jax.jvp(vjp_map, (active, base, covector), (dstate, dparameters, dcovector))


def source_invariant_diagnostics(source, beta2):
    """Read-only structural diagnostics; no value is projected or corrected."""
    values = {name: np.asarray(source[name], dtype=np.float64) for name in SOURCE_ORDER}
    water = values["Qv"] + values["Qc"] + values["Qr"]
    thermo = values["S"] - float(beta2) * values["Qv"]
    return {
        "water_maximum_absolute": float(np.max(np.abs(water))),
        "water_rms": float(np.sqrt(np.mean(water * water))),
        "S_minus_beta2_Qv_maximum_absolute": float(np.max(np.abs(thermo))),
        "S_minus_beta2_Qv_rms": float(np.sqrt(np.mean(thermo * thermo))),
    }


def structural_diagnostics(prediction, truth, beta2, source_scales, weights=None):
    """Normalized two-rate-manifold and component diagnostics, never a projection."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 4)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth source arrays differ")
    scales = np.asarray(source_scales, dtype=np.float64)
    weights = np.ones(prediction.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if scales.shape != (4,) or weights.shape != (prediction.shape[0],):
        raise ValueError("structural diagnostic metric shape changed")
    normalized = prediction / scales
    normalized_truth = truth / scales
    basis = np.asarray(((float(beta2), 1.0, -1.0, 0.0), (0.0, 0.0, -1.0, 1.0))).T / scales[:, None]
    coefficients = np.linalg.solve(basis.T @ basis, basis.T @ normalized.T).T
    residual = normalized - coefficients @ basis.T
    weighted_sum = float(np.sum(weights))
    rms = lambda values: float(np.sqrt(np.sum(weights[:, None] * np.asarray(values) ** 2) / weighted_sum))
    dot = float(np.sum(weights[:, None] * normalized * normalized_truth))
    pred_norm = float(np.sqrt(np.sum(weights[:, None] * normalized**2)))
    truth_norm = float(np.sqrt(np.sum(weights[:, None] * normalized_truth**2)))
    water = prediction[:, 1] + prediction[:, 2] + prediction[:, 3]
    thermo = prediction[:, 0] - float(beta2) * prediction[:, 1]
    return {
        "water_source_defect_rms": float(np.sqrt(np.sum(weights * water**2) / weighted_sum)),
        "S_minus_beta2_Qv_defect_rms": float(np.sqrt(np.sum(weights * thermo**2) / weighted_sum)),
        "normalized_two_rate_manifold_residual_rms": rms(residual),
        "normalized_source_vector_cosine": dot / max(pred_norm * truth_norm, np.finfo(np.float64).tiny),
        "component_physical_RMS_error": {name: float(np.sqrt(np.sum(weights * (prediction[:, index] - truth[:, index]) ** 2) / weighted_sum)) for index, name in enumerate(SOURCE_ORDER)},
        "rain_source_RMS_error": float(np.sqrt(np.sum(weights * (prediction[:, 3] - truth[:, 3]) ** 2) / weighted_sum)),
        "negative_Qr_t_fraction": float(np.mean(prediction[:, 3] < 0.0)),
        "metric_sha256": canonical_sha256({"source_order": SOURCE_ORDER, "source_scales": scales.tolist(), "basis": basis.tolist()}),
    }


__all__ = (
    "FEATURE_ORDER", "LINEAR_EXCEEDANCE_VARIANTS", "OUTPUT_ORDERS",
    "PARAMETER_COUNTS", "PHYSICS_MODES",
    "REPRESENTATIONS", "RainActiveNeuralMoistPhysics", "RainLearningNormalization",
    "RainMLPConfiguration", "SOURCE_ORDER", "STRUCTURED_RAIN_VARIANTS",
    "bplus_physical_rates", "btp_physical_rates", "btpl_physical_rates",
    "build_model", "canonical_sha256",
    "initial_parameters", "source_invariant_diagnostics",
    "load_parameters", "save_parameters", "structural_diagnostics",
    "structured_rain_physical_rates",
)
