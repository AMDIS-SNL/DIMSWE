"""Pure JAX replica of the deployed local moist-physics algebra.

This module deliberately has no Firedrake dependency.  It represents only the
pointwise rate/source calculation; finite-element interpolation, weak assembly,
the mixed mass solve, and the applied Euler timestep belong to
``dimswe.jax_moist_adapter``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class JAXMoistConfigurationError(RuntimeError):
    """Raised when JAX cannot reproduce the float64 production arithmetic."""


def _require_x64() -> None:
    if not bool(jax.config.read("jax_enable_x64")):
        raise JAXMoistConfigurationError(
            "DIMSWE JAX moist physics requires jax_enable_x64=True; "
            "enable JAX 64-bit mode before importing dimswe.jax_moist"
        )


_require_x64()

_FLOAT64 = jnp.dtype(jnp.float64)
_ZERO = jnp.float64(0.0)
_ONE = jnp.float64(1.0)
_TWENTY = jnp.float64(20.0)

_STATE_KEYS = ("h", "S", "Qv", "Qc")
_FIELD_KEYS = ("B",)
_PARAMETER_KEYS = (
    "g",
    "q0",
    "H0",
    "gamma_r",
    "qprecip",
    "L",
    "configured_dt",
)


def _float64_leaf(container_name, key, value):
    array = jnp.asarray(value)
    if array.dtype != _FLOAT64:
        raise TypeError(
            f"{container_name}['{key}'] must have dtype float64, "
            f"got {array.dtype}"
        )
    return array


def _validated_inputs(state_q, fields_q, parameters):
    _require_x64()
    state = {
        key: _float64_leaf("state_q", key, state_q[key])
        for key in _STATE_KEYS
    }
    fields = {
        key: _float64_leaf("fields_q", key, fields_q[key])
        for key in _FIELD_KEYS
    }
    params = {
        key: _float64_leaf("parameters", key, parameters[key])
        for key in _PARAMETER_KEYS
    }
    return state, fields, params


def _moist_algebra(state_q, fields_q, parameters):
    """Evaluate the deployed algebra once and return float-valued quantities."""
    state, fields, params = _validated_inputs(state_q, fields_q, parameters)

    h = state["h"]
    entropy_density = state["S"]
    vapour_density = state["Qv"]
    cloud_density = state["Qc"]
    topography = fields["B"]

    gravity = params["g"]
    q0 = params["q0"]
    reference_depth = params["H0"]
    rain_rate = params["gamma_r"]
    precipitation_threshold = params["qprecip"]
    latent_ratio = params["L"]
    configured_dt = params["configured_dt"]

    qv = vapour_density / h
    qc = cloud_density / h
    specific_entropy = entropy_density / h

    beta2 = gravity * latent_ratio
    depth_denominator = h + topography
    qsat = (
        q0
        * reference_depth
        / depth_denominator
        * jnp.exp(
            _TWENTY * (_ONE - specific_entropy / gravity)
        )
    )
    gamma_v_denominator = (
        _ONE + _TWENTY * qsat * beta2 / gravity
    )
    gamma_v = _ONE / gamma_v_denominator

    # The deployed constructor sets both relaxation times to configured_dt.
    tau_v = configured_dt
    tau_r = configured_dt

    condensation_argument = gamma_v * (qv - qsat) / tau_v
    condensation = jnp.where(
        condensation_argument < _ZERO,
        _ZERO,
        condensation_argument,
    )

    evaporation_argument = gamma_v * (qsat - qv) / tau_v
    evaporation_positive = jnp.where(
        evaporation_argument < _ZERO,
        _ZERO,
        evaporation_argument,
    )
    evaporation_cap = qc / configured_dt
    evaporation = jnp.where(
        evaporation_cap < evaporation_positive,
        evaporation_cap,
        evaporation_positive,
    )

    rain_argument = (
        rain_rate * (qc - precipitation_threshold) / tau_r
    )
    rain = jnp.where(rain_argument < _ZERO, _ZERO, rain_argument)
    net_vapour_rate = evaporation - condensation

    return {
        "h": h,
        "qv": qv,
        "qc": qc,
        "s": specific_entropy,
        "beta2": beta2,
        "depth_denominator": depth_denominator,
        "qsat": qsat,
        "gamma_v_denominator": gamma_v_denominator,
        "gamma_v": gamma_v,
        "condensation_argument": condensation_argument,
        "evaporation_argument": evaporation_argument,
        "evaporation_cap": evaporation_cap,
        "evaporation_cap_difference": (
            evaporation_cap - evaporation_positive
        ),
        "rain_argument": rain_argument,
        "C": condensation,
        "E_positive": evaporation_positive,
        "E": evaporation,
        "A": net_vapour_rate,
        "R": rain,
    }


def moist_rates_jax(state_q, fields_q, parameters):
    """Return only the differentiable deployed net-vapour and rain rates."""
    algebra = _moist_algebra(state_q, fields_q, parameters)
    return {"A": algebra["A"], "R": algebra["R"]}


def moist_rates_and_source_density_jax(state_q, fields_q, parameters):
    """Return rates and coupled sources from one evaluation of the algebra."""
    algebra = _moist_algebra(state_q, fields_q, parameters)
    h = algebra["h"]
    net_vapour_rate = algebra["A"]
    rain = algebra["R"]
    return {
        "rates": {"A": net_vapour_rate, "R": rain},
        "source": {
            "S": h * algebra["beta2"] * net_vapour_rate,
            "Qv": h * net_vapour_rate,
            "Qc": -h * (net_vapour_rate + rain),
            "Qr": h * rain,
        },
    }


def moist_source_density_jax(state_q, fields_q, parameters):
    """Return the four invariant-coupled source densities.

    This evaluates the rate algebra once.  The velocity and depth source
    blocks are structural zeros supplied by the Firedrake adapter.
    """
    algebra = _moist_algebra(state_q, fields_q, parameters)
    h = algebra["h"]
    net_vapour_rate = algebra["A"]
    rain = algebra["R"]
    return {
        "S": h * algebra["beta2"] * net_vapour_rate,
        "Qv": h * net_vapour_rate,
        "Qc": -h * (net_vapour_rate + rain),
        "Qr": h * rain,
    }


def moist_diagnostics_jax(state_q, fields_q, parameters):
    """Return nondifferentiated inspection data, masks, and local margins."""
    algebra = _moist_algebra(state_q, fields_q, parameters)
    diagnostics = dict(algebra)
    diagnostics.update(
        {
            "condensation_mask": (
                algebra["condensation_argument"] > _ZERO
            ),
            "evaporation_mask": (
                algebra["evaporation_argument"] > _ZERO
            ),
            "uncapped_evaporation_mask": (
                algebra["evaporation_cap_difference"] > _ZERO
            ),
            "rain_mask": algebra["rain_argument"] > _ZERO,
            "condensation_margin": jnp.min(
                jnp.abs(algebra["condensation_argument"])
            ),
            "evaporation_margin": jnp.min(
                jnp.abs(algebra["evaporation_argument"])
            ),
            "evaporation_cap_margin": jnp.min(
                jnp.abs(algebra["evaporation_cap_difference"])
            ),
            "rain_margin": jnp.min(jnp.abs(algebra["rain_argument"])),
            "depth_denominator_margin": jnp.min(
                jnp.abs(algebra["depth_denominator"])
            ),
        }
    )
    return diagnostics


moist_rates_jit = jax.jit(moist_rates_jax)
moist_rates_and_source_density_jit = jax.jit(
    moist_rates_and_source_density_jax
)
moist_source_density_jit = jax.jit(moist_source_density_jax)
moist_diagnostics_jit = jax.jit(moist_diagnostics_jax)


__all__ = (
    "JAXMoistConfigurationError",
    "moist_rates_jax",
    "moist_rates_jit",
    "moist_rates_and_source_density_jax",
    "moist_rates_and_source_density_jit",
    "moist_source_density_jax",
    "moist_source_density_jit",
    "moist_diagnostics_jax",
    "moist_diagnostics_jit",
)
