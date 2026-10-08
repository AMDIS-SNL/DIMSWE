"""Pure JAX derivatives of the certified local moist source density.

The finite-element maps surrounding this kernel live in
``dimswe.jax_moist_hvp``.  This module deliberately imports no Firedrake
objects and differentiates the unchanged J1 source-density function directly.
Moist parameters and topography are explicit, separate pytrees, but are held
fixed by the full J2 production-oracle comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_moist import moist_source_density_jax


_FLOAT64 = jnp.dtype(jnp.float64)
_STATE_KEYS = ("h", "S", "Qv", "Qc")
_SOURCE_KEYS = ("S", "Qv", "Qc", "Qr")


def _require_float64_tree(name, values, keys):
    """Return a keyed float64 pytree without admitting diagnostic leaves."""
    result = {}
    for key in keys:
        value = jnp.asarray(values[key])
        if value.dtype != _FLOAT64:
            raise TypeError(
                f"{name}['{key}'] must have dtype float64, got {value.dtype}"
            )
        result[key] = value
    return result


def moist_source_jvp(state_q, dstate_q, fields_q, parameters):
    """Return the J1 source density and its state-directional derivative.

    Only ``state_q`` is an active differentiation argument.  The topography
    and deployed moist parameters remain fixed, so their tangents are
    structural zeros in the certified J2 full-child comparison.
    """
    state = _require_float64_tree("state_q", state_q, _STATE_KEYS)
    direction = _require_float64_tree("dstate_q", dstate_q, _STATE_KEYS)

    def source(active_state):
        return moist_source_density_jax(active_state, fields_q, parameters)

    return jax.jvp(source, (state,), (direction,))


def moist_source_vjp(
    state_q,
    source_covector_q,
    fields_q,
    parameters,
):
    """Apply the ordinary transpose of the local J1 source Jacobian."""
    state = _require_float64_tree("state_q", state_q, _STATE_KEYS)
    source_covector = _require_float64_tree(
        "source_covector_q", source_covector_q, _SOURCE_KEYS
    )

    def source(active_state):
        return moist_source_density_jax(active_state, fields_q, parameters)

    _, pullback = jax.vjp(source, state)
    return pullback(source_covector)[0]


def moist_source_differentiated_vjp(
    state_q,
    source_covector_q,
    dstate_q,
    dsource_covector_q,
    fields_q,
    parameters,
):
    """Return a local VJP and its joint state/covector directional derivative.

    The second result is exactly

    ``J(q).T @ dbar_source + D[J(q).T @ bar_source][dq]``.

    No dense Jacobian or Hessian is formed and no pullback closure escapes this
    call.
    """
    state = _require_float64_tree("state_q", state_q, _STATE_KEYS)
    source_covector = _require_float64_tree(
        "source_covector_q", source_covector_q, _SOURCE_KEYS
    )
    direction = _require_float64_tree("dstate_q", dstate_q, _STATE_KEYS)
    covector_direction = _require_float64_tree(
        "dsource_covector_q", dsource_covector_q, _SOURCE_KEYS
    )

    def vjp_map(active_state, active_source_covector):
        def source(local_state):
            return moist_source_density_jax(
                local_state, fields_q, parameters
            )

        _, pullback = jax.vjp(source, active_state)
        return pullback(active_source_covector)[0]

    return jax.jvp(
        vjp_map,
        (state, source_covector),
        (direction, covector_direction),
    )


moist_source_jvp_jit = jax.jit(moist_source_jvp)
moist_source_vjp_jit = jax.jit(moist_source_vjp)
moist_source_differentiated_vjp_jit = jax.jit(
    moist_source_differentiated_vjp
)


__all__ = (
    "moist_source_jvp",
    "moist_source_jvp_jit",
    "moist_source_vjp",
    "moist_source_vjp_jit",
    "moist_source_differentiated_vjp",
    "moist_source_differentiated_vjp_jit",
)
