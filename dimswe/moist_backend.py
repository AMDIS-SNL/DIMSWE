"""Narrow runtime dispatch for the final production moist Euler child.

The default UFL child is returned unchanged.  The opt-in JAX wrapper owns the
certified J1 primal adapter while retaining the independently constructed UFL
Euler object as an oracle and as coefficient storage.  No other Euler child or
forcing term is affected by this module.
"""

from __future__ import annotations

from dataclasses import dataclass


MOIST_BACKENDS = ("ufl", "jax")


def validate_moist_backend(value):
    """Return a validated backend name without coercing arbitrary values."""
    if not isinstance(value, str):
        raise TypeError("moist_backend must be the string 'ufl' or 'jax'")
    if value not in MOIST_BACKENDS:
        raise ValueError(
            f"invalid moist_backend {value!r}; expected 'ufl' or 'jax'"
        )
    return value


@dataclass(frozen=True)
class JAXMoistFixedControlReverseResult:
    """JAX moist reverse plus its structural physical-c0 zero."""

    derivative: object
    c0_gradient: float = 0.0

    def __getattr__(self, name):
        return getattr(self.derivative, name)


@dataclass(frozen=True)
class JAXMoistFixedControlHVPResult:
    """JAX moist incremental reverse plus structural physical-c0 zeros."""

    derivative: object
    ordinary: JAXMoistFixedControlReverseResult
    c0_hvp: float = 0.0

    def __getattr__(self, name):
        return getattr(self.derivative, name)


class JAXMoistEulerIntegrator:
    """Production-child facade backed only by the certified J1 primal map."""

    moist_backend = "jax"

    def __init__(self, ufl_oracle, *, local_physics=None):
        if ufl_oracle.__class__.__name__ != "Euler":
            raise ValueError("JAX moist backend requires the production Euler")
        if ufl_oracle.terms != ["threewayphysics"]:
            raise ValueError(
                "JAX moist backend requires terms=['threewayphysics'] exactly"
            )
        matches = [
            term
            for term in ufl_oracle.model.dynamics.forcing_terms
            if term.name == "threewayphysics"
        ]
        if len(matches) != 1 or matches[0].treat_as_coeffs:
            raise ValueError(
                "JAX complete-split backend requires one fixed-parameter "
                "ThreeWayPhysics oracle"
            )

        # Importing the default UFL backend must not import JAX.
        from .jax_moist_adapter import JAXMoistEulerPrimal

        self.ufl_oracle = ufl_oracle
        self.model = ufl_oracle.model
        self.logger = ufl_oracle.logger
        self.solver_parameters = ufl_oracle.solver_parameters
        self.terms = ufl_oracle.terms
        self.local_physics = local_physics
        self.moist_A_model = (
            "analytical" if local_physics is None else "neural"
        )
        self.primal_helper = JAXMoistEulerPrimal(
            self.model,
            self.solver_parameters,
            local_physics=local_physics,
        )
        self.last_primal_cache = None

    @property
    def coeff(self):
        return self.ufl_oracle.coeff

    def reset_internal_vars(self):
        """Reset retained UFL oracle storage without selecting its primal."""
        self.ufl_oracle.reset_internal_vars()

    def set_coeff(self, coeff_val):
        self.ufl_oracle.set_coeff(coeff_val)

    def set_numpy_coeff(self, coeff_val_arr):
        self.ufl_oracle.set_numpy_coeff(coeff_val_arr)

    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):
        """Apply J1 and publish output only after a successful evaluation."""
        del xnp1_sub, tn
        state = xn[0]
        cache = self.primal_helper.evaluate(state, dt)
        xnp1[0].assign(cache.state_out)
        self.last_primal_cache = cache

    def take_adjoint_step(self, *args, **kwargs):
        """Prevent accidental fallback to the legacy UFL derivative path."""
        del args, kwargs
        raise NotImplementedError(
            "the JAX moist backend requires the dual-native MTSWE cached "
            "reverse API; legacy LieSplittingIntegrator.take_adjoint_step "
            "would mix JAX primal and UFL derivatives"
        )


def build_moist_integrator(
    ufl_integrator, moist_backend, *, local_physics=None
):
    """Return the unchanged UFL child or the narrowly wrapped JAX child."""
    backend = validate_moist_backend(moist_backend)
    if backend == "ufl":
        if local_physics is not None:
            raise ValueError("learned local physics requires moist_backend='jax'")
        return ufl_integrator
    return JAXMoistEulerIntegrator(
        ufl_integrator, local_physics=local_physics
    )


def wrap_jax_moist_reverse(result):
    return JAXMoistFixedControlReverseResult(derivative=result)


def wrap_jax_moist_hvp(result):
    return JAXMoistFixedControlHVPResult(
        derivative=result,
        ordinary=wrap_jax_moist_reverse(result.ordinary),
    )


__all__ = (
    "JAXMoistEulerIntegrator",
    "JAXMoistFixedControlHVPResult",
    "JAXMoistFixedControlReverseResult",
    "MOIST_BACKENDS",
    "build_moist_integrator",
    "validate_moist_backend",
    "wrap_jax_moist_hvp",
    "wrap_jax_moist_reverse",
)
