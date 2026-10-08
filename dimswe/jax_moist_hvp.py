"""Dual-native J2 derivatives of the certified JAX moist Euler child.

This helper is independent of ``ProductionMoistEulerHVP``.  It composes the
unchanged J1 primal adapter with exact JAX local derivatives, Firedrake's
adjoint interpolation for ``P*``, an assembled weak carrier form for ``A*``,
and the same complete mixed mass solve used by J1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import numpy as np
from firedrake import (
    Cofunction,
    Function,
    TestFunction,
    action,
    assemble,
    inner,
    interpolate,
)

from .jax_moist_adapter import (
    JAXMoistEulerPrimal,
    MoistActiveSetSnapshot,
)
from .learned_physics.parameters import tree_copy, validate_float64_tree
from .jax_moist_derivatives import (
    moist_source_differentiated_vjp,
    moist_source_differentiated_vjp_jit,
    moist_source_jvp,
    moist_source_jvp_jit,
    moist_source_vjp,
    moist_source_vjp_jit,
)


_STATE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")
_PACKED_STATE_FIELDS = ("h", "S", "Qv", "Qc")
_SOURCE_FIELDS = ("S", "Qv", "Qc", "Qr")
_VANILLA_FORM_COMPILER_PARAMETERS = MappingProxyType({"mode": "vanilla"})


def _copy_function(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _copy_cofunction(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _readonly_array(value, *, dtype=None):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_mapping(values, *, dtype=np.float64):
    return MappingProxyType(
        {
            key: _readonly_array(jax.device_get(value), dtype=dtype)
            for key, value in values.items()
        }
    )


def _as_float(name, value):
    if hasattr(value, "values"):
        array = np.asarray(value.values(), dtype=np.float64).reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar value")
    return float(array[0])


def _copy_active_set(value):
    return MoistActiveSetSnapshot(
        sampling=value.sampling,
        masks=_readonly_mapping(value.masks, dtype=bool),
        signature=tuple(tuple(row) for row in value.signature),
        margins=_readonly_mapping(value.margins, dtype=np.float64),
    )


@dataclass(frozen=True)
class JAXMoistActiveSetDiagnostics:
    """Both diagnostic grids required to qualify a J2 derivative."""

    gll: MoistActiveSetSnapshot
    legacy: MoistActiveSetSnapshot


@dataclass(frozen=True)
class JAXMoistEulerPrimalCache:
    """Owned primal and local-boundary data for one JAX moist Euler step."""

    t0: float
    dt: float
    configured_dt: float
    physics_mode: str
    state_in: Function
    stage_state: Function
    packed_state: Mapping
    packed_fields: Mapping
    parameters: Mapping
    neural_parameters: object | None
    rates: Mapping
    source_density: Mapping
    gll_diagnostics: Mapping
    source_dual: Cofunction
    tendency: Function
    state_out: Function
    gll_active_set: MoistActiveSetSnapshot
    legacy_active_set: MoistActiveSetSnapshot


@dataclass(frozen=True)
class JAXMoistEulerTangentCache:
    """Owned tangent data, including the exact packed JAX direction."""

    primal: JAXMoistEulerPrimalCache
    state_direction_in: Function
    stage_state_direction: Function
    packed_state_direction: Mapping
    source_density_direction: Mapping
    source_dual_direction: Cofunction
    tendency_direction: Function
    state_direction_out: Function


@dataclass(frozen=True)
class JAXMoistEulerParameterTangentCache:
    """Exact complete-child tangent for a neural-parameter direction."""

    primal: JAXMoistEulerPrimalCache
    parameter_direction: object
    source_density_direction: Mapping
    source_dual_direction: Cofunction
    tendency_direction: Function
    state_direction_out: Function


@dataclass(frozen=True)
class JAXMoistEulerReverseResult:
    """Owned dual-native reverse result for the complete Euler child."""

    state_adjoint_in: Cofunction
    tendency_adjoint: Cofunction
    reverse_auxiliary: Function
    source_covector: Mapping
    packed_state_covector: Mapping
    stage_state_adjoint: Cofunction
    reverse_stage_order: tuple[int, ...]


@dataclass(frozen=True)
class JAXMoistEulerHVPResult:
    """Owned differentiated-reverse result with fixed moist parameters."""

    ordinary: JAXMoistEulerReverseResult
    incremental_state_adjoint_in: Cofunction
    incremental_tendency_adjoint: Cofunction
    incremental_reverse_auxiliary: Function
    incremental_source_covector: Mapping
    incremental_packed_state_covector: Mapping
    incremental_stage_state_adjoint: Cofunction
    reverse_stage_order: tuple[int, ...]


@dataclass(frozen=True)
class JAXMoistEulerParameterReverseResult:
    """Complete-child reverse contribution in the neural parameter pytree."""

    ordinary_state_reverse: JAXMoistEulerReverseResult
    parameter_adjoint: object


@dataclass(frozen=True)
class JAXMoistEulerJointHVPResult:
    """State/parameter differentiated reverse of the neural moist child."""

    ordinary_state_reverse: JAXMoistEulerReverseResult
    ordinary_parameter_adjoint: object
    incremental_state_adjoint_in: Cofunction
    incremental_parameter_adjoint: object
    incremental_source_covector: Mapping
    incremental_packed_state_covector: Mapping


class JAXMoistEulerHVP:
    """Exact JAX JVP/VJP/differentiated-VJP moist Euler helper.

    The public caches contain copied arrays and Firedrake objects only.  JAX
    pullback closures are created and consumed within one method call.
    """

    def __init__(self, timestepper, *, use_jit=True, local_physics=None):
        self.timestepper = timestepper
        self.model = timestepper.model
        self.state_space = self.model.dynamics.xspace
        self.state_dual_space = self.state_space.dual()
        self.use_jit = bool(use_jit)
        self.local_physics = local_physics
        self._validate_timestepper()
        self.primal_helper = JAXMoistEulerPrimal(
            self.model,
            timestepper.solver_parameters,
            use_jit=self.use_jit,
            local_physics=local_physics,
        )
        self.layout = self.primal_helper.layout
        self.carrier_space = self.primal_helper.carrier_space
        self._field_indices = {
            name: index for index, name in enumerate(_STATE_FIELDS)
        }
        self._state_test = TestFunction(self.state_space)
        self._carrier_test = TestFunction(self.carrier_space)
        if local_physics is None:
            self._jvp_kernel = (
                moist_source_jvp_jit if self.use_jit else moist_source_jvp
            )
            self._vjp_kernel = (
                moist_source_vjp_jit if self.use_jit else moist_source_vjp
            )
            self._differentiated_vjp_kernel = (
                moist_source_differentiated_vjp_jit
                if self.use_jit
                else moist_source_differentiated_vjp
            )
        else:
            if getattr(local_physics, "physics_mode", None) not in (
                "neural_A_original_R",
                "neural_A_R",
                "neural_four_tendency",
                "neural_A_threshold_nonnegative_R",
                "neural_A_threshold_positive_gate_R",
            ):
                raise ValueError("unsupported JAX moist derivative local physics")
            self._jvp_kernel = local_physics.state_jvp_kernel
            self._vjp_kernel = local_physics.state_vjp_kernel
            self._differentiated_vjp_kernel = (
                local_physics.state_differentiated_vjp_kernel
            )

    def _validate_timestepper(self):
        if self.timestepper.__class__.__name__ != "Euler":
            raise ValueError("JAX moist J2 requires the production Euler class")
        if self.timestepper.terms != ["threewayphysics"]:
            raise ValueError(
                "JAX moist J2 requires terms=['threewayphysics'] exactly"
            )
        if tuple(self.model.get_x_var_list()) != _STATE_FIELDS:
            raise ValueError("JAX moist J2 requires [v,h,S,Qv,Qc,Qr]")
        matches = [
            term
            for term in self.model.dynamics.forcing_terms
            if term.name == "threewayphysics"
        ]
        if len(matches) != 1 or matches[0].treat_as_coeffs:
            raise ValueError(
                "JAX moist J2 full-child derivatives require fixed parameters"
            )

    def _require_state(self, name, value):
        if not isinstance(value, Function):
            raise TypeError(f"{name} must be a Firedrake Function")
        if value.function_space() != self.state_space:
            raise ValueError(f"{name} belongs to the wrong mixed state space")

    def _require_dual(self, name, value):
        if not isinstance(value, Cofunction):
            raise TypeError(f"{name} must be a Firedrake Cofunction")
        if value.function_space() != self.state_dual_space:
            raise ValueError(f"{name} belongs to the wrong mixed dual space")

    def _state_from_container(self, name, value):
        state = value if isinstance(value, Function) else value[0]
        self._require_state(name, state)
        return state

    def _to_device_tree(self, values):
        return self.primal_helper._to_device_tree(values)

    @staticmethod
    def _from_device_tree(values):
        return {
            key: np.array(jax.device_get(value), dtype=np.float64, copy=True)
            for key, value in values.items()
        }

    def _dual_sum(self, terms, name):
        result = Cofunction(self.state_dual_space, name=name)
        result.zero()
        for scale, value in terms:
            self._require_dual("dual summand", value)
            with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
                result_vec.axpy(float(scale), value_vec)
        return result

    def _primal_axpy(self, base, scale, increment, name):
        self._require_state("base", base)
        self._require_state("increment", increment)
        result = _copy_function(base, name)
        with result.dat.vec as result_vec, increment.dat.vec_ro as increment_vec:
            result_vec.axpy(float(scale), increment_vec)
        return result

    def state_interpolation(self, value):
        """Apply the exact J1 state-to-packed-carrier operator ``P``."""
        self._require_state("value", value)
        packed = {}
        for name in _PACKED_STATE_FIELDS:
            _, packed[name] = self.primal_helper.interpolate_and_pack(
                value.sub(self._field_indices[name]),
                f"jax_moist_j2_P_{name}",
            )
        return _readonly_mapping(packed, dtype=np.float64)

    def _carrier_cofunction(self, values, name):
        array = np.asarray(values)
        expected = (
            self.layout.owned_cell_count,
            self.layout.points_per_cell,
        )
        if array.shape != expected:
            raise ValueError(
                f"packed carrier covector must have shape {expected}, "
                f"got {array.shape}"
            )
        if array.dtype != np.float64:
            raise TypeError(
                "packed carrier covector must have dtype float64, "
                f"got {array.dtype}"
            )
        result = Cofunction(self.carrier_space.dual(), name=name)
        result.zero()
        result.dat.data_wo[self.layout.cell_nodes] = np.array(
            array, dtype=np.float64, order="C", copy=True
        )
        return result

    def _pack_carrier_cofunction(self, value):
        if not isinstance(value, Cofunction):
            raise TypeError("carrier covector must be a Cofunction")
        if value.function_space() != self.carrier_space.dual():
            raise ValueError("carrier covector belongs to the wrong dual space")
        data = np.array(value.dat.data_ro, dtype=np.float64, copy=True)
        return _readonly_array(
            data[self.layout.cell_nodes], dtype=np.float64
        )

    def state_interpolation_transpose(
        self, packed_covectors, name="jax_moist_j2_P_transpose"
    ):
        """Apply ``P*`` with Firedrake's installed adjoint interpolation.

        Packed Euclidean covectors are first represented in the algebraic dual
        of the exact broken-GLL carrier.  ``firedrake.interpolate`` then applies
        the transpose of the same nodal interpolation used by J1.
        """
        result = Cofunction(self.state_dual_space, name=name)
        result.zero()
        for field in _PACKED_STATE_FIELDS:
            carrier_covector = self._carrier_cofunction(
                packed_covectors[field],
                f"{name}_{field}_carrier_covector",
            )
            source_space = self.state_space.sub(
                self._field_indices[field]
            )
            field_covector = assemble(
                interpolate(TestFunction(source_space), carrier_covector)
            )
            if not isinstance(field_covector, Cofunction):
                raise TypeError("adjoint interpolation did not return Cofunction")
            result.sub(self._field_indices[field]).assign(field_covector)
        return result

    def source_assembly(self, source_density):
        """Apply the unchanged J1 weak source operator ``A``."""
        return self.primal_helper._assemble_source_dual(source_density)

    def source_assembly_transpose(self, psi):
        """Apply ``A*`` by assembling the exact carrier-dual weak forms."""
        self._require_state("psi", psi)
        result = {}
        for field in _SOURCE_FIELDS:
            carrier_covector = assemble(
                inner(
                    self._carrier_test,
                    psi.sub(self._field_indices[field]),
                )
                * self.model.spaces.dx,
                form_compiler_parameters=dict(
                    _VANILLA_FORM_COMPILER_PARAMETERS
                ),
            )
            if not isinstance(carrier_covector, Cofunction):
                raise TypeError("weak A transpose did not return Cofunction")
            result[field] = self._pack_carrier_cofunction(carrier_covector)
        return MappingProxyType(result)

    def state_mass_map(self, value, name="jax_moist_state_mass_map"):
        self._require_state("value", value)
        result = assemble(inner(self._state_test, value) * self.model.spaces.dx)
        if not isinstance(result, Cofunction):
            raise TypeError("state mass map did not return Cofunction")
        if result.function_space() != self.state_dual_space:
            raise ValueError("state mass map belongs to the wrong dual space")
        result.rename(name)
        return result

    def state_riesz_representative(
        self, dual, name="jax_moist_state_riesz_representative"
    ):
        self._require_dual("dual", dual)
        return self.primal_helper.solve_mass(dual, name)

    def dual_pairing(self, dual, primal):
        self._require_dual("dual", dual)
        self._require_state("primal", primal)
        return float(assemble(action(dual, primal)))

    def active_set_diagnostics(self, state):
        """Return separate legacy-DG1 and actual production-GLL diagnostics."""
        self._require_state("state", state)
        parameters = self.primal_helper._parameters(None)
        packed_state = self.state_interpolation(state)
        _, packed_topography = self.primal_helper.interpolate_and_pack(
            self.primal_helper.term.B, "jax_moist_j2_active_B"
        )
        diagnostics = self.primal_helper._diagnostic_kernel(
            self._to_device_tree(packed_state),
            self._to_device_tree({"B": packed_topography}),
            self._to_device_tree(parameters),
        )
        return JAXMoistActiveSetDiagnostics(
            gll=_copy_active_set(
                self.primal_helper._gll_active_set(diagnostics)
            ),
            legacy=_copy_active_set(
                self.primal_helper._legacy_active_set(state, parameters)
            ),
        )

    def take_forward_step_cached(self, xn, tn, dt, *, neural_parameters=None):
        state = self._state_from_container("xn", xn)
        t0 = _as_float("tn", tn)
        step = _as_float("dt", dt)
        j1 = self.primal_helper.evaluate(
            state, step, neural_parameters=neural_parameters
        )
        return JAXMoistEulerPrimalCache(
            t0=t0,
            dt=step,
            configured_dt=j1.configured_dt,
            physics_mode=j1.physics_mode,
            state_in=_copy_function(j1.state_in, "jax_moist_j2_state_in"),
            stage_state=_copy_function(j1.state_in, "jax_moist_j2_stage_state"),
            packed_state=_readonly_mapping(j1.packed_state),
            packed_fields=_readonly_mapping(j1.packed_fields),
            parameters=_readonly_mapping(j1.parameters),
            neural_parameters=(
                None
                if j1.neural_parameters is None
                else tree_copy(j1.neural_parameters)
            ),
            rates=_readonly_mapping(j1.rates),
            source_density=_readonly_mapping(j1.source_density),
            gll_diagnostics=_readonly_mapping(
                j1.gll_diagnostics, dtype=None
            ),
            source_dual=_copy_cofunction(
                j1.source_dual, "jax_moist_j2_source_dual"
            ),
            tendency=_copy_function(j1.tendency, "jax_moist_j2_tendency"),
            state_out=_copy_function(j1.state_out, "jax_moist_j2_state_out"),
            gll_active_set=_copy_active_set(j1.gll_active_set),
            legacy_active_set=_copy_active_set(j1.legacy_active_set),
        )

    def take_tangent_step(self, primal, delta_xn):
        if not isinstance(primal, JAXMoistEulerPrimalCache):
            raise TypeError("primal must be a JAXMoistEulerPrimalCache")
        direction = self._state_from_container("delta_xn", delta_xn)
        packed_direction = self.state_interpolation(direction)
        if primal.neural_parameters is None:
            _, source_direction_device = self._jvp_kernel(
                self._to_device_tree(primal.packed_state),
                self._to_device_tree(packed_direction),
                self._to_device_tree(primal.packed_fields),
                self._to_device_tree(primal.parameters),
            )
        else:
            physics = self._require_neural_physics()
            source = lambda active_state: physics.combined_parameterized_kernel(
                active_state,
                self._to_device_tree(primal.packed_fields),
                self._to_device_tree(primal.parameters),
                primal.neural_parameters,
            )["source"]
            _, source_direction_device = jax.jvp(
                source,
                (self._to_device_tree(primal.packed_state),),
                (self._to_device_tree(packed_direction),),
            )
        source_direction = self._from_device_tree(source_direction_device)
        source_dual_direction = self.source_assembly(source_direction)
        tendency_direction = self.state_riesz_representative(
            source_dual_direction, "jax_moist_j2_tendency_direction"
        )
        state_direction_out = self._primal_axpy(
            direction,
            primal.dt,
            tendency_direction,
            "jax_moist_j2_state_direction_out",
        )
        return JAXMoistEulerTangentCache(
            primal=primal,
            state_direction_in=_copy_function(
                direction, "jax_moist_j2_state_direction_in"
            ),
            stage_state_direction=_copy_function(
                direction, "jax_moist_j2_stage_state_direction"
            ),
            packed_state_direction=_readonly_mapping(packed_direction),
            source_density_direction=_readonly_mapping(source_direction),
            source_dual_direction=_copy_cofunction(
                source_dual_direction,
                "jax_moist_j2_source_dual_direction",
            ),
            tendency_direction=_copy_function(
                tendency_direction, "jax_moist_j2_tendency_direction_cache"
            ),
            state_direction_out=_copy_function(
                state_direction_out, "jax_moist_j2_state_direction_out_cache"
            ),
        )

    def _require_neural_physics(self):
        if self.local_physics is None:
            raise ValueError(
                "neural parameter derivatives require opt-in learned moist physics"
            )
        return self.local_physics

    def take_parameter_tangent_step(self, primal, parameter_direction):
        """Differentiate the complete Euler output in a neural direction."""
        if not isinstance(primal, JAXMoistEulerPrimalCache):
            raise TypeError("primal must be a JAXMoistEulerPrimalCache")
        physics = self._require_neural_physics()
        direction = validate_float64_tree(
            parameter_direction, name="parameter_direction"
        )
        _, source_direction_device = physics.parameter_jvp(
            self._to_device_tree(primal.packed_state),
            direction,
            self._to_device_tree(primal.packed_fields),
            self._to_device_tree(primal.parameters),
            base_parameters=primal.neural_parameters,
        )
        source_direction = self._from_device_tree(source_direction_device)
        source_dual_direction = self.source_assembly(source_direction)
        tendency_direction = self.state_riesz_representative(
            source_dual_direction,
            "jax_moist_j2_parameter_tendency_direction",
        )
        zero = _copy_function(
            primal.state_in, "jax_moist_j2_zero_parameter_direction"
        )
        zero.assign(0.0)
        state_direction_out = self._primal_axpy(
            zero,
            primal.dt,
            tendency_direction,
            "jax_moist_j2_parameter_state_direction_out",
        )
        return JAXMoistEulerParameterTangentCache(
            primal=primal,
            parameter_direction=tree_copy(direction),
            source_density_direction=_readonly_mapping(source_direction),
            source_dual_direction=_copy_cofunction(
                source_dual_direction,
                "jax_moist_j2_parameter_source_dual_direction",
            ),
            tendency_direction=_copy_function(
                tendency_direction,
                "jax_moist_j2_parameter_tendency_direction_cache",
            ),
            state_direction_out=_copy_function(
                state_direction_out,
                "jax_moist_j2_parameter_state_direction_out_cache",
            ),
        )

    def take_adjoint_step_cached(self, primal, lambda_plus_star):
        if not isinstance(primal, JAXMoistEulerPrimalCache):
            raise TypeError("primal must be a JAXMoistEulerPrimalCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        tendency_adjoint = self._dual_sum(
            [(primal.dt, lambda_plus_star)],
            "jax_moist_j2_tendency_adjoint",
        )
        reverse_auxiliary = self.state_riesz_representative(
            tendency_adjoint, "jax_moist_j2_reverse_auxiliary"
        )
        source_covector = self.source_assembly_transpose(reverse_auxiliary)
        if primal.neural_parameters is None:
            state_covector_device = self._vjp_kernel(
                self._to_device_tree(primal.packed_state),
                self._to_device_tree(source_covector),
                self._to_device_tree(primal.packed_fields),
                self._to_device_tree(primal.parameters),
            )
        else:
            physics = self._require_neural_physics()
            active_state = self._to_device_tree(primal.packed_state)
            source = lambda state_value: physics.combined_parameterized_kernel(
                state_value,
                self._to_device_tree(primal.packed_fields),
                self._to_device_tree(primal.parameters),
                primal.neural_parameters,
            )["source"]
            _, pullback = jax.vjp(source, active_state)
            state_covector_device = pullback(
                self._to_device_tree(source_covector)
            )[0]
        state_covector = self._from_device_tree(state_covector_device)
        stage_state_adjoint = self.state_interpolation_transpose(
            state_covector, "jax_moist_j2_stage_state_adjoint"
        )
        state_adjoint_in = self._dual_sum(
            [(1.0, lambda_plus_star), (1.0, stage_state_adjoint)],
            "jax_moist_j2_state_adjoint_in",
        )
        return JAXMoistEulerReverseResult(
            state_adjoint_in=_copy_cofunction(
                state_adjoint_in, "jax_moist_j2_state_adjoint_in_result"
            ),
            tendency_adjoint=_copy_cofunction(
                tendency_adjoint, "jax_moist_j2_tendency_adjoint_result"
            ),
            reverse_auxiliary=_copy_function(
                reverse_auxiliary, "jax_moist_j2_reverse_auxiliary_result"
            ),
            source_covector=_readonly_mapping(source_covector),
            packed_state_covector=_readonly_mapping(state_covector),
            stage_state_adjoint=_copy_cofunction(
                stage_state_adjoint,
                "jax_moist_j2_stage_state_adjoint_result",
            ),
            reverse_stage_order=(0,),
        )

    def take_parameter_adjoint_step(self, primal, lambda_plus_star):
        """Apply the complete-child transpose to all neural parameters."""
        physics = self._require_neural_physics()
        ordinary = self.take_adjoint_step_cached(primal, lambda_plus_star)
        parameter_adjoint_device = physics.parameter_vjp(
            self._to_device_tree(primal.packed_state),
            self._to_device_tree(ordinary.source_covector),
            self._to_device_tree(primal.packed_fields),
            self._to_device_tree(primal.parameters),
            base_parameters=primal.neural_parameters,
        )
        return JAXMoistEulerParameterReverseResult(
            ordinary_state_reverse=ordinary,
            parameter_adjoint=tree_copy(parameter_adjoint_device),
        )

    def take_incremental_adjoint_step(
        self, tangent, lambda_plus_star, mu_plus_star
    ):
        if not isinstance(tangent, JAXMoistEulerTangentCache):
            raise TypeError("tangent must be a JAXMoistEulerTangentCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        self._require_dual("mu_plus_star", mu_plus_star)
        primal = tangent.primal
        ordinary = self.take_adjoint_step_cached(primal, lambda_plus_star)
        incremental_tendency_adjoint = self._dual_sum(
            [(primal.dt, mu_plus_star)],
            "jax_moist_j2_incremental_tendency_adjoint",
        )
        incremental_reverse_auxiliary = self.state_riesz_representative(
            incremental_tendency_adjoint,
            "jax_moist_j2_incremental_reverse_auxiliary",
        )
        incremental_source_covector = self.source_assembly_transpose(
            incremental_reverse_auxiliary
        )
        _, incremental_state_covector_device = (
            self._differentiated_vjp_kernel(
                self._to_device_tree(primal.packed_state),
                self._to_device_tree(ordinary.source_covector),
                self._to_device_tree(tangent.packed_state_direction),
                self._to_device_tree(incremental_source_covector),
                self._to_device_tree(primal.packed_fields),
                self._to_device_tree(primal.parameters),
            )
        )
        incremental_state_covector = self._from_device_tree(
            incremental_state_covector_device
        )
        incremental_stage_state_adjoint = (
            self.state_interpolation_transpose(
                incremental_state_covector,
                "jax_moist_j2_incremental_stage_state_adjoint",
            )
        )
        incremental_state_adjoint_in = self._dual_sum(
            [
                (1.0, mu_plus_star),
                (1.0, incremental_stage_state_adjoint),
            ],
            "jax_moist_j2_incremental_state_adjoint_in",
        )
        return JAXMoistEulerHVPResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=_copy_cofunction(
                incremental_state_adjoint_in,
                "jax_moist_j2_incremental_state_adjoint_in_result",
            ),
            incremental_tendency_adjoint=_copy_cofunction(
                incremental_tendency_adjoint,
                "jax_moist_j2_incremental_tendency_adjoint_result",
            ),
            incremental_reverse_auxiliary=_copy_function(
                incremental_reverse_auxiliary,
                "jax_moist_j2_incremental_reverse_auxiliary_result",
            ),
            incremental_source_covector=_readonly_mapping(
                incremental_source_covector
            ),
            incremental_packed_state_covector=_readonly_mapping(
                incremental_state_covector
            ),
            incremental_stage_state_adjoint=_copy_cofunction(
                incremental_stage_state_adjoint,
                "jax_moist_j2_incremental_stage_state_adjoint_result",
            ),
            reverse_stage_order=(0,),
        )

    def take_joint_incremental_adjoint_step(
        self,
        primal,
        state_direction,
        parameter_direction,
        lambda_plus_star,
        mu_plus_star,
    ):
        """Differentiate state and neural-parameter VJPs without dense maps."""
        if not isinstance(primal, JAXMoistEulerPrimalCache):
            raise TypeError("primal must be a JAXMoistEulerPrimalCache")
        physics = self._require_neural_physics()
        direction = self._state_from_container("state_direction", state_direction)
        self._require_dual("lambda_plus_star", lambda_plus_star)
        self._require_dual("mu_plus_star", mu_plus_star)
        parameter_direction = validate_float64_tree(
            parameter_direction, name="parameter_direction"
        )
        packed_direction = self.state_interpolation(direction)
        ordinary = self.take_adjoint_step_cached(primal, lambda_plus_star)
        ordinary_parameter_adjoint = physics.parameter_vjp(
            self._to_device_tree(primal.packed_state),
            self._to_device_tree(ordinary.source_covector),
            self._to_device_tree(primal.packed_fields),
            self._to_device_tree(primal.parameters),
            base_parameters=primal.neural_parameters,
        )
        incremental_tendency_adjoint = self._dual_sum(
            [(primal.dt, mu_plus_star)],
            "jax_moist_j2_joint_incremental_tendency_adjoint",
        )
        incremental_reverse_auxiliary = self.state_riesz_representative(
            incremental_tendency_adjoint,
            "jax_moist_j2_joint_incremental_reverse_auxiliary",
        )
        incremental_source_covector = self.source_assembly_transpose(
            incremental_reverse_auxiliary
        )
        _, differentiated = physics.joint_differentiated_vjp(
            self._to_device_tree(primal.packed_state),
            self._to_device_tree(ordinary.source_covector),
            self._to_device_tree(packed_direction),
            parameter_direction,
            self._to_device_tree(incremental_source_covector),
            self._to_device_tree(primal.packed_fields),
            self._to_device_tree(primal.parameters),
            base_parameters=primal.neural_parameters,
        )
        incremental_state_covector_device, incremental_parameter_adjoint = differentiated
        incremental_state_covector = self._from_device_tree(
            incremental_state_covector_device
        )
        incremental_stage_state_adjoint = self.state_interpolation_transpose(
            incremental_state_covector,
            "jax_moist_j2_joint_incremental_stage_state_adjoint",
        )
        incremental_state_adjoint_in = self._dual_sum(
            [(1.0, mu_plus_star), (1.0, incremental_stage_state_adjoint)],
            "jax_moist_j2_joint_incremental_state_adjoint_in",
        )
        return JAXMoistEulerJointHVPResult(
            ordinary_state_reverse=ordinary,
            ordinary_parameter_adjoint=tree_copy(ordinary_parameter_adjoint),
            incremental_state_adjoint_in=_copy_cofunction(
                incremental_state_adjoint_in,
                "jax_moist_j2_joint_incremental_state_adjoint_in_result",
            ),
            incremental_parameter_adjoint=tree_copy(
                incremental_parameter_adjoint
            ),
            incremental_source_covector=_readonly_mapping(
                incremental_source_covector
            ),
            incremental_packed_state_covector=_readonly_mapping(
                incremental_state_covector
            ),
        )


__all__ = (
    "JAXMoistActiveSetDiagnostics",
    "JAXMoistEulerHVP",
    "JAXMoistEulerHVPResult",
    "JAXMoistEulerJointHVPResult",
    "JAXMoistEulerParameterReverseResult",
    "JAXMoistEulerParameterTangentCache",
    "JAXMoistEulerPrimalCache",
    "JAXMoistEulerReverseResult",
    "JAXMoistEulerTangentCache",
)
