"""Serial primal Firedrake adapter for the exact JAX moist source.

The accepted UFL moist child remains independent and unchanged.  This module
is an opt-in J1 comparison path only: it performs no tangent, reverse, HVP,
PyROL, runtime-switch, MPI, or accelerator integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import numpy as np
import ufl
from firedrake import (
    Cofunction,
    Function,
    LinearSolver,
    TestFunction,
    TestFunctions,
    TrialFunction,
    action,
    assemble,
    inner,
)

from .jax_moist import (
    JAXMoistConfigurationError,
    moist_diagnostics_jax,
    moist_diagnostics_jit,
    moist_rates_and_source_density_jax,
    moist_rates_and_source_density_jit,
)
from .physics import qsat


_STATE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")
_PACKED_STATE_FIELDS = ("h", "S", "Qv", "Qc")
_SOURCE_FIELDS = ("S", "Qv", "Qc", "Qr")
_MASK_KEYS = (
    "condensation_mask",
    "evaporation_mask",
    "uncapped_evaporation_mask",
    "rain_mask",
)
_MARGIN_KEYS = (
    "condensation_margin",
    "evaporation_margin",
    "evaporation_cap_margin",
    "rain_margin",
    "depth_denominator_margin",
)
_VANILLA_FORM_COMPILER_PARAMETERS = MappingProxyType({"mode": "vanilla"})


def _readonly_array(value, *, dtype=None):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_mapping(values, *, dtype=None):
    return MappingProxyType(
        {
            key: _readonly_array(value, dtype=dtype)
            for key, value in values.items()
        }
    )


def _copy_function(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _copy_cofunction(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _real_scalar(name, value):
    if isinstance(value, (Function, Cofunction)):
        data = np.asarray(value.dat.data_ro, dtype=np.float64).reshape(-1)
        if data.size != 1:
            raise ValueError(f"{name} must contain exactly one Real-space value")
        result = data[0]
    elif hasattr(value, "values"):
        data = np.asarray(value.values(), dtype=np.float64).reshape(-1)
        if data.size != 1:
            raise ValueError(f"{name} must contain exactly one scalar value")
        result = data[0]
    else:
        result = value
    scalar = np.asarray(result, dtype=np.float64)
    if scalar.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {scalar.shape}")
    return scalar.copy()


@dataclass(frozen=True)
class BrokenGLLLayout:
    """Read-only metadata for the exact cell-local GLL carrier."""

    carrier_space: object
    owned_cell_count: int
    points_per_cell: int
    cell_nodes: np.ndarray
    reference_points: np.ndarray
    field_order: tuple[str, ...]
    interpolation_contract: str
    source_assembly_contract: str
    form_compiler_parameters: Mapping


@dataclass(frozen=True)
class MoistActiveSetSnapshot:
    """Owned masks, signature, and switch margins on one sampling grid."""

    sampling: str
    masks: Mapping
    signature: tuple[tuple[bool, ...], ...]
    margins: Mapping


@dataclass(frozen=True)
class JAXMoistEulerPrimalCache:
    """Owned result of one standalone JAX moist Euler evaluation."""

    state_in: Function
    physics_mode: str
    applied_dt: float
    configured_dt: float
    parameters: Mapping
    neural_parameters: object | None
    packed_state: Mapping
    packed_fields: Mapping
    rates: Mapping
    source_density: Mapping
    gll_diagnostics: Mapping
    source_dual: Cofunction
    tendency: Function
    state_out: Function
    gll_active_set: MoistActiveSetSnapshot
    legacy_active_set: MoistActiveSetSnapshot


class JAXMoistEulerPrimal:
    """Standalone serial primal replica of the deployed moist Euler child."""

    def __init__(
        self, model, solver_parameters, *, use_jit=True, local_physics=None
    ):
        self.model = model
        self.spaces = model.spaces
        self.mesh = self.spaces.mesh
        self.state_space = model.dynamics.xspace
        self.state_dual_space = self.state_space.dual()
        self.dx = self.spaces.dx
        self.use_jit = bool(use_jit)
        self.local_physics = local_physics
        if local_physics is None:
            self.physics_mode = "analytical_A_original_R"
            self._combined_kernel = (
                moist_rates_and_source_density_jit
                if self.use_jit
                else moist_rates_and_source_density_jax
            )
            self._diagnostic_kernel = (
                moist_diagnostics_jit
                if self.use_jit
                else moist_diagnostics_jax
            )
        else:
            if getattr(local_physics, "physics_mode", None) not in (
                "neural_A_original_R",
                "neural_A_R",
                "neural_four_tendency",
                "neural_A_threshold_nonnegative_R",
                "neural_A_threshold_positive_gate_R",
            ):
                raise ValueError("unsupported opt-in JAX moist local physics")
            if bool(getattr(local_physics, "use_jit", self.use_jit)) != self.use_jit:
                raise ValueError("local physics and adapter JIT settings differ")
            self.physics_mode = local_physics.physics_mode
            self._combined_kernel = local_physics.combined_kernel
            self._diagnostic_kernel = local_physics.diagnostic_kernel

        if tuple(model.get_x_var_list()) != _STATE_FIELDS:
            raise ValueError(
                "JAX moist J1 requires state order (v,h,S,Qv,Qc,Qr)"
            )
        if self.mesh.comm.size != 1:
            raise NotImplementedError(
                "JAX moist J1 adapter is serial-only; MPI is not certified"
            )
        cellname = self.mesh.ufl_cell().cellname
        cellname = cellname() if callable(cellname) else cellname
        if cellname != "quadrilateral":
            raise ValueError("JAX moist J1 requires quadrilateral cells")
        if int(self.spaces.order) != 3:
            raise ValueError("JAX moist J1 requires deployed spatial order 3")

        matches = [
            term
            for term in model.dynamics.forcing_terms
            if term.name == "threewayphysics"
        ]
        if len(matches) != 1:
            raise ValueError("JAX moist J1 requires one ThreeWayPhysics term")
        self.term = matches[0]
        self._field_indices = {
            name: index for index, name in enumerate(_STATE_FIELDS)
        }

        self.carrier_space = self.spaces.CG.broken_space()
        self.layout = self._build_layout()
        self._state_test = TestFunction(self.state_space)
        self._state_tests = TestFunctions(self.state_space)
        state_trial = TrialFunction(self.state_space)
        state_mass = assemble(
            inner(self._state_test, state_trial) * self.dx,
            mat_type="aij",
        )
        stage_solver_parameters = (
            solver_parameters["erkstage-f"]
            if "erkstage-f" in solver_parameters
            else solver_parameters
        )
        self._mass_solver = LinearSolver(
            state_mass,
            solver_parameters=dict(stage_solver_parameters),
        )

        cpu_devices = [
            device for device in jax.devices() if device.platform == "cpu"
        ]
        if not cpu_devices:
            raise JAXMoistConfigurationError(
                "JAX moist J1 requires an available CPU device"
            )
        self._device = cpu_devices[0]

    def _build_layout(self):
        cell_nodes = np.asarray(
            self.carrier_space.cell_node_map().values, dtype=np.int64
        )
        owned_cell_count = int(self.mesh.cell_set.size)
        cell_nodes = np.array(
            cell_nodes[:owned_cell_count], dtype=np.int64, order="C", copy=True
        )
        if cell_nodes.ndim != 2 or cell_nodes.shape[1] != 16:
            raise ValueError(
                "broken CG3 carrier must have 16 cell-local GLL values"
            )
        flattened = cell_nodes.reshape(-1)
        if np.unique(flattened).size != flattened.size:
            raise ValueError("broken GLL carrier unexpectedly shares cell DOFs")
        if flattened.size != self.carrier_space.dim():
            raise ValueError(
                "serial broken GLL cell map does not cover its Function data"
            )

        dual_basis = self.carrier_space.finat_element.dual_basis
        # ``cell_node_map`` uses Firedrake's flattened tensor-product data
        # order (first physical coordinate fastest).  FInAT reports the
        # tensor point-set axes in the opposite table order.  Reverse the
        # coordinate columns so this metadata describes the actual packed
        # column order rather than the abstract dual-basis table order.
        reference_points = np.asarray(
            dual_basis[1].points, dtype=np.float64
        )[:, ::-1]
        if reference_points.shape != (16, 2):
            raise ValueError(
                "broken CG3 carrier must expose sixteen 2D reference points"
            )

        cell_nodes.setflags(write=False)
        reference_points = _readonly_array(reference_points, dtype=np.float64)
        return BrokenGLLLayout(
            carrier_space=self.carrier_space,
            owned_cell_count=owned_cell_count,
            points_per_cell=16,
            cell_nodes=cell_nodes,
            reference_points=reference_points,
            field_order=_PACKED_STATE_FIELDS + ("B",),
            interpolation_contract=(
                "Function(broken_CG3).interpolate(source); explicit "
                "cell_node_map packing"
            ),
            source_assembly_contract=(
                "broken-CG3 source coefficient tested in the production mixed "
                "space with production GLL dx"
            ),
            form_compiler_parameters=_VANILLA_FORM_COMPILER_PARAMETERS,
        )

    def _require_state(self, state):
        if not isinstance(state, Function):
            raise TypeError("state must be a Firedrake Function")
        if state.function_space() != self.state_space:
            raise ValueError("state belongs to the wrong mixed function space")

    def interpolate_and_pack(self, expression, name):
        """Interpolate a scalar expression to exact GLL points and copy it."""
        carrier = Function(self.carrier_space, name=name)
        carrier.interpolate(expression)
        return carrier, self.pack_carrier(carrier)

    def pack_carrier(self, carrier):
        if not isinstance(carrier, Function):
            raise TypeError("carrier must be a Firedrake Function")
        if carrier.function_space() != self.carrier_space:
            raise ValueError("carrier belongs to the wrong broken GLL space")
        data = np.array(carrier.dat.data_ro, dtype=np.float64, copy=True)
        packed = data[self.layout.cell_nodes]
        return _readonly_array(packed, dtype=np.float64)

    def unpack_carrier(self, values, name):
        array = np.asarray(values)
        expected = (
            self.layout.owned_cell_count,
            self.layout.points_per_cell,
        )
        if array.shape != expected:
            raise ValueError(
                f"carrier values must have shape {expected}, got {array.shape}"
            )
        if array.dtype != np.float64:
            raise TypeError(
                f"carrier values must have dtype float64, got {array.dtype}"
            )
        carrier = Function(self.carrier_space, name=name)
        carrier.dat.data[self.layout.cell_nodes] = np.array(
            array, dtype=np.float64, order="C", copy=True
        )
        return carrier

    def _coefficient_mapping(self, coefficient):
        if isinstance(coefficient, Mapping):
            return coefficient
        if isinstance(coefficient, Function):
            if coefficient.function_space() != self.model.dynamics.coeffspace:
                raise ValueError("coefficient belongs to the wrong function space")
            return {
                name: coefficient.sub(index)
                for index, name in enumerate(self.model.get_coeff_list())
            }
        raise TypeError(
            "coefficient mode requires a coefficient mapping or mixed Function"
        )

    def _parameters(self, coefficient):
        if self.term.treat_as_coeffs:
            values = self._coefficient_mapping(coefficient)
            gamma_r = _real_scalar("gamma_r", values["gamma_r"])
            qprecip = _real_scalar("qprecip", values["qprecip"])
            latent_ratio = _real_scalar("L", values["L"])
        else:
            gamma_r = _real_scalar("gamma_r", self.term.gamma_r)
            qprecip = _real_scalar("qprecip", self.term.qprecip)
            latent_ratio = _real_scalar("L", self.term.L)
        return {
            "g": _real_scalar("g", self.term.g),
            "q0": _real_scalar("q0", self.term.q0),
            "H0": _real_scalar("H0", self.term.H0),
            "gamma_r": gamma_r,
            "qprecip": qprecip,
            "L": latent_ratio,
            "configured_dt": _real_scalar(
                "configured_dt", self.term.dt
            ),
        }

    def _to_device_tree(self, values):
        return {
            key: jax.device_put(
                np.array(value, dtype=np.float64, order="C", copy=True),
                self._device,
            )
            for key, value in values.items()
        }

    @staticmethod
    def _from_device_tree(values):
        return {
            key: np.array(jax.device_get(value), dtype=np.float64, copy=True)
            for key, value in values.items()
        }

    @staticmethod
    def _owned_diagnostics(values):
        return MappingProxyType(
            {
                key: _readonly_array(jax.device_get(value))
                for key, value in values.items()
            }
        )

    def _gll_active_set(self, diagnostics):
        masks = {
            key: np.array(
                jax.device_get(diagnostics[key]), dtype=bool, copy=True
            )
            for key in _MASK_KEYS
        }
        readonly_masks = _readonly_mapping(masks, dtype=bool)
        signature = tuple(
            tuple(bool(value) for value in masks[key].reshape(-1))
            for key in _MASK_KEYS
        )
        margins = {
            key: np.asarray(
                jax.device_get(diagnostics[key]), dtype=np.float64
            )
            for key in _MARGIN_KEYS
        }
        return MoistActiveSetSnapshot(
            sampling="production broken-CG3/GLL points",
            masks=readonly_masks,
            signature=signature,
            margins=_readonly_mapping(margins, dtype=np.float64),
        )

    def _legacy_active_set(self, state, parameters):
        if self.physics_mode in (
            "neural_A_R", "neural_four_tendency",
            "neural_A_threshold_nonnegative_R",
            "neural_A_threshold_positive_gate_R",
        ):
            # These learned-R sources have no matching UFL analytical switch.
            # Return a neutral certification snapshot without evaluating the
            # analytical A/R switching law, which is absent from Problem B.
            size = int(np.asarray(state.sub(3).dat.data_ro).size)
            masks = {
                key: np.zeros(size, dtype=bool) for key in _MASK_KEYS
            }
            margins = {key: np.inf for key in _MARGIN_KEYS}
            return MoistActiveSetSnapshot(
                sampling="smooth neural-four-tendency DG diagnostic",
                masks=_readonly_mapping(masks, dtype=bool),
                signature=tuple(
                    tuple(False for _ in range(size)) for _ in _MASK_KEYS
                ),
                margins=_readonly_mapping(margins, dtype=np.float64),
            )
        fields = {
            name: state.sub(index)
            for index, name in enumerate(_STATE_FIELDS)
        }
        h = fields["h"]
        qv = fields["Qv"] / h
        qc = fields["Qc"] / h
        specific_entropy = fields["S"] / h
        beta2 = float(parameters["g"] * parameters["L"])
        saturation = qsat(
            h,
            specific_entropy,
            self.term.B,
            float(parameters["q0"]),
            float(parameters["H0"]),
            float(parameters["g"]),
        )
        gamma_v = 1.0 / (
            1.0 + saturation * 20.0 * beta2 / float(parameters["g"])
        )
        configured_dt = float(parameters["configured_dt"])
        condensation = gamma_v * (qv - saturation) / configured_dt
        evaporation = gamma_v * (saturation - qv) / configured_dt
        evaporation_positive = ufl.max_value(0.0, evaporation)
        cap_difference = qc / configured_dt - evaporation_positive
        rain = (
            float(parameters["gamma_r"])
            * (qc - float(parameters["qprecip"]))
            / configured_dt
        )
        expressions = {
            "condensation_mask": condensation,
            "evaporation_mask": evaporation,
            "uncapped_evaporation_mask": cap_difference,
            "rain_mask": rain,
            "depth_denominator": h + self.term.B,
        }
        water_space = fields["Qv"].function_space()
        values = {}
        for key, expression in expressions.items():
            sample = Function(water_space, name=f"jax_moist_legacy_{key}")
            sample.interpolate(expression)
            values[key] = np.array(
                sample.dat.data_ro, dtype=np.float64, copy=True
            ).reshape(-1)

        masks = {
            key: values[key] > 0.0
            for key in _MASK_KEYS
        }
        signature = tuple(
            tuple(bool(value) for value in masks[key]) for key in _MASK_KEYS
        )
        margins = {
            "condensation_margin": np.min(
                np.abs(values["condensation_mask"])
            ),
            "evaporation_margin": np.min(
                np.abs(values["evaporation_mask"])
            ),
            "evaporation_cap_margin": np.min(
                np.abs(values["uncapped_evaporation_mask"])
            ),
            "rain_margin": np.min(np.abs(values["rain_mask"])),
            "depth_denominator_margin": np.min(
                np.abs(values["depth_denominator"])
            ),
        }
        return MoistActiveSetSnapshot(
            sampling="legacy moisture-DG1 interpolation nodes",
            masks=_readonly_mapping(masks, dtype=bool),
            signature=signature,
            margins=_readonly_mapping(margins, dtype=np.float64),
        )

    def _assemble_source_dual(self, source_density):
        carriers = {
            name: self.unpack_carrier(
                source_density[name], f"jax_moist_{name}_source_carrier"
            )
            for name in _SOURCE_FIELDS
        }
        form = sum(
            inner(
                self._state_tests[self._field_indices[name]],
                carriers[name],
            )
            * self.dx
            for name in _SOURCE_FIELDS
        )
        result = assemble(
            form,
            form_compiler_parameters=dict(
                _VANILLA_FORM_COMPILER_PARAMETERS
            ),
        )
        if not isinstance(result, Cofunction):
            raise TypeError("JAX moist source assembly did not return Cofunction")
        if result.function_space() != self.state_dual_space:
            raise ValueError("JAX moist source belongs to the wrong dual space")
        result.rename("jax_moist_source_dual")
        return result

    def solve_mass(self, dual, name="jax_moist_mass_representative"):
        """Apply the same complete mixed mass inverse used by this adapter."""
        if not isinstance(dual, Cofunction):
            raise TypeError("dual must be a Firedrake Cofunction")
        if dual.function_space() != self.state_dual_space:
            raise ValueError("dual belongs to the wrong mixed dual space")
        result = Function(self.state_space, name=name)
        self._mass_solver.solve(
            result, _copy_cofunction(dual, f"{name}_rhs")
        )
        return result

    def dual_pairing(self, dual, primal):
        if not isinstance(dual, Cofunction):
            raise TypeError("dual must be a Firedrake Cofunction")
        self._require_state(primal)
        return float(assemble(action(dual, primal)))

    def evaluate(
        self, state, applied_dt, *, coefficient=None, neural_parameters=None
    ):
        """Return an owned cache for one standalone primal moist Euler step."""
        self._require_state(state)
        step = float(applied_dt)
        parameters = self._parameters(coefficient)

        packed_state = {}
        for name in _PACKED_STATE_FIELDS:
            _, packed_state[name] = self.interpolate_and_pack(
                state.sub(self._field_indices[name]),
                f"jax_moist_{name}_gll",
            )
        _, topography = self.interpolate_and_pack(
            self.term.B, "jax_moist_B_gll"
        )
        packed_fields = {"B": topography}

        state_device = self._to_device_tree(packed_state)
        fields_device = self._to_device_tree(packed_fields)
        parameter_device = self._to_device_tree(parameters)
        if neural_parameters is None:
            combined = self._combined_kernel(
                state_device, fields_device, parameter_device
            )
            diagnostics = self._diagnostic_kernel(
                state_device, fields_device, parameter_device
            )
            owned_neural_parameters = None
        else:
            if self.local_physics is None:
                raise ValueError(
                    "explicit neural parameters require opt-in learned moist physics"
                )
            from .learned_physics.parameters import (
                tree_copy,
                validate_float64_tree,
            )

            owned_neural_parameters = validate_float64_tree(
                neural_parameters, name="neural_parameters"
            )
            combined = self.local_physics.combined_parameterized_kernel(
                state_device,
                fields_device,
                parameter_device,
                owned_neural_parameters,
            )
            diagnostics = self.local_physics.diagnostic_parameterized_kernel(
                state_device,
                fields_device,
                parameter_device,
                owned_neural_parameters,
            )
            owned_neural_parameters = tree_copy(owned_neural_parameters)
        rates = self._from_device_tree(combined["rates"])
        source_density = self._from_device_tree(combined["source"])

        source_dual = self._assemble_source_dual(source_density)
        tendency = self.solve_mass(source_dual, "jax_moist_tendency")
        state_out = _copy_function(state, "jax_moist_state_out")
        with state_out.dat.vec as output_vec, tendency.dat.vec_ro as tendency_vec:
            output_vec.axpy(step, tendency_vec)

        return JAXMoistEulerPrimalCache(
            state_in=_copy_function(state, "jax_moist_state_in_cache"),
            physics_mode=self.physics_mode,
            applied_dt=step,
            configured_dt=float(parameters["configured_dt"]),
            parameters=_readonly_mapping(parameters, dtype=np.float64),
            neural_parameters=owned_neural_parameters,
            packed_state=_readonly_mapping(packed_state, dtype=np.float64),
            packed_fields=_readonly_mapping(
                packed_fields, dtype=np.float64
            ),
            rates=_readonly_mapping(rates, dtype=np.float64),
            source_density=_readonly_mapping(
                source_density, dtype=np.float64
            ),
            gll_diagnostics=self._owned_diagnostics(diagnostics),
            source_dual=_copy_cofunction(
                source_dual, "jax_moist_source_dual_cache"
            ),
            tendency=_copy_function(
                tendency, "jax_moist_tendency_cache"
            ),
            state_out=_copy_function(
                state_out, "jax_moist_state_out_cache"
            ),
            gll_active_set=self._gll_active_set(diagnostics),
            legacy_active_set=self._legacy_active_set(state, parameters),
        )


__all__ = (
    "BrokenGLLLayout",
    "JAXMoistEulerPrimal",
    "JAXMoistEulerPrimalCache",
    "MoistActiveSetSnapshot",
)
