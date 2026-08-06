"""Exact dual-native HVPs for the deployed six-field MTSWE split step.

The active RK derivative paths consume the exact per-stage forms retained by
``GeneralRK``.  The complete parent step is the production [2, 1, 2, 1]
subcycled Lie composition.  This module deliberately exposes physical ``c0``
and the full initial state only; moist parameters and ``s`` remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import ufl
from firedrake import (
    Cofunction,
    Function,
    LinearSolver,
    TestFunction,
    TrialFunction,
    action,
    assemble,
    derivative,
    inner,
)

from .dry_lie_hvp import ProductionDryRK4HVP
from .hyperviscosity_hvp import (
    HyperviscosityHVPResult,
    HyperviscosityPrimalCache,
    HyperviscosityReverseResult,
    HyperviscosityTangentCache,
    _as_float,
    _copy_cofunction,
    _copy_function,
    _form_items,
    _form_metadata,
    _normalize_derivative_form,
)
from .physics import qsat


_MTSWE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")
_RK4_A = np.array(
    (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
)
_RK4_B = np.array((1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0))
_RK4_C = np.array((0.0, 0.5, 0.5, 1.0))
_SSPRK43_A = np.array(
    (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0),
    )
)
_SSPRK43_B = np.array((1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5))
_SSPRK43_C = np.array((0.0, 0.5, 1.0, 0.5))
_REVERSE_RK4_ORDER = (3, 2, 1, 0)
_FORWARD_CHILD_ORDER = (
    "dry_rk4_0",
    "dry_rk4_1",
    "hyperviscosity_euler",
    "dg_ssprk43_0",
    "dg_ssprk43_1",
    "moist_euler",
)
_REVERSE_CHILD_ORDER = tuple(reversed(_FORWARD_CHILD_ORDER))


class ProductionMTSWEDryRK4HVP(ProductionDryRK4HVP):
    """The accepted exact-form dry RK4 graph on the six-field state space."""

    def _validate_timestepper(self):
        if self.timestepper.__class__.__name__ != "RK4":
            raise ValueError("MTSWE dry HVP requires the production RK4 class")
        if self.timestepper.terms != ["model"]:
            raise ValueError("MTSWE dry HVP requires terms=['model'] exactly")
        if self.timestepper.nstages != 4 or not self.timestepper.is_explicit:
            raise ValueError("MTSWE dry HVP requires four explicit RK stages")
        if not (
            np.array_equal(self.timestepper.A, _RK4_A)
            and np.array_equal(self.timestepper.b, _RK4_B)
            and np.array_equal(self.timestepper.c, _RK4_C)
        ):
            raise ValueError("MTSWE dry HVP is certified only for classical RK4")
        if tuple(self.model.get_x_var_list()) != _MTSWE_FIELDS:
            raise ValueError("MTSWE dry HVP requires [v,h,S,Qv,Qc,Qr]")
        if self.model.get_aux_var_list(terms=self.timestepper.terms):
            raise ValueError("MTSWE dry child must not expose stage diagnostics")


class ProductionDGSSPRK43HVP(ProductionDryRK4HVP):
    """Exact cached derivative graph for production DG SSPRK43 transport."""

    def _validate_timestepper(self):
        if self.timestepper.__class__.__name__ != "SSPRK43":
            raise ValueError("DG HVP requires the production SSPRK43 class")
        if self.timestepper.terms != ["dg1limiter"]:
            raise ValueError("DG HVP requires terms=['dg1limiter'] exactly")
        if self.timestepper.nstages != 4 or not self.timestepper.is_explicit:
            raise ValueError("DG HVP requires four explicit stages")
        if not (
            np.array_equal(self.timestepper.A, _SSPRK43_A)
            and np.array_equal(self.timestepper.b, _SSPRK43_B)
            and np.array_equal(self.timestepper.c, _SSPRK43_C)
        ):
            raise ValueError("deployed DG tableau differs from SSPRK43 audit")
        if tuple(self.model.get_x_var_list()) != _MTSWE_FIELDS:
            raise ValueError("DG HVP requires [v,h,S,Qv,Qc,Qr]")
        if self.model.get_aux_var_list(terms=self.timestepper.terms):
            raise ValueError("DG transport child must not expose diagnostics")
        matching = [
            term
            for term in self.model.dynamics.forcing_terms
            if term.name == "dg1limiter"
        ]
        if len(matching) != 1:
            raise ValueError("DG HVP requires one production dg1limiter term")


@dataclass(frozen=True)
class MoistActiveSetDiagnostic:
    """Owned active-branch signature and distances to every moist switch."""

    signature: tuple[tuple[bool, ...], ...]
    condensation_margin: float
    evaporation_margin: float
    evaporation_cap_margin: float
    rain_margin: float
    depth_denominator_margin: float


@dataclass(frozen=True)
class MoistEulerPrimalCache:
    """Owned primal data for one exact production moist Euler child."""

    t0: float
    dt: float
    state_in: Function
    stage_state: Function
    tendency: Function
    state_out: Function
    active_set: MoistActiveSetDiagnostic


@dataclass(frozen=True)
class MoistEulerTangentCache:
    """Owned tangent data for one moist Euler cache."""

    primal: MoistEulerPrimalCache
    state_direction_in: Function
    stage_state_direction: Function
    tendency_direction: Function
    state_direction_out: Function


@dataclass(frozen=True)
class MoistEulerReverseResult:
    """Owned dual-native ordinary reverse for moist Euler."""

    state_adjoint_in: Cofunction
    c0_gradient: float
    tendency_adjoint: Cofunction
    reverse_auxiliary: Function
    stage_state_adjoint: Cofunction
    reverse_stage_order: tuple[int, ...]


@dataclass(frozen=True)
class MoistEulerHVPResult:
    """Owned exact incremental reverse for moist Euler."""

    ordinary: MoistEulerReverseResult
    incremental_state_adjoint_in: Cofunction
    c0_hvp: float
    incremental_tendency_adjoint: Cofunction
    incremental_reverse_auxiliary: Function
    incremental_stage_state_adjoint: Cofunction
    reverse_stage_order: tuple[int, ...]


class ProductionMoistEulerHVP:
    """Differentiate the identical production-owned moist Euler stage form."""

    def __init__(self, timestepper):
        self.timestepper = timestepper
        self.model = timestepper.model
        self._validate_timestepper()
        self.state_space = self.model.dynamics.xspace
        self.state_dual_space = self.state_space.dual()
        self._production_state = timestepper.production_stage_base_state
        self._production_rhs = timestepper.production_stage_rhs_forms[0]
        if self._production_state is not timestepper.xk[0]:
            raise ValueError("moist production base-state identity changed")
        if self._production_rhs is not timestepper.production_stage_rhs_forms[0]:
            raise ValueError("moist production stage-form identity changed")

        state_trial = TrialFunction(self.state_space)
        self._state_test = TestFunction(self.state_space)
        state_mass = assemble(
            inner(self._state_test, state_trial) * self.model.spaces.dx,
            mat_type="aij",
        )
        self._state_mass_solver = LinearSolver(
            state_mass,
            solver_parameters=dict(timestepper.solver_parameters["erkstage-f"]),
        )
        self.term = next(
            term
            for term in self.model.dynamics.forcing_terms
            if term.name == "threewayphysics"
        )
        self._field_indices = {
            name: i for i, name in enumerate(self.model.get_x_var_list())
        }

    def _validate_timestepper(self):
        if self.timestepper.__class__.__name__ != "Euler":
            raise ValueError("moist HVP requires the production Euler class")
        if self.timestepper.terms != ["threewayphysics"]:
            raise ValueError(
                "moist HVP requires terms=['threewayphysics'] exactly"
            )
        if not (
            self.timestepper.nstages == 1
            and self.timestepper.is_explicit
            and self.timestepper.A.shape == (1, 1)
            and self.timestepper.A[0, 0] == 0.0
            and self.timestepper.b[0] == 1.0
            and self.timestepper.c[0] == 0.0
        ):
            raise ValueError("moist HVP is certified only for explicit Euler")
        if tuple(self.model.get_x_var_list()) != _MTSWE_FIELDS:
            raise ValueError("moist HVP requires [v,h,S,Qv,Qc,Qr]")
        if self.model.get_aux_var_list(terms=self.timestepper.terms):
            raise ValueError("moist Euler must not expose stage diagnostics")
        matching = [
            term
            for term in self.model.dynamics.forcing_terms
            if term.name == "threewayphysics"
        ]
        if len(matching) != 1 or matching[0].treat_as_coeffs:
            raise ValueError(
                "moist HVP requires one fixed-parameter production physics term"
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

    @staticmethod
    def _zero_dual(space, name):
        result = Cofunction(space, name=name)
        result.zero()
        return result

    def _assemble_dual(self, form, name):
        normalized = _normalize_derivative_form(form)
        if normalized.structural_zero:
            return self._zero_dual(self.state_dual_space, name)
        if len(_form_items(normalized.normalized, "arguments")) != 1:
            raise TypeError(
                "nonzero moist derivative is not a one-form: "
                f"{_form_metadata(normalized.normalized)}"
            )
        result = assemble(normalized.normalized)
        if not isinstance(result, Cofunction):
            raise TypeError("moist one-form assembly did not return Cofunction")
        if result.function_space() != self.state_dual_space:
            raise ValueError("moist one-form belongs to the wrong dual space")
        result.rename(name)
        return result

    def _solve_mass(self, dual, name):
        self._require_dual("dual", dual)
        result = Function(self.state_space, name=name)
        self._state_mass_solver.solve(
            result, _copy_cofunction(dual, f"{name}_rhs")
        )
        return result

    def _dual_sum(self, terms, name):
        result = Cofunction(self.state_dual_space, name=name)
        result.zero()
        for scale, value in terms:
            self._require_dual("dual summand", value)
            with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
                result_vec.axpy(float(scale), value_vec)
        return result

    @staticmethod
    def _primal_axpy(base, scale, increment, name):
        result = _copy_function(base, name)
        with result.dat.vec as result_vec, increment.dat.vec_ro as increment_vec:
            result_vec.axpy(float(scale), increment_vec)
        return result

    def _populate_graph(self, primal):
        self.timestepper.t.assign(primal.t0)
        self.timestepper.dt.assign(primal.dt)
        self._production_state.assign(primal.state_in)
        self.timestepper.Fi[0][0][0].assign(primal.tendency)

    def state_mass_map(self, value, name="moist_state_mass_map"):
        self._require_state("value", value)
        return self._assemble_dual(
            inner(self._state_test, value) * self.model.spaces.dx, name
        )

    def state_riesz_representative(
        self, dual, name="moist_state_riesz_representative"
    ):
        return self._solve_mass(dual, name)

    def dual_pairing(self, dual, primal):
        self._require_dual("dual", dual)
        self._require_state("primal", primal)
        return float(assemble(action(dual, primal)))

    def _sample(self, expression, space, name):
        sampled = Function(space, name=name)
        sampled.interpolate(expression)
        return np.array(sampled.dat.data_ro, dtype=float, copy=True).reshape(-1)

    def active_set_diagnostics(self, state):
        """Evaluate all deployed moist switch selections on DG moisture nodes."""
        self._require_state("state", state)
        sub = {
            name: state.sub(index)
            for name, index in self._field_indices.items()
        }
        h = sub["h"]
        S = sub["S"]
        qv = sub["Qv"] / h
        qc = sub["Qc"] / h
        s = S / h
        beta2 = self.term.g * self.term.L
        q_sat = qsat(
            h,
            s,
            self.term.B,
            self.term.q0,
            self.term.H0,
            self.term.g,
        )
        gamma_v = 1.0 / (1.0 + q_sat * 20.0 * beta2 / self.term.g)
        condensation = gamma_v * (qv - q_sat) / self.term.tau_v
        evaporation = gamma_v * (q_sat - qv) / self.term.tau_v
        evaporation_positive = ufl.max_value(0.0, evaporation)
        evaporation_cap = qc / self.term.dt
        rain = self.term.gamma_r * (qc - self.term.qprecip) / self.term.tau_r
        space = sub["Qv"].function_space()
        values = {
            "condensation": self._sample(
                condensation, space, "moist_condensation_switch"
            ),
            "evaporation": self._sample(
                evaporation, space, "moist_evaporation_switch"
            ),
            "cap_difference": self._sample(
                evaporation_cap - evaporation_positive,
                space,
                "moist_evaporation_cap_switch",
            ),
            "rain": self._sample(rain, space, "moist_rain_switch"),
            "depth_denominator": self._sample(
                h + self.term.B, space, "moist_depth_denominator"
            ),
        }
        if any(array.size == 0 for array in values.values()):
            raise ValueError("moist active-set sampling produced no values")
        signature = tuple(
            tuple(bool(value) for value in (values[name] > 0.0))
            for name in (
                "condensation",
                "evaporation",
                "cap_difference",
                "rain",
            )
        )
        return MoistActiveSetDiagnostic(
            signature=signature,
            condensation_margin=float(np.min(np.abs(values["condensation"]))),
            evaporation_margin=float(np.min(np.abs(values["evaporation"]))),
            evaporation_cap_margin=float(
                np.min(np.abs(values["cap_difference"]))
            ),
            rain_margin=float(np.min(np.abs(values["rain"]))),
            depth_denominator_margin=float(
                np.min(np.abs(values["depth_denominator"]))
            ),
        )

    def form_identity_diagnostics(self):
        known = [
            ("production_base_state_xk", self._production_state),
            ("production_time", self.timestepper.t),
            ("production_dt", self.timestepper.dt),
            ("production_coefficient", self.timestepper.coeff),
            ("production_stage_tendency_F0", self.timestepper.Fi[0][0][0]),
        ]
        coefficients = _form_items(self._production_rhs, "coefficients")
        return {
            "production_form_python_id": id(self._production_rhs),
            "registered_form_python_id": id(
                self.timestepper.production_stage_rhs_forms[0]
            ),
            "production_residual_python_id": id(
                self.timestepper.production_stage_residuals[0]
            ),
            "form_is_registered_generalrk_form": (
                self._production_rhs
                is self.timestepper.production_stage_rhs_forms[0]
            ),
            "derivative_variable_python_id": id(self._production_state),
            "derivative_variable_is_live": any(
                coefficient is self._production_state
                for coefficient in coefficients
            ),
            "coefficient_identities": tuple(
                {
                    "python_id": id(coefficient),
                    "name": (
                        coefficient.name()
                        if callable(getattr(coefficient, "name", None))
                        else "<unnamed>"
                    ),
                    "identity_labels": tuple(
                        label
                        for label, candidate in known
                        if coefficient is candidate
                    ),
                }
                for coefficient in coefficients
            ),
            "direct_c0_dependency": any(
                coefficient is self.timestepper.coeff
                for coefficient in coefficients
            ),
            "reverse_stage_order": (0,),
        }

    def take_forward_step_cached(self, xn, tn, dt):
        state_in = self._state_from_container("xn", xn)
        t0 = _as_float("tn", tn)
        step = _as_float("dt", dt)
        state_out, state_out_sub, _ = self.model.get_full_var(
            "moist_cached_state_out", split_x_and_aux=True
        )
        legacy_input = xn if isinstance(xn, (list, tuple)) else [state_in]
        self.timestepper.reset_internal_vars()
        self.timestepper.take_forward_step(
            state_out, state_out_sub, legacy_input, t0, step
        )
        return MoistEulerPrimalCache(
            t0=t0,
            dt=step,
            state_in=_copy_function(state_in, "moist_state_in_cache"),
            stage_state=_copy_function(state_in, "moist_stage_state_cache"),
            tendency=_copy_function(
                self.timestepper.Fi[0][0][0], "moist_tendency_cache"
            ),
            state_out=_copy_function(state_out[0], "moist_state_out_cache"),
            active_set=self.active_set_diagnostics(state_in),
        )

    def take_tangent_step(self, primal, delta_xn):
        if not isinstance(primal, MoistEulerPrimalCache):
            raise TypeError("primal must be a MoistEulerPrimalCache")
        direction = self._state_from_container("delta_xn", delta_xn)
        self._populate_graph(primal)
        rhs = self._assemble_dual(
            derivative(self._production_rhs, self._production_state, direction),
            "moist_tangent_rhs",
        )
        tendency_direction = self._solve_mass(rhs, "moist_tendency_direction")
        state_direction_out = self._primal_axpy(
            direction,
            primal.dt,
            tendency_direction,
            "moist_state_direction_out",
        )
        return MoistEulerTangentCache(
            primal=primal,
            state_direction_in=_copy_function(
                direction, "moist_state_direction_in_cache"
            ),
            stage_state_direction=_copy_function(
                direction, "moist_stage_state_direction_cache"
            ),
            tendency_direction=_copy_function(
                tendency_direction, "moist_tendency_direction_cache"
            ),
            state_direction_out=_copy_function(
                state_direction_out, "moist_state_direction_out_cache"
            ),
        )

    def take_adjoint_step_cached(self, primal, lambda_plus_star):
        if not isinstance(primal, MoistEulerPrimalCache):
            raise TypeError("primal must be a MoistEulerPrimalCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        tendency_adjoint = self._dual_sum(
            [(primal.dt, lambda_plus_star)], "moist_tendency_adjoint"
        )
        reverse_auxiliary = self._solve_mass(
            tendency_adjoint, "moist_reverse_auxiliary"
        )
        self._populate_graph(primal)
        stage_state_adjoint = self._assemble_dual(
            derivative(
                action(self._production_rhs, reverse_auxiliary),
                self._production_state,
            ),
            "moist_stage_state_adjoint",
        )
        state_adjoint_in = self._dual_sum(
            [(1.0, lambda_plus_star), (1.0, stage_state_adjoint)],
            "moist_state_adjoint_in",
        )
        return MoistEulerReverseResult(
            state_adjoint_in=state_adjoint_in,
            c0_gradient=0.0,
            tendency_adjoint=_copy_cofunction(
                tendency_adjoint, "moist_tendency_adjoint_result"
            ),
            reverse_auxiliary=_copy_function(
                reverse_auxiliary, "moist_reverse_auxiliary_result"
            ),
            stage_state_adjoint=_copy_cofunction(
                stage_state_adjoint, "moist_stage_state_adjoint_result"
            ),
            reverse_stage_order=(0,),
        )

    def take_incremental_adjoint_step(
        self, tangent, lambda_plus_star, mu_plus_star
    ):
        if not isinstance(tangent, MoistEulerTangentCache):
            raise TypeError("tangent must be a MoistEulerTangentCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        self._require_dual("mu_plus_star", mu_plus_star)
        primal = tangent.primal
        ordinary = self.take_adjoint_step_cached(primal, lambda_plus_star)
        incremental_tendency_adjoint = self._dual_sum(
            [(primal.dt, mu_plus_star)],
            "moist_incremental_tendency_adjoint",
        )
        incremental_reverse_auxiliary = self._solve_mass(
            incremental_tendency_adjoint,
            "moist_incremental_reverse_auxiliary",
        )
        self._populate_graph(primal)
        ordinary_contracted = action(
            self._production_rhs, ordinary.reverse_auxiliary
        )
        incremental_contracted = action(
            self._production_rhs, incremental_reverse_auxiliary
        )
        ordinary_pullback = derivative(
            ordinary_contracted, self._production_state
        )
        incremental_stage_state_adjoint = self._assemble_dual(
            derivative(incremental_contracted, self._production_state)
            + derivative(
                ordinary_pullback,
                self._production_state,
                tangent.state_direction_in,
            ),
            "moist_incremental_stage_state_adjoint",
        )
        incremental_state_adjoint_in = self._dual_sum(
            [
                (1.0, mu_plus_star),
                (1.0, incremental_stage_state_adjoint),
            ],
            "moist_incremental_state_adjoint_in",
        )
        return MoistEulerHVPResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=incremental_state_adjoint_in,
            c0_hvp=0.0,
            incremental_tendency_adjoint=_copy_cofunction(
                incremental_tendency_adjoint,
                "moist_incremental_tendency_adjoint_result",
            ),
            incremental_reverse_auxiliary=_copy_function(
                incremental_reverse_auxiliary,
                "moist_incremental_reverse_auxiliary_result",
            ),
            incremental_stage_state_adjoint=_copy_cofunction(
                incremental_stage_state_adjoint,
                "moist_incremental_stage_state_adjoint_result",
            ),
            reverse_stage_order=(0,),
        )


@dataclass(frozen=True)
class MTSWEChildPrimalCache:
    name: str
    integrator_index: int
    subcycle_index: int
    t0: float
    dt: float
    cache: object


@dataclass(frozen=True)
class MTSWEChildTangentCache:
    primal: MTSWEChildPrimalCache
    cache: object


@dataclass(frozen=True)
class MTSWESplitPrimalCache:
    t0: float
    dt: float
    state_in: Function
    boundary_states: tuple[Function, ...]
    children: tuple[MTSWEChildPrimalCache, ...]
    state_out: Function
    forward_child_order: tuple[str, ...]


@dataclass(frozen=True)
class MTSWESplitTangentCache:
    primal: MTSWESplitPrimalCache
    delta_c0: float
    state_direction_in: Function
    boundary_state_directions: tuple[Function, ...]
    children: tuple[MTSWEChildTangentCache, ...]
    state_direction_out: Function
    forward_child_order: tuple[str, ...]


@dataclass(frozen=True)
class MTSWEChildReverseData:
    name: str
    result: object


@dataclass(frozen=True)
class MTSWESplitReverseResult:
    state_adjoint_in: Cofunction
    physical_c0_gradient: float
    children: tuple[MTSWEChildReverseData, ...]
    reverse_child_order: tuple[str, ...]


@dataclass(frozen=True)
class MTSWESplitHVPResult:
    ordinary: MTSWESplitReverseResult
    incremental_state_adjoint_in: Cofunction
    physical_c0_hvp: float
    children: tuple[MTSWEChildReverseData, ...]
    reverse_child_order: tuple[str, ...]


@dataclass(frozen=True)
class MTSWEReducedGradientResult:
    objective_value: float
    initial_condition_gradient: Cofunction
    physical_c0_gradient: float
    terminal_adjoint: Cofunction
    states: tuple[Function, ...]
    primal_caches: tuple[MTSWESplitPrimalCache, ...]
    reverse_results: tuple[MTSWESplitReverseResult, ...]


@dataclass(frozen=True)
class MTSWEReducedHVPResult:
    objective_value: float
    initial_condition_gradient: Cofunction
    physical_c0_gradient: float
    initial_condition_hvp: Cofunction
    physical_c0_hvp: float
    terminal_adjoint: Cofunction
    terminal_incremental_adjoint: Cofunction
    states: tuple[Function, ...]
    state_directions: tuple[Function, ...]
    primal_caches: tuple[MTSWESplitPrimalCache, ...]
    tangent_caches: tuple[MTSWESplitTangentCache, ...]
    reverse_results: tuple[MTSWESplitHVPResult, ...]


class ProductionMTSWESplitHVP:
    """Exact cached derivatives of one or more complete production MTSWE steps."""

    def __init__(self, timestepper):
        self.timestepper = timestepper
        self._validate_timestepper()
        self.dry_child, self.hyper_child, self.dg_child, self.moist_child = (
            timestepper.time_integrators
        )
        self.model = self.dry_child.model
        self.dry_helper = ProductionMTSWEDryRK4HVP(self.dry_child)
        self.hyper_helper = self.hyper_child._get_hyperviscosity_hvp_helper()
        self.dg_helper = ProductionDGSSPRK43HVP(self.dg_child)
        self.moist_helper = ProductionMoistEulerHVP(self.moist_child)
        self.state_space = self.dry_helper.state_space
        self.state_dual_space = self.dry_helper.state_dual_space

    def _validate_timestepper(self):
        if getattr(self.timestepper, "timestepper_list", None) != [
            "RK4",
            "Euler",
            "SSPRK43",
            "Euler",
        ]:
            raise ValueError("MTSWE HVP requires [RK4,Euler,SSPRK43,Euler]")
        if getattr(self.timestepper, "termlist", None) != [
            ["model"],
            ["hyperviscosity"],
            ["dg1limiter"],
            ["threewayphysics"],
        ]:
            raise ValueError("MTSWE HVP requires the deployed four-way term split")
        if list(getattr(self.timestepper, "subcycle_list", ())) != [2, 1, 2, 1]:
            raise ValueError("MTSWE HVP requires deployed subcycles [2,1,2,1]")
        children = getattr(self.timestepper, "time_integrators", ())
        if len(children) != 4:
            raise ValueError("MTSWE HVP requires four production integrator objects")
        model = children[0].model
        if tuple(model.get_x_var_list()) != _MTSWE_FIELDS:
            raise ValueError("MTSWE HVP requires [v,h,S,Qv,Qc,Qr]")
        if model.get_coeff_list() != ["s", "c0"]:
            raise ValueError(
                "MTSWE HVP control mode requires coefficient order ['s','c0']"
            )
        hyper = [
            term
            for term in model.dynamics.forcing_terms
            if term.name == "hyperviscosity"
        ]
        moist = [
            term
            for term in model.dynamics.forcing_terms
            if term.name == "threewayphysics"
        ]
        if len(hyper) != 1 or not hyper[0].treat_as_coeffs:
            raise ValueError("physical c0 must be retained by the production form")
        if len(moist) != 1 or moist[0].treat_as_coeffs:
            raise ValueError("moist parameters must remain fixed in this milestone")

    def _require_state(self, name, value):
        self.dry_helper._require_state(name, value)

    def _require_dual(self, name, value):
        self.dry_helper._require_state_dual(name, value)

    def _state_from_container(self, name, value):
        return self.dry_helper._state_from_container(name, value)

    @staticmethod
    def _require_nsteps(nsteps):
        if not isinstance(nsteps, Integral) or isinstance(nsteps, bool):
            raise TypeError("nsteps must be an integer")
        if int(nsteps) < 1:
            raise ValueError("nsteps must be positive")
        return int(nsteps)

    def state_mass_map(self, value, name="mtswe_state_mass_map"):
        return self.dry_helper.state_mass_map(value, name=name)

    def state_riesz_representative(
        self, dual, name="mtswe_state_riesz_representative"
    ):
        return self.dry_helper.state_riesz_representative(dual, name=name)

    def dual_pairing(self, dual, primal):
        return self.dry_helper.dual_pairing(dual, primal)

    def _child_specs(self, t0, dt):
        specs = []
        counters = {0: 0, 2: 0}
        for integrator_index, count in enumerate(self.timestepper.subcycle_list):
            child_dt = dt / count
            for subcycle_index in range(count):
                if integrator_index == 0:
                    name = f"dry_rk4_{counters[0]}"
                    counters[0] += 1
                elif integrator_index == 1:
                    name = "hyperviscosity_euler"
                elif integrator_index == 2:
                    name = f"dg_ssprk43_{counters[2]}"
                    counters[2] += 1
                else:
                    name = "moist_euler"
                specs.append(
                    (
                        name,
                        integrator_index,
                        subcycle_index,
                        t0 + subcycle_index * child_dt,
                        child_dt,
                    )
                )
        if tuple(spec[0] for spec in specs) != _FORWARD_CHILD_ORDER:
            raise RuntimeError("expanded production MTSWE child order changed")
        return tuple(specs)

    def _forward_child(self, name, state, child_t0, child_dt):
        if name.startswith("dry_rk4"):
            return self.dry_helper.take_forward_step_cached(
                state, child_t0, child_dt
            )
        if name == "hyperviscosity_euler":
            return self.hyper_helper.take_forward_step_cached(
                state, child_t0, child_dt
            )
        if name.startswith("dg_ssprk43"):
            return self.dg_helper.take_forward_step_cached(
                state, child_t0, child_dt
            )
        if name == "moist_euler":
            return self.moist_helper.take_forward_step_cached(
                state, child_t0, child_dt
            )
        raise RuntimeError(f"unknown MTSWE child {name}")

    def take_forward_step_cached(self, xn, tn, dt):
        state_in = self._state_from_container("xn", xn)
        t0 = _as_float("tn", tn)
        step = _as_float("dt", dt)
        current = _copy_function(state_in, "mtswe_current_state_0")
        boundaries = [_copy_function(state_in, "mtswe_boundary_state_0")]
        children = []
        for boundary_index, spec in enumerate(self._child_specs(t0, step), 1):
            name, integrator_index, subcycle_index, child_t0, child_dt = spec
            cache = self._forward_child(name, current, child_t0, child_dt)
            children.append(
                MTSWEChildPrimalCache(
                    name=name,
                    integrator_index=integrator_index,
                    subcycle_index=subcycle_index,
                    t0=child_t0,
                    dt=child_dt,
                    cache=cache,
                )
            )
            current = _copy_function(
                cache.state_out, f"mtswe_current_state_{boundary_index}"
            )
            boundaries.append(
                _copy_function(
                    current, f"mtswe_boundary_state_{boundary_index}"
                )
            )
        return MTSWESplitPrimalCache(
            t0=t0,
            dt=step,
            state_in=_copy_function(state_in, "mtswe_state_in_cache"),
            boundary_states=tuple(boundaries),
            children=tuple(children),
            state_out=_copy_function(current, "mtswe_state_out_cache"),
            forward_child_order=_FORWARD_CHILD_ORDER,
        )

    def _tangent_child(self, child, direction, delta_c0):
        if child.name.startswith("dry_rk4"):
            return self.dry_helper.take_tangent_step(child.cache, direction)
        if child.name == "hyperviscosity_euler":
            return self.hyper_helper.take_tangent_step(
                child.cache, direction, delta_c0
            )
        if child.name.startswith("dg_ssprk43"):
            return self.dg_helper.take_tangent_step(child.cache, direction)
        return self.moist_helper.take_tangent_step(child.cache, direction)

    def take_tangent_step(self, primal, delta_x_in, delta_c0):
        if not isinstance(primal, MTSWESplitPrimalCache):
            raise TypeError("primal must be a MTSWESplitPrimalCache")
        direction = self._state_from_container("delta_x_in", delta_x_in)
        parameter_direction = _as_float("delta_c0", delta_c0)
        current = _copy_function(direction, "mtswe_current_direction_0")
        boundaries = [
            _copy_function(direction, "mtswe_boundary_state_direction_0")
        ]
        children = []
        for boundary_index, child in enumerate(primal.children, 1):
            cache = self._tangent_child(child, current, parameter_direction)
            children.append(MTSWEChildTangentCache(primal=child, cache=cache))
            current = _copy_function(
                cache.state_direction_out,
                f"mtswe_current_state_direction_{boundary_index}",
            )
            boundaries.append(
                _copy_function(
                    current,
                    f"mtswe_boundary_state_direction_{boundary_index}",
                )
            )
        return MTSWESplitTangentCache(
            primal=primal,
            delta_c0=parameter_direction,
            state_direction_in=_copy_function(
                direction, "mtswe_state_direction_in_cache"
            ),
            boundary_state_directions=tuple(boundaries),
            children=tuple(children),
            state_direction_out=_copy_function(
                current, "mtswe_state_direction_out_cache"
            ),
            forward_child_order=_FORWARD_CHILD_ORDER,
        )

    def _reverse_child(self, child, lambda_plus_star):
        if child.name.startswith("dry_rk4"):
            return self.dry_helper.take_adjoint_step_cached(
                child.cache, lambda_plus_star
            )
        if child.name == "hyperviscosity_euler":
            return self.hyper_helper.take_adjoint_step_cached(
                child.cache, lambda_plus_star
            )
        if child.name.startswith("dg_ssprk43"):
            return self.dg_helper.take_adjoint_step_cached(
                child.cache, lambda_plus_star
            )
        return self.moist_helper.take_adjoint_step_cached(
            child.cache, lambda_plus_star
        )

    def take_adjoint_step_cached(self, primal, lambda_plus_star):
        if not isinstance(primal, MTSWESplitPrimalCache):
            raise TypeError("primal must be a MTSWESplitPrimalCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        current = _copy_cofunction(
            lambda_plus_star, "mtswe_current_state_adjoint"
        )
        c0_gradient = 0.0
        children = []
        for child in reversed(primal.children):
            result = self._reverse_child(child, current)
            children.append(MTSWEChildReverseData(child.name, result))
            c0_gradient += float(result.c0_gradient)
            current = _copy_cofunction(
                result.state_adjoint_in, "mtswe_current_state_adjoint"
            )
        return MTSWESplitReverseResult(
            state_adjoint_in=_copy_cofunction(
                current, "mtswe_state_adjoint_in"
            ),
            physical_c0_gradient=c0_gradient,
            children=tuple(children),
            reverse_child_order=_REVERSE_CHILD_ORDER,
        )

    def _incremental_reverse_child(
        self, child, lambda_plus_star, mu_plus_star
    ):
        name = child.primal.name
        if name.startswith("dry_rk4"):
            return self.dry_helper.take_incremental_adjoint_step(
                child.cache, lambda_plus_star, mu_plus_star
            )
        if name == "hyperviscosity_euler":
            return self.hyper_helper.take_incremental_adjoint_step(
                child.cache, lambda_plus_star, mu_plus_star
            )
        if name.startswith("dg_ssprk43"):
            return self.dg_helper.take_incremental_adjoint_step(
                child.cache, lambda_plus_star, mu_plus_star
            )
        return self.moist_helper.take_incremental_adjoint_step(
            child.cache, lambda_plus_star, mu_plus_star
        )

    def take_incremental_adjoint_step(
        self, tangent, lambda_plus_star, mu_plus_star
    ):
        if not isinstance(tangent, MTSWESplitTangentCache):
            raise TypeError("tangent must be a MTSWESplitTangentCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        self._require_dual("mu_plus_star", mu_plus_star)
        current = _copy_cofunction(
            lambda_plus_star, "mtswe_hvp_current_state_adjoint"
        )
        current_incremental = _copy_cofunction(
            mu_plus_star, "mtswe_current_incremental_state_adjoint"
        )
        c0_gradient = 0.0
        c0_hvp = 0.0
        children = []
        ordinary_children = []
        for child in reversed(tangent.children):
            result = self._incremental_reverse_child(
                child, current, current_incremental
            )
            name = child.primal.name
            children.append(MTSWEChildReverseData(name, result))
            ordinary_children.append(MTSWEChildReverseData(name, result.ordinary))
            c0_gradient += float(result.ordinary.c0_gradient)
            c0_hvp += float(result.c0_hvp)
            current = _copy_cofunction(
                result.ordinary.state_adjoint_in,
                "mtswe_hvp_current_state_adjoint",
            )
            current_incremental = _copy_cofunction(
                result.incremental_state_adjoint_in,
                "mtswe_current_incremental_state_adjoint",
            )
        ordinary = MTSWESplitReverseResult(
            state_adjoint_in=_copy_cofunction(
                current, "mtswe_hvp_ordinary_state_adjoint_in"
            ),
            physical_c0_gradient=c0_gradient,
            children=tuple(ordinary_children),
            reverse_child_order=_REVERSE_CHILD_ORDER,
        )
        return MTSWESplitHVPResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=_copy_cofunction(
                current_incremental, "mtswe_incremental_state_adjoint_in"
            ),
            physical_c0_hvp=c0_hvp,
            children=tuple(children),
            reverse_child_order=_REVERSE_CHILD_ORDER,
        )

    def production_graph_diagnostics(self):
        return {
            "timestepper_list": tuple(self.timestepper.timestepper_list),
            "termlist": tuple(tuple(terms) for terms in self.timestepper.termlist),
            "subcycle_list": tuple(self.timestepper.subcycle_list),
            "forward_child_order": _FORWARD_CHILD_ORDER,
            "reverse_child_order": _REVERSE_CHILD_ORDER,
            "dry_reverse_stage_order": _REVERSE_RK4_ORDER,
            "dg_reverse_stage_order": _REVERSE_RK4_ORDER,
            "moist_reverse_stage_order": (0,),
            "limiter_post_step_invoked": False,
            "moist_rate_dt_source": "ThreeWayPhysics.dt configured at construction",
            "coefficient_order": tuple(self.model.get_coeff_list()),
            "dry_form_identities": self.dry_helper.stage_form_identity_diagnostics(),
            "dg_form_identities": self.dg_helper.stage_form_identity_diagnostics(),
            "moist_form_identity": self.moist_helper.form_identity_diagnostics(),
        }

    def _forward_trajectory(self, nsteps, state_initial, t0, dt):
        count = self._require_nsteps(nsteps)
        self._require_state("state_initial", state_initial)
        start = _as_float("t0", t0)
        step = _as_float("dt", dt)
        states = [_copy_function(state_initial, "mtswe_state_0")]
        caches = []
        current = _copy_function(state_initial, "mtswe_trajectory_current")
        for n in range(count):
            cache = self.take_forward_step_cached(
                current, start + n * step, step
            )
            caches.append(cache)
            current = _copy_function(cache.state_out, f"mtswe_state_{n + 1}")
            states.append(_copy_function(current, f"mtswe_state_owned_{n + 1}"))
        return tuple(states), tuple(caches)

    def _tangent_trajectory(
        self, nsteps, state_initial, t0, dt, delta_x0, delta_c0
    ):
        count = self._require_nsteps(nsteps)
        self._require_state("state_initial", state_initial)
        self._require_state("delta_x0", delta_x0)
        start = _as_float("t0", t0)
        step = _as_float("dt", dt)
        parameter_direction = _as_float("delta_c0", delta_c0)
        states = [_copy_function(state_initial, "mtswe_state_0")]
        directions = [_copy_function(delta_x0, "mtswe_direction_0")]
        tangents = []
        current = _copy_function(state_initial, "mtswe_tangent_current_state")
        current_direction = _copy_function(
            delta_x0, "mtswe_tangent_current_direction"
        )
        for n in range(count):
            primal = self.take_forward_step_cached(
                current, start + n * step, step
            )
            tangent = self.take_tangent_step(
                primal, current_direction, parameter_direction
            )
            tangents.append(tangent)
            current = _copy_function(primal.state_out, f"mtswe_state_{n + 1}")
            current_direction = _copy_function(
                tangent.state_direction_out, f"mtswe_direction_{n + 1}"
            )
            states.append(_copy_function(current, f"mtswe_state_owned_{n + 1}"))
            directions.append(
                _copy_function(
                    current_direction, f"mtswe_direction_owned_{n + 1}"
                )
            )
        return tuple(states), tuple(directions), tuple(tangents)

    def _terminal_residual(self, state, target):
        self._require_state("state", state)
        self._require_state("target", target)
        residual = _copy_function(state, "mtswe_terminal_residual")
        with residual.dat.vec as residual_vec, target.dat.vec_ro as target_vec:
            residual_vec.axpy(-1.0, target_vec)
        return residual

    def terminal_least_squares_gradient(
        self, nsteps, state_initial, t0, dt, target
    ):
        states, caches = self._forward_trajectory(
            nsteps, state_initial, t0, dt
        )
        residual = self._terminal_residual(states[-1], target)
        objective = 0.5 * float(
            assemble(inner(residual, residual) * self.model.spaces.dx)
        )
        terminal = self.state_mass_map(residual, "mtswe_terminal_adjoint")
        current = _copy_cofunction(terminal, "mtswe_reduced_current_adjoint")
        c0_gradient = 0.0
        reverses = []
        for cache in reversed(caches):
            reverse = self.take_adjoint_step_cached(cache, current)
            reverses.append(reverse)
            c0_gradient += reverse.physical_c0_gradient
            current = _copy_cofunction(
                reverse.state_adjoint_in, "mtswe_reduced_current_adjoint"
            )
        return MTSWEReducedGradientResult(
            objective_value=objective,
            initial_condition_gradient=_copy_cofunction(
                current, "mtswe_initial_condition_gradient"
            ),
            physical_c0_gradient=c0_gradient,
            terminal_adjoint=_copy_cofunction(
                terminal, "mtswe_terminal_adjoint_result"
            ),
            states=states,
            primal_caches=caches,
            reverse_results=tuple(reverses),
        )

    def terminal_least_squares_hvp(
        self,
        nsteps,
        state_initial,
        t0,
        dt,
        target,
        delta_x0,
        delta_c0,
    ):
        states, directions, tangents = self._tangent_trajectory(
            nsteps, state_initial, t0, dt, delta_x0, delta_c0
        )
        residual = self._terminal_residual(states[-1], target)
        objective = 0.5 * float(
            assemble(inner(residual, residual) * self.model.spaces.dx)
        )
        terminal = self.state_mass_map(
            residual, "mtswe_hvp_terminal_adjoint"
        )
        terminal_incremental = self.state_mass_map(
            directions[-1], "mtswe_terminal_incremental_adjoint"
        )
        current = _copy_cofunction(terminal, "mtswe_hvp_current_adjoint")
        current_incremental = _copy_cofunction(
            terminal_incremental, "mtswe_hvp_current_incremental_adjoint"
        )
        c0_gradient = 0.0
        c0_hvp = 0.0
        reverses = []
        for tangent in reversed(tangents):
            reverse = self.take_incremental_adjoint_step(
                tangent, current, current_incremental
            )
            reverses.append(reverse)
            c0_gradient += reverse.ordinary.physical_c0_gradient
            c0_hvp += reverse.physical_c0_hvp
            current = _copy_cofunction(
                reverse.ordinary.state_adjoint_in,
                "mtswe_hvp_current_adjoint",
            )
            current_incremental = _copy_cofunction(
                reverse.incremental_state_adjoint_in,
                "mtswe_hvp_current_incremental_adjoint",
            )
        return MTSWEReducedHVPResult(
            objective_value=objective,
            initial_condition_gradient=_copy_cofunction(
                current, "mtswe_hvp_initial_condition_gradient"
            ),
            physical_c0_gradient=c0_gradient,
            initial_condition_hvp=_copy_cofunction(
                current_incremental, "mtswe_initial_condition_hvp"
            ),
            physical_c0_hvp=c0_hvp,
            terminal_adjoint=_copy_cofunction(
                terminal, "mtswe_hvp_terminal_adjoint_result"
            ),
            terminal_incremental_adjoint=_copy_cofunction(
                terminal_incremental,
                "mtswe_terminal_incremental_adjoint_result",
            ),
            states=states,
            state_directions=directions,
            primal_caches=tuple(tangent.primal for tangent in tangents),
            tangent_caches=tangents,
            reverse_results=tuple(reverses),
        )
