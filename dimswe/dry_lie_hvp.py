"""Exact dual-native HVPs for the production dry RK4--Euler Lie step.

The implementation is deliberately restricted to the dry three-field TSWE
configuration used by ``tests/tswe_rol_small.cfg``.  It differentiates the
deployed ``GeneralRK`` dry weak form stage by stage, composes that child with
the already-certified production hyperviscosity Euler child, and reverses one
or more complete Lie timesteps.  Legacy forward and adjoint entry points are
not used as derivative implementations and are not modified here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
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
_REVERSE_STAGE_ORDER = (3, 2, 1, 0)
_FORWARD_CHILD_ORDER = ("dry_rk4", "hyperviscosity_euler")
_REVERSE_CHILD_ORDER = ("hyperviscosity_euler", "dry_rk4")


@dataclass(frozen=True)
class DryRK4PrimalCache:
    """Owned primal data for one production dry RK4 child."""

    t0: float
    dt: float
    state_in: Function
    stage_states: tuple[Function, ...]
    stage_tendencies: tuple[Function, ...]
    state_out: Function


@dataclass(frozen=True)
class DryRK4TangentCache:
    """Owned tangent data for one owned dry RK4 primal cache."""

    primal: DryRK4PrimalCache
    state_direction_in: Function
    stage_state_directions: tuple[Function, ...]
    stage_tendency_directions: tuple[Function, ...]
    state_direction_out: Function


@dataclass(frozen=True)
class DryRK4ReverseStageData:
    """Owned ordinary reverse data for one RK stage."""

    stage_index: int
    tendency_adjoint: Cofunction
    reverse_auxiliary: Function
    stage_state_adjoint: Cofunction
    predecessor_tendency_adjoint_contributions: tuple[
        tuple[int, Cofunction], ...
    ]


@dataclass(frozen=True)
class DryRK4ReverseResult:
    """Owned ordinary dual reverse of one production dry RK4 child."""

    state_adjoint_in: Cofunction
    c0_gradient: float
    stages: tuple[DryRK4ReverseStageData, ...]
    reverse_stage_order: tuple[int, ...]


@dataclass(frozen=True)
class DryRK4StagePairingDiagnostic:
    """Natural-pairing residual for one isolated dry RK4 stage pullback."""

    stage_index: int
    tangent_pairing: float
    reverse_pairing: float
    absolute_error: float
    relative_error: float


@dataclass(frozen=True)
class DryRK4IncrementalReverseStageData:
    """Owned incremental reverse data for one RK stage."""

    stage_index: int
    incremental_tendency_adjoint: Cofunction
    incremental_reverse_auxiliary: Function
    incremental_stage_state_adjoint: Cofunction
    incremental_predecessor_tendency_adjoint_contributions: tuple[
        tuple[int, Cofunction], ...
    ]


@dataclass(frozen=True)
class DryRK4HVPResult:
    """Owned exact incremental reverse of one production dry RK4 child."""

    ordinary: DryRK4ReverseResult
    incremental_state_adjoint_in: Cofunction
    c0_hvp: float
    incremental_stages: tuple[DryRK4IncrementalReverseStageData, ...]
    reverse_stage_order: tuple[int, ...]


@dataclass(frozen=True)
class DryLiePrimalCache:
    """Owned primal data for dry RK4 followed by hyperviscosity Euler."""

    t0: float
    dt: float
    state_in: Function
    dry: DryRK4PrimalCache
    hyperviscosity: HyperviscosityPrimalCache
    state_out: Function
    forward_child_order: tuple[str, ...]


@dataclass(frozen=True)
class DryLieTangentCache:
    """Owned tangent data for one complete production dry Lie step."""

    primal: DryLiePrimalCache
    delta_c0: float
    state_direction_in: Function
    dry: DryRK4TangentCache
    hyperviscosity: HyperviscosityTangentCache
    state_direction_out: Function
    forward_child_order: tuple[str, ...]


@dataclass(frozen=True)
class DryLieReverseResult:
    """Owned ordinary reverse of one complete production dry Lie step."""

    state_adjoint_in: Cofunction
    physical_c0_gradient: float
    hyperviscosity: HyperviscosityReverseResult
    dry: DryRK4ReverseResult
    reverse_child_order: tuple[str, ...]


@dataclass(frozen=True)
class DryLieHVPResult:
    """Owned exact incremental reverse of one complete dry Lie step."""

    ordinary: DryLieReverseResult
    incremental_state_adjoint_in: Cofunction
    physical_c0_hvp: float
    hyperviscosity: HyperviscosityHVPResult
    dry: DryRK4HVPResult
    reverse_child_order: tuple[str, ...]


@dataclass(frozen=True)
class DryLieReducedGradientResult:
    """Reduced terminal least-squares value and ordinary gradient."""

    objective_value: float
    physical_c0_gradient: float
    initial_condition_gradient: Cofunction
    terminal_adjoint: Cofunction
    states: tuple[Function, ...]
    primal_caches: tuple[DryLiePrimalCache, ...]
    reverse_results: tuple[DryLieReverseResult, ...]


@dataclass(frozen=True)
class DryLieReducedHVPResult:
    """Reduced terminal least-squares gradient and HVP for ``(dx0, dc0)``."""

    objective_value: float
    physical_c0_gradient: float
    initial_condition_gradient: Cofunction
    physical_c0_hvp: float
    initial_condition_hvp: Cofunction
    terminal_adjoint: Cofunction
    terminal_incremental_adjoint: Cofunction
    states: tuple[Function, ...]
    state_directions: tuple[Function, ...]
    primal_caches: tuple[DryLiePrimalCache, ...]
    tangent_caches: tuple[DryLieTangentCache, ...]
    reverse_results: tuple[DryLieHVPResult, ...]


class ProductionDryRK4HVP:
    """Differentiate exactly one deployed dry ``GeneralRK`` RK4 child."""

    def __init__(self, timestepper):
        self.timestepper = timestepper
        self.model = timestepper.model
        self._validate_timestepper()

        self.state_space = self.model.dynamics.xspace
        self.state_dual_space = self.state_space.dual()

        self._state, _, self._state_split = self.model.get_x_var(
            "dry_rk4_hvp_form_state"
        )
        (
            self._coefficient,
            _,
            self._coefficient_split,
        ) = self.model.get_coeff_var("dry_rk4_hvp_form_coefficient")
        self._time = self.model.get_t_var()
        self._state_test, state_test_subs = self.model.get_x_test_vars()

        # Keep the former independently reconstructed form solely as a
        # diagnostic comparator.  The active derivative path below uses the
        # exact per-stage form objects built and solved by GeneralRK.
        self._reconstructed_stage_rhs = -self.model.rhs(
            self._state_split,
            self._time,
            self._coefficient_split,
            state_test_subs,
            terms=self.timestepper.terms,
        )
        self._production_stage_rhs = tuple(
            self.timestepper.production_stage_rhs_forms
        )
        self._production_state = self.timestepper.production_stage_base_state
        if len(self._production_stage_rhs) != 4:
            raise ValueError("production RK4 must expose four exact stage forms")
        if self._production_state is not self.timestepper.xk[0]:
            raise ValueError("production dry stage base-state identity changed")
        if any(
            coefficient is self._coefficient
            for coefficient in _form_items(
                self._reconstructed_stage_rhs, "coefficients"
            )
        ):
            raise ValueError(
                "production dry RK4 stage form unexpectedly depends on "
                "trainable coefficients"
            )
        for stage_rhs in self._production_stage_rhs:
            if any(
                coefficient is self.timestepper.coeff
                for coefficient in _form_items(stage_rhs, "coefficients")
            ):
                raise ValueError(
                    "exact production dry RK4 stage form unexpectedly "
                    "depends on trainable coefficients"
                )

        state_trial = TrialFunction(self.state_space)
        state_test = TestFunction(self.state_space)
        state_mass = assemble(
            inner(state_test, state_trial) * self.model.spaces.dx,
            mat_type="aij",
        )
        self._state_mass_solver = LinearSolver(
            state_mass,
            solver_parameters=dict(
                self.timestepper.solver_parameters["erkstage-f"]
            ),
        )

    def _validate_timestepper(self):
        if self.timestepper.__class__.__name__ != "RK4":
            raise ValueError("dry cached HVP API requires the production RK4 class")
        if self.timestepper.terms != ["model"]:
            raise ValueError("dry cached HVP API requires terms=['model'] exactly")
        if self.timestepper.nstages != 4 or not self.timestepper.is_explicit:
            raise ValueError("dry cached HVP API requires four explicit stages")
        if not (
            np.array_equal(self.timestepper.A, _RK4_A)
            and np.array_equal(self.timestepper.b, _RK4_B)
            and np.array_equal(self.timestepper.c, _RK4_C)
        ):
            raise ValueError(
                "dry cached HVP API is certified only for classical RK4"
            )
        if self.model.get_x_var_list() != ["v", "h", "S"]:
            raise ValueError(
                "dry cached HVP API is certified only for dry TSWE [v,h,S]"
            )
        if self.model.get_aux_var_list(terms=self.timestepper.terms):
            raise ValueError("dry RK4 child must not expose stage diagnostics")

    def _require_state(self, name, value):
        if not isinstance(value, Function):
            raise TypeError(f"{name} must be a Firedrake Function")
        if value.function_space() != self.state_space:
            raise ValueError(f"{name} belongs to the wrong mixed state space")

    def _require_state_dual(self, name, value):
        if not isinstance(value, Cofunction):
            raise TypeError(f"{name} must be a Firedrake Cofunction")
        if value.function_space() != self.state_dual_space:
            raise ValueError(f"{name} belongs to the wrong mixed dual space")

    def _state_from_container(self, name, value):
        if isinstance(value, Function):
            result = value
        elif isinstance(value, (list, tuple)) and value:
            result = value[0]
        else:
            raise TypeError(
                f"{name} must be a mixed Function or production state container"
            )
        self._require_state(name, result)
        return result

    @staticmethod
    def _zero_dual(dual_space, name):
        result = Cofunction(dual_space, name=name)
        result.zero()
        result.rename(name)
        return result

    def _assemble_dual(self, form, name):
        normalized = _normalize_derivative_form(form)
        if normalized.structural_zero:
            return self._zero_dual(self.state_dual_space, name)
        arguments = _form_items(normalized.normalized, "arguments")
        if len(arguments) != 1:
            metadata = _form_metadata(normalized.normalized)
            raise TypeError(
                "nonzero dry RK4 dual form has invalid arity: "
                f"{metadata}"
            )
        result = assemble(normalized.normalized)
        if not isinstance(result, Cofunction):
            raise TypeError("expected one-form assembly to return a Cofunction")
        if result.function_space() != self.state_dual_space:
            raise ValueError("assembled dry RK4 one-form has the wrong dual space")
        result.rename(name)
        return result

    def _solve_state_mass(self, dual, name):
        self._require_state_dual("dual", dual)
        rhs = _copy_cofunction(dual, f"{name}_rhs")
        result = Function(self.state_space, name=name)
        self._state_mass_solver.solve(result, rhs)
        return result

    def _primal_sum(self, base, terms, name):
        self._require_state("base", base)
        result = _copy_function(base, name)
        for scale, value in terms:
            self._require_state("primal summand", value)
            with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
                result_vec.axpy(float(scale), value_vec)
        return result

    def _dual_sum(self, terms, name):
        result = Cofunction(self.state_dual_space, name=name)
        result.zero()
        for scale, value in terms:
            self._require_state_dual("dual summand", value)
            with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
                result_vec.axpy(float(scale), value_vec)
        return result

    @staticmethod
    def _require_stage_index(stage_index):
        if not isinstance(stage_index, Integral) or isinstance(stage_index, bool):
            raise TypeError("stage_index must be an integer")
        index = int(stage_index)
        if index < 0 or index >= 4:
            raise ValueError("stage_index must be in [0, 3]")
        return index

    def _populate_production_graph(self, primal):
        """Assign the copied cache to the exact GeneralRK coefficient graph."""
        self.timestepper.t.assign(primal.t0)
        self.timestepper.dt.assign(primal.dt)
        self._production_state.assign(primal.state_in)
        for i in range(4):
            self.timestepper.Fi[i][0][0].assign(primal.stage_tendencies[i])

    def _populate_isolated_production_stage(
        self, primal, stage_index, stage_state
    ):
        """Perturb only Y_i while retaining its exact production expression."""
        self._require_state("stage_state", stage_state)
        self._populate_production_graph(primal)
        if stage_state is not primal.stage_states[stage_index]:
            offset = _copy_function(
                stage_state,
                f"dry_rk4_stage_{stage_index}_isolated_state_offset",
            )
            with (
                offset.dat.vec as offset_vec,
                primal.stage_states[stage_index].dat.vec_ro as cached_vec,
            ):
                offset_vec.axpy(-1.0, cached_vec)
            with (
                self._production_state.dat.vec as production_vec,
                offset.dat.vec_ro as offset_vec,
            ):
                production_vec.axpy(1.0, offset_vec)
        # Fi predecessors remain the exact copied production tendencies.
        # Changing xk by stage_state-Y_i therefore changes only the symbolic
        # stage state xk+dt*A[i,j]*Fi_j and preserves its deployed evaluation
        # graph, including coefficient identities and operation ordering.

    @staticmethod
    def _coefficient_name(coefficient):
        name_method = getattr(coefficient, "name", None)
        if callable(name_method):
            try:
                return str(name_method())
            except TypeError:
                pass
        return "<unnamed>"

    @staticmethod
    def _coefficient_count(coefficient):
        count_method = getattr(coefficient, "count", None)
        if callable(count_method):
            try:
                return int(count_method())
            except (TypeError, ValueError):
                pass
        return None

    def _coefficient_identity_records(self, form, known_coefficients):
        records = []
        for coefficient in _form_items(form, "coefficients"):
            records.append(
                {
                    "python_id": id(coefficient),
                    "ufl_count": self._coefficient_count(coefficient),
                    "name": self._coefficient_name(coefficient),
                    "type": (
                        f"{type(coefficient).__module__}."
                        f"{type(coefficient).__qualname__}"
                    ),
                    "identity_labels": tuple(
                        label
                        for label, candidate in known_coefficients
                        if candidate is not None and coefficient is candidate
                    ),
                }
            )
        return tuple(records)

    def stage_form_identity_diagnostics(self):
        """Report UFL coefficient membership using Python object identity."""
        known = [
            ("production_base_state_xk", self._production_state),
            ("production_time", self.timestepper.t),
            ("production_dt", self.timestepper.dt),
            ("production_trainable_coefficient", self.timestepper.coeff),
            ("reconstructed_state", self._state),
            ("reconstructed_time", self._time),
            ("reconstructed_trainable_coefficient", self._coefficient),
        ]
        known.extend(
            (
                f"production_stage_tendency_F{j}",
                self.timestepper.Fi[j][0][0],
            )
            for j in range(4)
        )
        reconstructed_coefficients = _form_items(
            self._reconstructed_stage_rhs, "coefficients"
        )
        reconstructed_ids = {id(item) for item in reconstructed_coefficients}
        reconstructed_records = self._coefficient_identity_records(
            self._reconstructed_stage_rhs, known
        )
        stages = []
        for i, stage_rhs in enumerate(self._production_stage_rhs):
            production_coefficients = _form_items(stage_rhs, "coefficients")
            production_ids = {id(item) for item in production_coefficients}
            stages.append(
                {
                    "stage_index": i,
                    "production_form_python_id": id(stage_rhs),
                    "production_residual_python_id": id(
                        self.timestepper.production_stage_residuals[i]
                    ),
                    "production_form_is_registered_generalrk_form": (
                        stage_rhs
                        is self.timestepper.production_stage_rhs_forms[i]
                    ),
                    "reconstructed_form_python_id": id(
                        self._reconstructed_stage_rhs
                    ),
                    "forms_are_identical_objects": (
                        stage_rhs is self._reconstructed_stage_rhs
                    ),
                    "production_stage_state_function": (
                        "xk with UFL stage expression "
                        "xk + dt*sum_j A[i,j]*F_j"
                    ),
                    "production_stage_predecessor_edges": tuple(
                        j
                        for j in range(i)
                        if self.timestepper.A[i, j] != 0.0
                    ),
                    "production_derivative_variable_python_id": id(
                        self._production_state
                    ),
                    "reconstructed_derivative_variable_python_id": id(
                        self._state
                    ),
                    "active_derivative_variable": "production_base_state_xk",
                    "former_derivative_variable": "reconstructed_state",
                    "production_derivative_variable_is_live_coefficient": any(
                        item is self._production_state
                        for item in production_coefficients
                    ),
                    "reconstructed_derivative_variable_is_live_coefficient": any(
                        item is self._state
                        for item in reconstructed_coefficients
                    ),
                    "production_coefficients": self._coefficient_identity_records(
                        stage_rhs, known
                    ),
                    "reconstructed_coefficients": reconstructed_records,
                    "production_coefficients_absent_from_reconstruction": tuple(
                        id(item)
                        for item in production_coefficients
                        if id(item) not in reconstructed_ids
                    ),
                    "reconstructed_coefficients_absent_from_production": tuple(
                        id(item)
                        for item in reconstructed_coefficients
                        if id(item) not in production_ids
                    ),
                    "production_integral_metadata": self._integral_metadata(
                        stage_rhs
                    ),
                    "reconstructed_integral_metadata": self._integral_metadata(
                        self._reconstructed_stage_rhs
                    ),
                }
            )
        return tuple(stages)

    def production_stage_tendency(
        self, primal, stage_index, stage_state
    ):
        """Run one exact deployed stage solve with only Y_i varied."""
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")
        index = self._require_stage_index(stage_index)
        self._populate_isolated_production_stage(primal, index, stage_state)
        self.timestepper.Fi[index][0][0].assign(0)
        self.timestepper.auxsolvers[index].solve()
        self.timestepper.Fsolvers[index].solve()
        return _copy_function(
            self.timestepper.Fi[index][0][0],
            f"dry_rk4_stage_{index}_isolated_production_tendency",
        )

    def perturbed_production_stage_tendency(
        self, primal, stage_index, stage_direction, direction_scale
    ):
        """Run exact B_i after the symbolic Y_i is changed by scale*W_i."""
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")
        index = self._require_stage_index(stage_index)
        self._require_state("stage_direction", stage_direction)
        scale = _as_float("direction_scale", direction_scale)
        self._populate_production_graph(primal)
        with (
            self._production_state.dat.vec as production_vec,
            stage_direction.dat.vec_ro as direction_vec,
        ):
            production_vec.axpy(scale, direction_vec)
        self.timestepper.Fi[index][0][0].assign(0)
        self.timestepper.auxsolvers[index].solve()
        self.timestepper.Fsolvers[index].solve()
        return _copy_function(
            self.timestepper.Fi[index][0][0],
            f"dry_rk4_stage_{index}_perturbed_production_tendency",
        )

    def production_stage_tangent(
        self, primal, stage_index, stage_state, stage_direction
    ):
        """Apply D B_i from the exact deployed production form object."""
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")
        index = self._require_stage_index(stage_index)
        self._require_state("stage_direction", stage_direction)
        self._populate_isolated_production_stage(primal, index, stage_state)
        tangent_rhs = self._assemble_dual(
            derivative(
                self._production_stage_rhs[index],
                self._production_state,
                stage_direction,
            ),
            f"dry_rk4_stage_{index}_isolated_production_tangent_rhs",
        )
        return self._solve_state_mass(
            tangent_rhs,
            f"dry_rk4_stage_{index}_isolated_production_tangent",
        )

    def reconstructed_stage_tangent(
        self, primal, stage_index, stage_state, stage_direction
    ):
        """Apply the former independently reconstructed D B comparator."""
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")
        index = self._require_stage_index(stage_index)
        self._require_state("stage_state", stage_state)
        self._require_state("stage_direction", stage_direction)
        self._state.assign(stage_state)
        self._time.assign(
            primal.t0 + float(self.timestepper.c[index]) * primal.dt
        )
        tangent_rhs = self._assemble_dual(
            derivative(
                self._reconstructed_stage_rhs,
                self._state,
                stage_direction,
            ),
            f"dry_rk4_stage_{index}_isolated_reconstructed_tangent_rhs",
        )
        return self._solve_state_mass(
            tangent_rhs,
            f"dry_rk4_stage_{index}_isolated_reconstructed_tangent",
        )

    @staticmethod
    def _integral_metadata(form):
        result = []
        for integral in _form_items(form, "integrals"):
            metadata_method = getattr(integral, "metadata", None)
            metadata = metadata_method() if callable(metadata_method) else {}
            integral_type_method = getattr(integral, "integral_type", None)
            integral_type = (
                integral_type_method()
                if callable(integral_type_method)
                else "<unavailable>"
            )
            result.append(
                {
                    "integral_type": str(integral_type),
                    "metadata": tuple(
                        sorted(
                            (str(key), repr(value))
                            for key, value in metadata.items()
                        )
                    ),
                    "integrand_type": (
                        f"{type(integral.integrand()).__module__}."
                        f"{type(integral.integrand()).__qualname__}"
                    ),
                }
            )
        return tuple(result)

    @staticmethod
    def _solver_parameters(parameters):
        return tuple(
            sorted((str(key), repr(value)) for key, value in parameters.items())
        )

    @staticmethod
    def _relative_function_difference(left, right):
        with left.dat.vec_ro as left_vec, right.dat.vec_ro as right_vec:
            difference = left_vec.copy()
            difference.axpy(-1.0, right_vec)
            denominator = max(right_vec.norm(), np.finfo(float).tiny)
            return difference.norm() / denominator

    def graph_diagnostics(self, primal):
        """Describe the exact cached and deployed RK4 graph without mutation."""
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")

        reconstructed_stages = tuple(
            self._primal_sum(
                primal.state_in,
                (
                    (
                        primal.dt * float(self.timestepper.A[i, j]),
                        primal.stage_tendencies[j],
                    )
                    for j in range(i)
                    if self.timestepper.A[i, j] != 0.0
                ),
                f"dry_rk4_stage_{i}_graph_reconstruction",
            )
            for i in range(4)
        )
        reconstructed_out = self._primal_sum(
            primal.state_in,
            (
                (
                    primal.dt * float(self.timestepper.b[i]),
                    primal.stage_tendencies[i],
                )
                for i in range(4)
            ),
            "dry_rk4_graph_output_reconstruction",
        )
        stage_aliases_scratch = tuple(
            (
                primal.stage_states[i].dat is self.timestepper.xk[0].dat,
                primal.stage_states[i].dat
                is self.timestepper.Fi[i][0][0].dat,
                primal.stage_tendencies[i].dat
                is self.timestepper.Fi[i][0][0].dat,
            )
            for i in range(4)
        )
        cached_pair_aliases = tuple(
            primal.stage_states[i].dat is primal.stage_tendencies[i].dat
            for i in range(4)
        )
        return {
            "terms": tuple(self.timestepper.terms),
            "tableau_a": tuple(
                tuple(float(value) for value in row)
                for row in self.timestepper.A
            ),
            "tableau_b": tuple(float(value) for value in self.timestepper.b),
            "tableau_c": tuple(float(value) for value in self.timestepper.c),
            "stage_times": tuple(
                primal.t0 + float(value) * primal.dt
                for value in self.timestepper.c
            ),
            "final_tendency_coefficients": tuple(
                primal.dt * float(value) for value in self.timestepper.b
            ),
            "later_stage_to_earlier_tendency_coefficients": tuple(
                tuple(
                    primal.dt * float(self.timestepper.A[j, i])
                    for j in range(i + 1, 4)
                )
                for i in range(4)
            ),
            "reverse_stage_order": _REVERSE_STAGE_ORDER,
            "stage_state_reconstruction_relative_errors": tuple(
                self._relative_function_difference(
                    reconstructed_stages[i], primal.stage_states[i]
                )
                for i in range(4)
            ),
            "final_state_reconstruction_relative_error": (
                self._relative_function_difference(
                    reconstructed_out, primal.state_out
                )
            ),
            "stage_aliases_scratch": stage_aliases_scratch,
            "cached_state_tendency_aliases": cached_pair_aliases,
            "new_stage_rhs_integral_metadata": self._integral_metadata(
                self._reconstructed_stage_rhs
            ),
            "active_exact_production_stage_integral_metadata": tuple(
                self._integral_metadata(form)
                for form in self._production_stage_rhs
            ),
            "legacy_forward_stage_integral_metadata": tuple(
                self._integral_metadata(form)
                for form in self.timestepper.production_stage_rhs_forms
            ),
            "legacy_reverse_stage_integral_metadata": tuple(
                self._integral_metadata(form)
                for form in self.timestepper.production_reverse_stage_residuals
            ),
            "new_mass_solver_parameters": self._solver_parameters(
                self.timestepper.solver_parameters["erkstage-f"]
            ),
            "legacy_forward_solver_parameters": self._solver_parameters(
                self.timestepper.solver_parameters["erkstage-f"]
            ),
            "legacy_reverse_solver_parameters": self._solver_parameters(
                self.timestepper.solver_parameters["erkstage-mu"]
            ),
            "incoming_identity_coefficient": 1.0,
            "stage_state_identity_coefficients": (1.0, 1.0, 1.0, 1.0),
            "stage_adjoint_initialization": (
                "q_i=dt*b_i*lambda_plus_star; then add the exact "
                "D_Fi <B_j,psi_j> production-form edge for j>i"
            ),
            "rhs_sign_convention": "B=-model.rhs",
            "adjoint_convention": "Cofunction in V*; explicit M psi=dual",
        }

    def stage_pairing_diagnostics(self, tangent, reverse):
        """Measure each exact production-form tangent/pullback pairing."""
        if not isinstance(tangent, DryRK4TangentCache):
            raise TypeError("tangent must be a DryRK4TangentCache")
        if not isinstance(reverse, DryRK4ReverseResult):
            raise TypeError("reverse must be a DryRK4ReverseResult")
        diagnostics = []
        for i in range(4):
            tangent_pairing = self.dual_pairing(
                reverse.stages[i].tendency_adjoint,
                tangent.stage_tendency_directions[i],
            )
            reverse_pairing = self.dual_pairing(
                reverse.stages[i].stage_state_adjoint,
                tangent.state_direction_in,
            )
            for predecessor_index, contribution in (
                reverse.stages[
                    i
                ].predecessor_tendency_adjoint_contributions
            ):
                reverse_pairing += self.dual_pairing(
                    contribution,
                    tangent.stage_tendency_directions[predecessor_index],
                )
            absolute_error = abs(tangent_pairing - reverse_pairing)
            scale = max(
                abs(tangent_pairing),
                abs(reverse_pairing),
                np.finfo(float).tiny,
            )
            diagnostics.append(
                DryRK4StagePairingDiagnostic(
                    stage_index=i,
                    tangent_pairing=tangent_pairing,
                    reverse_pairing=reverse_pairing,
                    absolute_error=absolute_error,
                    relative_error=absolute_error / scale,
                )
            )
        return tuple(diagnostics)

    def state_mass_map(self, value, name="dry_rk4_state_mass_map"):
        self._require_state("value", value)
        return self._assemble_dual(
            inner(self._state_test, value) * self.model.spaces.dx,
            name,
        )

    def state_riesz_representative(
        self, dual, name="dry_rk4_state_riesz_representative"
    ):
        return self._solve_state_mass(dual, name)

    def dual_pairing(self, dual, primal):
        self._require_state_dual("dual", dual)
        self._require_state("primal", primal)
        return float(assemble(action(dual, primal)))

    def _production_graph_directional_derivative(
        self,
        form,
        stage_index,
        state_direction,
        predecessor_tendency_directions,
    ):
        """Differentiate every live state edge in exact production B_i."""
        self._require_state("state_direction", state_direction)
        result = derivative(form, self._production_state, state_direction)
        for j in range(stage_index):
            if self.timestepper.A[stage_index, j] == 0.0:
                continue
            direction = predecessor_tendency_directions[j]
            self._require_state("predecessor tendency direction", direction)
            result = result + derivative(
                form,
                self.timestepper.Fi[j][0][0],
                direction,
            )
        return result

    @staticmethod
    def _predecessor_contribution(stage, predecessor_index):
        for index, contribution in (
            stage.predecessor_tendency_adjoint_contributions
        ):
            if index == predecessor_index:
                return contribution
        return None

    @staticmethod
    def _incremental_predecessor_contribution(stage, predecessor_index):
        for index, contribution in (
            stage.incremental_predecessor_tendency_adjoint_contributions
        ):
            if index == predecessor_index:
                return contribution
        return None

    def take_forward_step_cached(self, xn, tn, dt):
        state_in = self._state_from_container("xn", xn)
        t0 = _as_float("tn", tn)
        step_size = _as_float("dt", dt)

        state_out_container, state_out_sub, _ = self.model.get_full_var(
            "dry_rk4_cached_state_out", split_x_and_aux=True
        )
        legacy_input = xn if isinstance(xn, (list, tuple)) else [state_in]
        self.timestepper.reset_internal_vars()
        self.timestepper.take_forward_step(
            state_out_container,
            state_out_sub,
            legacy_input,
            t0,
            step_size,
        )

        stage_tendencies = tuple(
            _copy_function(
                self.timestepper.Fi[i][0][0],
                f"dry_rk4_stage_{i}_tendency_cache",
            )
            for i in range(4)
        )
        stage_states = tuple(
            self._primal_sum(
                state_in,
                (
                    (
                        step_size * float(self.timestepper.A[i, j]),
                        stage_tendencies[j],
                    )
                    for j in range(i)
                    if self.timestepper.A[i, j] != 0.0
                ),
                f"dry_rk4_stage_{i}_state_cache",
            )
            for i in range(4)
        )
        return DryRK4PrimalCache(
            t0=t0,
            dt=step_size,
            state_in=_copy_function(state_in, "dry_rk4_state_in_cache"),
            stage_states=stage_states,
            stage_tendencies=stage_tendencies,
            state_out=_copy_function(
                state_out_container[0], "dry_rk4_state_out_cache"
            ),
        )

    def take_tangent_step(self, primal, delta_xn):
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")
        state_direction = self._state_from_container("delta_xn", delta_xn)
        stage_state_directions = []
        stage_tendency_directions = []

        for i in range(4):
            stage_direction = self._primal_sum(
                state_direction,
                (
                    (
                        primal.dt * float(self.timestepper.A[i, j]),
                        stage_tendency_directions[j],
                    )
                    for j in range(i)
                    if self.timestepper.A[i, j] != 0.0
                ),
                f"dry_rk4_stage_{i}_state_direction",
            )
            self._populate_production_graph(primal)
            tangent_rhs = self._assemble_dual(
                self._production_graph_directional_derivative(
                    self._production_stage_rhs[i],
                    i,
                    state_direction,
                    stage_tendency_directions,
                ),
                f"dry_rk4_stage_{i}_tangent_rhs",
            )
            tendency_direction = self._solve_state_mass(
                tangent_rhs,
                f"dry_rk4_stage_{i}_tendency_direction",
            )
            stage_state_directions.append(
                _copy_function(
                    stage_direction,
                    f"dry_rk4_stage_{i}_state_direction_cache",
                )
            )
            stage_tendency_directions.append(
                _copy_function(
                    tendency_direction,
                    f"dry_rk4_stage_{i}_tendency_direction_cache",
                )
            )

        state_direction_out = self._primal_sum(
            state_direction,
            (
                (
                    primal.dt * float(self.timestepper.b[i]),
                    stage_tendency_directions[i],
                )
                for i in range(4)
            ),
            "dry_rk4_state_direction_out",
        )
        return DryRK4TangentCache(
            primal=primal,
            state_direction_in=_copy_function(
                state_direction, "dry_rk4_state_direction_in_cache"
            ),
            stage_state_directions=tuple(stage_state_directions),
            stage_tendency_directions=tuple(stage_tendency_directions),
            state_direction_out=_copy_function(
                state_direction_out, "dry_rk4_state_direction_out_cache"
            ),
        )

    def _reverse_stage(self, primal, stage_index, tendency_adjoint):
        self._populate_production_graph(primal)
        reverse_auxiliary = self._solve_state_mass(
            tendency_adjoint,
            f"dry_rk4_stage_{stage_index}_reverse_auxiliary",
        )
        contracted = action(
            self._production_stage_rhs[stage_index], reverse_auxiliary
        )
        stage_state_adjoint = self._assemble_dual(
            derivative(contracted, self._production_state),
            f"dry_rk4_stage_{stage_index}_state_adjoint",
        )
        predecessor_contributions = []
        for j in range(stage_index):
            if self.timestepper.A[stage_index, j] == 0.0:
                continue
            contribution = self._assemble_dual(
                derivative(
                    contracted,
                    self.timestepper.Fi[j][0][0],
                ),
                (
                    f"dry_rk4_stage_{stage_index}_to_stage_{j}_"
                    "tendency_adjoint"
                ),
            )
            predecessor_contributions.append(
                (
                    j,
                    _copy_cofunction(
                        contribution,
                        (
                            f"dry_rk4_stage_{stage_index}_to_stage_{j}_"
                            "tendency_adjoint_result"
                        ),
                    ),
                )
            )
        return DryRK4ReverseStageData(
            stage_index=stage_index,
            tendency_adjoint=_copy_cofunction(
                tendency_adjoint,
                f"dry_rk4_stage_{stage_index}_tendency_adjoint_result",
            ),
            reverse_auxiliary=_copy_function(
                reverse_auxiliary,
                f"dry_rk4_stage_{stage_index}_reverse_auxiliary_result",
            ),
            stage_state_adjoint=_copy_cofunction(
                stage_state_adjoint,
                f"dry_rk4_stage_{stage_index}_state_adjoint_result",
            ),
            predecessor_tendency_adjoint_contributions=tuple(
                predecessor_contributions
            ),
        )

    def take_adjoint_step_cached(self, primal, lambda_plus_star):
        if not isinstance(primal, DryRK4PrimalCache):
            raise TypeError("primal must be a DryRK4PrimalCache")
        self._require_state_dual("lambda_plus_star", lambda_plus_star)
        stages = [None] * 4

        for i in _REVERSE_STAGE_ORDER:
            terms = [
                (primal.dt * float(self.timestepper.b[i]), lambda_plus_star)
            ]
            for j in range(i + 1, 4):
                later = stages[j]
                if later is None:
                    raise RuntimeError("dry RK4 reverse stage order was violated")
                contribution = self._predecessor_contribution(later, i)
                if contribution is not None:
                    terms.append((1.0, contribution))
            tendency_adjoint = self._dual_sum(
                terms, f"dry_rk4_stage_{i}_tendency_adjoint_sum"
            )
            stages[i] = self._reverse_stage(primal, i, tendency_adjoint)

        completed = tuple(stage for stage in stages if stage is not None)
        state_adjoint_in = self._dual_sum(
            [(1.0, lambda_plus_star)]
            + [(1.0, stage.stage_state_adjoint) for stage in completed],
            "dry_rk4_state_adjoint_in",
        )
        return DryRK4ReverseResult(
            state_adjoint_in=state_adjoint_in,
            c0_gradient=0.0,
            stages=completed,
            reverse_stage_order=_REVERSE_STAGE_ORDER,
        )

    def take_incremental_adjoint_step(
        self, tangent, lambda_plus_star, mu_plus_star
    ):
        if not isinstance(tangent, DryRK4TangentCache):
            raise TypeError("tangent must be a DryRK4TangentCache")
        self._require_state_dual("lambda_plus_star", lambda_plus_star)
        self._require_state_dual("mu_plus_star", mu_plus_star)
        primal = tangent.primal
        ordinary = self.take_adjoint_step_cached(primal, lambda_plus_star)
        incremental_stages = [None] * 4

        for i in _REVERSE_STAGE_ORDER:
            terms = [(primal.dt * float(self.timestepper.b[i]), mu_plus_star)]
            for j in range(i + 1, 4):
                later = incremental_stages[j]
                if later is None:
                    raise RuntimeError(
                        "dry RK4 incremental reverse stage order was violated"
                    )
                contribution = self._incremental_predecessor_contribution(
                    later, i
                )
                if contribution is not None:
                    terms.append((1.0, contribution))
            incremental_tendency_adjoint = self._dual_sum(
                terms,
                f"dry_rk4_stage_{i}_incremental_tendency_adjoint_sum",
            )
            incremental_reverse_auxiliary = self._solve_state_mass(
                incremental_tendency_adjoint,
                f"dry_rk4_stage_{i}_incremental_reverse_auxiliary",
            )

            self._populate_production_graph(primal)
            ordinary_contracted = action(
                self._production_stage_rhs[i],
                ordinary.stages[i].reverse_auxiliary,
            )
            incremental_contracted = action(
                self._production_stage_rhs[i], incremental_reverse_auxiliary
            )
            ordinary_state_pullback = derivative(
                ordinary_contracted, self._production_state
            )
            incremental_stage_state_adjoint = self._assemble_dual(
                derivative(incremental_contracted, self._production_state)
                + self._production_graph_directional_derivative(
                    ordinary_state_pullback,
                    i,
                    tangent.state_direction_in,
                    tangent.stage_tendency_directions,
                ),
                f"dry_rk4_stage_{i}_incremental_state_adjoint",
            )
            incremental_predecessor_contributions = []
            for j in range(i):
                if self.timestepper.A[i, j] == 0.0:
                    continue
                predecessor = self.timestepper.Fi[j][0][0]
                ordinary_predecessor_pullback = derivative(
                    ordinary_contracted, predecessor
                )
                incremental_predecessor = self._assemble_dual(
                    derivative(incremental_contracted, predecessor)
                    + self._production_graph_directional_derivative(
                        ordinary_predecessor_pullback,
                        i,
                        tangent.state_direction_in,
                        tangent.stage_tendency_directions,
                    ),
                    (
                        f"dry_rk4_stage_{i}_to_stage_{j}_"
                        "incremental_tendency_adjoint"
                    ),
                )
                incremental_predecessor_contributions.append(
                    (
                        j,
                        _copy_cofunction(
                            incremental_predecessor,
                            (
                                f"dry_rk4_stage_{i}_to_stage_{j}_"
                                "incremental_tendency_adjoint_result"
                            ),
                        ),
                    )
                )
            incremental_stages[i] = DryRK4IncrementalReverseStageData(
                stage_index=i,
                incremental_tendency_adjoint=_copy_cofunction(
                    incremental_tendency_adjoint,
                    f"dry_rk4_stage_{i}_incremental_tendency_adjoint_result",
                ),
                incremental_reverse_auxiliary=_copy_function(
                    incremental_reverse_auxiliary,
                    f"dry_rk4_stage_{i}_incremental_reverse_auxiliary_result",
                ),
                incremental_stage_state_adjoint=_copy_cofunction(
                    incremental_stage_state_adjoint,
                    f"dry_rk4_stage_{i}_incremental_state_adjoint_result",
                ),
                incremental_predecessor_tendency_adjoint_contributions=tuple(
                    incremental_predecessor_contributions
                ),
            )

        completed = tuple(
            stage for stage in incremental_stages if stage is not None
        )
        incremental_state_adjoint_in = self._dual_sum(
            [(1.0, mu_plus_star)]
            + [
                (1.0, stage.incremental_stage_state_adjoint)
                for stage in completed
            ],
            "dry_rk4_incremental_state_adjoint_in",
        )
        return DryRK4HVPResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=incremental_state_adjoint_in,
            c0_hvp=0.0,
            incremental_stages=completed,
            reverse_stage_order=_REVERSE_STAGE_ORDER,
        )


class ProductionDryLieHVP:
    """Exact cached derivatives of the deployed two-child dry Lie step."""

    def __init__(self, timestepper):
        self.timestepper = timestepper
        self._validate_timestepper()
        self.model = self.dry_child.model
        self.dry_helper = self.dry_child._get_dry_rk4_hvp_helper()
        self.hyperviscosity_helper = (
            self.hyperviscosity_child._get_hyperviscosity_hvp_helper()
        )
        self.state_space = self.dry_helper.state_space
        self.state_dual_space = self.dry_helper.state_dual_space

    def _validate_timestepper(self):
        children = getattr(self.timestepper, "time_integrators", None)
        if children is None or len(children) != 2:
            raise ValueError("dry Lie HVP requires exactly two production children")
        if getattr(self.timestepper, "timestepper_list", None) != ["RK4", "Euler"]:
            raise ValueError("dry Lie HVP requires child methods [RK4, Euler]")
        if getattr(self.timestepper, "termlist", None) != [
            ["model"],
            ["hyperviscosity"],
        ]:
            raise ValueError(
                "dry Lie HVP requires terms [model] then [hyperviscosity]"
            )
        if list(getattr(self.timestepper, "subcycle_list", ())) != [1, 1]:
            raise ValueError("dry Lie HVP is certified only for subcycles [1, 1]")
        self.dry_child, self.hyperviscosity_child = children

    def _require_primal(self, name, value):
        self.dry_helper._require_state(name, value)

    def _require_dual(self, name, value):
        self.dry_helper._require_state_dual(name, value)

    def state_mass_map(self, value, name="dry_lie_state_mass_map"):
        return self.dry_helper.state_mass_map(value, name=name)

    def state_riesz_representative(
        self, dual, name="dry_lie_state_riesz_representative"
    ):
        return self.dry_helper.state_riesz_representative(dual, name=name)

    def dual_pairing(self, dual, primal):
        return self.dry_helper.dual_pairing(dual, primal)

    def take_forward_step_cached(self, xn, tn, dt):
        state_in = self.dry_helper._state_from_container("xn", xn)
        t0 = _as_float("tn", tn)
        step_size = _as_float("dt", dt)
        dry = self.dry_child.take_dry_forward_step_cached(
            state_in, t0, step_size
        )
        # The deployed Lie loop supplies tn+k*sub_dt to each child.  With the
        # certified [1,1] subcycles, the Euler child therefore also starts t0.
        hyperviscosity = self.hyperviscosity_child.take_forward_step_cached(
            dry.state_out, t0, step_size
        )
        return DryLiePrimalCache(
            t0=t0,
            dt=step_size,
            state_in=_copy_function(state_in, "dry_lie_state_in_cache"),
            dry=dry,
            hyperviscosity=hyperviscosity,
            state_out=_copy_function(
                hyperviscosity.state_out, "dry_lie_state_out_cache"
            ),
            forward_child_order=_FORWARD_CHILD_ORDER,
        )

    def take_tangent_step(self, primal, delta_x_in, delta_c0):
        if not isinstance(primal, DryLiePrimalCache):
            raise TypeError("primal must be a DryLiePrimalCache")
        state_direction = self.dry_helper._state_from_container(
            "delta_x_in", delta_x_in
        )
        parameter_direction = _as_float("delta_c0", delta_c0)
        dry = self.dry_child.take_dry_tangent_step(
            primal.dry, state_direction
        )
        hyperviscosity = self.hyperviscosity_child.take_tangent_step(
            primal.hyperviscosity,
            dry.state_direction_out,
            parameter_direction,
        )
        return DryLieTangentCache(
            primal=primal,
            delta_c0=parameter_direction,
            state_direction_in=_copy_function(
                state_direction, "dry_lie_state_direction_in_cache"
            ),
            dry=dry,
            hyperviscosity=hyperviscosity,
            state_direction_out=_copy_function(
                hyperviscosity.state_direction_out,
                "dry_lie_state_direction_out_cache",
            ),
            forward_child_order=_FORWARD_CHILD_ORDER,
        )

    def take_adjoint_step_cached(self, primal, lambda_plus_star):
        if not isinstance(primal, DryLiePrimalCache):
            raise TypeError("primal must be a DryLiePrimalCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        hyperviscosity = (
            self.hyperviscosity_child.take_adjoint_step_cached(
                primal.hyperviscosity, lambda_plus_star
            )
        )
        dry = self.dry_child.take_dry_adjoint_step_cached(
            primal.dry, hyperviscosity.state_adjoint_in
        )
        return DryLieReverseResult(
            state_adjoint_in=_copy_cofunction(
                dry.state_adjoint_in, "dry_lie_state_adjoint_in"
            ),
            physical_c0_gradient=hyperviscosity.c0_gradient,
            hyperviscosity=hyperviscosity,
            dry=dry,
            reverse_child_order=_REVERSE_CHILD_ORDER,
        )

    def take_incremental_adjoint_step(
        self, tangent, lambda_plus_star, mu_plus_star
    ):
        if not isinstance(tangent, DryLieTangentCache):
            raise TypeError("tangent must be a DryLieTangentCache")
        self._require_dual("lambda_plus_star", lambda_plus_star)
        self._require_dual("mu_plus_star", mu_plus_star)

        hyperviscosity = (
            self.hyperviscosity_child.take_incremental_adjoint_step(
                tangent.hyperviscosity,
                lambda_plus_star,
                mu_plus_star,
            )
        )
        dry = self.dry_child.take_dry_incremental_adjoint_step(
            tangent.dry,
            hyperviscosity.ordinary.state_adjoint_in,
            hyperviscosity.incremental_state_adjoint_in,
        )
        ordinary = DryLieReverseResult(
            state_adjoint_in=_copy_cofunction(
                dry.ordinary.state_adjoint_in,
                "dry_lie_hvp_ordinary_state_adjoint_in",
            ),
            physical_c0_gradient=hyperviscosity.ordinary.c0_gradient,
            hyperviscosity=hyperviscosity.ordinary,
            dry=dry.ordinary,
            reverse_child_order=_REVERSE_CHILD_ORDER,
        )
        return DryLieHVPResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=_copy_cofunction(
                dry.incremental_state_adjoint_in,
                "dry_lie_incremental_state_adjoint_in",
            ),
            physical_c0_hvp=hyperviscosity.c0_hvp,
            hyperviscosity=hyperviscosity,
            dry=dry,
            reverse_child_order=_REVERSE_CHILD_ORDER,
        )

    @staticmethod
    def _require_nsteps(nsteps):
        if not isinstance(nsteps, Integral) or isinstance(nsteps, bool):
            raise TypeError("nsteps must be an integer")
        if int(nsteps) < 1:
            raise ValueError("nsteps must be positive")
        return int(nsteps)

    def _require_target(self, target):
        self._require_primal("target", target)

    def _forward_trajectory(self, nsteps, state_initial, t0, dt):
        count = self._require_nsteps(nsteps)
        self._require_primal("state_initial", state_initial)
        start_time = _as_float("t0", t0)
        step_size = _as_float("dt", dt)
        states = [_copy_function(state_initial, "dry_lie_state_0")]
        caches = []
        current = _copy_function(state_initial, "dry_lie_current_state")
        for n in range(count):
            cache = self.take_forward_step_cached(
                current, start_time + n * step_size, step_size
            )
            caches.append(cache)
            current = _copy_function(
                cache.state_out, f"dry_lie_current_state_{n + 1}"
            )
            states.append(
                _copy_function(current, f"dry_lie_state_{n + 1}")
            )
        return tuple(states), tuple(caches)

    def _tangent_trajectory(
        self, nsteps, state_initial, t0, dt, delta_x0, delta_c0
    ):
        count = self._require_nsteps(nsteps)
        self._require_primal("state_initial", state_initial)
        self._require_primal("delta_x0", delta_x0)
        start_time = _as_float("t0", t0)
        step_size = _as_float("dt", dt)
        parameter_direction = _as_float("delta_c0", delta_c0)
        states = [_copy_function(state_initial, "dry_lie_state_0")]
        directions = [
            _copy_function(delta_x0, "dry_lie_state_direction_0")
        ]
        tangent_caches = []
        current = _copy_function(state_initial, "dry_lie_current_state")
        current_direction = _copy_function(
            delta_x0, "dry_lie_current_state_direction"
        )
        for n in range(count):
            primal = self.take_forward_step_cached(
                current, start_time + n * step_size, step_size
            )
            tangent = self.take_tangent_step(
                primal, current_direction, parameter_direction
            )
            tangent_caches.append(tangent)
            current = _copy_function(
                primal.state_out, f"dry_lie_current_state_{n + 1}"
            )
            current_direction = _copy_function(
                tangent.state_direction_out,
                f"dry_lie_current_state_direction_{n + 1}",
            )
            states.append(
                _copy_function(current, f"dry_lie_state_{n + 1}")
            )
            directions.append(
                _copy_function(
                    current_direction, f"dry_lie_state_direction_{n + 1}"
                )
            )
        return tuple(states), tuple(directions), tuple(tangent_caches)

    def _terminal_residual(self, state, target):
        self._require_primal("terminal state", state)
        self._require_target(target)
        residual = _copy_function(state, "dry_lie_terminal_residual")
        with residual.dat.vec as residual_vec, target.dat.vec_ro as target_vec:
            residual_vec.axpy(-1.0, target_vec)
        return residual

    def terminal_least_squares_gradient(
        self, nsteps, state_initial, t0, dt, target
    ):
        states, primal_caches = self._forward_trajectory(
            nsteps, state_initial, t0, dt
        )
        residual = self._terminal_residual(states[-1], target)
        objective = 0.5 * float(
            assemble(inner(residual, residual) * self.model.spaces.dx)
        )
        terminal_adjoint = self.state_mass_map(
            residual, name="dry_lie_terminal_adjoint"
        )
        current = _copy_cofunction(
            terminal_adjoint, "dry_lie_current_state_adjoint"
        )
        physical_c0_gradient = 0.0
        reverse_results = []
        for primal in reversed(primal_caches):
            reverse = self.take_adjoint_step_cached(primal, current)
            reverse_results.append(reverse)
            physical_c0_gradient += reverse.physical_c0_gradient
            current = _copy_cofunction(
                reverse.state_adjoint_in,
                "dry_lie_current_state_adjoint",
            )
        return DryLieReducedGradientResult(
            objective_value=objective,
            physical_c0_gradient=physical_c0_gradient,
            initial_condition_gradient=_copy_cofunction(
                current, "dry_lie_initial_condition_gradient"
            ),
            terminal_adjoint=_copy_cofunction(
                terminal_adjoint, "dry_lie_terminal_adjoint_result"
            ),
            states=states,
            primal_caches=primal_caches,
            reverse_results=tuple(reverse_results),
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
        states, directions, tangent_caches = self._tangent_trajectory(
            nsteps,
            state_initial,
            t0,
            dt,
            delta_x0,
            delta_c0,
        )
        residual = self._terminal_residual(states[-1], target)
        objective = 0.5 * float(
            assemble(inner(residual, residual) * self.model.spaces.dx)
        )
        terminal_adjoint = self.state_mass_map(
            residual, name="dry_lie_hvp_terminal_adjoint"
        )
        terminal_incremental_adjoint = self.state_mass_map(
            directions[-1], name="dry_lie_terminal_incremental_adjoint"
        )
        current = _copy_cofunction(
            terminal_adjoint, "dry_lie_hvp_current_state_adjoint"
        )
        current_incremental = _copy_cofunction(
            terminal_incremental_adjoint,
            "dry_lie_current_incremental_state_adjoint",
        )
        physical_c0_gradient = 0.0
        physical_c0_hvp = 0.0
        reverse_results = []
        for tangent in reversed(tangent_caches):
            reverse = self.take_incremental_adjoint_step(
                tangent, current, current_incremental
            )
            reverse_results.append(reverse)
            physical_c0_gradient += reverse.ordinary.physical_c0_gradient
            physical_c0_hvp += reverse.physical_c0_hvp
            current = _copy_cofunction(
                reverse.ordinary.state_adjoint_in,
                "dry_lie_hvp_current_state_adjoint",
            )
            current_incremental = _copy_cofunction(
                reverse.incremental_state_adjoint_in,
                "dry_lie_current_incremental_state_adjoint",
            )

        return DryLieReducedHVPResult(
            objective_value=objective,
            physical_c0_gradient=physical_c0_gradient,
            initial_condition_gradient=_copy_cofunction(
                current, "dry_lie_hvp_initial_condition_gradient"
            ),
            physical_c0_hvp=physical_c0_hvp,
            initial_condition_hvp=_copy_cofunction(
                current_incremental, "dry_lie_initial_condition_hvp"
            ),
            terminal_adjoint=_copy_cofunction(
                terminal_adjoint, "dry_lie_hvp_terminal_adjoint_result"
            ),
            terminal_incremental_adjoint=_copy_cofunction(
                terminal_incremental_adjoint,
                "dry_lie_terminal_incremental_adjoint_result",
            ),
            states=states,
            state_directions=directions,
            primal_caches=tuple(
                tangent.primal for tangent in tangent_caches
            ),
            tangent_caches=tangent_caches,
            reverse_results=tuple(reverse_results),
        )
