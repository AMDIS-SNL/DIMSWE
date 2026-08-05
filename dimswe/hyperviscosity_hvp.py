"""Dual-native cached derivatives of the production hyperviscosity Euler child.

This module is intentionally narrow.  It differentiates the weak forms exposed
by a production ``GeneralRK`` instance configured as one explicit Euler stage
with ``terms=["hyperviscosity"]``.  It does not implement a generic RK or Lie
split HVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

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
from ufl import ZeroBaseForm
from ufl.algorithms import expand_derivatives
from ufl.classes import Zero as UFLZero
from ufl.form import Form


def _as_float(name, value):
    if isinstance(value, Real):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        if hasattr(value, "values"):
            values = value.values()
            if len(values) == 1:
                return float(values[0])
        raise TypeError(f"{name} must be a real scalar") from None


def _copy_function(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _copy_cofunction(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


@dataclass(frozen=True)
class _NormalizedDerivativeForm:
    expanded: object
    normalized: object
    structural_zero: bool


@dataclass(frozen=True)
class _ZeroDerivativeDebug:
    contracted: object
    raw_derivative: object
    expanded_derivative: object
    normalized_derivative: object
    dependency_absent: bool | None
    structural_zero: bool
    assembly_bypassed: bool
    raw_metadata: dict
    expanded_metadata: dict
    normalized_metadata: dict
    assembled_scalar_value: float | None


def _form_items(form, accessor):
    method = getattr(form, accessor, None)
    return tuple(method()) if callable(method) else ()


def _argument_space(argument):
    method = getattr(argument, "ufl_function_space", None)
    if callable(method):
        return str(method())
    return "<unavailable>"


def _form_metadata(form):
    arguments = _form_items(form, "arguments")
    return {
        "type": f"{type(form).__module__}.{type(form).__qualname__}",
        "argument_count": len(arguments),
        "argument_spaces": [_argument_space(arg) for arg in arguments],
        "integral_count": len(_form_items(form, "integrals")),
        "domain_count": len(_form_items(form, "ufl_domains")),
    }


def _normalize_derivative_form(form):
    """Expand derivatives and remove only canonically zero integrals."""
    if isinstance(form, Real):
        return _NormalizedDerivativeForm(form, form, form == 0)

    expanded = expand_derivatives(form)
    if isinstance(expanded, (ZeroBaseForm, UFLZero)):
        return _NormalizedDerivativeForm(expanded, expanded, True)

    integrals_method = getattr(expanded, "integrals", None)
    if not callable(integrals_method):
        return _NormalizedDerivativeForm(expanded, expanded, False)

    integrals = tuple(integrals_method())
    nonzero_integrals = tuple(
        integral
        for integral in integrals
        if not isinstance(integral.integrand(), UFLZero)
    )
    normalized = (
        expanded
        if len(nonzero_integrals) == len(integrals)
        else Form(list(nonzero_integrals))
    )
    return _NormalizedDerivativeForm(
        expanded,
        normalized,
        len(nonzero_integrals) == 0,
    )


@dataclass(frozen=True)
class HyperviscosityPrimalCache:
    """Owned primal data for one production hyperviscosity Euler child."""

    t0: float
    dt: float
    c0: float
    s: float
    state_in: Function
    diagnostic: Function
    tendency: Function
    state_out: Function


@dataclass(frozen=True)
class HyperviscosityTangentCache:
    """Owned tangent data associated with one owned primal cache."""

    primal: HyperviscosityPrimalCache
    delta_c0: float
    state_direction_in: Function
    diagnostic_direction: Function
    tendency_direction: Function
    state_direction_out: Function


@dataclass(frozen=True)
class HyperviscosityReverseResult:
    """Owned dual-native ordinary reverse data for one child step."""

    state_adjoint_in: Cofunction
    c0_gradient: float
    tendency_adjoint: Cofunction
    main_reverse_auxiliary: Function
    diagnostic_adjoint: Cofunction
    diagnostic_reverse_auxiliary: Function
    main_state_adjoint: Cofunction
    diagnostic_state_adjoint: Cofunction


@dataclass(frozen=True)
class HyperviscosityHVPResult:
    """Owned exact incremental reverse data for one child step."""

    ordinary: HyperviscosityReverseResult
    incremental_state_adjoint_in: Cofunction
    c0_hvp: float
    c0_hvp_from_incremental_adjoint: float
    c0_hvp_from_state_direction: float
    c0_hvp_from_pure_control_curvature: float
    incremental_tendency_adjoint: Cofunction
    incremental_main_reverse_auxiliary: Function
    incremental_diagnostic_adjoint: Cofunction
    incremental_diagnostic_reverse_auxiliary: Function
    incremental_main_state_adjoint: Cofunction
    incremental_diagnostic_state_adjoint: Cofunction


class ProductionHyperviscosityEulerHVP:
    """Differentiate the deployed hyperviscosity forms for one Euler child."""

    def __init__(self, timestepper):
        self.timestepper = timestepper
        self.model = timestepper.model
        self._validate_timestepper()

        self.state_space = self.model.dynamics.xspace
        self.state_dual_space = self.state_space.dual()
        self.auxiliary_space = self.model.dynamics.auxspace
        self.auxiliary_dual_space = self.auxiliary_space.dual()
        self.coefficient_space = self.model.dynamics.coeffspace

        # These are private form coefficients, independent of GeneralRK scratch.
        full, full_sub, full_split = self.model.get_full_var(
            "hyperviscosity_hvp_form_state", split_x_and_aux=True
        )
        self._state = full[0]
        self._auxiliary = full[1]
        self._state_sub = {
            name: full_sub[name] for name in self.model.get_x_var_list()
        }
        self._full_split = full_split
        self._coefficient, self._coefficient_sub, self._coefficient_split = (
            self.model.get_coeff_var("hyperviscosity_hvp_form_coefficient")
        )
        self._time = self.model.get_t_var()

        full_tests, full_test_subs = self.model.get_full_test_vars(
            split_x_and_aux=True
        )
        self._state_test = full_tests[0]
        self._auxiliary_test = full_tests[1]

        auxiliary_expressions = self.model.compute_aux_expressions(
            self._full_split,
            self._time,
            self._coefficient_split,
            full_test_subs,
            terms=self.timestepper.terms,
        )
        auxiliary_names = self.model.get_aux_var_list(
            terms=self.timestepper.terms
        )
        if not auxiliary_names:
            raise ValueError("hyperviscosity child must expose diagnostic fields")
        self._auxiliary_lhs = sum(
            auxiliary_expressions[name][0] for name in auxiliary_names
        )
        self._auxiliary_rhs = sum(
            auxiliary_expressions[name][1] for name in auxiliary_names
        )

        # This is the same B=-R form used by GeneralRK at production line 154.
        self._main_rhs = -self.model.rhs(
            self._full_split,
            self._time,
            self._coefficient_split,
            full_test_subs,
            terms=self.timestepper.terms,
        )

        state_trial = TrialFunction(self.state_space)
        state_test = TestFunction(self.state_space)
        state_mass = assemble(
            inner(state_test, state_trial) * self.model.spaces.dx,
            mat_type="aij",
        )
        auxiliary_trial = TrialFunction(self.auxiliary_space)
        auxiliary_test = TestFunction(self.auxiliary_space)
        auxiliary_mass = assemble(
            inner(auxiliary_test, auxiliary_trial) * self.model.spaces.dx,
            mat_type="aij",
        )
        self._state_mass_solver = LinearSolver(
            state_mass,
            solver_parameters=dict(self.timestepper.solver_parameters["erkstage-f"]),
        )
        self._auxiliary_mass_solver = LinearSolver(
            auxiliary_mass,
            solver_parameters=dict(
                self.timestepper.solver_parameters["erkstage-aux"]
            ),
        )

        self._c0_unit_direction = Function(
            self.coefficient_space, name="physical_c0_unit_direction"
        )
        self._c0_unit_direction.assign(0)
        self._c0_unit_direction.sub(1).assign(1.0)

    def _validate_timestepper(self):
        if self.timestepper.terms != ["hyperviscosity"]:
            raise ValueError(
                "cached HVP API requires terms=['hyperviscosity'] exactly"
            )
        if self.timestepper.nstages != 1:
            raise ValueError("cached HVP API is certified only for one-stage Euler")
        if not (
            self.timestepper.A.shape == (1, 1)
            and self.timestepper.A[0, 0] == 0.0
            and self.timestepper.b.shape == (1,)
            and self.timestepper.b[0] == 1.0
            and self.timestepper.c.shape == (1,)
            and self.timestepper.c[0] == 0.0
        ):
            raise ValueError("cached HVP API is certified only for explicit Euler")
        if self.model.get_coeff_list() != ["s", "c0"]:
            raise ValueError(
                "cached HVP API requires trainable coefficient order ['s', 'c0']"
            )
        matching_terms = [
            term
            for term in self.model.dynamics.forcing_terms
            if term.name == "hyperviscosity"
        ]
        if len(matching_terms) != 1 or not matching_terms[0].treat_as_coeffs:
            raise ValueError(
                "cached HVP API requires one trainable production hyperviscosity term"
            )
        if not self.model.has_aux():
            raise ValueError("cached HVP API requires hyperviscosity diagnostics")

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

    def _require_auxiliary_dual(self, name, value):
        if not isinstance(value, Cofunction):
            raise TypeError(f"{name} must be a Firedrake Cofunction")
        if value.function_space() != self.auxiliary_dual_space:
            raise ValueError(f"{name} belongs to the wrong auxiliary dual space")

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

    def _coefficient_value(self, name):
        data = self.timestepper.coeff_sub[name].dat.data_ro
        if data.size != 1:
            raise ValueError(f"physical coefficient {name} must be scalar")
        return float(data.reshape(-1)[0])

    def _populate_primal_coefficients(self, primal):
        self._state.assign(primal.state_in)
        self._auxiliary.assign(primal.diagnostic)
        self._coefficient.assign(0)
        self._coefficient_sub["s"].assign(primal.s)
        self._coefficient_sub["c0"].assign(primal.c0)
        self._time.assign(primal.t0)

    def _coefficient_direction(self, delta_c0):
        result = Function(
            self.coefficient_space, name="physical_coefficient_direction"
        )
        result.assign(0)
        result.sub(1).assign(delta_c0)
        return result

    @staticmethod
    def _zero_dual(dual_space, name):
        result = Cofunction(dual_space, name=name)
        result.zero()
        result.rename(name)
        return result

    @staticmethod
    def _arity_error(kind, form, expected_arguments):
        metadata = _form_metadata(form)
        return TypeError(
            f"nonzero {kind} form has invalid arity: "
            f"type={metadata['type']}; "
            f"arguments={metadata['argument_count']}; "
            f"argument_spaces={metadata['argument_spaces']}; "
            f"integrals={metadata['integral_count']}; "
            f"domains={metadata['domain_count']}; "
            f"expected_arguments={expected_arguments}"
        )

    def _assemble_dual(self, form, dual_space, name):
        normalized = _normalize_derivative_form(form)
        if normalized.structural_zero:
            return self._zero_dual(dual_space, name)
        arguments = _form_items(normalized.normalized, "arguments")
        if len(arguments) != 1:
            raise self._arity_error(
                "dual", normalized.normalized, expected_arguments=1
            )
        result = assemble(normalized.normalized)
        if not isinstance(result, Cofunction):
            raise TypeError("expected one-form assembly to return a Cofunction")
        if result.function_space() != dual_space:
            raise ValueError("assembled one-form belongs to the wrong dual space")
        result.rename(name)
        return result

    @staticmethod
    def _assemble_scalar(form):
        normalized = _normalize_derivative_form(form)
        if normalized.structural_zero:
            return 0.0
        arguments = _form_items(normalized.normalized, "arguments")
        if len(arguments) != 0:
            raise ProductionHyperviscosityEulerHVP._arity_error(
                "scalar", normalized.normalized, expected_arguments=0
            )
        return float(assemble(normalized.normalized))

    @staticmethod
    def _dual_sum(dual_space, terms, name):
        result = Cofunction(dual_space, name=name)
        result.zero()
        for scale, value in terms:
            with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
                result_vec.axpy(float(scale), value_vec)
        return result

    @staticmethod
    def _primal_axpy(base, scale, increment, name):
        result = _copy_function(base, name)
        with result.dat.vec as result_vec, increment.dat.vec_ro as increment_vec:
            result_vec.axpy(float(scale), increment_vec)
        return result

    def _solve_state_mass(self, dual, name):
        self._require_state_dual("dual", dual)
        rhs = _copy_cofunction(dual, f"{name}_rhs")
        result = Function(self.state_space, name=name)
        self._state_mass_solver.solve(result, rhs)
        return result

    def _solve_auxiliary_mass(self, dual, name):
        self._require_auxiliary_dual("dual", dual)
        rhs = _copy_cofunction(dual, f"{name}_rhs")
        result = Function(self.auxiliary_space, name=name)
        self._auxiliary_mass_solver.solve(result, rhs)
        return result

    def state_mass_map(self, value, name="state_mass_map"):
        self._require_state("value", value)
        return self._assemble_dual(
            inner(self._state_test, value) * self.model.spaces.dx,
            self.state_dual_space,
            name,
        )

    def state_riesz_representative(self, dual, name="state_riesz_representative"):
        return self._solve_state_mass(dual, name)

    def dual_pairing(self, dual, primal):
        self._require_state_dual("dual", dual)
        self._require_state("primal", primal)
        return float(assemble(action(dual, primal)))

    def take_forward_step_cached(self, xn, tn, dt):
        state_in = self._state_from_container("xn", xn)
        t0 = _as_float("tn", tn)
        step_size = _as_float("dt", dt)
        c0 = self._coefficient_value("c0")
        s = self._coefficient_value("s")

        state_out_container, state_out_sub, _ = self.model.get_full_var(
            "hyperviscosity_cached_state_out", split_x_and_aux=True
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
        return HyperviscosityPrimalCache(
            t0=t0,
            dt=step_size,
            c0=c0,
            s=s,
            state_in=_copy_function(state_in, "hyperviscosity_state_in_cache"),
            diagnostic=_copy_function(
                self.timestepper.Fi[0][0][1],
                "hyperviscosity_diagnostic_cache",
            ),
            tendency=_copy_function(
                self.timestepper.Fi[0][0][0],
                "hyperviscosity_tendency_cache",
            ),
            state_out=_copy_function(
                state_out_container[0], "hyperviscosity_state_out_cache"
            ),
        )

    def take_tangent_step(self, primal, delta_xn, delta_c0):
        if not isinstance(primal, HyperviscosityPrimalCache):
            raise TypeError("primal must be a HyperviscosityPrimalCache")
        state_direction = self._state_from_container("delta_xn", delta_xn)
        parameter_direction = _as_float("delta_c0", delta_c0)
        self._populate_primal_coefficients(primal)
        coefficient_direction = self._coefficient_direction(parameter_direction)

        diagnostic_tangent_rhs_form = derivative(
            self._auxiliary_rhs, self._state, state_direction
        )
        diagnostic_tangent_rhs = self._assemble_dual(
            diagnostic_tangent_rhs_form,
            self.auxiliary_dual_space,
            "hyperviscosity_diagnostic_tangent_rhs",
        )
        diagnostic_direction = self._solve_auxiliary_mass(
            diagnostic_tangent_rhs,
            "hyperviscosity_diagnostic_direction",
        )

        tendency_tangent_rhs_form = (
            derivative(self._main_rhs, self._state, state_direction)
            + derivative(self._main_rhs, self._auxiliary, diagnostic_direction)
            + derivative(
                self._main_rhs, self._coefficient, coefficient_direction
            )
        )
        tendency_tangent_rhs = self._assemble_dual(
            tendency_tangent_rhs_form,
            self.state_dual_space,
            "hyperviscosity_tendency_tangent_rhs",
        )
        tendency_direction = self._solve_state_mass(
            tendency_tangent_rhs,
            "hyperviscosity_tendency_direction",
        )
        state_direction_out = self._primal_axpy(
            state_direction,
            primal.dt,
            tendency_direction,
            "hyperviscosity_state_direction_out",
        )
        return HyperviscosityTangentCache(
            primal=primal,
            delta_c0=parameter_direction,
            state_direction_in=_copy_function(
                state_direction, "hyperviscosity_state_direction_in_cache"
            ),
            diagnostic_direction=_copy_function(
                diagnostic_direction,
                "hyperviscosity_diagnostic_direction_cache",
            ),
            tendency_direction=_copy_function(
                tendency_direction,
                "hyperviscosity_tendency_direction_cache",
            ),
            state_direction_out=_copy_function(
                state_direction_out,
                "hyperviscosity_state_direction_out_cache",
            ),
        )

    def take_adjoint_step_cached(self, primal, lambda_plus_star):
        if not isinstance(primal, HyperviscosityPrimalCache):
            raise TypeError("primal must be a HyperviscosityPrimalCache")
        self._require_state_dual("lambda_plus_star", lambda_plus_star)
        self._populate_primal_coefficients(primal)

        tendency_adjoint = self._dual_sum(
            self.state_dual_space,
            [(primal.dt, lambda_plus_star)],
            "hyperviscosity_tendency_adjoint",
        )
        main_reverse_auxiliary = self._solve_state_mass(
            tendency_adjoint,
            "hyperviscosity_main_reverse_auxiliary",
        )
        main_contracted = action(self._main_rhs, main_reverse_auxiliary)

        main_state_dependency_absent = all(
            coefficient is not self._state
            for coefficient in _form_items(main_contracted, "coefficients")
        )
        if not main_state_dependency_absent:
            raise ValueError(
                "production main hyperviscosity contraction unexpectedly "
                "depends directly on the incoming state"
            )
        raw_main_state_derivative = derivative(main_contracted, self._state)
        normalized_main_state = _normalize_derivative_form(
            raw_main_state_derivative
        )
        self._last_main_state_zero_debug = _ZeroDerivativeDebug(
            contracted=main_contracted,
            raw_derivative=raw_main_state_derivative,
            expanded_derivative=normalized_main_state.expanded,
            normalized_derivative=normalized_main_state.normalized,
            dependency_absent=main_state_dependency_absent,
            structural_zero=normalized_main_state.structural_zero,
            assembly_bypassed=True,
            raw_metadata=_form_metadata(raw_main_state_derivative),
            expanded_metadata=_form_metadata(normalized_main_state.expanded),
            normalized_metadata=_form_metadata(normalized_main_state.normalized),
            assembled_scalar_value=None,
        )
        if not normalized_main_state.structural_zero:
            raise TypeError(
                "expanded direct main-state derivative is not structural zero: "
                f"{_form_metadata(normalized_main_state.normalized)}"
            )
        main_state_adjoint = self._zero_dual(
            self.state_dual_space,
            "hyperviscosity_main_state_adjoint",
        )
        diagnostic_adjoint = self._assemble_dual(
            derivative(main_contracted, self._auxiliary),
            self.auxiliary_dual_space,
            "hyperviscosity_diagnostic_adjoint",
        )
        c0_gradient = self._assemble_scalar(
            derivative(
                main_contracted,
                self._coefficient,
                self._c0_unit_direction,
            )
        )

        diagnostic_reverse_auxiliary = self._solve_auxiliary_mass(
            diagnostic_adjoint,
            "hyperviscosity_diagnostic_reverse_auxiliary",
        )
        diagnostic_contracted = action(
            self._auxiliary_rhs, diagnostic_reverse_auxiliary
        )
        diagnostic_state_adjoint = self._assemble_dual(
            derivative(diagnostic_contracted, self._state),
            self.state_dual_space,
            "hyperviscosity_diagnostic_state_adjoint",
        )
        state_adjoint_in = self._dual_sum(
            self.state_dual_space,
            [
                (1.0, lambda_plus_star),
                (1.0, main_state_adjoint),
                (1.0, diagnostic_state_adjoint),
            ],
            "hyperviscosity_state_adjoint_in",
        )
        return HyperviscosityReverseResult(
            state_adjoint_in=state_adjoint_in,
            c0_gradient=c0_gradient,
            tendency_adjoint=_copy_cofunction(
                tendency_adjoint, "hyperviscosity_tendency_adjoint_result"
            ),
            main_reverse_auxiliary=_copy_function(
                main_reverse_auxiliary,
                "hyperviscosity_main_reverse_auxiliary_result",
            ),
            diagnostic_adjoint=_copy_cofunction(
                diagnostic_adjoint, "hyperviscosity_diagnostic_adjoint_result"
            ),
            diagnostic_reverse_auxiliary=_copy_function(
                diagnostic_reverse_auxiliary,
                "hyperviscosity_diagnostic_reverse_auxiliary_result",
            ),
            main_state_adjoint=_copy_cofunction(
                main_state_adjoint, "hyperviscosity_main_state_adjoint_result"
            ),
            diagnostic_state_adjoint=_copy_cofunction(
                diagnostic_state_adjoint,
                "hyperviscosity_diagnostic_state_adjoint_result",
            ),
        )

    def take_incremental_adjoint_step(
        self, tangent, lambda_plus_star, mu_plus_star
    ):
        if not isinstance(tangent, HyperviscosityTangentCache):
            raise TypeError("tangent must be a HyperviscosityTangentCache")
        self._require_state_dual("lambda_plus_star", lambda_plus_star)
        self._require_state_dual("mu_plus_star", mu_plus_star)
        primal = tangent.primal
        self._populate_primal_coefficients(primal)
        coefficient_direction = self._coefficient_direction(tangent.delta_c0)

        ordinary = self.take_adjoint_step_cached(primal, lambda_plus_star)
        # Restore form coefficients after the independent ordinary reverse call.
        self._populate_primal_coefficients(primal)

        incremental_tendency_adjoint = self._dual_sum(
            self.state_dual_space,
            [(primal.dt, mu_plus_star)],
            "hyperviscosity_incremental_tendency_adjoint",
        )
        incremental_main_reverse_auxiliary = self._solve_state_mass(
            incremental_tendency_adjoint,
            "hyperviscosity_incremental_main_reverse_auxiliary",
        )

        ordinary_main_contracted = action(
            self._main_rhs, ordinary.main_reverse_auxiliary
        )
        incremental_main_contracted = action(
            self._main_rhs, incremental_main_reverse_auxiliary
        )

        ordinary_diagnostic_pullback = derivative(
            ordinary_main_contracted, self._auxiliary
        )
        incremental_diagnostic_pullback_form = (
            derivative(incremental_main_contracted, self._auxiliary)
            + derivative(
                ordinary_diagnostic_pullback,
                self._state,
                tangent.state_direction_in,
            )
            + derivative(
                ordinary_diagnostic_pullback,
                self._auxiliary,
                tangent.diagnostic_direction,
            )
            + derivative(
                ordinary_diagnostic_pullback,
                self._coefficient,
                coefficient_direction,
            )
        )
        incremental_diagnostic_adjoint = self._assemble_dual(
            incremental_diagnostic_pullback_form,
            self.auxiliary_dual_space,
            "hyperviscosity_incremental_diagnostic_adjoint",
        )

        ordinary_main_state_pullback = derivative(
            ordinary_main_contracted, self._state
        )
        incremental_main_state_form = (
            derivative(incremental_main_contracted, self._state)
            + derivative(
                ordinary_main_state_pullback,
                self._state,
                tangent.state_direction_in,
            )
            + derivative(
                ordinary_main_state_pullback,
                self._auxiliary,
                tangent.diagnostic_direction,
            )
            + derivative(
                ordinary_main_state_pullback,
                self._coefficient,
                coefficient_direction,
            )
        )
        incremental_main_state_adjoint = self._assemble_dual(
            incremental_main_state_form,
            self.state_dual_space,
            "hyperviscosity_incremental_main_state_adjoint",
        )

        raw_ordinary_c0_pullback = derivative(
            ordinary_main_contracted,
            self._coefficient,
            self._c0_unit_direction,
        )
        normalized_ordinary_c0 = _normalize_derivative_form(
            raw_ordinary_c0_pullback
        )
        if normalized_ordinary_c0.structural_zero:
            raise TypeError(
                "ordinary physical-c0 pullback unexpectedly normalized to zero"
            )
        ordinary_c0_pullback = normalized_ordinary_c0.normalized
        hvp_from_incremental_adjoint = self._assemble_scalar(
            derivative(
                incremental_main_contracted,
                self._coefficient,
                self._c0_unit_direction,
            )
        )
        hvp_from_state_direction = self._assemble_scalar(
            derivative(
                ordinary_c0_pullback,
                self._state,
                tangent.state_direction_in,
            )
            + derivative(
                ordinary_c0_pullback,
                self._auxiliary,
                tangent.diagnostic_direction,
            )
        )
        raw_pure_control_derivative = derivative(
            ordinary_c0_pullback,
            self._coefficient,
            self._c0_unit_direction,
        )
        normalized_pure_control = _normalize_derivative_form(
            raw_pure_control_derivative
        )
        pure_control_metadata = _form_metadata(
            normalized_pure_control.normalized
        )
        if normalized_pure_control.structural_zero:
            hvp_from_pure_control = 0.0
            pure_control_assembly_bypassed = True
        else:
            if (
                pure_control_metadata["argument_count"] != 0
                or pure_control_metadata["domain_count"] < 1
            ):
                raise TypeError(
                    "nonzero pure-c0 derivative is not a valid scalar form: "
                    f"{pure_control_metadata}"
                )
            hvp_from_pure_control = self._assemble_scalar(
                raw_pure_control_derivative
            )
            pure_control_assembly_bypassed = False
        self._last_pure_control_zero_debug = _ZeroDerivativeDebug(
            contracted=ordinary_c0_pullback,
            raw_derivative=raw_pure_control_derivative,
            expanded_derivative=normalized_pure_control.expanded,
            normalized_derivative=normalized_pure_control.normalized,
            dependency_absent=None,
            structural_zero=normalized_pure_control.structural_zero,
            assembly_bypassed=pure_control_assembly_bypassed,
            raw_metadata=_form_metadata(raw_pure_control_derivative),
            expanded_metadata=_form_metadata(normalized_pure_control.expanded),
            normalized_metadata=pure_control_metadata,
            assembled_scalar_value=hvp_from_pure_control,
        )
        c0_hvp = (
            hvp_from_incremental_adjoint
            + hvp_from_state_direction
            + hvp_from_pure_control
        )

        incremental_diagnostic_reverse_auxiliary = self._solve_auxiliary_mass(
            incremental_diagnostic_adjoint,
            "hyperviscosity_incremental_diagnostic_reverse_auxiliary",
        )
        ordinary_diagnostic_contracted = action(
            self._auxiliary_rhs, ordinary.diagnostic_reverse_auxiliary
        )
        incremental_diagnostic_contracted = action(
            self._auxiliary_rhs,
            incremental_diagnostic_reverse_auxiliary,
        )
        ordinary_diagnostic_state_pullback = derivative(
            ordinary_diagnostic_contracted, self._state
        )
        incremental_diagnostic_state_form = (
            derivative(incremental_diagnostic_contracted, self._state)
            + derivative(
                ordinary_diagnostic_state_pullback,
                self._state,
                tangent.state_direction_in,
            )
            + derivative(
                ordinary_diagnostic_state_pullback,
                self._auxiliary,
                tangent.diagnostic_direction,
            )
            + derivative(
                ordinary_diagnostic_state_pullback,
                self._coefficient,
                coefficient_direction,
            )
        )
        incremental_diagnostic_state_adjoint = self._assemble_dual(
            incremental_diagnostic_state_form,
            self.state_dual_space,
            "hyperviscosity_incremental_diagnostic_state_adjoint",
        )
        incremental_state_adjoint_in = self._dual_sum(
            self.state_dual_space,
            [
                (1.0, mu_plus_star),
                (1.0, incremental_main_state_adjoint),
                (1.0, incremental_diagnostic_state_adjoint),
            ],
            "hyperviscosity_incremental_state_adjoint_in",
        )

        return HyperviscosityHVPResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=incremental_state_adjoint_in,
            c0_hvp=c0_hvp,
            c0_hvp_from_incremental_adjoint=hvp_from_incremental_adjoint,
            c0_hvp_from_state_direction=hvp_from_state_direction,
            c0_hvp_from_pure_control_curvature=hvp_from_pure_control,
            incremental_tendency_adjoint=_copy_cofunction(
                incremental_tendency_adjoint,
                "hyperviscosity_incremental_tendency_adjoint_result",
            ),
            incremental_main_reverse_auxiliary=_copy_function(
                incremental_main_reverse_auxiliary,
                "hyperviscosity_incremental_main_reverse_auxiliary_result",
            ),
            incremental_diagnostic_adjoint=_copy_cofunction(
                incremental_diagnostic_adjoint,
                "hyperviscosity_incremental_diagnostic_adjoint_result",
            ),
            incremental_diagnostic_reverse_auxiliary=_copy_function(
                incremental_diagnostic_reverse_auxiliary,
                "hyperviscosity_incremental_diagnostic_reverse_auxiliary_result",
            ),
            incremental_main_state_adjoint=_copy_cofunction(
                incremental_main_state_adjoint,
                "hyperviscosity_incremental_main_state_adjoint_result",
            ),
            incremental_diagnostic_state_adjoint=_copy_cofunction(
                incremental_diagnostic_state_adjoint,
                "hyperviscosity_incremental_diagnostic_state_adjoint_result",
            ),
        )
