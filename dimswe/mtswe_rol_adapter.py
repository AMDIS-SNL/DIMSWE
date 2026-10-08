"""Normalized-scalar PyROL access to the production MTSWE reduced HVP.

This module is intentionally not imported by :mod:`dimswe`.  PyROL is an
optional dependency, and importing the base package must not require it.

Only the normalized scalar hyperviscosity control is exposed here.  The
production derivative helper continues to receive physical ``c0`` values and
physical ``delta_c0`` directions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from pyrol import Objective, Vector
from pyrol.vectors import NumPyVector


MTSWE_ACTIVE_SET_SWITCHES = (
    "condensation",
    "evaporation",
    "evaporation_cap",
    "rain",
    "depth_denominator",
)

_MARGIN_ATTRIBUTES = {
    "condensation": "condensation_margin",
    "evaporation": "evaporation_margin",
    "evaporation_cap": "evaporation_cap_margin",
    "rain": "rain_margin",
    "depth_denominator": "depth_denominator_margin",
}
_SIGNATURE_INDEX = {
    "condensation": 0,
    "evaporation": 1,
    "evaporation_cap": 2,
    "rain": 3,
}


@dataclass(frozen=True)
class MTSWEActiveSetEntry:
    """Qualification data for one switch at one complete timestep."""

    timestep: int
    switch: str
    minimum_margin: float
    configured_threshold: float
    qualified: bool
    signature: tuple[bool, ...] | None


@dataclass(frozen=True)
class MTSWEActiveSetReport:
    """Owned base-trajectory active-set report for one adapter call."""

    purpose: str
    entries: tuple[MTSWEActiveSetEntry, ...]

    @property
    def qualified(self):
        return all(entry.qualified for entry in self.entries)

    @property
    def failures(self):
        return tuple(entry for entry in self.entries if not entry.qualified)

    @property
    def signatures(self):
        timesteps = sorted({entry.timestep for entry in self.entries})
        return tuple(
            tuple(
                entry.signature
                for entry in self.entries
                if entry.timestep == timestep
                and entry.switch in _SIGNATURE_INDEX
            )
            for timestep in timesteps
        )


class MTSWEActiveSetQualificationError(RuntimeError):
    """Base exception for a nonqualified production MTSWE derivative."""

    def __init__(self, report):
        self.report = report
        detail = "; ".join(
            "timestep={0.timestep}, switch={0.switch}, "
            "minimum_margin={0.minimum_margin:.17g}, "
            "threshold={0.configured_threshold:.17g}".format(entry)
            for entry in report.failures
        )
        super().__init__(
            f"MTSWE {report.purpose} active-set qualification failed: {detail}"
        )


class MTSWEGradientActiveSetQualificationError(
    MTSWEActiveSetQualificationError
):
    """The branchwise gradient is ambiguous at a configured machine zero."""


class MTSWEHVPActiveSetQualificationError(MTSWEActiveSetQualificationError):
    """The fixed-active-set MTSWE Hessian action is not locally qualified."""


def _owned_copy(value, name):
    """Return an owned deep copy of a Firedrake-like function object."""
    try:
        result = value.copy(deepcopy=True)
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{name} must support copy(deepcopy=True)") from exc
    rename = getattr(result, "rename", None)
    if callable(rename):
        rename(name)
    return result


def _as_finite_float(name, value, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _normalize_tolerances(name, values):
    if values is None:
        values = {}
    if not hasattr(values, "items"):
        raise TypeError(f"{name} must be a mapping or None")
    unknown = set(values) - set(MTSWE_ACTIVE_SET_SWITCHES)
    if unknown:
        raise ValueError(f"{name} contains unknown switches: {sorted(unknown)}")
    result = {}
    for switch in MTSWE_ACTIVE_SET_SWITCHES:
        threshold = _as_finite_float(
            f"{name}[{switch!r}]", values.get(switch, 0.0)
        )
        if threshold < 0.0:
            raise ValueError(f"{name}[{switch!r}] must be nonnegative")
        result[switch] = threshold
    return result


def _selected_moist_active_set(parent_cache, timestep):
    """Use production GLL switches for JAX and legacy switches for UFL."""
    matches = tuple(
        child for child in parent_cache.children if child.name == "moist_euler"
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"MTSWE timestep {timestep} does not contain one moist_euler cache"
        )
    cache = matches[0].cache
    if hasattr(cache, "gll_active_set"):
        return cache.gll_active_set
    if hasattr(cache, "active_set"):
        return cache.active_set
    raise RuntimeError(
        f"MTSWE timestep {timestep} moist cache has no active-set diagnostic"
    )


def _active_set_margin(diagnostic, switch):
    attribute = _MARGIN_ATTRIBUTES[switch]
    if hasattr(diagnostic, attribute):
        return float(getattr(diagnostic, attribute))
    margins = getattr(diagnostic, "margins", None)
    if margins is not None and attribute in margins:
        return float(margins[attribute])
    raise RuntimeError(f"moist active-set diagnostic has no {attribute}")


class ProductionMTSWEScalarC0Objective(Objective):
    """PyROL objective for normalized scalar ``c0`` on the certified MTSWE map.

    The object exclusively uses ``timestepper`` and its cached-HVP helper.
    Concurrent use of the same timestepper by another simulation or objective
    is unsupported.  Caller-owned state, target, coefficient, and PyROL vector
    data are copied and never used as production scratch.
    """

    def __init__(
        self,
        timestepper,
        coefficient_template,
        fixed_initial_state,
        target,
        *,
        nsteps,
        t0,
        dt,
        c0_scale,
        gradient_zero_margin_tolerances=None,
        hvp_active_set_tolerances=None,
    ):
        super().__init__()
        if not isinstance(nsteps, Integral) or isinstance(nsteps, bool):
            raise TypeError("nsteps must be the integer 1 or 3")
        if int(nsteps) not in (1, 3):
            raise ValueError("certified MTSWE PyROL nsteps are exactly 1 or 3")
        self.nsteps = int(nsteps)
        self.t0 = _as_finite_float("t0", t0)
        self.dt = _as_finite_float("dt", dt, positive=True)
        self.c0_scale = _as_finite_float(
            "c0_scale", c0_scale, positive=True
        )
        self.gradient_zero_margin_tolerances = _normalize_tolerances(
            "gradient_zero_margin_tolerances",
            gradient_zero_margin_tolerances,
        )
        self.hvp_active_set_tolerances = _normalize_tolerances(
            "hvp_active_set_tolerances", hvp_active_set_tolerances
        )

        self.timestepper = timestepper
        try:
            self.helper = timestepper._get_mtswe_split_hvp_helper()
        except AttributeError as exc:
            raise TypeError(
                "timestepper does not expose the production MTSWE HVP helper"
            ) from exc

        names = list(self.helper.model.get_coeff_list())
        if names != ["s", "c0"]:
            raise ValueError(
                "production MTSWE coefficient order must be exactly ['s', 'c0']"
            )
        self._c0_index = 1
        self.helper._require_state("fixed_initial_state", fixed_initial_state)
        self.helper._require_state("target", target)
        self._fixed_initial_state = _owned_copy(
            fixed_initial_state, "mtswe_rol_fixed_initial_state"
        )
        self._target = _owned_copy(target, "mtswe_rol_target")
        self._coefficient_template = _owned_copy(
            coefficient_template, "mtswe_rol_coefficient_template"
        )
        self._validate_coefficient_spaces()

        self._point_z = None
        self._point_result = None
        self._hvp_z = None
        self._hvp_qz = None
        self._hvp_result = None
        self.production_gradient_evaluations = 0
        self.production_hvp_evaluations = 0
        self.last_active_set_report = None
        self.last_value_active_set_report = None
        self.last_gradient_active_set_report = None
        self.last_hvp_active_set_report = None

    def _validate_coefficient_spaces(self):
        children = tuple(getattr(self.timestepper, "time_integrators", ()))
        if not children:
            raise ValueError("production MTSWE timestepper has no child integrators")
        template_space = self._coefficient_template.function_space()
        for index, child in enumerate(children):
            coefficient = getattr(child, "coeff", None)
            if coefficient is None:
                raise ValueError(f"MTSWE child {index} has no coefficient storage")
            if coefficient.function_space() != template_space:
                raise ValueError(
                    f"coefficient_template belongs to the wrong space for child {index}"
                )

    @property
    def fixed_initial_state(self):
        return _owned_copy(
            self._fixed_initial_state, "mtswe_rol_fixed_initial_state_snapshot"
        )

    @property
    def target(self):
        return _owned_copy(self._target, "mtswe_rol_target_snapshot")

    @property
    def coefficient_template(self):
        return _owned_copy(
            self._coefficient_template, "mtswe_rol_coefficient_snapshot"
        )

    @property
    def cache_info(self):
        return {
            "point_z": self._point_z,
            "has_point_result": self._point_result is not None,
            "hvp_z": self._hvp_z,
            "hvp_qz": self._hvp_qz,
            "has_hvp_result": self._hvp_result is not None,
        }

    @staticmethod
    def _require_vector(name, value, *, finite_input=False):
        if not isinstance(value, NumPyVector) or value.dimension() != 1:
            raise TypeError(f"{name} must be a one-element pyrol NumPyVector")
        if value.array.shape != (1,):
            raise ValueError(f"{name} must store exactly one scalar entry")
        scalar = float(value.array[0])
        if finite_input and not np.isfinite(scalar):
            raise ValueError(f"{name} entry must be finite")
        return scalar

    @staticmethod
    def _require_independent_output(output_name, output, *inputs):
        for input_name, input_vector in inputs:
            if output is input_vector or np.shares_memory(
                output.array, input_vector.array
            ):
                raise ValueError(
                    f"{output_name} output must not alias input {input_name}"
                )

    def _clear_caches(self):
        self._point_z = None
        self._point_result = None
        self._hvp_z = None
        self._hvp_qz = None
        self._hvp_result = None

    def update(self, x, *args):
        self._require_vector("x", x, finite_input=True)
        self._clear_caches()

    def _prepare_point(self, z):
        if self._point_z != z:
            self._point_z = None
            self._point_result = None
        if self._hvp_z != z:
            self._hvp_z = None
            self._hvp_qz = None
            self._hvp_result = None

    @contextmanager
    def _physical_coefficient(self, physical_c0):
        children = tuple(self.timestepper.time_integrators)
        snapshots = tuple(
            _owned_copy(child.coeff, f"mtswe_rol_child_{index}_coefficient")
            for index, child in enumerate(children)
        )
        work = _owned_copy(
            self._coefficient_template, "mtswe_rol_working_coefficient"
        )
        work.sub(self._c0_index).assign(float(physical_c0))
        try:
            self.timestepper.reset_internal_vars()
            self.timestepper.set_coeff(work)
            yield
        finally:
            try:
                for child, snapshot in zip(children, snapshots):
                    child.set_coeff(snapshot)
            finally:
                self.timestepper.reset_internal_vars()

    def _production_gradient(self, z):
        physical_c0 = _as_finite_float("physical c0", self.c0_scale * z)
        state = _owned_copy(
            self._fixed_initial_state, "mtswe_rol_gradient_initial_state"
        )
        target = _owned_copy(self._target, "mtswe_rol_gradient_target")
        with self._physical_coefficient(physical_c0):
            result = self.timestepper.mtswe_terminal_least_squares_gradient(
                self.nsteps, state, self.t0, self.dt, target
            )
        self.production_gradient_evaluations += 1
        return result

    def _production_hvp(self, z, qz):
        physical_c0 = _as_finite_float("physical c0", self.c0_scale * z)
        delta_c0 = _as_finite_float(
            "physical delta_c0", self.c0_scale * qz
        )
        state = _owned_copy(
            self._fixed_initial_state, "mtswe_rol_hvp_initial_state"
        )
        target = _owned_copy(self._target, "mtswe_rol_hvp_target")
        zero_direction = _owned_copy(
            self._fixed_initial_state, "mtswe_rol_zero_state_direction"
        )
        zero_direction.assign(0)
        with self._physical_coefficient(physical_c0):
            result = self.timestepper.mtswe_terminal_least_squares_hvp(
                self.nsteps,
                state,
                self.t0,
                self.dt,
                target,
                zero_direction,
                delta_c0,
            )
        self.production_hvp_evaluations += 1
        return result

    def _point_result_at(self, z):
        self._prepare_point(z)
        if self._point_result is None:
            self._point_result = self._production_gradient(z)
            self._point_z = z
        return self._point_result

    def _hvp_result_at(self, z, qz):
        self._prepare_point(z)
        if self._hvp_qz != qz:
            self._hvp_z = None
            self._hvp_qz = None
            self._hvp_result = None
        if self._hvp_result is None:
            result = self._production_hvp(z, qz)
            self._hvp_z = z
            self._hvp_qz = qz
            self._hvp_result = result
            # The accepted HVP result contains the same objective and ordinary
            # reduced gradient, so it is also the bounded current-point entry.
            self._point_z = z
            self._point_result = result
        return self._hvp_result

    @staticmethod
    def _moist_active_set(parent_cache, timestep):
        return _selected_moist_active_set(parent_cache, timestep)

    def _active_set_report(self, result, purpose, tolerances):
        entries = []
        for timestep, parent_cache in enumerate(result.primal_caches):
            diagnostic = self._moist_active_set(parent_cache, timestep)
            signature = tuple(tuple(branch) for branch in diagnostic.signature)
            if len(signature) != 4:
                raise RuntimeError(
                    f"MTSWE timestep {timestep} has an invalid active signature"
                )
            for switch in MTSWE_ACTIVE_SET_SWITCHES:
                margin = _active_set_margin(diagnostic, switch)
                threshold = float(tolerances[switch])
                branch_signature = (
                    signature[_SIGNATURE_INDEX[switch]]
                    if switch in _SIGNATURE_INDEX
                    else None
                )
                entries.append(
                    MTSWEActiveSetEntry(
                        timestep=timestep,
                        switch=switch,
                        minimum_margin=margin,
                        configured_threshold=threshold,
                        qualified=np.isfinite(margin) and margin > threshold,
                        signature=branch_signature,
                    )
                )
        if len(entries) != self.nsteps * len(MTSWE_ACTIVE_SET_SWITCHES):
            raise RuntimeError("MTSWE active-set report has the wrong timestep count")
        return MTSWEActiveSetReport(purpose=purpose, entries=tuple(entries))

    def _record_report(self, kind, report):
        self.last_active_set_report = report
        if kind == "value":
            self.last_value_active_set_report = report
        elif kind == "gradient":
            self.last_gradient_active_set_report = report
        else:
            self.last_hvp_active_set_report = report

    def value(self, x, tol):
        z = self._require_vector("x", x, finite_input=True)
        result = self._point_result_at(z)
        report = self._active_set_report(
            result,
            "value",
            {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES},
        )
        self._record_report("value", report)
        value = float(result.objective_value)
        if not np.isfinite(value):
            raise FloatingPointError("production MTSWE objective is not finite")
        return value

    def gradient(self, g, x, tol):
        self._require_vector("g", g)
        z = self._require_vector("x", x, finite_input=True)
        self._require_independent_output("gradient", g, ("x", x))
        result = self._point_result_at(z)
        report = self._active_set_report(
            result,
            "gradient",
            self.gradient_zero_margin_tolerances,
        )
        self._record_report("gradient", report)
        if not report.qualified:
            raise MTSWEGradientActiveSetQualificationError(report)
        normalized_gradient = self.c0_scale * float(
            result.physical_c0_gradient
        )
        if not np.isfinite(normalized_gradient):
            raise FloatingPointError("production MTSWE c0 gradient is not finite")
        g.array[0] = normalized_gradient

    def hessVec(self, hv, v, x, tol):
        self._require_vector("hv", hv)
        qz = self._require_vector("v", v, finite_input=True)
        z = self._require_vector("x", x, finite_input=True)
        self._require_independent_output(
            "Hessian", hv, ("v", v), ("x", x)
        )
        result = self._hvp_result_at(z, qz)
        report = self._active_set_report(
            result, "hessVec", self.hvp_active_set_tolerances
        )
        self._record_report("hvp", report)
        if not report.qualified:
            raise MTSWEHVPActiveSetQualificationError(report)
        normalized_hvp = self.c0_scale * float(result.physical_c0_hvp)
        if not np.isfinite(normalized_hvp):
            raise FloatingPointError("production MTSWE c0 HVP is not finite")
        hv.array[0] = normalized_hvp


def _functions_equal(left, right):
    """Return exact equality without exporting a coefficient array."""
    if left.function_space() != right.function_space():
        return False
    with left.dat.vec_ro as left_vec, right.dat.vec_ro as right_vec:
        return bool(left_vec.equal(right_vec))


class MTSWEStateVector(Vector):
    """Owned PyROL primal vector in the certified mixed MTSWE L2 metric."""

    def __init__(self, function, helper):
        helper._require_state("MTSWEStateVector function", function)
        self.helper = helper
        self.function = _owned_copy(function, "mtswe_rol_state_vector")
        self._space = self.function.function_space()
        super().__init__()

    def _require_compatible(self, name, other):
        if not isinstance(other, MTSWEStateVector):
            raise TypeError(f"{name} must be an MTSWEStateVector")
        if other.helper is not self.helper:
            raise ValueError(f"{name} uses a different certified MTSWE metric")
        if other.function.function_space() != self._space:
            raise ValueError(f"{name} belongs to an incompatible mixed space")
        return other

    def clone(self):
        result = type(self)(self.function, self.helper)
        result.zero()
        return result

    def set(self, other):
        other = self._require_compatible("set input", other)
        self.function.assign(other.function)

    def plus(self, other):
        other = self._require_compatible("plus input", other)
        with (
            self.function.dat.vec as target,
            other.function.dat.vec_ro as increment,
        ):
            target.axpy(1.0, increment)

    def scale(self, scale_factor):
        factor = _as_finite_float("scale factor", scale_factor)
        with self.function.dat.vec as target:
            target.scale(factor)

    def dot(self, other):
        other = self._require_compatible("dot input", other)
        dual = self.helper.state_mass_map(
            self.function, "mtswe_rol_vector_mass_dual"
        )
        result = float(self.helper.dual_pairing(dual, other.function))
        if not np.isfinite(result):
            raise FloatingPointError("MTSWE state-vector dot product is not finite")
        return result

    def norm(self):
        squared = self.dot(self)
        if squared < 0.0:
            tolerance = 100.0 * np.finfo(float).eps * max(abs(squared), 1.0)
            if squared < -tolerance:
                raise FloatingPointError("MTSWE state-vector norm squared is negative")
        return float(np.sqrt(max(squared, 0.0)))

    def axpy(self, scale_factor, other):
        other = self._require_compatible("axpy input", other)
        factor = _as_finite_float("axpy scale factor", scale_factor)
        with (
            self.function.dat.vec as target,
            other.function.dat.vec_ro as increment,
        ):
            target.axpy(factor, increment)

    def zero(self):
        self.function.assign(0)

    def dual(self):
        return self

    def apply(self, other):
        return self.dot(other)

    def dimension(self):
        return int(self._space.dim())

    def same_values(self, other):
        other = self._require_compatible("comparison input", other)
        return _functions_equal(self.function, other.function)


class _ProductionMTSWEStateObjectiveBase(Objective):
    """Shared certified context for field-containing MTSWE objectives."""

    def __init__(
        self,
        timestepper,
        coefficient_template,
        target,
        *,
        nsteps,
        t0,
        dt,
        gradient_zero_margin_tolerances,
        hvp_active_set_tolerances,
    ):
        super().__init__()
        if not isinstance(nsteps, Integral) or isinstance(nsteps, bool):
            raise TypeError("nsteps must be the integer 1 or 3")
        if int(nsteps) not in (1, 3):
            raise ValueError("certified MTSWE PyROL nsteps are exactly 1 or 3")
        self.nsteps = int(nsteps)
        self.t0 = _as_finite_float("t0", t0)
        self.dt = _as_finite_float("dt", dt, positive=True)
        self.gradient_zero_margin_tolerances = _normalize_tolerances(
            "gradient_zero_margin_tolerances",
            gradient_zero_margin_tolerances,
        )
        self.hvp_active_set_tolerances = _normalize_tolerances(
            "hvp_active_set_tolerances", hvp_active_set_tolerances
        )
        self.timestepper = timestepper
        try:
            self.helper = timestepper._get_mtswe_split_hvp_helper()
        except AttributeError as exc:
            raise TypeError(
                "timestepper does not expose the production MTSWE HVP helper"
            ) from exc
        names = list(self.helper.model.get_coeff_list())
        if names != ["s", "c0"]:
            raise ValueError(
                "production MTSWE coefficient order must be exactly ['s', 'c0']"
            )
        self._c0_index = 1
        self.helper._require_state("target", target)
        self._target = _owned_copy(target, "mtswe_rol_state_objective_target")
        self._coefficient_template = _owned_copy(
            coefficient_template, "mtswe_rol_state_objective_coefficient"
        )
        self._validate_coefficient_spaces()
        self.production_gradient_evaluations = 0
        self.production_hvp_evaluations = 0
        self.last_active_set_report = None
        self.last_value_active_set_report = None
        self.last_gradient_active_set_report = None
        self.last_hvp_active_set_report = None

    def _validate_coefficient_spaces(self):
        children = tuple(getattr(self.timestepper, "time_integrators", ()))
        if not children:
            raise ValueError("production MTSWE timestepper has no child integrators")
        template_space = self._coefficient_template.function_space()
        for index, child in enumerate(children):
            coefficient = getattr(child, "coeff", None)
            if coefficient is None:
                raise ValueError(f"MTSWE child {index} has no coefficient storage")
            if coefficient.function_space() != template_space:
                raise ValueError(
                    "coefficient_template belongs to the wrong space for "
                    f"child {index}"
                )

    @property
    def target(self):
        return _owned_copy(self._target, "mtswe_rol_state_target_snapshot")

    @property
    def coefficient_template(self):
        return _owned_copy(
            self._coefficient_template,
            "mtswe_rol_state_coefficient_snapshot",
        )

    def _require_state_vector(self, name, value):
        if not isinstance(value, MTSWEStateVector):
            raise TypeError(f"{name} must be an MTSWEStateVector")
        if value.helper is not self.helper:
            raise ValueError(f"{name} uses a different certified MTSWE helper")
        self.helper._require_state(f"{name}.function", value.function)
        return value

    @staticmethod
    def _require_independent_state_output(output_name, output, *inputs):
        for input_name, input_vector in inputs:
            if (
                output is input_vector
                or output.function.dat is input_vector.function.dat
            ):
                raise ValueError(
                    f"{output_name} output must not alias input {input_name}"
                )

    @contextmanager
    def _physical_coefficient(self, physical_c0):
        physical_c0 = _as_finite_float("physical c0", physical_c0)
        children = tuple(self.timestepper.time_integrators)
        snapshots = tuple(
            _owned_copy(child.coeff, f"mtswe_rol_child_{index}_coefficient")
            for index, child in enumerate(children)
        )
        work = _owned_copy(
            self._coefficient_template, "mtswe_rol_state_working_coefficient"
        )
        work.sub(self._c0_index).assign(physical_c0)
        try:
            self.timestepper.reset_internal_vars()
            self.timestepper.set_coeff(work)
            yield
        finally:
            try:
                for child, snapshot in zip(children, snapshots):
                    child.set_coeff(snapshot)
            finally:
                self.timestepper.reset_internal_vars()

    @staticmethod
    def _moist_active_set(parent_cache, timestep):
        return _selected_moist_active_set(parent_cache, timestep)

    def _active_set_report(self, result, purpose, tolerances):
        entries = []
        for timestep, parent_cache in enumerate(result.primal_caches):
            diagnostic = self._moist_active_set(parent_cache, timestep)
            signature = tuple(tuple(branch) for branch in diagnostic.signature)
            if len(signature) != 4:
                raise RuntimeError(
                    f"MTSWE timestep {timestep} has an invalid active signature"
                )
            for switch in MTSWE_ACTIVE_SET_SWITCHES:
                margin = _active_set_margin(diagnostic, switch)
                threshold = float(tolerances[switch])
                branch_signature = (
                    signature[_SIGNATURE_INDEX[switch]]
                    if switch in _SIGNATURE_INDEX
                    else None
                )
                entries.append(
                    MTSWEActiveSetEntry(
                        timestep=timestep,
                        switch=switch,
                        minimum_margin=margin,
                        configured_threshold=threshold,
                        qualified=np.isfinite(margin) and margin > threshold,
                        signature=branch_signature,
                    )
                )
        if len(entries) != self.nsteps * len(MTSWE_ACTIVE_SET_SWITCHES):
            raise RuntimeError("MTSWE active-set report has the wrong timestep count")
        return MTSWEActiveSetReport(purpose=purpose, entries=tuple(entries))

    def _record_report(self, kind, report):
        self.last_active_set_report = report
        if kind == "value":
            self.last_value_active_set_report = report
        elif kind == "gradient":
            self.last_gradient_active_set_report = report
        else:
            self.last_hvp_active_set_report = report

    def _value_report(self, result):
        report = self._active_set_report(
            result,
            "value",
            {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES},
        )
        self._record_report("value", report)
        value = float(result.objective_value)
        if not np.isfinite(value):
            raise FloatingPointError("production MTSWE objective is not finite")
        return value

    def _qualify_gradient(self, result):
        report = self._active_set_report(
            result,
            "gradient",
            self.gradient_zero_margin_tolerances,
        )
        self._record_report("gradient", report)
        if not report.qualified:
            raise MTSWEGradientActiveSetQualificationError(report)

    def _qualify_hvp(self, result):
        report = self._active_set_report(
            result, "hessVec", self.hvp_active_set_tolerances
        )
        self._record_report("hvp", report)
        if not report.qualified:
            raise MTSWEHVPActiveSetQualificationError(report)

    def _copy_riesz_result(self, output, dual, name):
        representative = self.helper.state_riesz_representative(dual, name)
        output.function.assign(representative)


class ProductionMTSWEInitialConditionObjective(
    _ProductionMTSWEStateObjectiveBase
):
    """PyROL objective for the physical six-field MTSWE initial condition."""

    def __init__(
        self,
        timestepper,
        coefficient_template,
        target,
        *,
        fixed_c0_physical,
        nsteps,
        t0,
        dt,
        gradient_zero_margin_tolerances=None,
        hvp_active_set_tolerances=None,
    ):
        super().__init__(
            timestepper,
            coefficient_template,
            target,
            nsteps=nsteps,
            t0=t0,
            dt=dt,
            gradient_zero_margin_tolerances=gradient_zero_margin_tolerances,
            hvp_active_set_tolerances=hvp_active_set_tolerances,
        )
        self.fixed_c0_physical = _as_finite_float(
            "fixed_c0_physical", fixed_c0_physical
        )
        self._point_x = None
        self._point_result = None
        self._hvp_x = None
        self._hvp_v = None
        self._hvp_result = None

    @property
    def cache_info(self):
        return {
            "has_point": self._point_x is not None,
            "has_point_result": self._point_result is not None,
            "has_hvp_point": self._hvp_x is not None,
            "has_hvp_direction": self._hvp_v is not None,
            "has_hvp_result": self._hvp_result is not None,
        }

    def _clear_caches(self):
        self._point_x = None
        self._point_result = None
        self._hvp_x = None
        self._hvp_v = None
        self._hvp_result = None

    def update(self, x, *args):
        self._require_state_vector("x", x)
        self._clear_caches()

    @staticmethod
    def _same(snapshot, value):
        return snapshot is not None and snapshot.same_values(value)

    def _prepare_point(self, x):
        if not self._same(self._point_x, x):
            self._point_x = None
            self._point_result = None
        if not self._same(self._hvp_x, x):
            self._hvp_x = None
            self._hvp_v = None
            self._hvp_result = None

    def _production_gradient(self, x):
        state = _owned_copy(x.function, "mtswe_rol_ic_gradient_state")
        target = _owned_copy(self._target, "mtswe_rol_ic_gradient_target")
        with self._physical_coefficient(self.fixed_c0_physical):
            result = self.timestepper.mtswe_terminal_least_squares_gradient(
                self.nsteps, state, self.t0, self.dt, target
            )
        self.production_gradient_evaluations += 1
        return result

    def _production_hvp(self, x, v):
        state = _owned_copy(x.function, "mtswe_rol_ic_hvp_state")
        direction = _owned_copy(v.function, "mtswe_rol_ic_hvp_direction")
        target = _owned_copy(self._target, "mtswe_rol_ic_hvp_target")
        with self._physical_coefficient(self.fixed_c0_physical):
            result = self.timestepper.mtswe_terminal_least_squares_hvp(
                self.nsteps,
                state,
                self.t0,
                self.dt,
                target,
                direction,
                0.0,
            )
        self.production_hvp_evaluations += 1
        return result

    def _point_result_at(self, x):
        self._prepare_point(x)
        if self._point_result is None:
            self._point_result = self._production_gradient(x)
            self._point_x = MTSWEStateVector(x.function, self.helper)
        return self._point_result

    def _hvp_result_at(self, x, v):
        self._prepare_point(x)
        if not self._same(self._hvp_v, v):
            self._hvp_x = None
            self._hvp_v = None
            self._hvp_result = None
        if self._hvp_result is None:
            result = self._production_hvp(x, v)
            self._hvp_x = MTSWEStateVector(x.function, self.helper)
            self._hvp_v = MTSWEStateVector(v.function, self.helper)
            self._hvp_result = result
            self._point_x = MTSWEStateVector(x.function, self.helper)
            self._point_result = result
        return self._hvp_result

    def value(self, x, tol):
        x = self._require_state_vector("x", x)
        return self._value_report(self._point_result_at(x))

    def gradient(self, g, x, tol):
        g = self._require_state_vector("g", g)
        x = self._require_state_vector("x", x)
        self._require_independent_state_output("gradient", g, ("x", x))
        result = self._point_result_at(x)
        self._qualify_gradient(result)
        self._copy_riesz_result(
            g,
            result.initial_condition_gradient,
            "mtswe_rol_ic_gradient_riesz",
        )

    def hessVec(self, hv, v, x, tol):
        hv = self._require_state_vector("hv", hv)
        v = self._require_state_vector("v", v)
        x = self._require_state_vector("x", x)
        self._require_independent_state_output(
            "Hessian", hv, ("v", v), ("x", x)
        )
        result = self._hvp_result_at(x, v)
        self._qualify_hvp(result)
        self._copy_riesz_result(
            hv,
            result.initial_condition_hvp,
            "mtswe_rol_ic_hvp_riesz",
        )


class MTSWECombinedVector(Vector):
    """Owned product vector ``(MTSWEStateVector, normalized scalar)``."""

    def __init__(self, field, scalar):
        if not isinstance(field, MTSWEStateVector):
            raise TypeError("field must be an MTSWEStateVector")
        scalar_value = ProductionMTSWEScalarC0Objective._require_vector(
            "scalar", scalar
        )
        self.field = MTSWEStateVector(field.function, field.helper)
        self.scalar = NumPyVector(
            np.array([scalar_value], dtype=np.float64)
        )
        super().__init__()

    def _require_compatible(self, name, other):
        if not isinstance(other, MTSWECombinedVector):
            raise TypeError(f"{name} must be an MTSWECombinedVector")
        self.field._require_compatible(f"{name}.field", other.field)
        ProductionMTSWEScalarC0Objective._require_vector(
            f"{name}.scalar", other.scalar
        )
        return other

    def clone(self):
        result = type(self)(self.field, self.scalar)
        result.zero()
        return result

    def set(self, other):
        other = self._require_compatible("set input", other)
        self.field.set(other.field)
        self.scalar.array[0] = float(other.scalar.array[0])

    def plus(self, other):
        other = self._require_compatible("plus input", other)
        self.field.plus(other.field)
        self.scalar.plus(other.scalar)

    def scale(self, scale_factor):
        factor = _as_finite_float("scale factor", scale_factor)
        self.field.scale(factor)
        self.scalar.scale(factor)

    def dot(self, other):
        other = self._require_compatible("dot input", other)
        result = self.field.dot(other.field) + float(
            self.scalar.dot(other.scalar)
        )
        if not np.isfinite(result):
            raise FloatingPointError("MTSWE combined dot product is not finite")
        return result

    def norm(self):
        squared = self.dot(self)
        if squared < 0.0:
            tolerance = 100.0 * np.finfo(float).eps * max(abs(squared), 1.0)
            if squared < -tolerance:
                raise FloatingPointError("MTSWE combined norm squared is negative")
        return float(np.sqrt(max(squared, 0.0)))

    def axpy(self, scale_factor, other):
        other = self._require_compatible("axpy input", other)
        factor = _as_finite_float("axpy scale factor", scale_factor)
        self.field.axpy(factor, other.field)
        self.scalar.axpy(factor, other.scalar)

    def zero(self):
        self.field.zero()
        self.scalar.zero()

    def dual(self):
        return self

    def apply(self, other):
        return self.dot(other)

    def dimension(self):
        return self.field.dimension() + 1

    def same_values(self, other):
        other = self._require_compatible("comparison input", other)
        return self.field.same_values(other.field) and (
            float(self.scalar.array[0]) == float(other.scalar.array[0])
        )


class ProductionMTSWECombinedObjective(_ProductionMTSWEStateObjectiveBase):
    """PyROL objective for physical initial state plus normalized ``c0``."""

    def __init__(
        self,
        timestepper,
        coefficient_template,
        target,
        *,
        nsteps,
        t0,
        dt,
        c0_scale,
        gradient_zero_margin_tolerances=None,
        hvp_active_set_tolerances=None,
    ):
        super().__init__(
            timestepper,
            coefficient_template,
            target,
            nsteps=nsteps,
            t0=t0,
            dt=dt,
            gradient_zero_margin_tolerances=gradient_zero_margin_tolerances,
            hvp_active_set_tolerances=hvp_active_set_tolerances,
        )
        self.c0_scale = _as_finite_float(
            "c0_scale", c0_scale, positive=True
        )
        self._point_y = None
        self._point_result = None
        self._hvp_y = None
        self._hvp_q = None
        self._hvp_result = None

    @property
    def cache_info(self):
        return {
            "has_point": self._point_y is not None,
            "has_point_result": self._point_result is not None,
            "has_hvp_point": self._hvp_y is not None,
            "has_hvp_direction": self._hvp_q is not None,
            "has_hvp_result": self._hvp_result is not None,
        }

    def _require_combined(self, name, value, *, finite_scalar=False):
        if not isinstance(value, MTSWECombinedVector):
            raise TypeError(f"{name} must be an MTSWECombinedVector")
        self._require_state_vector(f"{name}.field", value.field)
        ProductionMTSWEScalarC0Objective._require_vector(
            f"{name}.scalar", value.scalar, finite_input=finite_scalar
        )
        return value

    @staticmethod
    def _require_independent_combined_output(output_name, output, *inputs):
        for input_name, input_vector in inputs:
            if (
                output is input_vector
                or output.field.function.dat is input_vector.field.function.dat
                or np.shares_memory(
                    output.scalar.array, input_vector.scalar.array
                )
            ):
                raise ValueError(
                    f"{output_name} output must not alias input {input_name}"
                )

    def _clear_caches(self):
        self._point_y = None
        self._point_result = None
        self._hvp_y = None
        self._hvp_q = None
        self._hvp_result = None

    def update(self, x, *args):
        self._require_combined("x", x, finite_scalar=True)
        self._clear_caches()

    @staticmethod
    def _same(snapshot, value):
        return snapshot is not None and snapshot.same_values(value)

    def _prepare_point(self, y):
        if not self._same(self._point_y, y):
            self._point_y = None
            self._point_result = None
        if not self._same(self._hvp_y, y):
            self._hvp_y = None
            self._hvp_q = None
            self._hvp_result = None

    def _physical_c0(self, y):
        return _as_finite_float(
            "physical c0", self.c0_scale * float(y.scalar.array[0])
        )

    def _production_gradient(self, y):
        state = _owned_copy(y.field.function, "mtswe_rol_combined_state")
        target = _owned_copy(self._target, "mtswe_rol_combined_target")
        with self._physical_coefficient(self._physical_c0(y)):
            result = self.timestepper.mtswe_terminal_least_squares_gradient(
                self.nsteps, state, self.t0, self.dt, target
            )
        self.production_gradient_evaluations += 1
        return result

    def _production_hvp(self, y, q):
        state = _owned_copy(y.field.function, "mtswe_rol_combined_hvp_state")
        direction = _owned_copy(
            q.field.function, "mtswe_rol_combined_hvp_direction"
        )
        target = _owned_copy(self._target, "mtswe_rol_combined_hvp_target")
        delta_c0 = _as_finite_float(
            "physical delta_c0",
            self.c0_scale * float(q.scalar.array[0]),
        )
        with self._physical_coefficient(self._physical_c0(y)):
            result = self.timestepper.mtswe_terminal_least_squares_hvp(
                self.nsteps,
                state,
                self.t0,
                self.dt,
                target,
                direction,
                delta_c0,
            )
        self.production_hvp_evaluations += 1
        return result

    def _point_result_at(self, y):
        self._prepare_point(y)
        if self._point_result is None:
            self._point_result = self._production_gradient(y)
            self._point_y = MTSWECombinedVector(y.field, y.scalar)
        return self._point_result

    def _hvp_result_at(self, y, q):
        self._prepare_point(y)
        if not self._same(self._hvp_q, q):
            self._hvp_y = None
            self._hvp_q = None
            self._hvp_result = None
        if self._hvp_result is None:
            result = self._production_hvp(y, q)
            self._hvp_y = MTSWECombinedVector(y.field, y.scalar)
            self._hvp_q = MTSWECombinedVector(q.field, q.scalar)
            self._hvp_result = result
            self._point_y = MTSWECombinedVector(y.field, y.scalar)
            self._point_result = result
        return self._hvp_result

    def value(self, x, tol):
        x = self._require_combined("x", x, finite_scalar=True)
        return self._value_report(self._point_result_at(x))

    def gradient(self, g, x, tol):
        g = self._require_combined("g", g)
        x = self._require_combined("x", x, finite_scalar=True)
        self._require_independent_combined_output("gradient", g, ("x", x))
        result = self._point_result_at(x)
        self._qualify_gradient(result)
        normalized_scalar = self.c0_scale * float(
            result.physical_c0_gradient
        )
        if not np.isfinite(normalized_scalar):
            raise FloatingPointError(
                "production MTSWE combined scalar gradient is not finite"
            )
        self._copy_riesz_result(
            g.field,
            result.initial_condition_gradient,
            "mtswe_rol_combined_gradient_riesz",
        )
        g.scalar.array[0] = normalized_scalar

    def hessVec(self, hv, v, x, tol):
        hv = self._require_combined("hv", hv)
        v = self._require_combined("v", v, finite_scalar=True)
        x = self._require_combined("x", x, finite_scalar=True)
        self._require_independent_combined_output(
            "Hessian", hv, ("v", v), ("x", x)
        )
        result = self._hvp_result_at(x, v)
        self._qualify_hvp(result)
        normalized_scalar = self.c0_scale * float(result.physical_c0_hvp)
        if not np.isfinite(normalized_scalar):
            raise FloatingPointError(
                "production MTSWE combined scalar HVP is not finite"
            )
        self._copy_riesz_result(
            hv.field,
            result.initial_condition_hvp,
            "mtswe_rol_combined_hvp_riesz",
        )
        hv.scalar.array[0] = normalized_scalar
