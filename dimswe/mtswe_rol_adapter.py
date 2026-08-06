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
from pyrol import Objective
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
        matches = tuple(
            child
            for child in parent_cache.children
            if child.name == "moist_euler"
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"MTSWE timestep {timestep} does not contain one moist_euler cache"
            )
        return matches[0].cache.active_set

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
                margin = float(getattr(diagnostic, _MARGIN_ATTRIBUTES[switch]))
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
