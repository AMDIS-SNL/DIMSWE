"""Opt-in production DIMSWE benchmark for recovery of a hidden ``c0``.

The generic learned-physics package imports neither this module nor Firedrake.
This adapter deliberately reuses the unchanged J3 production split, its
hyperviscosity child, and its certified physical-c0 gradient/HVP helper.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np
from firedrake import Function, SpatialCoordinate, as_vector, assemble, cos, inner, pi, sin

from .learned_physics.experiment import (
    ExperimentDefinition,
    ExperimentResult,
    TruthDataset,
    TruthMetadata,
)
from .learned_physics.objectives import LossAccumulation, TrainingMode
from .logger import EmptyLogger
from .models import get_model
from .numpy_helpers import (
    create_flattened_numpy_arr_from_mixed_function,
    set_mixed_function_from_flattened_array,
)
from .parameters import get_parameters, overall_solver_parameters
from .timestepping import get_timestepper


C0_SCALE = 0.07
DEFAULT_TRUTH_C0 = 0.14
DEFAULT_INITIAL_C0 = 0.07
DEFAULT_TRUTH_STEPS = 4
STATE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")


def _finite_float(name, value, *, positive=False):
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(name, value):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _copy_function(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _flat_values(value):
    result = np.asarray(
        create_flattened_numpy_arr_from_mixed_function(value),
        dtype=np.float64,
    ).copy()
    result.setflags(write=False)
    return result


def _serial_solver_parameters():
    result = deepcopy(overall_solver_parameters)
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    for name in (
        "erkstage-f",
        "erkstage-aux",
        "erkstage-mu",
        "erkstage-muaux",
        "erk-dlambda",
        "erk-grad",
    ):
        result[name] = dict(direct)
    return result


@dataclass(frozen=True)
class HiddenC0Case:
    """One serial production model and owned templates used by the benchmark."""

    parameters: dict[str, Any]
    model: Any
    timestepper: Any
    helper: Any
    coefficient_template: Any
    initial_state: Function
    t0: float
    dt: float
    moist_backend: str
    c0_lower: float
    c0_upper: float
    c0_scale: float
    field_sizes: tuple[int, ...]

    def new_state(self, name):
        return self.model.get_x_var(name)[0]

    @contextmanager
    def physical_c0(self, value):
        """Temporarily set physical c0 and restore every child coefficient."""
        c0 = _finite_float("physical c0", value)
        if not self.c0_lower <= c0 <= self.c0_upper:
            raise ValueError(
                f"physical c0 {c0} is outside [{self.c0_lower}, {self.c0_upper}]"
            )
        children = tuple(self.timestepper.time_integrators)
        snapshots = tuple(
            child.coeff.copy(deepcopy=True) for child in children
        )
        working = self.coefficient_template.copy(deepcopy=True)
        c0_index = self.model.get_coeff_list().index("c0")
        working.sub(c0_index).assign(c0)
        try:
            self.timestepper.reset_internal_vars()
            self.timestepper.set_coeff(working)
            yield
        finally:
            try:
                for child, snapshot in zip(children, snapshots):
                    child.set_coeff(snapshot)
            finally:
                self.timestepper.reset_internal_vars()

    def state_from_values(self, values, name):
        result = self.new_state(name)
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (sum(self.field_sizes),):
            raise ValueError("serialized state has the wrong size")
        set_mixed_function_from_flattened_array(result, array)
        return result


def build_hidden_c0_case(
    config_path=None,
    *,
    moist_backend="ufl",
):
    """Construct the deterministic 2-by-2 serial production MTSWE case."""
    if config_path is None:
        config_path = files("dimswe").joinpath(
            "configs", "hidden_c0_tiny.cfg"
        )
    parameters = get_parameters(str(config_path))
    parameters["timestepping"]["subcycle_list"] = [2, 1, 2, 1]
    parameters["hyperviscosity"]["treat_as_coeffs"] = True
    parameters["threewayphysics"]["treat_as_coeffs"] = False

    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    if model.mesh.comm.size != 1:
        raise RuntimeError("hidden-c0 J4A benchmark is serial-only")
    if tuple(model.get_x_var_list()) != STATE_FIELDS:
        raise RuntimeError("hidden-c0 benchmark requires the six-field MTSWE state")

    coefficient, coefficient_sub, _ = model.get_coeff_var(
        "hidden_c0_coefficient_template"
    )
    state_container, state_sub, _ = model.get_full_var(
        "hidden_c0_initial_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)

    # A deterministic, resolved, nonconstant state makes c0 identifiable.
    coordinate = SpatialCoordinate(model.mesh)
    mode_x = sin(2.0 * pi * coordinate[0] / model.initcond.Lx)
    mode_y = cos(2.0 * pi * coordinate[1] / model.initcond.Ly)
    height = 750.0 + 4.0 * mode_x + 3.0 * mode_y
    entropy_density = height * model.initcond.g * (
        1.02 + 0.0015 * mode_x - 0.0010 * mode_y
    )
    state_sub["v"].project(
        as_vector([25.0 + 1.5 * mode_y, 17.0 + 1.0 * mode_x])
    )
    state_sub["h"].project(height)
    state_sub["S"].project(entropy_density)
    state_sub["Qv"].project(0.0030 * height)
    state_sub["Qc"].project(0.0010 * height)
    state_sub["Qr"].project(0.0002 * height)

    timestepper = get_timestepper(
        parameters,
        model,
        logger,
        _serial_solver_parameters(),
        moist_backend=moist_backend,
    )
    timestepper.set_coeff(coefficient)
    helper = timestepper._get_mtswe_split_hvp_helper()
    scales = np.asarray(model.get_coeff_scaling_factors(), dtype=np.float64)
    lower, upper = model.get_coeff_bounds()
    if model.get_coeff_list() != ["s", "c0"] or scales.shape != (2,):
        raise RuntimeError("production coefficient convention changed")
    if float(scales[1]) != C0_SCALE:
        raise RuntimeError("certified c0 = 0.07 z scaling changed")
    field_sizes = tuple(int(block.size) for block in state_container[0].dat.data)
    return HiddenC0Case(
        parameters=parameters,
        model=model,
        timestepper=timestepper,
        helper=helper,
        coefficient_template=coefficient.copy(deepcopy=True),
        initial_state=_copy_function(state_container[0], "hidden_c0_initial_owned"),
        t0=float(time),
        dt=float(parameters["timestepping"]["dt"]),
        moist_backend=str(moist_backend),
        c0_lower=float(lower[1]),
        c0_upper=float(upper[1]),
        c0_scale=float(scales[1]),
        field_sizes=field_sizes,
    )


@dataclass(frozen=True)
class HiddenC0Truth:
    """Owned Firedrake snapshots paired with their portable dense dataset."""

    states: tuple[Function, ...]
    dataset: TruthDataset

    def __post_init__(self):
        owned = tuple(
            _copy_function(state, f"hidden_c0_truth_owned_{index}")
            for index, state in enumerate(self.states)
        )
        if len(owned) != self.dataset.metadata.num_steps + 1:
            raise ValueError("truth state count disagrees with dense dataset")
        object.__setattr__(self, "states", owned)


def _advance(case, initial_state, c0, nsteps, *, start_time=None, prefix="advance"):
    count = _positive_integer("nsteps", nsteps)
    time = case.t0 if start_time is None else _finite_float("start_time", start_time)
    current = _copy_function(initial_state, f"{prefix}_state_0")
    states = [_copy_function(current, f"{prefix}_owned_0")]
    with case.physical_c0(c0):
        for step in range(count):
            cache = case.helper.take_forward_step_cached(
                current, time + step * case.dt, case.dt
            )
            current = _copy_function(cache.state_out, f"{prefix}_state_{step + 1}")
            states.append(_copy_function(current, f"{prefix}_owned_{step + 1}"))
    return tuple(states)


def generate_hidden_c0_truth(
    case: HiddenC0Case,
    *,
    c0_truth=DEFAULT_TRUTH_C0,
    initial_c0=DEFAULT_INITIAL_C0,
    num_steps=DEFAULT_TRUTH_STEPS,
    seed=0,
):
    """Generate explicit truth with production equations differing only in c0."""
    truth_c0 = _finite_float("c0_truth", c0_truth, positive=True)
    guessed_c0 = _finite_float("initial_c0", initial_c0, positive=True)
    if truth_c0 == guessed_c0:
        raise ValueError("hidden c0 must differ from the initial guess")
    if not case.c0_lower <= truth_c0 <= case.c0_upper:
        raise ValueError("truth c0 is outside production bounds")
    count = _positive_integer("num_steps", num_steps)
    states = _advance(
        case,
        case.initial_state,
        truth_c0,
        count,
        prefix="hidden_c0_truth",
    )
    dense_states = np.stack(tuple(_flat_values(state) for state in states))
    times = case.t0 + case.dt * np.arange(count + 1, dtype=np.float64)
    offsets = np.cumsum((0,) + case.field_sizes)
    field_slices = {
        name: [int(offsets[index]), int(offsets[index + 1])]
        for index, name in enumerate(STATE_FIELDS)
    }
    metadata = TruthMetadata(
        benchmark="hidden_c0",
        solver_backend="production_firedrake_mtswe_lie_split",
        timestep=case.dt,
        num_steps=count,
        initial_condition={
            "name": "deterministic_identifiable_mtswe_modes",
            "mesh": "2x2_periodic_quadrilateral",
            "formula_version": 1,
        },
        physical_parameters={
            "hyperviscosity": {
                "s": float(case.parameters["hyperviscosity"]["s"]),
                "c0": truth_c0,
            },
            "threewayphysics": deepcopy(case.parameters["threewayphysics"]),
        },
        truth_c0=truth_c0,
        moist_backend=case.moist_backend,
        random_seed=int(seed),
        state_control_convention={
            "state_fields": STATE_FIELDS,
            "flattening": "mixed_dat_blocks_in_state_field_order",
            "field_sizes": case.field_sizes,
            "field_slices": field_slices,
            "control": "normalized scalar z",
            "physical_map": "c0 = 0.07 z",
            "c0_scale": case.c0_scale,
            "initial_c0": guessed_c0,
        },
        solver_configuration={
            "timestepper_list": tuple(case.timestepper.timestepper_list),
            "termlist": tuple(tuple(x) for x in case.timestepper.termlist),
            "subcycle_list": tuple(case.timestepper.subcycle_list),
            "serial_only": True,
            "accelerator": False,
            "checkpointing": False,
        },
    )
    dataset = TruthDataset(states=dense_states, times=times, metadata=metadata)
    return HiddenC0Truth(states=states, dataset=dataset)


def truth_from_dataset(case: HiddenC0Case, dataset: TruthDataset):
    """Reconstruct owned production-space snapshots from portable truth data."""
    convention = dataset.metadata.state_control_convention
    if tuple(convention["state_fields"]) != STATE_FIELDS:
        raise ValueError("truth state-field convention is incompatible")
    if tuple(convention["field_sizes"]) != case.field_sizes:
        raise ValueError("truth field layout is incompatible with this case")
    if float(dataset.metadata.timestep) != case.dt:
        raise ValueError("truth timestep is incompatible with this case")
    states = tuple(
        case.state_from_values(row, f"hidden_c0_loaded_{index}")
        for index, row in enumerate(dataset.states)
    )
    return HiddenC0Truth(states=states, dataset=dataset)


@dataclass(frozen=True)
class ObjectiveCounts:
    objective_evaluations: int
    gradient_evaluations: int
    hvp_evaluations: int
    solver_calls: int


class HiddenC0Objective:
    """Common normalized-scalar objective interface and cost counters."""

    def __init__(self, mode, c0_scale):
        self.mode = TrainingMode(mode)
        self.c0_scale = _finite_float("c0_scale", c0_scale, positive=True)
        self.objective_evaluations = 0
        self.gradient_evaluations = 0
        self.hvp_evaluations = 0
        self.solver_calls = 0

    def value(self, normalized_z):
        raise NotImplementedError

    def value_and_gradient(self, normalized_z):
        value = self.value(normalized_z)
        return value, self.gradient(normalized_z)

    def gradient(self, normalized_z):
        raise NotImplementedError

    def hess_vec(self, normalized_z, direction):
        raise NotImplementedError

    def counts(self):
        return ObjectiveCounts(
            objective_evaluations=self.objective_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            hvp_evaluations=self.hvp_evaluations,
            solver_calls=self.solver_calls,
        )


@dataclass(frozen=True)
class OfflineObservation:
    """One fixed production hyperviscosity observation at a truth state."""

    target: np.ndarray
    semantics: str

    def __post_init__(self):
        target = np.array(self.target, dtype=np.float64, copy=True)
        if target.ndim != 1 or not np.all(np.isfinite(target)):
            raise ValueError("offline target must be a finite vector")
        if float(np.dot(target, target)) <= np.finfo(np.float64).tiny:
            raise ValueError("offline target must be nonzero")
        target.setflags(write=False)
        object.__setattr__(self, "target", target)


class OfflineC0Objective(HiddenC0Objective):
    """Exact quadratic c0 fit to fixed weak-operator or child-update data."""

    def __init__(self, mode, c0_scale, truth_c0, observations):
        if mode not in (
            TrainingMode.APRIORI_OFFLINE,
            TrainingMode.DISCRETE_OFFLINE,
        ):
            raise ValueError("OfflineC0Objective requires an offline mode")
        super().__init__(mode, c0_scale)
        self.truth_c0 = _finite_float("truth_c0", truth_c0, positive=True)
        self.observations = tuple(observations)
        if not self.observations:
            raise ValueError("offline objective requires observations")

    def _value_gradient(self, normalized_z):
        z = _finite_float("normalized_z", normalized_z)
        ratio_error = self.c0_scale * z / self.truth_c0 - 1.0
        values = []
        gradients = []
        for observation in self.observations:
            target = observation.target
            residual = ratio_error * target
            normalizer = float(np.dot(target, target))
            values.append(0.5 * float(np.dot(residual, residual)) / normalizer)
            gradients.append(
                float(np.dot(residual, target))
                / normalizer
                * self.c0_scale
                / self.truth_c0
            )
        return float(np.mean(values)), float(np.mean(gradients))

    def value(self, normalized_z):
        self.objective_evaluations += 1
        return self._value_gradient(normalized_z)[0]

    def value_and_gradient(self, normalized_z):
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        return self._value_gradient(normalized_z)

    def gradient(self, normalized_z):
        self.gradient_evaluations += 1
        return self._value_gradient(normalized_z)[1]

    def hess_vec(self, normalized_z, direction):
        _finite_float("normalized_z", normalized_z)
        q = _finite_float("direction", direction)
        self.hvp_evaluations += 1
        return (self.c0_scale / self.truth_c0) ** 2 * q


@dataclass(frozen=True)
class SolverObservation:
    """A trusted start/target pair evaluated by an autonomous solver prefix."""

    initial_state: Function
    target: Function
    start_time: float
    nsteps: int
    normalizer: float
    window_index: int
    target_step: int

    def __post_init__(self):
        object.__setattr__(
            self,
            "initial_state",
            _copy_function(self.initial_state, "hidden_c0_observation_initial"),
        )
        object.__setattr__(
            self,
            "target",
            _copy_function(self.target, "hidden_c0_observation_target"),
        )
        object.__setattr__(self, "nsteps", _positive_integer("nsteps", self.nsteps))
        object.__setattr__(
            self, "normalizer", _finite_float("normalizer", self.normalizer, positive=True)
        )


def _state_squared_difference(case, left, right, name):
    residual = _copy_function(left, name)
    with residual.dat.vec as residual_vec, right.dat.vec_ro as right_vec:
        residual_vec.axpy(-1.0, right_vec)
    return float(assemble(inner(residual, residual) * case.model.spaces.dx))


class SolverInLoopC0Objective(HiddenC0Objective):
    """Aggregate exact production terminal losses over explicit observations."""

    def __init__(self, mode, case, observations, accumulation):
        if mode not in (TrainingMode.TRUTH_RESET, TrainingMode.ROLLOUT):
            raise ValueError("solver-in-loop objective requires reset or rollout mode")
        super().__init__(mode, case.c0_scale)
        self.case = case
        self.observations = tuple(observations)
        self.accumulation = LossAccumulation(accumulation)
        if not self.observations:
            raise ValueError("solver-in-loop objective requires observations")

    def _physical_c0(self, normalized_z):
        return self.c0_scale * _finite_float("normalized_z", normalized_z)

    def _value_observation(self, observation, physical_c0):
        prediction = _advance(
            self.case,
            observation.initial_state,
            physical_c0,
            observation.nsteps,
            start_time=observation.start_time,
            prefix=f"hidden_c0_{self.mode.value}_value",
        )[-1]
        self.solver_calls += observation.nsteps
        squared = _state_squared_difference(
            self.case,
            prediction,
            observation.target,
            f"hidden_c0_{self.mode.value}_residual",
        )
        return 0.5 * squared / observation.normalizer

    def value(self, normalized_z):
        physical_c0 = self._physical_c0(normalized_z)
        self.objective_evaluations += 1
        values = tuple(
            self._value_observation(observation, physical_c0)
            for observation in self.observations
        )
        return float(np.mean(values))

    def _gradient_observation(self, observation, physical_c0):
        with self.case.physical_c0(physical_c0):
            result = self.case.helper.terminal_least_squares_gradient(
                observation.nsteps,
                _copy_function(
                    observation.initial_state,
                    f"hidden_c0_{self.mode.value}_gradient_initial",
                ),
                observation.start_time,
                self.case.dt,
                _copy_function(
                    observation.target,
                    f"hidden_c0_{self.mode.value}_gradient_target",
                ),
            )
        self.solver_calls += observation.nsteps
        value = float(result.objective_value) / observation.normalizer
        gradient = (
            self.c0_scale
            * float(result.physical_c0_gradient)
            / observation.normalizer
        )
        return value, gradient

    def value_and_gradient(self, normalized_z):
        physical_c0 = self._physical_c0(normalized_z)
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        pairs = tuple(
            self._gradient_observation(observation, physical_c0)
            for observation in self.observations
        )
        return (
            float(np.mean(tuple(value for value, _ in pairs))),
            float(np.mean(tuple(gradient for _, gradient in pairs))),
        )

    def gradient(self, normalized_z):
        physical_c0 = self._physical_c0(normalized_z)
        self.gradient_evaluations += 1
        gradients = tuple(
            self._gradient_observation(observation, physical_c0)[1]
            for observation in self.observations
        )
        return float(np.mean(gradients))

    def _hvp_observation(self, observation, physical_c0, normalized_direction):
        zero = self.case.new_state(f"hidden_c0_{self.mode.value}_zero_direction")
        zero.assign(0)
        physical_direction = self.c0_scale * normalized_direction
        with self.case.physical_c0(physical_c0):
            result = self.case.helper.terminal_least_squares_hvp(
                observation.nsteps,
                _copy_function(
                    observation.initial_state,
                    f"hidden_c0_{self.mode.value}_hvp_initial",
                ),
                observation.start_time,
                self.case.dt,
                _copy_function(
                    observation.target,
                    f"hidden_c0_{self.mode.value}_hvp_target",
                ),
                zero,
                physical_direction,
            )
        self.solver_calls += observation.nsteps
        return (
            self.c0_scale
            * float(result.physical_c0_hvp)
            / observation.normalizer
        )

    def hess_vec(self, normalized_z, direction):
        physical_c0 = self._physical_c0(normalized_z)
        q = _finite_float("direction", direction)
        self.hvp_evaluations += 1
        values = tuple(
            self._hvp_observation(observation, physical_c0, q)
            for observation in self.observations
        )
        return float(np.mean(values))


@dataclass(frozen=True)
class HiddenC0ObjectiveSuite:
    """The four mode-specific objectives prepared from one truth trajectory."""

    objectives: Any
    preprocessing_hyperviscosity_child_calls: int
    preprocessing_complete_solver_calls: int
    truth_reset_horizon: int
    rollout_horizon: int
    rollout_accumulation: LossAccumulation

    def __post_init__(self):
        expected = set(TrainingMode)
        if set(self.objectives) != expected:
            raise ValueError("objective suite must contain all four training modes")
        object.__setattr__(
            self, "objectives", MappingProxyType(dict(self.objectives))
        )

    def __getitem__(self, mode):
        return self.objectives[TrainingMode(mode)]


def _offline_observations(case, truth):
    weak_observations = []
    update_observations = []
    truth_c0 = truth.dataset.metadata.truth_c0
    hyper_helper = case.helper.hyper_helper
    with case.physical_c0(truth_c0):
        for index, state in enumerate(truth.states[:-1]):
            cache = hyper_helper.take_forward_step_cached(
                state, case.t0 + index * case.dt, case.dt
            )
            # M*tendency is algebraically the assembled deployed weak RHS.  It
            # removes the child mass solve and Euler dt from the a-priori loss.
            weak_dual = hyper_helper.state_mass_map(
                cache.tendency, f"hidden_c0_weak_target_{index}"
            )
            weak_observations.append(
                OfflineObservation(
                    target=_flat_values(weak_dual),
                    semantics="assembled weak hyperviscosity RHS before M^-1 and dt",
                )
            )
            update = _flat_values(cache.state_out) - _flat_values(state)
            update_observations.append(
                OfflineObservation(
                    target=update,
                    semantics="deployed hyperviscosity Euler child increment dt*M^-1*b",
                )
            )
    return tuple(weak_observations), tuple(update_observations)


def _baseline_normalizer(case, initial, target, start_time, nsteps, initial_c0, name):
    prediction = _advance(
        case,
        initial,
        initial_c0,
        nsteps,
        start_time=start_time,
        prefix=f"hidden_c0_{name}_baseline",
    )[-1]
    squared = _state_squared_difference(
        case, prediction, target, f"hidden_c0_{name}_baseline_residual"
    )
    if not np.isfinite(squared) or squared <= np.finfo(np.float64).tiny:
        raise RuntimeError(
            f"{name} baseline has no resolvable hidden-c0 signal"
        )
    return squared


def prepare_hidden_c0_objectives(
    case: HiddenC0Case,
    truth: HiddenC0Truth,
    *,
    initial_c0=DEFAULT_INITIAL_C0,
    truth_reset_horizon=1,
    rollout_horizon=3,
    truth_reset_accumulation=LossAccumulation.TERMINAL,
    rollout_accumulation=LossAccumulation.ACCUMULATED,
):
    """Explicitly prepare all four losses from the same stored truth data."""
    guessed_c0 = _finite_float("initial_c0", initial_c0, positive=True)
    reset_horizon = _positive_integer("truth_reset_horizon", truth_reset_horizon)
    autonomous_horizon = _positive_integer("rollout_horizon", rollout_horizon)
    if reset_horizon not in (1, 3):
        raise ValueError("certified truth-reset horizons are 1 or 3")
    if autonomous_horizon not in (1, 3):
        raise ValueError("certified rollout horizons are 1 or 3")
    available = truth.dataset.metadata.num_steps
    if reset_horizon > available or autonomous_horizon > available:
        raise ValueError("truth trajectory is shorter than an objective horizon")
    reset_accumulation = LossAccumulation(truth_reset_accumulation)
    rollout_accumulation = LossAccumulation(rollout_accumulation)
    truth_c0 = truth.dataset.metadata.truth_c0

    weak, updates = _offline_observations(case, truth)
    apriori = OfflineC0Objective(
        TrainingMode.APRIORI_OFFLINE, case.c0_scale, truth_c0, weak
    )
    discrete = OfflineC0Objective(
        TrainingMode.DISCRETE_OFFLINE, case.c0_scale, truth_c0, updates
    )

    reset_observations = []
    reset_prefixes = (
        range(1, reset_horizon + 1)
        if reset_accumulation is LossAccumulation.ACCUMULATED
        else (reset_horizon,)
    )
    reset_windows = available - reset_horizon + 1
    reset_solver_calls = 0
    for window in range(reset_windows):
        for prefix in reset_prefixes:
            normalizer = _baseline_normalizer(
                case,
                truth.states[window],
                truth.states[window + prefix],
                case.t0 + window * case.dt,
                prefix,
                guessed_c0,
                f"truth_reset_{window}_{prefix}",
            )
            reset_solver_calls += prefix
            reset_observations.append(
                SolverObservation(
                    initial_state=truth.states[window],
                    target=truth.states[window + prefix],
                    start_time=case.t0 + window * case.dt,
                    nsteps=prefix,
                    normalizer=normalizer,
                    window_index=window,
                    target_step=window + prefix,
                )
            )
    reset = SolverInLoopC0Objective(
        TrainingMode.TRUTH_RESET,
        case,
        tuple(reset_observations),
        reset_accumulation,
    )

    rollout_prefixes = (
        range(1, autonomous_horizon + 1)
        if rollout_accumulation is LossAccumulation.ACCUMULATED
        else (autonomous_horizon,)
    )
    rollout_observations = []
    rollout_solver_calls = 0
    for prefix in rollout_prefixes:
        normalizer = _baseline_normalizer(
            case,
            truth.states[0],
            truth.states[prefix],
            case.t0,
            prefix,
            guessed_c0,
            f"rollout_{prefix}",
        )
        rollout_solver_calls += prefix
        rollout_observations.append(
            SolverObservation(
                initial_state=truth.states[0],
                target=truth.states[prefix],
                start_time=case.t0,
                nsteps=prefix,
                normalizer=normalizer,
                window_index=0,
                target_step=prefix,
            )
        )
    autonomous = SolverInLoopC0Objective(
        TrainingMode.ROLLOUT,
        case,
        tuple(rollout_observations),
        rollout_accumulation,
    )
    return HiddenC0ObjectiveSuite(
        objectives={
            TrainingMode.APRIORI_OFFLINE: apriori,
            TrainingMode.DISCRETE_OFFLINE: discrete,
            TrainingMode.TRUTH_RESET: reset,
            TrainingMode.ROLLOUT: autonomous,
        },
        preprocessing_hyperviscosity_child_calls=len(truth.states) - 1,
        preprocessing_complete_solver_calls=(
            reset_solver_calls + rollout_solver_calls
        ),
        truth_reset_horizon=reset_horizon,
        rollout_horizon=autonomous_horizon,
        rollout_accumulation=rollout_accumulation,
    )


@dataclass(frozen=True)
class ScalarOptimizerConfiguration:
    """Common scale-invariant bounded-Newton budget for every c0 mode.

    The three tolerances are dimensionless.  ``gradient_tolerance`` bounds
    reduction relative to the initial nonzero gradient,
    ``minimum_curvature`` bounds positive curvature relative to the first
    nonzero curvature seen, and ``step_tolerance`` bounds a parameter step
    relative to ``max(1, abs(z))``.  Retaining the established field names
    avoids changing serialized Test-1A/Test-1B optimizer configurations.
    """

    physical_lower: float = 0.01
    physical_upper: float = 0.30
    max_iterations: int = 8
    max_line_search_steps: int = 6
    gradient_tolerance: float = 1.0e-9
    step_tolerance: float = 1.0e-11
    minimum_curvature: float = 1.0e-12
    armijo_constant: float = 1.0e-4
    use_hvp: bool = True

    def __post_init__(self):
        lower = _finite_float("physical_lower", self.physical_lower)
        upper = _finite_float("physical_upper", self.physical_upper)
        if lower >= upper:
            raise ValueError("physical optimizer bounds are empty")
        _positive_integer("max_iterations", self.max_iterations)
        _positive_integer("max_line_search_steps", self.max_line_search_steps)
        dimensionless_tolerances = {
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "minimum_curvature": self.minimum_curvature,
        }
        for name, value in dimensionless_tolerances.items():
            tolerance = _finite_float(name, value, positive=True)
            if tolerance >= 1.0:
                raise ValueError(f"{name} must be less than one")
        armijo = _finite_float(
            "armijo_constant", self.armijo_constant, positive=True
        )
        if armijo >= 1.0:
            raise ValueError("armijo_constant must be less than one")


@dataclass(frozen=True)
class ScalarOptimizationResult:
    """Owned scalar fit history and accounting independent of PyROL."""

    starting_c0: float
    recovered_c0: float
    starting_normalized_z: float
    recovered_normalized_z: float
    objective_history: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    normalized_iterates: tuple[float, ...]
    counts: ObjectiveCounts
    wall_time_seconds: float
    success: bool
    termination_reason: str
    failure_reason: str | None


def _relative_gradient_converged(gradient, reference, tolerance):
    """Return a positive-objective-scale-invariant gradient decision."""
    magnitude = abs(float(gradient))
    if reference == 0.0:
        return magnitude == 0.0
    return magnitude / reference <= tolerance


def _bound_stationary(z, gradient, lower, upper):
    """Apply the scalar first-order bound condition without a scale."""
    return bool(
        (z <= lower and gradient > 0.0)
        or (z >= upper and gradient < 0.0)
    )


def _positive_curvature(candidate, reference, relative_tolerance):
    """Accept positive curvature by a homogeneous relative-magnitude test."""
    if not np.isfinite(candidate):
        return None, reference
    value = float(candidate)
    if reference is None and value != 0.0:
        reference = abs(value)
    if value <= 0.0 or reference is None:
        return None, reference
    if value / reference <= relative_tolerance:
        return None, reference
    return value, reference


def optimize_hidden_c0(
    objective: HiddenC0Objective,
    initial_c0=DEFAULT_INITIAL_C0,
    configuration=ScalarOptimizerConfiguration(),
):
    """Fit normalized c0 with scale-invariant safeguarded Newton steps."""
    if not isinstance(configuration, ScalarOptimizerConfiguration):
        raise TypeError("configuration must be ScalarOptimizerConfiguration")
    initial = _finite_float("initial_c0", initial_c0, positive=True)
    scale = objective.c0_scale
    lower = max(configuration.physical_lower, 0.0) / scale
    upper = configuration.physical_upper / scale
    z = float(np.clip(initial / scale, lower, upper))
    objective_history = []
    gradient_norms = []
    iterates = []
    success = False
    failure_reason = None
    termination_reason = None
    initial_gradient_norm = None
    curvature_reference = None
    started = perf_counter()

    previous_z = None
    previous_gradient = None
    for _ in range(configuration.max_iterations):
        value, gradient = objective.value_and_gradient(z)
        if not np.isfinite(value) or not np.isfinite(gradient):
            failure_reason = "nonfinite objective or gradient"
            break
        objective_history.append(float(value))
        gradient_norms.append(abs(float(gradient)))
        iterates.append(float(z))
        if initial_gradient_norm is None:
            initial_gradient_norm = abs(float(gradient))
        if _relative_gradient_converged(
            gradient,
            initial_gradient_norm,
            configuration.gradient_tolerance,
        ):
            success = True
            termination_reason = (
                "initial gradient is exactly zero"
                if initial_gradient_norm == 0.0
                else "relative gradient tolerance satisfied"
            )
            break
        if _bound_stationary(z, gradient, lower, upper):
            success = True
            termination_reason = "projected gradient satisfies bound constraint"
            break

        curvature = None
        if configuration.use_hvp:
            candidate_curvature = objective.hess_vec(z, 1.0)
            curvature, curvature_reference = _positive_curvature(
                candidate_curvature,
                curvature_reference,
                configuration.minimum_curvature,
            )
        if curvature is None and previous_z is not None:
            dz = z - previous_z
            relative_dz = abs(dz) / max(1.0, abs(previous_z))
            if relative_dz > configuration.step_tolerance:
                secant = (gradient - previous_gradient) / dz
                curvature, curvature_reference = _positive_curvature(
                    secant,
                    curvature_reference,
                    configuration.minimum_curvature,
                )
        if curvature is None:
            # A bounded gradient step is deterministic and only a fallback for
            # a nonpositive local scalar curvature.
            proposal = z - np.sign(gradient) * 0.25 * (upper - lower)
        else:
            proposal = z - gradient / curvature
        proposal = float(np.clip(proposal, lower, upper))
        relative_step = abs(proposal - z) / max(1.0, abs(z))
        if relative_step <= configuration.step_tolerance:
            success = True
            termination_reason = "relative parameter step tolerance satisfied"
            break

        accepted = False
        trial = proposal
        for _ in range(configuration.max_line_search_steps):
            trial_value = objective.value(trial)
            armijo_bound = value + (
                configuration.armijo_constant * gradient * (trial - z)
            )
            if np.isfinite(trial_value) and trial_value <= armijo_bound:
                accepted = True
                break
            trial = 0.5 * (z + trial)
        if not accepted:
            failure_reason = "line search failed to reduce the objective"
            termination_reason = failure_reason
            break
        accepted_relative_step = abs(trial - z) / max(1.0, abs(z))
        previous_z, previous_gradient = z, gradient
        z = float(trial)
        if accepted_relative_step <= configuration.step_tolerance:
            success = True
            termination_reason = "relative parameter step tolerance satisfied"
            break

    if not success and failure_reason is None:
        # A last requested gradient makes the iteration-limit result explicit.
        value, gradient = objective.value_and_gradient(z)
        objective_history.append(float(value))
        gradient_norms.append(abs(float(gradient)))
        iterates.append(float(z))
        if not np.isfinite(value) or not np.isfinite(gradient):
            failure_reason = "nonfinite objective or gradient"
            termination_reason = failure_reason
        else:
            if initial_gradient_norm is None:
                initial_gradient_norm = abs(float(gradient))
            success = _relative_gradient_converged(
                gradient,
                initial_gradient_norm,
                configuration.gradient_tolerance,
            )
            if success:
                termination_reason = (
                    "initial gradient is exactly zero"
                    if initial_gradient_norm == 0.0
                    else "relative gradient tolerance satisfied"
                )
            elif _bound_stationary(z, gradient, lower, upper):
                success = True
                termination_reason = (
                    "projected gradient satisfies bound constraint"
                )
        if not success and failure_reason is None:
            failure_reason = "iteration limit reached"
            termination_reason = failure_reason
    if termination_reason is None:
        termination_reason = failure_reason or "optimizer terminated"
    elapsed = perf_counter() - started
    return ScalarOptimizationResult(
        starting_c0=initial,
        recovered_c0=scale * z,
        starting_normalized_z=initial / scale,
        recovered_normalized_z=z,
        objective_history=tuple(objective_history),
        gradient_norms=tuple(gradient_norms),
        normalized_iterates=tuple(iterates),
        counts=objective.counts(),
        wall_time_seconds=float(elapsed),
        success=success,
        termination_reason=termination_reason,
        failure_reason=failure_reason,
    )


@dataclass(frozen=True)
class ObjectiveScan:
    """Small independent scalar landscape oracle; plotting is external."""

    physical_c0: np.ndarray
    normalized_z: np.ndarray
    objective: np.ndarray
    minimum_physical_c0: float

    def __post_init__(self):
        for name in ("physical_c0", "normalized_z", "objective"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def scan_hidden_c0_objective(objective, physical_c0_values):
    """Evaluate a supplied dense one-dimensional physical-c0 grid."""
    physical = np.asarray(physical_c0_values, dtype=np.float64)
    if physical.ndim != 1 or physical.size < 3:
        raise ValueError("objective scan requires at least three scalar points")
    if not np.all(np.isfinite(physical)) or np.any(np.diff(physical) <= 0.0):
        raise ValueError("objective scan points must be finite and increasing")
    normalized = physical / objective.c0_scale
    values = np.array(tuple(objective.value(z) for z in normalized))
    minimum_index = int(np.argmin(values))
    return ObjectiveScan(
        physical_c0=physical,
        normalized_z=normalized,
        objective=values,
        minimum_physical_c0=float(physical[minimum_index]),
    )


def default_hidden_c0_scan(objective, truth_c0=DEFAULT_TRUTH_C0, points=17):
    """Return the standard small scan centered exactly on the truth value."""
    count = _positive_integer("points", points)
    if count < 3 or count % 2 == 0:
        raise ValueError("default scan uses an odd number of at least three points")
    truth_value = _finite_float("truth_c0", truth_c0, positive=True)
    return scan_hidden_c0_objective(
        objective,
        np.linspace(0.5 * truth_value, 1.5 * truth_value, count),
    )


def _state_relative_error(case, predicted, target, name):
    numerator = _state_squared_difference(case, predicted, target, f"{name}_residual")
    zero = case.new_state(f"{name}_zero")
    zero.assign(0)
    denominator = _state_squared_difference(case, target, zero, f"{name}_target")
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def _field_relative_errors(case, predicted, target, name):
    result = {}
    for index, field_name in enumerate(STATE_FIELDS):
        residual = case.new_state(f"{name}_{field_name}_residual")
        residual.assign(predicted)
        residual.sub(index).assign(predicted.sub(index) - target.sub(index))
        numerator = float(
            assemble(inner(residual.sub(index), residual.sub(index)) * case.model.spaces.dx)
        )
        denominator = float(
            assemble(inner(target.sub(index), target.sub(index)) * case.model.spaces.dx)
        )
        result[field_name] = float(
            np.sqrt(numerator / max(denominator, np.finfo(float).tiny))
        )
    return result


def _trajectory_errors(case, predicted, targets, name):
    if len(predicted) != len(targets):
        raise ValueError("prediction and truth trajectories differ in length")
    step_errors = tuple(
        _state_relative_error(case, left, right, f"{name}_{index}")
        for index, (left, right) in enumerate(zip(predicted[1:], targets[1:]), 1)
    )
    squared_numerator = sum(
        _state_squared_difference(case, left, right, f"{name}_accum_{index}")
        for index, (left, right) in enumerate(zip(predicted[1:], targets[1:]), 1)
    )
    zero = case.new_state(f"{name}_accum_zero")
    zero.assign(0)
    squared_denominator = sum(
        _state_squared_difference(case, target, zero, f"{name}_denom_{index}")
        for index, target in enumerate(targets[1:], 1)
    )
    return {
        "per_step": step_errors,
        "final": step_errors[-1],
        "accumulated": float(
            np.sqrt(
                squared_numerator
                / max(squared_denominator, np.finfo(float).tiny)
            )
        ),
    }


def evaluate_hidden_c0(
    case,
    truth,
    suite,
    recovered_c0,
    *,
    initial_c0=DEFAULT_INITIAL_C0,
    cost_metrics=None,
):
    """Cross-evaluate one fitted c0 under all losses and deployment metrics."""
    recovered = _finite_float("recovered_c0", recovered_c0, positive=True)
    initial = _finite_float("initial_c0", initial_c0, positive=True)
    truth_c0 = truth.dataset.metadata.truth_c0
    horizon = suite.rollout_horizon
    predicted = _advance(
        case,
        truth.states[0],
        recovered,
        horizon,
        prefix="hidden_c0_evaluation_recovered",
    )
    repeated = _advance(
        case,
        truth.states[0],
        recovered,
        horizon,
        prefix="hidden_c0_evaluation_repeat",
    )
    target = truth.states[: horizon + 1]
    trajectory = _trajectory_errors(
        case, predicted, target, "hidden_c0_evaluation"
    )
    finite = all(np.all(np.isfinite(_flat_values(state))) for state in predicted)
    repeatable = all(
        np.array_equal(_flat_values(left), _flat_values(right))
        for left, right in zip(predicted, repeated)
    )
    cross = {}
    for mode in TrainingMode:
        objective = suite[mode]
        cross[mode.value] = {
            "initial": objective.value(initial / case.c0_scale),
            "recovered": objective.value(recovered / case.c0_scale),
        }
    field_final = _field_relative_errors(
        case, predicted[-1], target[-1], "hidden_c0_final_fields"
    )
    field_one_step = _field_relative_errors(
        case, predicted[1], target[1], "hidden_c0_one_step_fields"
    )
    metrics = {
        "truth_c0": truth_c0,
        "recovered_c0": recovered,
        "normalized_z": recovered / case.c0_scale,
        "physical_c0_absolute_error": abs(recovered - truth_c0),
        "physical_c0_relative_error": abs(recovered - truth_c0) / abs(truth_c0),
        "one_step_state_prediction_error": trajectory["per_step"][0],
        "short_autonomous_rollout_error": trajectory["accumulated"],
        "final_state_error": trajectory["final"],
        "accumulated_trajectory_error": trajectory["accumulated"],
        "per_step_state_errors": trajectory["per_step"],
        "objectives_under_all_training_modes": cross,
        "state_field_block_errors": {
            "one_step": field_one_step,
            "final": field_final,
        },
        "stability": {
            "all_states_finite": bool(finite),
            "num_deployed_steps": horizon,
        },
        "repeatability": {
            "exact_repeated_state_vectors": bool(repeatable),
            "seed": truth.dataset.metadata.random_seed,
        },
        "cost_metrics": {} if cost_metrics is None else dict(cost_metrics),
    }
    return metrics


def hidden_c0_experiment_definition(
    mode,
    truth,
    suite,
    configuration=ScalarOptimizerConfiguration(),
):
    """Build the generic immutable record contract for one Benchmark-1 fit."""
    selected = TrainingMode(mode)
    return ExperimentDefinition(
        benchmark="hidden_c0",
        truth_configuration=truth.dataset.metadata.to_dict(),
        baseline_configuration={
            "equations": "identical_to_truth",
            "only_difference": "physical_c0",
            "initial_c0": DEFAULT_INITIAL_C0,
            "moist_backend": truth.dataset.metadata.moist_backend,
        },
        model_configuration={
            "parameter_pytree": "scalar normalized_z",
            "physical_parameterization": "c0 = 0.07 z",
            "architecture": None,
            "feature_map": None,
            "output_map": "production hyperviscosity coefficient",
        },
        training_mode=selected,
        observation_definition={
            "truth_reset_horizon": suite.truth_reset_horizon,
            "rollout_horizon": suite.rollout_horizon,
            "rollout_accumulation": suite.rollout_accumulation.value,
        },
        rollout_horizon=suite.rollout_horizon,
        seed=truth.dataset.metadata.random_seed,
        optimizer_configuration={
            "name": "deterministic_bounded_newton",
            **configuration.__dict__,
        },
        evaluation_metrics=(
            "physical_c0_error",
            "one_step_state_prediction_error",
            "short_autonomous_rollout_error",
            "final_state_error",
            "accumulated_trajectory_error",
            "all_training_mode_objectives",
            "state_field_block_errors",
            "finite_state",
            "repeatability",
            "cost",
            "objective_scan",
        ),
    )


def run_hidden_c0_experiment(
    case,
    truth,
    suite,
    mode,
    *,
    initial_c0=DEFAULT_INITIAL_C0,
    configuration=ScalarOptimizerConfiguration(),
    scan_points=17,
):
    """Optimize one mode and package its mandatory common cross-evaluation."""
    selected = TrainingMode(mode)
    objective = suite[selected]
    initial_value = objective.value(initial_c0 / case.c0_scale)
    fit = optimize_hidden_c0(
        objective, initial_c0=initial_c0, configuration=configuration
    )
    final_value = objective.value(fit.recovered_normalized_z)
    scan_counts_before = objective.counts()
    scan = default_hidden_c0_scan(
        objective,
        truth_c0=truth.dataset.metadata.truth_c0,
        points=scan_points,
    )
    scan_counts_after = objective.counts()
    relative_error = abs(
        fit.recovered_c0 - truth.dataset.metadata.truth_c0
    ) / abs(truth.dataset.metadata.truth_c0)
    cost = {
        "objective_evaluations": fit.counts.objective_evaluations,
        "gradient_evaluations": fit.counts.gradient_evaluations,
        "hvp_evaluations": fit.counts.hvp_evaluations,
        "solver_calls": fit.counts.solver_calls,
        "wall_time_seconds": fit.wall_time_seconds,
        "preprocessing_hyperviscosity_child_calls": (
            suite.preprocessing_hyperviscosity_child_calls
        ),
        "preprocessing_complete_solver_calls": (
            suite.preprocessing_complete_solver_calls
        ),
        "objective_scan": {
            "objective_evaluations": (
                scan_counts_after.objective_evaluations
                - scan_counts_before.objective_evaluations
            ),
            "solver_calls": (
                scan_counts_after.solver_calls - scan_counts_before.solver_calls
            ),
        },
    }
    evaluation = evaluate_hidden_c0(
        case,
        truth,
        suite,
        fit.recovered_c0,
        initial_c0=initial_c0,
        cost_metrics=cost,
    )
    evaluation["training_objective_initial"] = initial_value
    evaluation["training_objective_final"] = final_value
    evaluation["training_objective_reduction"] = initial_value - final_value
    evaluation["relative_parameter_error"] = relative_error
    evaluation["objective_scan"] = {
        "physical_c0": scan.physical_c0.tolist(),
        "normalized_z": scan.normalized_z.tolist(),
        "objective": scan.objective.tolist(),
        "minimum_physical_c0": scan.minimum_physical_c0,
    }
    definition = hidden_c0_experiment_definition(
        selected, truth, suite, configuration
    )
    return ExperimentResult(
        benchmark="hidden_c0",
        training_mode=selected,
        seed=definition.seed,
        truth_configuration=definition.truth_configuration,
        baseline_configuration=definition.baseline_configuration,
        model_configuration=definition.model_configuration,
        initial_parameters={
            "normalized_z": initial_c0 / case.c0_scale,
            "physical_c0": initial_c0,
        },
        final_parameters={
            "normalized_z": fit.recovered_normalized_z,
            "physical_c0": fit.recovered_c0,
        },
        objective_history=fit.objective_history,
        gradient_norms=fit.gradient_norms,
        objective_evaluations=fit.counts.objective_evaluations,
        gradient_evaluations=fit.counts.gradient_evaluations,
        hvp_evaluations=fit.counts.hvp_evaluations,
        solver_calls=fit.counts.solver_calls,
        timing={"optimization_wall_time_seconds": fit.wall_time_seconds},
        deployment_evaluation_metrics=evaluation,
        success=fit.success,
        failure_reason=fit.failure_reason,
    )


__all__ = (
    "C0_SCALE",
    "DEFAULT_INITIAL_C0",
    "DEFAULT_TRUTH_C0",
    "DEFAULT_TRUTH_STEPS",
    "HiddenC0Case",
    "HiddenC0ObjectiveSuite",
    "HiddenC0Truth",
    "ObjectiveCounts",
    "ObjectiveScan",
    "ScalarOptimizationResult",
    "ScalarOptimizerConfiguration",
    "build_hidden_c0_case",
    "default_hidden_c0_scan",
    "evaluate_hidden_c0",
    "generate_hidden_c0_truth",
    "hidden_c0_experiment_definition",
    "optimize_hidden_c0",
    "prepare_hidden_c0_objectives",
    "run_hidden_c0_experiment",
    "scan_hidden_c0_objective",
    "truth_from_dataset",
)
