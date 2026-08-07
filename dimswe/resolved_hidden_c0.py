"""Pure configuration and diagnostic contracts for resolved hidden-c0 work.

This module is the inexpensive J4B-PREP boundary.  It imports neither
Firedrake nor the production timestepper.  The opt-in execution adapter lives
in :mod:`dimswe.resolved_hidden_c0_driver`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .learned_physics.experiment import canonical_json
from .learned_physics.objectives import LossAccumulation


RESOLVED_FORMAT_VERSION = 1
C0_SCALE = 0.07
STATE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")


def _positive_integer(name: str, value: Any) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class CandidateCase:
    """Repository-archeology record, not a scientific case selection."""

    name: str
    source: str
    readiness: str
    expected_dynamics: str
    identifiability_risk: str


CANDIDATE_CASES = MappingProxyType(
    {
        "doublevortex": CandidateCase(
            name="doublevortex",
            source="dimswe/initial_conditions.py:DoubleVortex; mtswe.cfg",
            readiness="configured production MTSWE case",
            expected_dynamics=(
                "two moist thermal vortices on a doubly periodic plane"
            ),
            identifiability_risk=(
                "smooth initial vortices may require sufficient duration and "
                "resolution before high modes are populated"
            ),
        ),
        "TC5": CandidateCase(
            name="TC5",
            source="dimswe/initial_conditions.py:TC5",
            readiness="complete initial condition but no dedicated repository cfg",
            expected_dynamics=(
                "balanced moist zonal flow interacting with a conical mountain"
            ),
            identifiability_risk=(
                "the mountain is narrow on coarse meshes and the planar periodic "
                "adaptation needs external inspection"
            ),
        ),
        "TC2": CandidateCase(
            name="TC2",
            source="dimswe/initial_conditions.py:TC2",
            readiness="complete initial condition but no dedicated repository cfg",
            expected_dynamics="smooth balanced moist zonal flow",
            identifiability_risk=(
                "near-steady large-scale flow is unlikely to populate modes on "
                "which hyperviscosity is identifiable"
            ),
        ),
    }
)


@dataclass(frozen=True)
class ResolvedPilotConfiguration:
    """One complete-production serial pilot run.

    ``output_directory`` is operational provenance and is deliberately
    excluded from :meth:`physics_configuration`; paired physics configurations
    can therefore be proved equal except for ``c0``.
    """

    case: str = "doublevortex"
    nx: int = 16
    ny: int = 16
    dt: float = 400.0
    nsteps: int = 20
    output_stride: int = 2
    c0: float = 0.07
    s: float = 3.2
    moist_backend: str = "ufl"
    seed: int = 0
    output_directory: str = "resolved_hidden_c0/doublevortex_n16_c0_0.07"
    base_config: str | None = None
    spectral_nx: int | None = None
    spectral_ny: int | None = None
    high_wavenumber_fraction: float = 2.0 / 3.0
    write_vtk: bool = False
    format_version: int = RESOLVED_FORMAT_VERSION

    def __post_init__(self):
        if self.case not in CANDIDATE_CASES:
            raise ValueError(
                f"unsupported resolved pilot case {self.case!r}; "
                f"choose one of {tuple(CANDIDATE_CASES)}"
            )
        _positive_integer("nx", self.nx)
        _positive_integer("ny", self.ny)
        _finite_float("dt", self.dt, positive=True)
        _positive_integer("nsteps", self.nsteps)
        stride = _positive_integer("output_stride", self.output_stride)
        if stride > self.nsteps:
            raise ValueError("output_stride cannot exceed nsteps")
        _finite_float("c0", self.c0, positive=True)
        _finite_float("s", self.s, positive=True)
        if self.moist_backend not in ("ufl", "jax"):
            raise ValueError("moist_backend must be 'ufl' or 'jax'")
        _nonnegative_integer("seed", self.seed)
        if not str(self.output_directory):
            raise ValueError("output_directory must be nonempty")
        for name in ("spectral_nx", "spectral_ny"):
            value = getattr(self, name)
            if value is not None and _positive_integer(name, value) < 4:
                raise ValueError(f"{name} must be at least four")
        high = _finite_float(
            "high_wavenumber_fraction", self.high_wavenumber_fraction,
            positive=True,
        )
        if high >= 1.0:
            raise ValueError("high_wavenumber_fraction must be below one")
        if int(self.format_version) != RESOLVED_FORMAT_VERSION:
            raise ValueError("unsupported resolved pilot format version")

    @property
    def final_time(self) -> float:
        return float(self.dt) * int(self.nsteps)

    @property
    def output_steps(self) -> tuple[int, ...]:
        steps = list(range(0, int(self.nsteps) + 1, int(self.output_stride)))
        if steps[-1] != int(self.nsteps):
            steps.append(int(self.nsteps))
        return tuple(steps)

    @property
    def sampling_shape(self) -> tuple[int, int]:
        return (
            int(self.spectral_nx or 2 * int(self.nx)),
            int(self.spectral_ny or 2 * int(self.ny)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def physics_configuration(self, *, include_c0: bool = True) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("output_directory")
        result.pop("write_vtk")
        if not include_c0:
            result.pop("c0")
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]):
        return cls(**dict(values))


def paired_pilot_configurations(
    base: ResolvedPilotConfiguration,
    c0_a: float = 0.07,
    c0_b: float = 0.14,
    *,
    parent_directory: str | Path | None = None,
) -> tuple[ResolvedPilotConfiguration, ResolvedPilotConfiguration]:
    """Return a pair whose physical setup differs only in physical ``c0``."""
    if not isinstance(base, ResolvedPilotConfiguration):
        raise TypeError("base must be a ResolvedPilotConfiguration")
    left_c0 = _finite_float("c0_a", c0_a, positive=True)
    right_c0 = _finite_float("c0_b", c0_b, positive=True)
    if left_c0 == right_c0:
        raise ValueError("paired c0 values must differ")
    parent = Path(parent_directory or base.output_directory).resolve()
    stem = f"{base.case}_n{base.nx}x{base.ny}"
    left = replace(
        base,
        c0=left_c0,
        output_directory=str(parent / f"{stem}_c0_{left_c0:.8g}"),
    )
    right = replace(
        base,
        c0=right_c0,
        output_directory=str(parent / f"{stem}_c0_{right_c0:.8g}"),
    )
    if left.physics_configuration(include_c0=False) != right.physics_configuration(
        include_c0=False
    ):
        raise AssertionError("paired pilot construction changed non-c0 physics")
    return left, right


def normalized_separation(left, right, *, weights=None) -> float:
    """Return ``||left-right|| / ||right||`` for synthetic diagnostics.

    Production mixed-state separation is assembled with the Firedrake mixed
    mass measure by the analysis adapter; this helper intentionally does not
    pretend coefficient-vector Euclidean norm is a finite-element mass norm.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("separation arrays must have one equal nonempty shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise FloatingPointError("separation arrays must be finite")
    residual = a - b
    if weights is None:
        numerator = float(np.vdot(residual, residual).real)
        denominator = float(np.vdot(b, b).real)
    else:
        mass = np.asarray(weights, dtype=np.float64)
        if mass.ndim == 1:
            if mass.shape != (a.size,):
                raise ValueError("diagonal weights have the wrong size")
            flat_residual = residual.reshape(-1)
            flat_b = b.reshape(-1)
            numerator = float(np.dot(mass, flat_residual * flat_residual))
            denominator = float(np.dot(mass, flat_b * flat_b))
        elif mass.ndim == 2:
            if mass.shape != (a.size, a.size):
                raise ValueError("mass matrix has the wrong shape")
            flat_residual = residual.reshape(-1)
            flat_b = b.reshape(-1)
            numerator = float(flat_residual @ mass @ flat_residual)
            denominator = float(flat_b @ mass @ flat_b)
        else:
            raise ValueError("weights must be a diagonal or dense mass matrix")
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def fieldwise_normalized_separation(
    left,
    right,
    field_slices: Mapping[str, tuple[int, int] | list[int]],
) -> dict[str, float]:
    """Coefficient-space reference calculation used only by cheap tests."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("state arrays have different shapes")
    result = {}
    for name in STATE_FIELDS:
        if name not in field_slices:
            raise ValueError(f"missing field slice for {name}")
        start, stop = (int(value) for value in field_slices[name])
        if not 0 <= start < stop <= a.size:
            raise ValueError(f"invalid field slice for {name}")
        result[name] = normalized_separation(a[start:stop], b[start:stop])
    return result


@dataclass(frozen=True)
class VectorSpectrum:
    """Shell-averaged spectrum of a uniformly sampled periodic vector field."""

    shell: np.ndarray
    shell_mode_count: np.ndarray
    shell_energy_sum: np.ndarray
    shell_energy_mean: np.ndarray
    physical_shell_wavenumber: np.ndarray
    high_wavenumber_fraction: float
    parseval_mean_kinetic_energy: float

    def __post_init__(self):
        for name in (
            "shell",
            "shell_mode_count",
            "shell_energy_sum",
            "shell_energy_mean",
            "physical_shell_wavenumber",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def shell_averaged_vector_spectrum(
    velocity,
    *,
    lx: float,
    ly: float,
    high_wavenumber_fraction: float = 2.0 / 3.0,
) -> VectorSpectrum:
    """FFT a physical vector field sampled on a uniform periodic grid.

    The input shape is ``(ny, nx, ncomponents)``.  ``fft2(..., norm='forward')``
    makes the sum of modal ``0.5*|v_hat|^2`` equal the grid-mean kinetic
    energy by Parseval.  Integer cycle modes define radial shells rounded to
    the nearest integer.  Both shell sums and shell means are retained.
    """
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError("velocity must have shape (ny, nx, ncomponents)")
    if min(values.shape[:2]) < 4 or not np.all(np.isfinite(values)):
        raise ValueError("velocity samples must be finite on at least a 4x4 grid")
    domain_x = _finite_float("lx", lx, positive=True)
    domain_y = _finite_float("ly", ly, positive=True)
    high = _finite_float(
        "high_wavenumber_fraction", high_wavenumber_fraction, positive=True
    )
    if high >= 1.0:
        raise ValueError("high_wavenumber_fraction must be below one")
    ny, nx, _ = values.shape
    transformed = np.fft.fft2(values, axes=(0, 1), norm="forward")
    modal_energy = 0.5 * np.sum(np.abs(transformed) ** 2, axis=2)
    mode_x = np.fft.fftfreq(nx, d=1.0 / nx)
    mode_y = np.fft.fftfreq(ny, d=1.0 / ny)
    kx, ky = np.meshgrid(mode_x, mode_y)
    radius = np.sqrt(kx * kx + ky * ky)
    shell_index = np.floor(radius + 0.5).astype(np.int64)
    max_shell = int(shell_index.max())
    counts = np.bincount(shell_index.ravel(), minlength=max_shell + 1)
    sums = np.bincount(
        shell_index.ravel(), weights=modal_energy.ravel(), minlength=max_shell + 1
    )
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    physical_radius = np.sqrt(
        (2.0 * np.pi * kx / domain_x) ** 2
        + (2.0 * np.pi * ky / domain_y) ** 2
    )
    physical_shell = np.divide(
        np.bincount(
            shell_index.ravel(),
            weights=physical_radius.ravel(),
            minlength=max_shell + 1,
        ),
        counts,
        out=np.zeros(max_shell + 1, dtype=np.float64),
        where=counts > 0,
    )
    nonzero_radius = radius[radius > 0.0]
    threshold = high * float(nonzero_radius.max())
    total = float(np.sum(modal_energy))
    high_energy = float(np.sum(modal_energy[radius >= threshold]))
    grid_mean = float(0.5 * np.mean(np.sum(values * values, axis=2)))
    if not np.isclose(total, grid_mean, rtol=2.0e-13, atol=2.0e-15):
        raise FloatingPointError("FFT normalization failed its Parseval check")
    return VectorSpectrum(
        shell=np.arange(max_shell + 1, dtype=np.int64),
        shell_mode_count=counts.astype(np.int64),
        shell_energy_sum=sums,
        shell_energy_mean=means,
        physical_shell_wavenumber=physical_shell,
        high_wavenumber_fraction=(
            high_energy / max(total, np.finfo(float).tiny)
        ),
        parseval_mean_kinetic_energy=grid_mean,
    )


@dataclass(frozen=True)
class LateTimeGrowthConfiguration:
    """Transparent thresholds for non-conclusive pilot growth warnings.

    The first ``baseline_fraction`` of saved samples supplies a baseline
    median; the final ``tail_fraction`` supplies a tail median.  A metric is
    flagged when the tail-to-baseline ratio reaches its configured factor.
    These data-dependent warnings are intentionally separate from both the
    finite-state check and the exact linear Euler-child spectral audit.
    """

    baseline_fraction: float = 0.5
    tail_fraction: float = 0.25
    kinetic_energy_factor: float = 1.25
    projected_enstrophy_factor: float = 2.0
    hyperviscosity_tendency_factor: float = 10.0
    high_wavenumber_fraction_factor: float = 10.0
    absolute_floor: float = 1.0e-30
    high_wavenumber_absolute_floor: float = 1.0e-12

    def __post_init__(self):
        for name in ("baseline_fraction", "tail_fraction"):
            value = _finite_float(name, getattr(self, name), positive=True)
            if value > 0.5:
                raise ValueError(f"{name} must not exceed one half")
        for name in (
            "kinetic_energy_factor",
            "projected_enstrophy_factor",
            "hyperviscosity_tendency_factor",
            "high_wavenumber_fraction_factor",
        ):
            if _finite_float(name, getattr(self, name), positive=True) <= 1.0:
                raise ValueError(f"{name} must exceed one")
        _finite_float("absolute_floor", self.absolute_floor, positive=True)
        _finite_float(
            "high_wavenumber_absolute_floor",
            self.high_wavenumber_absolute_floor,
            positive=True,
        )


def late_time_growth_indicator(
    times,
    values,
    *,
    growth_factor: float,
    baseline_fraction: float = 0.5,
    tail_fraction: float = 0.25,
    absolute_floor: float = 1.0e-30,
) -> dict[str, Any]:
    """Return an explicit heuristic warning, never a stability proof.

    Inputs are copied into float64 arrays and never mutated.  The indicator
    requires at least four strictly time-ordered, finite, nonnegative samples.
    """
    sample_times = np.asarray(times, dtype=np.float64).reshape(-1)
    history = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample_times.shape != history.shape or history.size < 4:
        raise ValueError("growth histories need at least four matching samples")
    if not np.all(np.isfinite(sample_times)) or not np.all(np.isfinite(history)):
        raise FloatingPointError("growth histories must be finite")
    if np.any(np.diff(sample_times) <= 0.0):
        raise ValueError("growth sample times must be strictly increasing")
    if np.any(history < 0.0):
        raise ValueError("growth metrics must be nonnegative")
    factor = _finite_float("growth_factor", growth_factor, positive=True)
    if factor <= 1.0:
        raise ValueError("growth_factor must exceed one")
    baseline_part = _finite_float(
        "baseline_fraction", baseline_fraction, positive=True
    )
    tail_part = _finite_float("tail_fraction", tail_fraction, positive=True)
    if baseline_part > 0.5 or tail_part > 0.5:
        raise ValueError("growth window fractions must not exceed one half")
    floor = _finite_float("absolute_floor", absolute_floor, positive=True)
    baseline_count = max(2, int(np.ceil(history.size * baseline_part)))
    tail_count = max(2, int(np.ceil(history.size * tail_part)))
    baseline = history[:baseline_count]
    tail = history[-tail_count:]
    baseline_median = float(np.median(baseline))
    tail_median = float(np.median(tail))
    reference = max(baseline_median, floor)
    ratio = tail_median / reference
    centred_time = sample_times[-tail_count:] - sample_times[-tail_count]
    slope = float(np.polyfit(centred_time, tail, 1)[0])
    normalized_tail_change = float(
        slope * centred_time[-1] / max(tail_median, floor)
    )
    suspicious = bool(tail_median >= factor * reference)
    return {
        "kind": "late-time growth heuristic; not a proof of instability",
        "sample_count": int(history.size),
        "baseline_sample_count": baseline_count,
        "tail_sample_count": tail_count,
        "growth_factor": factor,
        "absolute_floor": floor,
        "baseline_median": baseline_median,
        "tail_median": tail_median,
        "tail_maximum": float(np.max(tail)),
        "tail_to_baseline_ratio": float(ratio),
        "tail_slope_per_unit_time": slope,
        "normalized_tail_change": normalized_tail_change,
        "suspicious_late_time_growth": suspicious,
    }


class RolloutLoss(str, Enum):
    TERMINAL = "terminal"
    ACCUMULATED = "accumulated"

    def as_framework_accumulation(self) -> LossAccumulation:
        return LossAccumulation(self.value)


class ScanDerivativeLevel(str, Enum):
    """Explicit derivative work requested for a scalar landscape scan."""

    OBJECTIVE_ONLY = "objective_only"
    OBJECTIVE_GRADIENT = "objective_gradient"
    OBJECTIVE_GRADIENT_HESSIAN = "objective_gradient_hessian"

    @property
    def includes_gradient(self) -> bool:
        return self is not ScanDerivativeLevel.OBJECTIVE_ONLY

    @property
    def includes_hessian(self) -> bool:
        return self is ScanDerivativeLevel.OBJECTIVE_GRADIENT_HESSIAN


class SolverLossNormalization(str, Enum):
    """Fixed normalizer used by resolved solver-in-loop observations.

    ``INITIAL_GUESS_RESIDUAL`` preserves the generic/Test-1A convention in
    which each start/prefix pair is normalized by its own initial-guess
    prediction error.  ``TRUTH_TARGET_MASS`` gives reset and rollout the same
    normalizer for a shared target state and is the selected Test-1B fairness
    convention.
    """

    INITIAL_GUESS_RESIDUAL = "initial_guess_residual"
    TRUTH_TARGET_MASS = "truth_target_mass"


@dataclass(frozen=True)
class ResolvedInferenceConfiguration:
    """Selection-neutral Test-1B indexing and loss configuration.

    Transition intervals are half open.  The truth state at
    ``training_stop_step`` is shared only as the initial state of the held-out
    deployment; no held-out transition participates in fitting.
    """

    truth_run_directory: str
    c0_truth: float = 0.14
    c0_initial: float = 0.07
    training_start_step: int = 0
    training_stop_step: int = 10
    heldout_stop_step: int = 20
    observation_stride: int = 1
    truth_reset_horizon: int = 1
    truth_reset_window_stride: int | None = None
    rollout_horizon: int = 5
    truth_reset_loss: RolloutLoss = RolloutLoss.TERMINAL
    rollout_loss: RolloutLoss = RolloutLoss.ACCUMULATED
    solver_loss_normalization: SolverLossNormalization = (
        SolverLossNormalization.INITIAL_GUESS_RESIDUAL
    )
    c0_scale: float = C0_SCALE

    def __post_init__(self):
        if not str(self.truth_run_directory):
            raise ValueError("truth_run_directory must be nonempty")
        truth = _finite_float("c0_truth", self.c0_truth, positive=True)
        initial = _finite_float("c0_initial", self.c0_initial, positive=True)
        if truth == initial:
            raise ValueError("truth and initial c0 must differ")
        start = _nonnegative_integer(
            "training_start_step", self.training_start_step
        )
        stop = _positive_integer("training_stop_step", self.training_stop_step)
        heldout = _positive_integer("heldout_stop_step", self.heldout_stop_step)
        if not start < stop < heldout:
            raise ValueError(
                "require training_start_step < training_stop_step < "
                "heldout_stop_step"
            )
        stride = _positive_integer("observation_stride", self.observation_stride)
        if (stop - start) % stride or (heldout - stop) % stride:
            raise ValueError(
                "training and held-out interval endpoints must lie on the "
                "configured observation cadence"
            )
        reset = _positive_integer("truth_reset_horizon", self.truth_reset_horizon)
        window_stride = (
            stride
            if self.truth_reset_window_stride is None
            else _positive_integer(
                "truth_reset_window_stride", self.truth_reset_window_stride
            )
        )
        rollout = _positive_integer("rollout_horizon", self.rollout_horizon)
        if reset > stop - start or rollout > stop - start:
            raise ValueError("solver-in-loop horizons must fit in training interval")
        if reset % stride or rollout % stride:
            raise ValueError(
                "each horizon endpoint must lie on the configured observation "
                "cadence (horizon must be divisible by observation_stride)"
            )
        if window_stride % stride:
            raise ValueError(
                "truth_reset_window_stride must be divisible by "
                "observation_stride"
            )
        object.__setattr__(self, "truth_reset_window_stride", window_stride)
        object.__setattr__(
            self, "truth_reset_loss", RolloutLoss(self.truth_reset_loss)
        )
        object.__setattr__(self, "rollout_loss", RolloutLoss(self.rollout_loss))
        object.__setattr__(
            self,
            "solver_loss_normalization",
            SolverLossNormalization(self.solver_loss_normalization),
        )
        scale = _finite_float("c0_scale", self.c0_scale, positive=True)
        if scale != C0_SCALE:
            raise ValueError("the certified c0 = 0.07 z scaling is immutable")

    @property
    def training_transition_steps(self) -> tuple[int, ...]:
        return tuple(range(self.training_start_step, self.training_stop_step))

    @property
    def heldout_transition_steps(self) -> tuple[int, ...]:
        return tuple(range(self.training_stop_step, self.heldout_stop_step))

    @property
    def training_observation_steps(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.training_start_step,
                self.training_stop_step + 1,
                self.observation_stride,
            )
        )

    @property
    def heldout_observation_steps(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.training_stop_step,
                self.heldout_stop_step + 1,
                self.observation_stride,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["truth_reset_loss"] = self.truth_reset_loss.value
        result["rollout_loss"] = self.rollout_loss.value
        result["solver_loss_normalization"] = self.solver_loss_normalization.value
        return result


@dataclass(frozen=True)
class InferenceIndexPlan:
    training_observations: tuple[int, ...]
    heldout_observations: tuple[int, ...]
    offline_transition_starts: tuple[int, ...]
    truth_reset_windows: tuple[tuple[int, int], ...]
    truth_reset_target_steps: tuple[int, ...]
    rollout_start_step: int
    rollout_prefixes: tuple[int, ...]
    rollout_target_steps: tuple[int, ...]


def build_inference_index_plan(
    configuration: ResolvedInferenceConfiguration,
) -> InferenceIndexPlan:
    """Construct explicit reset windows and autonomous rollout observations."""
    if not isinstance(configuration, ResolvedInferenceConfiguration):
        raise TypeError("configuration must be ResolvedInferenceConfiguration")
    observations = configuration.training_observation_steps
    observation_set = set(observations)
    latest_start = (
        configuration.training_stop_step - configuration.truth_reset_horizon
    )
    window_starts = range(
        configuration.training_start_step,
        latest_start + 1,
        configuration.truth_reset_window_stride,
    )
    windows = tuple(
        (start, start + configuration.truth_reset_horizon)
        for start in window_starts
        if start in observation_set
        and start + configuration.truth_reset_horizon in observation_set
    )
    if not windows:
        raise ValueError("configuration produces no truth-reset window")
    if configuration.rollout_loss is RolloutLoss.TERMINAL:
        prefixes = (configuration.rollout_horizon,)
    else:
        prefixes = tuple(
            range(
                configuration.observation_stride,
                configuration.rollout_horizon + 1,
                configuration.observation_stride,
            )
        )
    if configuration.truth_reset_loss is RolloutLoss.TERMINAL:
        reset_prefixes = (configuration.truth_reset_horizon,)
    else:
        reset_prefixes = tuple(
            range(
                configuration.observation_stride,
                configuration.truth_reset_horizon + 1,
                configuration.observation_stride,
            )
        )
    reset_targets = tuple(
        start + prefix for start, _ in windows for prefix in reset_prefixes
    )
    rollout_start = configuration.training_start_step
    return InferenceIndexPlan(
        training_observations=observations,
        heldout_observations=configuration.heldout_observation_steps,
        offline_transition_starts=observations[:-1],
        truth_reset_windows=windows,
        truth_reset_target_steps=reset_targets,
        rollout_start_step=rollout_start,
        rollout_prefixes=prefixes,
        rollout_target_steps=tuple(rollout_start + prefix for prefix in prefixes),
    )


def resolved_truth_state_indices(
    configuration: ResolvedInferenceConfiguration,
    *,
    include_heldout: bool = False,
) -> tuple[int, ...]:
    """Return the only truth states a fitting or evaluation command may load."""
    if not isinstance(configuration, ResolvedInferenceConfiguration):
        raise TypeError("configuration must be ResolvedInferenceConfiguration")
    if not isinstance(include_heldout, bool):
        raise TypeError("include_heldout must be bool")
    final_step = (
        configuration.heldout_stop_step
        if include_heldout
        else configuration.training_stop_step
    )
    return tuple(range(configuration.training_start_step, final_step + 1))


@dataclass(frozen=True)
class ObjectiveScanConfiguration:
    derivative_level: ScanDerivativeLevel
    physical_lower: float = 0.03
    physical_upper: float = 0.20
    points: int = 18

    def __post_init__(self):
        lower = _finite_float("physical_lower", self.physical_lower, positive=True)
        upper = _finite_float("physical_upper", self.physical_upper, positive=True)
        if lower >= upper:
            raise ValueError("objective scan interval is empty")
        if _positive_integer("points", self.points) < 3:
            raise ValueError("objective scan requires at least three points")
        object.__setattr__(
            self, "derivative_level", ScanDerivativeLevel(self.derivative_level)
        )

    @property
    def include_gradient(self) -> bool:
        return self.derivative_level.includes_gradient

    @property
    def include_hessian(self) -> bool:
        return self.derivative_level.includes_hessian

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["derivative_level"] = self.derivative_level.value
        return result

    @property
    def physical_values(self) -> np.ndarray:
        result = np.linspace(
            self.physical_lower, self.physical_upper, self.points, dtype=np.float64
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class LandscapePoint:
    physical_c0: float
    normalized_z: float
    objective: float | None
    physical_gradient: float | None
    physical_hessian: float | None
    finite: bool
    objective_evaluations: int
    gradient_evaluations: int
    hvp_evaluations: int
    solver_calls: int
    forward_steps: int
    reverse_steps: int
    tangent_steps: int
    incremental_reverse_steps: int
    wall_time_seconds: float
    failure_reason: str | None


def _objective_trajectory_work(objective):
    if hasattr(objective, "work_counts"):
        work = objective.work_counts()
        return (
            int(work.forward_steps),
            int(work.reverse_steps),
            int(work.tangent_steps),
            int(work.incremental_reverse_steps),
        )
    counts = objective.counts()
    return (int(counts.solver_calls), 0, 0, 0)


def scan_scalar_objective(
    objective,
    configuration,
    *,
    completed_points=(),
    completed_configuration=None,
    point_callback=None,
):
    """Record J, dJ/dc0, d2J/dc0^2, cost, and failures point by point.

    This routine is prepared for external resolved runs and is not invoked by
    import or pytest.  The objective itself owns all complete-solver counts.
    """
    from time import perf_counter

    if not isinstance(configuration, ObjectiveScanConfiguration):
        raise TypeError("configuration must be ObjectiveScanConfiguration")
    if completed_configuration is not None:
        if dict(completed_configuration) != configuration.to_dict():
            raise ValueError(
                "completed landscape points use an incompatible scan policy"
            )
    elif completed_points:
        raise ValueError(
            "completed landscape points require their scan configuration"
        )
    scale = _finite_float("objective.c0_scale", objective.c0_scale, positive=True)
    records = []
    completed = {
        float(
            point.physical_c0
            if isinstance(point, LandscapePoint)
            else point["physical_c0"]
        ): (
            point
            if isinstance(point, LandscapePoint)
            else LandscapePoint(**dict(point))
        )
        for point in completed_points
    }
    for physical in configuration.physical_values:
        if float(physical) in completed:
            records.append(completed[float(physical)])
            continue
        before = objective.counts()
        before_work = _objective_trajectory_work(objective)
        started = perf_counter()
        value = gradient = hessian = None
        failure = None
        try:
            z = float(physical / scale)
            if configuration.include_gradient:
                value, normalized_gradient = objective.value_and_gradient(z)
                gradient = float(normalized_gradient / scale)
            else:
                value = objective.value(z)
            if configuration.include_hessian:
                normalized_hessian = objective.hess_vec(z, 1.0)
                hessian = float(normalized_hessian / (scale * scale))
        except Exception as exc:  # landscape diagnostics must retain failures
            failure = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter() - started
        after = objective.counts()
        after_work = _objective_trajectory_work(objective)
        numeric = tuple(x for x in (value, gradient, hessian) if x is not None)
        finite = failure is None and all(np.isfinite(x) for x in numeric)
        record = LandscapePoint(
            physical_c0=float(physical),
            normalized_z=float(physical / scale),
            objective=None if value is None else float(value),
            physical_gradient=gradient,
            physical_hessian=hessian,
            finite=bool(finite),
            objective_evaluations=(
                after.objective_evaluations - before.objective_evaluations
            ),
            gradient_evaluations=(
                after.gradient_evaluations - before.gradient_evaluations
            ),
            hvp_evaluations=after.hvp_evaluations - before.hvp_evaluations,
            solver_calls=after.solver_calls - before.solver_calls,
            forward_steps=after_work[0] - before_work[0],
            reverse_steps=after_work[1] - before_work[1],
            tangent_steps=after_work[2] - before_work[2],
            incremental_reverse_steps=after_work[3] - before_work[3],
            wall_time_seconds=float(elapsed),
            failure_reason=failure,
        )
        records.append(record)
        if point_callback is not None:
            point_callback(record, tuple(records))
    return tuple(records)


COMMON_EVALUATION_METRICS = (
    "relative_c0_error",
    "one_step_state_error",
    "training_autonomous_trajectory_error",
    "heldout_autonomous_trajectory_error",
    "final_state_error",
    "accumulated_trajectory_error",
    "fieldwise_errors",
    "kinetic_energy_history_mismatch",
    "projected_enstrophy_history_mismatch",
    "high_wavenumber_mismatch",
    "hyperviscosity_diagnostic_mismatch",
    "finite_state_status",
    "numerical_stability_heuristics",
    "objectives_under_all_training_modes",
    "objective_evaluations",
    "gradient_evaluations",
    "hvp_evaluations",
    "solver_calls",
    "trajectory_traversal_step_counts",
    "wall_time_seconds",
)


def write_json_record(path: str | Path, value: Any) -> Path:
    """Atomically write canonical JSON for incremental experiment records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(destination)
    return destination


def read_json_record(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = (
    "C0_SCALE",
    "CANDIDATE_CASES",
    "COMMON_EVALUATION_METRICS",
    "CandidateCase",
    "InferenceIndexPlan",
    "LandscapePoint",
    "LateTimeGrowthConfiguration",
    "ObjectiveScanConfiguration",
    "RESOLVED_FORMAT_VERSION",
    "ResolvedInferenceConfiguration",
    "ResolvedPilotConfiguration",
    "RolloutLoss",
    "ScanDerivativeLevel",
    "SolverLossNormalization",
    "STATE_FIELDS",
    "VectorSpectrum",
    "build_inference_index_plan",
    "fieldwise_normalized_separation",
    "late_time_growth_indicator",
    "normalized_separation",
    "paired_pilot_configurations",
    "read_json_record",
    "resolved_truth_state_indices",
    "scan_scalar_objective",
    "shell_averaged_vector_spectrum",
    "write_json_record",
)
