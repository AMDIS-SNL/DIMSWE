"""Prepared Test-1B inference adapter for a user-selected resolved truth run.

Nothing in this module runs at import time.  External execution is authorized
only after Test 1B-0 evidence has selected a case, mesh, timestep, duration,
and output cadence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .cached_accumulated_objective import (
    AccumulatedTrajectoryTarget,
    AccumulatedTrajectoryWindow,
    CachedAccumulatedC0Objective,
)
from .hidden_c0 import (
    HiddenC0ObjectiveSuite,
    OfflineC0Objective,
    OfflineObservation,
    ScalarOptimizerConfiguration,
    SolverInLoopC0Objective,
    SolverObservation,
    _advance,
    _baseline_normalizer,
    _copy_function,
    _flat_values,
    _state_relative_error,
    _state_squared_difference,
    optimize_hidden_c0,
)
from .learned_physics.objectives import TrainingMode
from .resolved_hidden_c0 import (
    COMMON_EVALUATION_METRICS,
    LateTimeGrowthConfiguration,
    ObjectiveScanConfiguration,
    ResolvedInferenceConfiguration,
    ResolvedPilotConfiguration,
    RolloutLoss,
    ScanDerivativeLevel,
    SolverLossNormalization,
    STATE_FIELDS,
    build_inference_index_plan,
    late_time_growth_indicator,
    read_json_record,
    resolved_truth_state_indices,
    scan_scalar_objective,
    write_json_record,
)
from .resolved_hidden_c0_driver import (
    ResolvedDiagnosticEvaluator,
    _configuration_fingerprint,
    build_resolved_hidden_c0_case,
)
from .selected_test1b import load_selected_test1b_plan


GATE4_RELATIVE_ERROR_TOLERANCE = 1.0e-10
GATE4_RELATIVE_C0_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ResolvedTruthTrajectory:
    metadata: dict
    states: object

    def __post_init__(self):
        owned = {
            int(step): _copy_function(state, f"resolved_truth_owned_{step}")
            for step, state in dict(self.states).items()
        }
        object.__setattr__(self, "states", MappingProxyType(owned))


def load_resolved_truth(
    configuration: ResolvedInferenceConfiguration,
    *,
    include_heldout: bool = False,
):
    """Load training states, and held-out states only for evaluation."""
    run_directory = Path(configuration.truth_run_directory).resolve()
    metadata = read_json_record(run_directory / "metadata.json")
    if metadata.get("status") != "complete":
        raise RuntimeError("selected resolved truth run is incomplete")
    if float(metadata["physical_parameters"]["c0"]) != configuration.c0_truth:
        raise ValueError("inference c0_truth disagrees with selected truth metadata")
    pilot = ResolvedPilotConfiguration.from_dict(metadata["configuration"])
    case = build_resolved_hidden_c0_case(pilot)
    expected_fingerprint = _configuration_fingerprint(
        pilot.physics_configuration(include_c0=False)
    )
    if metadata.get("paired_non_c0_physics_sha256") != expected_fingerprint:
        raise ValueError("truth non-c0 physics fingerprint is invalid")
    states = {}
    for step in resolved_truth_state_indices(
        configuration, include_heldout=include_heldout
    ):
        path = run_directory / "restart" / f"step_{step:08d}.npy"
        if not path.exists():
            raise RuntimeError(
                "selected truth lacks model-step snapshot "
                f"{step}; production Test 1B truth must use output_stride=1"
            )
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if values.shape != (sum(case.field_sizes),) or not np.all(
            np.isfinite(values)
        ):
            raise FloatingPointError(f"invalid selected truth snapshot {path}")
        states[step] = case.state_from_values(values, f"resolved_truth_{step}")
    return case, ResolvedTruthTrajectory(metadata=metadata, states=states)


def _fixed_offline_objectives(case, trajectory, configuration):
    weak = []
    updates = []
    indices = configuration.training_observation_steps[:-1]
    with case.physical_c0(configuration.c0_truth):
        for step in indices:
            state = trajectory.states[step]
            cache = case.helper.hyper_helper.take_forward_step_cached(
                state, case.t0 + step * case.dt, case.dt
            )
            dual = case.helper.hyper_helper.state_mass_map(
                cache.tendency, f"resolved_operator_target_{step}"
            )
            weak.append(
                OfflineObservation(
                    target=_flat_values(dual),
                    semantics=(
                        "assembled production weak hyperviscosity RHS at a fixed "
                        "truth state, before M^-1 and dt"
                    ),
                )
            )
            updates.append(
                OfflineObservation(
                    target=_flat_values(cache.state_out) - _flat_values(state),
                    semantics=(
                        "actual deployed hyperviscosity Euler-child update "
                        "dt*M^-1*b at a fixed truth state"
                    ),
                )
            )
    return (
        OfflineC0Objective(
            TrainingMode.APRIORI_OFFLINE,
            case.c0_scale,
            configuration.c0_truth,
            weak,
        ),
        OfflineC0Objective(
            TrainingMode.DISCRETE_OFFLINE,
            case.c0_scale,
            configuration.c0_truth,
            updates,
        ),
        len(indices),
    )


def _observation(
    case,
    trajectory,
    configuration,
    start,
    prefix,
    name,
    window_index,
    normalizer=None,
):
    target_step = start + prefix
    normalizer = _solver_observation_normalizer(
        case,
        trajectory,
        configuration,
        start,
        prefix,
        name,
        normalizer,
    )
    return SolverObservation(
        initial_state=trajectory.states[start],
        target=trajectory.states[target_step],
        start_time=case.t0 + start * case.dt,
        nsteps=prefix,
        normalizer=normalizer,
        window_index=window_index,
        target_step=target_step,
    )


def _solver_observation_normalizer(
    case,
    trajectory,
    configuration,
    start,
    prefix,
    name,
    normalizer=None,
):
    if normalizer is not None:
        return normalizer
    return _baseline_normalizer(
        case,
        trajectory.states[start],
        trajectory.states[start + prefix],
        case.t0 + start * case.dt,
        prefix,
        configuration.c0_initial,
        name,
    )


def _truth_target_mass_normalizers(case, trajectory, target_steps):
    """Return one immutable mixed-mass normalizer per truth target state."""
    zero = case.new_state("resolved_solver_loss_normalizer_zero")
    zero.assign(0)
    normalizers = {}
    for step in sorted(set(target_steps)):
        squared = _state_squared_difference(
            case,
            trajectory.states[step],
            zero,
            f"resolved_solver_loss_target_mass_{step}",
        )
        if not np.isfinite(squared) or squared <= np.finfo(np.float64).tiny:
            raise RuntimeError(
                f"truth target state {step} has no positive finite mixed-mass norm"
            )
        normalizers[step] = squared
    return MappingProxyType(normalizers)


def prepare_resolved_hidden_c0_objectives(
    case,
    trajectory,
    configuration: ResolvedInferenceConfiguration,
):
    """Build all four exact objective families from training states only."""
    plan = build_inference_index_plan(configuration)
    operator, deployed, hyper_calls = _fixed_offline_objectives(
        case, trajectory, configuration
    )
    shared_normalizers = None
    if (
        configuration.solver_loss_normalization
        is SolverLossNormalization.TRUTH_TARGET_MASS
    ):
        shared_normalizers = _truth_target_mass_normalizers(
            case,
            trajectory,
            plan.truth_reset_target_steps + plan.rollout_target_steps,
        )
    reset_prefixes = (
        (configuration.truth_reset_horizon,)
        if configuration.truth_reset_loss is RolloutLoss.TERMINAL
        else tuple(
            range(
                configuration.observation_stride,
                configuration.truth_reset_horizon + 1,
                configuration.observation_stride,
            )
        )
    )
    reset_observations = []
    reset_cached_windows = []
    reset_calls = 0
    for window_index, (start, _) in enumerate(plan.truth_reset_windows):
        cached_targets = []
        for prefix in reset_prefixes:
            normalizer = (
                None
                if shared_normalizers is None
                else shared_normalizers[start + prefix]
            )
            normalizer = _solver_observation_normalizer(
                case,
                trajectory,
                configuration,
                start,
                prefix,
                f"resolved_truth_reset_{start}_{prefix}",
                normalizer,
            )
            if configuration.truth_reset_loss is RolloutLoss.TERMINAL:
                reset_observations.append(
                    _observation(
                        case,
                        trajectory,
                        configuration,
                        start,
                        prefix,
                        f"resolved_truth_reset_{start}_{prefix}",
                        window_index,
                        normalizer,
                    )
                )
            cached_targets.append(
                AccumulatedTrajectoryTarget(
                    offset=prefix,
                    target=trajectory.states[start + prefix],
                    normalizer=normalizer,
                    target_step=start + prefix,
                )
            )
            if shared_normalizers is None:
                reset_calls += prefix
        reset_cached_windows.append(
            AccumulatedTrajectoryWindow(
                initial_state=trajectory.states[start],
                start_time=case.t0 + start * case.dt,
                targets=tuple(cached_targets),
            )
        )
    if configuration.truth_reset_loss is RolloutLoss.ACCUMULATED:
        reset = CachedAccumulatedC0Objective(
            TrainingMode.TRUTH_RESET, case, reset_cached_windows
        )
    else:
        reset = SolverInLoopC0Objective(
            TrainingMode.TRUTH_RESET,
            case,
            reset_observations,
            configuration.truth_reset_loss.as_framework_accumulation(),
        )
    rollout_observations = []
    rollout_cached_targets = []
    rollout_calls = 0
    start = configuration.training_start_step
    for prefix in plan.rollout_prefixes:
        normalizer = (
            None
            if shared_normalizers is None
            else shared_normalizers[start + prefix]
        )
        normalizer = _solver_observation_normalizer(
            case,
            trajectory,
            configuration,
            start,
            prefix,
            f"resolved_rollout_{start}_{prefix}",
            normalizer,
        )
        if configuration.rollout_loss is RolloutLoss.TERMINAL:
            rollout_observations.append(
                _observation(
                    case,
                    trajectory,
                    configuration,
                    start,
                    prefix,
                    f"resolved_rollout_{start}_{prefix}",
                    0,
                    normalizer,
                )
            )
        rollout_cached_targets.append(
            AccumulatedTrajectoryTarget(
                offset=prefix,
                target=trajectory.states[start + prefix],
                normalizer=normalizer,
                target_step=start + prefix,
            )
        )
        if shared_normalizers is None:
            rollout_calls += prefix
    if configuration.rollout_loss is RolloutLoss.ACCUMULATED:
        rollout = CachedAccumulatedC0Objective(
            TrainingMode.ROLLOUT,
            case,
            (
                AccumulatedTrajectoryWindow(
                    initial_state=trajectory.states[start],
                    start_time=case.t0 + start * case.dt,
                    targets=tuple(rollout_cached_targets),
                ),
            ),
        )
    else:
        rollout = SolverInLoopC0Objective(
            TrainingMode.ROLLOUT,
            case,
            rollout_observations,
            configuration.rollout_loss.as_framework_accumulation(),
        )
    return HiddenC0ObjectiveSuite(
        objectives={
            TrainingMode.APRIORI_OFFLINE: operator,
            TrainingMode.DISCRETE_OFFLINE: deployed,
            TrainingMode.TRUTH_RESET: reset,
            TrainingMode.ROLLOUT: rollout,
        },
        preprocessing_hyperviscosity_child_calls=hyper_calls,
        preprocessing_complete_solver_calls=reset_calls + rollout_calls,
        truth_reset_horizon=configuration.truth_reset_horizon,
        rollout_horizon=configuration.rollout_horizon,
        rollout_accumulation=configuration.rollout_loss.as_framework_accumulation(),
    )


def _relative_error_record(squared_error, squared_reference):
    """Represent a relative norm without hiding a zero reference norm."""
    numerator = float(squared_error)
    denominator = float(squared_reference)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise FloatingPointError("relative-error norm inputs must be finite")
    if numerator < 0.0 or denominator < 0.0:
        raise ValueError("relative-error squared norms must be nonnegative")
    absolute_error = float(np.sqrt(numerator))
    reference_norm = float(np.sqrt(denominator))
    reference_zero = denominator == 0.0
    if reference_zero:
        relative_error = 0.0 if numerator == 0.0 else None
    else:
        relative_error = float(np.sqrt(numerator / denominator))
    return {
        "absolute_error": absolute_error,
        "reference_norm": reference_norm,
        "relative_error": relative_error,
        "reference_norm_zero": reference_zero,
        "relative_error_defined": relative_error is not None,
    }


def _optional_maximum(values):
    sequence = tuple(values)
    if not sequence or any(value is None for value in sequence):
        return None
    return float(max(sequence))


def _trajectory_metric(case, predicted, truth, steps, name):
    numerators = []
    denominators = []
    records = []
    zero = case.new_state(f"{name}_zero")
    zero.assign(0)
    for step in steps:
        prediction = predicted[step]
        target = truth.states[step]
        numerator = _state_squared_difference(
            case, prediction, target, f"{name}_residual_{step}"
        )
        denominator = _state_squared_difference(
            case, target, zero, f"{name}_target_{step}"
        )
        numerators.append(numerator)
        denominators.append(denominator)
        records.append(_relative_error_record(numerator, denominator))
    relative = tuple(record["relative_error"] for record in records)
    accumulated = _relative_error_record(sum(numerators), sum(denominators))
    step_tuple = tuple(int(step) for step in steps)
    return {
        "steps": step_tuple,
        "times": tuple(float(case.t0 + step * case.dt) for step in step_tuple),
        "relative_mass_norm_error": relative,
        "absolute_mass_norm_error": tuple(
            record["absolute_error"] for record in records
        ),
        "reference_mass_norm": tuple(
            record["reference_norm"] for record in records
        ),
        "reference_norm_zero": tuple(
            record["reference_norm_zero"] for record in records
        ),
        "all_relative_errors_defined": all(
            record["relative_error_defined"] for record in records
        ),
        # Retain the established key for consumers of prepared J4B output.
        "per_observation": relative,
        "maximum": _optional_maximum(relative),
        "final": relative[-1],
        "accumulated": accumulated["relative_error"],
        "accumulated_record": accumulated,
    }


def _field_squared_norms(case, predicted, target, field_index, name):
    from firedrake import assemble, inner

    residual = case.new_state(f"{name}_residual")
    residual.assign(predicted)
    residual.sub(field_index).assign(
        predicted.sub(field_index) - target.sub(field_index)
    )
    numerator = float(
        assemble(
            inner(residual.sub(field_index), residual.sub(field_index))
            * case.model.spaces.dx
        )
    )
    denominator = float(
        assemble(
            inner(target.sub(field_index), target.sub(field_index))
            * case.model.spaces.dx
        )
    )
    return numerator, denominator


def _field_trajectory_metric(case, predicted, truth, steps, name):
    step_tuple = tuple(int(step) for step in steps)
    times = tuple(float(case.t0 + step * case.dt) for step in step_tuple)
    fields = {}
    for index, field_name in enumerate(STATE_FIELDS):
        records = []
        for step in step_tuple:
            numerator, denominator = _field_squared_norms(
                case,
                predicted[step],
                truth.states[step],
                index,
                f"{name}_{field_name}_{step}",
            )
            records.append(_relative_error_record(numerator, denominator))
        relative = tuple(record["relative_error"] for record in records)
        fields[field_name] = {
            "steps": step_tuple,
            "times": times,
            "relative_mass_norm_error": relative,
            "reference_norm_zero": tuple(
                record["reference_norm_zero"] for record in records
            ),
            "all_relative_errors_defined": all(
                record["relative_error_defined"] for record in records
            ),
            "maximum": _optional_maximum(relative),
            "final": relative[-1],
        }
    return fields


def _diagnostic_mismatch(predicted, truth, steps, times):
    predicted_values = np.asarray(predicted, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if predicted_values.shape != truth_values.shape or predicted_values.ndim != 1:
        raise ValueError("diagnostic histories must be equal one-dimensional arrays")
    if not np.all(np.isfinite(predicted_values)) or not np.all(
        np.isfinite(truth_values)
    ):
        raise FloatingPointError("diagnostic histories must be finite")
    absolute = np.abs(predicted_values - truth_values)
    relative = tuple(
        _relative_error_record(error * error, reference * reference)[
            "relative_error"
        ]
        for error, reference in zip(absolute, truth_values)
    )
    history = _relative_error_record(
        float(np.dot(predicted_values - truth_values, predicted_values - truth_values)),
        float(np.dot(truth_values, truth_values)),
    )
    return {
        "steps": tuple(int(step) for step in steps),
        "times": tuple(float(value) for value in times),
        "truth": tuple(float(value) for value in truth_values),
        "predicted": tuple(float(value) for value in predicted_values),
        "absolute_mismatch": tuple(float(value) for value in absolute),
        "relative_mismatch": relative,
        "maximum_absolute_mismatch": float(np.max(absolute)),
        "final_absolute_mismatch": float(absolute[-1]),
        "maximum_relative_mismatch": _optional_maximum(relative),
        "final_relative_mismatch": relative[-1],
        "all_relative_mismatches_defined": all(
            value is not None for value in relative
        ),
        "relative_history_l2_mismatch": history["relative_error"],
        "history_reference_norm_zero": history["reference_norm_zero"],
    }


def _autonomous_map(case, initial, c0, start, stop, name):
    states = _advance(
        case,
        initial,
        c0,
        stop - start,
        start_time=case.t0 + start * case.dt,
        prefix=name,
    )
    return {start + index: state for index, state in enumerate(states)}


def _diagnostic_growth(times, diagnostics):
    configuration = LateTimeGrowthConfiguration()
    common = {
        "baseline_fraction": configuration.baseline_fraction,
        "tail_fraction": configuration.tail_fraction,
    }
    specifications = {
        "kinetic_energy": (
            configuration.kinetic_energy_factor,
            configuration.absolute_floor,
        ),
        "projected_enstrophy": (
            configuration.projected_enstrophy_factor,
            configuration.absolute_floor,
        ),
        "hyperviscosity_tendency_mass_norm": (
            configuration.hyperviscosity_tendency_factor,
            configuration.absolute_floor,
        ),
        "velocity_high_wavenumber_energy_fraction": (
            configuration.high_wavenumber_fraction_factor,
            configuration.high_wavenumber_absolute_floor,
        ),
    }
    return {
        key: late_time_growth_indicator(
            times,
            [record[key] for record in diagnostics],
            growth_factor=factor,
            absolute_floor=floor,
            **common,
        )
        for key, (factor, floor) in specifications.items()
    }


def _objective_work_record(objective):
    if hasattr(objective, "work_counts"):
        counts = objective.work_counts()
        return {
            "forward": int(counts.forward_steps),
            "reverse": int(counts.reverse_steps),
            "tangent": int(counts.tangent_steps),
            "incremental_reverse": int(counts.incremental_reverse_steps),
        }
    return {
        "forward": int(objective.counts().solver_calls),
        "reverse": 0,
        "tangent": 0,
        "incremental_reverse": 0,
    }


def evaluate_resolved_hidden_c0(
    case,
    trajectory,
    suite,
    configuration,
    recovered_c0,
):
    """Common train/held-out deployment evaluation for every fitted mode."""
    from time import perf_counter

    started = perf_counter()
    recovered = float(recovered_c0)
    started_counts = {mode: suite[mode].counts() for mode in TrainingMode}
    started_work = {
        mode: _objective_work_record(suite[mode]) for mode in TrainingMode
    }
    # Construct the held-out prediction before consulting any held-out target.
    # _advance recursively reuses only its predicted state after trusted X_80.
    heldout = _autonomous_map(
        case,
        trajectory.states[configuration.training_stop_step],
        recovered,
        configuration.training_stop_step,
        configuration.heldout_stop_step,
        "resolved_evaluation_heldout",
    )
    training = _autonomous_map(
        case,
        trajectory.states[configuration.training_start_step],
        recovered,
        configuration.training_start_step,
        configuration.training_stop_step,
        "resolved_evaluation_training",
    )
    training_metric = _trajectory_metric(
        case,
        training,
        trajectory,
        configuration.training_observation_steps[1:],
        "resolved_training",
    )
    heldout_metric = _trajectory_metric(
        case,
        heldout,
        trajectory,
        configuration.heldout_observation_steps[1:],
        "resolved_heldout",
    )
    fieldwise = _field_trajectory_metric(
        case,
        heldout,
        trajectory,
        configuration.heldout_observation_steps[1:],
        "resolved_evaluation_heldout",
    )
    first_step = configuration.training_start_step + 1
    one_step = _state_relative_error(
        case,
        training[first_step],
        trajectory.states[first_step],
        "resolved_evaluation_one_step",
    )
    evaluator = ResolvedDiagnosticEvaluator(
        case,
        ResolvedPilotConfiguration.from_dict(trajectory.metadata["configuration"]),
    )
    # State training_stop is the trusted initializer, not a held-out target.
    diagnostic_steps = configuration.heldout_observation_steps[1:]
    predicted_diagnostics = []
    truth_diagnostics = []
    with case.physical_c0(recovered):
        for step in diagnostic_steps:
            predicted_diagnostics.append(
                evaluator.evaluate(
                    heldout[step], step, case.t0 + step * case.dt
                )[0]
            )
    with case.physical_c0(configuration.c0_truth):
        for step in diagnostic_steps:
            truth_diagnostics.append(
                evaluator.evaluate(
                    trajectory.states[step], step, case.t0 + step * case.dt
                )[0]
            )
    cross = {}
    for mode in TrainingMode:
        objective = suite[mode]
        cross[mode.value] = {
            "initial": objective.value(configuration.c0_initial / case.c0_scale),
            "recovered": objective.value(recovered / case.c0_scale),
        }
    all_states = tuple(training.values()) + tuple(heldout.values())
    finite = all(np.all(np.isfinite(_flat_values(state))) for state in all_states)
    final_counts = {mode: suite[mode].counts() for mode in TrainingMode}
    final_work = {
        mode: _objective_work_record(suite[mode]) for mode in TrainingMode
    }
    cross_cost = {
        mode.value: {
            "objective_evaluations": (
                final_counts[mode].objective_evaluations
                - started_counts[mode].objective_evaluations
            ),
            "gradient_evaluations": 0,
            "hvp_evaluations": 0,
            "solver_calls": (
                final_counts[mode].solver_calls
                - started_counts[mode].solver_calls
            ),
            "trajectory_traversal_step_counts": {
                key: final_work[mode][key] - started_work[mode][key]
                for key in final_work[mode]
            },
        }
        for mode in TrainingMode
    }
    kinetic_prediction = np.array(
        [value["kinetic_energy"] for value in predicted_diagnostics]
    )
    kinetic_truth = np.array(
        [value["kinetic_energy"] for value in truth_diagnostics]
    )
    high_prediction = np.array(
        [
            value["velocity_high_wavenumber_energy_fraction"]
            for value in predicted_diagnostics
        ]
    )
    high_truth = np.array(
        [
            value["velocity_high_wavenumber_energy_fraction"]
            for value in truth_diagnostics
        ]
    )
    hyper_prediction = np.array(
        [
            value["hyperviscosity_tendency_mass_norm"]
            for value in predicted_diagnostics
        ]
    )
    hyper_truth = np.array(
        [
            value["hyperviscosity_tendency_mass_norm"]
            for value in truth_diagnostics
        ]
    )
    enstrophy_prediction = np.array(
        [value["projected_enstrophy"] for value in predicted_diagnostics]
    )
    enstrophy_truth = np.array(
        [value["projected_enstrophy"] for value in truth_diagnostics]
    )
    diagnostic_times = np.array(
        [case.t0 + step * case.dt for step in diagnostic_steps],
        dtype=np.float64,
    )
    predicted_growth = _diagnostic_growth(
        diagnostic_times, predicted_diagnostics
    )
    truth_growth = _diagnostic_growth(diagnostic_times, truth_diagnostics)
    growth_warning = any(
        value["suspicious_late_time_growth"]
        for value in tuple(predicted_growth.values()) + tuple(truth_growth.values())
    )
    kinetic_mismatch = _diagnostic_mismatch(
        kinetic_prediction,
        kinetic_truth,
        diagnostic_steps,
        diagnostic_times,
    )
    enstrophy_mismatch = _diagnostic_mismatch(
        enstrophy_prediction,
        enstrophy_truth,
        diagnostic_steps,
        diagnostic_times,
    )
    high_mismatch = _diagnostic_mismatch(
        high_prediction,
        high_truth,
        diagnostic_steps,
        diagnostic_times,
    )
    hyper_mismatch = _diagnostic_mismatch(
        hyper_prediction,
        hyper_truth,
        diagnostic_steps,
        diagnostic_times,
    )
    relative_c0_error = abs(recovered - configuration.c0_truth) / abs(
        configuration.c0_truth
    )
    field_maxima = tuple(value["maximum"] for value in fieldwise.values())
    relative_checks = (
        heldout_metric["maximum"],
        kinetic_mismatch["maximum_relative_mismatch"],
        enstrophy_mismatch["maximum_relative_mismatch"],
        *field_maxima,
    )
    certification_reasons = []
    if not finite:
        certification_reasons.append("a deployed state contains nonfinite values")
    if growth_warning:
        certification_reasons.append("a late-time growth heuristic issued a warning")
    if relative_c0_error > GATE4_RELATIVE_C0_TOLERANCE:
        certification_reasons.append("recovered c0 exceeds the certification tolerance")
    if any(value is None for value in relative_checks):
        certification_reasons.append(
            "at least one required relative error has a zero reference and is undefined"
        )
    elif any(
        value > GATE4_RELATIVE_ERROR_TOLERANCE for value in relative_checks
    ):
        certification_reasons.append(
            "a held-out relative error exceeds the numerical-precision tolerance"
        )
    minimum_height = float(
        min(
            value["minimum_height_coefficient"]
            for value in predicted_diagnostics
        )
    )
    if minimum_height <= 0.0:
        certification_reasons.append("deployed height is not admissible")
    evaluation = {
        "evaluation_metrics_contract": COMMON_EVALUATION_METRICS,
        "truth_c0": configuration.c0_truth,
        "recovered_c0": recovered,
        "normalized_z": recovered / case.c0_scale,
        "relative_c0_error": relative_c0_error,
        "heldout_deployment_contract": {
            "trusted_initial_state_index": configuration.training_stop_step,
            "target_state_indices": (
                configuration.training_stop_step + 1,
                configuration.heldout_stop_step,
            ),
            "complete_production_steps": (
                configuration.heldout_stop_step
                - configuration.training_stop_step
            ),
            "truth_resets": 0,
            "predicted_state_recursively_reused": True,
            "truth_targets_consulted_only_after_prediction": True,
        },
        "one_step_state_error": one_step,
        "training_autonomous_trajectory_error": training_metric,
        "heldout_autonomous_trajectory_error": heldout_metric,
        "final_state_error": heldout_metric["final"],
        "accumulated_trajectory_error": heldout_metric["accumulated"],
        "fieldwise_heldout_errors": fieldwise,
        "fieldwise_heldout_final_errors": {
            field: values["final"] for field, values in fieldwise.items()
        },
        "kinetic_energy_mismatch": kinetic_mismatch,
        "kinetic_energy_history_mismatch": kinetic_mismatch[
            "relative_history_l2_mismatch"
        ],
        "projected_enstrophy_mismatch": enstrophy_mismatch,
        "projected_enstrophy_history_mismatch": enstrophy_mismatch[
            "relative_history_l2_mismatch"
        ],
        "high_wavenumber_mismatch": high_mismatch,
        "high_wavenumber_history_mismatch": high_mismatch[
            "relative_history_l2_mismatch"
        ],
        "hyperviscosity_diagnostic": hyper_mismatch,
        "hyperviscosity_diagnostic_mismatch": hyper_mismatch[
            "relative_history_l2_mismatch"
        ],
        "all_deployed_states_finite": bool(finite),
        "minimum_deployed_height_coefficient": minimum_height,
        "numerical_stability_status": {
            "finite_state_check_passed": bool(finite),
            "prediction_growth_heuristics": predicted_growth,
            "truth_growth_heuristics": truth_growth,
            "any_late_time_growth_warning": bool(growth_warning),
            "interpretation": (
                "growth warnings are diagnostic only; no warning is not a "
                "stand-alone stability proof"
            ),
        },
        "objectives_under_all_training_modes": cross,
        "cross_evaluation_cost": cross_cost,
        "deployment_solver_calls": (
            configuration.training_stop_step
            - configuration.training_start_step
            + configuration.heldout_stop_step
            - configuration.training_stop_step
        ),
        "deployment_step_accounting": {
            "training_autonomous_steps": (
                configuration.training_stop_step
                - configuration.training_start_step
            ),
            "canonical_heldout_autonomous_steps": (
                configuration.heldout_stop_step
                - configuration.training_stop_step
            ),
        },
    }
    evaluation["gate4_certification"] = {
        "passed": not certification_reasons,
        "relative_error_tolerance": GATE4_RELATIVE_ERROR_TOLERANCE,
        "relative_c0_tolerance": GATE4_RELATIVE_C0_TOLERANCE,
        "failure_reasons": tuple(certification_reasons),
        "interpretation": (
            "deterministic end-to-end workflow certification; not an ML "
            "generalization claim"
        ),
    }
    evaluation["wall_time_seconds"] = float(perf_counter() - started)
    return evaluation


def _configuration_from_arguments(arguments):
    if arguments.selected_plan is not None:
        override_names = (
            "c0_truth",
            "c0_initial",
            "training_start",
            "training_stop",
            "heldout_stop",
            "observation_stride",
            "truth_reset_horizon",
            "truth_reset_window_stride",
            "rollout_horizon",
            "truth_reset_loss",
            "rollout_loss",
            "solver_loss_normalization",
        )
        supplied = tuple(
            name for name in override_names if getattr(arguments, name) is not None
        )
        if supplied:
            raise ValueError(
                "--selected-plan owns canonical scientific indexing; remove "
                f"conflicting overrides {supplied}"
            )
        _, selected = load_selected_test1b_plan(arguments.selected_plan)
        return selected.inference_configuration(arguments.truth_run)
    if arguments.training_stop is None or arguments.heldout_stop is None:
        raise ValueError(
            "generic inference requires --training-stop and --heldout-stop; "
            "canonical Test 1B should use --selected-plan"
        )
    return ResolvedInferenceConfiguration(
        truth_run_directory=arguments.truth_run,
        c0_truth=0.14 if arguments.c0_truth is None else arguments.c0_truth,
        c0_initial=0.07 if arguments.c0_initial is None else arguments.c0_initial,
        training_start_step=(
            0 if arguments.training_start is None else arguments.training_start
        ),
        training_stop_step=arguments.training_stop,
        heldout_stop_step=arguments.heldout_stop,
        observation_stride=(
            1
            if arguments.observation_stride is None
            else arguments.observation_stride
        ),
        truth_reset_horizon=(
            1
            if arguments.truth_reset_horizon is None
            else arguments.truth_reset_horizon
        ),
        truth_reset_window_stride=arguments.truth_reset_window_stride,
        rollout_horizon=(
            5 if arguments.rollout_horizon is None else arguments.rollout_horizon
        ),
        truth_reset_loss=(
            "terminal"
            if arguments.truth_reset_loss is None
            else arguments.truth_reset_loss
        ),
        rollout_loss=(
            "accumulated" if arguments.rollout_loss is None else arguments.rollout_loss
        ),
        solver_loss_normalization=(
            SolverLossNormalization.INITIAL_GUESS_RESIDUAL
            if arguments.solver_loss_normalization is None
            else arguments.solver_loss_normalization
        ),
    )


def _scan_configuration_from_arguments(arguments):
    if arguments.selected_plan is not None:
        supplied = tuple(
            name
            for name in (
                "scan_lower",
                "scan_upper",
                "scan_points",
                "scan_derivative_level",
            )
            if getattr(arguments, name) is not None
        )
        if supplied:
            raise ValueError(
                "--selected-plan owns the canonical objective scan; remove "
                f"conflicting overrides {supplied}"
            )
        selected_plan, _ = load_selected_test1b_plan(arguments.selected_plan)
        scan = selected_plan["objective_scan"]
        return ObjectiveScanConfiguration(
            physical_lower=scan["physical_lower"],
            physical_upper=scan["physical_upper"],
            points=scan["points"],
            derivative_level=scan["derivative_level"],
        )
    if arguments.scan_derivative_level is None:
        raise ValueError(
            "generic scans require explicit --scan-derivative-level"
        )
    return ObjectiveScanConfiguration(
        physical_lower=(0.03 if arguments.scan_lower is None else arguments.scan_lower),
        physical_upper=(0.20 if arguments.scan_upper is None else arguments.scan_upper),
        points=18 if arguments.scan_points is None else arguments.scan_points,
        derivative_level=arguments.scan_derivative_level,
    )


def _validated_fit_result(fitted, configuration):
    """Validate a completed successful fit and return its mode and c0."""
    if not isinstance(fitted, Mapping):
        raise TypeError("fit result must be a mapping")
    if fitted.get("status") != "complete":
        raise ValueError("evaluate requires a complete fit result")
    intent = fitted.get("intent")
    if not isinstance(intent, Mapping) or intent.get("command") != "fit":
        raise ValueError("evaluate input is not a fit result")
    if intent.get("configuration") != configuration.to_dict():
        raise ValueError("fit result uses a different inference configuration")
    try:
        mode = TrainingMode(intent["training_mode"])
    except (KeyError, ValueError) as exc:
        raise ValueError("fit result has an invalid training mode") from exc
    result = fitted.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("fit result lacks optimizer output")
    if result.get("success") is not True:
        raise ValueError("evaluate requires a successful fit result")
    if result.get("failure_reason") is not None:
        raise ValueError("successful fit result has a failure reason")
    if fitted.get("termination_claim") != "converged":
        raise ValueError("fit result lacks an independent convergence claim")
    starting_c0 = float(result["starting_c0"])
    recovered_c0 = float(result["recovered_c0"])
    if not np.isfinite(starting_c0) or starting_c0 != configuration.c0_initial:
        raise ValueError("fit result has an incompatible starting c0")
    if not np.isfinite(recovered_c0) or recovered_c0 <= 0.0:
        raise ValueError("fit result recovered c0 must be positive and finite")
    summary = fitted.get("fit_summary")
    if not isinstance(summary, Mapping) or float(
        summary.get("recovered_c0", np.nan)
    ) != recovered_c0:
        raise ValueError("fit summary and optimizer output disagree on recovered c0")
    return mode, starting_c0, recovered_c0


def _parser():
    parser = argparse.ArgumentParser(
        description="Prepared Test-1B inference; use only after pilot selection"
    )
    parser.add_argument("command", choices=("plan", "scan", "fit", "evaluate"))
    parser.add_argument("--truth-run", required=True)
    parser.add_argument("--selected-plan")
    parser.add_argument("--c0-truth", type=float)
    parser.add_argument("--c0-initial", type=float)
    parser.add_argument("--training-start", type=int)
    parser.add_argument("--training-stop", type=int)
    parser.add_argument("--heldout-stop", type=int)
    parser.add_argument("--observation-stride", type=int)
    parser.add_argument("--truth-reset-horizon", type=int)
    parser.add_argument("--truth-reset-window-stride", type=int)
    parser.add_argument("--rollout-horizon", type=int)
    parser.add_argument(
        "--truth-reset-loss", choices=("terminal", "accumulated")
    )
    parser.add_argument(
        "--rollout-loss", choices=("terminal", "accumulated")
    )
    parser.add_argument(
        "--solver-loss-normalization",
        choices=tuple(value.value for value in SolverLossNormalization),
    )
    parser.add_argument("--mode", choices=tuple(mode.value for mode in TrainingMode))
    parser.add_argument("--scan-lower", type=float)
    parser.add_argument("--scan-upper", type=float)
    parser.add_argument("--scan-points", type=int)
    parser.add_argument(
        "--scan-derivative-level",
        choices=tuple(level.value for level in ScanDerivativeLevel),
    )
    parser.add_argument("--fit-result")
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    configuration = _configuration_from_arguments(arguments)
    scan_configuration = (
        _scan_configuration_from_arguments(arguments)
        if arguments.command == "scan"
        else None
    )
    if arguments.command == "plan":
        plan = build_inference_index_plan(configuration)
        write_json_record(
            arguments.output,
            {"configuration": configuration.to_dict(), "index_plan": asdict(plan)},
        )
        return 0
    if arguments.mode is None and arguments.command in ("scan", "fit"):
        raise ValueError("scan and fit require --mode")
    output_path = Path(arguments.output).resolve()
    mode_value = None if arguments.mode is None else TrainingMode(arguments.mode).value
    intent = {
        "command": arguments.command,
        "selected_plan": (
            None
            if arguments.selected_plan is None
            else str(Path(arguments.selected_plan).resolve())
        ),
        "configuration": configuration.to_dict(),
        "training_mode": mode_value,
        "scan_configuration": (
            None
            if arguments.command != "scan"
            else scan_configuration.to_dict()
        ),
        "fit_result": (
            None
            if arguments.fit_result is None
            else str(Path(arguments.fit_result).resolve())
        ),
    }
    validated_fit = None
    if arguments.command == "evaluate":
        if not arguments.fit_result:
            raise ValueError("evaluate requires --fit-result")
        fitted = read_json_record(arguments.fit_result)
        mode, starting_c0, recovered = _validated_fit_result(
            fitted, configuration
        )
        validated_fit = (fitted, mode, starting_c0, recovered)
    existing = None
    if output_path.exists():
        existing = read_json_record(output_path)
        if existing.get("intent") != intent:
            raise RuntimeError("existing inference output belongs to another intent")
        if existing.get("status") == "complete":
            print(json.dumps({"output": str(output_path), "status": "complete"}))
            return 0
    record = {
        "intent": intent,
        "status": "running",
        "restartability": (
            "landscape points resume incrementally; deterministic fit/evaluation "
            "rerun from the beginning after interruption"
        ),
        "failure_reason": None,
    }
    if existing is not None and arguments.command == "scan":
        record["points"] = existing.get("points", [])
    write_json_record(output_path, record)
    try:
        include_heldout = arguments.command == "evaluate"
        case, truth = load_resolved_truth(
            configuration, include_heldout=include_heldout
        )
        loaded_steps = tuple(truth.states)
        record["truth_data_access"] = {
            "loaded_state_index_range": (loaded_steps[0], loaded_steps[-1]),
            "heldout_target_states_loaded": bool(include_heldout),
            "contract": (
                "scan and fit load only states through training_stop; held-out "
                "targets are loaded only for post-fit evaluation"
            ),
        }
        suite = prepare_resolved_hidden_c0_objectives(case, truth, configuration)
        record["objective_preprocessing"] = {
            "hyperviscosity_child_calls": (
                suite.preprocessing_hyperviscosity_child_calls
            ),
            "complete_solver_steps": suite.preprocessing_complete_solver_calls,
            "solver_loss_normalization": (
                configuration.solver_loss_normalization.value
            ),
            "accounting_scope": (
                "fixed truth-derived targets and any initial-guess normalizers; "
                "truth-target-mass normalization needs no solver steps; separate "
                "from scan-point, optimizer, and evaluation counts"
            ),
        }
        if arguments.command == "scan":
            mode = TrainingMode(arguments.mode)

            def save_point(_point, points):
                record["points"] = tuple(asdict(point) for point in points)
                write_json_record(output_path, record)

            scan = scan_scalar_objective(
                suite[mode],
                scan_configuration,
                completed_points=record.get("points", ()),
                completed_configuration=(
                    None
                    if existing is None
                    else existing.get("intent", {}).get("scan_configuration")
                ),
                point_callback=save_point,
            )
            record["points"] = tuple(asdict(point) for point in scan)
        elif arguments.command == "fit":
            mode = TrainingMode(arguments.mode)
            result = optimize_hidden_c0(
                suite[mode],
                initial_c0=configuration.c0_initial,
                configuration=ScalarOptimizerConfiguration(),
            )
            record.update(
                {
                    "optimizer": asdict(ScalarOptimizerConfiguration()),
                    "result": asdict(result),
                    "fit_summary": {
                        "recovered_c0": result.recovered_c0,
                        "relative_c0_error": (
                            abs(result.recovered_c0 - configuration.c0_truth)
                            / abs(configuration.c0_truth)
                        ),
                        "accepted_optimizer_steps": (
                            max(len(result.normalized_iterates) - 1, 0)
                            + int(
                                bool(result.normalized_iterates)
                                and result.recovered_normalized_z
                                != result.normalized_iterates[-1]
                            )
                        ),
                        "objective_evaluations": (
                            result.counts.objective_evaluations
                        ),
                        "gradient_evaluations": (
                            result.counts.gradient_evaluations
                        ),
                        "hvp_evaluations": result.counts.hvp_evaluations,
                        "complete_solver_steps": result.counts.solver_calls,
                        "trajectory_traversal_step_counts": _objective_work_record(
                            suite[mode]
                        ),
                        "wall_time_seconds": result.wall_time_seconds,
                        "termination_reason": result.termination_reason,
                    },
                    "termination_claim": (
                        "converged" if result.success else "not converged"
                    ),
                }
            )
        else:
            fitted, mode, starting_c0, recovered = validated_fit
            fit_result = fitted["result"]
            record["fitted_training_mode"] = mode.value
            record["fit_provenance"] = {
                "path": str(Path(arguments.fit_result).resolve()),
                "status": fitted["status"],
                "success": fit_result["success"],
                "termination_claim": fitted["termination_claim"],
                "starting_c0": starting_c0,
                "recovered_c0": recovered,
                "accepted_optimizer_steps": fitted["fit_summary"][
                    "accepted_optimizer_steps"
                ],
                "objective_evaluations": fit_result["counts"][
                    "objective_evaluations"
                ],
                "gradient_evaluations": fit_result["counts"][
                    "gradient_evaluations"
                ],
                "hvp_evaluations": fit_result["counts"]["hvp_evaluations"],
                "fit_wall_time_seconds": fit_result["wall_time_seconds"],
            }
            record["evaluation"] = evaluate_resolved_hidden_c0(
                case, truth, suite, configuration, recovered
            )
            if not record["evaluation"]["gate4_certification"]["passed"]:
                reasons = record["evaluation"]["gate4_certification"][
                    "failure_reasons"
                ]
                raise RuntimeError(
                    "Gate-4 deterministic certification failed: "
                    + "; ".join(reasons)
                )
        record["status"] = "complete"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["failure_reason"] = "KeyboardInterrupt"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json_record(output_path, record)
    print(json.dumps({"output": str(output_path), "status": record["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ResolvedTruthTrajectory",
    "evaluate_resolved_hidden_c0",
    "load_resolved_truth",
    "prepare_resolved_hidden_c0_objectives",
)
