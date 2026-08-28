"""Sparse-endpoint direct training and two-stage FIML for Test 2A.

This module is opt-in preparation/production plumbing.  It reuses the exact
six-child trajectory implementation and never changes the deployed solver.
Only truth boundaries 0..80 are permitted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .hidden_c0 import _copy_function, _flat_values
from .jax_moist import moist_diagnostics_jax
from .learned_physics.parameters import (
    tree_dot,
    tree_norm,
    validate_float64_tree,
)
from .resolved_hidden_c0 import (
    ResolvedPilotConfiguration,
    read_json_record,
    write_json_record,
)
from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
from .resolved_hidden_c0_inference import load_resolved_truth
from .selected_test1b import load_selected_test1b_plan
from .test2a_apriori_autonomous import load_compatible_neural_physics
from .test2a_discrete_training import _file_sha256
from .test2a_discrete_training import CompactCheckpointObjective
from .test2a_embedded_moist import (
    PHYSICS_MODE_NEURAL_A,
    parameter_pytree_sha256,
)
from .test2a_horizon_curriculum import _canonical_json_sha256
from .test2a_operator import (
    DenseMLP,
    load_mlp_parameters,
    load_operator_dataset,
    mlp_configuration_from_record,
    normalization_from_record,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters_atomic,
)
from .test2a_pyrol import JAXPytreeObjective, build_test2a_lbfgs_parameters
from .test2a_trajectory import (
    GlobalMixedMassMetric,
    NeuralTrajectoryObjective,
    TrajectoryPyROLObjective,
    estimate_owned_bytes,
    reset_windows,
)


HORIZONS = (2, 5)
H1_BASELINE_SHA256 = (
    "ebc49083bda299d91e614adeaeefdda0400ca1e8cfccc95a3b4ba953044f963c"
)
DENOMINATOR = 4090171967662.3027
DENOMINATOR_SHA256 = (
    "10bda77bf2e003802c560ef1218fe28b17531da6b30e3f97cf22fa04a62d4753"
)
_STATE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")
_STATE_KEYS = ("h", "S", "Qv", "Qc")
_SOURCE_KEYS = ("S", "Qv", "Qc", "Qr")


def load_fiml_configuration(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("benchmark_stage") != "Test 2A sparse-endpoint FIML H2-H5":
        raise ValueError("not the selected Test-2A FIML configuration")
    if record["truth"]["allowed_state_indices"] != [0, 80] or not record[
        "truth"
    ]["states_after_80_forbidden"]:
        raise ValueError("FIML may use only truth states 0..80")
    if record["baseline"]["parameter_pytree_sha256"] != H1_BASELINE_SHA256:
        raise ValueError("FIML baseline is not the completed H1 artifact")
    baseline_path = Path(record["baseline"]["parameter_file"])
    if not baseline_path.exists() or _file_sha256(baseline_path) != record[
        "baseline"
    ]["parameter_npz_sha256"]:
        raise ValueError("FIML H1 baseline NPZ fingerprint changed")
    objective = record["objective"]
    if (
        objective["loss_mode"] != "endpoint"
        or float(objective["endpoint_weight"]) != 1.0
        or objective["per_target_normalization"] is not False
        or float(objective["common_denominator_D"]) != DENOMINATOR
        or objective["common_denominator_sha256"] != DENOMINATOR_SHA256
    ):
        raise ValueError("sparse-endpoint objective contract changed")
    field = record["field_inversion"]
    if (
        field["parameterization"]
        != "A_FI=A_H1(current postprefix state)+A_scale*c"
        or field["control_shape_per_step"] != [256, 16]
        or field["initial_control"] != 0.0
        or field["truth_A_regularization"] is not False
        or field["spatial_smoothness"] is not False
        or field["temporal_smoothness"] is not False
    ):
        raise ValueError("primary field-inversion contract changed")
    optimizer = record["optimizer"]
    if (
        optimizer["library"] != "PyROL/ROL"
        or optimizer["method"] != "line-search L-BFGS"
        or int(optimizer["maximum_secant_storage"]) != 20
        or float(optimizer["gradient_tolerance"]) != 1.0e-8
        or float(optimizer["step_tolerance"]) != 1.0e-12
        or optimizer["production_HVP"] is not False
        or optimizer["new_optimizer_and_empty_secant_history"] is not True
    ):
        raise ValueError("FIML optimizer contract changed")
    for horizon in HORIZONS:
        starts = sparse_starts(horizon)
        encoded = record["truth"]["optimization_observations"][str(horizon)]
        actual = tuple(range(encoded["starts"][0], encoded["starts"][1] + 1, encoded["starts"][2]))
        endpoints = tuple(range(encoded["endpoints"][0], encoded["endpoints"][1] + 1, encoded["endpoints"][2]))
        if actual != starts or endpoints != tuple(value + horizon for value in starts):
            raise ValueError(f"H{horizon} sparse observation schedule changed")
        if encoded["intermediate_truth_in_objective"] is not False:
            raise ValueError("intermediate truth entered sparse optimization")
    return record


def sparse_starts(horizon):
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError("sparse FIML horizon must be 2 or 5")
    return tuple(range(0, 80, horizon))


def sparse_observation_indices(horizon):
    starts = sparse_starts(horizon)
    return tuple(sorted(set(starts) | {value + int(horizon) for value in starts}))


def sparse_windows(horizon):
    return reset_windows(sparse_starts(horizon), horizon, "endpoint", (1.0,))


def control_sha256(controls):
    values = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    digest = sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _require_tree(values, keys, name):
    result = {}
    for key in keys:
        value = jnp.asarray(values[key])
        if value.dtype != jnp.float64:
            raise TypeError(f"{name}[{key!r}] must be float64")
        result[key] = value
    return result


class FieldControlAMoistPhysics:
    """Add a dimensionless free GLL correction around frozen H1 neural A."""

    physics_mode = PHYSICS_MODE_NEURAL_A

    def __init__(self, baseline_physics, a_scale, *, use_jit=True):
        self.baseline_physics = baseline_physics
        self.model_configuration = baseline_physics.model_configuration
        self.normalization = baseline_physics.normalization
        self.provenance = dict(baseline_physics.provenance)
        self.use_jit = bool(use_jit)
        self.a_scale = float(a_scale)
        if not np.isfinite(self.a_scale) or self.a_scale <= 0.0:
            raise ValueError("field-control A scale must be positive")
        self._baseline_parameters = baseline_physics.parameters

        def combined(state, fields, moist_parameters, controls):
            state_values = _require_tree(state, _STATE_KEYS, "state")
            field_values = _require_tree(fields, ("B",), "fields")
            moist_values = _require_tree(
                moist_parameters,
                ("g", "q0", "H0", "gamma_r", "qprecip", "L", "configured_dt"),
                "moist_parameters",
            )
            correction = jnp.asarray(controls)
            if correction.dtype != jnp.float64:
                raise TypeError("field controls must be float64")
            if correction.shape != state_values["h"].shape:
                raise ValueError("field controls must match deployed GLL shape")
            baseline = baseline_physics.combined_with_parameters(
                state_values,
                field_values,
                moist_values,
                self._baseline_parameters,
            )
            baseline_a = jnp.asarray(baseline["rates"]["A"], dtype=jnp.float64)
            original_r = jnp.asarray(baseline["rates"]["R"], dtype=jnp.float64)
            active_a = baseline_a + self.a_scale * correction
            h = state_values["h"]
            beta2 = moist_values["g"] * moist_values["L"]
            return {
                "rates": {"A": active_a, "R": original_r},
                "source": {
                    "S": h * beta2 * active_a,
                    "Qv": h * active_a,
                    "Qc": -h * (active_a + original_r),
                    "Qr": h * original_r,
                },
            }

        self._combined_explicit = combined
        self.combined_parameterized_kernel = jax.jit(combined) if use_jit else combined

        def frozen_combined(state, fields, moist_parameters):
            return combined(
                state,
                fields,
                moist_parameters,
                jnp.zeros_like(jnp.asarray(state["h"], dtype=jnp.float64)),
            )

        self.combined_kernel = jax.jit(frozen_combined) if use_jit else frozen_combined

        def diagnostics(state, fields, moist_parameters, controls):
            original = moist_diagnostics_jax(state, fields, moist_parameters)
            baseline = baseline_physics.combined_with_parameters(
                state, fields, moist_parameters, self._baseline_parameters
            )
            active = combined(state, fields, moist_parameters, controls)
            return {
                **original,
                "analytical_A_reference": original["A"],
                "baseline_neural_A": baseline["rates"]["A"],
                "field_control": controls,
                "field_controlled_A": active["rates"]["A"],
                "A": active["rates"]["A"],
                "R": active["rates"]["R"],
            }

        self.diagnostic_parameterized_kernel = (
            jax.jit(diagnostics) if use_jit else diagnostics
        )

        def frozen_diagnostics(state, fields, moist_parameters):
            return diagnostics(
                state,
                fields,
                moist_parameters,
                jnp.zeros_like(jnp.asarray(state["h"], dtype=jnp.float64)),
            )

        self.diagnostic_kernel = (
            jax.jit(frozen_diagnostics) if use_jit else frozen_diagnostics
        )

        def state_jvp(state, direction, fields, moist_parameters):
            source = lambda active_state: frozen_combined(
                active_state, fields, moist_parameters
            )["source"]
            return jax.jvp(source, (state,), (direction,))

        def state_vjp(state, source_covector, fields, moist_parameters):
            source = lambda active_state: frozen_combined(
                active_state, fields, moist_parameters
            )["source"]
            _, pullback = jax.vjp(source, state)
            return pullback(source_covector)[0]

        def state_differentiated_vjp(
            state,
            source_covector,
            direction,
            source_covector_direction,
            fields,
            moist_parameters,
        ):
            def action(active_state, active_covector):
                source = lambda value: frozen_combined(
                    value, fields, moist_parameters
                )["source"]
                _, pullback = jax.vjp(source, active_state)
                return pullback(active_covector)[0]

            return jax.jvp(
                action,
                (state, source_covector),
                (direction, source_covector_direction),
            )

        self.state_jvp_kernel = jax.jit(state_jvp) if use_jit else state_jvp
        self.state_vjp_kernel = jax.jit(state_vjp) if use_jit else state_vjp
        self.state_differentiated_vjp_kernel = (
            jax.jit(state_differentiated_vjp)
            if use_jit
            else state_differentiated_vjp
        )

    def parameter_jvp(
        self,
        state,
        parameter_direction,
        fields,
        moist_parameters,
        *,
        base_parameters=None,
    ):
        if base_parameters is None:
            raise ValueError("field-control JVP requires explicit base controls")
        base = validate_float64_tree(base_parameters, name="field controls")
        direction = validate_float64_tree(
            parameter_direction, name="field-control direction"
        )
        source = lambda active: self._combined_explicit(
            state, fields, moist_parameters, active
        )["source"]
        return jax.jvp(source, (base,), (direction,))

    def parameter_vjp(
        self,
        state,
        source_covector,
        fields,
        moist_parameters,
        *,
        base_parameters=None,
    ):
        if base_parameters is None:
            raise ValueError("field-control VJP requires explicit base controls")
        base = validate_float64_tree(base_parameters, name="field controls")
        covector = _require_tree(source_covector, _SOURCE_KEYS, "source_covector")
        source = lambda active: self._combined_explicit(
            state, fields, moist_parameters, active
        )["source"]
        _, pullback = jax.vjp(source, base)
        return pullback(covector)[0]


@dataclass(frozen=True)
class FIMLCase:
    configuration: dict
    case: object
    truth: dict
    baseline_physics: object
    field_physics: FieldControlAMoistPhysics | None
    model_configuration: object
    normalization: object
    materialized_truth_indices: tuple[int, ...]


def _build_case(configuration, horizon, *, field_control, truth_indices=None):
    from dataclasses import replace
    from .test2a_trajectory_certification import (
        load_trajectory_preparation_configuration,
    )

    selected = load_trajectory_preparation_configuration(
        configuration["truth"]["trajectory_configuration"]
    )
    _, plan = load_selected_test1b_plan(selected["truth"]["selected_plan"])
    inference = plan.inference_configuration(
        Path(selected["truth"]["run_directory"]).resolve()
    )
    _, loaded = load_resolved_truth(inference, include_heldout=False)
    if tuple(loaded.states) != tuple(range(81)):
        raise ValueError("FIML loader requires exactly truth states 0..80")
    baseline = load_compatible_neural_physics(
        configuration["baseline"]["embedding_configuration"],
        configuration["baseline"]["parameter_file"],
        expected_pytree_sha256=configuration["baseline"][
            "parameter_pytree_sha256"
        ],
    )
    local_physics = (
        FieldControlAMoistPhysics(
            baseline, configuration["field_inversion"]["A_scale"], use_jit=True
        )
        if field_control
        else baseline
    )
    pilot = ResolvedPilotConfiguration.from_dict(loaded.metadata["configuration"])
    neural_pilot = replace(
        pilot,
        moist_backend="jax",
        output_directory="/tmp/test2a-fiml-no-output",
    )
    case = build_resolved_hidden_c0_case(
        neural_pilot, jax_moist_local_physics=local_physics
    )
    if truth_indices is None:
        indices = sparse_observation_indices(horizon)
    else:
        indices = tuple(sorted(set(int(value) for value in truth_indices)))
        if not indices or min(indices) < 0 or max(indices) > 80:
            raise ValueError("FIML truth materialization is restricted to 0..80")
    truth = {
        step: case.state_from_values(
            _flat_values(loaded.states[step]), f"test2a_fiml_truth_{step}"
        )
        for step in indices
    }
    parameters, model_configuration = load_mlp_parameters(
        configuration["baseline"]["parameter_file"]
    )
    if parameter_pytree_sha256(parameters) != H1_BASELINE_SHA256:
        raise ValueError("H1 baseline fingerprint changed")
    return FIMLCase(
        configuration=configuration,
        case=case,
        truth=truth,
        baseline_physics=baseline,
        field_physics=local_physics if field_control else None,
        model_configuration=model_configuration,
        normalization=baseline.normalization,
        materialized_truth_indices=indices,
    )


def build_direct_objective(problem, horizon):
    metric = GlobalMixedMassMetric(
        problem.case.helper,
        DENOMINATOR,
        denominator_sha256=DENOMINATOR_SHA256,
    )
    return NeuralTrajectoryObjective(
        problem.case,
        problem.truth,
        sparse_windows(horizon),
        metric=metric,
        c0=0.14,
        use_fixed_prefix=True,
    )


@dataclass(frozen=True)
class FieldInversionTape:
    control_sha256: str
    states: tuple[object, ...]
    step_caches: tuple[object, ...]
    data_misfit: float
    regularizer: float
    objective: float
    estimated_owned_bytes: int


class FieldInversionWindowObjective:
    """Exact sparse-endpoint inversion for one independent reset window."""

    def __init__(self, problem, horizon, start, regularization_lambda):
        self.problem = problem
        self.case = problem.case
        self.helper = problem.case.helper
        self.horizon = int(horizon)
        self.start = int(start)
        production_window = (
            self.horizon in HORIZONS and self.start in sparse_starts(self.horizon)
        )
        h1_control = self.horizon == 1 and self.start == 0
        if not production_window and not h1_control:
            raise ValueError("field-inversion window is outside selected schedule")
        self.endpoint = self.start + self.horizon
        required = {self.start, self.endpoint}
        if not required.issubset(problem.truth):
            raise ValueError("FI problem lacks its observed origin or endpoint")
        if production_window and set(problem.truth) != set(
            sparse_observation_indices(self.horizon)
        ):
            raise ValueError("FI problem materialized unexpected truth indices")
        self.regularization_lambda = float(regularization_lambda)
        if not np.isfinite(self.regularization_lambda) or self.regularization_lambda < 0:
            raise ValueError("lambda must be finite and nonnegative")
        self.metric = GlobalMixedMassMetric(
            self.helper, DENOMINATOR, denominator_sha256=DENOMINATOR_SHA256
        )
        with self.case.physical_c0(0.14):
            self.prefix = self.helper.take_fixed_prefix_cached(
                problem.truth[self.start],
                self.case.t0 + self.start * self.case.dt,
                self.case.dt,
            )
        points = int(self.helper.moist_helper.layout.owned_cell_count)
        local = int(self.helper.moist_helper.layout.points_per_cell)
        if (points, local) != tuple(
            problem.configuration["field_inversion"]["control_shape_per_step"]
        ):
            raise ValueError("deployed GLL control shape changed")
        self.control_shape = (self.horizon, points, local)
        self._last_tape = None
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.forward_complete_steps = 0
        self.reverse_complete_steps = 0
        self.same_control_tape_hits = 0
        self.tape_invalidations = 0

    def zero_controls(self):
        return jnp.zeros(self.control_shape, dtype=jnp.float64)

    def _require_controls(self, controls):
        owned = validate_float64_tree(controls, name="field controls")
        if not isinstance(owned, jax.Array) or owned.shape != self.control_shape:
            raise ValueError(f"controls must have shape {self.control_shape}")
        return owned

    def _forward(self, controls):
        owned = self._require_controls(controls)
        states = [
            _copy_function(
                self.problem.truth[self.start],
                f"test2a_fiml_H{self.horizon}_{self.start}_state0",
            )
        ]
        caches = []
        with self.case.physical_c0(0.14):
            first = self.helper.take_forward_step_from_prefix(self.prefix, owned[0])
            caches.append(first)
            states.append(_copy_function(first.state_out, "test2a_fiml_state1"))
            for offset in range(1, self.horizon):
                cache = self.helper.take_forward_step_cached(
                    states[-1],
                    self.case.t0 + (self.start + offset) * self.case.dt,
                    self.case.dt,
                    neural_parameters=owned[offset],
                )
                caches.append(cache)
                states.append(
                    _copy_function(cache.state_out, f"test2a_fiml_state{offset + 1}")
                )
        data = self.metric.value(
            states[-1],
            self.problem.truth[self.endpoint],
            self.endpoint,
            1.0,
            f"test2a_fiml_H{self.horizon}_{self.start}_endpoint",
        )
        regularizer = float(jnp.mean(owned * owned))
        tape = FieldInversionTape(
            control_sha256=control_sha256(owned),
            states=tuple(states),
            step_caches=tuple(caches),
            data_misfit=float(data),
            regularizer=regularizer,
            objective=float(data + self.regularization_lambda * regularizer),
            estimated_owned_bytes=0,
        )
        tape = FieldInversionTape(
            **{
                **tape.__dict__,
                "estimated_owned_bytes": estimate_owned_bytes(
                    (tape.states, tape.step_caches)
                ),
            }
        )
        self.forward_complete_steps += self.horizon
        self._last_tape = tape
        return tape

    def tape(self, controls):
        owned = self._require_controls(controls)
        fingerprint = control_sha256(owned)
        if self._last_tape is not None and self._last_tape.control_sha256 == fingerprint:
            self.same_control_tape_hits += 1
            return self._last_tape
        if self._last_tape is not None:
            self.tape_invalidations += 1
        return self._forward(owned)

    def value(self, controls):
        self.value_evaluations += 1
        return float(self.tape(controls).objective)

    def _zero_state(self, name):
        value = self.case.new_state(name)
        value.assign(0.0)
        return value

    def value_and_gradient(self, controls):
        self.gradient_evaluations += 1
        owned = self._require_controls(controls)
        tape = self.tape(owned)
        _, current = self.metric.value_and_dual(
            tape.states[-1],
            self.problem.truth[self.endpoint],
            self.endpoint,
            1.0,
            f"test2a_fiml_H{self.horizon}_{self.start}_gradient",
        )
        gradient = np.zeros(self.control_shape, dtype=np.float64)
        for offset in range(self.horizon - 1, -1, -1):
            reverse = self.helper.take_neural_parameter_adjoint_step(
                tape.step_caches[offset],
                current,
                stop_at_fixed_prefix=(offset == 0),
            )
            gradient[offset] = np.asarray(
                reverse.parameter_adjoint, dtype=np.float64
            )
            current = reverse.state_adjoint_in
            self.reverse_complete_steps += 1
        gradient += (
            2.0 * self.regularization_lambda / float(np.prod(self.control_shape))
        ) * np.asarray(owned, dtype=np.float64)
        return float(tape.objective), jnp.asarray(gradient, dtype=jnp.float64)

    def gradient(self, controls):
        return self.value_and_gradient(controls)[1]

    def tangent(self, controls, direction):
        owned = self._require_controls(controls)
        delta = self._require_controls(direction)
        tape = self.tape(owned)
        current = self._zero_state("test2a_fiml_control_tangent_initial")
        for offset in range(self.horizon):
            tangent = self.helper.take_neural_parameter_tangent_step(
                tape.step_caches[offset], current, delta[offset]
            )
            current = tangent.state_direction_out
        return current

    def diagnostics(self, controls):
        owned = self._require_controls(controls)
        tape = self.tape(owned)
        return {
            "horizon": self.horizon,
            "window_origin": self.start,
            "observed_truth_indices": [self.start, self.endpoint],
            "intermediate_truth_indices_used": [],
            "data_misfit": tape.data_misfit,
            "regularizer_mean_control_squared": tape.regularizer,
            "control_rms": float(np.sqrt(tape.regularizer)),
            "control_maximum_absolute": float(np.max(np.abs(np.asarray(owned)))),
            "regularization_lambda": self.regularization_lambda,
            "regularization_contribution": self.regularization_lambda
            * tape.regularizer,
            "objective": tape.objective,
            "gradient_norm": float(tree_norm(self.gradient(owned))),
            "control_sha256": control_sha256(owned),
            "estimated_owned_tape_bytes": tape.estimated_owned_bytes,
        }


def _fieldwise_maximum(left, right):
    return {
        name: float(
            np.max(
                np.abs(
                    np.asarray(actual.dat.data_ro, dtype=np.float64)
                    - np.asarray(expected.dat.data_ro, dtype=np.float64)
                )
            )
        )
        for name, actual, expected in zip(
            _STATE_FIELDS, left.subfunctions, right.subfunctions
        )
    }


def _gradient_relation(left, right):
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    difference = x - y
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    dot = float(np.dot(x, y))
    return {
        "left_norm": nx,
        "right_norm": ny,
        "absolute_difference": float(np.linalg.norm(difference)),
        "relative_difference": float(
            np.linalg.norm(difference) / max(ny, np.finfo(np.float64).tiny)
        ),
        "dot": dot,
        "cosine": dot / max(nx * ny, np.finfo(np.float64).tiny),
        "maximum_absolute_component_difference": float(
            np.max(np.abs(difference))
        ),
    }


def _direction(shape):
    values = np.linspace(-0.9, 0.7, int(np.prod(shape)), dtype=np.float64).reshape(shape)
    # Unit RMS, rather than unit Euclidean norm, keeps each physical GLL
    # perturbation above full-state roundoff as control dimension grows.
    return jnp.asarray(values / np.sqrt(np.mean(values * values)), dtype=jnp.float64)


def _single_direct_objective(problem, horizon, start):
    metric = GlobalMixedMassMetric(
        problem.case.helper,
        DENOMINATOR,
        denominator_sha256=DENOMINATOR_SHA256,
    )
    return NeuralTrajectoryObjective(
        problem.case,
        problem.truth,
        reset_windows((start,), horizon, "endpoint", (1.0,)),
        metric=metric,
        c0=0.14,
        use_fixed_prefix=True,
    )


def _source_invariants(tape):
    records = []
    for offset, cache in enumerate(tape.step_caches, 1):
        moist = cache.children[-1].cache
        source = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in moist.source_density.items()
        }
        beta2 = float(moist.parameters["g"] * moist.parameters["L"])
        water = source["Qv"] + source["Qc"] + source["Qr"]
        thermal = source["S"] - beta2 * source["Qv"]
        records.append(
            {
                "internal_step": offset,
                "water_maximum_absolute_residual": float(np.max(np.abs(water))),
                "thermal_maximum_absolute_residual": float(
                    np.max(np.abs(thermal))
                ),
            }
        )
    return records


def _truth_control_oracle(configuration, horizon, start, *, problem=None):
    """Post-hoc control using analytical A on the generated oracle path."""

    if problem is None:
        problem = _build_case(
            configuration,
            horizon,
            field_control=True,
            truth_indices=(start, start + horizon),
        )
    helper = problem.case.helper
    controls = []
    current = _copy_function(problem.truth[start], "test2a_fiml_truth_oracle_start")
    caches = []
    with problem.case.physical_c0(0.14):
        for offset in range(horizon):
            prefix = helper.take_fixed_prefix_cached(
                current,
                problem.case.t0 + (start + offset) * problem.case.dt,
                problem.case.dt,
            )
            zeros = jnp.zeros(
                (
                    helper.moist_helper.layout.owned_cell_count,
                    helper.moist_helper.layout.points_per_cell,
                ),
                dtype=jnp.float64,
            )
            baseline = helper.take_forward_step_from_prefix(prefix, zeros)
            moist = baseline.children[-1].cache
            analytical = np.asarray(
                moist.gll_diagnostics["analytical_A_reference"], dtype=np.float64
            )
            h1 = np.asarray(
                moist.gll_diagnostics["baseline_neural_A"], dtype=np.float64
            )
            control = jnp.asarray(
                (analytical - h1) / float(problem.field_physics.a_scale),
                dtype=jnp.float64,
            )
            active = helper.take_forward_step_from_prefix(prefix, control)
            controls.append(control)
            caches.append(active)
            current = _copy_function(active.state_out, "test2a_fiml_truth_oracle_state")
    return {
        "controls": jnp.stack(controls),
        "endpoint_fieldwise_maximum_absolute_difference": _fieldwise_maximum(
            current, problem.truth[start + horizon]
        ),
        "intermediate_truth_boundaries_compared": [],
        "optimizer_saw_truth_control": False,
        "purpose": "post-hoc implementation/controllability oracle only",
    }


def certify_fiml(configuration_path, output_path):
    configuration = load_fiml_configuration(configuration_path)
    baseline_parameters, _ = load_mlp_parameters(
        configuration["baseline"]["parameter_file"]
    )
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A sparse-endpoint/FIML derivative certification",
        "configuration_sha256": _canonical_json_sha256(configuration),
        "baseline_parameter_pytree_sha256": parameter_pytree_sha256(
            baseline_parameters
        ),
        "common_denominator_D": DENOMINATOR,
        "common_denominator_sha256": DENOMINATOR_SHA256,
        "horizons": {},
        "optimization_performed": False,
        "states_after_80_accessed": False,
    }
    tolerance = float(
        configuration["preparation"]["derivative_relative_tolerance"]
    )
    finite_difference_tolerance = float(
        configuration["preparation"]["finite_difference_relative_tolerance"]
    )
    for horizon, start in ((2, 0), (5, 40)):
        direct_problem = _build_case(configuration, horizon, field_control=False)
        field_problem = _build_case(configuration, horizon, field_control=True)
        direct = _single_direct_objective(direct_problem, horizon, start)
        field = FieldInversionWindowObjective(field_problem, horizon, start, 0.0)
        zero = field.zero_controls()
        direct_value = direct.value(baseline_parameters)
        zero_value, zero_gradient = field.value_and_gradient(zero)
        direct_tape = direct._last_tape.windows[0]
        field_tape = field.tape(zero)
        zero_fields = _fieldwise_maximum(
            field_tape.states[-1], direct_tape.states[-1]
        )
        zero_value_difference = abs(zero_value - direct_value)
        direction = _direction(field.control_shape)
        tangent = field.tangent(zero, direction)
        _, endpoint_dual = field.metric.value_and_dual(
            field_tape.states[-1],
            field_problem.truth[start + horizon],
            start + horizon,
            1.0,
            "test2a_fiml_tangent_adjoint",
        )
        tangent_pairing = float(field.helper.dual_pairing(endpoint_dual, tangent))
        adjoint_pairing = float(tree_dot(zero_gradient, direction))
        epsilon = 1.0e-3
        centered = (
            field.value(zero + epsilon * direction)
            - field.value(zero - epsilon * direction)
        ) / (2.0 * epsilon)
        scale = max(
            abs(tangent_pairing),
            abs(adjoint_pairing),
            abs(centered),
            np.finfo(np.float64).tiny,
        )
        tangent_relative = abs(tangent_pairing - adjoint_pairing) / scale
        fd_relative = abs(centered - adjoint_pairing) / scale
        if (
            tangent_relative > tolerance
            or fd_relative > finite_difference_tolerance
        ):
            raise RuntimeError(
                f"H{horizon} field-control derivative failed: "
                f"tangent={tangent_pairing:.17g}, adjoint={adjoint_pairing:.17g}, "
                f"centered={centered:.17g}, tangent_rel={tangent_relative:.3e}, "
                f"fd_rel={fd_relative:.3e}"
            )
        truth = _truth_control_oracle(
            configuration, horizon, start, problem=field_problem
        )
        if max(zero_fields.values()) != 0.0:
            raise RuntimeError("zero FI controls do not reproduce H1 baseline")
        result["horizons"][str(horizon)] = {
            "representative_window_origin": start,
            "observed_truth_indices": [start, start + horizon],
            "intermediate_truth_in_objective": [],
            "zero_control_data_misfit": zero_value,
            "zero_control_direct_endpoint_value": direct_value,
            "zero_control_objective_absolute_difference": zero_value_difference,
            "zero_control_endpoint_fieldwise_maximum_difference": zero_fields,
            "control_dimension": int(np.prod(field.control_shape)),
            "endpoint_owned_state_coefficient_dimension": int(
                _flat_values(field_tape.states[-1]).size
            ),
            "control_to_endpoint_coefficient_dimension_ratio": float(
                np.prod(field.control_shape)
                / _flat_values(field_tape.states[-1]).size
            ),
            "all_control_gradient_norm": float(tree_norm(zero_gradient)),
            "tangent_pairing": tangent_pairing,
            "adjoint_pairing": adjoint_pairing,
            "tangent_adjoint_absolute_discrepancy": abs(
                tangent_pairing - adjoint_pairing
            ),
            "tangent_adjoint_relative_discrepancy": tangent_relative,
            "centered_directional_derivative": centered,
            "directional_adjoint": adjoint_pairing,
            "directional_relative_discrepancy": fd_relative,
            "source_invariants": _source_invariants(field_tape),
            "truth_control_oracle": {
                key: value for key, value in truth.items() if key != "controls"
            },
        }
    h1_problem = _build_case(
        configuration, 1, field_control=True, truth_indices=(0, 1)
    )
    h1 = FieldInversionWindowObjective(h1_problem, 1, 0, 0.0)
    h1_zero = h1.zero_controls()
    h1_direction = _direction(h1.control_shape)
    h1_truth = _truth_control_oracle(configuration, 1, 0, problem=h1_problem)
    h1_value, h1_gradient = h1.value_and_gradient(h1_zero)
    h1_tangent = h1.tangent(h1_zero, h1_direction)
    _, h1_dual = h1.metric.value_and_dual(
        h1.tape(h1_zero).states[-1], h1_problem.truth[1], 1, 1.0, "fiml_h1"
    )
    result["H1_control_sanity"] = {
        "production_FIML_experiment": False,
        "data_misfit_at_H1_baseline": h1_value,
        "control_dimension": int(np.prod(h1.control_shape)),
        "gradient_norm": float(tree_norm(h1_gradient)),
        "tangent_adjoint_discrepancy": abs(
            float(h1.helper.dual_pairing(h1_dual, h1_tangent))
            - float(tree_dot(h1_gradient, h1_direction))
        ),
        "truth_control_endpoint_fieldwise_maximum_difference": h1_truth[
            "endpoint_fieldwise_maximum_absolute_difference"
        ],
        "interpretation": (
            "one final moist-child control maps linearly through dt*G(Y0), "
            "the fixed one-step post-prefix M2-Y map"
        ),
    }
    write_json_record(output_path, result)
    return result


def _solve_exact_lbfgs(objective, initial, iteration_limit, *, accepted_callback=None):
    """Run a fresh exact-gradient ROL L-BFGS process and return accounting."""

    from pyrol import Problem, Solver

    count = int(iteration_limit)
    if count < 1:
        raise ValueError("iteration limit must be positive")
    adapter = TrajectoryPyROLObjective(
        objective, initial, accepted_callback=accepted_callback
    )
    control = adapter.vector_from_pytree(initial)
    rol = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": 1.0e-8,
            "step_tolerance": 1.0e-12,
            "iteration_limit": count,
            "maximum_secant_storage": 20,
        }
    )
    started = perf_counter()
    solver = Solver(Problem(adapter, control), rol)
    solver.solve()
    wall = perf_counter() - started
    final = adapter.pytree_from_vector(control)
    state = solver.getAlgorithmState()
    if adapter.hvp_evaluations != 0:
        raise RuntimeError("FIML L-BFGS unexpectedly requested an HVP")
    return final, {
        "accepted_iterations": int(state.iter),
        "termination_reason": str(state.statusFlag),
        "objective_evaluations": int(adapter.value_evaluations),
        "gradient_evaluations": int(adapter.gradient_evaluations),
        "HVP_evaluations": int(adapter.hvp_evaluations),
        "wall_time_seconds": float(wall),
        "new_optimizer_process": True,
        "source_secant_history_reused": False,
    }


def _control_truth_metrics(field_objective, controls):
    tape = field_objective.tape(controls)
    predicted = []
    analytical = []
    baseline = []
    for cache in tape.step_caches:
        moist = cache.children[-1].cache
        predicted.append(np.asarray(moist.rates["A"], dtype=np.float64).reshape(-1))
        analytical.append(
            np.asarray(
                moist.gll_diagnostics["analytical_A_reference"], dtype=np.float64
            ).reshape(-1)
        )
        baseline.append(
            np.asarray(
                moist.gll_diagnostics["baseline_neural_A"], dtype=np.float64
            ).reshape(-1)
        )
    target = np.concatenate(analytical)
    return {
        "inferred_A_versus_analytical_A_on_same_FI_states": operator_metrics(
            np.concatenate(predicted), target
        ),
        "H1_A_versus_analytical_A_on_same_FI_states": operator_metrics(
            np.concatenate(baseline), target
        ),
        "role": "post-hoc synthetic-truth diagnostic; never used for lambda selection",
    }


def _lcurve_selection(records, candidates):
    """Select an interior log-log L-curve knee without truth-A information."""

    positive = [float(value) for value in candidates if float(value) > 0.0]
    points = []
    for value in positive:
        subset = [item for item in records if float(item["lambda"]) == value]
        data = float(np.median([item["final_data_misfit"] for item in subset]))
        control = float(np.median([item["final_control_rms"] for item in subset]))
        points.append(
            {
                "lambda": value,
                "median_data_misfit": data,
                "median_control_rms": control,
                "x": float(np.log10(max(control, np.finfo(np.float64).tiny))),
                "y": float(np.log10(max(data, np.finfo(np.float64).tiny))),
            }
        )
    if len(points) < 3:
        raise RuntimeError("lambda sweep has too few positive candidates")
    chord_lengths = [
        float(
            np.linalg.norm(
                np.array([points[index + 1]["x"], points[index + 1]["y"]])
                - np.array([points[index]["x"], points[index]["y"]])
            )
        )
        for index in range(len(points) - 1)
    ]
    total_arc = float(sum(chord_lengths))
    minimum_chord = 1.0e-3 * total_arc
    curvatures = []
    for index in range(1, len(points) - 1):
        p0 = np.array([points[index - 1]["x"], points[index - 1]["y"]])
        p1 = np.array([points[index]["x"], points[index]["y"]])
        p2 = np.array([points[index + 1]["x"], points[index + 1]["y"]])
        first = p1 - p0
        second = p2 - p0
        area2 = abs(float(first[0] * second[1] - first[1] * second[0]))
        lengths = (
            np.linalg.norm(p1 - p0)
            * np.linalg.norm(p2 - p1)
            * np.linalg.norm(p2 - p0)
        )
        curvature = float(2.0 * area2 / max(lengths, np.finfo(np.float64).tiny))
        points[index]["discrete_curvature"] = curvature
        points[index]["adjacent_chords_non_degenerate"] = bool(
            chord_lengths[index - 1] >= minimum_chord
            and chord_lengths[index] >= minimum_chord
        )
        if points[index]["adjacent_chords_non_degenerate"]:
            curvatures.append((curvature, index))
    if not curvatures:
        raise RuntimeError("all apparent L-curve knees use degenerate chords")
    curvature, selected_index = max(curvatures)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise RuntimeError("lambda sweep did not identify a finite L-curve knee")
    selected = points[selected_index]
    return {
        "selected_lambda": selected["lambda"],
        "selection_rule": (
            "maximum interior discrete curvature of median log10(data misfit) "
            "versus log10(control RMS); no true-A information"
        ),
        "selected_curvature": curvature,
        "minimum_adjacent_chord": minimum_chord,
        "degenerate_chord_rule": "reject either adjacent chord shorter than 1e-3 of total log-log arc length",
        "points": points,
    }


def reselect_regularization(configuration_path, sweep_directory):
    """Correct a degenerate-chord L-curve selection without repeating probes."""

    configuration = load_fiml_configuration(configuration_path)
    root = Path(sweep_directory)
    path = root / "regularization_sweep.json"
    result = read_json_record(path)
    for horizon in HORIZONS:
        section = result["horizons"][str(horizon)]
        selection = _lcurve_selection(
            section["records"], configuration["field_inversion"]["lambda_candidates"]
        )
        selected_lambda = float(selection["selected_lambda"])
        problem = _build_case(configuration, horizon, field_control=True)
        posthoc = []
        for start in section["representative_window_origins"]:
            active = FieldInversionWindowObjective(
                problem, horizon, start, selected_lambda
            )
            controls, accounting = _solve_exact_lbfgs(
                active,
                active.zero_controls(),
                int(section["accepted_iteration_cap_per_probe"]),
            )
            posthoc.append(
                {
                    "window_origin": int(start),
                    "selected_control_optimizer": accounting,
                    **_control_truth_metrics(active, controls),
                }
            )
            _npz_atomic(
                root / f"H{horizon}_start_{int(start):03d}_selected_controls.npz",
                controls=np.asarray(controls, dtype=np.float64),
            )
        section["selection"] = selection
        section["posthoc_truth_A_diagnostics_after_selection"] = posthoc
    result["selection_revision"] = (
        "near-coincident log-log points are excluded from curvature selection; "
        "no true-A information used"
    )
    write_json_record(path, result)
    return result


def run_regularization_sweep(configuration_path, output_directory):
    """Short representative-window L-curve study; explicitly nonscientific."""

    configuration = load_fiml_configuration(configuration_path)
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError("refusing to overwrite regularization sweep")
    root.mkdir(parents=True)
    iterations = int(
        configuration["field_inversion"]["lambda_sweep_accepted_iteration_cap"]
    )
    result = {
        "status": "complete",
        "interpretation": "REPRESENTATIVE REGULARIZATION PREPARATION SWEEP",
        "configuration_sha256": _canonical_json_sha256(configuration),
        "lambda_selection_used_true_A": False,
        "states_after_80_accessed": False,
        "horizons": {},
    }
    for horizon in HORIZONS:
        problem = _build_case(configuration, horizon, field_control=True)
        records = []
        final_controls = {}
        starts = configuration["field_inversion"]["representative_starts"][str(horizon)]
        for start in starts:
            zero_objective = FieldInversionWindowObjective(problem, horizon, start, 0.0)
            zero = zero_objective.zero_controls()
            zero_value, zero_gradient = zero_objective.value_and_gradient(zero)
            dimension = int(np.prod(zero.shape))
            natural = {
                "zero_data_misfit": zero_value,
                "zero_endpoint_gradient_norm": float(tree_norm(zero_gradient)),
                "unit_rms_regularizer_gradient_norm": 2.0 / np.sqrt(dimension),
            }
            for regularization_lambda in configuration["field_inversion"][
                "lambda_candidates"
            ]:
                active = FieldInversionWindowObjective(
                    problem, horizon, start, regularization_lambda
                )
                final, accounting = _solve_exact_lbfgs(active, zero, iterations)
                diagnostics = active.diagnostics(final)
                record = {
                    "horizon": horizon,
                    "window_origin": int(start),
                    "observed_truth_indices": [int(start), int(start + horizon)],
                    "intermediate_truth_indices_used": [],
                    "lambda": float(regularization_lambda),
                    **natural,
                    "initial_objective": zero_value,
                    "final_objective": diagnostics["objective"],
                    "final_data_misfit": diagnostics["data_misfit"],
                    "final_regularization_contribution": diagnostics[
                        "regularization_contribution"
                    ],
                    "final_control_rms": diagnostics["control_rms"],
                    "final_control_maximum_absolute": diagnostics[
                        "control_maximum_absolute"
                    ],
                    "final_gradient_norm": diagnostics["gradient_norm"],
                    "control_sha256": control_sha256(final),
                    "optimizer": accounting,
                }
                records.append(record)
                final_controls[(int(start), float(regularization_lambda))] = final
                print(json.dumps({"event": "lambda_sweep", **record}, sort_keys=True), flush=True)
        selection = _lcurve_selection(
            records, configuration["field_inversion"]["lambda_candidates"]
        )
        selected_lambda = float(selection["selected_lambda"])
        posthoc = []
        for start in starts:
            active = FieldInversionWindowObjective(
                problem, horizon, start, selected_lambda
            )
            controls = final_controls[(int(start), selected_lambda)]
            posthoc.append(
                {
                    "window_origin": int(start),
                    **_control_truth_metrics(active, controls),
                }
            )
            path = root / f"H{horizon}_start_{int(start):03d}_selected_controls.npz"
            np.savez(path, controls=np.asarray(controls, dtype=np.float64))
        result["horizons"][str(horizon)] = {
            "representative_window_origins": list(starts),
            "accepted_iteration_cap_per_probe": iterations,
            "records": records,
            "selection": selection,
            "posthoc_truth_A_diagnostics_after_selection": posthoc,
        }
    write_json_record(root / "regularization_sweep.json", result)
    return result


def _timed(callable_object, repeats):
    values = []
    result = None
    for _ in range(int(repeats)):
        started = perf_counter()
        result = callable_object()
        values.append(perf_counter() - started)
    return {
        "first_seconds": float(values[0]),
        "steady_median_seconds": float(np.median(values[1:] or values)),
        "all_seconds": [float(value) for value in values],
    }, result


def _extract_pseudo_arrays(field_objective, controls):
    tape = field_objective.tape(controls)
    feature_fields = []
    targets = []
    analytical_targets = []
    step_records = []
    offset = 0
    for internal, cache in enumerate(tape.step_caches, 1):
        moist = cache.children[-1].cache
        feature = np.stack(
            [
                np.asarray(moist.packed_state[key], dtype=np.float64)
                for key in ("h", "S", "Qv", "Qc")
            ]
            + [np.asarray(moist.packed_fields["B"], dtype=np.float64)],
            axis=-1,
        ).reshape(-1, 5)
        target = np.asarray(moist.rates["A"], dtype=np.float64).reshape(-1)
        analytical_target = np.asarray(
            moist.gll_diagnostics["analytical_A_reference"], dtype=np.float64
        ).reshape(-1)
        feature_fields.append(feature)
        targets.append(target)
        analytical_targets.append(analytical_target)
        step_records.append(
            {
                "window_origin": int(field_objective.start),
                "internal_step": int(internal),
                "sample_offset": int(offset),
                "sample_count": int(target.size),
                "control_sha256": control_sha256(np.asarray(controls)[internal - 1]),
            }
        )
        offset += target.size
    return (
        np.concatenate(feature_fields),
        np.concatenate(targets),
        np.concatenate(analytical_targets),
        step_records,
    )


def _stage2_context(configuration, features, targets):
    parameters, model_configuration = load_mlp_parameters(
        configuration["baseline"]["parameter_file"]
    )
    model = DenseMLP(model_configuration)
    normalization = normalization_from_record(
        load_compatible_neural_physics(
            configuration["baseline"]["embedding_configuration"],
            configuration["baseline"]["parameter_file"],
            expected_pytree_sha256=H1_BASELINE_SHA256,
        ).normalization.to_record()
    )
    x = jnp.asarray(normalization.normalize_features(features), dtype=jnp.float64)
    y = jnp.asarray(normalization.normalize_a(targets), dtype=jnp.float64).reshape(-1, 1)

    def objective(active_parameters):
        difference = model(active_parameters, x) - y
        return jnp.mean(difference * difference)

    return parameters, model_configuration, normalization, objective


def _npz_atomic(path, **arrays):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(destination)


def build_pseudo_label_dataset(
    configuration_path, horizon, controls_directory, output_path
):
    configuration = load_fiml_configuration(configuration_path)
    horizon = int(horizon)
    sweep = read_json_record(
        Path(configuration["output_root"]) / "regularization-sweep" / "regularization_sweep.json"
    )
    selected_lambda = float(
        sweep["horizons"][str(horizon)]["selection"]["selected_lambda"]
    )
    problem = _build_case(configuration, horizon, field_control=True)
    features = []
    targets = []
    analytical_targets = []
    steps = []
    control_records = []
    for start in sparse_starts(horizon):
        metadata_path = Path(controls_directory) / f"window_{start:03d}.json"
        controls_path = Path(controls_directory) / f"window_{start:03d}.npz"
        metadata = read_json_record(metadata_path)
        if metadata.get("status") != "complete":
            raise ValueError(f"FI window {start} is incomplete")
        if float(metadata["lambda"]) != selected_lambda:
            raise ValueError(f"FI window {start} lambda differs")
        values = np.load(controls_path, allow_pickle=False)
        controls = jnp.asarray(values["controls"], dtype=jnp.float64)
        if control_sha256(controls) != metadata["control_sha256"]:
            raise ValueError(f"FI window {start} control fingerprint changed")
        active = FieldInversionWindowObjective(problem, horizon, start, selected_lambda)
        x, y, analytical, local_steps = _extract_pseudo_arrays(active, controls)
        features.append(x)
        targets.append(y)
        analytical_targets.append(analytical)
        steps.extend(local_steps)
        control_records.append(metadata)
    x = np.concatenate(features)
    y = np.concatenate(targets)
    analytical = np.concatenate(analytical_targets)
    expected = 80 * 256 * 16
    if x.shape != (expected, 5) or y.shape != (expected,):
        raise RuntimeError("FIML pseudo-label sample accounting changed")
    _npz_atomic(output_path, features=x, targets=y)
    posthoc_path = Path(output_path).with_name(
        Path(output_path).stem + "_posthoc_truth_A.npz"
    )
    _npz_atomic(posthoc_path, analytical_A_on_FI_states=analytical)
    metadata = {
        "status": "complete",
        "horizon": horizon,
        "sample_count": int(y.size),
        "model_step_fields": 80,
        "samples_per_model_step": 4096,
        "feature_order": ["h", "S", "Qv", "Qc", "B"],
        "representation": "cell-local 256 cells x 4x4 GLL; shared CG points not deduplicated",
        "pseudo_target": "A_H1(Y_FI)+A_scale*c_opt",
        "intermediate_truth_used": False,
        "baseline_parameter_pytree_sha256": H1_BASELINE_SHA256,
        "lambda": selected_lambda,
        "step_provenance": steps,
        "field_control_metadata": control_records,
        "dataset_npz_sha256": _file_sha256(output_path),
        "posthoc_truth_A_file": str(posthoc_path.resolve()),
        "posthoc_truth_A_file_npz_sha256": _file_sha256(posthoc_path),
        "posthoc_inferred_A_versus_analytical_A": operator_metrics(y, analytical),
        "posthoc_truth_A_used_by_stage2_training": False,
    }
    write_json_record(Path(output_path).with_suffix(".json"), metadata)
    return metadata


class _JAXValueGradientObjective:
    """Small value/gradient interface used by the offline Stage-2 smoke."""

    def __init__(self, objective):
        self._value = jax.jit(objective)
        self._value_gradient = jax.jit(jax.value_and_grad(objective))
        self.value_evaluations = 0
        self.gradient_evaluations = 0

    def value(self, parameters):
        self.value_evaluations += 1
        return float(self._value(parameters))

    def value_and_gradient(self, parameters):
        self.gradient_evaluations += 1
        value, gradient = self._value_gradient(parameters)
        return float(value), gradient

    def gradient(self, parameters):
        return self.value_and_gradient(parameters)[1]


def _selected_lambda(configuration, horizon):
    sweep_path = (
        Path(configuration["output_root"])
        / "regularization-sweep"
        / "regularization_sweep.json"
    )
    sweep = read_json_record(sweep_path)
    if (
        sweep.get("status") != "complete"
        or sweep.get("configuration_sha256") != _canonical_json_sha256(configuration)
        or sweep.get("lambda_selection_used_true_A") is not False
    ):
        raise ValueError("lambda selection contract changed")
    return float(
        sweep["horizons"][str(int(horizon))]["selection"]["selected_lambda"]
    )


def benchmark_and_smoke(
    configuration_path, output_directory, *, repeats=3, horizon=None
):
    """Full production-semantics timings plus tiny nonscientific smokes."""

    configuration = load_fiml_configuration(configuration_path)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("refusing to overwrite FIML benchmark/smoke output")
    output.mkdir(parents=True)
    baseline, _ = load_mlp_parameters(configuration["baseline"]["parameter_file"])
    records = {
        "status": "complete",
        "interpretation": "NONSCIENTIFIC IMPLEMENTATION TIMINGS AND SMOKES",
        "configuration_sha256": _canonical_json_sha256(configuration),
        "direct": {},
        "field_inversion": {},
        "stage2": {},
        "states_after_80_accessed": False,
    }
    representative_arrays = {}
    active_horizons = HORIZONS if horizon is None else (int(horizon),)
    if any(value not in HORIZONS for value in active_horizons):
        raise ValueError("benchmark horizon must be 2 or 5")
    for horizon in active_horizons:
        direct_problem = _build_case(configuration, horizon, field_control=False)
        direct = build_direct_objective(direct_problem, horizon)
        direct.value(baseline)
        direct.value_and_gradient(baseline)
        direct_cached_value_time, _ = _timed(
            lambda: direct.value(baseline), repeats
        )

        def fresh_direct_value():
            direct.clear_parameter_tape()
            return direct.value(baseline)

        def fresh_direct_value_gradient():
            direct.clear_parameter_tape()
            return direct.value_and_gradient(baseline)

        direct_value_time, _ = _timed(fresh_direct_value, repeats)
        direct_gradient_time, _ = _timed(fresh_direct_value_gradient, repeats)
        initial_value = direct.value(baseline)
        direct_final, direct_smoke = _solve_exact_lbfgs(
            direct,
            baseline,
            int(configuration["preparation"]["direct_smoke_accepted_iterations"]),
        )
        final_value = direct.value(direct_final)
        if not final_value < initial_value:
            raise RuntimeError(f"H{horizon} direct sparse smoke did not descend")
        direct_path = output / f"direct_H{horizon}_smoke_parameters.npz"
        save_mlp_parameters_atomic(
            direct_path, direct_final, direct_problem.model_configuration
        )
        records["direct"][str(horizon)] = {
            "window_count": len(sparse_starts(horizon)),
            "observed_truth_indices": list(sparse_observation_indices(horizon)),
            "intermediate_truth_indices_used": [],
            "full_value_timing": direct_value_time,
            "full_value_and_gradient_timing": direct_gradient_time,
            "same_theta_cached_value_timing": direct_cached_value_time,
            "smoke": {
                **direct_smoke,
                "interpretation": "NONSCIENTIFIC DIRECT SPARSE-ENDPOINT SMOKE",
                "initial_objective": initial_value,
                "final_objective": final_value,
                "objective_decreased": True,
                "final_parameter_pytree_sha256": parameter_pytree_sha256(
                    direct_final
                ),
            },
        }
        direct.clear_parameter_tape()
        del direct, direct_problem, direct_final
        gc.collect()

        field_problem = _build_case(configuration, horizon, field_control=True)
        selected_lambda = _selected_lambda(configuration, horizon)
        representative_start = configuration["field_inversion"][
            "representative_starts"
        ][str(horizon)][2]
        field = FieldInversionWindowObjective(
            field_problem, horizon, representative_start, selected_lambda
        )
        zero = field.zero_controls()
        field.value(zero)
        field.value_and_gradient(zero)
        field_cached_value_time, _ = _timed(lambda: field.value(zero), repeats)

        def fresh_field_value():
            field._last_tape = None
            return field.value(zero)

        def fresh_field_value_gradient():
            field._last_tape = None
            return field.value_and_gradient(zero)

        field_value_time, _ = _timed(fresh_field_value, repeats)
        field_gradient_time, _ = _timed(fresh_field_value_gradient, repeats)
        field_initial = field.value(zero)
        field_final, field_smoke = _solve_exact_lbfgs(
            field,
            zero,
            int(
                configuration["preparation"][
                    "field_inversion_smoke_accepted_iterations"
                ]
            ),
        )
        field_final_value = field.value(field_final)
        if not field_final_value < field_initial:
            raise RuntimeError(f"H{horizon} FI smoke did not descend")
        _npz_atomic(
            output / f"FI_H{horizon}_smoke_controls.npz",
            controls=np.asarray(field_final, dtype=np.float64),
        )
        x, y, _, step_records = _extract_pseudo_arrays(field, field_final)
        representative_arrays[horizon] = (x, y)
        records["field_inversion"][str(horizon)] = {
            "selected_lambda": selected_lambda,
            "representative_window_origin": representative_start,
            "control_dimension": int(np.prod(field.control_shape)),
            "per_window_value_timing": field_value_time,
            "per_window_value_and_gradient_timing": field_gradient_time,
            "same_control_cached_value_timing": field_cached_value_time,
            "projected_serial_all_window_value_seconds": field_value_time[
                "steady_median_seconds"
            ]
            * len(sparse_starts(horizon)),
            "projected_serial_all_window_value_and_gradient_seconds": field_gradient_time[
                "steady_median_seconds"
            ]
            * len(sparse_starts(horizon)),
            "smoke": {
                **field_smoke,
                "interpretation": "NONSCIENTIFIC FIELD-INVERSION WINDOW SMOKE",
                "initial_objective": field_initial,
                "final_objective": field_final_value,
                "objective_decreased": True,
                "final_control_sha256": control_sha256(field_final),
            },
            "representative_pseudo_label_step_records": step_records,
        }
        field._last_tape = None
        del field, field_problem, field_final
        gc.collect()

    for horizon, (features, targets) in representative_arrays.items():
        initial, model_configuration, normalization, objective = _stage2_context(
            configuration, features, targets
        )
        active = _JAXValueGradientObjective(objective)
        active.value(initial)
        active.value_and_gradient(initial)
        value_time, _ = _timed(lambda: active.value(initial), repeats)
        gradient_time, _ = _timed(
            lambda: active.value_and_gradient(initial), repeats
        )
        initial_value = active.value(initial)
        final, accounting = _solve_exact_lbfgs(
            active,
            initial,
            int(configuration["preparation"]["stage2_smoke_accepted_iterations"]),
        )
        final_value = active.value(final)
        if not final_value < initial_value:
            raise RuntimeError(f"H{horizon} Stage-2 smoke did not descend")
        path = output / f"stage2_H{horizon}_smoke_parameters.npz"
        save_mlp_parameters_atomic(path, final, model_configuration)
        predictions = physical_predictions(
            final, DenseMLP(model_configuration), normalization, features
        )
        records["stage2"][str(horizon)] = {
            "representative_sample_count": int(targets.size),
            "solver_calls_per_objective_or_gradient": 0,
            "value_timing": value_time,
            "value_and_gradient_timing": gradient_time,
            "smoke": {
                **accounting,
                "interpretation": "NONSCIENTIFIC OFFLINE STAGE-2 SMOKE",
                "initial_objective": initial_value,
                "final_objective": final_value,
                "objective_decreased": True,
                "pseudo_label_metrics": operator_metrics(predictions, targets),
                "final_parameter_pytree_sha256": parameter_pytree_sha256(final),
            },
        }
    write_json_record(output / "benchmark_and_smokes.json", records)
    return records


def merge_benchmark_records(h2_path, h5_path, output_path):
    h2 = read_json_record(h2_path)
    h5 = read_json_record(h5_path)
    for record, horizon in ((h2, "2"), (h5, "5")):
        if record.get("states_after_80_accessed", True):
            raise ValueError("benchmark record accessed held-out truth")
        for family in ("direct", "field_inversion", "stage2"):
            if set(record[family]) != {horizon}:
                raise ValueError("isolated benchmark horizon changed")
    merged = {
        "status": "complete",
        "interpretation": "MERGED ISOLATED NONSCIENTIFIC IMPLEMENTATION TIMINGS AND SMOKES",
        "configuration_sha256": h2["configuration_sha256"],
        "direct": {**h2["direct"], **h5["direct"]},
        "field_inversion": {
            **h2["field_inversion"],
            **h5["field_inversion"],
        },
        "stage2": {**h2["stage2"], **h5["stage2"]},
        "isolated_processes_used_to_bound_trajectory_tape_memory": True,
        "states_after_80_accessed": False,
    }
    if h2["configuration_sha256"] != h5["configuration_sha256"]:
        raise ValueError("benchmark configuration fingerprints differ")
    write_json_record(output_path, merged)
    return merged


def benchmark_stage2_full_batch(configuration_path, output_path, *, repeats=5):
    """Pure-JAX production-size Stage-2 cost without solver or FI truth use."""

    configuration = load_fiml_configuration(configuration_path)
    dataset, _ = load_operator_dataset(configuration["model"]["operator_dataset"])
    sample_count = 80 * 256 * 16
    features = np.asarray(dataset.features[:sample_count], dtype=np.float64)
    baseline, model_configuration = load_mlp_parameters(
        configuration["baseline"]["parameter_file"]
    )
    physics = load_compatible_neural_physics(
        configuration["baseline"]["embedding_configuration"],
        configuration["baseline"]["parameter_file"],
        expected_pytree_sha256=H1_BASELINE_SHA256,
    )
    normalization = physics.normalization
    targets = physical_predictions(
        baseline, DenseMLP(model_configuration), normalization, features
    )
    initial, _, _, function = _stage2_context(configuration, features, targets)
    active = _JAXValueGradientObjective(function)
    active.value(initial)
    active.value_and_gradient(initial)
    value, _ = _timed(lambda: active.value(initial), repeats)
    gradient, _ = _timed(lambda: active.value_and_gradient(initial), repeats)
    record = {
        "status": "complete",
        "interpretation": "PURE-JAX PRODUCTION-SIZE STAGE-2 COMPUTE BENCHMARK",
        "sample_count": sample_count,
        "feature_count": 5,
        "parameter_count": 1281,
        "targets": "H1 predictions used only to instantiate an exact-size timing objective",
        "value_timing": value,
        "value_and_gradient_timing": gradient,
        "solver_calls": 0,
        "states_after_80_accessed": False,
    }
    write_json_record(output_path, record)
    return record


def _checkpoint_iterations(limit):
    candidates = {0, int(limit)}
    candidates.update(
        value for value in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000, 25000, 50000)
        if value <= int(limit)
    )
    return candidates


def train_direct_sparse(
    configuration_path, horizon, iteration_limit, output_directory, *, resume=False
):
    """User-launched direct sparse endpoint fit with parameter-only recovery."""

    configuration = load_fiml_configuration(configuration_path)
    horizon = int(horizon)
    problem = _build_case(configuration, horizon, field_control=False)
    objective = build_direct_objective(problem, horizon)
    initial, _ = load_mlp_parameters(configuration["baseline"]["parameter_file"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "fit_result.json"
    progress_path = output / "fit_progress.json"
    if result_path.exists():
        raise FileExistsError("direct sparse fit is already complete")
    offset = 0
    start = initial
    if resume:
        progress = read_json_record(progress_path)
        if progress["configuration_sha256"] != _canonical_json_sha256(configuration):
            raise ValueError("direct sparse resume configuration changed")
        if progress["baseline_parameter_pytree_sha256"] != H1_BASELINE_SHA256:
            raise ValueError("direct sparse baseline changed")
        offset = int(progress["accepted_iteration"])
        start, _ = load_mlp_parameters(progress["parameter_file"])
        if parameter_pytree_sha256(start) != progress["parameter_pytree_sha256"]:
            raise ValueError("direct sparse resume parameter fingerprint changed")
    elif progress_path.exists():
        raise FileExistsError("incomplete fit exists; use explicit --resume")
    checkpoints = _checkpoint_iterations(iteration_limit)

    if not resume:
        initial_path = output / "parameters_iter_000000.npz"
        save_mlp_parameters_atomic(
            initial_path, initial, problem.model_configuration
        )
        write_json_record(
            progress_path,
            {
                "status": "in_progress",
                "horizon": horizon,
                "accepted_iteration": 0,
                "configuration_sha256": _canonical_json_sha256(configuration),
                "baseline_parameter_pytree_sha256": H1_BASELINE_SHA256,
                "parameter_pytree_sha256": H1_BASELINE_SHA256,
                "parameter_file": str(initial_path.resolve()),
                "parameter_only_resume_restores_LBFGS_history": False,
            },
        )

    def callback(control, local_index, adapter):
        if local_index == 0:
            return
        global_index = offset + local_index
        parameters = adapter.pytree_from_vector(control)
        record = {
            "status": "in_progress",
            "horizon": horizon,
            "accepted_iteration": global_index,
            "configuration_sha256": _canonical_json_sha256(configuration),
            "baseline_parameter_pytree_sha256": H1_BASELINE_SHA256,
            "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
            "objective_evaluations_this_process": adapter.value_evaluations,
            "gradient_evaluations_this_process": adapter.gradient_evaluations,
            "parameter_only_resume_restores_LBFGS_history": False,
        }
        print(json.dumps({"event": "direct_progress", **record}, sort_keys=True), flush=True)
        if global_index in checkpoints:
            path = output / f"parameters_iter_{global_index:06d}.npz"
            save_mlp_parameters_atomic(path, parameters, problem.model_configuration)
            record["parameter_file"] = str(path.resolve())
            write_json_record(progress_path, record)

    remaining = int(iteration_limit) - offset
    if remaining <= 0:
        raise ValueError("direct sparse budget already exhausted")
    final, accounting = _solve_exact_lbfgs(
        objective, start, remaining, accepted_callback=callback
    )
    final_path = output / "final_parameters.npz"
    save_mlp_parameters_atomic(final_path, final, problem.model_configuration)
    result = {
        "status": "complete",
        "method": f"direct sparse endpoint H{horizon}",
        "observed_truth_indices": list(sparse_observation_indices(horizon)),
        "intermediate_truth_indices_used": [],
        "initialization_parameter_pytree_sha256": H1_BASELINE_SHA256,
        "new_optimizer_process": True,
        "source_optimizer_secant_history_reused": False,
        "parameter_only_resume_restored_secant_history": False,
        "accepted_iteration_offset": offset,
        "optimizer": {
            **accounting,
            "cumulative_accepted_iterations": offset
            + int(accounting["accepted_iterations"]),
        },
        "final_objective": objective.value(final),
        "final_parameter_file": str(final_path.resolve()),
        "final_parameter_pytree_sha256": parameter_pytree_sha256(final),
        "states_after_80_accessed": False,
    }
    write_json_record(result_path, result)
    write_json_record(progress_path, {**result, "status": "complete"})
    return result


def train_field_inversion(
    configuration_path, horizon, iteration_limit, output_directory
):
    """User-launched deterministic serial independent-window FI campaign."""

    configuration = load_fiml_configuration(configuration_path)
    horizon = int(horizon)
    regularization_lambda = _selected_lambda(configuration, horizon)
    problem = _build_case(configuration, horizon, field_control=True)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for start in sparse_starts(horizon):
        control_path = output / f"window_{start:03d}.npz"
        metadata_path = output / f"window_{start:03d}.json"
        if metadata_path.exists():
            record = read_json_record(metadata_path)
            if (
                record.get("status") != "complete"
                or float(record["lambda"]) != regularization_lambda
                or record["baseline_parameter_pytree_sha256"] != H1_BASELINE_SHA256
                or record["configuration_sha256"] != _canonical_json_sha256(configuration)
            ):
                raise ValueError(f"existing FI window {start} is incompatible")
            controls = np.load(control_path, allow_pickle=False)["controls"]
            if control_sha256(controls) != record["control_sha256"]:
                raise ValueError(f"existing FI window {start} fingerprint changed")
            summary.append(record)
            continue
        active = FieldInversionWindowObjective(
            problem, horizon, start, regularization_lambda
        )
        zero = active.zero_controls()
        initial = active.diagnostics(zero)
        final, accounting = _solve_exact_lbfgs(active, zero, iteration_limit)
        final_diagnostics = active.diagnostics(final)
        _npz_atomic(control_path, controls=np.asarray(final, dtype=np.float64))
        record = {
            "status": "complete",
            "horizon": horizon,
            "window_origin": start,
            "observed_truth_indices": [start, start + horizon],
            "intermediate_truth_indices_used": [],
            "lambda": regularization_lambda,
            "configuration_sha256": _canonical_json_sha256(configuration),
            "baseline_parameter_pytree_sha256": H1_BASELINE_SHA256,
            "zero_control_initialized": True,
            "new_optimizer_process": True,
            "other_window_secant_histories_reused": False,
            "optimizer": accounting,
            "initial_diagnostics": initial,
            "final_diagnostics": final_diagnostics,
            "control_file": str(control_path.resolve()),
            "control_npz_sha256": _file_sha256(control_path),
            "control_sha256": control_sha256(final),
            "states_after_80_accessed": False,
        }
        write_json_record(metadata_path, record)
        summary.append(record)
    result = {
        "status": "complete",
        "horizon": horizon,
        "window_count": len(summary),
        "lambda": regularization_lambda,
        "serial_reference": True,
        "process_parallelism_used": False,
        "records": summary,
        "states_after_80_accessed": False,
    }
    write_json_record(output / "field_inversion_summary.json", result)
    return result


def train_stage2(
    configuration_path,
    horizon,
    dataset_path,
    iteration_limit,
    output_directory,
    *,
    resume=False,
):
    configuration = load_fiml_configuration(configuration_path)
    metadata = read_json_record(Path(dataset_path).with_suffix(".json"))
    if metadata["horizon"] != int(horizon) or metadata["sample_count"] != 327680:
        raise ValueError("Stage-2 pseudo-label dataset is incompatible")
    if metadata.get("intermediate_truth_used") is not False:
        raise ValueError("intermediate truth entered Stage-2 pseudo-labels")
    values = np.load(dataset_path, allow_pickle=False)
    features = np.asarray(values["features"], dtype=np.float64)
    targets = np.asarray(values["targets"], dtype=np.float64)
    initial, model_configuration, normalization, function = _stage2_context(
        configuration, features, targets
    )
    active = _JAXValueGradientObjective(function)
    output = Path(output_directory)
    result_path = output / "fit_result.json"
    progress_path = output / "fit_progress.json"
    if result_path.exists():
        raise FileExistsError("Stage-2 fit is already complete")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_iterations(iteration_limit)
    offset = 0
    start = initial
    if resume:
        progress = read_json_record(progress_path)
        if (
            int(progress["horizon"]) != int(horizon)
            or progress["configuration_sha256"]
            != _canonical_json_sha256(configuration)
            or progress["pseudo_label_dataset_npz_sha256"]
            != _file_sha256(dataset_path)
        ):
            raise ValueError("Stage-2 resume contract changed")
        offset = int(progress["accepted_iteration"])
        start, _ = load_mlp_parameters(progress["parameter_file"])
        if parameter_pytree_sha256(start) != progress["parameter_pytree_sha256"]:
            raise ValueError("Stage-2 resume parameter fingerprint changed")
    elif progress_path.exists():
        raise FileExistsError("incomplete Stage-2 fit exists; use explicit --resume")
    else:
        initial_path = output / "parameters_iter_000000.npz"
        save_mlp_parameters_atomic(initial_path, initial, model_configuration)
        write_json_record(
            progress_path,
            {
                "status": "in_progress",
                "horizon": int(horizon),
                "configuration_sha256": _canonical_json_sha256(configuration),
                "pseudo_label_dataset_npz_sha256": _file_sha256(dataset_path),
                "accepted_iteration": 0,
                "parameter_file": str(initial_path.resolve()),
                "parameter_pytree_sha256": H1_BASELINE_SHA256,
                "parameter_only_resume_restores_LBFGS_history": False,
            },
        )

    def callback(control, local_index, adapter):
        global_index = offset + local_index
        if local_index == 0 or global_index not in checkpoints:
            return
        parameters = adapter.pytree_from_vector(control)
        path = output / f"parameters_iter_{global_index:06d}.npz"
        save_mlp_parameters_atomic(path, parameters, model_configuration)
        record = {
            "status": "in_progress",
            "horizon": int(horizon),
            "configuration_sha256": _canonical_json_sha256(configuration),
            "pseudo_label_dataset_npz_sha256": _file_sha256(dataset_path),
            "accepted_iteration": global_index,
            "parameter_file": str(path.resolve()),
            "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
            "objective_evaluations": adapter.value_evaluations,
            "gradient_evaluations": adapter.gradient_evaluations,
            "parameter_only_resume_restores_LBFGS_history": False,
        }
        print(json.dumps({"event": "stage2_progress", **record}, sort_keys=True), flush=True)
        write_json_record(progress_path, record)

    remaining = int(iteration_limit) - offset
    if remaining <= 0:
        raise ValueError("Stage-2 iteration budget already exhausted")
    final, accounting = _solve_exact_lbfgs(
        active, start, remaining, accepted_callback=callback
    )
    final_path = output / "final_parameters.npz"
    save_mlp_parameters_atomic(final_path, final, model_configuration)
    predictions = physical_predictions(
        final, DenseMLP(model_configuration), normalization, features
    )
    posthoc = np.load(metadata["posthoc_truth_A_file"], allow_pickle=False)
    analytical = np.asarray(posthoc["analytical_A_on_FI_states"], dtype=np.float64)
    result = {
        "status": "complete",
        "method": f"FIML-H{int(horizon)} offline Stage 2",
        "initial_parameter_pytree_sha256": H1_BASELINE_SHA256,
        "accepted_iteration_offset": offset,
        "parameter_only_resume_restored_secant_history": False,
        "pseudo_label_dataset": str(Path(dataset_path).resolve()),
        "pseudo_label_dataset_npz_sha256": _file_sha256(dataset_path),
        "solver_calls": 0,
        "optimizer": {
            **accounting,
            "cumulative_accepted_iterations": offset
            + int(accounting["accepted_iterations"]),
        },
        "final_objective": active.value(final),
        "pseudo_label_regression_metrics": operator_metrics(predictions, targets),
        "true_A_metrics_on_FI_states": operator_metrics(predictions, analytical),
        "true_A_used_by_stage2_objective": False,
        "final_parameter_file": str(final_path.resolve()),
        "final_parameter_pytree_sha256": parameter_pytree_sha256(final),
        "states_after_80_accessed": False,
    }
    write_json_record(result_path, result)
    write_json_record(progress_path, {**result, "status": "complete"})
    return result


def cross_evaluate_sparse_network(
    configuration_path, parameter_file, expected_sha256, output_path
):
    """Read-only H2/H5 endpoint and direct-A diagnostics for one network."""

    configuration = load_fiml_configuration(configuration_path)
    parameters, model_configuration = load_mlp_parameters(parameter_file)
    actual_sha = parameter_pytree_sha256(parameters)
    if actual_sha != expected_sha256:
        raise ValueError("cross-evaluation parameter fingerprint changed")
    sparse = {}
    for horizon in HORIZONS:
        problem = _build_case(configuration, horizon, field_control=False)
        sparse[str(horizon)] = build_direct_objective(problem, horizon).value(
            parameters
        )
    dataset, metadata = load_operator_dataset(
        configuration["model"]["operator_dataset"]
    )
    normalization = normalization_from_record(metadata["normalization"])
    predictions = physical_predictions(
        parameters, DenseMLP(model_configuration), normalization, dataset.features
    )
    result = {
        "status": "complete",
        "parameter_file": str(Path(parameter_file).resolve()),
        "parameter_pytree_sha256": actual_sha,
        "sparse_endpoint_objectives": {"H2": sparse["2"], "H5": sparse["5"]},
        "J_op": float(
            np.mean(
                (
                    (predictions - dataset.targets.reshape(-1))
                    / normalization.output_scale
                )
                ** 2
            )
        ),
        "direct_A_metrics": operator_metrics(predictions, dataset.targets),
        "autonomous_metrics_used_for_selection": False,
        "states_after_80_accessed": False,
    }
    write_json_record(output_path, result)
    return result


def write_production_plan(
    configuration_path, benchmark_path, stage2_benchmark_path, output_path
):
    """Freeze transparent preparation-derived budgets without starting work."""

    configuration = load_fiml_configuration(configuration_path)
    benchmark = read_json_record(benchmark_path)
    stage2_full = read_json_record(stage2_benchmark_path)
    if stage2_full["sample_count"] != 80 * 256 * 16 or stage2_full["solver_calls"] != 0:
        raise ValueError("production-size Stage-2 benchmark contract changed")
    budgets = {
        "direct": {"2": 100, "5": 100},
        "field_inversion_per_window": {"2": 25, "5": 50},
        "stage2": {"2": 50000, "5": 50000},
    }
    sweep_path = (
        Path(configuration["output_root"])
        / "regularization-sweep"
        / "regularization_sweep.json"
    )
    selected_lambdas = {
        str(horizon): _selected_lambda(configuration, horizon)
        for horizon in HORIZONS
    }
    projections = {"direct": {}, "field_inversion_serial": {}, "stage2": {}}
    for horizon in HORIZONS:
        key = str(horizon)
        direct_smoke = benchmark["direct"][key]["smoke"]
        direct_rate = direct_smoke["wall_time_seconds"] / max(
            1, direct_smoke["accepted_iterations"]
        )
        fi_smoke = benchmark["field_inversion"][key]["smoke"]
        fi_rate = fi_smoke["wall_time_seconds"] / max(1, fi_smoke["accepted_iterations"])
        stage2_smoke = benchmark["stage2"][key]["smoke"]
        stage2_rate = (
            float(stage2_smoke["objective_evaluations"])
            / max(1, stage2_smoke["accepted_iterations"])
            * stage2_full["value_timing"]["steady_median_seconds"]
            + float(stage2_smoke["gradient_evaluations"])
            / max(1, stage2_smoke["accepted_iterations"])
            * stage2_full["value_and_gradient_timing"]["steady_median_seconds"]
        )
        projections["direct"][key] = {
            "smoke_seconds_per_accepted_iteration": direct_rate,
            "candidate_seconds": {
                str(count): direct_rate * count for count in (25, 50, 100, 250)
            },
            "recommended_cap": budgets["direct"][key],
        }
        projections["field_inversion_serial"][key] = {
            "smoke_seconds_per_window_accepted_iteration": fi_rate,
            "window_count": len(sparse_starts(horizon)),
            "candidate_all_window_seconds": {
                str(count): fi_rate * count * len(sparse_starts(horizon))
                for count in (10, 25, 50, 100)
            },
            "recommended_cap_per_window": budgets["field_inversion_per_window"][key],
        }
        projections["stage2"][key] = {
            "representative_smoke_seconds_per_accepted_iteration": stage2_rate,
            "candidate_seconds": {
                str(count): stage2_rate * count for count in (1000, 5000, 10000, 50000)
            },
            "recommended_cap": budgets["stage2"][key],
            "solver_calls": 0,
        }
    direct_total = sum(
        projections["direct"][key]["smoke_seconds_per_accepted_iteration"]
        * budgets["direct"][key]
        for key in ("2", "5")
    )
    fi_total = sum(
        projections["field_inversion_serial"][key][
            "smoke_seconds_per_window_accepted_iteration"
        ]
        * budgets["field_inversion_per_window"][key]
        * projections["field_inversion_serial"][key]["window_count"]
        for key in ("2", "5")
    )
    offline_total = sum(
        projections["stage2"][key]["representative_smoke_seconds_per_accepted_iteration"]
        * budgets["stage2"][key]
        for key in ("2", "5")
    )
    record = {
        "status": "ready",
        "configuration_sha256": _canonical_json_sha256(configuration),
        "baseline_parameter_pytree_sha256": H1_BASELINE_SHA256,
        "budgets": budgets,
        "regularization_selection": {
            "file": str(sweep_path.resolve()),
            "file_sha256": _file_sha256(sweep_path),
            "selected_lambdas": selected_lambdas,
            "used_true_A": False,
        },
        "projections": projections,
        "projected_primary_direct_seconds": direct_total,
        "projected_primary_FIML_seconds": fi_total + offline_total,
        "projection_is_authoritative": False,
        "caveat": "short nonscientific line-search behavior extrapolated linearly",
        "architecture_amortization_formula": "C_FIML(N)=C_FI+N*C_offline_ML; C_direct(N)=N*C_solver_in_loop",
        "break_even_architecture_count": (
            fi_total
            / max(direct_total - offline_total, np.finfo(np.float64).tiny)
            if direct_total > offline_total
            else None
        ),
        "states_after_80_accessed": False,
        "production_launched": False,
        "production_size_stage2_benchmark": stage2_full,
    }
    write_json_record(output_path, record)
    return record


def write_fiml_postprocess_report(root_directory, output_json, output_markdown):
    root = Path(root_directory)
    post = root / "postprocess"
    entries = []
    for label in ("h1-baseline", "direct-h2", "fiml-h2", "direct-h5", "fiml-h5"):
        sparse = read_json_record(post / "sparse" / f"{label}.json")
        dense = read_json_record(post / "dense" / f"{label}.json")
        rollout = read_json_record(post / "autonomous" / label / "rollout_summary.json")
        if sparse.get("states_after_80_accessed", True) or dense.get(
            "states_after_80_accessed", True
        ):
            raise ValueError("postprocessing accessed held-out truth")
        if rollout["deployment_contract"].get("states_after_80_accessed", True):
            raise ValueError("autonomous postprocessing accessed held-out truth")
        entries.append(
            {
                "label": label,
                "sparse_endpoint": sparse,
                "dense_and_fixed_support": dense,
                "autonomous_training_support": {
                    "mixed_state_error": rollout["mixed_state_error"],
                    "fieldwise_errors": rollout["fieldwise_errors"],
                    "off_manifold_A": rollout["aggregate_off_manifold_A_diagnostic"],
                    "kinetic_energy": rollout["kinetic_energy"],
                    "projected_enstrophy": rollout["projected_enstrophy"],
                    "rain_activity": rollout["rain_activity_summary"],
                    "source_structural_invariants": rollout[
                        "source_structural_invariants"
                    ],
                },
            }
        )
    fi = {
        key: read_json_record(root / "field-inversion" / key / "field_inversion_summary.json")
        for key in ("h2", "h5")
    }
    raw_data = {
        key: float(
            sum(item["final_diagnostics"]["data_misfit"] for item in record["records"])
        )
        for key, record in fi.items()
    }
    fi_cost = {
        key: {
            "window_count": len(record["records"]),
            "accepted_iterations": int(
                sum(item["optimizer"]["accepted_iterations"] for item in record["records"])
            ),
            "objective_evaluations": int(
                sum(item["optimizer"]["objective_evaluations"] for item in record["records"])
            ),
            "gradient_evaluations": int(
                sum(item["optimizer"]["gradient_evaluations"] for item in record["records"])
            ),
            "serial_equivalent_wall_seconds": float(
                sum(item["optimizer"]["wall_time_seconds"] for item in record["records"])
            ),
            "raw_stage1_sparse_endpoint_data_misfit": raw_data[key],
        }
        for key, record in fi.items()
    }
    network_by_label = {entry["label"]: entry for entry in entries}
    amortization = {}
    for key, horizon_label in (("h2", "H2"), ("h5", "H5")):
        nn_value = network_by_label[f"fiml-{key}"]["sparse_endpoint"][
            "sparse_endpoint_objectives"
        ][horizon_label]
        amortization[key] = {
            "raw_field_inversion_endpoint_data_misfit": raw_data[key],
            "stage2_network_endpoint_objective": nn_value,
            "absolute_compression_loss": float(nn_value - raw_data[key]),
            "ratio_stage2_to_raw_FI": float(
                nn_value / max(raw_data[key], np.finfo(np.float64).tiny)
            ),
        }
    direct_cost = {
        key: read_json_record(root / f"direct-endpoint-{key}" / "fit_result.json")[
            "optimizer"
        ]
        for key in ("h2", "h5")
    }
    stage2_cost = {
        key: read_json_record(root / "stage2" / key / "fit_result.json")["optimizer"]
        for key in ("h2", "h5")
    }
    amortized_cost = {}
    for key in ("h2", "h5"):
        direct_wall = float(direct_cost[key]["wall_time_seconds"])
        fi_wall = float(fi_cost[key]["serial_equivalent_wall_seconds"])
        offline_wall = float(stage2_cost[key]["wall_time_seconds"])
        saving_per_architecture = direct_wall - offline_wall
        amortized_cost[key] = {
            "direct_solver_in_loop_seconds_per_architecture": direct_wall,
            "one_time_field_inversion_serial_equivalent_seconds": fi_wall,
            "offline_stage2_seconds_per_architecture": offline_wall,
            "break_even_architecture_count": (
                float(fi_wall / saving_per_architecture)
                if saving_per_architecture > 0.0
                else None
            ),
        }
    record = {
        "status": "complete",
        "diagnostic": "TEST2A_FIML_SPARSE_ENDPOINT_H2_H5_MATCHED_POSTPROCESS",
        "information_set": (
            "all methods start from H1; sparse endpoint objectives use only origins "
            "and endpoints; intermediate truth is post-hoc only"
        ),
        "networks": entries,
        "raw_field_inversion": fi_cost,
        "amortization_compression_loss": amortization,
        "cost_accounting": {
            "direct_sparse_endpoint": direct_cost,
            "FIML_stage1": fi_cost,
            "FIML_stage2": stage2_cost,
            "architecture_amortization": amortized_cost,
            "formula": "C_FIML(N)=C_FI+N*C_offline_ML; C_direct(N)=N*C_solver_in_loop",
        },
        "autonomous_metrics_used_for_selection": False,
        "true_A_used_for_lambda_selection": False,
        "states_after_80_accessed": False,
    }
    write_json_record(output_json, record)
    lines = [
        "# Test 2A sparse-endpoint direct versus FIML",
        "",
        "All branches start from the same completed H1 model. Intermediate truth was excluded from optimization and used only for labeled post-hoc diagnostics.",
        "",
        "| network | sparse H2 | sparse H5 | dense H1 | dense H2 | dense H5 | autonomous final | autonomous max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        sparse = entry["sparse_endpoint"]["sparse_endpoint_objectives"]
        dense = entry["dense_and_fixed_support"]["horizon_objectives"]
        mixed = entry["autonomous_training_support"]["mixed_state_error"]
        lines.append(
            f"| {entry['label']} | {sparse['H2']:.12g} | {sparse['H5']:.12g} | "
            f"{dense['H1']:.12g} | {dense['H2']:.12g} | {dense['H5']:.12g} | "
            f"{mixed['final']:.12g} | {mixed['maximum']:.12g} |"
        )
    lines.extend(
        [
            "",
            "Raw Stage-1 endpoint fits and the Stage-2 NN endpoint fits are both retained so amortization/compression loss is explicit.",
        ]
    )
    Path(output_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    certify = commands.add_parser("certify")
    certify.add_argument("--configuration", required=True)
    certify.add_argument("--output", required=True)
    sweep = commands.add_parser("lambda-sweep")
    sweep.add_argument("--configuration", required=True)
    sweep.add_argument("--output-directory", required=True)
    benchmark = commands.add_parser("benchmark-smoke")
    benchmark.add_argument("--configuration", required=True)
    benchmark.add_argument("--output-directory", required=True)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--horizon", type=int)
    merge = commands.add_parser("merge-benchmarks")
    merge.add_argument("--h2", required=True)
    merge.add_argument("--h5", required=True)
    merge.add_argument("--output", required=True)
    stage2_benchmark = commands.add_parser("benchmark-stage2-full")
    stage2_benchmark.add_argument("--configuration", required=True)
    stage2_benchmark.add_argument("--output", required=True)
    stage2_benchmark.add_argument("--repeats", type=int, default=5)
    reselect = commands.add_parser("reselect-lambda")
    reselect.add_argument("--configuration", required=True)
    reselect.add_argument("--sweep-directory", required=True)
    plan = commands.add_parser("production-plan")
    plan.add_argument("--configuration", required=True)
    plan.add_argument("--benchmark", required=True)
    plan.add_argument("--stage2-benchmark", required=True)
    plan.add_argument("--output", required=True)
    direct = commands.add_parser("train-direct")
    direct.add_argument("--configuration", required=True)
    direct.add_argument("--horizon", type=int, required=True)
    direct.add_argument("--iterations", type=int, required=True)
    direct.add_argument("--output-directory", required=True)
    direct.add_argument("--resume", action="store_true")
    field = commands.add_parser("train-field-inversion")
    field.add_argument("--configuration", required=True)
    field.add_argument("--horizon", type=int, required=True)
    field.add_argument("--iterations", type=int, required=True)
    field.add_argument("--output-directory", required=True)
    pseudo = commands.add_parser("build-pseudo-labels")
    pseudo.add_argument("--configuration", required=True)
    pseudo.add_argument("--horizon", type=int, required=True)
    pseudo.add_argument("--controls-directory", required=True)
    pseudo.add_argument("--output", required=True)
    stage2 = commands.add_parser("train-stage2")
    stage2.add_argument("--configuration", required=True)
    stage2.add_argument("--horizon", type=int, required=True)
    stage2.add_argument("--dataset", required=True)
    stage2.add_argument("--iterations", type=int, required=True)
    stage2.add_argument("--output-directory", required=True)
    stage2.add_argument("--resume", action="store_true")
    cross = commands.add_parser("cross-evaluate")
    cross.add_argument("--configuration", required=True)
    cross.add_argument("--parameter-file", required=True)
    cross.add_argument("--expected-sha256", required=True)
    cross.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("--root", required=True)
    report.add_argument("--output-json", required=True)
    report.add_argument("--output-markdown", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "certify":
        certify_fiml(arguments.configuration, arguments.output)
    elif arguments.command == "lambda-sweep":
        run_regularization_sweep(arguments.configuration, arguments.output_directory)
    elif arguments.command == "benchmark-smoke":
        benchmark_and_smoke(
            arguments.configuration,
            arguments.output_directory,
            repeats=arguments.repeats,
            horizon=arguments.horizon,
        )
    elif arguments.command == "merge-benchmarks":
        merge_benchmark_records(arguments.h2, arguments.h5, arguments.output)
    elif arguments.command == "benchmark-stage2-full":
        benchmark_stage2_full_batch(
            arguments.configuration, arguments.output, repeats=arguments.repeats
        )
    elif arguments.command == "reselect-lambda":
        reselect_regularization(
            arguments.configuration, arguments.sweep_directory
        )
    elif arguments.command == "production-plan":
        write_production_plan(
            arguments.configuration,
            arguments.benchmark,
            arguments.stage2_benchmark,
            arguments.output,
        )
    elif arguments.command == "train-direct":
        train_direct_sparse(
            arguments.configuration,
            arguments.horizon,
            arguments.iterations,
            arguments.output_directory,
            resume=arguments.resume,
        )
    elif arguments.command == "train-field-inversion":
        train_field_inversion(
            arguments.configuration,
            arguments.horizon,
            arguments.iterations,
            arguments.output_directory,
        )
    elif arguments.command == "build-pseudo-labels":
        build_pseudo_label_dataset(
            arguments.configuration,
            arguments.horizon,
            arguments.controls_directory,
            arguments.output,
        )
    elif arguments.command == "train-stage2":
        train_stage2(
            arguments.configuration,
            arguments.horizon,
            arguments.dataset,
            arguments.iterations,
            arguments.output_directory,
            resume=arguments.resume,
        )
    elif arguments.command == "cross-evaluate":
        cross_evaluate_sparse_network(
            arguments.configuration,
            arguments.parameter_file,
            arguments.expected_sha256,
            arguments.output,
        )
    else:
        write_fiml_postprocess_report(
            arguments.root, arguments.output_json, arguments.output_markdown
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
