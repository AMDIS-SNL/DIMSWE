"""Prepare, certify, benchmark, and run the Test-2A H1/H2/H5 curriculum.

The production campaign is intentionally not launched by this module unless
the explicit ``train`` command is invoked by the user.  Cache preparation and
certification use truth states 0..80 only.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .learned_physics.parameters import tree_norm
from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_discrete_training import (
    CompactCheckpointObjective,
    FastFixedDiscreteObjective,
    FixedDiscreteCache,
    _file_sha256,
    load_fixed_cache,
    save_fixed_cache,
)
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import (
    DenseMLP,
    initialize_mlp,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters_atomic,
)
from .test2a_pyrol import build_test2a_lbfgs_parameters
from .test2a_trajectory import (
    GlobalMixedMassMetric,
    NeuralTrajectoryObjective,
    TrajectoryPyROLObjective,
    reset_windows,
)
from .test2a_trajectory_certification import _build_case


CANONICAL_SEED_SHA256 = (
    "6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975"
)
HORIZONS = (1, 2, 5)
TARGET_STEPS = tuple(range(1, 81))


def _canonical_json_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_stage_progress_record(
    record, configuration_sha256, horizon, h1_cache_npz_sha256
):
    if record.get("status") != "in_progress":
        raise ValueError("horizon progress is not resumable")
    if record.get("configuration_sha256") != configuration_sha256:
        raise ValueError("horizon resume configuration changed")
    if int(record.get("horizon", -1)) != int(horizon):
        raise ValueError("horizon resume stage changed")
    if record.get("h1_cache_npz_sha256") != h1_cache_npz_sha256:
        raise ValueError("horizon resume H1 cache changed")
    iteration = int(record.get("last_checkpoint_accepted_iteration", -1))
    if iteration < 0:
        raise ValueError("horizon resume lacks a parameter checkpoint")
    return iteration


def validate_complete_stage_result(
    record,
    configuration_sha256,
    horizon,
    h1_cache_npz_sha256,
    expected_source_parameter_sha256,
):
    """Reject incomplete or incompatible upstream curriculum artifacts."""

    if record.get("status") != "complete":
        raise ValueError("upstream horizon stage is incomplete")
    if record.get("configuration_sha256") != configuration_sha256:
        raise ValueError("upstream horizon configuration changed")
    if int(record.get("horizon", -1)) != int(horizon):
        raise ValueError("upstream horizon stage changed")
    if record.get("h1_cache_npz_sha256") != h1_cache_npz_sha256:
        raise ValueError("upstream horizon H1 cache changed")
    initialization = record.get("initialization", {})
    if (
        initialization.get("source_parameter_pytree_sha256")
        != expected_source_parameter_sha256
    ):
        raise ValueError("upstream horizon initialization changed")
    if (
        initialization.get("new_optimizer_process") is not True
        or initialization.get("source_optimizer_secant_history_reused") is not False
    ):
        raise ValueError("upstream horizon optimizer-history contract changed")
    final_sha = record.get("final_parameter_pytree_sha256")
    final_file = record.get("final_parameter_file")
    if not isinstance(final_sha, str) or len(final_sha) != 64 or not final_file:
        raise ValueError("upstream horizon final parameter artifact is incomplete")
    return final_file, final_sha


def load_curriculum_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("benchmark_stage") != "Test 2A H1-H2-H5 horizon curriculum":
        raise ValueError("not the selected Test-2A horizon curriculum")
    if record["truth"]["state_indices"] != [0, 80] or not record["truth"][
        "states_after_80_forbidden"
    ]:
        raise ValueError("horizon curriculum may use only truth states 0..80")
    if tuple(sorted(int(value) for value in record["stages"])) != HORIZONS:
        raise ValueError("selected curriculum horizons changed")
    optimizer = record["optimizer"]
    if (
        optimizer["library"] != "PyROL/ROL"
        or optimizer["method"] != "line-search L-BFGS"
        or int(optimizer["maximum_secant_storage"]) != 20
        or float(optimizer["gradient_tolerance"]) != 1.0e-8
        or float(optimizer["step_tolerance"]) != 1.0e-12
        or optimizer["production_HVP"] is not False
        or optimizer["new_process_and_empty_secant_history_each_stage"] is not True
    ):
        raise ValueError("curriculum optimizer contract changed")
    for horizon in HORIZONS:
        stage = record["stages"][str(horizon)]
        windows = production_windows(horizon)
        expected_starts = tuple(range(0, 80, horizon))
        encoded = tuple(
            range(
                int(stage["starts"][0]),
                int(stage["starts"][1]) + 1,
                int(stage["starts"][2]),
            )
        )
        if encoded != expected_starts or len(windows) != int(stage["window_count"]):
            raise ValueError(f"H={horizon} non-overlapping window schedule changed")
        targets = tuple(step for window in windows for step in window.target_steps)
        if targets != TARGET_STEPS or len(set(targets)) != 80:
            raise ValueError(f"H={horizon} target coverage changed")
        checkpoints = tuple(int(value) for value in stage["checkpoint_accepted_iterations"])
        if (
            checkpoints != tuple(sorted(set(checkpoints)))
            or checkpoints[0] != 0
            or checkpoints[-1]
            != int(stage["production_accepted_iteration_limit"])
        ):
            raise ValueError(f"H={horizon} checkpoint schedule changed")
    loss = record["loss"]
    if (
        loss["mode"] != "accumulated"
        or float(loss["target_weights"]) != 1.0
        or loss["target_normalization"] != "none"
        or loss["common_denominator_identical_across_horizons"] is not True
    ):
        raise ValueError("production horizon loss contract changed")
    return record


def production_windows(horizon):
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError("production curriculum horizon must be 1, 2, or 5")
    starts = tuple(range(0, 80, horizon))
    return reset_windows(starts, horizon, "accumulated", (1.0,) * horizon)


def _tree_relation(left, right):
    left_flat, _ = ravel_pytree(left)
    right_flat, _ = ravel_pytree(right)
    left_values = np.asarray(left_flat, dtype=np.float64)
    right_values = np.asarray(right_flat, dtype=np.float64)
    difference = left_values - right_values
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    denominator = left_norm * right_norm
    dot = float(np.dot(left_values, right_values))
    alpha = dot / max(float(np.dot(right_values, right_values)), np.finfo(float).tiny)
    nonproportional = left_values - alpha * right_values
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "dot": dot,
        "cosine": None if denominator == 0.0 else float(dot / denominator),
        "best_alpha_left_approximately_alpha_right": float(alpha),
        "relative_nonproportional_residual": float(
            np.linalg.norm(nonproportional)
            / max(left_norm, np.finfo(np.float64).tiny)
        ),
        "absolute_difference": float(np.linalg.norm(difference)),
        "relative_difference_to_right": float(
            np.linalg.norm(difference)
            / max(right_norm, np.finfo(np.float64).tiny)
        ),
        "maximum_absolute_component_difference": float(
            np.max(np.abs(difference))
        ),
    }


def _load_model_context(configuration):
    selected = load_selected_configuration(
        configuration["model"]["selected_operator_configuration"]
    )
    model_configuration = mlp_configuration_from_record(selected["model"])
    dataset, dataset_metadata = load_operator_dataset(
        configuration["model"]["operator_dataset"]
    )
    normalization = normalization_from_record(dataset_metadata["normalization"])
    return model_configuration, dataset, dataset_metadata, normalization


def _load_parameter(path, expected_sha256, model_configuration):
    parameters, configuration = load_mlp_parameters(path)
    if configuration != model_configuration:
        raise ValueError(f"parameter architecture changed for {path}")
    actual = parameter_pytree_sha256(parameters)
    if actual != expected_sha256:
        raise ValueError(f"parameter fingerprint changed for {path}")
    return parameters


def _parameter_probes(configuration, model_configuration):
    seed = initialize_mlp(model_configuration)
    if parameter_pytree_sha256(seed) != CANONICAL_SEED_SHA256:
        raise ValueError("canonical Test-2A seed changed")
    probes = {"canonical_seed0": seed}
    for name, record in configuration["certification"]["probes"].items():
        if record is None:
            continue
        probes[name] = _load_parameter(
            record["path"], record["sha256"], model_configuration
        )
    flat, unravel = ravel_pytree(seed)
    direction = np.linspace(-0.8, 0.6, int(flat.size), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    amplitude = 1.0e-3 * max(1.0, float(np.linalg.norm(np.asarray(flat))))
    probes["deterministic_seed_relative_1e-3_perturbation"] = unravel(
        flat + amplitude * jnp.asarray(direction, dtype=jnp.float64)
    )
    return probes


def _postprefix_arrays(configuration, normalization):
    from .jax_moist_hvp import JAXMoistEulerHVP

    _, case, truth, _, _ = _build_case(
        configuration["truth"]["trajectory_configuration"], maximum_truth_step=80
    )
    analytical = JAXMoistEulerHVP(
        case.helper.moist_child.ufl_oracle, use_jit=True, local_physics=None
    )
    if analytical.primal_helper.physics_mode != "analytical_A_original_R":
        raise RuntimeError("post-prefix target helper is not genuinely analytical")
    features = []
    targets = []
    depths = []
    maximum_r = 0.0
    with case.physical_c0(float(configuration["truth"]["c0"])):
        for step in range(80):
            prefix = case.helper.take_fixed_prefix_cached(
                truth[step], case.t0 + step * case.dt, case.dt
            )
            target = analytical.take_forward_step_cached(
                prefix.state_out, case.t0 + step * case.dt, case.dt
            )
            packed = target.packed_state
            local_features = np.stack(
                (
                    packed["h"],
                    packed["S"],
                    packed["Qv"],
                    packed["Qc"],
                    target.packed_fields["B"],
                ),
                axis=-1,
            )
            features.append(np.asarray(local_features, dtype=np.float64).reshape(-1, 5))
            targets.append(np.asarray(target.rates["A"], dtype=np.float64).reshape(-1))
            depths.append(np.asarray(packed["h"], dtype=np.float64).reshape(-1))
            maximum_r = max(
                maximum_r,
                float(np.max(np.abs(np.asarray(target.rates["R"], dtype=np.float64)))),
            )
    physical_features = np.concatenate(features)
    physical_targets = np.concatenate(targets)
    h = np.stack(depths)
    if physical_features.shape != (327_680, 5) or h.shape != (80, 4096):
        raise RuntimeError("post-prefix deployed sample accounting changed")
    return {
        "case": case,
        "truth": truth,
        "normalized_features": np.asarray(
            normalization.normalize_features(physical_features), dtype=np.float64
        ),
        "normalized_targets": np.asarray(
            normalization.normalize_a(physical_targets), dtype=np.float64
        ),
        "physical_features": physical_features,
        "physical_targets": physical_targets,
        "h": h,
        "maximum_original_R": maximum_r,
    }


def _fixed_a_tendency_energy(cache):
    from scipy.sparse import coo_matrix

    w_s = coo_matrix(
        (cache.w_s_data, (cache.w_s_indices[:, 0], cache.w_s_indices[:, 1])),
        shape=cache.w_s_shape,
    ).tocsr()
    w_q = coo_matrix(
        (cache.w_q_data, (cache.w_q_indices[:, 0], cache.w_q_indices[:, 1])),
        shape=cache.w_q_shape,
    ).tocsr()
    inverse_q = coo_matrix(
        (
            cache.mass_inverse_q_data,
            (
                cache.mass_inverse_q_indices[:, 0],
                cache.mass_inverse_q_indices[:, 1],
            ),
        ),
        shape=cache.mass_inverse_q_shape,
    ).tocsr()
    physical_a = float(cache.output_scale) * cache.normalized_targets
    source_q = cache.h * physical_a
    source_s = float(cache.beta2) * source_q
    weak_s = (w_s @ source_s.T).T
    weak_q = (w_q @ source_q.T).T
    weak_s_grid = weak_s[:, cache.mass_s_grid_order].reshape(
        (cache.h.shape[0], *cache.mass_s_grid_shape)
    )
    riesz_s = np.einsum(
        "ij,bjk,lk->bil",
        cache.mass_inverse_s_y,
        weak_s_grid,
        cache.mass_inverse_s_x,
    )
    riesz_q = (inverse_q @ weak_q.T).T
    return float(
        np.sum(weak_s_grid * riesz_s) + 2.0 * np.sum(weak_q * riesz_q)
    )


def _copy_fixed_operators(source, *, features, targets, h, normalizer, metadata):
    return FixedDiscreteCache(
        normalized_features=features,
        normalized_targets=targets,
        h=h,
        beta2=source.beta2,
        output_scale=source.output_scale,
        w_s_data=source.w_s_data,
        w_s_indices=source.w_s_indices,
        w_s_shape=source.w_s_shape,
        w_q_data=source.w_q_data,
        w_q_indices=source.w_q_indices,
        w_q_shape=source.w_q_shape,
        mass_inverse_s_x=source.mass_inverse_s_x,
        mass_inverse_s_y=source.mass_inverse_s_y,
        mass_s_grid_order=source.mass_s_grid_order,
        mass_s_grid_shape=source.mass_s_grid_shape,
        mass_inverse_q_data=source.mass_inverse_q_data,
        mass_inverse_q_indices=source.mass_inverse_q_indices,
        mass_inverse_q_shape=source.mass_inverse_q_shape,
        normalizer=normalizer,
        metadata=metadata,
    )


def _trajectory_objective(case, truth, horizon, denominator, denominator_sha256):
    metric = GlobalMixedMassMetric(
        case.helper, denominator, denominator_sha256=denominator_sha256
    )
    return NeuralTrajectoryObjective(
        case,
        truth,
        production_windows(horizon),
        metric=metric,
        c0=0.14,
        use_fixed_prefix=True,
    )


def _parity_record(fast_value, fast_gradient, literal_value, literal_gradient):
    relation = _tree_relation(fast_gradient, literal_gradient)
    return {
        "cached_value": float(fast_value),
        "literal_H1_value": float(literal_value),
        "absolute_value_difference": abs(float(fast_value) - float(literal_value)),
        "relative_value_difference": abs(float(fast_value) - float(literal_value))
        / max(abs(float(literal_value)), np.finfo(np.float64).tiny),
        "gradient": relation,
    }


def prepare_h1_cache(configuration_path, output_path):
    configuration = load_curriculum_configuration(configuration_path)
    destination = Path(output_path)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite H1 M2-Y cache")
    model_configuration, _, dataset_metadata, normalization = _load_model_context(
        configuration
    )
    source_path = configuration["fixed_cache"]["source_M2_X_cache"]
    source = load_fixed_cache(source_path)
    if source.metadata["cache_npz_sha256"] != configuration["fixed_cache"][
        "source_M2_X_cache_sha256"
    ]:
        raise ValueError("source M2-X cache fingerprint changed")
    started = perf_counter()
    data = _postprefix_arrays(configuration, normalization)
    preliminary = _copy_fixed_operators(
        source,
        features=data["normalized_features"],
        targets=data["normalized_targets"],
        h=data["h"],
        normalizer=1.0,
        metadata={},
    )
    tendency_denominator = _fixed_a_tendency_energy(preliminary)
    dt = float(data["case"].dt)
    state_denominator = dt * dt * tendency_denominator
    denominator_digest = sha256()
    denominator_digest.update(np.float64(state_denominator).tobytes())
    denominator_digest.update(
        np.ascontiguousarray(data["physical_targets"], dtype=np.float64).tobytes()
    )
    denominator_digest.update(
        np.ascontiguousarray(data["h"], dtype=np.float64).tobytes()
    )
    denominator_sha256 = denominator_digest.hexdigest()
    candidate = _copy_fixed_operators(
        source,
        features=data["normalized_features"],
        targets=data["normalized_targets"],
        h=data["h"],
        normalizer=tendency_denominator,
        metadata={},
    )
    fast = FastFixedDiscreteObjective(candidate, model_configuration, use_jit=True)
    literal = _trajectory_objective(
        data["case"], data["truth"], 1, state_denominator, denominator_sha256
    )
    probes = _parameter_probes(configuration, model_configuration)
    certifications = []
    value_tolerance = float(configuration["certification"]["value_relative_tolerance"])
    gradient_tolerance = float(
        configuration["certification"]["gradient_relative_tolerance"]
    )
    absolute_floor = float(configuration["certification"]["absolute_floor"])
    for name, parameters in probes.items():
        cached_value, cached_gradient = fast.value_and_gradient(parameters)
        literal.clear_parameter_tape()
        literal_value, literal_gradient = literal.value_and_gradient(parameters)
        record = _parity_record(
            cached_value, cached_gradient, literal_value, literal_gradient
        )
        value_passed = record["absolute_value_difference"] <= max(
            absolute_floor, value_tolerance * abs(literal_value)
        )
        gradient_passed = record["gradient"]["absolute_difference"] <= max(
            absolute_floor,
            gradient_tolerance * record["gradient"]["right_norm"],
        )
        record.update(
            {
                "name": name,
                "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
                "value_passed": bool(value_passed),
                "gradient_passed": bool(gradient_passed),
                "passed": bool(value_passed and gradient_passed),
            }
        )
        certifications.append(record)
    if not all(record["passed"] for record in certifications):
        raise RuntimeError("cached H1 failed literal complete-trajectory parity")
    metadata = {
        "benchmark_stage": "Test 2A fixed post-prefix M2-Y H1 cache",
        "configuration_sha256": _canonical_json_sha256(configuration),
        "source_M2_X_cache": str(Path(source_path).resolve()),
        "source_M2_X_cache_sha256": source.metadata["cache_npz_sha256"],
        "production_oracle_certified": True,
        "oracle": "literal complete H1 trajectory with stored UFL boundary targets",
        "analytical_target": "genuine analytical JAX A at Y_k; local_physics=None",
        "analytical_UFL_JAX_parity": "ordinary float64 operation-order accuracy",
        "sample_count": 327_680,
        "postprefix_state_indices": [0, 79],
        "target_state_indices": [1, 80],
        "maximum_original_R_on_postprefix_support": data["maximum_original_R"],
        "original_R_treatment": (
            "retained in deployment; cancels exactly between analytical and neural "
            "children at each common fixed Y_k"
        ),
        "normalization_refitted": False,
        "operator_dataset_content_sha256": dataset_metadata["sha256_float64_content"],
        "common_denominator_tendency_mass": tendency_denominator,
        "common_denominator_state_mass_D": state_denominator,
        "common_denominator_dt": dt,
        "common_denominator_sha256": denominator_sha256,
        "dt_squared_cancels_in_H1_ratio": True,
        "oracle_certifications": certifications,
        "hot_loop_firedrake_or_PETSc_actions": 0,
        "dense_G_or_K_formed": False,
        "cache_construction_and_certification_wall_seconds": float(
            perf_counter() - started
        ),
    }
    certified = _copy_fixed_operators(
        candidate,
        features=candidate.normalized_features,
        targets=candidate.normalized_targets,
        h=candidate.h,
        normalizer=candidate.normalizer,
        metadata=metadata,
    )
    save_fixed_cache(destination, certified)
    return load_h1_cache(destination, configuration)


def load_h1_cache(path, configuration=None):
    cache = load_fixed_cache(path, require_canonical=False)
    if cache.sample_count != 327_680 or cache.h.shape != (80, 4096):
        raise ValueError("H1 M2-Y cache sample accounting changed")
    if cache.metadata.get("benchmark_stage") != (
        "Test 2A fixed post-prefix M2-Y H1 cache"
    ):
        raise ValueError("not a certified H1 M2-Y cache")
    if configuration is not None and cache.metadata["configuration_sha256"] != (
        _canonical_json_sha256(configuration)
    ):
        raise ValueError("H1 cache and curriculum configuration differ")
    return cache


def certify_curriculum(configuration_path, h1_cache_path, output_path):
    configuration = load_curriculum_configuration(configuration_path)
    cache = load_h1_cache(h1_cache_path, configuration)
    model_configuration, _, _, _ = _load_model_context(configuration)
    parameters = _load_parameter(
        configuration["initialization"]["parameter_file"],
        configuration["initialization"]["parameter_pytree_sha256"],
        model_configuration,
    )
    _, case, truth, _, loaded_model = _build_case(
        configuration["truth"]["trajectory_configuration"], maximum_truth_step=80
    )
    if loaded_model != model_configuration:
        raise ValueError("trajectory architecture changed")
    denominator = float(cache.metadata["common_denominator_state_mass_D"])
    denominator_sha = cache.metadata["common_denominator_sha256"]
    objectives = {
        horizon: _trajectory_objective(
            case, truth, horizon, denominator, denominator_sha
        )
        for horizon in HORIZONS
    }
    records = {}
    h1_value, h1_gradient = objectives[1].value_and_gradient(parameters)
    for horizon in (2, 5):
        value, gradient = objectives[horizon].value_and_gradient(parameters)
        records[f"H{horizon}_recursive_vs_80_independent_H1"] = {
            "recursive_objective": value,
            "independent_H1_objective": h1_value,
            "objective_ratio_recursive_over_independent": value / h1_value,
            "gradient": _tree_relation(gradient, h1_gradient),
            "recursive_feedback_present": True,
        }
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A H1-H2-H5 recursion certification",
        "parameter_file": str(
            Path(configuration["initialization"]["parameter_file"]).resolve()
        ),
        "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
        "common_denominator_state_mass_D": denominator,
        "common_denominator_sha256": denominator_sha,
        "window_schedules": {
            f"H{horizon}": [window.to_record() for window in production_windows(horizon)]
            for horizon in HORIZONS
        },
        "target_multisets": {
            f"H{horizon}": [
                step
                for window in production_windows(horizon)
                for step in window.target_steps
            ]
            for horizon in HORIZONS
        },
        "recursion_certificates": records,
        "existing_derivative_certification": {
            "path": str(
                Path("external-results/test2a/m3-m4-prep/trajectory_certification.json").resolve()
            ),
            "retained": True,
            "certifies": [
                "H2 state tangent-adjoint duality",
                "H1/H2 all-1281-parameter gradients",
                "same-theta tape and changed-theta invalidation",
                "fixed-prefix primal/gradient parity",
            ],
        },
        "truth_state_access": {
            "minimum": 0,
            "maximum": 80,
            "states_after_80_accessed": False,
        },
        "optimization_performed": False,
    }
    write_json_record(output_path, result)
    return result


def benchmark_curriculum(configuration_path, h1_cache_path, output_path):
    configuration = load_curriculum_configuration(configuration_path)
    cache = load_h1_cache(h1_cache_path, configuration)
    model_configuration, _, _, _ = _load_model_context(configuration)
    parameters = _load_parameter(
        configuration["initialization"]["parameter_file"],
        configuration["initialization"]["parameter_pytree_sha256"],
        model_configuration,
    )
    started = perf_counter()
    fast = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    first_started = perf_counter()
    fast.value_and_gradient(parameters)
    first_h1 = perf_counter() - first_started
    h1_values = []
    h1_gradients = []
    for _ in range(int(configuration["benchmark"]["steady_H1_repeats"])):
        call = perf_counter()
        fast.value(parameters)
        h1_values.append(perf_counter() - call)
        call = perf_counter()
        fast.value_and_gradient(parameters)
        h1_gradients.append(perf_counter() - call)
    _, case, truth, _, _ = _build_case(
        configuration["truth"]["trajectory_configuration"], maximum_truth_step=80
    )
    denominator = cache.metadata["common_denominator_state_mass_D"]
    denominator_sha = cache.metadata["common_denominator_sha256"]
    timings = {
        "H1": {
            "implementation": "exact fixed post-prefix M2-Y JAX cache",
            "window_count": 80,
            "target_count": 80,
            "first_value_and_gradient_seconds": first_h1,
            "steady_value_seconds": float(np.median(h1_values)),
            "steady_value_and_gradient_seconds": float(np.median(h1_gradients)),
            "hot_loop_firedrake_or_PETSc_solves": 0,
        }
    }
    for horizon in (2, 5):
        setup = perf_counter()
        objective = _trajectory_objective(
            case, truth, horizon, denominator, denominator_sha
        )
        setup_wall = perf_counter() - setup
        objective.clear_parameter_tape()
        call = perf_counter()
        value = objective.value(parameters)
        value_wall = perf_counter() - call
        tape_bytes = objective._last_tape.estimated_owned_bytes
        call = perf_counter()
        same_tape_value, gradient = objective.value_and_gradient(parameters)
        cached_gradient_wall = perf_counter() - call
        objective.clear_parameter_tape()
        call = perf_counter()
        fresh_value, fresh_gradient = objective.value_and_gradient(parameters)
        fresh_value_gradient_wall = perf_counter() - call
        relation = _tree_relation(gradient, fresh_gradient)
        if value != same_tape_value or abs(value - fresh_value) > 0.0:
            raise RuntimeError(f"H={horizon} benchmark objective changed across calls")
        timings[f"H{horizon}"] = {
            "implementation": "exact serial recursive complete-split trajectory",
            "window_count": 80 // horizon,
            "target_count": 80,
            "complete_steps_per_value": 80,
            "complete_reverse_steps_per_gradient": 80,
            "fixed_prefix_setup_seconds": setup_wall,
            "full_objective_value": value,
            "fresh_value_seconds": value_wall,
            "gradient_after_same_theta_value_seconds": cached_gradient_wall,
            "same_theta_value_plus_gradient_seconds": value_wall
            + cached_gradient_wall,
            "fresh_value_and_gradient_seconds": fresh_value_gradient_wall,
            "fresh_vs_same_tape_gradient": relation,
            "estimated_owned_tape_bytes": int(tape_bytes),
        }
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A production-window horizon timing",
        "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
        "common_denominator_sha256": denominator_sha,
        "timings": timings,
        "total_benchmark_wall_seconds": float(perf_counter() - started),
        "interpretation": "engineering timing only; no optimization",
        "truth_state_access": [0, 80],
        "states_after_80_accessed": False,
    }
    write_json_record(output_path, result)
    return result


def _build_active_objective(configuration, cache, model_configuration, horizon):
    if int(horizon) == 1:
        return FastFixedDiscreteObjective(cache, model_configuration, use_jit=True), None
    _, case, truth, _, _ = _build_case(
        configuration["truth"]["trajectory_configuration"], maximum_truth_step=80
    )
    trajectory = _trajectory_objective(
        case,
        truth,
        horizon,
        cache.metadata["common_denominator_state_mass_D"],
        cache.metadata["common_denominator_sha256"],
    )
    return trajectory, case


def _active_value_gradient(active, parameters):
    return active.value_and_gradient(parameters)


def run_nonscientific_smokes(
    configuration_path, h1_cache_path, output_directory, *, iterations=None
):
    from pyrol import Problem, Solver

    configuration = load_curriculum_configuration(configuration_path)
    cache = load_h1_cache(h1_cache_path, configuration)
    model_configuration, _, _, _ = _load_model_context(configuration)
    initial = _load_parameter(
        configuration["initialization"]["parameter_file"],
        configuration["initialization"]["parameter_pytree_sha256"],
        model_configuration,
    )
    count = int(
        iterations
        if iterations is not None
        else configuration["benchmark"]["nonscientific_smoke_accepted_iterations"]
    )
    if count < 1 or count > 20:
        raise ValueError("nonscientific smoke requires 1..20 accepted iterations")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("refusing to overwrite horizon smoke output")
    output.mkdir(parents=True)
    results = {}
    for horizon in HORIZONS:
        active, _ = _build_active_objective(
            configuration, cache, model_configuration, horizon
        )
        if horizon == 1:
            adapter = CompactCheckpointObjective(
                active.jax_value, initial, use_jit=True
            )
        else:
            adapter = TrajectoryPyROLObjective(active, initial)
        control = adapter.vector_from_pytree(initial)
        initial_value = float(active.value(initial))
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
        final_parameters = adapter.pytree_from_vector(control)
        final_value = float(active.value(final_parameters))
        state = solver.getAlgorithmState()
        artifact = output / f"H{horizon}_final_parameters.npz"
        save_mlp_parameters_atomic(artifact, final_parameters, model_configuration)
        loaded, _ = load_mlp_parameters(artifact)
        if parameter_pytree_sha256(loaded) != parameter_pytree_sha256(final_parameters):
            raise RuntimeError("smoke parameter checkpoint did not round-trip")
        results[f"H{horizon}"] = {
            "interpretation": "NONSCIENTIFIC HORIZON-CURRICULUM IMPLEMENTATION SMOKE",
            "initial_parameter_pytree_sha256": parameter_pytree_sha256(initial),
            "new_optimizer_process": True,
            "source_optimizer_secant_history_reused": False,
            "initial_objective": initial_value,
            "final_objective": final_value,
            "objective_decreased": final_value < initial_value,
            "accepted_iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": int(adapter.value_evaluations),
            "gradient_evaluations": int(adapter.gradient_evaluations),
            "HVP_evaluations": int(adapter.hvp_evaluations),
            "wall_time_seconds": wall,
            "final_parameter_file": str(artifact.resolve()),
            "final_parameter_pytree_sha256": parameter_pytree_sha256(
                final_parameters
            ),
            "checkpoint_roundtrip_verified": True,
        }
        if adapter.hvp_evaluations != 0 or not results[f"H{horizon}"][
            "objective_decreased"
        ]:
            raise RuntimeError(f"H={horizon} nonscientific smoke failed")
    record = {
        "status": "complete",
        "interpretation": "NONSCIENTIFIC HORIZON-CURRICULUM IMPLEMENTATION SMOKES",
        "methods": results,
        "states_after_80_accessed": False,
    }
    write_json_record(output / "smoke_results.json", record)
    return record


def _monitoring_record(
    parameters, active, horizon, h1_fast, m2x_fast, model, normalization, dataset
):
    active_value, active_gradient = _active_value_gradient(active, parameters)
    predictions = physical_predictions(
        parameters, model, normalization, dataset.features
    )
    m2x, jop = m2x_fast.objectives(parameters)
    return {
        "active_horizon": int(horizon),
        "J_active": float(active_value),
        "gradient_norm_active": float(tree_norm(active_gradient)),
        "J_H1_M2_Y": float(h1_fast.value(parameters)),
        "J_M2_X": float(m2x),
        "J_op": float(jop),
        "direct_A_metrics": operator_metrics(predictions, dataset.targets),
        "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
    }


def train_stage(
    configuration_path,
    h1_cache_path,
    horizon,
    initial_parameter_file,
    expected_initial_sha256,
    source_stage,
    output_directory,
    *,
    resume=False,
):
    from pyrol import Problem, Solver

    configuration = load_curriculum_configuration(configuration_path)
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError("training horizon must be 1, 2, or 5")
    cache = load_h1_cache(h1_cache_path, configuration)
    model_configuration, dataset, _, normalization = _load_model_context(configuration)
    configured_initial = _load_parameter(
        initial_parameter_file, expected_initial_sha256, model_configuration
    )
    if horizon == 1 and expected_initial_sha256 != configuration[
        "initialization"
    ]["parameter_pytree_sha256"]:
        raise ValueError("H1 must start from matched M1 200k")
    source_cache = load_fixed_cache(configuration["fixed_cache"]["source_M2_X_cache"])
    m2x_fast = FastFixedDiscreteObjective(source_cache, model_configuration, use_jit=True)
    h1_fast = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    active, _ = _build_active_objective(
        configuration, cache, model_configuration, horizon
    )
    stage = configuration["stages"][str(horizon)]
    limit = int(stage["production_accepted_iteration_limit"])
    checkpoint_set = set(int(value) for value in stage["checkpoint_accepted_iterations"])
    output = Path(output_directory)
    result_path = output / "fit_result.json"
    progress_path = output / "fit_progress.json"
    if result_path.exists():
        raise FileExistsError("refusing to overwrite completed horizon stage")
    output.mkdir(parents=True, exist_ok=True)
    offset = 0
    cumulative = {"objective": 0, "gradient": 0, "HVP": 0, "wall_seconds": 0.0}
    checkpoints = {}
    start_parameters = configured_initial
    configuration_sha = _canonical_json_sha256(configuration)
    if resume:
        if not progress_path.exists():
            raise FileNotFoundError("no horizon-stage checkpoint to resume")
        progress = read_json_record(progress_path)
        offset = validate_stage_progress_record(
            progress,
            configuration_sha,
            horizon,
            cache.metadata["cache_npz_sha256"],
        )
        start_parameters = _load_parameter(
            progress["last_checkpoint_parameter_file"],
            progress["last_checkpoint_parameter_pytree_sha256"],
            model_configuration,
        )
        if _file_sha256(progress["last_checkpoint_parameter_file"]) != progress[
            "last_checkpoint_parameter_npz_sha256"
        ]:
            raise ValueError("resume parameter file fingerprint changed")
        cumulative = dict(progress.get("cumulative_accounting", cumulative))
        checkpoints = {
            int(value["accepted_iteration"]): value
            for value in progress.get("checkpoint_diagnostics", [])
        }
    elif progress_path.exists():
        raise FileExistsError("incomplete stage exists; use explicit --resume")
    remaining = limit - offset
    if remaining <= 0:
        raise ValueError("configured stage budget already exhausted")
    model = DenseMLP(model_configuration)
    run_started = None
    progress_stride = max(1, limit // 100)

    def checkpoint(parameters, global_iteration, adapter, *, force=False):
        is_checkpoint = global_iteration in checkpoint_set or force
        is_progress = global_iteration % progress_stride == 0
        if not is_checkpoint and not is_progress:
            return
        elapsed = 0.0 if run_started is None else perf_counter() - run_started
        if is_checkpoint:
            record = _monitoring_record(
                parameters,
                active,
                horizon,
                h1_fast,
                m2x_fast,
                model,
                normalization,
                dataset,
            )
        else:
            record = {
                "J_active": float(active.value(parameters)),
                "gradient_norm_active": None,
                "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
            }
        record.update(
            {
                "accepted_iteration": int(global_iteration),
                "elapsed_wall_seconds_this_process": elapsed,
                "objective_evaluations_this_process": int(adapter.value_evaluations),
                "gradient_evaluations_this_process": int(adapter.gradient_evaluations),
            }
        )
        print(json.dumps({"event": "progress", **record}, sort_keys=True), flush=True)
        if not is_checkpoint:
            return
        parameter_path = output / f"parameters_iter_{global_iteration:06d}.npz"
        if not parameter_path.exists():
            save_mlp_parameters_atomic(
                parameter_path, parameters, model_configuration
            )
        record["parameter_file"] = str(parameter_path.resolve())
        record["parameter_npz_sha256"] = _file_sha256(parameter_path)
        checkpoints[int(global_iteration)] = record
        progress = {
            "status": "in_progress",
            "configuration_sha256": configuration_sha,
            "horizon": horizon,
            "h1_cache_npz_sha256": cache.metadata["cache_npz_sha256"],
            "last_checkpoint_accepted_iteration": int(global_iteration),
            "last_checkpoint_parameter_file": str(parameter_path.resolve()),
            "last_checkpoint_parameter_npz_sha256": record["parameter_npz_sha256"],
            "last_checkpoint_parameter_pytree_sha256": record[
                "parameter_pytree_sha256"
            ],
            "checkpoint_diagnostics": [
                checkpoints[key] for key in sorted(checkpoints)
            ],
            "cumulative_accounting": {
                "objective": int(cumulative["objective"])
                + int(adapter.value_evaluations),
                "gradient": int(cumulative["gradient"])
                + int(adapter.gradient_evaluations),
                "HVP": int(cumulative["HVP"]) + int(adapter.hvp_evaluations),
                "wall_seconds": float(cumulative["wall_seconds"]) + elapsed,
            },
            "resume_contract": configuration["resume_contract"],
        }
        write_json_record(progress_path, progress)
        write_json_record(
            output / f"checkpoint_iter_{global_iteration:06d}.json", record
        )

    def accepted_callback(control, local_index, adapter):
        if local_index == 0:
            return
        parameters = adapter.pytree_from_vector(control)
        checkpoint(parameters, offset + local_index, adapter)

    if horizon == 1:
        adapter = CompactCheckpointObjective(
            active.jax_value,
            start_parameters,
            use_jit=True,
            accepted_callback=accepted_callback,
        )
    else:
        adapter = TrajectoryPyROLObjective(
            active, start_parameters, accepted_callback=accepted_callback
        )
    control = adapter.vector_from_pytree(start_parameters)
    if not resume:
        checkpoint(configured_initial, 0, adapter, force=True)
    rol = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": 1.0e-8,
            "step_tolerance": 1.0e-12,
            "iteration_limit": remaining,
            "maximum_secant_storage": 20,
        }
    )
    print(
        json.dumps(
            {
                "event": "horizon_stage_start",
                "horizon": horizon,
                "accepted_iteration_offset": offset,
                "accepted_iteration_limit": limit,
                "initial_parameter_pytree_sha256": parameter_pytree_sha256(
                    start_parameters
                ),
                "source_stage": source_stage,
                "new_optimizer_process": True,
                "source_optimizer_secant_history_reused": False,
                "resume_secant_history_restored": False if resume else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run_started = perf_counter()
    solver = Solver(Problem(adapter, control), rol)
    solver.solve()
    wall = perf_counter() - run_started
    algorithm_state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    final_iteration = offset + int(algorithm_state.iter)
    final_path = output / "final_parameters.npz"
    if final_path.exists() or final_path.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite final horizon parameters")
    save_mlp_parameters_atomic(final_path, final_parameters, model_configuration)
    final_record = _monitoring_record(
        final_parameters,
        active,
        horizon,
        h1_fast,
        m2x_fast,
        model,
        normalization,
        dataset,
    )
    final_record["accepted_iteration"] = final_iteration
    final_record["parameter_file"] = str(final_path.resolve())
    final_record["parameter_npz_sha256"] = _file_sha256(final_path)
    checkpoints[final_iteration] = final_record
    write_json_record(
        output / f"checkpoint_iter_{final_iteration:06d}.json", final_record
    )
    counts = {
        "objective": int(cumulative["objective"]) + int(adapter.value_evaluations),
        "gradient": int(cumulative["gradient"]) + int(adapter.gradient_evaluations),
        "HVP": int(cumulative["HVP"]) + int(adapter.hvp_evaluations),
    }
    if counts["HVP"] != 0:
        raise RuntimeError("curriculum L-BFGS unexpectedly requested an HVP")
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A horizon curriculum production stage",
        "configuration_sha256": configuration_sha,
        "horizon": horizon,
        "scientific_name": stage["scientific_name"],
        "h1_cache_npz_sha256": cache.metadata["cache_npz_sha256"],
        "initialization": {
            "source_stage": source_stage,
            "source_parameter_file": str(Path(initial_parameter_file).resolve()),
            "source_parameter_pytree_sha256": expected_initial_sha256,
            "new_optimizer_process": True,
            "source_optimizer_secant_history_reused": False,
        },
        "optimizer": {
            **configuration["optimizer"],
            "accepted_iterations": final_iteration,
            "actual_ROL_termination_reason": str(algorithm_state.statusFlag),
            "objective_evaluations": counts["objective"],
            "gradient_evaluations": counts["gradient"],
            "HVP_evaluations": counts["HVP"],
            "wall_time_seconds": float(cumulative["wall_seconds"]) + wall,
            "secant_history_restored_on_parameter_resume": False,
        },
        "final_diagnostics": final_record,
        "checkpoint_diagnostics": [checkpoints[key] for key in sorted(checkpoints)],
        "final_parameter_file": str(final_path.resolve()),
        "final_parameter_npz_sha256": _file_sha256(final_path),
        "final_parameter_pytree_sha256": parameter_pytree_sha256(final_parameters),
        "common_denominator_state_mass_D": cache.metadata[
            "common_denominator_state_mass_D"
        ],
        "common_denominator_sha256": cache.metadata["common_denominator_sha256"],
        "truth_state_access": [0, 80],
        "states_after_80_accessed": False,
    }
    write_json_record(result_path, result)
    write_json_record(
        progress_path,
        {
            "status": "complete",
            "configuration_sha256": configuration_sha,
            "horizon": horizon,
            "h1_cache_npz_sha256": cache.metadata["cache_npz_sha256"],
            "last_checkpoint_accepted_iteration": final_iteration,
            "last_checkpoint_parameter_file": str(final_path.resolve()),
            "last_checkpoint_parameter_npz_sha256": _file_sha256(final_path),
            "last_checkpoint_parameter_pytree_sha256": result[
                "final_parameter_pytree_sha256"
            ],
            "cumulative_accounting": {
                **counts,
                "wall_seconds": result["optimizer"]["wall_time_seconds"],
            },
            "resume_contract": configuration["resume_contract"],
        },
    )
    return result


def cross_evaluate_artifact(
    configuration_path, h1_cache_path, parameter_file, expected_sha256, output_path
):
    configuration = load_curriculum_configuration(configuration_path)
    cache = load_h1_cache(h1_cache_path, configuration)
    model_configuration, dataset, _, normalization = _load_model_context(configuration)
    parameters = _load_parameter(parameter_file, expected_sha256, model_configuration)
    h1 = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    m2x_cache = load_fixed_cache(configuration["fixed_cache"]["source_M2_X_cache"])
    m2x = FastFixedDiscreteObjective(m2x_cache, model_configuration, use_jit=True)
    _, case, truth, _, _ = _build_case(
        configuration["truth"]["trajectory_configuration"], maximum_truth_step=80
    )
    values = {"H1": h1.value(parameters)}
    for horizon in (2, 5):
        values[f"H{horizon}"] = _trajectory_objective(
            case,
            truth,
            horizon,
            cache.metadata["common_denominator_state_mass_D"],
            cache.metadata["common_denominator_sha256"],
        ).value(parameters)
    j_m2x, j_op = m2x.objectives(parameters)
    predictions = physical_predictions(
        parameters, DenseMLP(model_configuration), normalization, dataset.features
    )
    result = {
        "status": "complete",
        "parameter_file": str(Path(parameter_file).resolve()),
        "parameter_pytree_sha256": expected_sha256,
        "horizon_objectives": values,
        "J_M2_X": j_m2x,
        "J_op": j_op,
        "direct_A_metrics": operator_metrics(predictions, dataset.targets),
        "autonomous_metrics_used_for_selection": False,
        "states_after_80_accessed": False,
    }
    write_json_record(output_path, result)
    return result


def project_curriculum_runtime(
    configuration_path, benchmark_path, smoke_path, output_path
):
    """Create transparent, smoke-based engineering runtime projections."""

    configuration = load_curriculum_configuration(configuration_path)
    benchmark = read_json_record(benchmark_path)
    smoke = read_json_record(smoke_path)
    projections = {}
    for horizon in HORIZONS:
        item = smoke["methods"][f"H{horizon}"]
        accepted = int(item["accepted_iterations"])
        if accepted <= 0:
            raise ValueError("smoke did not accept an optimizer iteration")
        seconds_per_accepted = float(item["wall_time_seconds"]) / accepted
        projections[str(horizon)] = {
            "smoke_seconds_per_accepted_iteration": seconds_per_accepted,
            "smoke_objective_evaluations_per_accepted_iteration": (
                float(item["objective_evaluations"]) / accepted
            ),
            "smoke_gradient_evaluations_per_accepted_iteration": (
                float(item["gradient_evaluations"]) / accepted
            ),
            "projected_wall_seconds": {
                str(count): seconds_per_accepted * count
                for count in (100, 500, 1000)
            },
            "recommended_production_cap": int(
                configuration["stages"][str(horizon)][
                    "production_accepted_iteration_limit"
                ]
            ),
            "recommended_cap_projected_wall_seconds": seconds_per_accepted
            * int(
                configuration["stages"][str(horizon)][
                    "production_accepted_iteration_limit"
                ]
            ),
        }
    result = {
        "status": "complete",
        "diagnostic": "NONSCIENTIFIC_SMOKE_LINEAR_RUNTIME_PROJECTION",
        "projection_is_authoritative": False,
        "caveat": (
            "Two-iteration line-search behavior is extrapolated linearly; "
            "later line searches and convergence can change the cost."
        ),
        "full_objective_benchmark": benchmark,
        "projections": projections,
        "recommended_campaign_projected_wall_seconds": sum(
            item["recommended_cap_projected_wall_seconds"]
            for item in projections.values()
        ),
        "states_after_80_accessed": False,
    }
    write_json_record(output_path, result)
    return result


def _compact_rollout(record):
    return {
        "mixed_state_error": record["mixed_state_error"],
        "fieldwise_errors": record["fieldwise_errors"],
        "off_manifold_A": record["aggregate_off_manifold_A_diagnostic"],
        "kinetic_energy": record["kinetic_energy"],
        "projected_enstrophy": record["projected_enstrophy"],
        "rain_activity": record["rain_activity_summary"],
        "source_structural_invariants": record["source_structural_invariants"],
        "all_states_finite": record["all_states_finite"],
    }


def write_curriculum_postprocess_report(manifest_path, output_json, output_markdown):
    """Combine stage-boundary cross-objectives and post-hoc rollouts."""

    manifest = read_json_record(manifest_path)
    entries = []
    for entry in manifest["entries"]:
        cross = read_json_record(entry["cross_objective_file"])
        rollout = read_json_record(entry["rollout_summary_file"])
        if cross.get("status") != "complete" or rollout.get("status") != "complete":
            raise ValueError(f"postprocessing for {entry['label']} is incomplete")
        if cross.get("states_after_80_accessed", True):
            raise ValueError("cross-objective postprocessing accessed held-out truth")
        contract = rollout["deployment_contract"]
        if (
            contract.get("states_after_80_accessed", True)
            or contract.get("truth_states_accessed") != [0, 80]
            or int(contract.get("truth_resets_after_initialization", -1)) != 0
        ):
            raise ValueError("autonomous postprocessing contract changed")
        if cross["parameter_pytree_sha256"] != entry["parameter_pytree_sha256"]:
            raise ValueError("cross-objective parameter fingerprint changed")
        if (
            rollout["parameter_provenance"]["parameter_pytree_sha256"]
            != entry["parameter_pytree_sha256"]
        ):
            raise ValueError("autonomous parameter fingerprint changed")
        entries.append(
            {
                "label": entry["label"],
                "parameter_file": entry["parameter_file"],
                "parameter_pytree_sha256": entry["parameter_pytree_sha256"],
                "offline_diagnostics": cross,
                "autonomous_training_support": _compact_rollout(rollout),
            }
        )
    result = {
        "status": "complete",
        "diagnostic": "H1_H2_H5_HORIZON_CURRICULUM_STAGE_BOUNDARIES",
        "entries": entries,
        "selection_contract": {
            "autonomous_metrics_used_for_optimizer_stopping": False,
            "autonomous_metrics_used_for_horizon_progression": False,
            "truth_states_accessed": [0, 80],
            "states_after_80_accessed": False,
        },
    }
    write_json_record(output_json, result)
    lines = [
        "# Test 2A H1-H2-H5 horizon-curriculum stage boundaries",
        "",
        "Autonomous metrics are post-hoc training-support diagnostics and did not select parameters or stop training.",
        "",
        "| artifact | J_H1 | J_H2 | J_H5 | J_M2-X | J_op | mixed final | mixed max | mixed accumulated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        offline = entry["offline_diagnostics"]
        mixed = entry["autonomous_training_support"]["mixed_state_error"]
        horizons = offline["horizon_objectives"]
        lines.append(
            f"| {entry['label']} | {horizons['H1']:.12g} | "
            f"{horizons['H2']:.12g} | {horizons['H5']:.12g} | "
            f"{offline['J_M2_X']:.12g} | {offline['J_op']:.12g} | "
            f"{mixed['final']:.12g} | {mixed['maximum']:.12g} | "
            f"{mixed['accumulated']:.12g} |"
        )
    lines.extend(
        [
            "",
            "All runs use truth only through state 80. Each horizon used a new optimizer process with empty L-BFGS history.",
        ]
    )
    Path(output_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-h1-cache")
    prepare.add_argument("--configuration", required=True)
    prepare.add_argument("--output", required=True)
    certify = commands.add_parser("certify")
    certify.add_argument("--configuration", required=True)
    certify.add_argument("--h1-cache", required=True)
    certify.add_argument("--output", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--configuration", required=True)
    benchmark.add_argument("--h1-cache", required=True)
    benchmark.add_argument("--output", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--configuration", required=True)
    smoke.add_argument("--h1-cache", required=True)
    smoke.add_argument("--output-directory", required=True)
    smoke.add_argument("--iterations", type=int)
    train = commands.add_parser("train")
    train.add_argument("--configuration", required=True)
    train.add_argument("--h1-cache", required=True)
    train.add_argument("--horizon", type=int, required=True)
    train.add_argument("--initial-parameter-file", required=True)
    train.add_argument("--expected-initial-sha256", required=True)
    train.add_argument("--source-stage", required=True)
    train.add_argument("--output-directory", required=True)
    train.add_argument("--resume", action="store_true")
    cross = commands.add_parser("cross-evaluate")
    cross.add_argument("--configuration", required=True)
    cross.add_argument("--h1-cache", required=True)
    cross.add_argument("--parameter-file", required=True)
    cross.add_argument("--expected-sha256", required=True)
    cross.add_argument("--output", required=True)
    runtime = commands.add_parser("project-runtime")
    runtime.add_argument("--configuration", required=True)
    runtime.add_argument("--benchmark", required=True)
    runtime.add_argument("--smoke", required=True)
    runtime.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("--manifest", required=True)
    report.add_argument("--output-json", required=True)
    report.add_argument("--output-markdown", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare-h1-cache":
        prepare_h1_cache(arguments.configuration, arguments.output)
    elif arguments.command == "certify":
        certify_curriculum(arguments.configuration, arguments.h1_cache, arguments.output)
    elif arguments.command == "benchmark":
        benchmark_curriculum(arguments.configuration, arguments.h1_cache, arguments.output)
    elif arguments.command == "smoke":
        run_nonscientific_smokes(
            arguments.configuration,
            arguments.h1_cache,
            arguments.output_directory,
            iterations=arguments.iterations,
        )
    elif arguments.command == "train":
        train_stage(
            arguments.configuration,
            arguments.h1_cache,
            arguments.horizon,
            arguments.initial_parameter_file,
            arguments.expected_initial_sha256,
            arguments.source_stage,
            arguments.output_directory,
            resume=arguments.resume,
        )
    elif arguments.command == "cross-evaluate":
        cross_evaluate_artifact(
            arguments.configuration,
            arguments.h1_cache,
            arguments.parameter_file,
            arguments.expected_sha256,
            arguments.output,
        )
    elif arguments.command == "project-runtime":
        project_curriculum_runtime(
            arguments.configuration,
            arguments.benchmark,
            arguments.smoke,
            arguments.output,
        )
    else:
        write_curriculum_postprocess_report(
            arguments.manifest, arguments.output_json, arguments.output_markdown
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "HORIZONS",
    "TARGET_STEPS",
    "benchmark_curriculum",
    "certify_curriculum",
    "cross_evaluate_artifact",
    "load_curriculum_configuration",
    "load_h1_cache",
    "prepare_h1_cache",
    "production_windows",
    "project_curriculum_runtime",
    "run_nonscientific_smokes",
    "train_stage",
    "validate_complete_stage_result",
    "validate_stage_progress_record",
    "write_curriculum_postprocess_report",
)
