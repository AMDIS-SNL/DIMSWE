"""Test-2A audit of Method 2 versus the H=1 truth-reset special case.

This module is diagnostic only.  It composes the certified fixed-state
Method-2 objective and shared Method-3/4 trajectory machinery without
changing either implementation and without performing optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .hidden_c0 import _copy_function
from .learned_physics.parameters import (
    tree_axpy,
    tree_dot,
    tree_norm,
    tree_zeros,
    validate_float64_tree,
)
from .resolved_hidden_c0 import write_json_record
from .test2a_discrete_offline import (
    DeployedDiscreteOfflineObjective,
    DiscreteOfflineObservation,
    ProductionDiscreteOfflineOperations,
    ProductionObservationPayload,
    _a_sensitive_source,
)
from .test2a_discrete_training import (
    FastFixedDiscreteObjective,
    load_fixed_cache,
)
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import (
    initialize_mlp,
    load_mlp_parameters,
    load_selected_configuration,
    mlp_configuration_from_record,
)
from .test2a_trajectory import (
    NeuralTrajectoryObjective,
    continuous_rollout,
    reset_windows,
)
from .test2a_trajectory_certification import _build_case


CANONICAL_SEED_SHA256 = (
    "6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975"
)
PROBE_ARTIFACTS = {
    "matched_m1_200k": {
        "path": (
            "external-results/test2a/fair-longfit/"
            "operator-seed0-m20-200k/final_parameters.npz"
        ),
        "sha256": (
            "f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8"
        ),
    },
    "matched_m2_200k": {
        "path": (
            "external-results/test2a/fair-longfit/"
            "discrete-seed0-m20-200k/final_parameters.npz"
        ),
        "sha256": (
            "94bb112961bc2f2e05cbca459bc50d64513110a077e2b15cded39fe8427de6f8"
        ),
    },
    "m1_to_m2_finetuned_50k": {
        "path": (
            "external-results/test2a/m1-to-m2-finetune/"
            "operator-200k-to-discrete-m20-50k/final_parameters.npz"
        ),
        "sha256": (
            "e68110b18ea29748830b70683da321bb8e670aa69ddc94598692d72a6f278fc3"
        ),
    },
}


def h1_structural_source_error(h, delta_a, beta2):
    """Return H(Y) delta-A in mixed-field order (v,h,S,Qv,Qc,Qr)."""
    h = np.asarray(h, dtype=np.float64)
    delta_a = np.asarray(delta_a, dtype=np.float64)
    if h.shape != delta_a.shape:
        raise ValueError("h and delta_a must use the same deployed GLL shape")
    water = h * delta_a
    zeros = np.zeros_like(water)
    return (zeros, zeros, float(beta2) * water, water, -water, zeros)


def h1_tendency_loss_coefficient(dt, target_squared_mass_norm, weight=1.0):
    """Coefficient multiplying ||G(Y) delta-A||_M^2 in the H=1 loss."""
    dt = float(dt)
    normalizer = float(target_squared_mass_norm)
    weight = float(weight)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("target squared mass norm must be positive and finite")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be nonnegative and finite")
    return 0.5 * weight * dt * dt / normalizer


def parameter_gradient_relation(left, right):
    """Describe the best relation left approximately alpha * right."""
    left = validate_float64_tree(left, name="left gradient")
    right = validate_float64_tree(right, name="right gradient")
    left_norm = float(tree_norm(left))
    right_norm = float(tree_norm(right))
    dot = float(tree_dot(left, right))
    denominator = float(tree_dot(right, right))
    alpha = None if denominator == 0.0 else dot / denominator
    if left_norm == 0.0 or right_norm == 0.0:
        cosine = None
    else:
        cosine = dot / (left_norm * right_norm)
    residual = None
    if alpha is not None and left_norm > 0.0:
        residual = float(tree_norm(tree_axpy(left, -alpha, right))) / left_norm
    return {
        "left_gradient_norm": left_norm,
        "right_gradient_norm": right_norm,
        "gradient_dot_product": dot,
        "gradient_cosine": cosine,
        "best_alpha_left_over_right": alpha,
        "relative_nonproportional_residual": residual,
        "relative_left_component_orthogonal_to_right": residual,
    }


@dataclass(frozen=True)
class _WeightedForward:
    flat_parameters: np.ndarray
    value: float
    entries: tuple


class WeightedFixedStateObjective:
    """Unnormalized weighted sum of exact fixed-state tendency defects.

    The operation and observation interfaces are the same production oracles
    used by Test-2A-3A.  The only new input is an explicit positive,
    parameter-independent coefficient per observation.
    """

    def __init__(self, observations, operations, coefficients):
        self.observations = tuple(observations)
        self.operations = operations
        self.coefficients = tuple(float(value) for value in coefficients)
        if len(self.observations) != len(self.coefficients):
            raise ValueError("one coefficient is required per observation")
        if not self.observations:
            raise ValueError("weighted fixed-state objective requires observations")
        if not all(np.isfinite(value) and value >= 0.0 for value in self.coefficients):
            raise ValueError("coefficients must be finite and nonnegative")
        self._forward_cache = None

    def _forward(self, parameters):
        parameters = validate_float64_tree(parameters, name="parameters")
        flat, _ = ravel_pytree(parameters)
        flat = np.asarray(flat, dtype=np.float64).copy()
        if self._forward_cache is not None and np.array_equal(
            flat, self._forward_cache.flat_parameters
        ):
            return parameters, self._forward_cache
        total = 0.0
        entries = []
        for observation, coefficient in zip(self.observations, self.coefficients):
            prediction = self.operations.predict(parameters, observation)
            residual = self.operations.subtract(
                prediction.tendency,
                observation.target_tendency,
                f"test2a_h1_weighted_residual_{observation.step}",
            )
            total += coefficient * float(
                self.operations.squared_mass_norm(residual)
            )
            entries.append((observation, prediction, residual, coefficient))
        self._forward_cache = _WeightedForward(flat, float(total), tuple(entries))
        return parameters, self._forward_cache

    def value(self, parameters):
        return self._forward(parameters)[1].value

    def value_and_gradient(self, parameters):
        parameters, forward = self._forward(parameters)
        gradient = tree_zeros(parameters)
        for observation, prediction, residual, coefficient in forward.entries:
            contribution = self.operations.gradient_contribution(
                parameters, observation, prediction, residual
            )
            gradient = tree_axpy(gradient, coefficient, contribution)
        return forward.value, gradient

    def gradient(self, parameters):
        return self.value_and_gradient(parameters)[1]


def _target_squared_mass_norm(helper, target, step):
    dual = helper.state_mass_map(target, f"test2a_h1_target_mass_{step}")
    return float(helper.dual_pairing(dual, target))


def _equivalent_tendency(truth_target, prefix_state, dt, step):
    result = _copy_function(truth_target, f"test2a_h1_equivalent_tendency_{step}")
    with result.dat.vec as output, prefix_state.dat.vec_ro as prefix:
        output.axpy(-1.0, prefix)
        output.scale(1.0 / float(dt))
    return result


def _fieldwise_maximum_difference(left, right):
    names = ("v", "h", "S", "Qv", "Qc", "Qr")
    return {
        name: float(
            np.max(
                np.abs(
                    np.asarray(actual.dat.data_ro, dtype=np.float64)
                    - np.asarray(expected.dat.data_ro, dtype=np.float64)
                )
            )
        )
        for name, actual, expected in zip(names, left.subfunctions, right.subfunctions)
    }


def _state_relative_mass_error(helper, actual, expected, name):
    residual = _copy_function(actual, f"{name}_residual")
    with residual.dat.vec as output, expected.dat.vec_ro as reference:
        output.axpy(-1.0, reference)
    residual_dual = helper.state_mass_map(residual, f"{name}_residual_mass")
    expected_dual = helper.state_mass_map(expected, f"{name}_expected_mass")
    numerator = float(helper.dual_pairing(residual_dual, residual))
    denominator = float(helper.dual_pairing(expected_dual, expected))
    return np.sqrt(numerator / denominator)


def _postprefix_objectives(case, truth, h1_objective):
    """Construct analytical and stored-target fixed post-prefix controls."""
    helper = case.helper
    moist = helper.moist_helper
    operations = ProductionDiscreteOfflineOperations(moist, moist.local_physics)
    analytical_observations = []
    stored_target_observations = []
    coefficients = []
    backend_parity = []
    maximum_original_r = 0.0
    for step in range(80):
        prefix_state = h1_objective._prefixes[step].state_out
        target = moist.take_forward_step_cached(
            prefix_state, case.t0 + step * case.dt, case.dt
        )
        maximum_original_r = max(
            maximum_original_r,
            float(np.max(np.abs(np.asarray(target.rates["R"], dtype=np.float64)))),
        )
        analytical_a_tendency = moist.state_riesz_representative(
            moist.source_assembly(_a_sensitive_source(target)),
            f"test2a_h1_postprefix_analytical_A_{step}",
        )
        payload = ProductionObservationPayload(
            packed_state=moist._to_device_tree(target.packed_state),
            packed_fields=moist._to_device_tree(target.packed_fields),
            moist_parameters=moist._to_device_tree(target.parameters),
            target_original_r=np.asarray(target.rates["R"], dtype=np.float64),
        )
        analytical_observations.append(
            DiscreteOfflineObservation(
                step=step,
                payload=payload,
                target_tendency=target.tendency.copy(deepcopy=True),
                analytical_a_tendency=analytical_a_tendency,
            )
        )
        equivalent = _equivalent_tendency(
            truth[step + 1], prefix_state, case.dt, step
        )
        stored_target_observations.append(
            DiscreteOfflineObservation(
                step=step,
                payload=payload,
                target_tendency=equivalent,
                analytical_a_tendency=analytical_a_tendency,
            )
        )
        target_norm = _target_squared_mass_norm(helper, truth[step + 1], step + 1)
        coefficients.append(
            h1_tendency_loss_coefficient(case.dt, target_norm, 1.0)
        )
        backend_parity.append(
            {
                "start_step": step,
                "target_step": step + 1,
                "relative_mixed_mass_error": _state_relative_mass_error(
                    helper,
                    target.state_out,
                    truth[step + 1],
                    f"test2a_h1_backend_parity_{step}",
                ),
                "fieldwise_maximum_absolute_difference": (
                    _fieldwise_maximum_difference(target.state_out, truth[step + 1])
                ),
            }
        )
    return {
        "operations": operations,
        "analytical_global": DeployedDiscreteOfflineObjective(
            analytical_observations, operations
        ),
        "analytical_h1_weighted": WeightedFixedStateObjective(
            analytical_observations, operations, coefficients
        ),
        "stored_target_h1_weighted": WeightedFixedStateObjective(
            stored_target_observations, operations, coefficients
        ),
        "coefficients": coefficients,
        "backend_parity": backend_parity,
        "maximum_absolute_original_R_at_postprefix_states": maximum_original_r,
    }


def _fixed_prefix_parameter_independence(case, truth, prefixes, left, right):
    """Numerically verify children 1--5 ignore neural parameters."""
    records = []
    with case.physical_c0(0.14):
        for step in (0, 40, 79):
            left_step = case.helper.take_forward_step_cached(
                truth[step],
                case.t0 + step * case.dt,
                case.dt,
                neural_parameters=left,
            )
            right_step = case.helper.take_forward_step_cached(
                truth[step],
                case.t0 + step * case.dt,
                case.dt,
                neural_parameters=right,
            )
            cached = prefixes[step].state_out
            records.append(
                {
                    "start_step": step,
                    "left_vs_right_children_1_to_5": _fieldwise_maximum_difference(
                        left_step.boundary_states[5], right_step.boundary_states[5]
                    ),
                    "left_vs_fixed_prefix_children_1_to_5": (
                        _fieldwise_maximum_difference(
                            left_step.boundary_states[5], cached
                        )
                    ),
                    "right_vs_fixed_prefix_children_1_to_5": (
                        _fieldwise_maximum_difference(
                            right_step.boundary_states[5], cached
                        )
                    ),
                    "complete_step_left_vs_right": _fieldwise_maximum_difference(
                        left_step.state_out, right_step.state_out
                    ),
                }
            )
    return records


def _parameters(model_configuration):
    seed = initialize_mlp(model_configuration)
    if parameter_pytree_sha256(seed) != CANONICAL_SEED_SHA256:
        raise ValueError("canonical Test-2A seed-0 pytree changed")
    result = {"canonical_seed0": seed}
    for name, record in PROBE_ARTIFACTS.items():
        parameters, configuration = load_mlp_parameters(record["path"])
        if configuration != model_configuration:
            raise ValueError(f"{name} architecture changed")
        if parameter_pytree_sha256(parameters) != record["sha256"]:
            raise ValueError(f"{name} parameter fingerprint changed")
        result[name] = parameters
    flat, unravel = ravel_pytree(seed)
    direction = np.linspace(-1.0, 1.0, int(flat.size), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    amplitude = 1.0e-3 * max(1.0, float(np.linalg.norm(np.asarray(flat))))
    result["seed0_plus_deterministic_relative_1e-3"] = unravel(
        flat + amplitude * jnp.asarray(direction, dtype=jnp.float64)
    )
    return result


def _objective_comparison(left_value, left_gradient, right_value, right_gradient):
    relation = parameter_gradient_relation(left_gradient, right_gradient)
    relation.update(
        {
            "left_objective": float(left_value),
            "right_objective": float(right_value),
            "objective_ratio_left_over_right": (
                None if float(right_value) == 0.0 else float(left_value / right_value)
            ),
        }
    )
    return relation


def run_equivalence_audit(
    trajectory_configuration,
    discrete_cache,
    selected_operator_configuration,
    output,
):
    selected_operator = load_selected_configuration(selected_operator_configuration)
    model_configuration = mlp_configuration_from_record(selected_operator["model"])
    cache = load_fixed_cache(discrete_cache)
    current_m2 = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)

    _, case, truth, _, loaded_configuration = _build_case(
        trajectory_configuration, maximum_truth_step=80
    )
    if loaded_configuration != model_configuration:
        raise ValueError("trajectory and Method-2 neural architectures differ")
    h1 = NeuralTrajectoryObjective(
        case,
        truth,
        reset_windows(range(80), 1, "endpoint", (1.0,)),
        c0=0.14,
        use_fixed_prefix=True,
    )
    postprefix = _postprefix_objectives(case, truth, h1)

    parameter_probes = _parameters(model_configuration)
    probes = []
    for name, parameters in parameter_probes.items():
        j_disc, g_disc = current_m2.value_and_gradient(parameters)
        j_h1, g_h1 = h1.value_and_gradient(parameters)
        j_post, g_post = postprefix["analytical_global"].value_and_gradient(parameters)
        j_post_weighted, g_post_weighted = postprefix[
            "analytical_h1_weighted"
        ].value_and_gradient(parameters)
        j_stored, g_stored = postprefix[
            "stored_target_h1_weighted"
        ].value_and_gradient(parameters)
        probes.append(
            {
                "name": name,
                "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
                "H1_vs_current_M2": _objective_comparison(
                    j_h1, g_h1, j_disc, g_disc
                ),
                "H1_vs_postprefix_global": _objective_comparison(
                    j_h1, g_h1, j_post, g_post
                ),
                "H1_vs_postprefix_analytical_H1_weighting": _objective_comparison(
                    j_h1, g_h1, j_post_weighted, g_post_weighted
                ),
                "H1_vs_postprefix_stored_target_exact_weighting": (
                    _objective_comparison(j_h1, g_h1, j_stored, g_stored)
                ),
                "objective_values": {
                    "current_M2_Xk_global_normalization": j_disc,
                    "literal_H1_reset_Yk_certification_metric": j_h1,
                    "postprefix_Yk_global_M2_normalization": j_post,
                    "postprefix_Yk_analytical_target_H1_weighting": j_post_weighted,
                    "postprefix_Yk_stored_target_exact_H1_weighting": j_stored,
                },
            }
        )

    # A fresh H=2 contrast at seed zero: continuous recursion versus two
    # independently supplied truth starts.  This is differentiation only.
    seed = parameter_probes["canonical_seed0"]
    continuous_h2 = NeuralTrajectoryObjective(
        case,
        truth,
        continuous_rollout(2, "accumulated", (0.5, 0.5)),
        c0=0.14,
        use_fixed_prefix=True,
    )
    independent_h1 = NeuralTrajectoryObjective(
        case,
        truth,
        reset_windows((0, 1), 1, "endpoint", (0.5,)),
        c0=0.14,
        use_fixed_prefix=True,
    )
    j_h2, g_h2 = continuous_h2.value_and_gradient(seed)
    j_independent, g_independent = independent_h1.value_and_gradient(seed)
    prefix_independence = _fixed_prefix_parameter_independence(
        case,
        truth,
        h1._prefixes,
        seed,
        parameter_probes["matched_m1_200k"],
    )

    parity = postprefix["backend_parity"]
    record = {
        "status": "complete",
        "benchmark_stage": "Test 2A H=1 truth-reset / Method-2 equivalence audit",
        "interpretation": "diagnostic only; no optimization and no production behavior change",
        "branch_science": {
            "current_Method2_fixed_evaluation_states": "X*_k for k=0,...,80",
            "literal_H1_fixed_evaluation_states": "Y_k=P(X*_k) for k=0,...,79",
            "H1_targets": "stored boundary states X*_{k+1}, k=0,...,79",
            "parameter_entry": "neural parameters enter child 6 only",
            "fixed_prefix_parameter_independence": prefix_independence,
            "H1_recursive_cross_time_feedback": False,
            "first_necessary_recursive_feedback_horizon": 2,
        },
        "time_indexing": {
            "boundary_truth_states": [0, 80],
            "Method2_observation_steps": [0, 80],
            "H1_reset_origin_steps": [0, 79],
            "H1_target_steps": [1, 80],
            "prefix_children": [
                "dry_rk4_0",
                "dry_rk4_1",
                "hyperviscosity_euler",
                "dg_ssprk43_0",
                "dg_ssprk43_1",
            ],
            "child6_time": "t_n",
            "child6_applied_dt": float(case.dt),
        },
        "analytical_relation": {
            "delta_A": "A_theta(Y_k)-A_star(Y_k)",
            "source_error_mixed_field_order_v_h_S_Qv_Qc_Qr": (
                "(0,0,h*beta2*delta_A,h*delta_A,-h*delta_A,0)"
            ),
            "tendency_error": "G(Y_k) delta_A = M^{-1} W H(Y_k) delta_A",
            "state_error_if_same_analytical_child_target": (
                "dt * G(Y_k) delta_A"
            ),
            "H1_term_if_same_analytical_child_target": (
                "0.5*w_k*dt^2*||G(Y_k)delta_A||_M^2/||X*_{k+1}||_M^2"
            ),
            "R_cancellation": (
                "exact because neural and analytical child 6 evaluate the same original R law at identical Y_k"
            ),
            "maximum_absolute_original_R_at_postprefix_states": postprefix[
                "maximum_absolute_original_R_at_postprefix_states"
            ],
        },
        "objective_convention_differences": [
            "current Method 2 evaluates X*_0,...,X*_80; H=1 evaluates post-prefix Y_0,...,Y_79",
            "current Method 2 has 81 fixed-state rate samples; H=1 has 80 transitions",
            "current Method 2 compares tendencies and therefore has no dt^2 factor",
            "H=1 certification loss has the positive 0.5*dt^2 factor",
            "current Method 2 uses one global analytical-A tendency-energy normalizer",
            "H=1 certification loss uses one full-state mixed-mass normalizer per target",
            "both use the production mixed mass metric; v,h,Qr H=1 analytical defects are structurally zero",
        ],
        "postprefix_coefficients": {
            "count": len(postprefix["coefficients"]),
            "minimum": float(min(postprefix["coefficients"])),
            "maximum": float(max(postprefix["coefficients"])),
            "all_equal": bool(
                np.allclose(
                    postprefix["coefficients"],
                    postprefix["coefficients"][0],
                    rtol=0.0,
                    atol=0.0,
                )
            ),
        },
        "analytical_JAX_child_vs_stored_UFL_truth": {
            "maximum_relative_mixed_mass_error": float(
                max(item["relative_mixed_mass_error"] for item in parity)
            ),
            "maximum_fieldwise_absolute_difference": {
                field: float(
                    max(
                        item["fieldwise_maximum_absolute_difference"][field]
                        for item in parity
                    )
                )
                for field in ("v", "h", "S", "Qv", "Qc", "Qr")
            },
            "per_transition": parity,
        },
        "parameter_probes": probes,
        "H2_cross_time_contrast": {
            "continuous_H2_objective": j_h2,
            "two_independent_H1_objective": j_independent,
            "gradient_relation_H2_over_independent_H1": parameter_gradient_relation(
                g_h2, g_independent
            ),
            "mechanism": (
                "after child 6 of step one, Xhat_1(theta) enters every child of step two and the subsequent neural A and original state-dependent R"
            ),
        },
        "classification": {
            "case": "C_PRECISE_FIXED_STATE_AND_TARGET_CONVENTION_DIFFERENCES",
            "statement": (
                "H=1 contains no recursive solver feedback and is exactly a fixed-state deployed-child objective when its Y_k states, stored targets, support, and weights are retained. Current Method 2 differs because it evaluates X*_k, includes k=80, uses a global normalization, and targets the analytical JAX moist tendency rather than the stored UFL complete-step boundary state."
            ),
            "first_recursive_horizon": 2,
        },
        "truth_state_access": {
            "loaded_state_indices": [0, 80],
            "states_after_80_accessed": False,
        },
        "optimization_performed": False,
    }
    write_json_record(output, record)
    return record


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-configuration",
        default="dimswe/configs/test2a_trajectory_prep.json",
    )
    parser.add_argument(
        "--discrete-cache",
        default=(
            "external-results/test2a/deployed-discrete-offline/"
            "fixed_operator_cache.npz"
        ),
    )
    parser.add_argument(
        "--selected-operator-configuration",
        default="dimswe/configs/test2a_selected_operator.json",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    run_equivalence_audit(
        args.trajectory_configuration,
        args.discrete_cache,
        args.selected_operator_configuration,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "WeightedFixedStateObjective",
    "h1_structural_source_error",
    "h1_tendency_loss_coefficient",
    "parameter_gradient_relation",
    "run_equivalence_audit",
)
