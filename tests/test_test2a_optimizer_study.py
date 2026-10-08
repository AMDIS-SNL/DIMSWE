"""Cheap dispatch/fairness tests for the controlled Test 2A optimizer study."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dimswe.test2a_operator import MLPConfiguration, initialize_mlp, operator_metrics
from dimswe.test2a_optimizer_study import (
    _conservative_classification,
    build_study_rol_parameters,
    load_optimizer_study_configuration,
    parameter_fingerprint,
    run_rol_method,
)


STUDY_CONFIGURATION = Path("dimswe/configs/test2a_optimizer_study.json")


def _quadratic(parameters):
    left = parameters["left"] - jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    right = parameters["nested"]["right"] - jnp.asarray([[1.25]], dtype=jnp.float64)
    return jnp.vdot(left, left) + 2.0 * jnp.vdot(right, right)


def _initial_parameters():
    return {
        "left": jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        "nested": {"right": jnp.asarray([[0.1]], dtype=jnp.float64)},
    }


def _lbfgs(memory=10, iterations=30):
    return {
        "name": f"tiny_lbfgs_m{memory}",
        "kind": "line_search_lbfgs",
        "maximum_secant_storage": memory,
        "gradient_tolerance": 1.0e-10,
        "step_tolerance": 1.0e-14,
        "iteration_limit": iterations,
    }


def _trust_region(iterations=20):
    return {
        "name": "tiny_trust_region_tcg",
        "kind": "trust_region_truncated_cg_exact_hvp",
        "gradient_tolerance": 1.0e-10,
        "step_tolerance": 1.0e-14,
        "iteration_limit": iterations,
        "initial_radius": 10.0,
        "maximum_radius": 5000.0,
        "krylov_absolute_tolerance": 1.0e-10,
        "krylov_relative_tolerance": 1.0e-4,
        "krylov_iteration_limit": 20,
    }


def test_selected_study_freezes_problem_and_common_initialization():
    study = load_optimizer_study_configuration(STUDY_CONFIGURATION)
    frozen = study["frozen_problem"]
    assert frozen["sample_count"] == 331_776
    assert frozen["features"] == ["h", "S", "Qv", "Qc", "B"]
    assert frozen["parameter_count"] == 1281
    assert "seed 0" in frozen["initialization"]
    first = initialize_mlp(MLPConfiguration())
    second = initialize_mlp(MLPConfiguration())
    assert parameter_fingerprint(first) == parameter_fingerprint(second)


def test_optimizer_dispatch_uses_exact_installed_rol_names():
    lbfgs = build_study_rol_parameters(_lbfgs(memory=20))
    assert lbfgs.sublist("Step").get("Type") == "Line Search"
    secant = lbfgs.sublist("General").sublist("Secant")
    assert secant.get("Type") == "Limited-Memory BFGS"
    assert secant.get("Maximum Storage") == 20

    trust = build_study_rol_parameters(_trust_region())
    assert trust.sublist("Step").get("Type") == "Trust Region"
    assert (
        trust.sublist("Step").sublist("Trust Region").get("Subproblem Solver")
        == "Truncated CG"
    )
    trust_secant = trust.sublist("General").sublist("Secant")
    assert trust_secant.get("Use as Hessian") is False
    assert trust_secant.get("Use as Preconditioner") is False


def test_common_start_lbfgs_uses_gradients_and_no_hvps():
    initial = _initial_parameters()
    fingerprint = parameter_fingerprint(initial)
    final, result = run_rol_method(_quadratic, initial, _lbfgs())
    assert result["starting_parameter_sha256"] == fingerprint
    assert parameter_fingerprint(initial) == fingerprint
    assert result["gradient_evaluations"] > 0
    assert result["HVP_evaluations"] == 0
    assert result["exact_JAX_HVP_requested"] is False
    assert result["final_objective"] < result["initial_objective"]
    assert "EExitStatus" in result["termination_reason"]
    assert result["accepted_iteration_history"]
    assert np.isfinite(np.asarray(final["left"])).all()


def test_trust_region_tcg_actually_calls_exact_hvp_from_same_start():
    initial = _initial_parameters()
    fingerprint = parameter_fingerprint(initial)
    _, result = run_rol_method(_quadratic, initial, _trust_region())
    assert result["starting_parameter_sha256"] == fingerprint
    assert parameter_fingerprint(initial) == fingerprint
    assert result["gradient_evaluations"] > 0
    assert result["HVP_evaluations"] > 0
    assert result["exact_JAX_HVP_requested"] is True
    assert result["final_objective"] < result["initial_objective"]
    assert "EExitStatus" in result["termination_reason"]


def test_metrics_are_method_independent_and_use_existing_definition():
    truth = np.asarray([-2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    prediction = np.asarray([-1.5, -0.5, 0.25, 2.5], dtype=np.float64)
    left = operator_metrics(prediction, truth)
    right = operator_metrics(prediction.copy(), truth.copy())
    assert left == right
    assert set(left) == {
        "sample_count",
        "normalized_mse",
        "physical_rmse_A",
        "physical_mae_A",
        "relative_rms_error",
        "maximum_absolute_error",
        "correlation",
        "sign_accuracy",
        "sign_accuracy_threshold",
        "sign_accuracy_strata",
        "magnitude_strata",
    }


def test_full_bfgs_is_audited_as_unavailable_not_silently_substituted():
    study = load_optimizer_study_configuration(STUDY_CONFIGURATION)
    audit = study["full_bfgs_availability_audit"]
    assert "does not expose dense full BFGS" in audit["disposition"]
    assert "Limited-Memory BFGS" in audit["installed_secant_types"]
    assert all(method["kind"] != "full_bfgs" for method in study["methods"])


def test_nonstationary_improvement_remains_optimization_and_model_unresolved():
    metrics = operator_metrics(
        np.asarray([0.5, -0.5], dtype=np.float64),
        np.asarray([1.0, -1.0], dtype=np.float64),
    )
    record = {
        "name": "improved_but_nonstationary",
        "status": "complete",
        "initial_gradient_norm": 1.0,
        "final_gradient_norm": 0.1,
        "final_objective": 0.25,
        "metrics_complete_training_support": metrics,
    }
    classification, evidence = _conservative_classification(
        [record], {"final_objective": 0.7, "relative_rms_error": 0.82}
    )
    assert classification == "OPTIMIZATION_AND_MODEL_BOTH_UNRESOLVED"
    assert evidence["credible_stationarity"] is False
    assert evidence["substantially_better_than_100_iteration_reference"] is True
