"""Cheap continuation-contract tests without the production operator fit."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from pyrol import Problem, Solver

from dimswe.test2a_continuation import (
    CheckpointingPytreeObjective,
    _continuation_decision,
    _physical_metric_stability,
    _relative_decrease,
    _verify_source_result,
    load_continuation_configuration,
    verify_parameter_artifact,
)
from dimswe.test2a_operator import (
    MLPConfiguration,
    initialize_mlp,
    save_mlp_parameters,
)
from dimswe.test2a_optimizer_study import parameter_fingerprint
from dimswe.test2a_pyrol import build_test2a_lbfgs_parameters


CONFIGURATION = Path("dimswe/configs/test2a_m20_continuation.json")
PLATEAU_CONFIGURATION = Path(
    "dimswe/configs/test2a_m20_plateau_continuation.json"
)


def test_selected_continuation_freezes_method_budget_and_hashes():
    record = load_continuation_configuration(CONFIGURATION)
    assert record["additional_accepted_iteration_limit"] == 1500
    assert record["checkpoint_additional_iterations"] == [100, 250, 500, 1000, 1500]
    assert record["optimizer"]["maximum_secant_storage"] == 20
    assert record["optimizer"]["production_HVP"] is False
    assert len(record["source"]["parameter_npz_sha256"]) == 64
    assert len(record["source"]["parameter_pytree_sha256"]) == 64


def test_plateau_continuation_freezes_latest_source_and_five_thousand_budget():
    record = load_continuation_configuration(PLATEAU_CONFIGURATION)
    assert record["additional_accepted_iteration_limit"] == 5000
    assert record["checkpoint_additional_iterations"][-3:] == [4750, 4900, 5000]
    assert record["optimizer"]["maximum_secant_storage"] == 20
    assert record["optimizer"]["production_HVP"] is False
    assert record["source"]["result_contract"] == "completed_continuation"
    assert record["source"]["accepted_iterations"] == 1500
    assert record["source"]["parameter_npz_sha256"].startswith("346691")


def test_completed_continuation_source_contract_rejects_wrong_decision():
    source = load_continuation_configuration(PLATEAU_CONFIGURATION)["source"]
    result = {
        "status": "complete",
        "benchmark_stage": source["expected_benchmark_stage"],
        "decision": source["expected_decision"],
        "final_metrics": {"normalized_mse": source["final_objective"]},
        "optimizer": {
            "additional_accepted_iterations": 1500,
            "maximum_secant_storage": 20,
            "HVP_evaluations": 0,
        },
    }
    _verify_source_result(source, result)
    result["decision"] = "PRACTICALLY_CONVERGED_MODEL_INADEQUATE"
    with pytest.raises(ValueError, match="source result"):
        _verify_source_result(source, result)


def test_parameter_artifact_verifies_file_and_pytree_fingerprints(tmp_path):
    configuration = MLPConfiguration(hidden_layers=(3,), seed=4)
    parameters = initialize_mlp(configuration)
    path = tmp_path / "parameters.npz"
    save_mlp_parameters(path, parameters, configuration)
    from dimswe.test2a_continuation import _file_sha256

    restored, restored_configuration = verify_parameter_artifact(
        path, _file_sha256(path), parameter_fingerprint(parameters)
    )
    assert restored_configuration == configuration
    assert parameter_fingerprint(restored) == parameter_fingerprint(parameters)
    with pytest.raises(ValueError, match="NPZ fingerprint mismatch"):
        verify_parameter_artifact(path, "0" * 64, parameter_fingerprint(parameters))
    with pytest.raises(ValueError, match="pytree fingerprint mismatch"):
        verify_parameter_artifact(path, _file_sha256(path), "0" * 64)


def test_checkpointing_objective_observes_accepts_and_lbfgs_uses_no_hvp():
    initial = {"x": jnp.asarray([2.0, -3.0], dtype=jnp.float64)}
    target = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    calls = []

    def objective(parameters):
        error = parameters["x"] - target
        return jnp.vdot(error, error)

    def callback(control, local_accepted, adapter):
        calls.append((local_accepted, control.array.copy()))

    adapter = CheckpointingPytreeObjective(
        objective, initial, use_jit=False, accepted_callback=callback
    )
    control = adapter.vector_from_pytree(initial)
    problem = Problem(adapter, control)
    solver = Solver(
        problem,
        build_test2a_lbfgs_parameters(
            {
                "gradient_tolerance": 1.0e-11,
                "step_tolerance": 1.0e-14,
                "iteration_limit": 20,
                "maximum_secant_storage": 20,
            }
        ),
    )
    solver.solve()
    assert calls[0][0] == 0
    assert calls[-1][0] == int(solver.getAlgorithmState().iter)
    assert adapter.gradient_evaluations > 0
    assert adapter.hvp_evaluations == 0


def test_recent_objective_decrease_uses_accepted_history_only():
    history = [
        {"objective": float(2.0 - 0.01 * index)} for index in range(121)
    ]
    record = _relative_decrease(history, 100)
    assert record["starting_objective"] == pytest.approx(1.8)
    assert record["ending_objective"] == pytest.approx(0.8)
    assert record["relative_decrease"] == pytest.approx(1.0 / 1.8)


def _metrics(relative_rms, sign_accuracy, active_relative_rms):
    return {
        "relative_rms_error": relative_rms,
        "sign_accuracy": sign_accuracy,
        "magnitude_strata": {
            "abs_A_gt_1e-03_max_abs_A": {
                "relative_rms_error": active_relative_rms
            }
        },
    }


def _stationarity(gradient_ratio, step_ratio, recent_decrease):
    return {
        "final_gradient_norm_relative_to_start": gradient_ratio,
        "final_step_norm_relative_to_parameter_norm": step_ratio,
        "recent_100_objective_decrease": {"relative_decrease": recent_decrease},
    }


def test_decision_requires_joint_objective_gradient_and_step_plateau():
    criteria = load_continuation_configuration(CONFIGURATION)[
        "decision_diagnostics"
    ]
    decision, evidence = _continuation_decision(
        _metrics(0.5, 0.7, 0.5),
        _stationarity(0.5, 1.0e-7, 1.0e-5),
        criteria,
    )
    assert decision == "CONTINUING_OPTIMIZER_LIMITED"
    assert evidence["practical_plateau"] is False

    decision, evidence = _continuation_decision(
        _metrics(0.5, 0.7, 0.5),
        _stationarity(0.01, 1.0e-7, 1.0e-5),
        criteria,
    )
    assert decision == "PRACTICALLY_CONVERGED_MODEL_INADEQUATE"
    assert evidence["practical_plateau"] is True

    decision, evidence = _continuation_decision(
        _metrics(0.1, 0.95, 0.1),
        _stationarity(0.01, 1.0e-7, 1.0e-5),
        criteria,
    )
    assert decision == "READY_FOR_EMBEDDING"
    assert evidence["strong_accuracy"] is True


def test_plateau_decision_requires_stable_physical_metrics_and_active_sign_accuracy():
    criteria = load_continuation_configuration(PLATEAU_CONFIGURATION)[
        "decision_diagnostics"
    ]
    metrics = _metrics(0.08, 0.96, 0.09)
    metrics["sign_accuracy_strata"] = {
        "abs_A_gt_1e-03_max_abs_A": {"accuracy": 0.97},
        "abs_A_gt_1e-02_max_abs_A": {"accuracy": 0.98},
    }
    stationarity = _stationarity(0.05, 1.0e-7, 1.0e-4)
    stationarity["physical_metrics_stability"] = {"stable": False}
    decision, evidence = _continuation_decision(metrics, stationarity, criteria)
    assert decision == "STILL_OPTIMIZER_LIMITED"
    assert evidence["practical_plateau"] is False
    stationarity["physical_metrics_stability"] = {"stable": True}
    decision, evidence = _continuation_decision(metrics, stationarity, criteria)
    assert decision == "OPERATOR_REPRESENTATION_ADEQUATE"
    assert evidence["strong_accuracy"] is True


def test_physical_metric_stability_compares_activity_stratified_signs():
    criteria = load_continuation_configuration(PLATEAU_CONFIGURATION)[
        "decision_diagnostics"
    ]

    def metric_record(iteration, relative_rms, correlation, sign):
        return {
            "additional_accepted_iteration": iteration,
            "metrics": {
                "relative_rms_error": relative_rms,
                "correlation": correlation,
                "sign_accuracy": sign,
                "sign_accuracy_strata": {
                    "abs_A_gt_1e-03_max_abs_A": {"accuracy": sign},
                    "abs_A_gt_1e-02_max_abs_A": {"accuracy": sign},
                },
            },
        }

    evidence = _physical_metric_stability(
        [
            metric_record(4900, 0.08, 0.99, 0.97),
            metric_record(5000, 0.07995, 0.99005, 0.9701),
        ],
        criteria,
    )
    assert evidence["stable"] is True
    assert evidence["starting_additional_iteration"] == 4900
