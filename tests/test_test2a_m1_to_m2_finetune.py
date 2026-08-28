"""Cheap contracts for the Test-2A M1-to-M2 fine-tuning diagnostic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyrol")

from dimswe.test2a_discrete_training import (
    load_discrete_training_configuration,
    load_training_initial_parameters,
    validate_resume_record,
)
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2a_m1_to_m2_finetune import (
    OPERATOR_200K_SHA256,
    gradient_geometry,
)
from dimswe.test2a_operator import (
    load_selected_configuration,
    mlp_configuration_from_record,
)


CONFIGURATION = "dimswe/configs/test2a_m1_to_m2_finetune_50k.json"


def test_finetune_configuration_is_new_history_warm_start():
    record = load_discrete_training_configuration(CONFIGURATION)
    initialization = record["initialization"]
    assert initialization == {
        "kind": "operator_200k_warm_start",
        "operator_pretraining": True,
        "source_parameter_file": (
            "external-results/test2a/fair-longfit/"
            "operator-seed0-m20-200k/final_parameters.npz"
        ),
        "parameter_pytree_sha256": OPERATOR_200K_SHA256,
        "initial_J_op": 0.000373006108792648,
        "initial_J_disc": 0.0008346864309047664,
        "new_optimizer_process": True,
        "source_optimizer_secant_history_reused": False,
    }
    assert record["checkpoint_accepted_iterations"] == [
        0,
        1000,
        5000,
        10000,
        25000,
        50000,
    ]
    assert record["optimizer"] == {
        "library": "PyROL/ROL",
        "method": "line-search L-BFGS",
        "maximum_secant_storage": 20,
        "accepted_iteration_limit": 50000,
        "gradient_tolerance": 1.0e-8,
        "step_tolerance": 1.0e-12,
        "exact_gradients": True,
        "production_HVP": False,
    }
    assert record["truth_state_indices"] == [0, 80]
    assert record["states_after_80_forbidden"] is True
    assert record["recursive_model_state_propagation"] is False


def test_exact_operator_200k_artifact_is_loaded_without_reinitialization():
    record = load_discrete_training_configuration(CONFIGURATION)
    selected = load_selected_configuration(record["selected_operator_configuration"])
    model = mlp_configuration_from_record(selected["model"])
    parameters = load_training_initial_parameters(record, model)
    assert parameter_pytree_sha256(parameters) == OPERATOR_200K_SHA256
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(parameters))


def test_gradient_geometry_reports_exact_orthogonal_residual():
    operator = {"x": jnp.array([1.0, 0.0], dtype=jnp.float64)}
    discrete = {"x": jnp.array([2.0, 3.0], dtype=jnp.float64)}
    result = gradient_geometry(4.0, operator, 5.0, discrete)
    assert result["operator_gradient_norm"] == 1.0
    assert result["deployed_discrete_gradient_norm"] == pytest.approx(np.sqrt(13.0))
    assert result["gradient_dot_product"] == 2.0
    assert result["best_scalar_alpha_for_g_disc_minus_alpha_g_op"] == 2.0
    assert result["relative_orthogonal_component_of_g_disc"] == pytest.approx(
        3.0 / np.sqrt(13.0)
    )


def test_iteration_zero_is_a_valid_parameter_checkpoint_for_explicit_resume():
    record = {
        "status": "in_progress",
        "configuration_sha256": "configuration",
        "cache_npz_sha256": "cache",
        "last_checkpoint_accepted_iteration": 0,
    }
    assert validate_resume_record(record, "configuration", "cache") == 0
