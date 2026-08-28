from types import SimpleNamespace

import numpy as np
import pytest

from dimswe.test2a_backend_offset_audit import (
    TRAINING_TRANSITIONS,
    array_error_statistics,
    coefficient_decomposition_error,
    jax_helper_physics_kind,
    require_analytical_jax_helper,
)


def test_jax_helper_mode_is_explicit_and_neural_default_is_not_analytical():
    analytical = SimpleNamespace(local_physics=None)
    neural = SimpleNamespace(
        local_physics=SimpleNamespace(physics_mode="neural_A_original_R")
    )

    assert jax_helper_physics_kind(analytical) == "analytical_A_original_R"
    assert jax_helper_physics_kind(neural) == "frozen_neural_A_original_R"
    assert require_analytical_jax_helper(analytical) is analytical
    with pytest.raises(ValueError, match="frozen neural parameters"):
        require_analytical_jax_helper(neural)


def test_unknown_local_physics_mode_is_rejected():
    helper = SimpleNamespace(
        local_physics=SimpleNamespace(physics_mode="unexpected")
    )
    with pytest.raises(ValueError, match="unsupported"):
        jax_helper_physics_kind(helper)


def test_array_error_statistics_reports_rates_and_signs():
    actual = np.array([1.0, -1.0, 0.0, -2.0])
    reference = np.array([1.0, 1.0, 0.0, -1.0])
    result = array_error_statistics(actual, reference)

    np.testing.assert_allclose(result["maximum_absolute_difference"], 2.0)
    np.testing.assert_allclose(result["RMS_difference"], np.sqrt(5.0 / 4.0))
    np.testing.assert_allclose(result["reference_RMS"], np.sqrt(3.0 / 4.0))
    assert result["sign_disagreement_count"] == 1
    assert result["sign_disagreement_fraction"] == 0.25


def test_zero_reference_rate_is_handled_explicitly():
    result = array_error_statistics(np.zeros(4), np.zeros(4))
    assert result["relative_RMS_difference"] is None
    assert result["sign_disagreement_count"] == 0


def test_three_way_decomposition_identity():
    stored = np.array([1.0, -2.0, 3.0])
    backend = np.array([-0.5, 4.0, 2.0])
    total = stored + backend
    assert coefficient_decomposition_error(stored, backend, total) == 0.0
    assert coefficient_decomposition_error(stored, backend, total + 1.0) == 1.0


def test_audit_transition_support_stops_at_truth_state_80():
    assert TRAINING_TRANSITIONS == tuple(range(80))
    assert min(TRAINING_TRANSITIONS) == 0
    assert max(TRAINING_TRANSITIONS) + 1 == 80

