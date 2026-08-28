import numpy as np

from dimswe.test2b_representation_a_postprocess import (
    _summarize_metric_series,
    _weighted_metrics,
)


def test_weighted_metrics_preserve_signed_bias_and_scales():
    result = _weighted_metrics(
        np.asarray([2.0, 0.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 3.0]),
        2.0,
    )
    assert np.isclose(result["physical_RMS_error"], 1.0)
    assert np.isclose(result["normalized_RMS_error"], 0.5)
    assert np.isclose(result["relative_RMS_error"], 1.0)
    assert np.isclose(result["signed_mass_weighted_bias"], -0.5)


def test_weighted_metrics_accept_already_broadcast_state_weights():
    result = _weighted_metrics(
        np.asarray([[2.0, 0.0], [1.0, 1.0]]),
        np.ones((2, 2)),
        np.asarray([[1.0, 3.0], [1.0, 3.0]]),
        2.0,
    )
    assert result["sample_count"] == 4
    assert np.isclose(result["signed_mass_weighted_bias"], -0.25)


def test_metric_series_accumulation_uses_mass_squares():
    records = {
        0: {"numerator": 1.0, "denominator": 4.0, "relative_error": 0.5},
        1: {"numerator": 9.0, "denominator": 16.0, "relative_error": 0.75},
    }
    result = _summarize_metric_series(records, (0, 1))
    assert result["final"] == 0.75
    assert result["maximum"] == 0.75
    assert result["maximum_step"] == 1
    assert np.isclose(result["accumulated"], np.sqrt(0.5))
