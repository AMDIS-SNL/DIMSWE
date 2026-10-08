import numpy as np

from dimswe.test2b_representation_b_postprocess import (
    _activity_masks,
    _r_metrics,
)


def test_activity_masks_distinguish_positive_and_negative_rain():
    scale = 1.0e-6
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    rate = np.asarray([[0.0, 2.0 * tolerance, -2.0 * tolerance]])
    h = np.ones_like(rate)
    qr = np.zeros_like(rate)
    masks = _activity_masks(rate, h, qr, scale, dt=100.0)
    assert masks["meaningful_positive"].tolist() == [[False, True, False]]
    assert masks["meaningful_negative"].tolist() == [[False, False, True]]


def test_r_metrics_expose_false_positive_and_false_negative_activity():
    scale = 1.0e-6
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    truth = np.asarray([[0.0, 4.0 * tolerance, 4.0 * tolerance]])
    prediction = np.asarray(
        [[2.0 * tolerance, 0.0, 5.0 * tolerance]]
    )
    result = _r_metrics(
        prediction,
        truth,
        np.ones_like(truth),
        np.ones_like(truth),
        np.zeros_like(truth),
        scale,
    )
    assert result["truth_active_sample_count"] == 2
    assert result["false_positive_count"] == 1
    assert result["false_negative_count"] == 1
    assert np.isclose(
        result["false_negative_rate_given_truth_active"], 0.5
    )
    assert np.isclose(
        result["false_positive_rate_given_truth_inactive"], 1.0
    )
    assert result["truth_active_samples"]["sample_count"] == 2
