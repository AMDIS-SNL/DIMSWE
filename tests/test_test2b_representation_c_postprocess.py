import numpy as np

from dimswe.test2b_representation_c_postprocess import (
    BETA2,
    _defect_metrics,
    _projection_diagnostics,
)


def test_two_rate_projection_recovers_rates_and_zero_residual():
    h = np.asarray((700.0, 800.0))
    a = np.asarray((1.0e-7, -2.0e-7))
    r = np.asarray((3.0e-11, 5.0e-11))
    source = np.stack(
        (h * BETA2 * a, h * a, -h * (a + r), h * r), axis=-1
    )
    result = _projection_diagnostics(
        source, source, h, np.ones(2),
        np.asarray((6.0e-3, 7.0e-5, 7.0e-5, 1.5e-8)),
    )
    assert result["normalized_off_manifold_RMS"] < 1.0e-14
    assert result["off_manifold_fraction_of_source_magnitude"] < 1.0e-14
    assert result["projected_A"]["maximum_absolute_error"] < 1.0e-20
    assert result["projected_R"]["maximum_absolute_error"] < 1.0e-22


def test_projection_reports_unconstrained_direction_without_modification():
    h = np.ones(3)
    truth = np.zeros((3, 4))
    prediction = np.asarray(
        ((0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0),
         (1.0, 0.0, 0.0, 0.0))
    )
    original = prediction.copy()
    result = _projection_diagnostics(
        prediction, truth, h, np.ones(3), np.ones(4)
    )
    assert result["normalized_off_manifold_RMS"] > 0.0
    assert result["off_manifold_fraction_of_source_magnitude"] > 0.0
    assert np.array_equal(prediction, original)


def test_signed_defect_distinguishes_bias_from_cancellation():
    weights = np.ones(2)
    cancelling = _defect_metrics(
        np.asarray(((1.0, -1.0), (2.0, -2.0))), weights
    )
    systematic = _defect_metrics(
        np.asarray(((1.0, 1.0), (2.0, 2.0))), weights
    )
    assert cancelling["RMS"] > 0.0
    assert cancelling["signed_to_absolute_ratio"] == 0.0
    assert systematic["signed_to_absolute_ratio"] == 1.0
