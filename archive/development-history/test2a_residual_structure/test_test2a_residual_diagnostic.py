"""Cheap tests for the frozen Test-2A residual diagnostic."""

import jax.numpy as jnp
import numpy as np
import pytest

from dimswe.jax_moist import moist_rates_jax
from dimswe.test2a_residual_diagnostic import (
    deployed_a_switch_diagnostics,
    residual_bins,
    residual_subset_metrics,
    time_residual_metrics,
)


def test_sign_subset_metrics_preserve_error_energy_accounting():
    target = np.asarray([-2.0, -1.0, 1.0, 3.0], dtype=np.float64)
    prediction = np.asarray([-1.0, 0.5, 0.5, 2.0], dtype=np.float64)
    error_energy = float(np.sum((prediction - target) ** 2))
    negative = residual_subset_metrics(
        prediction, target, target < 0.0, global_error_energy=error_energy
    )
    positive = residual_subset_metrics(
        prediction, target, target > 0.0, global_error_energy=error_energy
    )
    assert negative["sample_count"] == 2
    assert positive["sample_count"] == 2
    assert negative["sign_accuracy"] == 0.5
    assert positive["sign_accuracy"] == 1.0
    assert (
        negative["global_residual_squared_energy_fraction"]
        + positive["global_residual_squared_energy_fraction"]
    ) == pytest.approx(1.0)


def test_residual_bins_cover_each_sample_once_and_handle_zero_truth():
    target = np.asarray([0.0, 0.01, 0.1, 1.0], dtype=np.float64)
    prediction = np.asarray([0.1, 0.02, 0.09, 0.8], dtype=np.float64)
    records = residual_bins(
        np.abs(target), prediction, target, (0.0, 0.01, 0.1, 1.0), scale=1.0
    )
    assert sum(record["sample_count"] for record in records) == target.size
    assert sum(
        record.get("global_residual_squared_energy_fraction", 0.0)
        for record in records
    ) == pytest.approx(1.0)


def test_time_residuals_use_exactly_states_zero_through_eighty():
    target = np.ones(162, dtype=np.float64)
    prediction = target + np.repeat(
        np.linspace(0.0, 0.8, 81, dtype=np.float64), 2
    )
    records = time_residual_metrics(prediction, target, samples_per_state=2)
    assert [record["step"] for record in records] == list(range(81))
    assert records[0]["residual_rmse"] == 0.0
    assert records[-1]["residual_rmse"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="states 0..80"):
        time_residual_metrics(
            prediction, target, samples_per_state=2, steps=range(1, 82)
        )


def test_switch_diagnostic_reuses_exact_deployed_A_algebra():
    parameters = {
        "g": 9.80616,
        "q0": 0.002,
        "H0": 750.0,
        "gamma_r": 0.001,
        "qprecip": 0.0001,
        "L": 10.0,
        "configured_dt": 100.0,
    }
    h = np.asarray([750.0, 750.0], dtype=np.float64)
    s = np.asarray([parameters["g"], parameters["g"]], dtype=np.float64)
    qv = np.asarray([0.003, 0.001], dtype=np.float64)
    qc = np.asarray([0.001, 0.001], dtype=np.float64)
    features = np.column_stack((h, h * s, h * qv, h * qc, np.zeros(2)))
    state = {
        "h": jnp.asarray(features[:, 0]),
        "S": jnp.asarray(features[:, 1]),
        "Qv": jnp.asarray(features[:, 2]),
        "Qc": jnp.asarray(features[:, 3]),
    }
    fields = {"B": jnp.asarray(features[:, 4])}
    jax_parameters = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in parameters.items()
    }
    target = np.asarray(
        moist_rates_jax(state, fields, jax_parameters)["A"], dtype=np.float64
    )
    diagnostics = deployed_a_switch_diagnostics(features, target, parameters)
    np.testing.assert_array_equal(diagnostics["A"], target)
    assert diagnostics["saturation_excess"][0] > 0.0
    assert diagnostics["A"][0] < 0.0
    assert diagnostics["saturation_excess"][1] < 0.0
    assert diagnostics["A"][1] > 0.0
