"""Cheap contracts for the Test-2 deployed-GLL moist activity audit."""

from __future__ import annotations

import numpy as np
import pytest

from dimswe.resolved_hidden_c0 import (
    ResolvedInferenceConfiguration,
    resolved_truth_state_indices,
)
from dimswe.test2_moist_activity import (
    _require_training_state_keys,
    classify_moist_activity,
    deployed_sample_accounting,
    moist_source_terms,
    plot_moist_activity,
    rate_statistics,
    time_rate_statistics,
)


def test_exact_selected_deployed_gll_sample_accounting():
    accounting = deployed_sample_accounting(
        states=81,
        cells=16 * 16,
        points_per_cell=4 * 4,
    )
    assert accounting["stored_states_examined"] == 81
    assert accounting["number_of_cells"] == 256
    assert accounting["gll_points_per_cell"] == 16
    assert accounting["samples_per_state"] == 4096
    assert accounting["total_space_time_samples"] == 331776
    assert accounting["shared_cg_boundary_points_repeated"]
    assert not accounting["deduplicated"]


def test_known_rate_statistics_signs_percentiles_and_activity():
    values = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float64)
    result = rate_statistics(values)
    assert result["minimum"] == -2.0
    assert result["maximum"] == 3.0
    assert result["mean"] == 0.5
    assert result["mean_absolute"] == 1.5
    assert result["rms"] == pytest.approx(np.sqrt(3.5))
    assert result["exact_zero_fraction"] == 0.25
    assert result["positive_fraction"] == 0.5
    assert result["negative_fraction"] == 0.25
    assert result["percentiles"]["0"] == -2.0
    assert result["percentiles"]["100"] == 3.0
    assert result["scale_relative_activity_fractions"]["1e-03"] == 0.75


def test_exact_zero_rate_and_time_activity_are_explicit():
    values = np.zeros((3, 4), dtype=np.float64)
    result = rate_statistics(values)
    assert result["maximum_absolute"] == 0.0
    assert result["exact_zero_fraction"] == 1.0
    assert result["positive_fraction"] == 0.0
    assert result["negative_fraction"] == 0.0
    assert set(result["scale_relative_activity_fractions"].values()) == {0.0}
    assert result["scale_relative_activity_contract"] == (
        "max_abs_rate_is_exactly_zero"
    )
    time = time_rate_statistics(values, 0.0)
    assert set(time["global_scale_relative_activity_fractions"].values()) == {
        0.0
    }


def test_structural_moist_source_identities_and_process_split():
    h = np.array([[2.0, 3.0]], dtype=np.float64)
    a_rate = np.array([[-1.0, 4.0]], dtype=np.float64)
    r_rate = np.array([[5.0, 6.0]], dtype=np.float64)
    beta2 = 7.0
    source = moist_source_terms(h, a_rate, r_rate, beta2)
    np.testing.assert_array_equal(source["Qv"], h * a_rate)
    np.testing.assert_array_equal(source["Qc"], -h * (a_rate + r_rate))
    np.testing.assert_array_equal(source["Qr"], h * r_rate)
    np.testing.assert_array_equal(source["S"], h * beta2 * a_rate)
    np.testing.assert_array_equal(
        source["Qc"], (-h * a_rate) + (-h * r_rate)
    )


def test_numpy_source_structure_matches_certified_j1_primal_kernel():
    jnp = pytest.importorskip("jax.numpy")
    from dimswe.jax_moist import moist_rates_and_source_density_jax

    state = {
        "h": jnp.array([[750.0, 800.0]], dtype=jnp.float64),
        "S": jnp.array([[7350.0, 7840.0]], dtype=jnp.float64),
        "Qv": jnp.array([[2.0, 1.0]], dtype=jnp.float64),
        "Qc": jnp.array([[0.5, 0.2]], dtype=jnp.float64),
    }
    fields = {"B": jnp.zeros((1, 2), dtype=jnp.float64)}
    parameters = {
        "g": jnp.array(9.8, dtype=jnp.float64),
        "q0": jnp.array(0.002, dtype=jnp.float64),
        "H0": jnp.array(750.0, dtype=jnp.float64),
        "gamma_r": jnp.array(1.0, dtype=jnp.float64),
        "qprecip": jnp.array(1.0e-4, dtype=jnp.float64),
        "L": jnp.array(10.0, dtype=jnp.float64),
        "configured_dt": jnp.array(100.0, dtype=jnp.float64),
    }
    result = moist_rates_and_source_density_jax(state, fields, parameters)
    rates = {name: np.asarray(value) for name, value in result["rates"].items()}
    expected = moist_source_terms(
        np.asarray(state["h"]),
        rates["A"],
        rates["R"],
        float(parameters["g"] * parameters["L"]),
    )
    for name, values in result["source"].items():
        np.testing.assert_array_equal(np.asarray(values), expected[name])


def test_training_loader_contract_never_requests_states_after_80():
    configuration = ResolvedInferenceConfiguration(
        truth_run_directory="selected_truth",
        training_start_step=0,
        training_stop_step=80,
        heldout_stop_step=160,
    )
    training = resolved_truth_state_indices(configuration, include_heldout=False)
    assert training == tuple(range(81))
    assert max(training) == 80
    _require_training_state_keys({step: object() for step in training})
    with pytest.raises(ValueError, match="exactly truth states 0..80"):
        _require_training_state_keys(
            {step: object() for step in range(82)}
        )


def _classification_inputs(a_values, r_values):
    rates = {"A": rate_statistics(a_values), "R": rate_statistics(r_values)}
    times = {
        "A": [time_rate_statistics(a_values, rates["A"]["maximum_absolute"])],
        "R": [time_rate_statistics(r_values, rates["R"]["maximum_absolute"])],
    }
    effects = {
        "Qv": {"global_rms_increment_over_truth_rms": {"value": 1.0e-3}},
        "Qr": {"global_rms_increment_over_truth_rms": {"value": 1.0e-3}},
    }
    return rates, times, effects


def test_decision_screen_identifies_two_rate_and_r_degenerate_cases():
    active = np.ones(10000, dtype=np.float64)
    rates, times, effects = _classification_inputs(active, 2.0 * active)
    assert classify_moist_activity(rates, times, effects)["classification"] == (
        "RICH_TWO_RATE_SIGNAL"
    )

    rates, times, effects = _classification_inputs(active, np.zeros_like(active))
    assert classify_moist_activity(rates, times, effects)["classification"] == (
        "A_ACTIVE_R_WEAK"
    )


def test_lightweight_activity_and_spatial_plots(tmp_path):
    pytest.importorskip("matplotlib")
    record = {
        "time": 0.0,
        "maximum_absolute": 2.0,
        "rms": 1.0,
        "global_scale_relative_activity_fractions": {
            "1e-06": 0.5,
            "1e-03": 0.25,
        },
    }
    summary = {
        "time_resolved_rate_activity": {
            "A": [record],
            "R": [record],
        },
        "deployed_representation": {
            "physical_coordinates_cell_major": {
                "x": [[0.0, 1.0]],
                "y": [[0.0, 1.0]],
            }
        },
        "representative_spatial_activity": {
            "A": {"time": 0.0, "rate_values_cell_major": [[1.0, -1.0]]},
            "R": {"time": 0.0, "rate_values_cell_major": [[0.0, 2.0]]},
        },
    }
    plot_moist_activity(summary, tmp_path)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "A_maximum_rms_vs_time.png",
        "R_maximum_rms_vs_time.png",
        "A_active_fraction_vs_time.png",
        "R_active_fraction_vs_time.png",
        "A_rms_selected_spatial_map.png",
        "R_rms_selected_spatial_map.png",
    }
