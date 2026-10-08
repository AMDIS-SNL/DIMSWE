import json

import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip("pyrol", reason="PyROL is required for M1-Y campaign tests")

from dimswe.jax_moist import moist_rates_jax
from dimswe.test2b_m1y_campaign import (
    EXPECTED_NORMALIZATION,
    EXPECTED_SEED_SHA,
    FEATURES,
    independent_numpy_rates,
    load_m1y_configuration,
    representation_target,
)
from dimswe.test2b_rain_learning import (
    RainLearningNormalization,
    RainMLPConfiguration,
    initial_parameters,
)
from dimswe.test2a_embedded_moist import parameter_pytree_sha256


def _normalization():
    return RainLearningNormalization(
        input_offset=EXPECTED_NORMALIZATION["input_offset"],
        input_scale=EXPECTED_NORMALIZATION["input_scale"],
        sigma_a=EXPECTED_NORMALIZATION["sigma_a"],
        sigma_r_active=EXPECTED_NORMALIZATION["sigma_r"],
        source_scales=EXPECTED_NORMALIZATION["source_scales"],
        provenance_sha256=EXPECTED_NORMALIZATION["provenance_sha256"],
    )


def test_campaign_configuration_freezes_state_features_and_optimizer():
    source, record = load_m1y_configuration(
        "dimswe/configs/test2b_m1y_20260828.json"
    )
    assert source.is_file()
    assert tuple(record["M1_Y"]["features"]) == FEATURES
    assert record["M1_Y"]["features_evaluated_at"] == "Y_n*"
    assert record["M1_Y"]["targets_evaluated_at"] == "Y_n*"
    assert record["M1_Y"]["training_states"] == [0, 80]
    assert record["normalization"]["refit_on_Y"] is False
    assert record["training"]["optimizer"]["accepted_iteration_limit"] == 10000


def test_independent_numpy_rates_reproduce_deployed_jax_law():
    physical = np.asarray(
        [
            [750.0, 7370.0, 1.45, 0.08, 0.0],
            [740.0, 7300.0, 1.20, 0.12, 0.0],
            [760.0, 7450.0, 1.70, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    parameters = {
        "g": jnp.float64(9.80616),
        "q0": jnp.float64(0.02),
        "H0": jnp.float64(750.0),
        "gamma_r": jnp.float64(0.1),
        "qprecip": jnp.float64(1.0e-4),
        "L": jnp.float64(10.0),
        "configured_dt": jnp.float64(100.0),
    }
    state = {
        name: jnp.asarray(physical[:, index])
        for index, name in enumerate(("h", "S", "Qv", "Qc"))
    }
    fields = {"B": jnp.asarray(physical[:, 4])}
    expected = moist_rates_jax(state, fields, parameters)
    actual = independent_numpy_rates(physical, parameters)
    np.testing.assert_allclose(actual["A"], expected["A"], rtol=4e-15, atol=0.0)
    np.testing.assert_allclose(actual["R"], expected["R"], rtol=4e-15, atol=0.0)


def test_representation_targets_preserve_production_order_and_exclude_qr():
    normalization = _normalization()
    physical = np.asarray(
        [[750.0, 7375.0, 1.4, 0.075, 0.0],
         [760.0, 7400.0, 1.5, 0.080, 0.0]],
        dtype=np.float64,
    )
    normalized = np.asarray(normalization.normalize_features(physical))
    a = np.asarray([2.0e-8, -3.0e-8])
    r = np.asarray([0.0, 4.0e-11])
    target_a = representation_target("A", normalized, a, r, normalization)
    target_b = representation_target("B", normalized, a, r, normalization)
    target_c = representation_target("C", normalized, a, r, normalization)
    np.testing.assert_array_equal(target_a, a[:, None])
    np.testing.assert_array_equal(target_b, np.stack((a, r), axis=-1))
    expected_c = np.stack(
        (
            physical[:, 0] * 98.0616 * a,
            physical[:, 0] * a,
            -physical[:, 0] * (a + r),
            physical[:, 0] * r,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(target_c, expected_c, rtol=0.0, atol=1e-18)
    assert FEATURES == ("h", "S", "Qv", "Qc", "B")
    assert "Qr" not in FEATURES


def test_frozen_architectures_and_seed_zero_initializations():
    expected_layers = {
        "A": (5, 32, 32, 1),
        "B": (5, 32, 32, 2),
        "C": (5, 32, 32, 4),
    }
    expected_counts = {"A": 1281, "B": 1314, "C": 1380}
    for representation in "ABC":
        configuration = RainMLPConfiguration(representation)
        assert configuration.layer_dimensions == expected_layers[representation]
        assert configuration.parameter_count == expected_counts[representation]
        assert parameter_pytree_sha256(initial_parameters(representation)) == EXPECTED_SEED_SHA[representation]


def test_configuration_is_machine_readable_json():
    with open("dimswe/configs/test2b_m1y_20260828.json", encoding="utf-8") as stream:
        record = json.load(stream)
    assert record["campaign_id"] == "m1y_test2b_20260828"
