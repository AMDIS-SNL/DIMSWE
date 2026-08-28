import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimswe.test2a_fiml import (
    DENOMINATOR,
    FieldControlAMoistPhysics,
    _lcurve_selection,
    control_sha256,
    load_fiml_configuration,
    sparse_observation_indices,
    sparse_starts,
    sparse_windows,
)


CONFIGURATION = "dimswe/configs/test2a_fiml_sparse_endpoint_h2_h5.json"


class _Baseline:
    model_configuration = object()
    normalization = object()
    provenance = {"test": True}
    parameters = {"p": jnp.array(0.0, dtype=jnp.float64)}

    def combined_with_parameters(self, state, fields, moist, parameters):
        del fields, parameters
        a = 0.25 * state["Qv"] - 0.1 * state["Qc"]
        r = 0.07 * jnp.maximum(state["Qc"] - 0.3, 0.0)
        h = state["h"]
        beta2 = moist["g"] * moist["L"]
        return {
            "rates": {"A": a, "R": r},
            "source": {
                "S": h * beta2 * a,
                "Qv": h * a,
                "Qc": -h * (a + r),
                "Qr": h * r,
            },
        }


def _local_values():
    shape = (2, 3)
    state = {
        "h": jnp.full(shape, 2.0, dtype=jnp.float64),
        "S": jnp.full(shape, 1.0, dtype=jnp.float64),
        "Qv": jnp.linspace(0.1, 0.6, 6, dtype=jnp.float64).reshape(shape),
        "Qc": jnp.linspace(0.2, 0.7, 6, dtype=jnp.float64).reshape(shape),
    }
    fields = {"B": jnp.zeros(shape, dtype=jnp.float64)}
    moist = {
        "g": jnp.array(9.8, dtype=jnp.float64),
        "q0": jnp.array(1.0, dtype=jnp.float64),
        "H0": jnp.array(1.0, dtype=jnp.float64),
        "gamma_r": jnp.array(1.0, dtype=jnp.float64),
        "qprecip": jnp.array(1.0, dtype=jnp.float64),
        "L": jnp.array(2.5, dtype=jnp.float64),
        "configured_dt": jnp.array(100.0, dtype=jnp.float64),
    }
    return state, fields, moist


def test_sparse_schedules_use_only_origins_and_endpoints():
    assert sparse_starts(2) == tuple(range(0, 80, 2))
    assert sparse_starts(5) == tuple(range(0, 80, 5))
    assert sparse_observation_indices(2) == tuple(range(0, 81, 2))
    assert sparse_observation_indices(5) == tuple(range(0, 81, 5))
    for horizon in (2, 5):
        windows = sparse_windows(horizon)
        assert len(windows) == 80 // horizon
        assert all(window.loss_mode == "endpoint" for window in windows)
        assert all(window.weights == (1.0,) for window in windows)
        assert max(sparse_observation_indices(horizon)) == 80


def test_selected_configuration_freezes_information_contract():
    record = load_fiml_configuration(CONFIGURATION)
    assert record["objective"]["common_denominator_D"] == DENOMINATOR
    assert record["field_inversion"]["truth_A_regularization"] is False
    assert record["field_inversion"]["spatial_smoothness"] is False
    assert record["field_inversion"]["temporal_smoothness"] is False
    assert record["optimizer"]["new_optimizer_and_empty_secant_history"] is True
    assert record["truth"]["allowed_state_indices"] == [0, 80]


def test_field_control_zero_parity_and_source_structure():
    state, fields, moist = _local_values()
    provider = FieldControlAMoistPhysics(_Baseline(), 3.0e-9, use_jit=False)
    zeros = jnp.zeros_like(state["h"])
    baseline = _Baseline().combined_with_parameters(
        state, fields, moist, _Baseline.parameters
    )
    zero = provider._combined_explicit(state, fields, moist, zeros)
    np.testing.assert_array_equal(zero["rates"]["A"], baseline["rates"]["A"])
    np.testing.assert_array_equal(zero["rates"]["R"], baseline["rates"]["R"])

    controls = jnp.linspace(-2.0, 1.0, 6, dtype=jnp.float64).reshape(2, 3)
    active = provider._combined_explicit(state, fields, moist, controls)
    expected_a = baseline["rates"]["A"] + 3.0e-9 * controls
    np.testing.assert_allclose(active["rates"]["A"], expected_a, rtol=0, atol=0)
    np.testing.assert_array_equal(active["rates"]["R"], baseline["rates"]["R"])
    source = active["source"]
    np.testing.assert_allclose(source["Qv"] + source["Qc"] + source["Qr"], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        source["S"] - moist["g"] * moist["L"] * source["Qv"], 0.0, atol=1e-15
    )


def test_field_control_parameter_jvp_vjp_transpose_identity():
    state, fields, moist = _local_values()
    provider = FieldControlAMoistPhysics(_Baseline(), 3.0e-9, use_jit=False)
    controls = jnp.zeros_like(state["h"])
    direction = jnp.linspace(-0.4, 0.6, 6, dtype=jnp.float64).reshape(2, 3)
    _, tangent = provider.parameter_jvp(
        state, direction, fields, moist, base_parameters=controls
    )
    covector = {
        key: jnp.linspace(0.2, 0.8, 6, dtype=jnp.float64).reshape(2, 3)
        for key in ("S", "Qv", "Qc", "Qr")
    }
    adjoint = provider.parameter_vjp(
        state, covector, fields, moist, base_parameters=controls
    )
    left = sum(float(jnp.vdot(tangent[key], covector[key])) for key in tangent)
    right = float(jnp.vdot(direction, adjoint))
    assert left == pytest.approx(right, rel=2e-15, abs=1e-20)


def test_control_fingerprint_is_shape_and_value_sensitive():
    a = np.zeros((2, 3, 4), dtype=np.float64)
    b = a.copy()
    b[0, 0, 0] = np.nextafter(0.0, 1.0)
    assert control_sha256(a) != control_sha256(b)
    assert control_sha256(a) != control_sha256(a.reshape(2, 12))


def test_lcurve_selection_uses_interior_nontruth_point():
    candidates = [0.0, 1e-4, 1e-2, 1.0, 100.0]
    data = {1e-4: 1.0e-4, 1e-2: 2.0e-4, 1.0: 2.0e-2, 100.0: 0.8}
    control = {1e-4: 1.0, 1e-2: 0.8, 1.0: 0.08, 100.0: 1.0e-3}
    records = [
        {
            "lambda": value,
            "final_data_misfit": data.get(value, 9.0e-5),
            "final_control_rms": control.get(value, 1.2),
        }
        for value in candidates
        for _ in range(2)
    ]
    selected = _lcurve_selection(records, candidates)
    assert selected["selected_lambda"] in (1e-2, 1.0)
    assert "truth" not in selected["selection_rule"].lower()
