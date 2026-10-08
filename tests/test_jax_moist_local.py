"""Pure-JAX specifications for the exact deployed moist algebra."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


jax = pytest.importorskip("jax", reason="JAX is optional for DIMSWE")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy", reason="JAX is optional for DIMSWE")

from dimswe.jax_moist import (  # noqa: E402
    moist_diagnostics_jax,
    moist_diagnostics_jit,
    moist_rates_jax,
    moist_rates_jit,
    moist_source_density_jax,
    moist_source_density_jit,
)


EPS = np.finfo(np.float64).eps
ROOT = Path(__file__).resolve().parents[1]


def _parameters(**updates):
    result = {
        "g": np.asarray(9.80616, dtype=np.float64),
        "q0": np.asarray(0.002, dtype=np.float64),
        "H0": np.asarray(750.0, dtype=np.float64),
        "gamma_r": np.asarray(0.001, dtype=np.float64),
        "qprecip": np.asarray(0.0001, dtype=np.float64),
        "L": np.asarray(10.0, dtype=np.float64),
        "configured_dt": np.asarray(400.0, dtype=np.float64),
    }
    for key, value in updates.items():
        result[key] = np.asarray(value, dtype=np.float64)
    return result


def _state(qv=0.003, qc=0.001, *, h=750.0, s=None, shape=()):
    h_array = np.full(shape, h, dtype=np.float64)
    gravity = 9.80616
    specific_entropy = gravity if s is None else s
    return {
        "h": h_array,
        "S": h_array * np.asarray(specific_entropy, dtype=np.float64),
        "Qv": h_array * np.asarray(qv, dtype=np.float64),
        "Qc": h_array * np.asarray(qc, dtype=np.float64),
    }


def _fields(B=0.0, shape=()):
    return {"B": np.full(shape, B, dtype=np.float64)}


def _reference(state, fields, parameters):
    """Independent NumPy transcription of ``ThreeWayPhysics.rhs``."""
    h = np.asarray(state["h"], dtype=np.float64)
    S = np.asarray(state["S"], dtype=np.float64)
    Qv = np.asarray(state["Qv"], dtype=np.float64)
    Qc = np.asarray(state["Qc"], dtype=np.float64)
    B = np.asarray(fields["B"], dtype=np.float64)
    p = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in parameters.items()
    }

    with np.errstate(all="ignore"):
        qv = Qv / h
        qc = Qc / h
        s = S / h
        beta2 = p["g"] * p["L"]
        depth_denominator = h + B
        qsat = (
            p["q0"]
            * p["H0"]
            / depth_denominator
            * np.exp(
                np.float64(20.0)
                * (np.float64(1.0) - s / p["g"])
            )
        )
        gamma_v_denominator = (
            np.float64(1.0)
            + np.float64(20.0) * qsat * beta2 / p["g"]
        )
        gamma_v = np.float64(1.0) / gamma_v_denominator
        c_argument = gamma_v * (qv - qsat) / p["configured_dt"]
        C = np.where(c_argument < np.float64(0.0), 0.0, c_argument)
        e_argument = gamma_v * (qsat - qv) / p["configured_dt"]
        E_positive = np.where(
            e_argument < np.float64(0.0), 0.0, e_argument
        )
        evaporation_cap = qc / p["configured_dt"]
        E = np.where(
            evaporation_cap < E_positive,
            evaporation_cap,
            E_positive,
        )
        r_argument = (
            p["gamma_r"]
            * (qc - p["qprecip"])
            / p["configured_dt"]
        )
        R = np.where(r_argument < np.float64(0.0), 0.0, r_argument)
        A = E - C

    return {
        "qv": qv,
        "qc": qc,
        "s": s,
        "beta2": beta2,
        "depth_denominator": depth_denominator,
        "qsat": qsat,
        "gamma_v_denominator": gamma_v_denominator,
        "gamma_v": gamma_v,
        "condensation_argument": c_argument,
        "evaporation_argument": e_argument,
        "evaporation_cap": evaporation_cap,
        "evaporation_cap_difference": evaporation_cap - E_positive,
        "rain_argument": r_argument,
        "C": C,
        "E_positive": E_positive,
        "E": E,
        "A": A,
        "R": R,
    }


def _numpy_tree(value):
    return jax.tree.map(lambda leaf: np.asarray(leaf), value)


def _classification(value):
    array = np.asarray(value)
    return {
        "finite": np.isfinite(array),
        "positive_inf": np.isposinf(array),
        "negative_inf": np.isneginf(array),
        "nan": np.isnan(array),
    }


def _assert_same_classification(actual, expected):
    actual_classes = _classification(actual)
    expected_classes = _classification(expected)
    for key in actual_classes:
        np.testing.assert_array_equal(
            actual_classes[key], expected_classes[key], err_msg=key
        )


def _assert_float_results_match(actual, expected):
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    _assert_same_classification(actual_array, expected_array)
    finite = np.isfinite(expected_array)
    if np.any(finite):
        np.testing.assert_allclose(
            actual_array[finite],
            expected_array[finite],
            rtol=32.0 * EPS,
            atol=32.0 * EPS * max(
                1.0, float(np.max(np.abs(expected_array[finite])))
            ),
        )


@pytest.mark.parametrize(
    ("name", "state", "parameters", "expected_masks"),
    (
        (
            "condensation_rain_active",
            _state(qv=0.003, qc=0.001),
            _parameters(),
            (True, False, True, True),
        ),
        (
            "condensation_rain_inactive",
            _state(qv=0.003, qc=0.00005),
            _parameters(),
            (True, False, True, False),
        ),
        (
            "evaporation_capped_rain_active",
            _state(qv=0.001, qc=0.0002),
            _parameters(),
            (False, True, False, True),
        ),
        (
            "evaporation_capped_rain_inactive",
            _state(qv=0.001, qc=0.00005),
            _parameters(),
            (False, True, False, False),
        ),
        (
            "evaporation_uncapped_rain_active",
            _state(qv=0.001, qc=0.001),
            _parameters(),
            (False, True, True, True),
        ),
        (
            "evaporation_uncapped_rain_inactive",
            _state(qv=0.001, qc=0.001),
            _parameters(qprecip=0.002),
            (False, True, True, False),
        ),
        (
            "finite_negative_cloud",
            _state(qv=0.003, qc=-0.0002),
            _parameters(),
            (True, False, False, False),
        ),
    ),
)
def test_feasible_active_sets_match_independent_reference(
    name, state, parameters, expected_masks
):
    fields = _fields()
    reference = _reference(state, fields, parameters)
    rates = _numpy_tree(moist_rates_jax(state, fields, parameters))
    diagnostics = _numpy_tree(
        moist_diagnostics_jax(state, fields, parameters)
    )

    for key in (
        "A",
        "R",
        "C",
        "E_positive",
        "E",
        "qsat",
        "gamma_v",
        "condensation_argument",
        "evaporation_argument",
        "evaporation_cap_difference",
        "rain_argument",
    ):
        actual = rates[key] if key in rates else diagnostics[key]
        _assert_float_results_match(actual, reference[key])

    masks = tuple(
        bool(np.asarray(diagnostics[key]))
        for key in (
            "condensation_mask",
            "evaporation_mask",
            "uncapped_evaporation_mask",
            "rain_mask",
        )
    )
    assert masks == expected_masks, name


def test_all_deployed_tie_values_select_expected_primal_values():
    parameters = _parameters()
    fields = _fields()

    saturation_state = _state(qv=0.002, qc=0.001)
    saturation = _numpy_tree(
        moist_diagnostics_jax(saturation_state, fields, parameters)
    )
    assert saturation["condensation_argument"] == 0.0
    assert saturation["evaporation_argument"] == 0.0
    assert saturation["C"] == 0.0
    assert saturation["E_positive"] == 0.0
    assert not bool(saturation["condensation_mask"])
    assert not bool(saturation["evaporation_mask"])

    evaporation_state = _state(qv=0.001, qc=0.0)
    first = _reference(evaporation_state, fields, parameters)
    cap_tie_qc = float(
        parameters["configured_dt"] * first["E_positive"]
    )
    cap_tie_state = _state(qv=0.001, qc=cap_tie_qc)
    cap_tie = _numpy_tree(
        moist_diagnostics_jax(cap_tie_state, fields, parameters)
    )
    assert abs(float(cap_tie["evaporation_cap_difference"])) <= 8.0 * EPS
    _assert_float_results_match(cap_tie["E"], cap_tie["E_positive"])

    rain_tie = _numpy_tree(
        moist_diagnostics_jax(
            _state(qv=0.003, qc=0.0001, h=1.0), fields, parameters
        )
    )
    assert rain_tie["rain_argument"] == 0.0
    assert rain_tie["R"] == 0.0
    assert not bool(rain_tie["rain_mask"])


@pytest.mark.parametrize(
    ("name", "state", "fields", "parameters"),
    (
        (
            "singular_h",
            _state(qv=0.003, qc=0.001, h=0.0),
            _fields(),
            _parameters(),
        ),
        (
            "singular_h_plus_B",
            _state(qv=0.003, qc=0.001),
            _fields(B=-750.0),
            _parameters(),
        ),
        (
            "singular_gamma_v_denominator",
            {
                "h": np.asarray(1.0, dtype=np.float64),
                "S": np.asarray(1.0, dtype=np.float64),
                "Qv": np.asarray(0.003, dtype=np.float64),
                "Qc": np.asarray(0.001, dtype=np.float64),
            },
            _fields(),
            _parameters(g=1.0, q0=0.05, H0=1.0, L=-1.0),
        ),
        (
            "exponential_overflow",
            _state(qv=0.003, qc=0.001, s=-100.0 * 9.80616),
            _fields(),
            _parameters(),
        ),
    ),
)
def test_nonfinite_behavior_matches_independent_reference(
    name, state, fields, parameters
):
    reference = _reference(state, fields, parameters)
    actual = _numpy_tree(moist_diagnostics_jax(state, fields, parameters))
    for key in (
        "qv",
        "qc",
        "s",
        "qsat",
        "gamma_v_denominator",
        "gamma_v",
        "condensation_argument",
        "evaporation_argument",
        "rain_argument",
        "A",
        "R",
    ):
        _assert_float_results_match(actual[key], reference[key])


@pytest.mark.parametrize("shape", ((), (7,), (3, 16), (2, 3, 4)))
def test_scalar_and_arbitrary_batch_shapes_and_jit(shape):
    indices = np.arange(np.prod(shape) if shape else 1, dtype=np.float64)
    perturbation = (indices.reshape(shape) if shape else indices[0]) * 1.0e-7
    state = _state(qv=0.0015 + perturbation, qc=0.0004, shape=shape)
    fields = _fields(shape=shape)
    parameters = _parameters()

    plain_rates = _numpy_tree(moist_rates_jax(state, fields, parameters))
    jitted_rates = _numpy_tree(moist_rates_jit(state, fields, parameters))
    plain_sources = _numpy_tree(
        moist_source_density_jax(state, fields, parameters)
    )
    jitted_sources = _numpy_tree(
        moist_source_density_jit(state, fields, parameters)
    )
    plain_diagnostics = _numpy_tree(
        moist_diagnostics_jax(state, fields, parameters)
    )
    jitted_diagnostics = _numpy_tree(
        moist_diagnostics_jit(state, fields, parameters)
    )

    for tree in (plain_rates, plain_sources):
        assert all(value.shape == shape for value in tree.values())
        assert all(value.dtype == np.float64 for value in tree.values())
    for left, right in (
        (plain_rates, jitted_rates),
        (plain_sources, jitted_sources),
        (plain_diagnostics, jitted_diagnostics),
    ):
        for key in left:
            if left[key].dtype == np.bool_:
                np.testing.assert_array_equal(left[key], right[key])
            else:
                _assert_float_results_match(left[key], right[key])


def test_source_density_invariants_are_algebraic():
    state = _state(qv=0.0017, qc=0.0004, shape=(4, 16))
    state["Qv"] *= np.linspace(0.8, 1.2, 64).reshape(4, 16)
    state["Qc"] *= np.linspace(1.1, 0.9, 64).reshape(4, 16)
    fields = _fields(shape=(4, 16))
    parameters = _parameters()
    source = _numpy_tree(
        moist_source_density_jax(state, fields, parameters)
    )
    beta2 = float(parameters["g"] * parameters["L"])

    total_water = source["Qv"] + source["Qc"] + source["Qr"]
    thermal_vapour = source["S"] - beta2 * source["Qv"]
    water_scale = max(
        1.0e-300,
        *(float(np.max(np.abs(source[key]))) for key in ("Qv", "Qc", "Qr")),
    )
    thermal_scale = max(
        1.0e-300,
        float(np.max(np.abs(source["S"]))),
        beta2 * float(np.max(np.abs(source["Qv"]))),
    )
    assert np.max(np.abs(total_water)) <= 8.0 * EPS * water_scale
    assert np.max(np.abs(thermal_vapour)) <= 8.0 * EPS * thermal_scale


def test_inputs_are_not_mutated_and_outputs_do_not_alias_inputs():
    state = _state(qv=0.001, qc=0.0002, shape=(2, 16))
    fields = _fields(shape=(2, 16))
    parameters = _parameters()
    snapshots = {
        "state": {key: value.copy() for key, value in state.items()},
        "fields": {key: value.copy() for key, value in fields.items()},
        "parameters": {key: value.copy() for key, value in parameters.items()},
    }

    rates = moist_rates_jax(state, fields, parameters)
    sources = moist_source_density_jax(state, fields, parameters)
    moist_diagnostics_jax(state, fields, parameters)

    for key in state:
        np.testing.assert_array_equal(state[key], snapshots["state"][key])
    for key in fields:
        np.testing.assert_array_equal(fields[key], snapshots["fields"][key])
    for key in parameters:
        np.testing.assert_array_equal(
            parameters[key], snapshots["parameters"][key]
        )
    for output in (*rates.values(), *sources.values()):
        assert not np.shares_memory(np.asarray(output), state["h"])


def test_float32_inputs_are_rejected_instead_of_silently_promoted():
    state = _state()
    state["h"] = np.asarray(state["h"], dtype=np.float32)
    with pytest.raises(TypeError, match=r"state_q\['h'\].*float64.*float32"):
        moist_rates_jax(state, _fields(), _parameters())


def test_x64_disabled_import_has_precise_configuration_error():
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-c", "import dimswe.jax_moist"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode != 0
    assert (
        "DIMSWE JAX moist physics requires jax_enable_x64=True"
        in process.stderr
    )
