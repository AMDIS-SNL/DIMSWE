"""Cheap Test 2A-3B autonomous-contract and local-diagnostic tests."""

import jax.numpy as jnp
import numpy as np
import pytest

from dimswe.test2a_apriori_autonomous import (
    autonomous_states,
    load_apriori_autonomous_configuration,
    load_compatible_neural_physics,
    local_a_diagnostic,
    rain_activity_diagnostic,
    source_invariant_diagnostic,
)
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2a_operator import (
    initialize_mlp,
    load_selected_configuration,
    mlp_configuration_from_record,
    save_mlp_parameters,
)


CONFIG = "dimswe/configs/test2a_apriori_autonomous.json"
EMBEDDING = "dimswe/configs/test2a_embedded_neural_a.json"
SELECTED = "dimswe/configs/test2a_selected_operator.json"


def test_selected_contract_is_one_truth_start_and_80_recursive_steps():
    record = load_apriori_autonomous_configuration(CONFIG)
    assert record["truth"]["state_indices"] == [0, 80]
    assert record["truth"]["states_after_80_forbidden"] is True
    assert record["deployment"]["initial_truth_state"] == 0
    assert record["deployment"]["complete_steps"] == 80
    assert record["deployment"]["truth_resets_after_initialization"] == 0
    assert record["deployment"]["R"] == "original deployed analytical R"


def test_autonomous_constructor_recursively_reuses_prediction_without_truth():
    inputs = []

    def step(value, index):
        inputs.append((index, value))
        return 1.5 * value + index

    states = autonomous_states(2.0, step, nsteps=80)
    assert len(states) == 81
    assert inputs[0] == (0, 2.0)
    assert all(inputs[index][1] == states[index] for index in range(80))
    assert tuple(index for index, _ in inputs) == tuple(range(80))


def test_default_frozen_artifact_and_arbitrary_compatible_artifact(tmp_path):
    record = load_apriori_autonomous_configuration(CONFIG)
    frozen = load_compatible_neural_physics(
        EMBEDDING,
        record["model"]["default_parameter_file"],
        expected_pytree_sha256=record["model"]["default_parameter_pytree_sha256"],
        use_jit=False,
    )
    assert parameter_pytree_sha256(frozen.parameters) == record["model"][
        "default_parameter_pytree_sha256"
    ]
    selected = load_selected_configuration(SELECTED)
    configuration = mlp_configuration_from_record(selected["model"])
    arbitrary = initialize_mlp(configuration)
    path = tmp_path / "arbitrary.npz"
    save_mlp_parameters(path, arbitrary, configuration)
    provider = load_compatible_neural_physics(EMBEDDING, path, use_jit=False)
    assert parameter_pytree_sha256(provider.parameters) == parameter_pytree_sha256(
        arbitrary
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_compatible_neural_physics(
            EMBEDDING, path, expected_pytree_sha256="0" * 64, use_jit=False
        )


def test_off_manifold_a_metrics_and_active_sign_strata():
    analytical = np.array([-2.0, -0.2, 0.0, 0.4, 3.0], dtype=np.float64)
    neural = np.array([-1.9, 0.1, 0.0, 0.5, 2.7], dtype=np.float64)
    record = local_a_diagnostic(neural, analytical)
    assert record["relative_rms_error"] > 0.0
    assert record["active_strata"]["abs_A_gt_1e-01_max_abs_A"][
        "sign_agreement"
    ] == 1.0
    assert record["active_strata"]["abs_A_gt_1e-03_max_abs_A"][
        "sign_agreement"
    ] == pytest.approx(0.75)


def test_rain_activity_separates_exact_roundoff_and_physical_increment():
    rain = np.array([0.0, 1.0e-25, 2.0e-8], dtype=np.float64)
    h = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    qr = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    record = rain_activity_diagnostic(rain, h, qr, 100.0, 1.0e-7)
    assert record["exact_nonzero_fraction"] == pytest.approx(2.0 / 3.0)
    assert record["above_float64_scale_fraction"] == pytest.approx(1.0 / 3.0)
    assert record["physically_meaningful_fraction"] == pytest.approx(1.0 / 3.0)


def test_neural_source_invariants_are_structural():
    h = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    a = jnp.asarray([-0.2, 0.4], dtype=jnp.float64)
    r = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    beta2 = 7.0
    source = {
        "Qv": h * a,
        "Qc": -h * (a + r),
        "Qr": h * r,
        "S": h * beta2 * a,
    }
    record = source_invariant_diagnostic(source, beta2)
    assert record["water_source_maximum_absolute_residual"] < 1.0e-15
    assert record["S_minus_beta2_Qv_maximum_absolute_residual"] < 4.0e-15
