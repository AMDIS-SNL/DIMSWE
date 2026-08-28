import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimswe.test2b_rain_learning import (
    PARAMETER_COUNTS,
    RainActiveNeuralMoistPhysics,
    RainLearningNormalization,
    RainMLPConfiguration,
    initial_parameters,
    source_invariant_diagnostics,
    structural_diagnostics,
)
from dimswe.test2b_rain_learning_prepare import proposed_sustained_interval


jax.config.update("jax_enable_x64", True)


def normalization():
    return RainLearningNormalization(
        input_offset=np.zeros(5),
        input_scale=np.ones(5),
        sigma_a=2.0e-8,
        sigma_r_active=5.0e-12,
        source_scales=np.asarray((2.0e-3, 2.0e-5, 2.0e-5, 4.0e-9)),
        provenance_sha256="0" * 64,
    )


def local_inputs():
    shape = (2, 3)
    state = {
        "h": jnp.full(shape, 700.0),
        "S": jnp.full(shape, 700.0 * 9.7),
        "Qv": jnp.full(shape, 1.4),
        "Qc": jnp.full(shape, 0.08),
    }
    fields = {"B": jnp.zeros(shape)}
    moist = {
        "g": jnp.asarray(9.80616), "q0": jnp.asarray(0.002),
        "H0": jnp.asarray(750.0), "gamma_r": jnp.asarray(0.001),
        "qprecip": jnp.asarray(1.0e-4), "L": jnp.asarray(10.0),
        "configured_dt": jnp.asarray(100.0),
    }
    return state, fields, moist


@pytest.mark.parametrize("representation", ("A", "B", "C"))
def test_architecture_and_local_parameter_duality(representation):
    config = RainMLPConfiguration(representation)
    assert config.parameter_count == PARAMETER_COUNTS[representation]
    parameters = initial_parameters(representation)
    provider = RainActiveNeuralMoistPhysics(
        representation, parameters, normalization(), use_jit=False
    )
    state, fields, moist = local_inputs()
    direction = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1.0e-4), parameters)
    source, tangent = provider.parameter_jvp(state, direction, fields, moist)
    covector = {name: jnp.full_like(value, 0.2 + index) for index, (name, value) in enumerate(source.items())}
    adjoint = provider.parameter_vjp(state, covector, fields, moist)
    left = sum(jnp.vdot(tangent[name], covector[name]) for name in tangent)
    right = sum(jnp.vdot(direction["layers"][i][leaf], adjoint["layers"][i][leaf]) for i in range(3) for leaf in ("weight", "bias"))
    assert float(abs(left - right)) <= 5.0e-13 * max(abs(float(left)), abs(float(right)), 1.0)


def test_two_rate_structure_is_exact_and_four_output_is_not_projected():
    state, fields, moist = local_inputs()
    provider_b = RainActiveNeuralMoistPhysics("B", initial_parameters("B"), normalization(), use_jit=False)
    source_b = provider_b.combined_kernel(state, fields, moist)["source"]
    audit_b = source_invariant_diagnostics(source_b, 9.80616 * 10.0)
    assert audit_b["water_maximum_absolute"] <= 1.0e-18
    assert audit_b["S_minus_beta2_Qv_maximum_absolute"] <= 1.0e-16

    provider_c = RainActiveNeuralMoistPhysics("C", initial_parameters("C"), normalization(), use_jit=False)
    source_c = provider_c.combined_kernel(state, fields, moist)["source"]
    audit_c = source_invariant_diagnostics(source_c, 9.80616 * 10.0)
    assert audit_c["water_maximum_absolute"] > 0.0
    assert audit_c["S_minus_beta2_Qv_maximum_absolute"] > 0.0


def test_analytical_r_is_retained_only_by_representation_a():
    state, fields, moist = local_inputs()
    outputs = {
        name: RainActiveNeuralMoistPhysics(name, initial_parameters(name), normalization(), use_jit=False).combined_kernel(state, fields, moist)
        for name in ("A", "B", "C")
    }
    assert set(outputs["A"]["rates"]) == {"A", "R"}
    assert set(outputs["B"]["rates"]) == {"A", "R"}
    assert outputs["C"]["rates"] == {}
    assert np.max(np.asarray(outputs["A"]["rates"]["R"])) > 0.0


def test_sustained_contract_is_unchanged():
    records = []
    for step in range(12):
        records.append({
            "step": step, "time": 100.0 * step,
            "physically_meaningful_R_fraction": 2.0e-4,
            "rain_source_mass_rate": 1.0,
            "rain_water_mass": float(step), "total_water_mass": 1.0e12,
        })
    result = proposed_sustained_interval(records)
    assert result["start_step"] == 0
    assert result["certification_step"] == 10


def test_two_rate_manifold_projection_is_diagnostic_only():
    beta2 = 98.0616
    a = np.asarray((1.0e-6, -2.0e-6))
    r = np.asarray((3.0e-10, 0.0))
    truth = np.stack((beta2 * a, a, -a - r, r), axis=-1)
    record = structural_diagnostics(
        truth, truth, beta2, np.ones(4), np.ones(2)
    )
    assert record["normalized_two_rate_manifold_residual_rms"] < 1.0e-20
    assert record["normalized_source_vector_cosine"] == pytest.approx(1.0)
    perturbed = truth.copy()
    perturbed[:, 2] += 1.0e-7
    changed = structural_diagnostics(
        perturbed, truth, beta2, np.ones(4), np.ones(2)
    )
    assert changed["water_source_defect_rms"] > 0.0
    assert changed["normalized_two_rate_manifold_residual_rms"] > 0.0
