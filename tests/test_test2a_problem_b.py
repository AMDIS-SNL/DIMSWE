import json
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from dimswe.learned_physics.parameters import tree_dot, tree_norm
from dimswe.resolved_hidden_c0 import ResolvedPilotConfiguration, write_json_record
from dimswe.test2a_problem_b import (
    FourTendencyNormalization,
    NeuralFourTendencyMoistPhysics,
    ProblemBMLPConfiguration,
    ProblemBOperatorDataset,
    ProblemBOperatorObjective,
    initial_problem_b_parameters,
    load_problem_b_parameters,
    parameter_pytree_sha256,
    save_problem_b_parameters,
    structural_diagnostics,
)
from dimswe.test2a_problem_b_campaign import (
    ProblemBDiagnosticConfiguration,
    RESOLVED_DIAGNOSTIC_CONFIGURATION_ATTRIBUTES,
    _verify_completed_training_artifact,
)


SEED_SHA = "e52dd73e3f97d44adf4d55354b1c8d9a9b252186a17cae4ad09410270b86df1e"


def normalization():
    return FourTendencyNormalization(
        input_offset=np.zeros(5),
        input_scale=np.ones(5),
        sigma_s=2.0,
        sigma_q=3.0,
        input_normalization_sha256="a" * 64,
        scale_provenance_sha256="b" * 64,
    )


def local_inputs(count=7):
    values = jnp.linspace(0.1, 0.8, count, dtype=jnp.float64)
    state = {
        "h": values + 1.0,
        "S": values + 2.0,
        "Qv": values + 3.0,
        "Qc": values + 4.0,
    }
    fields = {"B": values - 0.2}
    parameters = {
        name: jnp.float64(1.0)
        for name in ("g", "q0", "H0", "gamma_r", "qprecip", "L", "configured_dt")
    }
    return state, fields, parameters


def test_architecture_and_seed_fingerprint_are_frozen():
    configuration = ProblemBMLPConfiguration()
    parameters = initial_problem_b_parameters()
    assert configuration.layer_dimensions == (5, 32, 32, 4)
    assert configuration.parameter_count == 1380
    assert sum(x.size for x in jax.tree_util.tree_leaves(parameters)) == 1380
    assert all(x.dtype == jnp.float64 for x in jax.tree_util.tree_leaves(parameters))
    assert parameter_pytree_sha256(parameters) == SEED_SHA


def test_parameter_artifact_round_trip(tmp_path):
    path = tmp_path / "parameters.npz"
    parameters = initial_problem_b_parameters()
    record = save_problem_b_parameters(path, parameters, metadata={"smoke": True})
    loaded, configuration, sidecar = load_problem_b_parameters(path)
    assert configuration.parameter_count == 1380
    assert record == sidecar
    assert parameter_pytree_sha256(loaded) == SEED_SHA


def test_provider_outputs_four_independent_unprojected_sources():
    parameters = initial_problem_b_parameters()
    layers = list(parameters["layers"])
    last = dict(layers[-1])
    last["weight"] = jnp.zeros_like(last["weight"])
    last["bias"] = jnp.asarray((1.0, 2.0, 3.0, 4.0), dtype=jnp.float64)
    layers[-1] = last
    parameters = {"layers": tuple(layers)}
    physics = NeuralFourTendencyMoistPhysics(parameters, normalization(), use_jit=False)
    state, fields, moist = local_inputs()
    source = physics.combined_kernel(state, fields, moist)["source"]
    np.testing.assert_allclose(source["S"], 2.0)
    np.testing.assert_allclose(source["Qv"], 6.0)
    np.testing.assert_allclose(source["Qc"], 9.0)
    np.testing.assert_allclose(source["Qr"], 12.0)
    assert not np.allclose(source["Qv"] + source["Qc"] + source["Qr"], 0.0)
    assert physics.combined_kernel(state, fields, moist)["rates"] == {}


def test_local_parameter_tangent_adjoint_duality():
    parameters = initial_problem_b_parameters()
    physics = NeuralFourTendencyMoistPhysics(parameters, normalization(), use_jit=True)
    state, fields, moist = local_inputs()
    direction = jax.tree.map(lambda x: jnp.ones_like(x) / tree_norm(parameters), parameters)
    source, tangent = physics.parameter_jvp(state, direction, fields, moist)
    covector = {name: jnp.linspace(-0.3, 0.5, 7) for name in source}
    adjoint = physics.parameter_vjp(state, covector, fields, moist)
    left = sum(float(jnp.vdot(tangent[name], covector[name])) for name in tangent)
    right = float(tree_dot(direction, adjoint))
    assert left == pytest.approx(right, rel=2e-14, abs=2e-14)


def test_operator_gradient_matches_centered_directional_difference():
    rng = np.random.default_rng(20260810)
    count = 96
    norm = normalization()
    dataset = ProblemBOperatorDataset(
        normalized_features=rng.normal(size=(count, 5)),
        physical_targets=rng.normal(size=(count, 4)),
        spatial_weights=rng.uniform(0.2, 1.0, size=count),
        normalization=norm,
        metadata={"truth_state_indices": [0, 80], "states_after_80_accessed": False},
    )
    objective = ProblemBOperatorObjective(dataset, use_jit=True)
    parameters = initial_problem_b_parameters()
    direction = jax.tree.map(lambda x: jnp.ones_like(x) / tree_norm(parameters), parameters)
    _, gradient = objective.value_and_gradient(parameters)
    epsilon = 2e-6
    plus = jax.tree.map(lambda x, d: x + epsilon * d, parameters, direction)
    minus = jax.tree.map(lambda x, d: x - epsilon * d, parameters, direction)
    centered = (objective.value(plus) - objective.value(minus)) / (2 * epsilon)
    assert centered == pytest.approx(float(tree_dot(gradient, direction)), rel=2e-8)


def test_structural_quantities_are_diagnostics_not_constraints():
    prediction = np.asarray([[2.0, 1.0, -0.25, 0.5], [1.0, -2.0, 3.0, 4.0]])
    truth = np.asarray([[2.0, 1.0, -1.0, 0.0], [-4.0, -2.0, 2.0, 0.0]])
    record = structural_diagnostics(
        prediction, truth, 2.0, normalization(), np.ones(2)
    )
    assert record["water_source_defect_rms"] > 0.0
    assert record["beta_source_defect_rms"] > 0.0
    assert record["spurious_Qr_t_rms"] > 0.0
    assert record["normalized_manifold_residual_rms"] > 0.0


def test_frozen_configuration_uses_only_states_zero_through_eighty():
    record = json.loads(
        open("dimswe/configs/test2a_problem_b.json", encoding="utf-8").read()
    )
    assert record["truth"]["allowed_state_indices"] == [0, 80]
    assert record["truth"]["states_after_80_forbidden"] is True
    assert record["model"]["features"] == ["h", "S", "Qv", "Qc", "B"]
    assert record["model"]["source_structure_enforced"] is False


def test_problem_b_diagnostic_configuration_supplies_complete_evaluator_contract():
    resolved = ResolvedPilotConfiguration(nx=16, ny=16)
    diagnostic = ProblemBDiagnosticConfiguration.from_resolved_pilot(resolved)
    assert diagnostic.sampling_shape == resolved.sampling_shape == (32, 32)
    assert diagnostic.high_wavenumber_fraction == pytest.approx(2.0 / 3.0)
    assert all(
        hasattr(diagnostic, name)
        for name in RESOLVED_DIAGNOSTIC_CONFIGURATION_ATTRIBUTES
    )
    evaluator_source = Path("dimswe/resolved_hidden_c0_driver.py").read_text(
        encoding="utf-8"
    )
    accessed = set(re.findall(r"self\.configuration\.([A-Za-z_]\w*)", evaluator_source))
    assert accessed == set(RESOLVED_DIAGNOSTIC_CONFIGURATION_ATTRIBUTES)


def test_completed_problem_b_artifact_contract_is_checked_before_postprocessing(
    tmp_path,
):
    artifact = tmp_path / "final_parameters.npz"
    parameters = initial_problem_b_parameters()
    sidecar = save_problem_b_parameters(artifact, parameters)
    common = {
        "status": "complete",
        "stage": "M1",
        "final_parameter_file": str(artifact.resolve()),
        "final_parameter_pytree_sha256": sidecar["parameter_pytree_sha256"],
    }
    write_json_record(tmp_path / "fit_result.json", common)
    write_json_record(tmp_path / "fit_progress.json", common)
    record = _verify_completed_training_artifact("M1", artifact)
    assert record["status"] == "complete"
    assert record["parameter_pytree_sha256"] == SEED_SHA

    incomplete = {**common, "status": "in_progress"}
    write_json_record(tmp_path / "fit_progress.json", incomplete)
    with pytest.raises(ValueError, match="status complete"):
        _verify_completed_training_artifact("M1", artifact)
