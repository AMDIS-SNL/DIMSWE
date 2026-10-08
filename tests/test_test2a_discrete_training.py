"""Cheap exact-cache and checkpoint tests for Test 2A-3C."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyrol")

from pyrol import Problem, Solver

from dimswe.test2a_discrete_training import (
    CompactCheckpointObjective,
    FastFixedDiscreteObjective,
    FixedDiscreteCache,
    load_fixed_cache,
    load_discrete_training_configuration,
    save_fixed_cache,
    validate_resume_record,
)
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2a_operator import MLPConfiguration, initialize_mlp
from dimswe.test2a_pyrol import build_test2a_lbfgs_parameters


CONFIG = "dimswe/configs/test2a_deployed_discrete_50k.json"
SEED0 = "6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975"
DIRECT = "4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a"


def _tiny_cache():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(6, 5)).astype(np.float64)
    targets = rng.normal(size=(2, 3)).astype(np.float64)
    h = np.array([[2.0, 2.5, 3.0], [1.5, 2.0, 2.7]], dtype=np.float64)
    w_s = np.array([[1.0, 0.2, 0.0], [0.0, -0.4, 0.8]], dtype=np.float64)
    w_q = np.array([[0.5, 0.1, -0.2], [0.0, 0.7, 0.3]], dtype=np.float64)

    def coo(matrix):
        row, column = np.nonzero(matrix)
        return matrix[row, column], np.column_stack((row, column)).astype(np.int32)

    s_data, s_indices = coo(w_s)
    q_data, q_indices = coo(w_q)
    mass_s = np.array([[1.2, 0.1], [0.1, 0.8]], dtype=np.float64)
    mass_q = np.array([[0.7, 0.05], [0.05, 1.4]], dtype=np.float64)
    inverse_s = np.linalg.inv(mass_s)
    inverse_q = np.linalg.inv(mass_q)
    inverse_s_data, inverse_s_indices = coo(inverse_s)
    inverse_q_data, inverse_q_indices = coo(inverse_q)
    beta2 = 3.0
    physical_target = targets
    source_q = h * physical_target
    source_s = beta2 * source_q
    weak_s = source_s @ w_s.T
    weak_q = source_q @ w_q.T
    normalizer = np.sum(weak_s * (weak_s @ inverse_s.T)) + 2.0 * np.sum(
        weak_q * (weak_q @ inverse_q.T)
    )
    cache = FixedDiscreteCache(
        normalized_features=features,
        normalized_targets=targets,
        h=h,
        beta2=beta2,
        output_scale=1.0,
        w_s_data=s_data,
        w_s_indices=s_indices,
        w_s_shape=w_s.shape,
        w_q_data=q_data,
        w_q_indices=q_indices,
        w_q_shape=w_q.shape,
        mass_inverse_s_x=inverse_s,
        mass_inverse_s_y=np.ones((1, 1), dtype=np.float64),
        mass_s_grid_order=np.arange(2, dtype=np.int32),
        mass_s_grid_shape=(1, 2),
        mass_inverse_q_data=inverse_q_data,
        mass_inverse_q_indices=inverse_q_indices,
        mass_inverse_q_shape=inverse_q.shape,
        normalizer=float(normalizer),
        metadata={"production_oracle_certified": True},
    )
    return cache, w_s, w_q


def test_selected_training_contract_seed0_memory20_no_hvp_no_recursion():
    record = load_discrete_training_configuration(CONFIG)
    assert record["truth_state_indices"] == [0, 80]
    assert record["states_after_80_forbidden"] is True
    assert record["recursive_model_state_propagation"] is False
    assert record["optimizer"]["maximum_secant_storage"] == 20
    assert record["optimizer"]["production_HVP"] is False
    assert record["checkpoint_accepted_iterations"][-1] == 50000
    assert record["direct_production_method2_baseline"][
        "parameter_pytree_sha256"
    ] == DIRECT
    assert record["direct_production_method2_baseline"]["accepted_J_disc"] == (
        0.0017427829635521567
    )
    configuration = MLPConfiguration()
    assert parameter_pytree_sha256(initialize_mlp(configuration)) == SEED0


def test_sparse_fixed_objective_matches_manual_G_weighting_and_gradient_fd():
    cache, w_s, w_q = _tiny_cache()
    configuration = MLPConfiguration(hidden_layers=(4,), seed=2)
    parameters = initialize_mlp(configuration)
    objective = FastFixedDiscreteObjective(cache, configuration, use_jit=False)
    prediction = np.asarray(
        objective.model(parameters, jnp.asarray(cache.normalized_features))
    ).reshape(cache.h.shape)
    error = prediction - cache.normalized_targets
    source_q = cache.h * error
    source_s = cache.beta2 * source_q
    weak_s = source_s @ w_s.T
    weak_q = source_q @ w_q.T
    manual = (
        np.sum(weak_s * (weak_s @ np.linalg.inv(np.array([[1.2, 0.1], [0.1, 0.8]])).T))
        + 2.0 * np.sum(weak_q * (weak_q @ np.linalg.inv(np.array([[0.7, 0.05], [0.05, 1.4]])).T))
    ) / cache.normalizer
    value, gradient = objective.value_and_gradient(parameters)
    assert value == pytest.approx(float(manual), rel=2.0e-14)
    direction = jax.tree.map(lambda leaf: jnp.ones_like(leaf) * 0.03, parameters)
    epsilon = 2.0e-6
    plus = jax.tree.map(lambda p, d: p + epsilon * d, parameters, direction)
    minus = jax.tree.map(lambda p, d: p - epsilon * d, parameters, direction)
    centered = (objective.value(plus) - objective.value(minus)) / (2.0 * epsilon)
    reverse = sum(
        float(jnp.vdot(left, right))
        for left, right in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction))
    )
    assert centered == pytest.approx(reverse, rel=2.0e-8, abs=2.0e-9)
    hvp = objective.hess_vec(parameters, direction)
    gradient_plus = objective.gradient(plus)
    gradient_minus = objective.gradient(minus)
    centered_hvp = jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon),
        gradient_plus,
        gradient_minus,
    )
    hvp_difference = jax.tree.map(
        lambda left, right: left - right, hvp, centered_hvp
    )
    difference_norm = np.sqrt(
        sum(float(jnp.vdot(leaf, leaf)) for leaf in jax.tree.leaves(hvp_difference))
    )
    reference_norm = np.sqrt(
        sum(float(jnp.vdot(leaf, leaf)) for leaf in jax.tree.leaves(hvp))
    )
    assert difference_norm / reference_norm < 2.0e-8


def test_compact_checkpoint_adapter_dispatches_lbfgs_without_hvp():
    configuration = MLPConfiguration(hidden_layers=(3,), seed=3)
    parameters = initialize_mlp(configuration)

    def objective(value):
        return sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(value))

    accepted = []
    adapter = CompactCheckpointObjective(
        objective,
        parameters,
        use_jit=False,
        accepted_callback=lambda control, index, owner: accepted.append(index),
    )
    control = adapter.vector_from_pytree(parameters)
    rol = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": 1.0e-10,
            "step_tolerance": 1.0e-14,
            "iteration_limit": 5,
            "maximum_secant_storage": 20,
        }
    )
    assert rol.sublist("General").sublist("Secant").get("Maximum Storage") == 20
    Solver(Problem(adapter, control), rol).solve()
    assert accepted[0] == 0
    assert adapter.accepted_update_count >= 2
    assert adapter.hvp_evaluations == 0
    assert len(adapter.accepted_iteration_history) == adapter.accepted_update_count
    assert all("control" not in item for item in adapter.accepted_iteration_history)


def test_resume_policy_rejects_incompatible_configuration_and_cache():
    record = {
        "status": "in_progress",
        "configuration_sha256": "config",
        "cache_npz_sha256": "cache",
        "last_checkpoint_accepted_iteration": 2500,
    }
    assert validate_resume_record(record, "config", "cache") == 2500
    with pytest.raises(ValueError, match="configuration"):
        validate_resume_record(record, "changed", "cache")
    with pytest.raises(ValueError, match="cache"):
        validate_resume_record(record, "config", "changed")
    with pytest.raises(ValueError, match="not resumable"):
        validate_resume_record({**record, "status": "complete"}, "config", "cache")


def test_fixed_cache_roundtrip_preserves_certification_and_rejects_tampering(tmp_path):
    cache, _, _ = _tiny_cache()
    path = tmp_path / "fixed_cache.npz"
    save_fixed_cache(path, cache)
    loaded = load_fixed_cache(path, require_canonical=False)
    assert loaded.sample_count == cache.sample_count
    assert loaded.metadata["truth_state_indices"] == [0, 80]
    assert loaded.metadata["states_after_80_accessed"] is False
    assert loaded.metadata["recursive_model_state_propagation"] is False
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="fingerprint"):
        load_fixed_cache(path, require_canonical=False)

    incomplete = tmp_path / "incomplete.npz"
    incomplete.write_bytes(b"partial")
    with pytest.raises(FileNotFoundError, match="requires both"):
        load_fixed_cache(incomplete, require_canonical=False)
