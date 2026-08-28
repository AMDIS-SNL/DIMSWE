"""Tiny Firedrake certification of the Test 2A-3A offline objective."""

from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from firedrake import COMM_SELF, SpatialCoordinate, as_vector, cos, pi, sin

import dimswe.meshes as dimswe_meshes
from dimswe.jax_moist_hvp import JAXMoistEulerHVP
from dimswe.learned_physics.parameters import tree_axpy, tree_dot, tree_norm
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.test2a_discrete_offline import (
    DeployedDiscreteOfflineObjective,
    DiscreteOfflineObservation,
    ProductionDiscreteOfflineOperations,
    ProductionObservationPayload,
    _a_sensitive_source,
)
from dimswe.test2a_discrete_training import (
    FastFixedDiscreteObjective,
    FixedDiscreteCache,
    _matrix_cache_components,
)
from dimswe.test2a_embedded_moist import load_frozen_neural_a_physics
from dimswe.test2a_operator import (
    initialize_mlp,
    load_selected_configuration,
    mlp_configuration_from_record,
)
from dimswe.timestepping import Euler


CFG = "tests/mtswe_small.cfg"
EMBEDDING = "dimswe/configs/test2a_embedded_neural_a.json"
SELECTED_OPERATOR = "dimswe/configs/test2a_selected_operator.json"


def _direct_solver_parameters():
    parameters = deepcopy(overall_solver_parameters)
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    for name in (
        "erkstage-f",
        "erkstage-aux",
        "erkstage-mu",
        "erkstage-muaux",
        "erk-dlambda",
        "erk-grad",
    ):
        parameters[name] = direct
    return parameters


def _copy_state(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _state_axpy(base, scale, direction, name):
    result = _copy_state(base, name)
    with result.dat.vec as output, direction.dat.vec_ro as increment:
        output.axpy(float(scale), increment)
    return result


@pytest.fixture(scope="module")
def discrete_case():
    selected = load_selected_configuration(SELECTED_OPERATOR)
    model_configuration = mlp_configuration_from_record(selected["model"])
    initial_parameters = initialize_mlp(model_configuration)
    neural_physics = load_frozen_neural_a_physics(EMBEDDING, use_jit=True)

    parameters = get_parameters(CFG)
    parameters["mesh"]["type"] = "rectangle"
    parameters["mesh"]["nx"] = 2
    parameters["mesh"]["ny"] = 2
    parameters["timestepping"]["dt"] = 100.0
    parameters["threewayphysics"]["treat_as_coeffs"] = False
    logger = EmptyLogger()
    original_rectangle_mesh = dimswe_meshes.RectangleMesh

    def comm_self_mesh(*args, **kwargs):
        kwargs["comm"] = COMM_SELF
        mesh = original_rectangle_mesh(*args, **kwargs)
        mesh.dx = 1.0
        mesh.dy = 1.0
        return mesh

    dimswe_meshes.RectangleMesh = comm_self_mesh
    try:
        model = get_model(parameters, logger, has_dynamics_statistics=False)
    finally:
        dimswe_meshes.RectangleMesh = original_rectangle_mesh
    state_container, state_sub, _ = model.get_full_var(
        "test2a3a_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    x = SpatialCoordinate(model.mesh)
    lx = model.initcond.Lx
    ly = model.initcond.Ly
    mx = sin(2.0 * pi * x[0] / lx)
    my = cos(2.0 * pi * x[1] / ly)
    h = 750.0 + 3.0 * mx + 2.0 * my
    state_sub["v"].project(as_vector([1.0 + 0.1 * my, -0.5 + 0.1 * mx]))
    state_sub["h"].project(h)
    state_sub["S"].project(h * model.initcond.g * (1.01 + 0.0005 * mx))
    state_sub["Qv"].project(h * (0.0017 + 0.0003 * mx))
    state_sub["Qc"].project(h * (0.0010 + 0.0001 * my))
    state_sub["Qr"].project(h * 0.0002)

    state_direction = model.get_x_var("test2a3a_state_direction")[0]
    state_direction.assign(0.0)
    state_direction.sub(2).project(0.4 * h * my)
    state_direction.sub(3).project(2.0e-5 * h * mx)
    state_direction.sub(4).project(-1.0e-5 * h * my)
    states = (
        _copy_state(state_container[0], "test2a3a_truth_0"),
        _state_axpy(
            state_container[0], 0.25, state_direction, "test2a3a_truth_1"
        ),
    )

    euler = Euler(
        model,
        logger,
        _direct_solver_parameters(),
        terms=["threewayphysics"],
    )
    helper = JAXMoistEulerHVP(euler, use_jit=True)
    operations = ProductionDiscreteOfflineOperations(helper, neural_physics)
    observations = []
    target_caches = []
    for step, state in enumerate(states):
        target = helper.take_forward_step_cached(state, float(time), 100.0)
        target_caches.append(target)
        a_tendency = helper.state_riesz_representative(
            helper.source_assembly(_a_sensitive_source(target)),
            f"test2a3a_a_tendency_{step}",
        )
        observations.append(
            DiscreteOfflineObservation(
                step=step,
                payload=ProductionObservationPayload(
                    packed_state=helper._to_device_tree(target.packed_state),
                    packed_fields=helper._to_device_tree(target.packed_fields),
                    moist_parameters=helper._to_device_tree(target.parameters),
                    target_original_r=np.asarray(
                        target.rates["R"], dtype=np.float64
                    ).copy(),
                ),
                target_tendency=target.tendency.copy(deepcopy=True),
                analytical_a_tendency=a_tendency,
            )
        )
    objective = DeployedDiscreteOfflineObjective(observations, operations)
    parameter_direction = jax.tree.map(
        lambda value: jnp.linspace(-0.4, 0.3, value.size).reshape(value.shape),
        initial_parameters,
    )
    direction_norm = tree_norm(parameter_direction)
    parameter_direction = jax.tree.map(
        lambda value: value / direction_norm, parameter_direction
    )
    return {
        "states": states,
        "target_caches": tuple(target_caches),
        "helper": helper,
        "operations": operations,
        "objective": objective,
        "parameters": initial_parameters,
        "direction": parameter_direction,
    }


def _relative_tree_error(actual, expected):
    difference = jax.tree.map(lambda left, right: left - right, actual, expected)
    return float(tree_norm(difference)) / max(
        float(tree_norm(expected)), np.finfo(np.float64).tiny
    )


def test_primal_target_prediction_and_mass_metric_are_deployed(discrete_case):
    case = discrete_case
    objective = case["objective"]
    parameters = case["parameters"]
    assert case["operations"].helper is case["helper"]
    value = objective.value(parameters)
    assert np.isfinite(value) and value > 0.0
    numerator = 0.0
    for observation, target in zip(
        objective.observations, case["target_caches"]
    ):
        assert target.physics_mode == "analytical_A_original_R"
        prediction = case["operations"].predict(parameters, observation)
        assert float(np.max(np.abs(target.rates["R"]))) > 0.0
        np.testing.assert_allclose(
            prediction.auxiliary["rates"]["R"],
            target.rates["R"],
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )
        residual = case["operations"].subtract(
            prediction.tendency, target.tendency, "test2a3a_manual_residual"
        )
        numerator += case["operations"].squared_mass_norm(residual)
    assert value == pytest.approx(
        numerator / objective.normalizer, rel=2.0e-13
    )


def test_exact_all_parameter_gradient_matches_fd_and_independent_jvp_chain(
    discrete_case,
):
    case = discrete_case
    objective = case["objective"]
    parameters = case["parameters"]
    direction = case["direction"]
    value, gradient = objective.value_and_gradient(parameters)
    assert np.isfinite(value)
    epsilon = 2.0e-5
    centered = (
        objective.value(tree_axpy(parameters, epsilon, direction))
        - objective.value(tree_axpy(parameters, -epsilon, direction))
    ) / (2.0 * epsilon)
    reverse_directional = float(tree_dot(gradient, direction))
    np.testing.assert_allclose(
        centered, reverse_directional, rtol=3.0e-6, atol=3.0e-10
    )

    jvp_directional = 0.0
    for observation in objective.observations:
        prediction = case["operations"].predict(parameters, observation)
        residual = case["operations"].subtract(
            prediction.tendency,
            observation.target_tendency,
            "test2a3a_chain_residual",
        )
        _, source_direction_device = case["operations"]._parameter_jvp_kernel(
            parameters,
            direction,
            observation.payload.packed_state,
            observation.payload.packed_fields,
            observation.payload.moist_parameters,
        )
        source_direction = case["helper"]._from_device_tree(
            source_direction_device
        )
        tendency_direction = case["helper"].state_riesz_representative(
            case["helper"].source_assembly(source_direction),
            "test2a3a_chain_tendency_direction",
        )
        residual_dual = case["helper"].state_mass_map(
            residual, "test2a3a_chain_residual_dual"
        )
        jvp_directional += 2.0 * case["helper"].dual_pairing(
            residual_dual, tendency_direction
        )
    jvp_directional /= objective.normalizer
    np.testing.assert_allclose(
        jvp_directional, reverse_directional, rtol=3.0e-12, atol=3.0e-12
    )
    assert jax.tree.structure(gradient) == jax.tree.structure(parameters)
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(gradient))


def test_exact_deployed_discrete_hvp_matches_gradient_difference(discrete_case):
    case = discrete_case
    objective = case["objective"]
    parameters = case["parameters"]
    direction = case["direction"]
    action = objective.hess_vec(parameters, direction)
    epsilon = 5.0e-5
    plus = objective.gradient(tree_axpy(parameters, epsilon, direction))
    minus = objective.gradient(tree_axpy(parameters, -epsilon, direction))
    centered = jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon), plus, minus
    )
    assert _relative_tree_error(action, centered) < 8.0e-5


def test_exact_sparse_fixed_cache_matches_production_value_and_gradient(discrete_case):
    case = discrete_case
    objective = case["objective"]
    operations = case["operations"]
    physics = operations.neural_physics
    matrices, audit = _matrix_cache_components(
        SimpleNamespace(objective=objective), 2.0e-13, 16, 2.0e-13
    )
    feature_blocks = []
    target_blocks = []
    h_blocks = []
    for target in case["target_caches"]:
        block = np.column_stack(
            tuple(
                np.asarray(target.packed_state[name], dtype=np.float64).reshape(-1)
                for name in ("h", "S", "Qv", "Qc")
            )
            + (
                np.asarray(target.packed_fields["B"], dtype=np.float64).reshape(-1),
            )
        )
        feature_blocks.append(block)
        h_blocks.append(block[:, 0])
        target_blocks.append(
            np.asarray(target.rates["A"], dtype=np.float64).reshape(-1)
            / physics.normalization.output_scale
        )
    features = np.concatenate(feature_blocks)
    targets = np.stack(target_blocks)
    h = np.stack(h_blocks)
    cache = FixedDiscreteCache(
        normalized_features=np.asarray(
            physics.normalization.normalize_features(features), dtype=np.float64
        ),
        normalized_targets=targets,
        h=h,
        beta2=float(
            case["target_caches"][0].parameters["g"]
            * case["target_caches"][0].parameters["L"]
        ),
        output_scale=physics.normalization.output_scale,
        w_s_data=matrices["S"]["data"],
        w_s_indices=matrices["S"]["indices"],
        w_s_shape=matrices["S"]["shape"],
        w_q_data=matrices["Q"]["data"],
        w_q_indices=matrices["Q"]["indices"],
        w_q_shape=matrices["Q"]["shape"],
        mass_inverse_s_x=matrices["S"]["inverse_x"],
        mass_inverse_s_y=matrices["S"]["inverse_y"],
        mass_s_grid_order=matrices["S"]["grid_order"],
        mass_s_grid_shape=matrices["S"]["grid_shape"],
        mass_inverse_q_data=matrices["Q"]["mass_inverse_data"],
        mass_inverse_q_indices=matrices["Q"]["mass_inverse_indices"],
        mass_inverse_q_shape=matrices["Q"]["mass_inverse_shape"],
        normalizer=objective.normalizer,
        metadata={"production_oracle_certified": True},
    )
    fast = FastFixedDiscreteObjective(
        cache, physics.model_configuration, use_jit=True
    )
    production_value, production_gradient = objective.value_and_gradient(
        case["parameters"]
    )
    cached_value, cached_gradient = fast.value_and_gradient(case["parameters"])
    assert cached_value == pytest.approx(production_value, rel=2.0e-11)
    assert _relative_tree_error(cached_gradient, production_gradient) < 2.0e-10
    assert audit["S"]["tensor_factorization_relative_residual"] < 2.0e-13
    assert audit["Q"]["maximum_component_size"] <= 16
