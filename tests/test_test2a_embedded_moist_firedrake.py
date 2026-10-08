"""Tiny Firedrake certification for the opt-in Test-2A neural moist child."""

from copy import deepcopy
from contextlib import contextmanager
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from firedrake import COMM_SELF, Cofunction, SpatialCoordinate, as_vector, cos, norm, pi, sin

import dimswe.meshes as dimswe_meshes
from dimswe.jax_moist_hvp import JAXMoistEulerHVP
from dimswe.learned_physics.parameters import tree_axpy, tree_dot, tree_norm
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.moist_backend import JAXMoistEulerIntegrator
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.test2a_embedded_moist import (
    FrozenNeuralAMoistPhysics,
    load_frozen_neural_a_physics,
)
from dimswe.test2a_trajectory import (
    NeuralTrajectoryObjective,
    TrajectoryLossMode,
    continuous_rollout,
    reset_windows,
)
from dimswe.timestepping import Euler, get_timestepper


CFG = "tests/mtswe_small.cfg"
EMBEDDING_CONFIGURATION = "dimswe/configs/test2a_embedded_neural_a.json"
EPS = np.finfo(np.float64).eps
CHILD_ORDER = (
    "dry_rk4_0",
    "dry_rk4_1",
    "hyperviscosity_euler",
    "dg_ssprk43_0",
    "dg_ssprk43_1",
    "moist_euler",
)


def _serial_solver_parameters():
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


def _copy_function(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _function_axpy(base, scale, increment, name):
    result = _copy_function(base, name)
    with result.dat.vec as output, increment.dat.vec_ro as direction:
        output.axpy(float(scale), direction)
    return result


def _dual_axpy(base, scale, increment, name):
    result = base.copy(deepcopy=True)
    result.rename(name)
    with result.dat.vec as output, increment.dat.vec_ro as direction:
        output.axpy(float(scale), direction)
    return result


def _relative_function_error(actual, expected):
    difference = _function_axpy(actual, -1.0, expected, "test2a2_difference")
    return float(norm(difference)) / max(float(norm(expected)), np.finfo(float).tiny)


def _dual_norm(helper, value, name):
    representative = helper.state_riesz_representative(value, name)
    squared = helper.dual_pairing(value, representative)
    return float(np.sqrt(max(0.0, squared)))


def _relative_dual_error(helper, actual, expected):
    difference = _dual_axpy(actual, -1.0, expected, "test2a2_dual_difference")
    return _dual_norm(helper, difference, "test2a2_dual_difference_riesz") / max(
        _dual_norm(helper, expected, "test2a2_dual_expected_riesz"),
        np.finfo(float).tiny,
    )


def _tree_relative_error(actual, expected):
    numerator = float(tree_norm(jax.tree.map(lambda x, y: x - y, actual, expected)))
    denominator = max(float(tree_norm(expected)), np.finfo(np.float64).tiny)
    return numerator / denominator


@pytest.fixture(scope="module")
def embedded_case():
    physics = load_frozen_neural_a_physics(
        EMBEDDING_CONFIGURATION, use_jit=True
    )
    parameters = get_parameters(CFG)
    parameters["mesh"]["type"] = "rectangle"
    parameters["mesh"]["nx"] = 2
    parameters["mesh"]["ny"] = 2
    parameters["timestepping"]["dt"] = 100.0
    parameters["timestepping"]["subcycle_list"] = [2, 1, 2, 1]
    parameters["threewayphysics"]["treat_as_coeffs"] = False
    parameters["hyperviscosity"]["treat_as_coeffs"] = True
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
    coefficient, coefficient_sub, _ = model.get_coeff_var("test2a2_coefficient")
    state_container, state_sub, _ = model.get_full_var(
        "test2a2_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)
    coefficient_sub["c0"].assign(0.07)

    x = SpatialCoordinate(model.mesh)
    lx = model.initcond.Lx
    ly = model.initcond.Ly
    mode_x = sin(2.0 * pi * x[0] / lx)
    mode_y = cos(2.0 * pi * x[1] / ly)
    height = 750.0 + 2.0 * mode_x + 1.5 * mode_y
    state_sub["v"].project(as_vector([2.0 + 0.2 * mode_y, -1.0 + 0.1 * mode_x]))
    state_sub["h"].project(height)
    state_sub["S"].project(
        height * model.initcond.g * (1.01 + 0.0007 * mode_x - 0.0004 * mode_y)
    )
    state_sub["Qv"].project(height * (0.0018 + 0.0004 * mode_x))
    state_sub["Qc"].project(height * (0.0010 + 0.0001 * mode_y))
    state_sub["Qr"].project(0.0002 * height)

    direction = model.get_x_var("test2a2_direction")[0]
    direction.sub(0).project(as_vector([0.12 * mode_x, -0.08 * mode_y]))
    direction.sub(1).project(0.14 * mode_y - 0.05 * mode_x)
    direction.sub(2).project(1.1 * mode_x + 0.6 * mode_y)
    direction.sub(3).project(7.0e-6 * height * (1.0 + 0.1 * mode_x))
    direction.sub(4).project(-5.0e-6 * height * (1.0 - 0.1 * mode_y))
    direction.sub(5).project(3.0e-6 * height * mode_y)

    probe = model.get_x_var("test2a2_probe")[0]
    probe.sub(0).project(as_vector([-0.09 * mode_y, 0.11 * mode_x]))
    probe.sub(1).project(0.07 * mode_x)
    probe.sub(2).project(-0.8 * mode_y)
    probe.sub(3).project(4.0e-6 * height * mode_y)
    probe.sub(4).project(-6.0e-6 * height * mode_x)
    probe.sub(5).project(5.0e-6 * height * (mode_x + mode_y))

    incremental_probe = model.get_x_var("test2a2_incremental_probe")[0]
    incremental_probe.assign(-0.4 * probe)
    solver_parameters = _serial_solver_parameters()
    euler = Euler(model, logger, solver_parameters, terms=["threewayphysics"])
    helper = JAXMoistEulerHVP(euler, local_physics=physics)
    incoming = helper.state_mass_map(probe, "test2a2_incoming")
    incremental_incoming = helper.state_mass_map(
        incremental_probe, "test2a2_incremental_incoming"
    )
    analytical_split = get_timestepper(
        parameters, model, logger, solver_parameters, moist_backend="jax"
    )
    neural_split = get_timestepper(
        parameters,
        model,
        logger,
        solver_parameters,
        moist_backend="jax",
        jax_moist_local_physics=physics,
    )
    for split in (analytical_split, neural_split):
        split.set_coeff(coefficient)
    parameter_direction = jax.tree.map(
        lambda value: jnp.linspace(-0.2, 0.3, value.size).reshape(value.shape)
        / max(1, value.size),
        physics.parameters,
    )
    parameter_direction = jax.tree.map(
        lambda value: value / tree_norm(parameter_direction), parameter_direction
    )
    return {
        "model": model,
        "state": state_container[0],
        "time": float(time),
        "dt": 37.0,
        "configured_dt": 100.0,
        "direction": direction,
        "incoming": incoming,
        "incremental_incoming": incremental_incoming,
        "physics": physics,
        "euler": euler,
        "helper": helper,
        "parameter_direction": parameter_direction,
        "analytical_split": analytical_split,
        "neural_split": neural_split,
        "coefficient": coefficient,
    }


def test_weak_assembly_mass_solve_and_euler_match_manual_hybrid_source(embedded_case):
    case = embedded_case
    helper = case["helper"]
    primal = helper.take_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    assert primal.physics_mode == "neural_A_original_R"
    manual = case["physics"].combined_kernel(
        helper._to_device_tree(primal.packed_state),
        helper._to_device_tree(primal.packed_fields),
        helper._to_device_tree(primal.parameters),
    )
    manual_source = helper._from_device_tree(manual["source"])
    manual_rates = helper._from_device_tree(manual["rates"])
    for key in manual_source:
        np.testing.assert_array_equal(manual_source[key], primal.source_density[key])
    assert float(np.max(np.abs(manual_rates["R"]))) > 0.0
    source_dual = helper.source_assembly(manual_source)
    water_blocks = tuple(
        np.asarray(source_dual.sub(index).dat.data_ro, dtype=np.float64)
        for index in (3, 4, 5)
    )
    water_scale = max(
        *(float(np.max(np.abs(block))) for block in water_blocks),
        np.finfo(np.float64).tiny,
    )
    np.testing.assert_allclose(
        water_blocks[0] + water_blocks[1] + water_blocks[2],
        0.0,
        rtol=0.0,
        atol=64.0 * EPS * water_scale,
    )
    beta2 = float(primal.parameters["g"] * primal.parameters["L"])
    invariant_probe = case["model"].get_x_var("test2a2_invariant_probe")[0]
    invariant_probe.assign(0.0)
    invariant_probe.sub(2).assign(1.0)
    invariant_probe.sub(3).assign(-beta2)
    invariant_residual = helper.dual_pairing(source_dual, invariant_probe)
    entropy_probe = case["model"].get_x_var("test2a2_entropy_probe")[0]
    entropy_probe.assign(0.0)
    entropy_probe.sub(2).assign(1.0)
    entropy_scale = abs(helper.dual_pairing(source_dual, entropy_probe))
    assert abs(invariant_residual) <= 256.0 * EPS * max(
        entropy_scale, np.finfo(np.float64).tiny
    )
    tendency = helper.state_riesz_representative(source_dual, "test2a2_manual_tendency")
    expected = _function_axpy(case["state"], case["dt"], tendency, "test2a2_manual_out")
    assert _relative_function_error(tendency, primal.tendency) < 2.0e-13
    assert _relative_function_error(expected, primal.state_out) < 2.0e-13


def test_complete_state_jvp_matches_centered_fd_and_vjp_transpose(embedded_case):
    case = embedded_case
    helper = case["helper"]
    primal = helper.take_forward_step_cached(case["state"], case["time"], case["dt"])
    tangent = helper.take_tangent_step(primal, case["direction"])
    epsilon = 2.0e-5
    plus_state = _function_axpy(case["state"], epsilon, case["direction"], "test2a2_state_plus")
    minus_state = _function_axpy(case["state"], -epsilon, case["direction"], "test2a2_state_minus")
    plus = helper.take_forward_step_cached(plus_state, case["time"], case["dt"])
    minus = helper.take_forward_step_cached(minus_state, case["time"], case["dt"])
    centered = _function_axpy(plus.state_out, -1.0, minus.state_out, "test2a2_state_fd")
    with centered.dat.vec as vector:
        vector.scale(1.0 / (2.0 * epsilon))
    assert _relative_function_error(centered, tangent.state_direction_out) < 2.0e-7
    reverse = helper.take_adjoint_step_cached(primal, case["incoming"])
    left = helper.dual_pairing(case["incoming"], tangent.state_direction_out)
    right = helper.dual_pairing(reverse.state_adjoint_in, case["direction"])
    np.testing.assert_allclose(left, right, rtol=2.0e-12, atol=2.0e-12)


def test_complete_parameter_jvp_matches_fd_and_vjp_transpose(embedded_case):
    case = embedded_case
    helper = case["helper"]
    physics = case["physics"]
    primal = helper.take_forward_step_cached(case["state"], case["time"], case["dt"])
    direction = case["parameter_direction"]
    tangent = helper.take_parameter_tangent_step(primal, direction)
    epsilon = 2.0e-5
    source_plus = physics.combined_with_parameters(
        helper._to_device_tree(primal.packed_state),
        helper._to_device_tree(primal.packed_fields),
        helper._to_device_tree(primal.parameters),
        tree_axpy(physics.parameters, epsilon, direction),
    )["source"]
    source_minus = physics.combined_with_parameters(
        helper._to_device_tree(primal.packed_state),
        helper._to_device_tree(primal.packed_fields),
        helper._to_device_tree(primal.parameters),
        tree_axpy(physics.parameters, -epsilon, direction),
    )["source"]
    source_fd = helper._from_device_tree(
        jax.tree.map(
            lambda plus, minus: (plus - minus) / (2.0 * epsilon),
            source_plus,
            source_minus,
        )
    )
    tendency_fd = helper.state_riesz_representative(
        helper.source_assembly(source_fd), "test2a2_parameter_fd_tendency"
    )
    output_fd = _copy_function(tendency_fd, "test2a2_parameter_fd_output")
    with output_fd.dat.vec as vector:
        vector.scale(case["dt"])
    assert _relative_function_error(output_fd, tangent.state_direction_out) < 2.0e-7
    reverse = helper.take_parameter_adjoint_step(primal, case["incoming"])
    left = helper.dual_pairing(case["incoming"], tangent.state_direction_out)
    right = float(tree_dot(direction, reverse.parameter_adjoint))
    np.testing.assert_allclose(left, right, rtol=2.0e-12, atol=2.0e-12)


def test_complete_joint_differentiated_vjp_matches_centered_fd(embedded_case):
    case = embedded_case
    helper = case["helper"]
    physics = case["physics"]
    direction = case["parameter_direction"]
    primal = helper.take_forward_step_cached(case["state"], case["time"], case["dt"])
    joint = helper.take_joint_incremental_adjoint_step(
        primal,
        case["direction"],
        direction,
        case["incoming"],
        case["incremental_incoming"],
    )
    epsilon = 1.0e-4
    reverse_results = []
    for sign in (1.0, -1.0):
        local_physics = FrozenNeuralAMoistPhysics(
            tree_axpy(physics.parameters, sign * epsilon, direction),
            physics.model_configuration,
            physics.normalization,
            provenance=physics.provenance,
            use_jit=True,
        )
        local_helper = JAXMoistEulerHVP(
            case["euler"], local_physics=local_physics
        )
        local_state = _function_axpy(
            case["state"], sign * epsilon, case["direction"], f"test2a2_joint_state_{sign}"
        )
        local_incoming = _dual_axpy(
            case["incoming"], sign * epsilon, case["incremental_incoming"], f"test2a2_joint_incoming_{sign}"
        )
        local_primal = local_helper.take_forward_step_cached(
            local_state, case["time"], case["dt"]
        )
        reverse_results.append(
            local_helper.take_parameter_adjoint_step(local_primal, local_incoming)
        )
    state_centered = _dual_axpy(
        reverse_results[0].ordinary_state_reverse.state_adjoint_in,
        -1.0,
        reverse_results[1].ordinary_state_reverse.state_adjoint_in,
        "test2a2_joint_state_centered",
    )
    with state_centered.dat.vec as vector:
        vector.scale(1.0 / (2.0 * epsilon))
    parameter_centered = jax.tree.map(
        lambda plus, minus: (plus - minus) / (2.0 * epsilon),
        reverse_results[0].parameter_adjoint,
        reverse_results[1].parameter_adjoint,
    )
    assert _relative_dual_error(
        helper, state_centered, joint.incremental_state_adjoint_in
    ) < 3.0e-6
    assert _tree_relative_error(
        parameter_centered, joint.incremental_parameter_adjoint
    ) < 3.0e-6


def test_neural_mode_is_opt_in_and_complete_split_order_is_unchanged(embedded_case):
    case = embedded_case
    analytical = case["analytical_split"]
    neural = case["neural_split"]
    assert analytical.moist_backend == neural.moist_backend == "jax"
    assert analytical.time_integrators[-1].moist_A_model == "analytical"
    assert neural.time_integrators[-1].moist_A_model == "neural"
    assert isinstance(neural.time_integrators[-1], JAXMoistEulerIntegrator)
    assert [type(child) for child in analytical.time_integrators[:3]] == [
        type(child) for child in neural.time_integrators[:3]
    ]
    helper = neural._get_mtswe_split_hvp_helper()
    specs = helper._child_specs(case["time"], case["configured_dt"])
    assert tuple(spec[0] for spec in specs) == CHILD_ORDER
    assert helper.production_graph_diagnostics()["reverse_child_order"] == tuple(
        reversed(CHILD_ORDER)
    )
    cache = helper.take_forward_step_cached(
        case["state"], case["time"], case["configured_dt"]
    )
    assert cache.forward_child_order == CHILD_ORDER
    moist = [child for child in cache.children if child.name == "moist_euler"]
    assert len(moist) == 1
    assert moist[0].cache.physics_mode == "neural_A_original_R"
    for field in cache.state_out.subfunctions:
        assert np.all(np.isfinite(np.asarray(field.dat.data_ro)))


def _trajectory_case(case):
    helper = case["neural_split"]._get_mtswe_split_hvp_helper()
    coefficient = case["coefficient"]

    @contextmanager
    def physical_c0(value):
        c0 = coefficient.subfunctions[1]
        previous = float(c0)
        c0.assign(float(value))
        try:
            yield
        finally:
            c0.assign(previous)

    return SimpleNamespace(
        helper=helper,
        dt=case["configured_dt"],
        t0=case["time"],
        physical_c0=physical_c0,
        new_state=lambda name: case["model"].get_x_var(name)[0],
    )


def _short_truth(case, count=3):
    helper = case["analytical_split"]._get_mtswe_split_hvp_helper()
    current = _copy_function(case["state"], "test2a_trajectory_truth_0")
    truth = {0: current}
    for step in range(count):
        current = _copy_function(
            helper.take_forward_step_cached(
                current,
                case["time"] + step * case["configured_dt"],
                case["configured_dt"],
            ).state_out,
            f"test2a_trajectory_truth_{step + 1}",
        )
        truth[step + 1] = current
    return truth


def test_explicit_parameters_and_fixed_prefix_match_complete_step(embedded_case):
    case = embedded_case
    helper = case["neural_split"]._get_mtswe_split_hvp_helper()
    parameters = case["physics"].parameters
    ordinary = helper.take_forward_step_cached(
        case["state"], case["time"], case["configured_dt"]
    )
    explicit = helper.take_forward_step_cached(
        case["state"],
        case["time"],
        case["configured_dt"],
        neural_parameters=parameters,
    )
    prefix = helper.take_fixed_prefix_cached(
        case["state"], case["time"], case["configured_dt"]
    )
    cached = helper.take_forward_step_from_prefix(prefix, parameters)
    assert _relative_function_error(explicit.state_out, ordinary.state_out) < 2.0e-13
    assert _relative_function_error(cached.state_out, explicit.state_out) < 2.0e-13
    assert tuple(child.name for child in prefix.children) == CHILD_ORDER[:-1]


def test_complete_split_neural_parameter_tangent_and_reverse_are_transposes(
    embedded_case,
):
    case = embedded_case
    helper = case["neural_split"]._get_mtswe_split_hvp_helper()
    parameters = case["physics"].parameters
    primal = helper.take_forward_step_cached(
        case["state"],
        case["time"],
        case["configured_dt"],
        neural_parameters=parameters,
    )
    zero = case["model"].get_x_var("test2a_trajectory_zero_direction")[0]
    zero.assign(0.0)
    tangent = helper.take_neural_parameter_tangent_step(
        primal, zero, case["parameter_direction"]
    )
    reverse = helper.take_neural_parameter_adjoint_step(
        primal, case["incoming"]
    )
    left = helper.dual_pairing(case["incoming"], tangent.state_direction_out)
    right = float(tree_dot(case["parameter_direction"], reverse.parameter_adjoint))
    np.testing.assert_allclose(left, right, rtol=3.0e-12, atol=3.0e-12)
    assert reverse.reverse_child_order == tuple(reversed(CHILD_ORDER))


def test_shared_H2_objective_gradient_tape_and_invalidation(embedded_case):
    case = embedded_case
    trajectory_case = _trajectory_case(case)
    truth = _short_truth(case, 2)
    parameters = case["physics"].parameters
    windows = continuous_rollout(
        2, TrajectoryLossMode.ACCUMULATED, (0.4, 0.6)
    )
    objective = NeuralTrajectoryObjective(
        trajectory_case, truth, windows, c0=0.14, use_fixed_prefix=True
    )
    value = objective.value(parameters)
    exact_value, gradient = objective.value_and_gradient(parameters)
    np.testing.assert_allclose(value, exact_value, rtol=0.0, atol=0.0)
    counts = objective.work_counts()
    assert counts.forward_complete_steps == 2
    assert counts.same_theta_tape_hits == 1
    direction = case["parameter_direction"]
    epsilon = 2.0e-5
    plus = objective.value(tree_axpy(parameters, epsilon, direction))
    minus = objective.value(tree_axpy(parameters, -epsilon, direction))
    centered = (plus - minus) / (2.0 * epsilon)
    adjoint = float(tree_dot(gradient, direction))
    np.testing.assert_allclose(centered, adjoint, rtol=4.0e-5, atol=2.0e-12)
    changed = objective.work_counts()
    assert changed.tape_invalidations >= 2
    assert changed.forward_complete_steps == 6


def test_reset_windows_use_requested_truth_origins_not_previous_endpoint(
    embedded_case,
):
    case = embedded_case
    trajectory_case = _trajectory_case(case)
    truth = _short_truth(case, 3)
    windows = reset_windows((0, 2), 1, TrajectoryLossMode.ENDPOINT, (1.0,))
    objective = NeuralTrajectoryObjective(
        trajectory_case,
        truth,
        windows,
        c0=0.14,
        use_fixed_prefix=True,
    )
    tape = objective._tape(case["physics"].parameters)
    assert len(tape.windows) == 2
    assert _relative_function_error(tape.windows[1].states[0], truth[2]) == 0.0
    assert _relative_function_error(
        tape.windows[1].states[0], tape.windows[0].states[-1]
    ) > 0.0
