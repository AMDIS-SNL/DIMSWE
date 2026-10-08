"""J2 certification tests for exact JAX moist derivatives.

The ``TestPureJAXMoistDerivatives`` class has no Firedrake dependency and can
be selected independently.  Firedrake transpose and production-oracle tests
are defined later in this file and import Firedrake only from their fixture.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest


jax = pytest.importorskip("jax", reason="JAX is optional for DIMSWE")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy", reason="JAX is optional for DIMSWE")
from jax.flatten_util import ravel_pytree  # noqa: E402

from dimswe.jax_moist import (  # noqa: E402
    moist_diagnostics_jax,
    moist_source_density_jax,
)
from dimswe.jax_moist_derivatives import (  # noqa: E402
    moist_source_differentiated_vjp,
    moist_source_differentiated_vjp_jit,
    moist_source_jvp,
    moist_source_jvp_jit,
    moist_source_vjp,
    moist_source_vjp_jit,
)


EPS = np.finfo(np.float64).eps
STATE_KEYS = ("h", "S", "Qv", "Qc")
SOURCE_KEYS = ("S", "Qv", "Qc", "Qr")
MASK_KEYS = (
    "condensation_mask",
    "evaporation_mask",
    "uncapped_evaporation_mask",
    "rain_mask",
)
SMOOTH_BRANCHES = (
    ("condensation_rain", 0.0030, 0.0010),
    ("condensation_no_rain", 0.0030, 0.00005),
    ("evaporation_uncapped_rain", 0.0010, 0.0010),
    ("evaporation_capped_rain", 0.0010, 0.0002),
    ("evaporation_capped_no_rain", 0.0010, 0.00005),
    ("deployed_negative_cloud_branch", 0.0030, -0.0002),
)


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


def _state(qv=0.003, qc=0.001, *, shape=()):
    h = np.full(shape, 750.0, dtype=np.float64)
    modulation = np.ones(shape, dtype=np.float64)
    if shape:
        modulation = np.linspace(0.97, 1.03, int(np.prod(shape))).reshape(shape)
    h = h * modulation
    entropy = np.asarray(9.80616, dtype=np.float64)
    return {
        "h": h,
        "S": h * entropy,
        "Qv": h * np.asarray(qv, dtype=np.float64),
        "Qc": h * np.asarray(qc, dtype=np.float64),
    }


def _fields(shape=()):
    return {"B": np.zeros(shape, dtype=np.float64)}


def _direction(shape=(), *, scale=1.0):
    pattern = np.ones(shape, dtype=np.float64)
    if shape:
        pattern = np.linspace(0.8, 1.2, int(np.prod(shape))).reshape(shape)
    factor = np.asarray(scale, dtype=np.float64)
    return {
        "h": factor * 0.31 * pattern,
        "S": factor * -1.7 * pattern[::-1] if shape else factor * -1.7,
        "Qv": factor * 0.017 * pattern,
        "Qc": factor * -0.011 * pattern,
    }


def _source_covector(shape=(), *, scale=1.0):
    pattern = np.ones(shape, dtype=np.float64)
    if shape:
        pattern = np.linspace(0.9, 1.1, int(np.prod(shape))).reshape(shape)
    factor = np.asarray(scale, dtype=np.float64)
    return {
        "S": factor * 0.31 * pattern,
        "Qv": factor * -0.73 * pattern,
        "Qc": factor * 1.17 * pattern,
        "Qr": factor * -0.43 * pattern,
    }


def _tree_numpy(tree):
    return jax.tree.map(lambda value: np.array(value, copy=True), tree)


def _tree_axpy(base, scale, direction):
    return jax.tree.map(
        lambda left, right: jnp.asarray(left) + scale * jnp.asarray(right),
        base,
        direction,
    )


def _tree_centered(plus, minus, epsilon):
    return jax.tree.map(
        lambda left, right: (left - right) / (2.0 * epsilon), plus, minus
    )


def _tree_relative_error(actual, expected):
    actual_flat, _ = ravel_pytree(actual)
    expected_flat, _ = ravel_pytree(expected)
    error = float(jnp.linalg.norm(actual_flat - expected_flat))
    scale = max(float(jnp.linalg.norm(expected_flat)), np.finfo(float).tiny)
    return error / scale


def _tree_dot(left, right):
    products = jax.tree.map(lambda x, y: jnp.vdot(x, y), left, right)
    return sum(jax.tree.leaves(products), start=jnp.float64(0.0))


def _fd_ladder():
    return tuple(np.float64(2.0 ** -power) for power in range(3, 15))


def _dense_maps(state, fields, parameters):
    state_vector, unravel_state = ravel_pytree(state)

    def source_vector(vector):
        source = moist_source_density_jax(
            unravel_state(vector), fields, parameters
        )
        return ravel_pytree(source)[0]

    return state_vector, unravel_state, source_vector


def _pure_active_set(state, fields, parameters):
    diagnostics = moist_diagnostics_jax(state, fields, parameters)
    signature = tuple(
        tuple(bool(value) for value in np.asarray(diagnostics[key]).reshape(-1))
        for key in MASK_KEYS
    )
    margin_pairs = (
        ("condensation_margin", "condensation_argument"),
        ("evaporation_margin", "evaporation_argument"),
        ("evaporation_cap_margin", "evaporation_cap_difference"),
        ("rain_margin", "rain_argument"),
        ("depth_denominator_margin", "depth_denominator"),
    )
    separation = {}
    for margin_name, value_name in margin_pairs:
        margin = float(np.asarray(diagnostics[margin_name]))
        scale = max(
            float(np.max(np.abs(np.asarray(diagnostics[value_name])))),
            np.finfo(float).tiny,
        )
        separation[margin_name] = margin / scale
    return signature, separation


def _assert_pure_active_sets_stable(base, *candidates):
    base_signature, base_separation = base
    assert min(base_separation.values()) > 1.0e-5, base_separation
    for signature, separation in candidates:
        assert signature == base_signature
        assert min(separation.values()) > 1.0e-5, separation


class TestPureJAXMoistDerivatives:
    @pytest.mark.parametrize(("name", "qv", "qc"), SMOOTH_BRANCHES)
    def test_jvp_centered_fd_on_every_smooth_branch(self, name, qv, qc):
        del name
        state = _state(qv=qv, qc=qc, shape=(3,))
        direction = _direction(shape=(3,))
        fields = _fields(shape=(3,))
        parameters = _parameters()
        _, exact = moist_source_jvp(state, direction, fields, parameters)
        base_active_set = _pure_active_set(state, fields, parameters)
        errors = []
        for epsilon in _fd_ladder():
            plus_state = _tree_axpy(state, epsilon, direction)
            minus_state = _tree_axpy(state, -epsilon, direction)
            plus = moist_source_density_jax(
                plus_state, fields, parameters
            )
            minus = moist_source_density_jax(
                minus_state, fields, parameters
            )
            _assert_pure_active_sets_stable(
                base_active_set,
                _pure_active_set(plus_state, fields, parameters),
                _pure_active_set(minus_state, fields, parameters),
            )
            errors.append(
                _tree_relative_error(
                    _tree_centered(plus, minus, epsilon), exact
                )
            )
        assert min(errors) < 2.0e-8, errors
        assert all(np.isfinite(error) for error in errors), errors

    @pytest.mark.parametrize("shape", ((), (2,)))
    def test_jvp_matches_tiny_dense_jacfwd(self, shape):
        state = _state(qv=0.001, qc=0.001, shape=shape)
        fields = _fields(shape=shape)
        parameters = _parameters()
        direction = _direction(shape=shape)
        _, actual = moist_source_jvp(state, direction, fields, parameters)
        vector, _, source_vector = _dense_maps(state, fields, parameters)
        direction_vector, _ = ravel_pytree(direction)
        expected = jax.jacfwd(source_vector)(vector) @ direction_vector
        actual_vector, _ = ravel_pytree(actual)
        np.testing.assert_allclose(
            actual_vector, expected, rtol=256.0 * EPS, atol=256.0 * EPS
        )

    def test_vjp_pairing(self):
        state = _state(qv=0.001, qc=0.0002, shape=(3,))
        direction = _direction(shape=(3,))
        covector = _source_covector(shape=(3,))
        fields = _fields(shape=(3,))
        parameters = _parameters()
        _, tangent = moist_source_jvp(
            state, direction, fields, parameters
        )
        pullback = moist_source_vjp(
            state, covector, fields, parameters
        )
        left = _tree_dot(covector, tangent)
        right = _tree_dot(pullback, direction)
        np.testing.assert_allclose(left, right, rtol=512.0 * EPS, atol=EPS)

    @pytest.mark.parametrize("shape", ((), (2,)))
    def test_vjp_matches_tiny_dense_jacobian_transpose(self, shape):
        state = _state(qv=0.003, qc=0.001, shape=shape)
        covector = _source_covector(shape=shape)
        fields = _fields(shape=shape)
        parameters = _parameters()
        actual = moist_source_vjp(state, covector, fields, parameters)
        vector, _, source_vector = _dense_maps(state, fields, parameters)
        covector_vector, _ = ravel_pytree(covector)
        expected = jax.jacfwd(source_vector)(vector).T @ covector_vector
        actual_vector, _ = ravel_pytree(actual)
        np.testing.assert_allclose(
            actual_vector, expected, rtol=512.0 * EPS, atol=512.0 * EPS
        )

    def test_differentiated_vjp_matches_centered_vjp_differences(self):
        state = _state(qv=0.001, qc=0.001, shape=(2,))
        direction = _direction(shape=(2,))
        covector = _source_covector(shape=(2,))
        dcovector = _source_covector(shape=(2,), scale=-0.37)
        fields = _fields(shape=(2,))
        parameters = _parameters()
        _, exact = moist_source_differentiated_vjp(
            state,
            covector,
            direction,
            dcovector,
            fields,
            parameters,
        )
        base_active_set = _pure_active_set(state, fields, parameters)
        errors = []
        for epsilon in _fd_ladder():
            plus_state = _tree_axpy(state, epsilon, direction)
            minus_state = _tree_axpy(state, -epsilon, direction)
            plus = moist_source_vjp(
                plus_state,
                _tree_axpy(covector, epsilon, dcovector),
                fields,
                parameters,
            )
            minus = moist_source_vjp(
                minus_state,
                _tree_axpy(covector, -epsilon, dcovector),
                fields,
                parameters,
            )
            _assert_pure_active_sets_stable(
                base_active_set,
                _pure_active_set(plus_state, fields, parameters),
                _pure_active_set(minus_state, fields, parameters),
            )
            errors.append(
                _tree_relative_error(
                    _tree_centered(plus, minus, epsilon), exact
                )
            )
        assert min(errors) < 2.0e-8, errors
        assert min(errors[2:8]) < errors[0], errors

    def test_differentiated_vjp_matches_tiny_dense_hessian_action(self):
        state = _state(qv=0.001, qc=0.0002)
        direction = _direction()
        covector = _source_covector()
        dcovector = _source_covector(scale=0.21)
        fields = _fields()
        parameters = _parameters()
        ordinary, actual = moist_source_differentiated_vjp(
            state,
            covector,
            direction,
            dcovector,
            fields,
            parameters,
        )
        vector, _, source_vector = _dense_maps(state, fields, parameters)
        direction_vector, _ = ravel_pytree(direction)
        covector_vector, _ = ravel_pytree(covector)
        dcovector_vector, _ = ravel_pytree(dcovector)
        jacobian = jax.jacfwd(source_vector)(vector)
        hessian = jax.hessian(
            lambda active: jnp.vdot(covector_vector, source_vector(active))
        )(vector)
        expected_ordinary = jacobian.T @ covector_vector
        expected_incremental = (
            jacobian.T @ dcovector_vector + hessian @ direction_vector
        )
        ordinary_vector, _ = ravel_pytree(ordinary)
        actual_vector, _ = ravel_pytree(actual)
        np.testing.assert_allclose(
            ordinary_vector,
            expected_ordinary,
            rtol=1024.0 * EPS,
            atol=1024.0 * EPS,
        )
        np.testing.assert_allclose(
            actual_vector,
            expected_incremental,
            rtol=2048.0 * EPS,
            atol=2048.0 * EPS,
        )

    def test_bilinear_hessian_symmetry_away_from_switches(self):
        state = _state(qv=0.001, qc=0.0002)
        fields = _fields()
        parameters = _parameters()
        covector = _source_covector()
        vector, _, source_vector = _dense_maps(state, fields, parameters)
        covector_vector, _ = ravel_pytree(covector)
        left_direction, _ = ravel_pytree(_direction(scale=0.7))
        right_direction, _ = ravel_pytree(
            {
                "h": np.asarray(-0.12),
                "S": np.asarray(0.83),
                "Qv": np.asarray(0.009),
                "Qc": np.asarray(0.014),
            }
        )
        hessian = jax.hessian(
            lambda active: jnp.vdot(covector_vector, source_vector(active))
        )(vector)
        left = left_direction @ hessian @ right_direction
        right = right_direction @ hessian @ left_direction
        np.testing.assert_allclose(left, right, rtol=2048.0 * EPS, atol=EPS)

    def test_primal_jvp_and_second_order_invariant_identities(self):
        state = _state(qv=0.001, qc=0.0002, shape=(3,))
        first_direction = _direction(shape=(3,))
        second_direction = _direction(shape=(3,), scale=-0.43)
        fields = _fields(shape=(3,))
        parameters = _parameters()
        beta2 = parameters["g"] * parameters["L"]
        source, tangent = moist_source_jvp(
            state, first_direction, fields, parameters
        )

        def first_derivative(active_state):
            return moist_source_jvp(
                active_state, first_direction, fields, parameters
            )[1]

        _, second = jax.jvp(
            first_derivative, (state,), (second_direction,)
        )
        source_scale = max(
            *(float(np.max(np.abs(np.asarray(leaf))))
              for leaf in source.values()),
            np.finfo(float).tiny,
        )
        for value in (source, tangent, second):
            water = value["Qv"] + value["Qc"] + value["Qr"]
            thermal = value["S"] - beta2 * value["Qv"]
            assert float(np.max(np.abs(np.asarray(water)))) <= (
                64.0 * EPS * source_scale
            )
            assert float(np.max(np.abs(np.asarray(thermal)))) <= 64.0 * EPS * max(
                source_scale, float(beta2) * source_scale
            )

    def test_invariant_covectors_have_zero_vjp_and_differentiated_vjp(self):
        state = _state(qv=0.001, qc=0.001, shape=(2,))
        fields = _fields(shape=(2,))
        parameters = _parameters()
        zero = np.zeros((2,), dtype=np.float64)
        one = np.ones((2,), dtype=np.float64)
        beta2 = float(parameters["g"] * parameters["L"])
        invariant_covectors = (
            {"S": zero, "Qv": one, "Qc": one, "Qr": one},
            {"S": one, "Qv": -beta2 * one, "Qc": zero, "Qr": zero},
        )
        for covector in invariant_covectors:
            ordinary, incremental = moist_source_differentiated_vjp(
                state,
                covector,
                _direction(shape=(2,)),
                {key: np.zeros((2,), dtype=np.float64) for key in SOURCE_KEYS},
                fields,
                parameters,
            )
            ordinary_vector, _ = ravel_pytree(ordinary)
            incremental_vector, _ = ravel_pytree(incremental)
            assert float(jnp.linalg.norm(ordinary_vector)) <= 128.0 * EPS
            assert float(jnp.linalg.norm(incremental_vector)) <= 128.0 * EPS

    def test_equality_switches_are_characterized_not_classically_certified(self):
        state = _state(qv=0.002, qc=0.0001)
        fields = _fields()
        parameters = _parameters()
        diagnostics = moist_diagnostics_jax(state, fields, parameters)
        source, tangent = moist_source_jvp(
            state, _direction(), fields, parameters
        )
        _, incremental = moist_source_differentiated_vjp(
            state,
            _source_covector(),
            _direction(),
            _source_covector(scale=0.0),
            fields,
            parameters,
        )
        assert not bool(diagnostics["condensation_mask"])
        assert not bool(diagnostics["evaporation_mask"])
        assert not bool(diagnostics["rain_mask"])
        assert all(
            np.all(np.isfinite(np.asarray(leaf)))
            for tree in (source, tangent, incremental)
            for leaf in tree.values()
        )

    def test_differentiated_pytrees_contain_no_boolean_diagnostic_leaf(self):
        state = _state(qv=0.001, qc=0.0002, shape=(2,))
        fields = _fields(shape=(2,))
        parameters = _parameters()
        outputs = (
            moist_source_jvp(state, _direction(shape=(2,)), fields, parameters),
            moist_source_differentiated_vjp(
                state,
                _source_covector(shape=(2,)),
                _direction(shape=(2,)),
                _source_covector(shape=(2,), scale=0.2),
                fields,
                parameters,
            ),
        )
        for output in outputs:
            for leaf in jax.tree.leaves(output):
                assert jnp.asarray(leaf).dtype == jnp.float64

    def test_x64_only_and_inputs_are_immutable(self):
        assert bool(jax.config.read("jax_enable_x64"))
        state = _state(qv=0.001, qc=0.0002, shape=(2,))
        direction = _direction(shape=(2,))
        covector = _source_covector(shape=(2,))
        dcovector = _source_covector(shape=(2,), scale=-0.4)
        fields = _fields(shape=(2,))
        parameters = _parameters()
        snapshots = copy.deepcopy(
            (state, direction, covector, dcovector, fields, parameters)
        )
        moist_source_jvp_jit(state, direction, fields, parameters)
        moist_source_vjp_jit(state, covector, fields, parameters)
        moist_source_differentiated_vjp_jit(
            state,
            covector,
            direction,
            dcovector,
            fields,
            parameters,
        )
        for actual_tree, snapshot_tree in zip(
            (state, direction, covector, dcovector, fields, parameters),
            snapshots,
        ):
            for key in actual_tree:
                np.testing.assert_array_equal(
                    actual_tree[key], snapshot_tree[key]
                )
        float32_direction = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in direction.items()
        }
        with pytest.raises(TypeError, match="must have dtype float64"):
            moist_source_jvp(state, float32_direction, fields, parameters)

    def test_local_only_parameter_second_order_blocks(self):
        """Dense tiny proof only; parameters remain fixed in the J2 helper."""
        state = _state(qv=0.001, qc=0.0002)
        parameters = _parameters()
        fields = _fields()
        state_vector, unravel_state = ravel_pytree(state)
        parameter_vector, unravel_parameters = ravel_pytree(parameters)
        weights, _ = ravel_pytree(_source_covector())

        def objective(active_state, active_parameters):
            source = moist_source_density_jax(
                unravel_state(active_state),
                fields,
                unravel_parameters(active_parameters),
            )
            source_vector, _ = ravel_pytree(source)
            return jnp.vdot(weights, source_vector)

        state_parameter = jax.jacfwd(
            jax.grad(objective, argnums=0), argnums=1
        )(state_vector, parameter_vector)
        parameter_state = jax.jacfwd(
            jax.grad(objective, argnums=1), argnums=0
        )(state_vector, parameter_vector)
        parameter_parameter = jax.jacfwd(
            jax.grad(objective, argnums=1), argnums=1
        )(state_vector, parameter_vector)
        assert state_parameter.shape == (
            state_vector.size,
            parameter_vector.size,
        )
        assert parameter_parameter.shape == (
            parameter_vector.size,
            parameter_vector.size,
        )
        np.testing.assert_allclose(
            state_parameter,
            parameter_state.T,
            rtol=4096.0 * EPS,
            atol=4096.0 * EPS,
        )
        assert np.all(np.isfinite(np.asarray(parameter_parameter)))
        assert float(jnp.linalg.norm(parameter_parameter)) > 0.0


def _build_firedrake_case(configured_dt):
    """Construct a tiny fixed-parameter serial oracle/JAX comparison case."""
    from copy import deepcopy

    firedrake = pytest.importorskip(
        "firedrake", reason="Firedrake is required for J2 external parity"
    )
    from firedrake import COMM_SELF, SpatialCoordinate, as_vector, cos, pi, sin

    import dimswe.meshes as dimswe_meshes
    from dimswe.jax_moist_hvp import JAXMoistEulerHVP
    from dimswe.logger import EmptyLogger
    from dimswe.models import get_model
    from dimswe.mtswe_split_hvp import ProductionMoistEulerHVP
    from dimswe.parameters import get_parameters, overall_solver_parameters
    from dimswe.timestepping import Euler

    parameters = get_parameters("tests/mtswe_small.cfg")
    parameters["mesh"]["type"] = "rectangle"
    parameters["mesh"]["nx"] = 2
    parameters["mesh"]["ny"] = 2
    parameters["timestepping"]["dt"] = float(configured_dt)
    parameters["threewayphysics"]["treat_as_coeffs"] = False
    parameters["hyperviscosity"]["treat_as_coeffs"] = False
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
    assert model.mesh.comm.size == 1

    coefficient, coefficient_sub, _ = model.get_coeff_var(
        f"jax_moist_j2_coefficient_{configured_dt}"
    )
    state_container, state_sub, _ = model.get_full_var(
        f"jax_moist_j2_state_{configured_dt}", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)

    x = SpatialCoordinate(model.mesh)
    lx = model.initcond.Lx
    ly = model.initcond.Ly
    mode_x = sin(2.0 * pi * x[0] / lx)
    mode_y = cos(2.0 * pi * x[1] / ly)
    height = 750.0 + 2.0 * mode_x + 1.5 * mode_y
    state_sub["v"].project(
        as_vector([2.0 + 0.2 * mode_y, -1.0 + 0.1 * mode_x])
    )
    state_sub["h"].project(height)
    state_sub["S"].project(
        height
        * model.initcond.g
        * (1.01 + 0.0007 * mode_x - 0.0004 * mode_y)
    )
    state_sub["Qv"].project(0.0030 * height)
    state_sub["Qc"].project(0.0010 * height)
    state_sub["Qr"].project(0.0002 * height)

    solver_parameters = deepcopy(overall_solver_parameters)
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    for name in (
        "erkstage-f",
        "erkstage-aux",
        "erkstage-mu",
        "erkstage-muaux",
        "erk-dlambda",
        "erk-grad",
    ):
        solver_parameters[name] = direct
    timestepper = Euler(
        model, logger, solver_parameters, terms=["threewayphysics"]
    )
    production = ProductionMoistEulerHVP(timestepper)
    jax_helper = JAXMoistEulerHVP(timestepper)

    directions = []
    for index, sign in enumerate((1.0, -0.7)):
        direction = model.get_x_var(
            f"jax_moist_j2_direction_{configured_dt}_{index}"
        )[0]
        direction.sub(0).project(
            as_vector([sign * 0.12 * mode_x, -sign * 0.08 * mode_y])
        )
        direction.sub(1).project(sign * (0.14 * mode_y - 0.05 * mode_x))
        direction.sub(2).project(sign * (1.1 * mode_x + 0.6 * mode_y))
        direction.sub(3).project(
            sign * 7.0e-6 * height * (1.0 + 0.1 * mode_x)
        )
        direction.sub(4).project(
            -sign * 5.0e-6 * height * (1.0 - 0.1 * mode_y)
        )
        direction.sub(5).project(sign * 3.0e-6 * height * mode_y)
        directions.append(direction)

    probes = []
    for index, sign in enumerate((1.0, -0.6)):
        probe = model.get_x_var(
            f"jax_moist_j2_probe_{configured_dt}_{index}"
        )[0]
        probe.sub(0).project(
            as_vector([-sign * 0.09 * mode_y, sign * 0.11 * mode_x])
        )
        probe.sub(1).project(sign * 0.07 * mode_x)
        probe.sub(2).project(-sign * 0.8 * mode_y)
        probe.sub(3).project(sign * 4.0e-6 * height * mode_y)
        probe.sub(4).project(-sign * 6.0e-6 * height * mode_x)
        probe.sub(5).project(sign * 5.0e-6 * height * (mode_x + mode_y))
        probes.append(probe)
    incoming = tuple(jax_helper.state_mass_map(probe) for probe in probes)

    return {
        "firedrake": firedrake,
        "parameters": parameters,
        "model": model,
        "state": state_container[0],
        "state_sub": state_sub,
        "time": float(time),
        "configured_dt": float(configured_dt),
        "applied_dt": 37.0,
        "timestepper": timestepper,
        "production": production,
        "jax": jax_helper,
        "directions": tuple(directions),
        "probes": tuple(probes),
        "incoming": incoming,
    }


@pytest.fixture(scope="module")
def firedrake_cases():
    return {
        "base": _build_firedrake_case(100.0),
        "configured_alt": _build_firedrake_case(250.0),
    }


def _fd_values(value):
    with value.dat.vec_ro as vector:
        return np.array(vector.array_r, dtype=np.float64, copy=True)


def _fd_function_axpy(base, scale, increment, name):
    result = base.copy(deepcopy=True)
    result.rename(name)
    with result.dat.vec as result_vec, increment.dat.vec_ro as increment_vec:
        result_vec.axpy(float(scale), increment_vec)
    return result


def _fd_dual_axpy(base, scale, increment, name):
    result = base.copy(deepcopy=True)
    result.rename(name)
    with result.dat.vec as result_vec, increment.dat.vec_ro as increment_vec:
        result_vec.axpy(float(scale), increment_vec)
    return result


def _fd_function_difference(left, right, name):
    return _fd_function_axpy(left, -1.0, right, name)


def _fd_dual_difference(left, right, name):
    return _fd_dual_axpy(left, -1.0, right, name)


def _fd_function_error(actual, expected, name):
    from firedrake import norm

    difference = _fd_function_difference(actual, expected, name)
    absolute = float(norm(difference))
    scale = float(norm(expected))
    return absolute, absolute / max(scale, np.finfo(float).tiny), scale


def _fd_dual_norm(helper, value, name):
    representative = helper.state_riesz_representative(value, name)
    squared = helper.dual_pairing(value, representative)
    assert squared >= -1024.0 * EPS * max(1.0, abs(squared))
    return float(np.sqrt(max(0.0, squared)))


def _fd_dual_error(helper, actual, expected, name):
    difference = _fd_dual_difference(actual, expected, name)
    absolute = _fd_dual_norm(helper, difference, f"{name}_riesz")
    scale = _fd_dual_norm(helper, expected, f"{name}_scale_riesz")
    return absolute, absolute / max(scale, np.finfo(float).tiny), scale


def _fd_isolated_dual_block(helper, value, block, name):
    from firedrake import Cofunction

    isolated = Cofunction(helper.state_dual_space, name=name)
    isolated.zero()
    isolated.sub(block).assign(value.sub(block))
    return isolated


def _assert_function_close(actual, expected, name, *, rtol=5.0e-11):
    absolute, relative, scale = _fd_function_error(actual, expected, name)
    assert absolute <= 1.0e-12 + rtol * scale, (
        f"{name}: absolute={absolute:.17g}, relative={relative:.17g}, "
        f"scale={scale:.17g}"
    )
    return absolute, relative


def _assert_dual_close(helper, actual, expected, name, *, rtol=1.0e-10):
    absolute, relative, scale = _fd_dual_error(
        helper, actual, expected, name
    )
    assert absolute <= 1.0e-12 + rtol * scale, (
        f"{name}: natural_absolute={absolute:.17g}, "
        f"natural_relative={relative:.17g}, scale={scale:.17g}"
    )
    return absolute, relative


def _assert_gll_separated(primal):
    pairs = (
        ("condensation_margin", "condensation_argument"),
        ("evaporation_margin", "evaporation_argument"),
        ("evaporation_cap_margin", "evaporation_cap_difference"),
        ("rain_margin", "rain_argument"),
        ("depth_denominator_margin", "depth_denominator"),
    )
    for margin_name, value_name in pairs:
        margin = float(primal.gll_active_set.margins[margin_name])
        values = np.asarray(primal.gll_diagnostics[value_name])
        scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        assert margin / scale > 1.0e-5, (
            margin_name,
            margin,
            scale,
            primal.gll_active_set.signature,
        )


def _assert_active_sets_stable(base, *candidates):
    _assert_gll_separated(base)
    for candidate in candidates:
        _assert_gll_separated(candidate)
        assert candidate.gll_active_set.signature == base.gll_active_set.signature
        assert (
            candidate.legacy_active_set.signature
            == base.legacy_active_set.signature
        )


class TestFiredrakeOperatorTransposes:
    def test_state_interpolation_P_and_Pstar_pairing(self, firedrake_cases):
        case = firedrake_cases["base"]
        helper = case["jax"]
        direction = case["directions"][0]
        packed_direction = helper.state_interpolation(direction)
        rng = np.random.default_rng(20260806)
        packed_covector = {
            key: rng.standard_normal(value.shape).astype(np.float64)
            for key, value in packed_direction.items()
        }
        transpose = helper.state_interpolation_transpose(packed_covector)
        left = sum(
            float(np.vdot(packed_covector[key], packed_direction[key]))
            for key in STATE_KEYS
        )
        right = helper.dual_pairing(transpose, direction)
        absolute = abs(left - right)
        relative = absolute / max(abs(left), abs(right), np.finfo(float).tiny)
        repeated = helper.state_interpolation(direction)
        maximum_packed_discrepancy = max(
            float(np.max(np.abs(packed_direction[key] - repeated[key])))
            for key in STATE_KEYS
        )
        block_norms = {
            key: float(np.linalg.norm(packed_direction[key]))
            for key in STATE_KEYS
        }
        print(
            {
                "operator": "P/P*",
                "left_pairing": left,
                "right_pairing": right,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "block_norms": block_norms,
                "maximum_packed_discrepancy": maximum_packed_discrepancy,
                "owned_cells": helper.layout.owned_cell_count,
                "points_per_cell": helper.layout.points_per_cell,
                "field_order": helper.layout.field_order,
                "reference_points": helper.layout.reference_points.tolist(),
            }
        )
        assert absolute <= 8192.0 * EPS * max(abs(left), abs(right), 1.0)
        np.testing.assert_array_equal(_fd_values(transpose.sub(0)), 0.0)
        np.testing.assert_array_equal(_fd_values(transpose.sub(5)), 0.0)

    def test_source_assembly_A_and_Astar_pairing(self, firedrake_cases):
        case = firedrake_cases["base"]
        helper = case["jax"]
        rng = np.random.default_rng(20260807)
        shape = (
            helper.layout.owned_cell_count,
            helper.layout.points_per_cell,
        )
        source = {
            key: rng.standard_normal(shape).astype(np.float64)
            for key in SOURCE_KEYS
        }
        psi = case["probes"][0]
        assembled = helper.source_assembly(source)
        transpose = helper.source_assembly_transpose(psi)
        left = helper.dual_pairing(assembled, psi)
        right = sum(
            float(np.vdot(source[key], transpose[key]))
            for key in SOURCE_KEYS
        )
        absolute = abs(left - right)
        relative = absolute / max(abs(left), abs(right), np.finfo(float).tiny)
        maximum_packed_discrepancy = 0.0
        for key in SOURCE_KEYS:
            carrier = helper.primal_helper.unpack_carrier(
                source[key], f"jax_moist_j2_A_roundtrip_{key}"
            )
            maximum_packed_discrepancy = max(
                maximum_packed_discrepancy,
                float(
                    np.max(
                        np.abs(
                            source[key]
                            - helper.primal_helper.pack_carrier(carrier)
                        )
                    )
                ),
            )
        block_norms = {
            key: float(np.linalg.norm(transpose[key]))
            for key in SOURCE_KEYS
        }
        print(
            {
                "operator": "A/A*",
                "left_pairing": left,
                "right_pairing": right,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "block_norms": block_norms,
                "maximum_packed_discrepancy": maximum_packed_discrepancy,
                "owned_cells": helper.layout.owned_cell_count,
                "points_per_cell": helper.layout.points_per_cell,
                "field_order": SOURCE_KEYS,
                "cell_nodes": helper.layout.cell_nodes.tolist(),
            }
        )
        assert absolute <= 8192.0 * EPS * max(abs(left), abs(right), 1.0)


class TestFiredrakeJAXMoistTangent:
    def test_jax_moist_primal_cache_matches_production_oracle(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        jax_primal = case["jax"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        production_primal = case["production"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        _assert_function_close(
            jax_primal.stage_state,
            production_primal.stage_state,
            "primal_stage_state",
        )
        _assert_function_close(
            jax_primal.tendency,
            production_primal.tendency,
            "primal_tendency",
        )
        _assert_function_close(
            jax_primal.state_out,
            production_primal.state_out,
            "primal_state_out",
        )
        assert (
            jax_primal.legacy_active_set.signature
            == production_primal.active_set.signature
        )
        for margin_name in (
            "condensation_margin",
            "evaporation_margin",
            "evaporation_cap_margin",
            "rain_margin",
            "depth_denominator_margin",
        ):
            np.testing.assert_allclose(
                float(jax_primal.legacy_active_set.margins[margin_name]),
                float(getattr(production_primal.active_set, margin_name)),
                rtol=4096.0 * EPS,
                atol=4096.0 * EPS,
            )
        _assert_gll_separated(jax_primal)

    def test_jax_moist_tangent_matches_production_oracle(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        jax_primal = case["jax"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        production_primal = case["production"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        discrepancies = []
        for index, direction in enumerate(case["directions"]):
            actual = case["jax"].take_tangent_step(jax_primal, direction)
            expected = case["production"].take_tangent_step(
                production_primal, direction
            )
            discrepancies.append(
                _assert_function_close(
                    actual.tendency_direction,
                    expected.tendency_direction,
                    f"tangent_tendency_{index}",
                )
            )
            discrepancies.append(
                _assert_function_close(
                    actual.state_direction_out,
                    expected.state_direction_out,
                    f"tangent_output_{index}",
                )
            )
            for block in range(6):
                _assert_function_close(
                    actual.tendency_direction.sub(block),
                    expected.tendency_direction.sub(block),
                    f"tangent_tendency_{index}_block_{block}",
                )
        print({"tangent_discrepancies": discrepancies})

    def test_jax_moist_tangent_centered_primal_fd_with_gll_active_set(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        helper = case["jax"]
        direction = case["directions"][0]
        base = helper.take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        exact = helper.take_tangent_step(base, direction).state_direction_out
        errors = []
        for epsilon in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            plus_state = _fd_function_axpy(
                case["state"], epsilon, direction, "jax_moist_j2_fd_plus"
            )
            minus_state = _fd_function_axpy(
                case["state"], -epsilon, direction, "jax_moist_j2_fd_minus"
            )
            plus = helper.take_forward_step_cached(
                plus_state, case["time"], case["applied_dt"]
            )
            minus = helper.take_forward_step_cached(
                minus_state, case["time"], case["applied_dt"]
            )
            _assert_active_sets_stable(base, plus, minus)
            centered = plus.state_out.copy(deepcopy=True)
            with centered.dat.vec as centered_vec, minus.state_out.dat.vec_ro as minus_vec:
                centered_vec.axpy(-1.0, minus_vec)
                centered_vec.scale(1.0 / (2.0 * epsilon))
            errors.append(_fd_function_error(centered, exact, "tangent_fd")[1])
        assert min(errors) < 5.0e-8, errors


class TestFiredrakeJAXMoistReverse:
    def test_jax_moist_reverse_matches_production_oracle(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        jax_primal = case["jax"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        production_primal = case["production"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        discrepancies = []
        for index, incoming in enumerate(case["incoming"]):
            actual = case["jax"].take_adjoint_step_cached(
                jax_primal, incoming
            )
            expected = case["production"].take_adjoint_step_cached(
                production_primal, incoming
            )
            discrepancies.append(
                _assert_function_close(
                    actual.reverse_auxiliary,
                    expected.reverse_auxiliary,
                    f"reverse_psi_{index}",
                )
            )
            discrepancies.append(
                _assert_dual_close(
                    case["jax"],
                    actual.stage_state_adjoint,
                    expected.stage_state_adjoint,
                    f"reverse_stage_pullback_{index}",
                )
            )
            discrepancies.append(
                _assert_dual_close(
                    case["jax"],
                    actual.state_adjoint_in,
                    expected.state_adjoint_in,
                    f"reverse_state_adjoint_{index}",
                )
            )
            for block in range(6):
                actual_block = _fd_isolated_dual_block(
                    case["jax"],
                    actual.state_adjoint_in,
                    block,
                    f"reverse_actual_{index}_{block}",
                )
                expected_block = _fd_isolated_dual_block(
                    case["jax"],
                    expected.state_adjoint_in,
                    block,
                    f"reverse_expected_{index}_{block}",
                )
                _assert_dual_close(
                    case["jax"],
                    actual_block,
                    expected_block,
                    f"reverse_state_adjoint_{index}_block_{block}",
                )
        print({"reverse_discrepancies": discrepancies})

    def test_complete_euler_jacobian_natural_pairing(self, firedrake_cases):
        case = firedrake_cases["base"]
        primal = case["jax"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        for direction, incoming in zip(case["directions"], case["incoming"]):
            tangent = case["jax"].take_tangent_step(primal, direction)
            reverse = case["jax"].take_adjoint_step_cached(primal, incoming)
            left = case["jax"].dual_pairing(
                incoming, tangent.state_direction_out
            )
            right = case["jax"].dual_pairing(
                reverse.state_adjoint_in, direction
            )
            np.testing.assert_allclose(
                left, right, rtol=16384.0 * EPS, atol=16384.0 * EPS
            )


class TestFiredrakeJAXMoistIncrementalReverse:
    def test_jax_moist_incremental_reverse_matches_production_oracle(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        jax_primal = case["jax"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        production_primal = case["production"].take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        discrepancies = []
        for direction_index, direction in enumerate(case["directions"]):
            jax_tangent = case["jax"].take_tangent_step(
                jax_primal, direction
            )
            production_tangent = case["production"].take_tangent_step(
                production_primal, direction
            )
            for incoming_index, (incoming, incremental_incoming) in enumerate(
                zip(case["incoming"], reversed(case["incoming"]))
            ):
                actual = case["jax"].take_incremental_adjoint_step(
                    jax_tangent, incoming, incremental_incoming
                )
                expected = case["production"].take_incremental_adjoint_step(
                    production_tangent, incoming, incremental_incoming
                )
                label = f"{direction_index}_{incoming_index}"
                discrepancies.append(
                    _assert_function_close(
                        actual.incremental_reverse_auxiliary,
                        expected.incremental_reverse_auxiliary,
                        f"incremental_psi_{label}",
                    )
                )
                discrepancies.append(
                    _assert_dual_close(
                        case["jax"],
                        actual.incremental_stage_state_adjoint,
                        expected.incremental_stage_state_adjoint,
                        f"incremental_stage_pullback_{label}",
                        rtol=2.0e-9,
                    )
                )
                discrepancies.append(
                    _assert_dual_close(
                        case["jax"],
                        actual.incremental_state_adjoint_in,
                        expected.incremental_state_adjoint_in,
                        f"incremental_state_adjoint_{label}",
                        rtol=2.0e-9,
                    )
                )
        print({"incremental_reverse_discrepancies": discrepancies})

    def test_incremental_reverse_centered_reverse_fd_and_active_set(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        helper = case["jax"]
        direction = case["directions"][0]
        incoming = case["incoming"][0]
        incremental_incoming = case["incoming"][1]
        base = helper.take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        tangent = helper.take_tangent_step(base, direction)
        exact = helper.take_incremental_adjoint_step(
            tangent, incoming, incremental_incoming
        ).incremental_state_adjoint_in
        errors = []
        for epsilon in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            plus_state = _fd_function_axpy(
                case["state"], epsilon, direction, "reverse_fd_state_plus"
            )
            minus_state = _fd_function_axpy(
                case["state"], -epsilon, direction, "reverse_fd_state_minus"
            )
            plus_incoming = _fd_dual_axpy(
                incoming,
                epsilon,
                incremental_incoming,
                "reverse_fd_incoming_plus",
            )
            minus_incoming = _fd_dual_axpy(
                incoming,
                -epsilon,
                incremental_incoming,
                "reverse_fd_incoming_minus",
            )
            plus_primal = helper.take_forward_step_cached(
                plus_state, case["time"], case["applied_dt"]
            )
            minus_primal = helper.take_forward_step_cached(
                minus_state, case["time"], case["applied_dt"]
            )
            _assert_active_sets_stable(base, plus_primal, minus_primal)
            plus = helper.take_adjoint_step_cached(
                plus_primal, plus_incoming
            ).state_adjoint_in
            minus = helper.take_adjoint_step_cached(
                minus_primal, minus_incoming
            ).state_adjoint_in
            centered = plus.copy(deepcopy=True)
            with centered.dat.vec as centered_vec, minus.dat.vec_ro as minus_vec:
                centered_vec.axpy(-1.0, minus_vec)
                centered_vec.scale(1.0 / (2.0 * epsilon))
            errors.append(
                _fd_dual_error(
                    helper, centered, exact, "incremental_reverse_fd"
                )[1]
            )
        assert min(errors) < 2.0e-7, errors

    def test_hessian_natural_bilinear_symmetry(self, firedrake_cases):
        case = firedrake_cases["base"]
        helper = case["jax"]
        primal = helper.take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        zero = case["incoming"][0].copy(deepcopy=True)
        zero.zero()
        tangent_left = helper.take_tangent_step(
            primal, case["directions"][0]
        )
        tangent_right = helper.take_tangent_step(
            primal, case["directions"][1]
        )
        hessian_left = helper.take_incremental_adjoint_step(
            tangent_left, case["incoming"][0], zero
        ).incremental_state_adjoint_in
        hessian_right = helper.take_incremental_adjoint_step(
            tangent_right, case["incoming"][0], zero
        ).incremental_state_adjoint_in
        left = helper.dual_pairing(hessian_left, case["directions"][1])
        right = helper.dual_pairing(hessian_right, case["directions"][0])
        np.testing.assert_allclose(left, right, rtol=2.0e-10, atol=1.0e-11)


class TestFiredrakeJAXMoistContracts:
    def test_zero_incoming_structural_cases(self, firedrake_cases):
        case = firedrake_cases["base"]
        helper = case["jax"]
        primal = helper.take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        zero_dual = case["incoming"][0].copy(deepcopy=True)
        zero_dual.zero()
        zero_direction = case["directions"][0].copy(deepcopy=True)
        zero_direction.assign(0.0)
        reverse_zero = helper.take_adjoint_step_cached(primal, zero_dual)
        np.testing.assert_array_equal(
            _fd_values(reverse_zero.stage_state_adjoint), 0.0
        )
        np.testing.assert_array_equal(
            _fd_values(reverse_zero.state_adjoint_in), 0.0
        )
        zero_tangent = helper.take_tangent_step(primal, zero_direction)
        incremental_zero = helper.take_incremental_adjoint_step(
            zero_tangent, case["incoming"][0], zero_dual
        )
        np.testing.assert_array_equal(
            _fd_values(incremental_zero.incremental_state_adjoint_in), 0.0
        )
        nonzero_mu = helper.take_incremental_adjoint_step(
            zero_tangent, zero_dual, case["incoming"][1]
        )
        reverse_mu = helper.take_adjoint_step_cached(
            primal, case["incoming"][1]
        )
        _assert_dual_close(
            helper,
            nonzero_mu.incremental_state_adjoint_in,
            reverse_mu.state_adjoint_in,
            "zero_lambda_incremental_equals_reverse_mu",
        )

    def test_applied_and_configured_timestep_scaling(self, firedrake_cases):
        base_case = firedrake_cases["base"]
        alt_case = firedrake_cases["configured_alt"]
        direction_base = base_case["directions"][0]
        direction_alt = alt_case["directions"][0]

        base_17 = base_case["jax"].take_forward_step_cached(
            base_case["state"], base_case["time"], 17.0
        )
        base_53 = base_case["jax"].take_forward_step_cached(
            base_case["state"], base_case["time"], 53.0
        )
        _assert_active_sets_stable(base_17, base_53)
        tangent_17 = base_case["jax"].take_tangent_step(
            base_17, direction_base
        )
        tangent_53 = base_case["jax"].take_tangent_step(
            base_53, direction_base
        )
        np.testing.assert_allclose(
            _fd_values(tangent_17.tendency_direction),
            _fd_values(tangent_53.tendency_direction),
            rtol=2.0e-13,
            atol=1.0e-13,
        )
        increment_17 = (
            _fd_values(tangent_17.state_direction_out)
            - _fd_values(direction_base)
        )
        increment_53 = (
            _fd_values(tangent_53.state_direction_out)
            - _fd_values(direction_base)
        )
        np.testing.assert_allclose(
            increment_53,
            (53.0 / 17.0) * increment_17,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

        alt_primal = alt_case["jax"].take_forward_step_cached(
            alt_case["state"], alt_case["time"], 17.0
        )
        _assert_gll_separated(alt_primal)
        alt_tangent = alt_case["jax"].take_tangent_step(
            alt_primal, direction_alt
        )
        expected_ratio = (
            base_case["configured_dt"] / alt_case["configured_dt"]
        )
        np.testing.assert_allclose(
            _fd_values(alt_tangent.tendency_direction),
            expected_ratio * _fd_values(tangent_17.tendency_direction),
            rtol=2.0e-10,
            atol=2.0e-12,
        )

        base_production_primal = base_case[
            "production"
        ].take_forward_step_cached(
            base_case["state"], base_case["time"], 17.0
        )
        alt_production_primal = alt_case[
            "production"
        ].take_forward_step_cached(
            alt_case["state"], alt_case["time"], 17.0
        )
        _assert_function_close(
            tangent_17.tendency_direction,
            base_case["production"].take_tangent_step(
                base_production_primal, direction_base
            ).tendency_direction,
            "configured_base_tangent_oracle",
        )
        _assert_function_close(
            alt_tangent.tendency_direction,
            alt_case["production"].take_tangent_step(
                alt_production_primal, direction_alt
            ).tendency_direction,
            "configured_alt_tangent_oracle",
        )

        reverse_17 = base_case["jax"].take_adjoint_step_cached(
            base_17, base_case["incoming"][0]
        )
        reverse_53 = base_case["jax"].take_adjoint_step_cached(
            base_53, base_case["incoming"][0]
        )
        np.testing.assert_allclose(
            _fd_values(reverse_53.stage_state_adjoint),
            (53.0 / 17.0) * _fd_values(reverse_17.stage_state_adjoint),
            rtol=2.0e-11,
            atol=2.0e-12,
        )
        reverse_alt = alt_case["jax"].take_adjoint_step_cached(
            alt_primal, alt_case["incoming"][0]
        )
        np.testing.assert_allclose(
            _fd_values(reverse_alt.stage_state_adjoint),
            expected_ratio * _fd_values(reverse_17.stage_state_adjoint),
            rtol=2.0e-10,
            atol=2.0e-12,
        )

        hvp_17 = base_case["jax"].take_incremental_adjoint_step(
            tangent_17,
            base_case["incoming"][0],
            base_case["incoming"][1],
        )
        hvp_53 = base_case["jax"].take_incremental_adjoint_step(
            tangent_53,
            base_case["incoming"][0],
            base_case["incoming"][1],
        )
        np.testing.assert_allclose(
            _fd_values(hvp_53.incremental_stage_state_adjoint),
            (53.0 / 17.0)
            * _fd_values(hvp_17.incremental_stage_state_adjoint),
            rtol=2.0e-9,
            atol=2.0e-11,
        )
        hvp_alt = alt_case["jax"].take_incremental_adjoint_step(
            alt_tangent,
            alt_case["incoming"][0],
            alt_case["incoming"][1],
        )
        np.testing.assert_allclose(
            _fd_values(hvp_alt.incremental_stage_state_adjoint),
            expected_ratio
            * _fd_values(hvp_17.incremental_stage_state_adjoint),
            rtol=2.0e-9,
            atol=2.0e-11,
        )

    def test_gll_sampling_detects_switch_missed_by_legacy_dg1(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        helper = case["jax"]
        candidate = case["state"].copy(deepcopy=True)
        candidate.sub(0).assign(0.0)
        candidate.sub(1).assign(750.0)
        candidate.sub(3).assign(750.0 * 0.00205)
        candidate.sub(4).assign(750.0 * 0.001)
        candidate.sub(5).assign(750.0 * 0.0002)
        state_space = candidate.sub(2).function_space()
        cell_nodes = np.asarray(state_space.cell_node_map().values, dtype=int)[
            : helper.layout.owned_cell_count
        ]
        xi = helper.layout.reference_points[:, 0]
        gauss_left = 0.5 * (1.0 - 1.0 / np.sqrt(3.0))
        gauss_right = 0.5 * (1.0 + 1.0 / np.sqrt(3.0))
        local_entropy = (
            750.0
            * case["model"].initcond.g
            * (1.0 + 0.2 * (xi - gauss_left) * (xi - gauss_right))
        )
        for nodes in cell_nodes:
            candidate.sub(2).dat.data[nodes] = local_entropy
        diagnostics = helper.active_set_diagnostics(candidate)
        legacy_condensation = np.asarray(
            diagnostics.legacy.masks["condensation_mask"]
        )
        gll_condensation = np.asarray(
            diagnostics.gll.masks["condensation_mask"]
        )
        assert legacy_condensation.size == 4 * helper.layout.owned_cell_count
        assert gll_condensation.size == 16 * helper.layout.owned_cell_count
        assert np.all(legacy_condensation)
        assert np.any(gll_condensation)
        assert np.any(~gll_condensation)

    def test_cache_ownership_repeatability_and_exception_recovery(
        self, firedrake_cases
    ):
        case = firedrake_cases["base"]
        helper = case["jax"]
        state_snapshot = _fd_values(case["state"])
        direction_snapshot = _fd_values(case["directions"][0])
        incoming_snapshot = _fd_values(case["incoming"][0])
        first = helper.take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        first_tangent = helper.take_tangent_step(
            first, case["directions"][0]
        )
        first_reverse = helper.take_adjoint_step_cached(
            first, case["incoming"][0]
        )
        first_hvp = helper.take_incremental_adjoint_step(
            first_tangent, case["incoming"][0], case["incoming"][1]
        )
        second = helper.take_forward_step_cached(
            case["state"], case["time"], case["applied_dt"]
        )
        second_tangent = helper.take_tangent_step(
            second, case["directions"][0]
        )
        second_reverse = helper.take_adjoint_step_cached(
            second, case["incoming"][0]
        )
        second_hvp = helper.take_incremental_adjoint_step(
            second_tangent, case["incoming"][0], case["incoming"][1]
        )
        np.testing.assert_array_equal(_fd_values(case["state"]), state_snapshot)
        np.testing.assert_array_equal(
            _fd_values(case["directions"][0]), direction_snapshot
        )
        np.testing.assert_array_equal(
            _fd_values(case["incoming"][0]), incoming_snapshot
        )
        np.testing.assert_array_equal(
            _fd_values(first.state_out), _fd_values(second.state_out)
        )
        np.testing.assert_array_equal(
            _fd_values(first_tangent.state_direction_out),
            _fd_values(second_tangent.state_direction_out),
        )
        np.testing.assert_array_equal(
            _fd_values(first_reverse.state_adjoint_in),
            _fd_values(second_reverse.state_adjoint_in),
        )
        np.testing.assert_array_equal(
            _fd_values(first_hvp.incremental_state_adjoint_in),
            _fd_values(second_hvp.incremental_state_adjoint_in),
        )
        for mapping in (
            first.packed_state,
            first.source_density,
            first_tangent.packed_state_direction,
            first_tangent.source_density_direction,
            first_reverse.source_covector,
            first_reverse.packed_state_covector,
            first_hvp.incremental_source_covector,
            first_hvp.incremental_packed_state_covector,
        ):
            assert all(not value.flags.writeable for value in mapping.values())
            assert all(not callable(value) for value in mapping.values())

        first.state_out.assign(0.0)
        first_reverse.state_adjoint_in.zero()
        np.testing.assert_array_equal(
            _fd_values(second.state_out),
            _fd_values(
                helper.take_forward_step_cached(
                    case["state"], case["time"], case["applied_dt"]
                ).state_out
            ),
        )
        np.testing.assert_array_equal(
            _fd_values(second_reverse.state_adjoint_in),
            _fd_values(
                helper.take_adjoint_step_cached(
                    second, case["incoming"][0]
                ).state_adjoint_in
            ),
        )

        original_source_assembly = helper.source_assembly

        def induced_failure(_):
            raise RuntimeError("induced J2 source assembly failure")

        helper.source_assembly = induced_failure
        try:
            with pytest.raises(
                RuntimeError, match="induced J2 source assembly failure"
            ):
                helper.take_tangent_step(second, case["directions"][0])
        finally:
            helper.source_assembly = original_source_assembly
        recovered = helper.take_tangent_step(second, case["directions"][0])
        np.testing.assert_array_equal(
            _fd_values(recovered.state_direction_out),
            _fd_values(second_tangent.state_direction_out),
        )
