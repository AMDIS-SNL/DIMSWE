"""Focused J3 parity tests for the opt-in complete-split JAX moist child."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


jax = pytest.importorskip("jax", reason="JAX is optional for DIMSWE")
jax.config.update("jax_enable_x64", True)

from firedrake import (  # noqa: E402
    COMM_SELF,
    Cofunction,
    SpatialCoordinate,
    as_vector,
    cos,
    norm,
    pi,
    sin,
)

import dimswe.meshes as dimswe_meshes  # noqa: E402
from dimswe.jax_moist_hvp import JAXMoistEulerHVP  # noqa: E402
from dimswe.logger import EmptyLogger  # noqa: E402
from dimswe.models import get_model  # noqa: E402
from dimswe.moist_backend import (  # noqa: E402
    JAXMoistEulerIntegrator,
    JAXMoistFixedControlHVPResult,
    JAXMoistFixedControlReverseResult,
)
from dimswe.mtswe_split_hvp import (  # noqa: E402
    MoistEulerPrimalCache,
    ProductionMoistEulerHVP,
)
from dimswe.parameters import get_parameters, overall_solver_parameters  # noqa: E402
from dimswe.timestepping import get_timestepper  # noqa: E402


CFG = "tests/mtswe_small.cfg"
PHYSICAL_C0 = 0.07
DELTA_C0 = 0.012
C0_SCALE = 0.07
EPS = np.finfo(np.float64).eps
RTOL = 16384.0 * EPS
ABS_FLOOR = 4096.0 * EPS
PRODUCTION_SYMMETRY_RTOL = 5.0e-10
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


def _values(value):
    with value.dat.vec_ro as vector:
        return np.array(vector.array_r, dtype=np.float64, copy=True)


def _copy(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _zero_state(case, name):
    result = case["model"].get_x_var(name)[0]
    result.assign(0.0)
    return result


def _set_c0(case, value=PHYSICAL_C0):
    case["coefficient_sub"]["c0"].assign(float(value))
    for split in case["splits"].values():
        split.set_coeff(case["coefficient"])


def _moist_cache(parent):
    matches = tuple(
        child for child in parent.children if child.name == "moist_euler"
    )
    assert len(matches) == 1
    return matches[0].cache


def _diagnostic_text(parent, *, timestep, child_index, field, backend):
    cache = _moist_cache(parent)
    if hasattr(cache, "gll_active_set"):
        legacy = cache.legacy_active_set
        gll = cache.gll_active_set
        configured_dt = cache.configured_dt
    else:
        legacy = cache.active_set
        gll = None
        configured_dt = "UFL ThreeWayPhysics.dt"
    child = parent.children[child_index]
    return (
        f"timestep={timestep}, child={child_index}/{child.name}, field={field}, "
        f"legacy_DG1_signature={legacy.signature}, "
        f"actual_GLL_signature={getattr(gll, 'signature', None)}, "
        f"GLL_minimum_switch_margins={getattr(gll, 'margins', None)}, "
        f"backend={backend}, configured_physics_dt={configured_dt}, "
        f"applied_child_dt={child.dt}"
    )


def _assert_function_parity(
    actual,
    reference,
    *,
    parent,
    timestep,
    child_index,
    label,
):
    assert actual.function_space() == reference.function_space()
    for block in range(6):
        difference = _copy(actual.sub(block), f"{label}_difference_{block}")
        difference.assign(actual.sub(block) - reference.sub(block))
        absolute = float(norm(difference))
        reference_norm = float(norm(reference.sub(block)))
        relative = absolute / max(reference_norm, np.finfo(float).tiny)
        context = _diagnostic_text(
            parent,
            timestep=timestep,
            child_index=child_index,
            field=f"block_{block}",
            backend="jax",
        )
        assert absolute <= ABS_FLOOR + RTOL * reference_norm, (
            f"{label}: absolute_error={absolute:.17g}, "
            f"relative_error={relative:.17g}, "
            f"reference_norm={reference_norm:.17g}; {context}"
        )
    difference = _copy(actual, f"{label}_mixed_difference")
    difference.assign(actual - reference)
    absolute = float(norm(difference))
    reference_norm = float(norm(reference))
    relative = absolute / max(reference_norm, np.finfo(float).tiny)
    context = _diagnostic_text(
        parent,
        timestep=timestep,
        child_index=child_index,
        field="complete_mixed",
        backend="jax",
    )
    assert absolute <= ABS_FLOOR + RTOL * reference_norm, (
        f"{label}: absolute_error={absolute:.17g}, "
        f"relative_error={relative:.17g}, "
        f"reference_norm={reference_norm:.17g}; {context}"
    )


def _dual_norm(helper, value, name):
    representative = helper.state_riesz_representative(value, name)
    squared = helper.dual_pairing(value, representative)
    assert squared >= -ABS_FLOOR * max(1.0, abs(squared))
    return float(np.sqrt(max(0.0, squared)))


def _assert_dual_parity(
    helper,
    actual,
    reference,
    *,
    parent,
    timestep,
    child_index,
    label,
    blockwise=False,
):
    assert isinstance(actual, Cofunction)
    assert isinstance(reference, Cofunction)
    difference = _copy(actual, f"{label}_difference")
    with difference.dat.vec as out, reference.dat.vec_ro as expected:
        out.axpy(-1.0, expected)
    absolute = _dual_norm(helper, difference, f"{label}_difference_riesz")
    reference_norm = _dual_norm(helper, reference, f"{label}_scale_riesz")
    relative = absolute / max(reference_norm, np.finfo(float).tiny)
    context = _diagnostic_text(
        parent,
        timestep=timestep,
        child_index=child_index,
        field="complete_mixed_dual",
        backend="jax",
    )
    assert absolute <= ABS_FLOOR + RTOL * reference_norm, (
        f"{label}: natural_absolute_error={absolute:.17g}, "
        f"natural_relative_error={relative:.17g}, "
        f"reference_norm={reference_norm:.17g}; {context}"
    )
    if blockwise:
        for block in range(6):
            actual_block = Cofunction(
                helper.state_dual_space, name=f"{label}_actual_block_{block}"
            )
            reference_block = Cofunction(
                helper.state_dual_space,
                name=f"{label}_reference_block_{block}",
            )
            actual_block.zero()
            reference_block.zero()
            actual_block.sub(block).assign(actual.sub(block))
            reference_block.sub(block).assign(reference.sub(block))
            block_difference = _copy(
                actual_block, f"{label}_block_{block}_difference"
            )
            with (
                block_difference.dat.vec as out,
                reference_block.dat.vec_ro as expected,
            ):
                out.axpy(-1.0, expected)
            block_absolute = _dual_norm(
                helper,
                block_difference,
                f"{label}_block_{block}_difference_riesz",
            )
            block_reference = _dual_norm(
                helper,
                reference_block,
                f"{label}_block_{block}_scale_riesz",
            )
            block_relative = block_absolute / max(
                block_reference, np.finfo(float).tiny
            )
            block_context = _diagnostic_text(
                parent,
                timestep=timestep,
                child_index=child_index,
                field=f"dual_block_{block}",
                backend="jax",
            )
            assert block_absolute <= ABS_FLOOR + RTOL * block_reference, (
                f"{label}: natural_absolute_error={block_absolute:.17g}, "
                f"natural_relative_error={block_relative:.17g}, "
                f"reference_norm={block_reference:.17g}; {block_context}"
            )


def _assert_scalar_parity(
    actual,
    reference,
    label,
    *,
    parent,
    timestep=0,
    child_index=5,
):
    absolute = abs(float(actual) - float(reference))
    scale = max(abs(float(reference)), 1.0)
    relative = absolute / scale
    context = _diagnostic_text(
        parent,
        timestep=timestep,
        child_index=child_index,
        field="scalar",
        backend="jax",
    )
    assert absolute <= ABS_FLOOR + RTOL * scale, (
        f"{label}: absolute_error={absolute:.17g}, "
        f"relative_error={relative:.17g}, reference_scale={scale:.17g}; "
        f"{context}"
    )


def _direction(case, kind):
    if kind == "ic":
        return case["direction"], 0.0
    if kind == "c0":
        return case["zero"], DELTA_C0
    if kind == "combined":
        return case["direction"], DELTA_C0
    raise ValueError(kind)


def _trajectory(case, backend, nsteps):
    helper = case["helpers"][backend]
    states = [_copy(case["state"], f"{backend}_state_0")]
    caches = []
    for step in range(nsteps):
        cache = helper.take_forward_step_cached(
            states[-1], case["t0"] + step * case["dt"], case["dt"]
        )
        caches.append(cache)
        states.append(_copy(cache.state_out, f"{backend}_state_{step + 1}"))
    return tuple(states), tuple(caches)


def _deployed_step(case, backend, state, time, name):
    output, output_sub, _ = case["model"].get_full_var(
        name, split_x_and_aux=True
    )
    split = case["splits"][backend]
    split.reset_internal_vars()
    split.take_forward_step(
        output, output_sub, [state], float(time), case["dt"]
    )
    return _copy(output[0], f"{name}_owned")


def _assert_gll_safe(cache):
    assert hasattr(cache, "gll_active_set")
    pairs = (
        ("condensation_margin", "condensation_argument"),
        ("evaporation_margin", "evaporation_argument"),
        ("evaporation_cap_margin", "evaporation_cap_difference"),
        ("rain_margin", "rain_argument"),
        ("depth_denominator_margin", "depth_denominator"),
    )
    for margin_name, value_name in pairs:
        margin = float(cache.gll_active_set.margins[margin_name])
        values = np.asarray(cache.gll_diagnostics[value_name])
        scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        assert margin / scale > 1.0e-5, (
            margin_name,
            margin,
            scale,
            cache.gll_active_set.signature,
        )


@pytest.fixture(scope="module")
def full_split_case():
    parameters = get_parameters(CFG)
    parameters["timestepping"]["subcycle_list"] = [2, 1, 2, 1]
    parameters["hyperviscosity"]["treat_as_coeffs"] = True
    parameters["threewayphysics"]["treat_as_coeffs"] = False
    logger = EmptyLogger()
    original_mesh = dimswe_meshes.PeriodicRectangleMesh

    def comm_self_mesh(*args, **kwargs):
        kwargs["comm"] = COMM_SELF
        return original_mesh(*args, **kwargs)

    dimswe_meshes.PeriodicRectangleMesh = comm_self_mesh
    try:
        model = get_model(parameters, logger, has_dynamics_statistics=False)
    finally:
        dimswe_meshes.PeriodicRectangleMesh = original_mesh
    assert model.mesh.comm.size == 1

    coefficient, coefficient_sub, _ = model.get_coeff_var("j3_coefficient")
    state_container, state_sub, _ = model.get_full_var(
        "j3_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)

    x = SpatialCoordinate(model.mesh)
    lx = model.initcond.Lx
    ly = model.initcond.Ly
    mode_x = sin(2.0 * pi * x[0] / lx)
    mode_y = cos(2.0 * pi * x[1] / ly)
    height = 750.0 + 4.0 * mode_x + 3.0 * mode_y
    state_sub["v"].project(
        as_vector([25.0 + 1.5 * mode_y, 17.0 + mode_x])
    )
    state_sub["h"].project(height)
    state_sub["S"].project(
        height
        * model.initcond.g
        * (1.02 + 0.0015 * mode_x - 0.0010 * mode_y)
    )
    state_sub["Qv"].project(0.0030 * height)
    state_sub["Qc"].project(0.0010 * height)
    state_sub["Qr"].project(0.0002 * height)

    solver_parameters = _serial_solver_parameters()
    default = get_timestepper(parameters, model, logger, solver_parameters)
    ufl = get_timestepper(
        parameters, model, logger, solver_parameters, moist_backend="ufl"
    )
    jax_split = get_timestepper(
        parameters, model, logger, solver_parameters, moist_backend="jax"
    )

    direction = model.get_x_var("j3_direction")[0]
    direction.sub(0).project(as_vector([0.22 * mode_x, -0.17 * mode_y]))
    direction.sub(1).project(0.18 * mode_y)
    direction.sub(2).project(1.7 * mode_x - 1.1 * mode_y)
    direction.sub(3).project(1.1e-5 * height * (1.0 + 0.2 * mode_x))
    direction.sub(4).project(-8.0e-6 * height * (1.0 - 0.2 * mode_y))
    direction.sub(5).project(6.0e-6 * height * (1.0 + 0.1 * mode_x))

    probe = model.get_x_var("j3_probe")[0]
    probe.sub(0).project(as_vector([-0.15 * mode_y, 0.19 * mode_x]))
    probe.sub(1).project(-0.12 * mode_x)
    probe.sub(2).project(1.3 * mode_y)
    probe.sub(3).project(-7.0e-6 * height * mode_y)
    probe.sub(4).project(5.0e-6 * height * mode_x)
    probe.sub(5).project(9.0e-6 * height * mode_y)

    target = model.get_x_var("j3_target")[0]
    target.assign(0.985 * state_container[0])
    case = {
        "parameters": parameters,
        "model": model,
        "coefficient": coefficient,
        "coefficient_sub": coefficient_sub,
        "splits": {"default": default, "ufl": ufl, "jax": jax_split},
        "state": state_container[0],
        "direction": direction,
        "probe": probe,
        "target": target,
        "t0": float(time),
        "dt": float(parameters["timestepping"]["dt"]),
    }
    case["zero"] = _zero_state(case, "j3_zero_direction")
    case["helpers"] = {
        name: split._get_mtswe_split_hvp_helper()
        for name, split in case["splits"].items()
    }
    _set_c0(case)
    return case


class TestBackendDefaultAndGraph:
    def test_backend_default_and_six_child_graph(self, full_split_case):
        case = full_split_case
        default = case["splits"]["default"]
        ufl = case["splits"]["ufl"]
        jax_split = case["splits"]["jax"]
        assert default.moist_backend == ufl.moist_backend == "ufl"
        assert jax_split.moist_backend == "jax"
        assert [type(child) for child in default.time_integrators] == [
            type(child) for child in ufl.time_integrators
        ]
        assert all(
            type(ufl.time_integrators[index])
            is type(jax_split.time_integrators[index])
            for index in range(3)
        )
        assert isinstance(jax_split.time_integrators[3], JAXMoistEulerIntegrator)
        assert jax_split.time_integrators[3].ufl_oracle.__class__.__name__ == "Euler"
        assert all(
            left is not right
            for left, right in zip(ufl.time_integrators, jax_split.time_integrators)
        )
        for backend in ("default", "ufl", "jax"):
            helper = case["helpers"][backend]
            specs = helper._child_specs(case["t0"], case["dt"])
            assert tuple(spec[0] for spec in specs) == CHILD_ORDER
            assert tuple(spec[3] for spec in specs) == (
                case["t0"],
                case["t0"] + case["dt"] / 2.0,
                case["t0"],
                case["t0"],
                case["t0"] + case["dt"] / 2.0,
                case["t0"],
            )
            assert tuple(spec[4] for spec in specs) == (
                case["dt"] / 2.0,
                case["dt"] / 2.0,
                case["dt"],
                case["dt"] / 2.0,
                case["dt"] / 2.0,
                case["dt"],
            )
            diagnostics = helper.production_graph_diagnostics()
            assert diagnostics["forward_child_order"] == CHILD_ORDER
            assert diagnostics["reverse_child_order"] == tuple(
                reversed(CHILD_ORDER)
            )
        assert isinstance(case["helpers"]["ufl"].moist_helper, ProductionMoistEulerHVP)
        assert isinstance(case["helpers"]["jax"].moist_helper, JAXMoistEulerHVP)

    def test_invalid_backend_is_rejected(self, full_split_case):
        case = full_split_case
        with pytest.raises(ValueError, match="invalid moist_backend"):
            get_timestepper(
                case["parameters"],
                case["model"],
                EmptyLogger(),
                _serial_solver_parameters(),
                moist_backend="neural",
            )
        with pytest.raises(TypeError, match="moist_backend must be"):
            get_timestepper(
                case["parameters"],
                case["model"],
                EmptyLogger(),
                _serial_solver_parameters(),
                moist_backend=None,
            )


class TestCompletePrimalParity:
    def test_one_step_primal_parity(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        ufl = case["helpers"]["ufl"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        jax_cache = case["helpers"]["jax"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        assert ufl.forward_child_order == jax_cache.forward_child_order == CHILD_ORDER
        assert len(ufl.boundary_states) == len(jax_cache.boundary_states) == 7
        for index, (actual, reference) in enumerate(
            zip(jax_cache.boundary_states, ufl.boundary_states)
        ):
            _assert_function_parity(
                actual,
                reference,
                parent=jax_cache,
                timestep=0,
                child_index=max(0, index - 1),
                label=f"one_step_boundary_{index}",
            )
            if index <= 5:
                assert np.array_equal(_values(actual), _values(reference)), (
                    "first-five boundary lost bitwise parity; "
                    + _diagnostic_text(
                        jax_cache,
                        timestep=0,
                        child_index=max(0, index - 1),
                        field="complete_mixed_coefficients",
                        backend="jax",
                    )
                )
        moist_ufl = _moist_cache(ufl)
        moist_jax = _moist_cache(jax_cache)
        assert isinstance(moist_ufl, MoistEulerPrimalCache)
        _assert_gll_safe(moist_jax)
        assert (
            moist_jax.legacy_active_set.signature
            == moist_ufl.active_set.signature
        ), _diagnostic_text(
            jax_cache,
            timestep=0,
            child_index=5,
            field="legacy_DG1_active_signature",
            backend="jax",
        )
        _assert_function_parity(
            moist_jax.tendency,
            moist_ufl.tendency,
            parent=jax_cache,
            timestep=0,
            child_index=5,
            label="one_step_moist_tendency",
        )
        ufl_source = case["helpers"]["ufl"].moist_helper.state_mass_map(
            moist_ufl.tendency, "one_step_ufl_moist_source"
        )
        _assert_dual_parity(
            case["helpers"]["jax"],
            moist_jax.source_dual,
            ufl_source,
            parent=jax_cache,
            timestep=0,
            child_index=5,
            label="one_step_moist_source",
            blockwise=True,
        )
        ufl_on_gll = case["helpers"]["jax"].moist_helper.active_set_diagnostics(
            ufl.boundary_states[5]
        )
        assert (
            ufl_on_gll.gll.signature == moist_jax.gll_active_set.signature
        ), _diagnostic_text(
            jax_cache,
            timestep=0,
            child_index=5,
            field="actual_GLL_active_signature",
            backend="jax",
        )
        for margin_name in moist_jax.gll_active_set.margins:
            np.testing.assert_allclose(
                float(moist_jax.gll_active_set.margins[margin_name]),
                float(ufl_on_gll.gll.margins[margin_name]),
                rtol=RTOL,
                atol=ABS_FLOOR,
                err_msg=_diagnostic_text(
                    jax_cache,
                    timestep=0,
                    child_index=5,
                    field=f"GLL_{margin_name}",
                    backend="jax",
                ),
            )
        default_output = _deployed_step(
            case, "default", case["state"], case["t0"], "j3_default_step"
        )
        ufl_output = _deployed_step(
            case, "ufl", case["state"], case["t0"], "j3_ufl_step"
        )
        jax_output = _deployed_step(
            case, "jax", case["state"], case["t0"], "j3_jax_step"
        )
        np.testing.assert_array_equal(_values(default_output), _values(ufl_output))
        _assert_function_parity(
            jax_output,
            ufl_output,
            parent=jax_cache,
            timestep=0,
            child_index=5,
            label="deployed_complete_step",
        )
        _assert_function_parity(
            jax_output,
            jax_cache.state_out,
            parent=jax_cache,
            timestep=0,
            child_index=5,
            label="deployed_cached_jax_step",
        )
        for child_index in range(5):
            assert type(ufl.children[child_index].cache) is type(
                jax_cache.children[child_index].cache
            )
            assert ufl.children[child_index].cache is not jax_cache.children[
                child_index
            ].cache

    def test_three_step_primal_parity(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        states_ufl, caches_ufl = _trajectory(case, "ufl", 3)
        states_jax, caches_jax = _trajectory(case, "jax", 3)
        for step, (actual, reference, parent) in enumerate(
            zip(states_jax, states_ufl, (caches_jax[0],) + caches_jax)
        ):
            _assert_function_parity(
                actual,
                reference,
                parent=parent,
                timestep=min(step, 2),
                child_index=5,
                label=f"three_step_boundary_{step}",
            )
        repeated, repeated_caches = _trajectory(case, "jax", 3)
        for first, second in zip(states_jax, repeated):
            np.testing.assert_array_equal(_values(first), _values(second))
        assert all(
            first is not second
            for first, second in zip(caches_jax, repeated_caches)
        )
        _assert_function_parity(
            states_jax[-1],
            states_ufl[-1],
            parent=caches_jax[-1],
            timestep=2,
            child_index=5,
            label="three_step_target_relevant_final_state",
        )


class TestCompleteTangentParity:
    def test_complete_tangent_parity(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        primal_ufl = case["helpers"]["ufl"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        primal_jax = case["helpers"]["jax"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        for kind in ("ic", "c0", "combined"):
            direction, delta_c0 = _direction(case, kind)
            expected = case["helpers"]["ufl"].take_tangent_step(
                primal_ufl, direction, delta_c0
            )
            actual = case["helpers"]["jax"].take_tangent_step(
                primal_jax, direction, delta_c0
            )
            assert len(actual.boundary_state_directions) == 7
            for index, (actual_state, reference_state) in enumerate(
                zip(
                    actual.boundary_state_directions,
                    expected.boundary_state_directions,
                )
            ):
                _assert_function_parity(
                    actual_state,
                    reference_state,
                    parent=primal_jax,
                    timestep=0,
                    child_index=max(0, index - 1),
                    label=f"tangent_{kind}_boundary_{index}",
                )
            assert all(
                type(expected.children[index].cache)
                is type(actual.children[index].cache)
                for index in range(5)
            )
            _assert_function_parity(
                actual.children[-1].cache.tendency_direction,
                expected.children[-1].cache.tendency_direction,
                parent=primal_jax,
                timestep=0,
                child_index=5,
                label=f"tangent_{kind}_moist_tendency",
            )


class TestCompleteReverseParity:
    def test_complete_reverse_and_gradient_parity(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        primal_ufl = case["helpers"]["ufl"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        primal_jax = case["helpers"]["jax"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        terminal = case["helpers"]["ufl"].state_mass_map(
            case["probe"], "j3_terminal_dual"
        )
        terminal_before = _values(terminal)
        expected = case["helpers"]["ufl"].take_adjoint_step_cached(
            primal_ufl, terminal
        )
        actual = case["helpers"]["jax"].take_adjoint_step_cached(
            primal_jax, terminal
        )
        assert actual.reverse_child_order == tuple(reversed(CHILD_ORDER))
        for index, (actual_child, expected_child) in enumerate(
            zip(actual.children, expected.children)
        ):
            assert actual_child.name == expected_child.name
            _assert_dual_parity(
                case["helpers"]["jax"],
                actual_child.result.state_adjoint_in,
                expected_child.result.state_adjoint_in,
                parent=primal_jax,
                timestep=0,
                child_index=5 - index,
                label=f"reverse_boundary_{actual_child.name}",
            )
        moist_result = actual.children[0].result
        assert isinstance(moist_result, JAXMoistFixedControlReverseResult)
        assert moist_result.c0_gradient == 0.0
        assert isinstance(moist_result.state_adjoint_in, Cofunction)
        _assert_dual_parity(
            case["helpers"]["jax"],
            moist_result.stage_state_adjoint,
            expected.children[0].result.stage_state_adjoint,
            parent=primal_jax,
            timestep=0,
            child_index=5,
            label="moist_pullback",
            blockwise=True,
        )
        _assert_scalar_parity(
            actual.physical_c0_gradient,
            expected.physical_c0_gradient,
            "physical_c0_gradient",
            parent=primal_jax,
        )
        np.testing.assert_array_equal(_values(terminal), terminal_before)


class TestCompleteIncrementalReverseParity:
    def test_complete_incremental_reverse_hvp_parity(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        for nsteps in (1, 3):
            for kind in ("ic", "c0", "combined"):
                direction, delta_c0 = _direction(case, kind)
                expected = case["helpers"]["ufl"].terminal_least_squares_hvp(
                    nsteps,
                    case["state"],
                    case["t0"],
                    case["dt"],
                    case["target"],
                    direction,
                    delta_c0,
                )
                actual = case["helpers"]["jax"].terminal_least_squares_hvp(
                    nsteps,
                    case["state"],
                    case["t0"],
                    case["dt"],
                    case["target"],
                    direction,
                    delta_c0,
                )
                for parent_cache in actual.primal_caches:
                    _assert_gll_safe(_moist_cache(parent_cache))
                _assert_dual_parity(
                    case["helpers"]["jax"],
                    actual.initial_condition_hvp,
                    expected.initial_condition_hvp,
                    parent=actual.primal_caches[0],
                    timestep=0,
                    child_index=5,
                    label=f"reduced_hvp_{kind}_{nsteps}",
                )
                _assert_scalar_parity(
                    actual.physical_c0_hvp,
                    expected.physical_c0_hvp,
                    f"physical_c0_hvp_{kind}_{nsteps}",
                    parent=actual.primal_caches[0],
                )
                for reverse_step, (actual_step, expected_step) in enumerate(
                    zip(actual.reverse_results, expected.reverse_results)
                ):
                    for index, (actual_child, expected_child) in enumerate(
                        zip(actual_step.children, expected_step.children)
                    ):
                        _assert_dual_parity(
                            case["helpers"]["jax"],
                            actual_child.result.incremental_state_adjoint_in,
                            expected_child.result.incremental_state_adjoint_in,
                            parent=actual.primal_caches[
                                nsteps - reverse_step - 1
                            ],
                            timestep=nsteps - reverse_step - 1,
                            child_index=5 - index,
                            label=(
                                f"incremental_{kind}_{nsteps}_"
                                f"{actual_child.name}"
                            ),
                        )
                        if actual_child.name == "moist_euler":
                            _assert_function_parity(
                                actual_child.result.incremental_reverse_auxiliary,
                                expected_child.result.incremental_reverse_auxiliary,
                                parent=actual.primal_caches[
                                    nsteps - reverse_step - 1
                                ],
                                timestep=nsteps - reverse_step - 1,
                                child_index=5,
                                label=(
                                    f"incremental_{kind}_{nsteps}_"
                                    "moist_reverse_auxiliary"
                                ),
                            )
                            _assert_dual_parity(
                                case["helpers"]["jax"],
                                actual_child.result.incremental_stage_state_adjoint,
                                expected_child.result.incremental_stage_state_adjoint,
                                parent=actual.primal_caches[
                                    nsteps - reverse_step - 1
                                ],
                                timestep=nsteps - reverse_step - 1,
                                child_index=5,
                                label=(
                                    f"incremental_{kind}_{nsteps}_"
                                    "moist_stage_pullback"
                                ),
                                blockwise=True,
                            )
                    moist = actual_step.children[0].result
                    assert isinstance(moist, JAXMoistFixedControlHVPResult)
                    assert moist.ordinary.c0_gradient == 0.0
                    assert moist.c0_hvp == 0.0

    def test_combined_hessian_bilinear_symmetry(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        left = case["helpers"]["jax"].terminal_least_squares_hvp(
            1,
            case["state"],
            case["t0"],
            case["dt"],
            case["target"],
            case["direction"],
            DELTA_C0,
        )
        right = case["helpers"]["jax"].terminal_least_squares_hvp(
            1,
            case["state"],
            case["t0"],
            case["dt"],
            case["target"],
            case["probe"],
            -0.008,
        )
        u_hv = case["helpers"]["jax"].dual_pairing(
            right.initial_condition_hvp, case["direction"]
        ) + DELTA_C0 * right.physical_c0_hvp
        v_hu = case["helpers"]["jax"].dual_pairing(
            left.initial_condition_hvp, case["probe"]
        ) - 0.008 * left.physical_c0_hvp
        absolute = abs(u_hv - v_hu)
        scale = max(abs(u_hv), abs(v_hu), 1.0)
        assert absolute <= ABS_FLOOR + PRODUCTION_SYMMETRY_RTOL * scale, (
            "combined_natural_hessian_symmetry: "
            f"absolute_error={absolute:.17g}, "
            f"relative_error={absolute / scale:.17g}, "
            f"reference_scale={scale:.17g}"
        )


class TestReducedObjectiveParity:
    def test_reduced_objective_gradient_hvp_parity(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        for nsteps in (1, 3):
            expected_gradient = case["helpers"]["ufl"].terminal_least_squares_gradient(
                nsteps,
                case["state"],
                case["t0"],
                case["dt"],
                case["target"],
            )
            actual_gradient = case["helpers"]["jax"].terminal_least_squares_gradient(
                nsteps,
                case["state"],
                case["t0"],
                case["dt"],
                case["target"],
            )
            for parent_cache in actual_gradient.primal_caches:
                _assert_gll_safe(_moist_cache(parent_cache))
            _assert_scalar_parity(
                actual_gradient.objective_value,
                expected_gradient.objective_value,
                f"reduced_objective_{nsteps}",
                parent=actual_gradient.primal_caches[0],
            )
            _assert_dual_parity(
                case["helpers"]["jax"],
                actual_gradient.initial_condition_gradient,
                expected_gradient.initial_condition_gradient,
                parent=actual_gradient.primal_caches[0],
                timestep=0,
                child_index=5,
                label=f"initial_condition_gradient_{nsteps}",
                blockwise=True,
            )
            _assert_scalar_parity(
                actual_gradient.physical_c0_gradient,
                expected_gradient.physical_c0_gradient,
                f"c0_gradient_{nsteps}",
                parent=actual_gradient.primal_caches[0],
            )
            for kind in ("ic", "c0", "combined"):
                direction, delta_c0 = _direction(case, kind)
                expected = case["helpers"]["ufl"].terminal_least_squares_hvp(
                    nsteps,
                    case["state"],
                    case["t0"],
                    case["dt"],
                    case["target"],
                    direction,
                    delta_c0,
                )
                actual = case["helpers"]["jax"].terminal_least_squares_hvp(
                    nsteps,
                    case["state"],
                    case["t0"],
                    case["dt"],
                    case["target"],
                    direction,
                    delta_c0,
                )
                _assert_dual_parity(
                    case["helpers"]["jax"],
                    actual.initial_condition_hvp,
                    expected.initial_condition_hvp,
                    parent=actual.primal_caches[0],
                    timestep=0,
                    child_index=5,
                    label=f"reduced_{kind}_{nsteps}_initial_condition_hvp",
                    blockwise=True,
                )
                _assert_scalar_parity(
                    actual.physical_c0_hvp,
                    expected.physical_c0_hvp,
                    f"reduced_{kind}_{nsteps}_c0_hvp",
                    parent=actual.primal_caches[0],
                )


class TestExistingPyROLBackendParity:
    def test_existing_pyrol_scalar_ic_combined_parity(self, full_split_case):
        pytest.importorskip("pyrol")
        from pyrol.vectors import NumPyVector

        from dimswe.mtswe_rol_adapter import (
            MTSWECombinedVector,
            MTSWEStateVector,
            ProductionMTSWECombinedObjective,
            ProductionMTSWEInitialConditionObjective,
            ProductionMTSWEScalarC0Objective,
        )

        case = full_split_case
        _set_c0(case)
        scalar_results = {}
        state_results = {}
        combined_results = {}
        for backend in ("ufl", "jax"):
            split = case["splits"][backend]
            helper = case["helpers"][backend]
            scalar_objective = ProductionMTSWEScalarC0Objective(
                split,
                case["coefficient"],
                case["state"],
                case["target"],
                nsteps=1,
                t0=case["t0"],
                dt=case["dt"],
                c0_scale=C0_SCALE,
            )
            z = NumPyVector(np.array([1.0], dtype=np.float64))
            qz = NumPyVector(np.array([0.3], dtype=np.float64))
            gz = z.clone()
            hz = z.clone()
            scalar_objective.gradient(gz, z, 0.0)
            scalar_objective.hessVec(hz, qz, z, 0.0)
            scalar_results[backend] = (
                scalar_objective.value(z, 0.0),
                float(gz.array[0]),
                float(hz.array[0]),
            )

            x = MTSWEStateVector(case["state"], helper)
            v = MTSWEStateVector(case["direction"], helper)
            gx = x.clone()
            hx = x.clone()
            state_objective = ProductionMTSWEInitialConditionObjective(
                split,
                case["coefficient"],
                case["target"],
                fixed_c0_physical=PHYSICAL_C0,
                nsteps=1,
                t0=case["t0"],
                dt=case["dt"],
            )
            state_objective.gradient(gx, x, 0.0)
            state_objective.hessVec(hx, v, x, 0.0)
            state_results[backend] = (
                state_objective.value(x, 0.0),
                _copy(gx.function, f"{backend}_pyrol_state_gradient"),
                _copy(hx.function, f"{backend}_pyrol_state_hvp"),
            )

            y = MTSWECombinedVector(
                x, NumPyVector(np.array([1.0], dtype=np.float64))
            )
            q = MTSWECombinedVector(
                v, NumPyVector(np.array([0.3], dtype=np.float64))
            )
            gy = y.clone()
            hy = y.clone()
            combined_objective = ProductionMTSWECombinedObjective(
                split,
                case["coefficient"],
                case["target"],
                nsteps=1,
                t0=case["t0"],
                dt=case["dt"],
                c0_scale=C0_SCALE,
            )
            combined_objective.gradient(gy, y, 0.0)
            combined_objective.hessVec(hy, q, y, 0.0)
            combined_results[backend] = (
                combined_objective.value(y, 0.0),
                _copy(gy.field.function, f"{backend}_combined_gradient"),
                float(gy.scalar.array[0]),
                _copy(hy.field.function, f"{backend}_combined_hvp"),
                float(hy.scalar.array[0]),
            )

        parent = case["helpers"]["jax"].take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        for actual, reference, label in zip(
            scalar_results["jax"],
            scalar_results["ufl"],
            ("pyrol_scalar_value", "pyrol_scalar_gradient", "pyrol_scalar_hvp"),
        ):
            _assert_scalar_parity(actual, reference, label, parent=parent)
        _assert_scalar_parity(
            state_results["jax"][0],
            state_results["ufl"][0],
            "pyrol_ic_value",
            parent=parent,
        )
        for index, label in ((1, "pyrol_ic_gradient"), (2, "pyrol_ic_hvp")):
            _assert_function_parity(
                state_results["jax"][index],
                state_results["ufl"][index],
                parent=parent,
                timestep=0,
                child_index=5,
                label=label,
            )
        _assert_scalar_parity(
            combined_results["jax"][0],
            combined_results["ufl"][0],
            "pyrol_combined_value",
            parent=parent,
        )
        for index, label in (
            (1, "pyrol_combined_gradient"),
            (3, "pyrol_combined_hvp"),
        ):
            _assert_function_parity(
                combined_results["jax"][index],
                combined_results["ufl"][index],
                parent=parent,
                timestep=0,
                child_index=5,
                label=label,
            )
        _assert_scalar_parity(
            combined_results["jax"][2],
            combined_results["ufl"][2],
            "pyrol_combined_scalar_gradient",
            parent=parent,
        )
        _assert_scalar_parity(
            combined_results["jax"][4],
            combined_results["ufl"][4],
            "pyrol_combined_scalar_hvp",
            parent=parent,
        )


class TestOwnershipAndRestoration:
    def test_ownership_repeatability_exception_restoration(self, full_split_case):
        case = full_split_case
        _set_c0(case)
        helper = case["helpers"]["jax"]
        state_before = _values(case["state"])
        direction_before = _values(case["direction"])
        terminal = helper.state_mass_map(case["probe"], "j3_ownership_terminal")
        terminal_before = _values(terminal)
        first = helper.take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        first_tangent = helper.take_tangent_step(
            first, case["direction"], DELTA_C0
        )
        first_reverse = helper.take_adjoint_step_cached(first, terminal)
        first_hvp = helper.take_incremental_adjoint_step(
            first_tangent, terminal, terminal
        )
        second = helper.take_forward_step_cached(
            case["state"], case["t0"], case["dt"]
        )
        second_tangent = helper.take_tangent_step(
            second, case["direction"], DELTA_C0
        )
        np.testing.assert_array_equal(_values(case["state"]), state_before)
        np.testing.assert_array_equal(_values(case["direction"]), direction_before)
        np.testing.assert_array_equal(_values(terminal), terminal_before)
        np.testing.assert_array_equal(_values(first.state_out), _values(second.state_out))
        np.testing.assert_array_equal(
            _values(first_tangent.state_direction_out),
            _values(second_tangent.state_direction_out),
        )
        assert first is not second
        assert first.state_out.dat is not second.state_out.dat
        assert all(
            first.boundary_states[i].dat is not first.boundary_states[j].dat
            for i in range(7)
            for j in range(i + 1, 7)
        )
        assert isinstance(first_reverse.state_adjoint_in, Cofunction)
        assert isinstance(first_hvp.incremental_state_adjoint_in, Cofunction)

        moist_helper = helper.moist_helper
        original_source_assembly = moist_helper.source_assembly

        def induced_failure(_):
            raise RuntimeError("induced J3 source assembly failure")

        moist_helper.source_assembly = induced_failure
        try:
            with pytest.raises(
                RuntimeError, match="induced J3 source assembly failure"
            ):
                helper.take_tangent_step(second, case["direction"], DELTA_C0)
        finally:
            moist_helper.source_assembly = original_source_assembly
        assert case["splits"]["jax"].moist_backend == "jax"
        recovered = helper.take_tangent_step(
            second, case["direction"], DELTA_C0
        )
        np.testing.assert_array_equal(
            _values(recovered.state_direction_out),
            _values(second_tangent.state_direction_out),
        )
        np.testing.assert_array_equal(_values(case["state"]), state_before)
        np.testing.assert_array_equal(_values(case["direction"]), direction_before)
        np.testing.assert_array_equal(_values(terminal), terminal_before)
