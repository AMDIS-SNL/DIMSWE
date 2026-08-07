"""Serial Firedrake parity tests for the standalone J1 JAX moist child."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest


jax = pytest.importorskip("jax", reason="JAX is optional for DIMSWE")
jax.config.update("jax_enable_x64", True)
firedrake = pytest.importorskip(
    "firedrake", reason="Firedrake is required for J1 external parity"
)

from firedrake import (  # noqa: E402
    COMM_SELF,
    Cofunction,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    action,
    assemble,
    cos,
    inner,
    norm,
    pi,
    sin,
)
import ufl  # noqa: E402

import dimswe.meshes as dimswe_meshes  # noqa: E402
from dimswe.jax_moist_adapter import JAXMoistEulerPrimal  # noqa: E402
from dimswe.logger import EmptyLogger  # noqa: E402
from dimswe.models import get_model  # noqa: E402
from dimswe.parameters import get_parameters, overall_solver_parameters  # noqa: E402
from dimswe.physics import qsat  # noqa: E402
from dimswe.timestepping import Euler  # noqa: E402


CFG = "tests/mtswe_small.cfg"
EPS = np.finfo(np.float64).eps
LOCAL_FACTOR = 512.0
FIELD_FACTOR = 4096.0


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


def _make_case(*, coefficient_mode):
    parameters = get_parameters(CFG)
    parameters["mesh"]["type"] = "rectangle"
    parameters["mesh"]["nx"] = 2
    parameters["mesh"]["ny"] = 2
    parameters["threewayphysics"]["treat_as_coeffs"] = coefficient_mode
    parameters["hyperviscosity"]["treat_as_coeffs"] = False
    logger = EmptyLogger()

    original_rectangle_mesh = dimswe_meshes.RectangleMesh

    def comm_self_mesh(*args, **kwargs):
        kwargs["comm"] = COMM_SELF
        mesh = original_rectangle_mesh(*args, **kwargs)
        # The production periodic mesh constructor adds these attributes;
        # Hyperviscosity expects them even though this isolated test uses a
        # nonperiodic mesh solely for transparent cell-orientation checks.
        mesh.dx = 1.0
        mesh.dy = 1.0
        return mesh

    dimswe_meshes.RectangleMesh = comm_self_mesh
    try:
        model = get_model(parameters, logger, has_dynamics_statistics=False)
    finally:
        dimswe_meshes.RectangleMesh = original_rectangle_mesh
    assert model.mesh.comm.size == 1

    coefficient, coefficient_sub, _ = model.get_coeff_var("jax_moist_coeff")
    state, state_sub, _ = model.get_full_var(
        "jax_moist_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)

    state_sub["v"].assign(0.0)
    state_sub["h"].assign(750.0)
    state_sub["S"].assign(750.0 * model.initcond.g)
    state_sub["Qv"].assign(750.0 * 0.003)
    state_sub["Qc"].assign(750.0 * 0.001)
    state_sub["Qr"].assign(750.0 * 0.0002)

    solver_parameters = _serial_solver_parameters()
    oracle = Euler(
        model, logger, solver_parameters, terms=["threewayphysics"]
    )
    if coefficient is not None:
        oracle.set_coeff(coefficient)
    adapter = JAXMoistEulerPrimal(model, solver_parameters)
    return {
        "parameters": parameters,
        "model": model,
        "coefficient": coefficient,
        "coefficient_sub": coefficient_sub,
        "state_container": state,
        "state": state[0],
        "state_sub": state_sub,
        "time": float(time),
        "term": adapter.term,
        "oracle": oracle,
        "adapter": adapter,
    }


@pytest.fixture(scope="module")
def fixed_case():
    return _make_case(coefficient_mode=False)


@pytest.fixture(scope="module")
def coefficient_case():
    return _make_case(coefficient_mode=True)


def _set_constant_state(case, *, qv, qc, h=750.0, qr=0.0002):
    gravity = case["model"].initcond.g
    case["state_sub"]["v"].assign(0.0)
    case["state_sub"]["h"].assign(h)
    case["state_sub"]["S"].assign(h * gravity)
    case["state_sub"]["Qv"].assign(h * qv)
    case["state_sub"]["Qc"].assign(h * qc)
    case["state_sub"]["Qr"].assign(h * qr)


def _coefficient_argument(case):
    return case["coefficient"] if case["coefficient"] is not None else None


def _oracle_source_dual(case, applied_dt):
    oracle = case["oracle"]
    oracle.production_stage_base_state.assign(case["state"])
    oracle.t.assign(case["time"])
    oracle.dt.assign(applied_dt)
    if case["coefficient"] is not None:
        oracle.set_coeff(case["coefficient"])
    result = assemble(oracle.production_stage_rhs_forms[0])
    assert isinstance(result, Cofunction)
    return result


def _oracle_step(case, applied_dt, name):
    output, output_sub, _ = case["model"].get_full_var(
        name, split_x_and_aux=True
    )
    oracle = case["oracle"]
    oracle.reset_internal_vars()
    if case["coefficient"] is not None:
        oracle.set_coeff(case["coefficient"])
    oracle.take_forward_step(
        output,
        output_sub,
        [case["state"]],
        case["time"],
        applied_dt,
    )
    return (
        oracle.Fi[0][0][0].copy(deepcopy=True),
        output[0].copy(deepcopy=True),
    )


def _ufl_diagnostics(case):
    adapter = case["adapter"]
    parameters = adapter._parameters(_coefficient_argument(case))
    state = case["state"]
    fields = {
        name: state.sub(index)
        for index, name in enumerate(("v", "h", "S", "Qv", "Qc", "Qr"))
    }
    h = fields["h"]
    qv = fields["Qv"] / h
    qc = fields["Qc"] / h
    s = fields["S"] / h
    beta2 = float(parameters["g"] * parameters["L"])
    saturation = qsat(
        h,
        s,
        case["term"].B,
        float(parameters["q0"]),
        float(parameters["H0"]),
        float(parameters["g"]),
    )
    gamma_v = 1.0 / (
        1.0
        + 20.0 * saturation * beta2 / float(parameters["g"])
    )
    configured_dt = float(parameters["configured_dt"])
    c_argument = gamma_v * (qv - saturation) / configured_dt
    condensation = ufl.max_value(0.0, c_argument)
    e_argument = gamma_v * (saturation - qv) / configured_dt
    evaporation_positive = ufl.max_value(0.0, e_argument)
    evaporation = ufl.min_value(
        qc / configured_dt, evaporation_positive
    )
    rain = ufl.max_value(
        0.0,
        float(parameters["gamma_r"])
        * (qc - float(parameters["qprecip"]))
        / configured_dt,
    )
    expressions = {
        "qsat": saturation,
        "gamma_v": gamma_v,
        "C": condensation,
        "E_positive": evaporation_positive,
        "E": evaporation,
        "A": evaporation - condensation,
        "R": rain,
    }
    return {
        key: adapter.interpolate_and_pack(
            expression, f"ufl_{key}_gll"
        )[1]
        for key, expression in expressions.items()
    }


def _function_difference(left, right, name):
    result = left.copy(deepcopy=True)
    result.rename(name)
    with result.dat.vec as result_vec, right.dat.vec_ro as right_vec:
        result_vec.axpy(-1.0, right_vec)
    return result


def _mixed_values(value):
    with value.dat.vec_ro as vector:
        return np.array(vector.array_r, dtype=np.float64, copy=True)


def _dual_difference(left, right, name):
    result = left.copy(deepcopy=True)
    result.rename(name)
    with result.dat.vec as result_vec, right.dat.vec_ro as right_vec:
        result_vec.axpy(-1.0, right_vec)
    return result


def _dual_natural_norm(adapter, value, name):
    representative = adapter.solve_mass(value, name)
    squared = adapter.dual_pairing(value, representative)
    scale = max(1.0, abs(squared))
    assert squared >= -64.0 * EPS * scale
    return math.sqrt(max(0.0, squared))


def _isolated_dual_block(adapter, value, index, name):
    result = Cofunction(adapter.state_dual_space, name=name)
    result.zero()
    result.sub(index).assign(value.sub(index))
    return result


def _assert_dual_parity(case, actual, reference, cache, label):
    adapter = case["adapter"]
    difference = _dual_difference(actual, reference, f"{label}_difference")
    mixed_error = _dual_natural_norm(adapter, difference, f"{label}_error_riesz")
    mixed_scale = _dual_natural_norm(adapter, reference, f"{label}_scale_riesz")
    maximum_carrier = max(
        float(np.max(np.abs(value)))
        for value in cache.source_density.values()
    )
    overall_tolerance = FIELD_FACTOR * EPS * max(mixed_scale, 1.0e-300)
    diagnostics = []
    for index, name in enumerate(("v", "h", "S", "Qv", "Qc", "Qr")):
        block_difference = _isolated_dual_block(
            adapter, difference, index, f"{label}_{name}_difference"
        )
        block_reference = _isolated_dual_block(
            adapter, reference, index, f"{label}_{name}_reference"
        )
        absolute = _dual_natural_norm(
            adapter, block_difference, f"{label}_{name}_error_riesz"
        )
        scale = _dual_natural_norm(
            adapter, block_reference, f"{label}_{name}_scale_riesz"
        )
        relative = absolute / max(scale, 1.0e-300)
        tolerance = FIELD_FACTOR * EPS * max(
            scale, mixed_scale * EPS, 1.0e-300
        )
        diagnostics.append((name, absolute, relative, scale, tolerance))
        assert absolute <= tolerance, (
            f"block={name}, absolute_error={absolute:.17g}, "
            f"relative_error={relative:.17g}, reference_scale={scale:.17g}, "
            f"maximum_carrier_discrepancy={maximum_carrier:.17g}, "
            f"gll_signature={cache.gll_active_set.signature}, "
            f"legacy_signature={cache.legacy_active_set.signature}, "
            f"configured_dt={cache.configured_dt:.17g}, "
            f"applied_dt={cache.applied_dt:.17g}"
        )
    assert mixed_error <= overall_tolerance, (
        f"mixed_absolute_error={mixed_error:.17g}, "
        f"mixed_relative_error={mixed_error/max(mixed_scale, 1.0e-300):.17g}, "
        f"reference_scale={mixed_scale:.17g}, "
        f"maximum_carrier_discrepancy={maximum_carrier:.17g}, "
        f"blocks={diagnostics}, configured_dt={cache.configured_dt:.17g}, "
        f"applied_dt={cache.applied_dt:.17g}"
    )


def _assert_function_parity(case, actual, reference, cache, label):
    difference = _function_difference(actual, reference, f"{label}_difference")
    absolute = float(norm(difference))
    scale = float(norm(reference))
    relative = absolute / max(scale, 1.0e-300)
    maximum = max(
        float(np.max(np.abs(value))) for value in cache.source_density.values()
    )
    tolerance = FIELD_FACTOR * EPS * max(scale, 1.0e-300)
    assert absolute <= tolerance, (
        f"field={label}, absolute_error={absolute:.17g}, "
        f"relative_error={relative:.17g}, reference_scale={scale:.17g}, "
        f"maximum_carrier_discrepancy={maximum:.17g}, "
        f"gll_signature={cache.gll_active_set.signature}, "
        f"legacy_signature={cache.legacy_active_set.signature}, "
        f"configured_dt={cache.configured_dt:.17g}, "
        f"applied_dt={cache.applied_dt:.17g}"
    )


def test_broken_gll_carrier_coordinates_order_and_roundtrip(fixed_case):
    adapter = fixed_case["adapter"]
    layout = adapter.layout
    assert layout.owned_cell_count == 4
    assert layout.points_per_cell == 16
    assert layout.cell_nodes.shape == (4, 16)

    gll = np.asarray((0.0, 0.276393202250021, 0.723606797749979, 1.0))
    expected_reference = np.asarray([(x, y) for y in gll for x in gll])
    np.testing.assert_allclose(
        layout.reference_points, expected_reference, rtol=0.0, atol=8.0 * EPS
    )

    x = SpatialCoordinate(fixed_case["model"].mesh)
    _, packed_x = adapter.interpolate_and_pack(x[0], "packed_x")
    _, packed_y = adapter.interpolate_and_pack(x[1], "packed_y")
    assert np.unique(np.column_stack((packed_x.mean(1), packed_y.mean(1))), axis=0).shape[0] == 4
    for cell in range(layout.owned_cell_count):
        normalized_x = (packed_x[cell] - packed_x[cell].min()) / (
            packed_x[cell].max() - packed_x[cell].min()
        )
        normalized_y = (packed_y[cell] - packed_y[cell].min()) / (
            packed_y[cell].max() - packed_y[cell].min()
        )
        np.testing.assert_allclose(
            np.column_stack((normalized_x, normalized_y)),
            expected_reference,
            rtol=0.0,
            atol=64.0 * EPS,
        )

    carrier = adapter.unpack_carrier(packed_x + 3.0 * packed_y, "roundtrip")
    repacked = adapter.pack_carrier(carrier)
    np.testing.assert_array_equal(repacked, packed_x + 3.0 * packed_y)
    np.testing.assert_array_equal(
        adapter.interpolate_and_pack(x[0], "packed_x_repeat")[1], packed_x
    )

    dg3 = FunctionSpace(
        fixed_case["model"].mesh, "DG", 3, variant="spectral"
    )
    dg3_points = np.asarray(dg3.finat_element.dual_basis[1].points)
    assert dg3_points.shape == (16, 2)
    assert not np.allclose(dg3_points, layout.reference_points)
    assert np.all(dg3_points > 0.0) and np.all(dg3_points < 1.0)


@pytest.mark.parametrize(
    ("qv", "qc"),
    (
        (0.003, 0.001),
        (0.003, 0.00005),
        (0.001, 0.001),
        (0.001, 0.0002),
        (0.001, 0.00005),
        (0.003, -0.0002),
    ),
)
def test_exact_gll_local_rates_match_ufl(fixed_case, qv, qc):
    _set_constant_state(fixed_case, qv=qv, qc=qc)
    cache = fixed_case["adapter"].evaluate(
        fixed_case["state"],
        37.0,
        coefficient=_coefficient_argument(fixed_case),
    )
    reference = _ufl_diagnostics(fixed_case)
    keys = ("C", "E_positive", "E", "A", "R", "qsat", "gamma_v")
    for key in keys:
        actual = (
            cache.rates[key]
            if key in cache.rates
            else cache.gll_diagnostics[key]
        )
        expected = reference[key]
        error = float(np.max(np.abs(actual - expected)))
        scale = max(float(np.max(np.abs(expected))), 1.0e-300)
        relative = error / scale
        assert error <= LOCAL_FACTOR * EPS * scale, (
            f"field={key}, absolute_error={error:.17g}, "
            f"relative_error={relative:.17g}, reference_scale={scale:.17g}, "
            f"maximum_carrier_discrepancy={error:.17g}, "
            f"gll_signature={cache.gll_active_set.signature}, "
            f"legacy_signature={cache.legacy_active_set.signature}, "
            f"configured_dt={cache.configured_dt:.17g}, "
            f"applied_dt={cache.applied_dt:.17g}"
        )


def test_source_dual_and_mass_solved_tendency_match_ufl(fixed_case):
    _set_constant_state(fixed_case, qv=0.003, qc=0.001)
    applied_dt = 73.0
    cache = fixed_case["adapter"].evaluate(
        fixed_case["state"], applied_dt
    )
    reference_dual = _oracle_source_dual(fixed_case, applied_dt)
    reference_tendency, _ = _oracle_step(
        fixed_case, applied_dt, "jax_moist_oracle_tendency"
    )
    _assert_dual_parity(
        fixed_case, cache.source_dual, reference_dual, cache, "source_dual"
    )
    _assert_function_parity(
        fixed_case,
        cache.tendency,
        reference_tendency,
        cache,
        "tendency",
    )
    for name in ("v", "h"):
        index = fixed_case["model"].get_x_var_list().index(name)
        np.testing.assert_array_equal(
            cache.source_dual.sub(index).dat.data_ro, 0.0
        )
        np.testing.assert_array_equal(
            cache.tendency.sub(index).dat.data_ro, 0.0
        )


@pytest.mark.parametrize("applied_dt", (100.0, 37.0, 0.0))
def test_complete_euler_output_matches_for_distinct_applied_dt(
    fixed_case, applied_dt
):
    _set_constant_state(fixed_case, qv=0.001, qc=0.0002)
    cache = fixed_case["adapter"].evaluate(
        fixed_case["state"], applied_dt
    )
    reference_tendency, reference_output = _oracle_step(
        fixed_case, applied_dt, f"jax_moist_oracle_output_{applied_dt}"
    )
    _assert_function_parity(
        fixed_case,
        cache.tendency,
        reference_tendency,
        cache,
        "tendency",
    )
    _assert_function_parity(
        fixed_case, cache.state_out, reference_output, cache, "state_out"
    )
    assert cache.configured_dt == 100.0
    assert cache.applied_dt == applied_dt
    for name in ("v", "h"):
        index = fixed_case["model"].get_x_var_list().index(name)
        assert norm(cache.state_out.sub(index) - fixed_case["state"].sub(index)) == 0.0


@pytest.mark.parametrize("case_name", ("fixed_case", "coefficient_case"))
def test_fixed_and_real_coefficient_parameter_modes(request, case_name):
    case = request.getfixturevalue(case_name)
    _set_constant_state(case, qv=0.001, qc=0.0002)
    applied_dt = 61.0
    cache = case["adapter"].evaluate(
        case["state"],
        applied_dt,
        coefficient=_coefficient_argument(case),
    )
    reference_dual = _oracle_source_dual(case, applied_dt)
    reference_tendency, reference_output = _oracle_step(
        case, applied_dt, f"jax_moist_{case_name}_oracle"
    )
    _assert_dual_parity(
        case, cache.source_dual, reference_dual, cache, case_name
    )
    _assert_function_parity(
        case, cache.tendency, reference_tendency, cache, f"{case_name}_tendency"
    )
    _assert_function_parity(
        case, cache.state_out, reference_output, cache, f"{case_name}_output"
    )


def _integrated_invariants(case, state, beta2):
    fields = {
        name: state.sub(index)
        for index, name in enumerate(("v", "h", "S", "Qv", "Qc", "Qr"))
    }
    water = float(
        assemble((fields["Qv"] + fields["Qc"] + fields["Qr"]) * case["model"].spaces.dx)
    )
    thermal = float(
        assemble((fields["S"] - beta2 * fields["Qv"]) * case["model"].spaces.dx)
    )
    return water, thermal


def test_local_weak_and_post_solve_invariants(fixed_case):
    _set_constant_state(fixed_case, qv=0.001, qc=0.0002)
    cache = fixed_case["adapter"].evaluate(fixed_case["state"], 100.0)
    source = cache.source_density
    beta2 = float(cache.parameters["g"] * cache.parameters["L"])
    water_source = source["Qv"] + source["Qc"] + source["Qr"]
    thermal_source = source["S"] - beta2 * source["Qv"]
    water_scale = max(
        *(float(np.max(np.abs(source[name]))) for name in ("Qv", "Qc", "Qr")),
        1.0e-300,
    )
    thermal_scale = max(
        float(np.max(np.abs(source["S"]))),
        beta2 * float(np.max(np.abs(source["Qv"]))),
        1.0e-300,
    )
    assert np.max(np.abs(water_source)) <= 8.0 * EPS * water_scale
    assert np.max(np.abs(thermal_source)) <= 8.0 * EPS * thermal_scale

    water_probe = Function(fixed_case["adapter"].state_space)
    water_probe.sub(3).assign(1.0)
    water_probe.sub(4).assign(1.0)
    water_probe.sub(5).assign(1.0)
    thermal_probe = Function(fixed_case["adapter"].state_space)
    thermal_probe.sub(2).assign(1.0)
    thermal_probe.sub(3).assign(-beta2)
    weak_water = float(assemble(action(cache.source_dual, water_probe)))
    weak_thermal = float(assemble(action(cache.source_dual, thermal_probe)))
    water_terms = []
    for index in (3, 4, 5):
        probe = Function(fixed_case["adapter"].state_space)
        probe.sub(index).assign(1.0)
        water_terms.append(float(assemble(action(cache.source_dual, probe))))
    thermal_terms = []
    for index, value in ((2, 1.0), (3, -beta2)):
        probe = Function(fixed_case["adapter"].state_space)
        probe.sub(index).assign(value)
        thermal_terms.append(float(assemble(action(cache.source_dual, probe))))
    weak_water_scale = max(sum(abs(value) for value in water_terms), 1.0e-300)
    weak_thermal_scale = max(
        sum(abs(value) for value in thermal_terms), 1.0e-300
    )
    assert abs(weak_water) <= 256.0 * EPS * weak_water_scale
    assert abs(weak_thermal) <= 256.0 * EPS * weak_thermal_scale

    before = _integrated_invariants(fixed_case, fixed_case["state"], beta2)
    after = _integrated_invariants(fixed_case, cache.state_out, beta2)
    np.testing.assert_allclose(after, before, rtol=64.0 * EPS, atol=0.0)


def test_gll_and_legacy_diagnostic_sampling_are_both_retained(fixed_case):
    model = fixed_case["model"]
    x = SpatialCoordinate(model.mesh)
    lx = model.initcond.Lx
    h = 750.0 + 2.0 * sin(2.0 * pi * x[0] / lx)
    specific_entropy = model.initcond.g * (
        1.0 + 0.006 * cos(2.0 * pi * x[0] / lx)
    )
    fixed_case["state_sub"]["h"].project(h)
    fixed_case["state_sub"]["S"].project(h * specific_entropy)
    fixed_case["state_sub"]["Qv"].project(0.002 * h)
    fixed_case["state_sub"]["Qc"].project(0.0003 * h)
    fixed_case["state_sub"]["Qr"].project(0.0001 * h)
    cache = fixed_case["adapter"].evaluate(fixed_case["state"], 100.0)

    assert len(cache.gll_active_set.signature[0]) == 16 * 4
    assert len(cache.legacy_active_set.signature[0]) == 4 * 4
    gll_margin = float(cache.gll_active_set.margins["condensation_margin"])
    legacy_margin = float(
        cache.legacy_active_set.margins["condensation_margin"]
    )
    assert gll_margin != legacy_margin


def test_ownership_repeatability_and_exception_safety(fixed_case, monkeypatch):
    _set_constant_state(fixed_case, qv=0.003, qc=0.001)
    state_snapshot = _mixed_values(fixed_case["state"])
    topography_snapshot = np.array(fixed_case["term"].B.dat.data_ro, copy=True)
    first = fixed_case["adapter"].evaluate(fixed_case["state"], 45.0)
    first_output_snapshot = _mixed_values(first.state_out)
    first_dual_snapshot = _mixed_values(first.source_dual)
    second = fixed_case["adapter"].evaluate(fixed_case["state"], 45.0)

    np.testing.assert_array_equal(_mixed_values(fixed_case["state"]), state_snapshot)
    np.testing.assert_array_equal(
        fixed_case["term"].B.dat.data_ro, topography_snapshot
    )
    np.testing.assert_array_equal(_mixed_values(first.state_out), first_output_snapshot)
    np.testing.assert_array_equal(_mixed_values(first.source_dual), first_dual_snapshot)
    np.testing.assert_array_equal(_mixed_values(first.state_out), _mixed_values(second.state_out))
    np.testing.assert_array_equal(_mixed_values(first.source_dual), _mixed_values(second.source_dual))
    assert all(not value.flags.writeable for value in first.packed_state.values())
    assert all(not value.flags.writeable for value in first.rates.values())
    assert all(not value.flags.writeable for value in first.source_density.values())

    first.state_out.assign(0.0)
    np.testing.assert_array_equal(_mixed_values(fixed_case["state"]), state_snapshot)
    np.testing.assert_array_equal(_mixed_values(second.state_out), first_output_snapshot)

    def induced_failure(_):
        raise RuntimeError("induced source assembly failure")

    monkeypatch.setattr(
        fixed_case["adapter"], "_assemble_source_dual", induced_failure
    )
    with pytest.raises(RuntimeError, match="induced source assembly failure"):
        fixed_case["adapter"].evaluate(fixed_case["state"], 45.0)
    np.testing.assert_array_equal(_mixed_values(fixed_case["state"]), state_snapshot)
    np.testing.assert_array_equal(
        fixed_case["term"].B.dat.data_ro, topography_snapshot
    )
