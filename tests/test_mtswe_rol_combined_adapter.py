"""Focused tests for combined state/normalized-c0 production PyROL access."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip(
    "pyrol",
    reason="PyROL is optional; install rol-python for MTSWE adapter tests",
)

from firedrake import COMM_SELF, SpatialCoordinate, as_vector, cos, pi, sin
from pyrol import Objective
from pyrol.pyrol.std import vector_double_t
from pyrol.vectors import NumPyVector

import dimswe.meshes as dimswe_meshes
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.mtswe_rol_adapter import (
    MTSWE_ACTIVE_SET_SWITCHES,
    MTSWECombinedVector,
    MTSWEGradientActiveSetQualificationError,
    MTSWEHVPActiveSetQualificationError,
    MTSWEStateVector,
    ProductionMTSWECombinedObjective,
)
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import get_timestepper


pytestmark = pytest.mark.rol

CFG = "tests/mtswe_small.cfg"
PHYSICAL_C0 = 0.07
PHYSICAL_C0_DIRECTION = 0.012


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
        return vector.getArray(readonly=True).copy()


def _relative_error(computed, expected):
    scale = max(np.linalg.norm(expected), np.finfo(float).tiny)
    return float(np.linalg.norm(computed - expected) / scale)


def _std_steps(values):
    result = vector_double_t()
    for value in values:
        result.push_back(float(value))
    return result


def _std_rows(values):
    return [
        [values[i][j] for j in range(values[i].size())]
        for i in range(values.size())
    ]


def _scalar(value):
    return NumPyVector(np.array([value], dtype=np.float64))


@pytest.fixture(scope="module")
def production_combined_case():
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

    coefficient, coefficient_sub, _ = model.get_coeff_var(
        "mtswe_rol_combined_coefficient"
    )
    state_container, state_sub, _ = model.get_full_var(
        "mtswe_rol_combined_initial", split_x_and_aux=True
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
    entropy_density = height * model.initcond.g * (
        1.02 + 0.0015 * mode_x - 0.0010 * mode_y
    )
    state_sub["v"].project(
        as_vector([25.0 + 1.5 * mode_y, 17.0 + 1.0 * mode_x])
    )
    state_sub["h"].project(height)
    state_sub["S"].project(entropy_density)
    state_sub["Qv"].project(0.0030 * height)
    state_sub["Qc"].project(0.0010 * height)
    state_sub["Qr"].project(0.0002 * height)

    direction = model.get_x_var("mtswe_rol_combined_direction")[0]
    direction.sub(0).project(as_vector([0.22 * mode_x, -0.17 * mode_y]))
    direction.sub(1).project(0.18 * mode_y)
    direction.sub(2).project(1.7 * mode_x - 1.1 * mode_y)
    direction.sub(3).project(1.1e-5 * height * (1.0 + 0.2 * mode_x))
    direction.sub(4).project(-8.0e-6 * height * (1.0 - 0.2 * mode_y))
    direction.sub(5).project(6.0e-6 * height * (1.0 + 0.1 * mode_x))

    probe = model.get_x_var("mtswe_rol_combined_probe")[0]
    probe.sub(0).project(as_vector([-0.15 * mode_y, 0.19 * mode_x]))
    probe.sub(1).project(-0.12 * mode_x)
    probe.sub(2).project(1.3 * mode_y)
    probe.sub(3).project(-7.0e-6 * height * mode_y)
    probe.sub(4).project(5.0e-6 * height * mode_x)
    probe.sub(5).project(9.0e-6 * height * mode_y)

    target = model.get_x_var("mtswe_rol_combined_target")[0]
    target.assign(0.985 * state_container[0])
    timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    timestepper.set_coeff(coefficient)
    helper = timestepper._get_mtswe_split_hvp_helper()
    scale = float(model.get_coeff_scaling_factors()[1])
    assert scale == 0.07
    return SimpleNamespace(
        model=model,
        timestepper=timestepper,
        helper=helper,
        coefficient=coefficient,
        state=state_container[0],
        direction=direction,
        probe=probe,
        target=target,
        t0=float(time),
        dt=float(parameters["timestepping"]["dt"]),
        scale=scale,
    )


@contextmanager
def _physical_c0(case, value):
    snapshots = tuple(
        child.coeff.copy(deepcopy=True)
        for child in case.timestepper.time_integrators
    )
    work = case.coefficient.copy(deepcopy=True)
    work.sub(1).assign(float(value))
    try:
        case.timestepper.reset_internal_vars()
        case.timestepper.set_coeff(work)
        yield
    finally:
        for child, snapshot in zip(
            case.timestepper.time_integrators, snapshots
        ):
            child.set_coeff(snapshot)
        case.timestepper.reset_internal_vars()


def _state_vector(case, value):
    return MTSWEStateVector(value, case.helper)


def _combined(case, field, scalar_value):
    return MTSWECombinedVector(
        _state_vector(case, field), _scalar(scalar_value)
    )


def _objective(case, nsteps=1, **kwargs):
    hvp_tolerances = kwargs.pop(
        "hvp_active_set_tolerances",
        {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES},
    )
    return ProductionMTSWECombinedObjective(
        case.timestepper,
        case.coefficient,
        case.target,
        nsteps=nsteps,
        t0=case.t0,
        dt=case.dt,
        c0_scale=case.scale,
        hvp_active_set_tolerances=hvp_tolerances,
        **kwargs,
    )


def _active_diagnostic(margins=None):
    values = {switch: 1.0 for switch in MTSWE_ACTIVE_SET_SWITCHES}
    if margins is not None:
        values.update(margins)
    return SimpleNamespace(
        signature=((True, False), (False, True), (True, True), (False, False)),
        condensation_margin=values["condensation"],
        evaporation_margin=values["evaporation"],
        evaporation_cap_margin=values["evaporation_cap"],
        rain_margin=values["rain"],
        depth_denominator_margin=values["depth_denominator"],
    )


def _primal_caches(nsteps, margins=None):
    moist = SimpleNamespace(
        name="moist_euler",
        cache=SimpleNamespace(active_set=_active_diagnostic(margins)),
    )
    return tuple(SimpleNamespace(children=(moist,)) for _ in range(nsteps))


def _analytic_result(
    case, y, q=None, margins=None, nsteps=1, center=None
):
    if center is None:
        center = _combined(
            case, case.state, PHYSICAL_C0 / case.scale
        )
    residual = y.clone()
    residual.set(y)
    residual.axpy(-1.0, center)
    gradient = case.helper.state_mass_map(
        residual.field.function, "mtswe_rol_combined_analytic_gradient"
    )
    result = {
        "objective_value": 0.5 * residual.dot(residual),
        "initial_condition_gradient": gradient,
        "physical_c0_gradient": (
            float(residual.scalar.array[0]) / case.scale
        ),
        "primal_caches": _primal_caches(nsteps, margins),
    }
    if q is not None:
        result["initial_condition_hvp"] = case.helper.state_mass_map(
            q.field.function, "mtswe_rol_combined_analytic_hvp"
        )
        result["physical_c0_hvp"] = float(q.scalar.array[0]) / case.scale
    return SimpleNamespace(**result)


class _CombinedQuadraticObjective(Objective):
    """Minimal native-ROL contract check in product Riesz coordinates."""

    def __init__(self):
        super().__init__()
        self.hessvec_calls = 0

    def value(self, x, tol):
        return 0.5 * x.dot(x)

    def gradient(self, g, x, tol):
        g.set(x)

    def hessVec(self, hv, v, x, tol):
        self.hessvec_calls += 1
        hv.set(v)

    def update(self, x, *args):
        pass


def test_combined_vector_ownership_clone_algebra_and_product_metric(
    production_combined_case,
):
    case = production_combined_case
    field_input = _state_vector(case, case.state)
    scalar_input = _scalar(1.25)
    vector = MTSWECombinedVector(field_input, scalar_input)
    field_before = _values(field_input.function)
    field_input.zero()
    scalar_input.array[0] = -9.0
    np.testing.assert_array_equal(_values(vector.field.function), field_before)
    assert vector.scalar.array[0] == 1.25

    clone = vector.clone()
    assert clone.field.function.dat is not vector.field.function.dat
    assert not np.shares_memory(clone.scalar.array, vector.scalar.array)
    clone.set(vector)
    direction = _combined(case, case.direction, -0.4)
    expected_field = field_before + _values(case.direction)
    clone.plus(direction)
    assert clone.scalar.array[0] == 0.85
    np.testing.assert_allclose(_values(clone.field.function), expected_field)
    clone.scale(0.5)
    expected_field *= 0.5
    clone.axpy(-0.25, direction)
    expected_field -= 0.25 * _values(case.direction)
    np.testing.assert_allclose(_values(clone.field.function), expected_field)
    np.testing.assert_allclose(clone.scalar.array[0], 0.525)

    expected_dot = vector.field.dot(direction.field) + (
        vector.scalar.array[0] * direction.scalar.array[0]
    )
    assert vector.dot(direction) == expected_dot
    assert vector.apply(direction) == expected_dot
    assert vector.dual() is vector
    assert vector.dimension() == vector.field.dimension() + 1
    np.testing.assert_allclose(vector.norm() ** 2, vector.dot(vector), rtol=2e-13)

    dual_copy = vector.dual().clone()
    dual_copy.set(vector.dual())
    dual_copy.zero()
    assert vector.dual() is vector
    assert vector.dot(direction) == expected_dot

    clone.zero()
    np.testing.assert_array_equal(
        _values(clone.field.function), np.zeros_like(field_before)
    )
    assert clone.scalar.array[0] == 0.0


@pytest.mark.parametrize("nsteps", (1, 3))
def test_combined_normalized_gradient_parity(
    production_combined_case, nsteps
):
    case = production_combined_case
    z = PHYSICAL_C0 / case.scale
    y = _combined(case, case.state, z)
    objective = _objective(case, nsteps)
    with _physical_c0(case, case.scale * z):
        direct = case.timestepper.mtswe_terminal_least_squares_gradient(
            nsteps, case.state, case.t0, case.dt, case.target
        )
    expected_field = case.helper.state_riesz_representative(
        direct.initial_condition_gradient,
        f"mtswe_rol_combined_expected_gradient_{nsteps}",
    )
    g = y.clone()
    assert objective.value(y, 0.0) == direct.objective_value
    objective.gradient(g, y, 0.0)
    assert _relative_error(
        _values(g.field.function), _values(expected_field)
    ) < 2.0e-13
    np.testing.assert_allclose(
        g.scalar.array[0],
        case.scale * direct.physical_c0_gradient,
        rtol=2.0e-13,
        atol=0.0,
    )
    assert objective.production_gradient_evaluations == 1


@pytest.mark.parametrize("nsteps", (1, 3))
@pytest.mark.parametrize("kind", ("c0", "ic", "combined"))
def test_combined_hvp_parity_for_all_direction_blocks(
    production_combined_case, nsteps, kind
):
    case = production_combined_case
    z = PHYSICAL_C0 / case.scale
    y = _combined(case, case.state, z)
    zero = case.state.copy(deepcopy=True)
    zero.assign(0)
    field_direction = zero if kind == "c0" else case.direction
    qz = 0.0 if kind == "ic" else PHYSICAL_C0_DIRECTION / case.scale
    q = _combined(case, field_direction, qz)
    delta_c0 = case.scale * qz
    objective = _objective(case, nsteps)
    y_before = _values(y.field.function)
    q_before = _values(q.field.function)
    z_before = float(y.scalar.array[0])
    qz_before = float(q.scalar.array[0])
    with _physical_c0(case, case.scale * z):
        direct = case.timestepper.mtswe_terminal_least_squares_hvp(
            nsteps,
            case.state,
            case.t0,
            case.dt,
            case.target,
            field_direction,
            delta_c0,
        )
    expected_field = case.helper.state_riesz_representative(
        direct.initial_condition_hvp,
        f"mtswe_rol_combined_expected_hvp_{kind}_{nsteps}",
    )
    hv = y.clone()
    objective.hessVec(hv, q, y, 0.0)
    assert _relative_error(
        _values(hv.field.function), _values(expected_field)
    ) < 2.0e-13
    np.testing.assert_allclose(
        hv.scalar.array[0],
        case.scale * direct.physical_c0_hvp,
        rtol=2.0e-13,
        atol=0.0,
    )
    if kind == "c0":
        physical_scalar_block = direct.physical_c0_hvp / delta_c0
        np.testing.assert_allclose(
            hv.scalar.array[0] / qz,
            case.scale**2 * physical_scalar_block,
            rtol=2.0e-13,
            atol=0.0,
        )
    np.testing.assert_array_equal(_values(y.field.function), y_before)
    np.testing.assert_array_equal(_values(q.field.function), q_before)
    assert y.scalar.array[0] == z_before
    assert q.scalar.array[0] == qz_before
    if nsteps == 1 and kind == "c0":
        assert np.linalg.norm(_values(hv.field.function)) > 0.0
    if nsteps == 1 and kind == "ic":
        assert abs(hv.scalar.array[0]) > 0.0


def test_combined_cache_reuse_invalidation_and_update(
    production_combined_case, monkeypatch
):
    case = production_combined_case
    y = _combined(case, case.state, PHYSICAL_C0 / case.scale)
    q = _combined(case, case.direction, 0.2)
    objective = _objective(case)
    calls = {"gradient": 0, "hvp": 0}

    def gradient_result(value):
        calls["gradient"] += 1
        return _analytic_result(case, value)

    def hvp_result(value, direction):
        calls["hvp"] += 1
        return _analytic_result(case, value, direction)

    monkeypatch.setattr(objective, "_production_gradient", gradient_result)
    monkeypatch.setattr(objective, "_production_hvp", hvp_result)
    g = y.clone()
    hv = y.clone()
    objective.value(y, 0.0)
    objective.gradient(g, y, 0.0)
    assert calls["gradient"] == 1
    objective.hessVec(hv, q, y, 0.0)
    repeated_field = _values(hv.field.function)
    repeated_scalar = float(hv.scalar.array[0])
    objective.hessVec(hv, q, y, 0.0)
    assert calls["hvp"] == 1
    np.testing.assert_array_equal(_values(hv.field.function), repeated_field)
    assert hv.scalar.array[0] == repeated_scalar

    changed_q = q.clone()
    changed_q.set(q)
    changed_q.scalar.array[0] += 0.1
    objective.hessVec(hv, changed_q, y, 0.0)
    objective.gradient(g, y, 0.0)
    assert calls == {"gradient": 1, "hvp": 2}

    changed_y = y.clone()
    changed_y.set(y)
    changed_y.scalar.array[0] += 0.05
    objective.value(changed_y, 0.0)
    assert calls["gradient"] == 2
    assert not objective.cache_info["has_hvp_result"]
    objective.update(changed_y)
    assert not any(objective.cache_info.values())
    objective.hessVec(hv, q, changed_y, 0.0)
    objective.gradient(g, changed_y, 0.0)
    assert calls == {"gradient": 2, "hvp": 3}


def test_combined_nonmutation_exception_restoration_and_owned_target(
    production_combined_case, monkeypatch
):
    case = production_combined_case
    target_input = case.target.copy(deepcopy=True)
    coefficient_input = case.coefficient.copy(deepcopy=True)
    target_before = _values(target_input)
    coefficient_before = _values(coefficient_input)
    objective = ProductionMTSWECombinedObjective(
        case.timestepper,
        coefficient_input,
        target_input,
        nsteps=1,
        t0=case.t0,
        dt=case.dt,
        c0_scale=case.scale,
    )
    target_input.assign(0)
    coefficient_input.assign(0)
    np.testing.assert_array_equal(_values(objective.target), target_before)
    np.testing.assert_array_equal(
        _values(objective.coefficient_template), coefficient_before
    )
    y = _combined(case, case.state, PHYSICAL_C0 / case.scale)
    q = _combined(case, case.direction, 0.2)
    y_before = _values(y.field.function)
    q_before = _values(q.field.function)
    z_before = float(y.scalar.array[0])
    qz_before = float(q.scalar.array[0])
    coefficients_before = tuple(
        _values(child.coeff) for child in case.timestepper.time_integrators
    )

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate combined production failure")

    monkeypatch.setattr(
        case.timestepper, "mtswe_terminal_least_squares_hvp", fail
    )
    with pytest.raises(RuntimeError, match="deliberate combined"):
        objective.hessVec(y.clone(), q, y, 0.0)
    np.testing.assert_array_equal(_values(y.field.function), y_before)
    np.testing.assert_array_equal(_values(q.field.function), q_before)
    assert y.scalar.array[0] == z_before
    assert q.scalar.array[0] == qz_before
    for child, before in zip(
        case.timestepper.time_integrators, coefficients_before
    ):
        np.testing.assert_array_equal(_values(child.coeff), before)


def test_combined_active_set_policies(
    production_combined_case, monkeypatch
):
    case = production_combined_case
    tolerances = {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES}
    tolerances["evaporation_cap"] = 0.2
    objective = _objective(
        case,
        hvp_active_set_tolerances=tolerances,
    )
    y = _combined(case, case.state, PHYSICAL_C0 / case.scale)
    q = _combined(case, case.direction, 0.2)
    near = {"evaporation_cap": 0.1}
    monkeypatch.setattr(
        objective,
        "_production_gradient",
        lambda value: _analytic_result(case, value, margins=near),
    )
    monkeypatch.setattr(
        objective,
        "_production_hvp",
        lambda value, direction: _analytic_result(
            case, value, direction, margins=near
        ),
    )
    objective.gradient(y.clone(), y, 0.0)
    with pytest.raises(MTSWEHVPActiveSetQualificationError):
        objective.hessVec(y.clone(), q, y, 0.0)

    objective.update(y)
    monkeypatch.setattr(
        objective,
        "_production_gradient",
        lambda value: _analytic_result(
            case, value, margins={"rain": 0.0}
        ),
    )
    with pytest.raises(MTSWEGradientActiveSetQualificationError):
        objective.gradient(y.clone(), y, 0.0)


def test_native_pyrol_combined_derivative_utilities_call_hessvec(
    production_combined_case, monkeypatch
):
    case = production_combined_case
    zero_field = case.state.copy(deepcopy=True)
    zero_field.assign(0)
    y_zero = _combined(case, zero_field, 0.0)
    q = _combined(case, case.direction, 0.2)
    probe = _combined(case, case.probe, -0.15)
    steps = _std_steps((1.0e-3, 1.0e-4, 1.0e-5))
    zero_before = _values(y_zero.field.function)
    q_before = _values(q.field.function)
    probe_before = _values(probe.field.function)
    q_scalar_before = float(q.scalar.array[0])
    probe_scalar_before = float(probe.scalar.array[0])

    quadratic = _CombinedQuadraticObjective()
    quadratic_gradient = _std_rows(
        quadratic.checkGradient(y_zero, q, steps, False)
    )
    quadratic_hessian = _std_rows(
        quadratic.checkHessVec(y_zero, q, steps, False)
    )
    quadratic_symmetry = quadratic.checkHessSym(
        y_zero, q, probe, False
    )
    print(
        "MTSWE combined zero-base quadratic:",
        {
            "gradient_rows": quadratic_gradient,
            "hessian_rows": quadratic_hessian,
            "symmetry": [
                quadratic_symmetry[index]
                for index in range(quadratic_symmetry.size())
            ],
            "y_norm": y_zero.norm(),
            "q_norm": q.norm(),
        },
    )
    assert quadratic_gradient[-1][-1] < quadratic_gradient[0][-1]
    assert quadratic_hessian[-1][-1] < 1.0e-8
    assert abs(
        quadratic_symmetry[quadratic_symmetry.size() - 1]
    ) < 1.0e-10
    assert quadratic.hessvec_calls > 0
    np.testing.assert_array_equal(
        _values(y_zero.field.function), zero_before
    )
    assert y_zero.scalar.array[0] == 0.0
    np.testing.assert_array_equal(_values(q.field.function), q_before)
    np.testing.assert_array_equal(
        _values(probe.field.function), probe_before
    )
    assert q.scalar.array[0] == q_scalar_before
    assert probe.scalar.array[0] == probe_scalar_before
    dual_work = y_zero.dual().clone()
    dual_work.set(y_zero.dual())
    dual_work.plus(q)
    np.testing.assert_array_equal(
        _values(y_zero.field.function), zero_before
    )
    assert y_zero.scalar.array[0] == 0.0

    y = _combined(case, zero_field, 0.25)
    center = y.clone()
    center.set(y)
    objective = _objective(case)
    calls = {"hvp": 0}

    monkeypatch.setattr(
        objective,
        "_production_gradient",
        lambda value: _analytic_result(case, value, center=center),
    )

    def hvp_result(value, direction):
        calls["hvp"] += 1
        return _analytic_result(
            case, value, direction, center=center
        )

    monkeypatch.setattr(objective, "_production_hvp", hvp_result)
    gradient_rows = _std_rows(objective.checkGradient(y, q, steps, False))
    hessian_rows = _std_rows(objective.checkHessVec(y, q, steps, False))
    symmetry = objective.checkHessSym(y, q, probe, False)
    assert len(gradient_rows) == len(hessian_rows) == 3
    assert gradient_rows[-1][-1] < gradient_rows[0][-1]
    assert hessian_rows[-1][-1] < 1.0e-8
    assert abs(symmetry[symmetry.size() - 1]) < 1.0e-10
    assert calls["hvp"] > 0
    np.testing.assert_array_equal(_values(y.field.function), zero_before)
    assert y.scalar.array[0] == 0.25
    np.testing.assert_array_equal(_values(q.field.function), q_before)
    np.testing.assert_array_equal(
        _values(probe.field.function), probe_before
    )
    assert q.scalar.array[0] == q_scalar_before
    assert probe.scalar.array[0] == probe_scalar_before


def test_production_combined_hessian_bilinear_symmetry(
    production_combined_case,
):
    case = production_combined_case
    y = _combined(case, case.state, PHYSICAL_C0 / case.scale)
    u = _combined(case, case.direction, 0.2)
    v = _combined(case, case.probe, -0.15)
    objective = _objective(case)
    hu = y.clone()
    hv = y.clone()
    objective.hessVec(hu, u, y, 0.0)
    objective.hessVec(hv, v, y, 0.0)
    left = u.dot(hv)
    right = v.dot(hu)
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    assert abs(left - right) / scale < 5.0e-9
