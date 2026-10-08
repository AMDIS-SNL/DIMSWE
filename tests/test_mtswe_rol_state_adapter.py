"""Focused tests for the production MTSWE initial-condition PyROL stage."""

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

import dimswe.meshes as dimswe_meshes
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.mtswe_rol_adapter import (
    MTSWE_ACTIVE_SET_SWITCHES,
    MTSWEGradientActiveSetQualificationError,
    MTSWEHVPActiveSetQualificationError,
    MTSWEStateVector,
    ProductionMTSWEInitialConditionObjective,
)
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import get_timestepper


pytestmark = pytest.mark.rol

CFG = "tests/mtswe_small.cfg"
PHYSICAL_C0 = 0.07
STATE_FIELD_NAMES = ("v", "h", "S", "Qv", "Qc", "Qr")


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


@pytest.fixture(scope="module")
def production_state_case():
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
        "mtswe_rol_state_coefficient"
    )
    state_container, state_sub, _ = model.get_full_var(
        "mtswe_rol_state_initial", split_x_and_aux=True
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

    direction = model.get_x_var("mtswe_rol_state_direction")[0]
    direction.sub(0).project(as_vector([0.22 * mode_x, -0.17 * mode_y]))
    direction.sub(1).project(0.18 * mode_y)
    direction.sub(2).project(1.7 * mode_x - 1.1 * mode_y)
    direction.sub(3).project(1.1e-5 * height * (1.0 + 0.2 * mode_x))
    direction.sub(4).project(-8.0e-6 * height * (1.0 - 0.2 * mode_y))
    direction.sub(5).project(6.0e-6 * height * (1.0 + 0.1 * mode_x))

    probe = model.get_x_var("mtswe_rol_state_probe")[0]
    probe.sub(0).project(as_vector([-0.15 * mode_y, 0.19 * mode_x]))
    probe.sub(1).project(-0.12 * mode_x)
    probe.sub(2).project(1.3 * mode_y)
    probe.sub(3).project(-7.0e-6 * height * mode_y)
    probe.sub(4).project(5.0e-6 * height * mode_x)
    probe.sub(5).project(9.0e-6 * height * mode_y)

    target = model.get_x_var("mtswe_rol_state_target")[0]
    target.assign(0.985 * state_container[0])
    timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    timestepper.set_coeff(coefficient)
    helper = timestepper._get_mtswe_split_hvp_helper()
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
    )


@contextmanager
def _physical_c0(case, value=PHYSICAL_C0):
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


def _objective(case, nsteps=1, **kwargs):
    hvp_tolerances = kwargs.pop(
        "hvp_active_set_tolerances",
        {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES},
    )
    return ProductionMTSWEInitialConditionObjective(
        case.timestepper,
        case.coefficient,
        case.target,
        fixed_c0_physical=PHYSICAL_C0,
        nsteps=nsteps,
        t0=case.t0,
        dt=case.dt,
        hvp_active_set_tolerances=hvp_tolerances,
        **kwargs,
    )


def _vector(case, value):
    return MTSWEStateVector(value, case.helper)


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
    case, x, v=None, margins=None, nsteps=1, center=None
):
    if center is None:
        center = _vector(case, case.state)
    residual = x.clone()
    residual.set(x)
    residual.axpy(-1.0, center)
    gradient = case.helper.state_mass_map(
        residual.function, "mtswe_rol_test_analytic_gradient"
    )
    result = {
        "objective_value": 0.5 * residual.dot(residual),
        "initial_condition_gradient": gradient,
        "primal_caches": _primal_caches(nsteps, margins),
    }
    if v is not None:
        result["initial_condition_hvp"] = case.helper.state_mass_map(
            v.function, "mtswe_rol_test_analytic_hvp"
        )
    return SimpleNamespace(**result)


class _StateQuadraticObjective(Objective):
    """Minimal native-ROL contract check in state Riesz coordinates."""

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


def _scaled_difference(left, right, scale):
    result = left.clone()
    result.set(left)
    result.axpy(-1.0, right)
    result.scale(scale)
    return result


def _field_maxima(vector):
    return {
        name: float(np.max(np.abs(_values(vector.function.sub(index)))))
        for index, name in enumerate(STATE_FIELD_NAMES)
    }


def _physical_base_quadratic_diagnostic(x, v, steps):
    objective = _StateQuadraticObjective()
    native_rows = _std_rows(objective.checkHessVec(x, v, steps, False))
    direct_rows = []
    for native_row in native_rows:
        eps = native_row[0]
        g0 = x.clone()
        objective.gradient(g0, x, 0.0)
        xp = x.clone()
        xp.set(x)
        xp.axpy(eps, v)
        gp = x.clone()
        objective.gradient(gp, xp, 0.0)
        fd = _scaled_difference(gp, g0, 1.0 / eps)
        error = _scaled_difference(fd, v, 1.0)
        direct_rows.append(
            {
                "step": eps,
                "fd_norm": fd.norm(),
                "v_norm": v.norm(),
                "fd_minus_v_norm": error.norm(),
                "step_times_error": eps * error.norm(),
            }
        )
    return native_rows, direct_rows


def test_state_vector_ownership_clone_and_algebra(production_state_case):
    case = production_state_case
    source = case.state.copy(deepcopy=True)
    source_before = _values(source)
    vector = _vector(case, source)
    source.assign(0)
    np.testing.assert_array_equal(_values(vector.function), source_before)

    clone = vector.clone()
    assert clone.function.dat is not vector.function.dat
    np.testing.assert_array_equal(
        _values(clone.function), np.zeros_like(source_before)
    )
    clone.set(vector)
    np.testing.assert_array_equal(_values(clone.function), source_before)

    direction = _vector(case, case.direction)
    expected = source_before + _values(case.direction)
    clone.plus(direction)
    np.testing.assert_allclose(_values(clone.function), expected)
    clone.scale(0.5)
    expected *= 0.5
    np.testing.assert_allclose(_values(clone.function), expected)
    clone.axpy(-0.25, direction)
    expected -= 0.25 * _values(case.direction)
    np.testing.assert_allclose(_values(clone.function), expected)
    clone.zero()
    np.testing.assert_array_equal(
        _values(clone.function), np.zeros_like(source_before)
    )


def test_state_vector_dimension_space_metric_apply_and_dual(
    production_state_case,
):
    case = production_state_case
    state = _vector(case, case.state)
    direction = _vector(case, case.direction)
    assert state.dimension() == case.state.function_space().dim()
    with pytest.raises((TypeError, ValueError)):
        MTSWEStateVector(case.coefficient, case.helper)

    mass_dual = case.helper.state_mass_map(
        state.function, "mtswe_rol_test_metric_mass"
    )
    expected = case.helper.dual_pairing(mass_dual, direction.function)
    assert state.dot(direction) == expected
    assert state.apply(direction) == expected
    assert state.dual() is state
    np.testing.assert_allclose(state.norm() ** 2, state.dot(state), rtol=2e-13)

    dual_copy = state.dual().clone()
    dual_copy.set(state.dual())
    dual_copy.zero()
    assert state.dual() is state
    assert state.dot(direction) == expected

    raw_dot = float(np.vdot(_values(state.function), _values(direction.function)))
    assert not np.isclose(expected, raw_dot, rtol=1.0e-8, atol=1.0e-12)


@pytest.mark.parametrize("nsteps", (1, 3))
def test_production_field_gradient_and_hvp_parity(
    production_state_case, nsteps
):
    case = production_state_case
    x = _vector(case, case.state)
    v = _vector(case, case.direction)
    objective = _objective(case, nsteps)
    state_before = _values(x.function)
    direction_before = _values(v.function)
    target_before = _values(case.target)
    coefficients_before = tuple(
        _values(child.coeff) for child in case.timestepper.time_integrators
    )

    with _physical_c0(case):
        direct_gradient = (
            case.timestepper.mtswe_terminal_least_squares_gradient(
                nsteps, case.state, case.t0, case.dt, case.target
            )
        )
        direct_hvp = case.timestepper.mtswe_terminal_least_squares_hvp(
            nsteps,
            case.state,
            case.t0,
            case.dt,
            case.target,
            case.direction,
            0.0,
        )

    expected_gradient = case.helper.state_riesz_representative(
        direct_gradient.initial_condition_gradient,
        f"mtswe_rol_expected_gradient_{nsteps}",
    )
    expected_hvp = case.helper.state_riesz_representative(
        direct_hvp.initial_condition_hvp,
        f"mtswe_rol_expected_hvp_{nsteps}",
    )
    g = x.clone()
    hv = x.clone()
    assert objective.value(x, 0.0) == direct_gradient.objective_value
    objective.gradient(g, x, 0.0)
    objective.hessVec(hv, v, x, 0.0)

    assert _relative_error(
        _values(g.function), _values(expected_gradient)
    ) < 2.0e-13
    assert _relative_error(
        _values(hv.function), _values(expected_hvp)
    ) < 2.0e-13
    assert objective.production_gradient_evaluations == 1
    assert objective.production_hvp_evaluations == 1
    np.testing.assert_array_equal(_values(x.function), state_before)
    np.testing.assert_array_equal(_values(v.function), direction_before)
    np.testing.assert_array_equal(_values(case.target), target_before)
    for child, before in zip(
        case.timestepper.time_integrators, coefficients_before
    ):
        np.testing.assert_array_equal(_values(child.coeff), before)


def test_field_cache_reuse_invalidation_and_update(
    production_state_case, monkeypatch
):
    case = production_state_case
    objective = _objective(case)
    x = _vector(case, case.state)
    v = _vector(case, case.direction)
    calls = {"gradient": 0, "hvp": 0}

    def gradient_result(value):
        calls["gradient"] += 1
        return _analytic_result(case, value)

    def hvp_result(value, direction):
        calls["hvp"] += 1
        return _analytic_result(case, value, direction)

    monkeypatch.setattr(objective, "_production_gradient", gradient_result)
    monkeypatch.setattr(objective, "_production_hvp", hvp_result)
    g = x.clone()
    hv = x.clone()
    objective.value(x, 0.0)
    objective.gradient(g, x, 0.0)
    assert calls["gradient"] == 1
    objective.hessVec(hv, v, x, 0.0)
    objective.hessVec(hv, v, x, 0.0)
    assert calls["hvp"] == 1

    changed_v = v.clone()
    changed_v.set(v)
    changed_v.scale(0.5)
    objective.hessVec(hv, changed_v, x, 0.0)
    objective.gradient(g, x, 0.0)
    assert calls == {"gradient": 1, "hvp": 2}

    changed_x = x.clone()
    changed_x.set(x)
    changed_x.axpy(1.0e-4, v)
    objective.value(changed_x, 0.0)
    assert calls["gradient"] == 2
    assert not objective.cache_info["has_hvp_result"]
    objective.update(changed_x)
    assert not any(objective.cache_info.values())

    objective.hessVec(hv, v, changed_x, 0.0)
    objective.gradient(g, changed_x, 0.0)
    assert calls == {"gradient": 2, "hvp": 3}


def test_field_exception_restores_coefficients_and_owned_target(
    production_state_case, monkeypatch
):
    case = production_state_case
    target_input = case.target.copy(deepcopy=True)
    coefficient_input = case.coefficient.copy(deepcopy=True)
    target_before = _values(target_input)
    coefficient_before = _values(coefficient_input)
    objective = ProductionMTSWEInitialConditionObjective(
        case.timestepper,
        coefficient_input,
        target_input,
        fixed_c0_physical=PHYSICAL_C0,
        nsteps=1,
        t0=case.t0,
        dt=case.dt,
    )
    target_input.assign(0)
    coefficient_input.assign(0)
    np.testing.assert_array_equal(_values(objective.target), target_before)
    np.testing.assert_array_equal(
        _values(objective.coefficient_template), coefficient_before
    )
    coefficients_before = tuple(
        _values(child.coeff) for child in case.timestepper.time_integrators
    )

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate field production failure")

    monkeypatch.setattr(
        case.timestepper, "mtswe_terminal_least_squares_gradient", fail
    )
    with pytest.raises(RuntimeError, match="deliberate field"):
        objective.value(_vector(case, case.state), 0.0)
    for child, before in zip(
        case.timestepper.time_integrators, coefficients_before
    ):
        np.testing.assert_array_equal(_values(child.coeff), before)


def test_field_active_set_gradient_and_hvp_policies(
    production_state_case, monkeypatch
):
    case = production_state_case
    tolerances = {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES}
    tolerances["rain"] = 0.2
    objective = _objective(
        case,
        hvp_active_set_tolerances=tolerances,
    )
    x = _vector(case, case.state)
    v = _vector(case, case.direction)
    near = {"rain": 0.1}
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
    objective.gradient(x.clone(), x, 0.0)
    with pytest.raises(MTSWEHVPActiveSetQualificationError):
        objective.hessVec(x.clone(), v, x, 0.0)

    objective.update(x)
    monkeypatch.setattr(
        objective,
        "_production_gradient",
        lambda value: _analytic_result(
            case, value, margins={"condensation": 0.0}
        ),
    )
    with pytest.raises(MTSWEGradientActiveSetQualificationError):
        objective.gradient(x.clone(), x, 0.0)


def test_native_pyrol_field_derivative_utilities_call_hessvec(
    production_state_case, monkeypatch
):
    case = production_state_case
    x_physical = _vector(case, case.state)
    v = _vector(case, case.direction)
    probe = _vector(case, case.probe)
    steps = _std_steps((1.0e-3, 1.0e-4, 1.0e-5))
    x_physical_before = _values(x_physical.function)
    v_before = _values(v.function)
    probe_before = _values(probe.function)

    physical_rows, direct_rows = _physical_base_quadratic_diagnostic(
        x_physical, v, steps
    )
    physical_report = {
        "native_rows": physical_rows,
        "direct_rows": direct_rows,
        "x_norm": x_physical.norm(),
        "v_norm": v.norm(),
        "x_field_maxima": _field_maxima(x_physical),
        "v_field_maxima": _field_maxima(v),
        "successive_error_growth": [
            direct_rows[index + 1]["fd_minus_v_norm"]
            / direct_rows[index]["fd_minus_v_norm"]
            for index in range(len(direct_rows) - 1)
        ],
    }
    print("MTSWE field physical-base quadratic:", physical_report)
    for native, direct in zip(physical_rows, direct_rows):
        np.testing.assert_allclose(native[0], direct["step"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            native[1], direct["v_norm"], rtol=5.0e-13, atol=1.0e-15
        )
        np.testing.assert_allclose(
            native[2], direct["fd_norm"], rtol=5.0e-13, atol=1.0e-15
        )
        np.testing.assert_allclose(
            native[3],
            direct["fd_minus_v_norm"],
            rtol=5.0e-13,
            atol=1.0e-15,
        )
    physical_errors = [
        row["fd_minus_v_norm"] for row in direct_rows
    ]
    assert physical_errors[0] < physical_errors[1] < physical_errors[2]
    np.testing.assert_array_equal(
        _values(x_physical.function), x_physical_before
    )
    np.testing.assert_array_equal(_values(v.function), v_before)
    np.testing.assert_array_equal(_values(probe.function), probe_before)

    x = x_physical.clone()
    x.zero()
    quadratic = _StateQuadraticObjective()
    quadratic_gradient = _std_rows(
        quadratic.checkGradient(x, v, steps, False)
    )
    quadratic_hessian = _std_rows(
        quadratic.checkHessVec(x, v, steps, False)
    )
    quadratic_symmetry = quadratic.checkHessSym(
        x, v, probe, False
    )
    print(
        "MTSWE field zero-base quadratic:",
        {
            "gradient_rows": quadratic_gradient,
            "hessian_rows": quadratic_hessian,
            "symmetry": [
                quadratic_symmetry[index]
                for index in range(quadratic_symmetry.size())
            ],
            "x_norm": x.norm(),
            "v_norm": v.norm(),
        },
    )
    assert quadratic_gradient[-1][-1] < quadratic_gradient[0][-1]
    assert quadratic_hessian[-1][-1] < 1.0e-8
    assert abs(
        quadratic_symmetry[quadratic_symmetry.size() - 1]
    ) < 1.0e-10
    assert quadratic.hessvec_calls > 0
    np.testing.assert_array_equal(
        _values(x.function), np.zeros_like(x_physical_before)
    )
    np.testing.assert_array_equal(_values(v.function), v_before)
    np.testing.assert_array_equal(_values(probe.function), probe_before)
    dual_work = x.dual().clone()
    dual_work.set(x.dual())
    dual_work.plus(v)
    np.testing.assert_array_equal(
        _values(x.function), np.zeros_like(x_physical_before)
    )

    objective = _objective(case)
    center = x.clone()
    center.set(x)
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
    gradient_rows = _std_rows(objective.checkGradient(x, v, steps, False))
    hessian_rows = _std_rows(objective.checkHessVec(x, v, steps, False))
    symmetry = objective.checkHessSym(x, v, probe, False)
    assert len(gradient_rows) == len(hessian_rows) == 3
    assert gradient_rows[-1][-1] < gradient_rows[0][-1]
    assert hessian_rows[-1][-1] < 1.0e-8
    assert abs(symmetry[symmetry.size() - 1]) < 1.0e-10
    assert calls["hvp"] > 0
    np.testing.assert_array_equal(
        _values(x.function), np.zeros_like(x_physical_before)
    )
    np.testing.assert_array_equal(_values(v.function), v_before)
    np.testing.assert_array_equal(_values(probe.function), probe_before)


def test_small_base_analytic_field_callbacks_are_directionally_consistent(
    production_state_case, monkeypatch, record_property
):
    case = production_state_case
    objective = _objective(case)
    x = _vector(case, case.state)
    x.zero()
    center = x.clone()
    center.set(x)
    v = _vector(case, case.direction)
    x_before = _values(x.function)
    v_before = _values(v.function)
    gradient_points = []
    hvp_points = []
    hvp_directions = []

    def gradient_result(value):
        gradient_points.append(_values(value.function))
        return _analytic_result(case, value, center=center)

    def hvp_result(value, direction):
        hvp_points.append(_values(value.function))
        hvp_directions.append(_values(direction.function))
        return _analytic_result(
            case, value, direction, center=center
        )

    monkeypatch.setattr(objective, "_production_gradient", gradient_result)
    monkeypatch.setattr(objective, "_production_hvp", hvp_result)
    g0 = x.clone()
    objective.gradient(g0, x, 0.0)
    objective.gradient(x.clone(), x, 0.0)
    assert len(gradient_points) == 1

    hv = x.clone()
    objective.hessVec(hv, v, x, 0.0)
    objective.hessVec(x.clone(), v, x, 0.0)
    assert len(hvp_points) == len(hvp_directions) == 1
    np.testing.assert_array_equal(hvp_points[0], x_before)
    np.testing.assert_array_equal(hvp_directions[0], v_before)

    metrics = []
    expected_gradient_points = [x_before]
    for eps in (1.0e-3, 1.0e-4, 1.0e-5):
        xp = x.clone()
        xp.set(x)
        xp.axpy(eps, v)
        xm = x.clone()
        xm.set(x)
        xm.axpy(-eps, v)
        gp = x.clone()
        gm = x.clone()
        expected_gradient_points.extend(
            (_values(xp.function), _values(xm.function))
        )
        objective.gradient(gp, xp, 0.0)
        objective.gradient(gm, xm, 0.0)
        forward = _scaled_difference(gp, g0, 1.0 / eps)
        centered = _scaled_difference(gp, gm, 0.5 / eps)
        forward_error = _scaled_difference(hv, forward, 1.0)
        centered_error = _scaled_difference(hv, centered, 1.0)
        hv_norm = hv.norm()
        relative_forward = forward_error.norm() / hv_norm
        relative_centered = centered_error.norm() / hv_norm
        metrics.append(
            {
                "eps": eps,
                "hv_norm": hv_norm,
                "forward_norm": forward.norm(),
                "centered_norm": centered.norm(),
                "forward_error_norm": forward_error.norm(),
                "centered_error_norm": centered_error.norm(),
                "relative_forward_error": relative_forward,
                "relative_centered_error": relative_centered,
            }
        )
        assert relative_forward < 1.0e-8
        assert relative_centered < 1.0e-8
        for field_index in range(6):
            expected = _values(hv.function.sub(field_index))
            actual = _values(centered.function.sub(field_index))
            assert _relative_error(actual, expected) < 1.0e-8

    record_property("small_base_analytic_callback_epsilon_ladder", metrics)
    assert len(gradient_points) == 1 + 2 * len(metrics)
    for received, expected in zip(
        gradient_points, expected_gradient_points
    ):
        np.testing.assert_array_equal(received, expected)
    np.testing.assert_array_equal(_values(x.function), x_before)
    np.testing.assert_array_equal(_values(v.function), v_before)


def test_production_field_hessian_bilinear_symmetry(production_state_case):
    case = production_state_case
    objective = _objective(case)
    x = _vector(case, case.state)
    u = _vector(case, case.direction)
    v = _vector(case, case.probe)
    hu = x.clone()
    hv = x.clone()
    objective.hessVec(hu, u, x, 0.0)
    objective.hessVec(hv, v, x, 0.0)
    left = u.dot(hv)
    right = v.dot(hu)
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    assert abs(left - right) / scale < 5.0e-9
