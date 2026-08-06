"""Focused tests for normalized-scalar production MTSWE PyROL access."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import numpy as np
import pytest


pytest.importorskip(
    "pyrol",
    reason="PyROL is optional; install rol-python for MTSWE adapter tests",
)

from pyrol import Bounds, Problem, Solver
from pyrol.pyrol.std import vector_double_t
from pyrol.vectors import NumPyVector

from dimswe.rol_adapter import (
    ScalarC0Objective,
    bound_constrained_lbfgs_parameters,
)
from dimswe.mtswe_rol_adapter import (
    MTSWE_ACTIVE_SET_SWITCHES,
    MTSWEGradientActiveSetQualificationError,
    MTSWEHVPActiveSetQualificationError,
    ProductionMTSWEScalarC0Objective,
)


pytestmark = pytest.mark.rol


class FakeSubFunction:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def assign(self, value):
        self.parent.data[self.index] = float(value)


class FakeFunction:
    """Small copy/assign-compatible stand-in for adapter contract tests."""

    def __init__(self, values, space, name="fake"):
        self.data = np.array(values, dtype=np.float64, copy=True)
        self._space = space
        self._name = name

    def copy(self, deepcopy=False):
        assert deepcopy
        return type(self)(self.data, self._space, self._name)

    def rename(self, name):
        self._name = name

    def assign(self, value):
        if isinstance(value, FakeFunction):
            self.data[:] = value.data
        else:
            self.data[:] = float(value)

    def sub(self, index):
        return FakeSubFunction(self, index)

    def function_space(self):
        return self._space


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


def _primal_caches(nsteps, diagnostic):
    moist = SimpleNamespace(
        name="moist_euler",
        cache=SimpleNamespace(active_set=diagnostic),
    )
    return tuple(SimpleNamespace(children=(moist,)) for _ in range(nsteps))


class FakeMTSWEHelper:
    def __init__(self):
        self.model = SimpleNamespace(get_coeff_list=lambda: ["s", "c0"])

    @staticmethod
    def _require_state(name, value):
        if not isinstance(value, FakeFunction) or value.function_space() != "state":
            raise TypeError(f"{name} is not in the fake state space")


class FakeChild:
    def __init__(self, coefficient):
        self.coeff = coefficient.copy(deepcopy=True)

    def set_coeff(self, coefficient):
        self.coeff.assign(coefficient)


class FakeMTSWETimestepper:
    """Quadratic physical-c0 oracle with production-shaped return objects."""

    def __init__(self, coefficient, *, margins=None, truth_c0=0.14):
        self.helper = FakeMTSWEHelper()
        self.time_integrators = tuple(
            FakeChild(coefficient) for _ in range(4)
        )
        self.diagnostic = _active_diagnostic(margins)
        self.truth_c0 = float(truth_c0)
        self.gradient_calls = 0
        self.hvp_calls = 0
        self.reset_calls = 0
        self.last_delta_c0 = None
        self.last_gradient_c0 = None
        self.last_hvp_c0 = None
        self.raise_next_gradient = False
        self.raise_next_hvp = False
        self.mutate_production_inputs = False

    def _get_mtswe_split_hvp_helper(self):
        return self.helper

    def reset_internal_vars(self):
        self.reset_calls += 1

    def set_coeff(self, coefficient):
        for child in self.time_integrators:
            child.set_coeff(coefficient)

    def _physical_c0(self):
        values = [child.coeff.data[1] for child in self.time_integrators]
        np.testing.assert_array_equal(values, np.full(4, values[0]))
        return float(values[0])

    @staticmethod
    def _factor(nsteps):
        return float(2 * nsteps + 1)

    def mtswe_terminal_least_squares_gradient(
        self, nsteps, state, t0, dt, target
    ):
        self.gradient_calls += 1
        c0 = self._physical_c0()
        self.last_gradient_c0 = c0
        if self.raise_next_gradient:
            self.raise_next_gradient = False
            raise RuntimeError("deliberate fake production gradient failure")
        factor = self._factor(nsteps)
        result = SimpleNamespace(
            objective_value=0.5 * factor * (c0 - self.truth_c0) ** 2,
            physical_c0_gradient=factor * (c0 - self.truth_c0),
            primal_caches=_primal_caches(nsteps, self.diagnostic),
        )
        if self.mutate_production_inputs:
            state.assign(91.0)
            target.assign(-37.0)
        return result

    def mtswe_terminal_least_squares_hvp(
        self,
        nsteps,
        state,
        t0,
        dt,
        target,
        delta_x0,
        delta_c0,
    ):
        self.hvp_calls += 1
        c0 = self._physical_c0()
        self.last_hvp_c0 = c0
        self.last_delta_c0 = float(delta_c0)
        np.testing.assert_array_equal(delta_x0.data, np.zeros_like(delta_x0.data))
        if self.raise_next_hvp:
            self.raise_next_hvp = False
            raise RuntimeError("deliberate fake production HVP failure")
        factor = self._factor(nsteps)
        result = SimpleNamespace(
            objective_value=0.5 * factor * (c0 - self.truth_c0) ** 2,
            physical_c0_gradient=factor * (c0 - self.truth_c0),
            physical_c0_hvp=factor * float(delta_c0),
            primal_caches=_primal_caches(nsteps, self.diagnostic),
        )
        if self.mutate_production_inputs:
            state.assign(73.0)
            target.assign(-19.0)
            delta_x0.assign(44.0)
        return result


def _fake_problem(
    *,
    nsteps=1,
    scale=0.07,
    margins=None,
    gradient_tolerances=None,
    hvp_tolerances=None,
):
    coefficient = FakeFunction([3.2, 0.11], "coefficient")
    state = FakeFunction([1.0, 2.0, 3.0], "state")
    target = FakeFunction([0.5, 1.5, 2.5], "state")
    timestepper = FakeMTSWETimestepper(coefficient, margins=margins)
    objective = ProductionMTSWEScalarC0Objective(
        timestepper,
        coefficient,
        state,
        target,
        nsteps=nsteps,
        t0=0.0,
        dt=100.0,
        c0_scale=scale,
        gradient_zero_margin_tolerances=gradient_tolerances,
        hvp_active_set_tolerances=hvp_tolerances,
    )
    return SimpleNamespace(
        coefficient=coefficient,
        state=state,
        target=target,
        timestepper=timestepper,
        objective=objective,
        scale=scale,
    )


def _vector(value):
    return NumPyVector(np.array([value], dtype=np.float64))


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


def test_std_steps_native_binding_type_length_and_order():
    expected = (1.0e-2, 1.0e-3, 1.0e-4)
    result = _std_steps(expected)

    assert type(result) is vector_double_t
    assert result.size() == len(expected)
    assert tuple(result.at(index) for index in range(result.size())) == expected


def test_base_import_does_not_require_new_optional_adapter():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockPyROL(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pyrol" or fullname.startswith("pyrol."):
                    raise ModuleNotFoundError("pyrol deliberately blocked")
                return None

        sys.meta_path.insert(0, BlockPyROL())
        import dimswe
        assert "dimswe.mtswe_rol_adapter" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_scalar_adapter_remains_the_first_order_class():
    assert ScalarC0Objective.__module__ == "dimswe.rol_adapter"
    assert "hessVec" not in ScalarC0Objective.__dict__


@pytest.mark.parametrize("nsteps", (0, 2, 4, True, 1.0))
def test_constructor_rejects_uncertified_nsteps(nsteps):
    coefficient = FakeFunction([3.2, 0.07], "coefficient")
    timestepper = FakeMTSWETimestepper(coefficient)
    kwargs = dict(
        timestepper=timestepper,
        coefficient_template=coefficient,
        fixed_initial_state=FakeFunction([1.0], "state"),
        target=FakeFunction([0.0], "state"),
        nsteps=nsteps,
        t0=0.0,
        dt=1.0,
        c0_scale=0.07,
    )
    with pytest.raises((TypeError, ValueError)):
        ProductionMTSWEScalarC0Objective(**kwargs)


@pytest.mark.parametrize("scale", (0.0, -1.0, np.inf, -np.inf, np.nan))
def test_constructor_rejects_nonpositive_or_nonfinite_scale(scale):
    coefficient = FakeFunction([3.2, 0.07], "coefficient")
    timestepper = FakeMTSWETimestepper(coefficient)
    with pytest.raises(ValueError):
        ProductionMTSWEScalarC0Objective(
            timestepper,
            coefficient,
            FakeFunction([1.0], "state"),
            FakeFunction([0.0], "state"),
            nsteps=1,
            t0=0.0,
            dt=1.0,
            c0_scale=scale,
        )


@pytest.mark.parametrize(
    "hvp_tolerances",
    (
        {"unknown_switch": 1.0},
        {"rain": -1.0},
        {"rain": np.inf},
    ),
)
def test_constructor_rejects_invalid_per_switch_tolerances(
    hvp_tolerances,
):
    with pytest.raises((ValueError, TypeError)):
        _fake_problem(hvp_tolerances=hvp_tolerances)


def test_one_entry_vector_validation_and_output_alias_rejection():
    objective = _fake_problem().objective
    two = NumPyVector(np.array([1.0, 2.0]))
    one = _vector(1.0)
    with pytest.raises(TypeError, match="one-element"):
        objective.value(two, 0.0)
    with pytest.raises(TypeError, match="one-element"):
        objective.gradient(two, one, 0.0)
    with pytest.raises(TypeError, match="one-element"):
        objective.hessVec(two, one, one, 0.0)
    with pytest.raises(TypeError, match="one-element"):
        objective.update(two)
    with pytest.raises(ValueError, match="must not alias"):
        objective.gradient(one, one, 0.0)
    with pytest.raises(ValueError, match="must not alias"):
        objective.hessVec(one, _vector(1.0), one, 0.0)

    shared = np.array([1.0], dtype=np.float64)
    shared_input = NumPyVector(shared)
    shared_output = NumPyVector(shared)
    if np.shares_memory(shared_input.array, shared_output.array):
        with pytest.raises(ValueError, match="must not alias"):
            objective.gradient(shared_output, shared_input, 0.0)


def test_constructor_owns_state_target_and_coefficient_template_copies():
    case = _fake_problem()
    owned_state = case.objective.fixed_initial_state.data.copy()
    owned_target = case.objective.target.data.copy()
    owned_coefficient = case.objective.coefficient_template.data.copy()

    case.state.assign(99.0)
    case.target.assign(-88.0)
    case.coefficient.assign(77.0)

    np.testing.assert_array_equal(
        case.objective.fixed_initial_state.data, owned_state
    )
    np.testing.assert_array_equal(case.objective.target.data, owned_target)
    np.testing.assert_array_equal(
        case.objective.coefficient_template.data, owned_coefficient
    )


def test_fake_normalized_value_gradient_and_hvp_scaling():
    case = _fake_problem(nsteps=3)
    z = _vector(1.5)
    qz = _vector(-0.25)
    gradient = _vector(np.nan)
    hvp = _vector(np.nan)
    factor = case.timestepper._factor(3)
    physical_c0 = case.scale * z.array[0]

    value = case.objective.value(z, 0.0)
    case.objective.gradient(gradient, z, 0.0)
    case.objective.hessVec(hvp, qz, z, 0.0)

    assert value == 0.5 * factor * (physical_c0 - 0.14) ** 2
    assert gradient.array[0] == (
        case.scale * factor * (physical_c0 - 0.14)
    )
    assert case.timestepper.last_delta_c0 == case.scale * qz.array[0]
    assert hvp.array[0] == case.scale**2 * factor * qz.array[0]


def test_bounded_single_entry_caches_and_update():
    case = _fake_problem()
    objective = case.objective
    z = _vector(1.0)
    qz = _vector(0.5)
    gradient = _vector(np.nan)
    hvp = _vector(np.nan)

    cached_value = objective.value(z, 0.0)
    objective.gradient(gradient, z, 0.0)
    assert case.timestepper.gradient_calls == 1

    objective.hessVec(hvp, qz, z, 0.0)
    objective.hessVec(hvp, qz, z, 0.0)
    assert case.timestepper.hvp_calls == 1
    assert objective.value(z, 0.0) == cached_value
    assert case.timestepper.gradient_calls == 1

    objective.hessVec(hvp, _vector(0.25), z, 0.0)
    assert case.timestepper.hvp_calls == 2
    assert objective.cache_info["hvp_qz"] == 0.25
    objective.gradient(gradient, z, 0.0)
    assert case.timestepper.gradient_calls == 1

    objective.value(_vector(1.25), 0.0)
    assert case.timestepper.gradient_calls == 2
    assert not objective.cache_info["has_hvp_result"]

    objective.update(_vector(1.25))
    assert objective.cache_info == {
        "point_z": None,
        "has_point_result": False,
        "hvp_z": None,
        "hvp_qz": None,
        "has_hvp_result": False,
    }

    objective.hessVec(hvp, qz, _vector(1.25), 0.0)
    objective.gradient(gradient, _vector(1.25), 0.0)
    assert np.isfinite(objective.value(_vector(1.25), 0.0))
    assert case.timestepper.gradient_calls == 2
    assert case.timestepper.hvp_calls == 3


def test_input_nonmutation_restoration_exception_safety_and_repeatability():
    case = _fake_problem()
    case.timestepper.mutate_production_inputs = True
    objective = case.objective
    z = _vector(1.3)
    qz = _vector(-0.2)
    z_before = z.array.copy()
    qz_before = qz.array.copy()
    state_before = case.state.data.copy()
    target_before = case.target.data.copy()
    coefficient_before = case.coefficient.data.copy()
    for index, child in enumerate(case.timestepper.time_integrators):
        child.coeff.data[1] = 0.03 * (index + 1)
    child_coefficients_before = tuple(
        child.coeff.data.copy() for child in case.timestepper.time_integrators
    )

    first_gradient = _vector(np.nan)
    first_hvp = _vector(np.nan)
    objective.gradient(first_gradient, z, 0.0)
    objective.hessVec(first_hvp, qz, z, 0.0)
    objective.update(z)
    repeated_gradient = _vector(np.nan)
    repeated_hvp = _vector(np.nan)
    objective.gradient(repeated_gradient, z, 0.0)
    objective.hessVec(repeated_hvp, qz, z, 0.0)

    np.testing.assert_array_equal(z.array, z_before)
    np.testing.assert_array_equal(qz.array, qz_before)
    np.testing.assert_array_equal(case.state.data, state_before)
    np.testing.assert_array_equal(case.target.data, target_before)
    np.testing.assert_array_equal(case.coefficient.data, coefficient_before)
    for child, before in zip(
        case.timestepper.time_integrators, child_coefficients_before
    ):
        np.testing.assert_array_equal(child.coeff.data, before)
    np.testing.assert_array_equal(repeated_gradient.array, first_gradient.array)
    np.testing.assert_array_equal(repeated_hvp.array, first_hvp.array)

    objective.update(z)
    case.timestepper.raise_next_gradient = True
    reset_calls = case.timestepper.reset_calls
    with pytest.raises(RuntimeError, match="deliberate"):
        objective.value(z, 0.0)
    assert case.timestepper.reset_calls == reset_calls + 2
    for child, before in zip(
        case.timestepper.time_integrators, child_coefficients_before
    ):
        np.testing.assert_array_equal(child.coeff.data, before)

    objective.update(z)
    case.timestepper.raise_next_hvp = True
    reset_calls = case.timestepper.reset_calls
    with pytest.raises(RuntimeError, match="deliberate"):
        objective.hessVec(_vector(np.nan), qz, z, 0.0)
    assert case.timestepper.reset_calls == reset_calls + 2
    for child, before in zip(
        case.timestepper.time_integrators, child_coefficients_before
    ):
        np.testing.assert_array_equal(child.coeff.data, before)


def test_value_records_but_does_not_reject_zero_active_margin():
    case = _fake_problem(margins={"rain": 0.0})
    assert np.isfinite(case.objective.value(_vector(1.0), 0.0))
    report = case.objective.last_value_active_set_report
    assert not report.qualified
    failure = report.failures[0]
    assert (failure.timestep, failure.switch) == (0, "rain")
    assert failure.minimum_margin == 0.0
    assert failure.configured_threshold == 0.0


def test_gradient_policy_is_weaker_than_hvp_policy():
    hvp_tolerances = {switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES}
    hvp_tolerances["evaporation_cap"] = 0.2
    case = _fake_problem(
        margins={"evaporation_cap": 0.1},
        hvp_tolerances=hvp_tolerances,
    )
    z = _vector(1.0)
    gradient = _vector(np.nan)
    case.objective.gradient(gradient, z, 0.0)
    assert np.isfinite(gradient.array[0])
    assert case.objective.last_gradient_active_set_report.qualified

    with pytest.raises(MTSWEHVPActiveSetQualificationError) as raised:
        case.objective.hessVec(_vector(np.nan), _vector(0.5), z, 0.0)
    failure = raised.value.report.failures[0]
    assert failure.timestep == 0
    assert failure.switch == "evaporation_cap"
    assert failure.minimum_margin == 0.1
    assert failure.configured_threshold == 0.2
    assert "timestep=0" in str(raised.value)
    assert "switch=evaporation_cap" in str(raised.value)


def test_actual_zero_margin_gradient_ambiguity_is_rejected():
    case = _fake_problem(margins={"condensation": 0.0})
    with pytest.raises(MTSWEGradientActiveSetQualificationError) as raised:
        case.objective.gradient(_vector(np.nan), _vector(1.0), 0.0)
    failure = raised.value.report.failures[0]
    assert failure.switch == "condensation"
    assert failure.minimum_margin == 0.0
    assert failure.configured_threshold == 0.0


def test_configured_machine_zero_gradient_ambiguity_is_rejected():
    case = _fake_problem(
        margins={"rain": 1.0e-13},
        gradient_tolerances={"rain": 1.0e-12},
    )
    with pytest.raises(MTSWEGradientActiveSetQualificationError) as raised:
        case.objective.gradient(_vector(np.nan), _vector(1.0), 0.0)
    failure = raised.value.report.failures[0]
    assert failure.switch == "rain"
    assert failure.minimum_margin == 1.0e-13
    assert failure.configured_threshold == 1.0e-12


def test_safe_hvp_active_set_is_accepted_and_reported_for_every_timestep():
    tolerances = {
        switch: 0.5 for switch in MTSWE_ACTIVE_SET_SWITCHES
    }
    case = _fake_problem(nsteps=3, hvp_tolerances=tolerances)
    case.objective.hessVec(
        _vector(np.nan), _vector(0.5), _vector(1.0), 0.0
    )
    report = case.objective.last_hvp_active_set_report
    assert report.qualified
    assert len(report.entries) == 3 * len(MTSWE_ACTIVE_SET_SWITCHES)
    assert {entry.timestep for entry in report.entries} == {0, 1, 2}


def test_pyrol_explicit_step_derivative_utilities_call_scalar_hessvec():
    case = _fake_problem(
        hvp_tolerances={
            switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES
        }
    )
    objective = case.objective
    x = _vector(1.4)
    direction = _vector(0.3)
    steps = _std_steps((1.0e-2, 1.0e-3, 1.0e-4))

    gradient_rows = _std_rows(
        objective.checkGradient(x, direction, steps, False)
    )
    hessian_rows = _std_rows(
        objective.checkHessVec(x, direction, steps, False)
    )
    symmetry = objective.checkHessSym(
        x, direction, _vector(-0.2), False
    )

    assert len(gradient_rows) == len(hessian_rows) == 3
    assert gradient_rows[-1][-1] < gradient_rows[0][-1]
    assert hessian_rows[-1][-1] < 1.0e-9
    assert symmetry.size() >= 3
    assert abs(symmetry[symmetry.size() - 1]) < 1.0e-12
    assert case.timestepper.hvp_calls > 0


def test_deterministic_bounded_lbfgs_optimization_smoke_is_not_hvp_test():
    case = _fake_problem()
    objective = case.objective
    z = _vector(0.25)
    initial_value = objective.value(z, 0.0)
    problem = Problem(objective, z)
    problem.addBoundConstraint(Bounds(_vector(0.0), _vector(3.0)))
    solver = Solver(
        problem,
        bound_constrained_lbfgs_parameters(
            gradient_tolerance=1.0e-12,
            step_tolerance=1.0e-14,
            iteration_limit=30,
        ),
    )
    solver.solve()
    final_value = objective.value(z, 0.0)

    assert final_value < initial_value
    assert 0.0 <= z.array[0] <= 3.0
    np.testing.assert_allclose(z.array[0], 0.14 / case.scale, atol=1.0e-8)
    # This is only optimization wiring coverage.  HVP coverage is supplied by
    # the direct and PyROL derivative-utility tests above.
    assert case.timestepper.hvp_calls == 0


# The remaining tests use the accepted production implementation as the
# oracle.  One module-scoped 2-by-2 serial case is shared across all checks;
# no child-stage or full certification ladder is repeated here.
from firedrake import COMM_SELF, SpatialCoordinate, as_vector, cos, pi, sin

import dimswe.meshes as dimswe_meshes
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import get_timestepper


PRODUCTION_CFG = "tests/mtswe_small.cfg"
PRODUCTION_C0 = 0.07
PHYSICAL_DIRECTION = 0.012


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


@pytest.fixture(scope="module")
def production_scalar_case():
    parameters = get_parameters(PRODUCTION_CFG)
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
        "mtswe_rol_production_coefficient"
    )
    state_container, state_sub, _ = model.get_full_var(
        "mtswe_rol_production_state", split_x_and_aux=True
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

    target = model.get_x_var("mtswe_rol_production_target")[0]
    target.assign(0.985 * state_container[0])
    timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    timestepper.set_coeff(coefficient)
    scale = float(model.get_coeff_scaling_factors()[1])
    assert scale == 0.07
    return SimpleNamespace(
        model=model,
        timestepper=timestepper,
        coefficient=coefficient,
        state=state_container[0],
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


def _production_objective(case, nsteps):
    return ProductionMTSWEScalarC0Objective(
        case.timestepper,
        case.coefficient,
        case.state,
        case.target,
        nsteps=nsteps,
        t0=case.t0,
        dt=case.dt,
        c0_scale=case.scale,
        hvp_active_set_tolerances={
            switch: 0.0 for switch in MTSWE_ACTIVE_SET_SWITCHES
        },
    )


def _zero_state(case, name):
    result = case.model.get_x_var(name)[0]
    result.assign(0)
    return result


@pytest.mark.parametrize("nsteps", (1, 3))
def test_production_normalized_value_and_gradient(
    production_scalar_case, nsteps
):
    case = production_scalar_case
    objective = _production_objective(case, nsteps)
    z = _vector(PRODUCTION_C0 / case.scale)
    physical_c0 = case.scale * z.array[0]
    with _physical_c0(case, physical_c0):
        direct = case.timestepper.mtswe_terminal_least_squares_gradient(
            nsteps, case.state, case.t0, case.dt, case.target
        )

    adapter_value = objective.value(z, 0.0)
    adapter_gradient = _vector(np.nan)
    objective.gradient(adapter_gradient, z, 0.0)
    assert adapter_value == direct.objective_value
    np.testing.assert_allclose(
        adapter_gradient.array[0],
        case.scale * direct.physical_c0_gradient,
        rtol=2.0e-13,
        atol=0.0,
    )
    assert objective.production_gradient_evaluations == 1


@pytest.mark.parametrize("nsteps", (1, 3))
def test_production_normalized_hessvec_scaling(
    production_scalar_case, nsteps
):
    case = production_scalar_case
    objective = _production_objective(case, nsteps)
    z = _vector(PRODUCTION_C0 / case.scale)
    qz = _vector(PHYSICAL_DIRECTION / case.scale)
    delta_c0 = case.scale * qz.array[0]
    zero = _zero_state(case, f"mtswe_rol_zero_{nsteps}")
    with _physical_c0(case, case.scale * z.array[0]):
        direct = case.timestepper.mtswe_terminal_least_squares_hvp(
            nsteps,
            case.state,
            case.t0,
            case.dt,
            case.target,
            zero,
            delta_c0,
        )

    adapter_hvp = _vector(np.nan)
    objective.hessVec(adapter_hvp, qz, z, 0.0)
    np.testing.assert_allclose(
        adapter_hvp.array[0],
        case.scale * direct.physical_c0_hvp,
        rtol=2.0e-13,
        atol=0.0,
    )
    physical_scalar_block = direct.physical_c0_hvp / delta_c0
    np.testing.assert_allclose(
        adapter_hvp.array[0] / qz.array[0],
        case.scale**2 * physical_scalar_block,
        rtol=2.0e-13,
        atol=0.0,
    )


def test_production_normalized_hessvec_centered_safe_gradients(
    production_scalar_case,
):
    case = production_scalar_case
    objective = _production_objective(case, 1)
    z0 = PRODUCTION_C0 / case.scale
    qz_value = PHYSICAL_DIRECTION / case.scale
    z = _vector(z0)
    qz = _vector(qz_value)
    exact = _vector(np.nan)
    objective.hessVec(exact, qz, z, 0.0)
    base_signatures = objective.last_hvp_active_set_report.signatures

    errors = []
    for epsilon in (0.05, 0.025, 0.0125):
        plus_gradient = _vector(np.nan)
        minus_gradient = _vector(np.nan)
        objective.gradient(
            plus_gradient, _vector(z0 + epsilon * qz_value), 0.0
        )
        plus_signatures = objective.last_gradient_active_set_report.signatures
        objective.gradient(
            minus_gradient, _vector(z0 - epsilon * qz_value), 0.0
        )
        minus_signatures = objective.last_gradient_active_set_report.signatures
        assert plus_signatures == base_signatures
        assert minus_signatures == base_signatures
        centered = (
            plus_gradient.array[0] - minus_gradient.array[0]
        ) / (2.0 * epsilon)
        errors.append(abs(centered - exact.array[0]))

    scale = max(abs(exact.array[0]), np.finfo(float).tiny)
    assert min(errors) / scale < 2.0e-6
