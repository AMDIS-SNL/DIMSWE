"""Specifications for the serial one-scalar dry DIMSWE PyROL adapter."""

from copy import deepcopy
import json

import numpy as np
import pytest


pytest.importorskip(
    "pyrol",
    reason="PyROL is optional; install the rol-python distribution for ROL adapter tests",
)

from pyrol import Problem, Solver
from pyrol.vectors import NumPyVector

from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.optimize import (
    L2Objective,
    Lagrangian_ODEConstrainedOptimization,
    compute_state_block,
)
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.rol_adapter import (
    ScalarC0Objective,
    bound_constrained_lbfgs_parameters,
    normalized_c0_bounds,
    numpy_c0_bounds,
)
from dimswe.timestepping import get_timestepper


pytestmark = pytest.mark.rol
CFG = "tests/tswe_rol_small.cfg"
TRUTH_C0 = 0.14
CANDIDATE_C0 = 0.05
INITIAL_C0 = 0.02
EPS = np.finfo(np.float64).eps


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
def rol_case():
    parameters = get_parameters(CFG)
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    coefficient, coefficient_sub, _ = model.get_coeff_var("coefficient")
    timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    state, state_sub, _ = model.get_full_var("initial", split_x_and_aux=True)
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)
    timestepper.set_coeff(coefficient)

    nsteps = 1
    dt = parameters["timestepping"]["dt"]
    data_blocks, _, _, time_blocks = compute_state_block(
        model, timestepper, 1, nsteps, dt, state, time
    )
    objective = L2Objective(data_blocks, time_blocks, nsteps, model.spaces.dx)
    reduced = Lagrangian_ODEConstrainedOptimization(
        model, timestepper, objective, dt
    )
    scales = model.get_coeff_scaling_factors()
    fixed_s_normalized = parameters["hyperviscosity"]["s"] / scales[0]

    # This profile is deliberately evaluated before constructing/running ROL.
    profile_c0 = np.array([0.01, 0.03, 0.05, 0.08, 0.14, 0.20, 0.30])
    profile_values = np.array(
        [
            reduced.obj(
                np.array([fixed_s_normalized, c0 / scales[1]]), None
            )
            for c0 in profile_c0
        ]
    )
    adapter = ScalarC0Objective(reduced, fixed_s_normalized)
    return {
        "parameters": parameters,
        "model": model,
        "reduced": reduced,
        "adapter": adapter,
        "scales": scales,
        "profile_c0": profile_c0,
        "profile_values": profile_values,
    }


def _control(c0, scale):
    return NumPyVector(np.array([c0 / scale], dtype=np.float64))


def test_scalar_packing_and_normalized_bounds(rol_case):
    adapter = rol_case["adapter"]
    scales = rol_case["scales"]
    z = _control(CANDIDATE_C0, scales[1])
    np.testing.assert_array_equal(
        adapter.pack_normalized_coefficients(z),
        np.array([1.0, CANDIDATE_C0 / scales[1]]),
    )

    lower, upper = normalized_c0_bounds(rol_case["model"])
    np.testing.assert_allclose(lower, 0.01 / scales[1], rtol=0.0, atol=EPS)
    np.testing.assert_allclose(upper, 2.0 / scales[1], rtol=0.0, atol=EPS)
    bounds = numpy_c0_bounds(rol_case["model"])
    np.testing.assert_allclose(bounds.getLowerBound().array, [lower])
    np.testing.assert_allclose(bounds.getUpperBound().array, [upper])


def test_profile_was_evaluated_before_rol_and_is_locally_identifiable(
    rol_case, record_property
):
    c0 = rol_case["profile_c0"]
    values = rol_case["profile_values"]
    assert np.isfinite(values).all()
    assert c0[np.argmin(values)] == TRUTH_C0
    truth_index = int(np.where(c0 == TRUTH_C0)[0][0])
    assert values[truth_index] == 0.0
    assert values[truth_index - 1] > values[truth_index]
    assert values[truth_index + 1] > values[truth_index]
    record_property(
        "c0_objective_profile",
        json.dumps(list(zip(c0.tolist(), values.tolist()))),
    )


def test_adapter_value_and_gradient_equal_existing_dimswe_paths(
    rol_case, record_property
):
    adapter = rol_case["adapter"]
    reduced = rol_case["reduced"]
    scales = rol_case["scales"]
    z = _control(CANDIDATE_C0, scales[1])
    packed = adapter.pack_normalized_coefficients(z)

    direct_value = float(reduced.obj(packed, None))
    adapter_value = adapter.value(z, 0.0)
    assert adapter_value == direct_value

    direct_gradient = np.array(reduced.jac(packed, None), copy=True)
    adapter_gradient = NumPyVector(np.array([np.nan]))
    adapter.gradient(adapter_gradient, z, 0.0)
    assert adapter_gradient.array[0] == direct_gradient[1]
    record_property(
        "direct_adapter_comparison",
        json.dumps(
            {
                "direct_value": direct_value,
                "adapter_value": adapter_value,
                "direct_c0_gradient": float(direct_gradient[1]),
                "adapter_c0_gradient": float(adapter_gradient.array[0]),
            }
        ),
    )


def test_scalar_gradient_centered_finite_difference(rol_case, record_property):
    adapter = rol_case["adapter"]
    scales = rol_case["scales"]
    z0 = CANDIDATE_C0 / scales[1]
    z = NumPyVector(np.array([z0]))
    gradient = NumPyVector(np.array([np.nan]))
    adapter.gradient(gradient, z, 0.0)

    step = 1.0e-3
    plus = adapter.value(NumPyVector(np.array([z0 + step])), 0.0)
    minus = adapter.value(NumPyVector(np.array([z0 - step])), 0.0)
    centered = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(centered, gradient.array[0], rtol=5.0e-7, atol=0.0)
    record_property(
        "centered_finite_difference",
        json.dumps(
            {
                "normalized_z": z0,
                "step": step,
                "adjoint_gradient": float(gradient.array[0]),
                "centered_gradient": centered,
                "relative_error": abs(centered - gradient.array[0])
                / abs(gradient.array[0]),
            }
        ),
    )


def test_repeated_adapter_calls_are_deterministic(rol_case):
    adapter = rol_case["adapter"]
    scales = rol_case["scales"]
    z = _control(CANDIDATE_C0, scales[1])
    values = [adapter.value(z, 0.0), adapter.value(z, 0.0)]
    gradients = []
    for _ in range(2):
        gradient = NumPyVector(np.array([np.nan]))
        adapter.gradient(gradient, z, 0.0)
        gradients.append(float(gradient.array[0]))
    np.testing.assert_allclose(values[1], values[0], rtol=64.0 * EPS, atol=0.0)
    np.testing.assert_allclose(
        gradients[1], gradients[0], rtol=64.0 * EPS, atol=0.0
    )


@pytest.fixture(scope="module")
def rol_solution(rol_case):
    adapter = rol_case["adapter"]
    scale = rol_case["scales"][1]
    z = _control(INITIAL_C0, scale)
    initial_value = adapter.value(z, 0.0)
    value_start = len(adapter.value_history)
    gradient_start = len(adapter.gradient_history)
    value_count_start = adapter.value_evaluations
    gradient_count_start = adapter.gradient_evaluations

    problem = Problem(adapter, z)
    problem.addBoundConstraint(numpy_c0_bounds(rol_case["model"]))
    solver = Solver(
        problem,
        bound_constrained_lbfgs_parameters(
            gradient_tolerance=1.0e-10,
            step_tolerance=1.0e-14,
            iteration_limit=50,
        ),
    )
    solver.solve()
    final_value = adapter.value(z, 0.0)
    state = solver.getAlgorithmState()
    return {
        "z": float(z.array[0]),
        "initial_value": initial_value,
        "final_value": final_value,
        "value_history": adapter.value_history[value_start:],
        "gradient_history": adapter.gradient_history[gradient_start:],
        "value_evaluations": adapter.value_evaluations - value_count_start,
        "gradient_evaluations": adapter.gradient_evaluations
        - gradient_count_start,
        "iterations": int(state.iter),
        "status": str(state.statusFlag),
    }


def test_bounded_rol_solve_decreases_objective_and_moves_toward_truth(
    rol_case, rol_solution, record_property
):
    lower, upper = normalized_c0_bounds(rol_case["model"])
    visited = [z for z, _ in rol_solution["value_history"]]
    assert visited
    bound_tolerance = 64.0 * EPS * max(1.0, abs(lower), abs(upper))
    assert all(
        lower - bound_tolerance <= z <= upper + bound_tolerance for z in visited
    )
    assert lower <= rol_solution["z"] <= upper
    assert rol_solution["final_value"] < rol_solution["initial_value"]

    scale = rol_case["scales"][1]
    recovered_c0 = scale * rol_solution["z"]
    assert abs(recovered_c0 - TRUTH_C0) < abs(INITIAL_C0 - TRUTH_C0)
    assert rol_solution["value_evaluations"] > 0
    assert rol_solution["gradient_evaluations"] > 0
    record_property(
        "rol_solution",
        json.dumps(
            {
                **rol_solution,
                "recovered_c0": recovered_c0,
                "normalized_bounds": [lower, upper],
            }
        ),
    )
