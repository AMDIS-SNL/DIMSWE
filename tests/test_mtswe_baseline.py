"""Permanent, serial specifications for the existing moist TSWE map."""

from copy import deepcopy
import json

import numpy as np
from firedrake import assemble, norm

from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import Euler, get_timestepper


CFG = "tests/mtswe_small.cfg"
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


def _initialized_model():
    parameters = get_parameters(CFG)
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    coefficient, coefficient_sub, _ = model.get_coeff_var("coefficient")
    state, state_sub, _ = model.get_full_var("state", split_x_and_aux=True)
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)
    return parameters, logger, model, coefficient, state, state_sub, time


def _set_switch_safe_moist_state(state_sub, gravity):
    """Use qv > qsat and qc > qprecip, away from every active switch."""
    state_sub["v"].assign(0.0)
    state_sub["h"].assign(750.0)
    state_sub["S"].assign(750.0 * gravity)
    state_sub["Qv"].assign(750.0 * 0.0030)
    state_sub["Qc"].assign(750.0 * 0.0010)
    state_sub["Qr"].assign(750.0 * 0.0002)


def _integrated_moist_invariants(model, state_sub, beta2):
    total_water = assemble(
        (state_sub["Qv"] + state_sub["Qc"] + state_sub["Qr"])
        * model.spaces.dx
    )
    thermal_vapour = assemble(
        (state_sub["S"] - beta2 * state_sub["Qv"]) * model.spaces.dx
    )
    return float(total_water), float(thermal_vapour)


def _field_data(state_sub):
    return {
        name: np.array(field.dat.data_ro, copy=True)
        for name, field in state_sub.items()
    }


def test_isolated_three_way_euler_update_preserves_invariants_and_dry_fields(
    record_property,
):
    parameters, logger, model, coefficient, state, state_sub, time = (
        _initialized_model()
    )
    _set_switch_safe_moist_state(state_sub, model.initcond.g)
    coefficient_sub = {
        name: coefficient.sub(index)
        for index, name in enumerate(model.get_coeff_list())
    }
    beta2 = model.initcond.g * float(coefficient_sub["L"].dat.data_ro[0])

    result, result_sub, _ = model.get_full_var("result", split_x_and_aux=True)
    physics_euler = Euler(
        model,
        logger,
        _serial_solver_parameters(),
        terms=["threewayphysics"],
    )
    # The timestepper receives the complete mixed [gamma_r, qprecip, L]
    # coefficient vector.
    physics_euler.set_coeff(coefficient)
    before = _integrated_moist_invariants(model, state_sub, beta2)
    h_before = state_sub["h"].copy(deepcopy=True)
    v_before = state_sub["v"].copy(deepcopy=True)
    physics_euler.reset_internal_vars()
    physics_euler.take_forward_step(
        result,
        result_sub,
        state,
        time,
        parameters["timestepping"]["dt"],
    )
    after = _integrated_moist_invariants(model, result_sub, beta2)

    # Each cancellation is algebraic; 64 eps allows for mass solves and the
    # two independently assembled global integrals, not modelling error.
    np.testing.assert_allclose(after, before, rtol=64.0 * EPS, atol=0.0)
    assert norm(result_sub["h"] - h_before) == 0.0
    assert norm(result_sub["v"] - v_before) == 0.0
    assert norm(result_sub["Qv"] - state_sub["Qv"]) > 0.0
    record_property(
        "moist_invariants",
        json.dumps(
            {
                "total_water_before": before[0],
                "total_water_after": after[0],
                "total_water_difference": after[0] - before[0],
                "thermal_vapour_before": before[1],
                "thermal_vapour_after": after[1],
                "thermal_vapour_difference": after[1] - before[1],
                "h_change_l2": float(norm(result_sub["h"] - h_before)),
                "v_change_l2": float(norm(result_sub["v"] - v_before)),
                "qv_change_l2": float(
                    norm(result_sub["Qv"] - state_sub["Qv"])
                ),
            }
        ),
    )


def test_tiny_complete_mtswe_step_is_finite_and_repeatable(record_property):
    parameters, logger, model, coefficient, state, state_sub, time = (
        _initialized_model()
    )
    # The initialized vapour is saturated.  Move it strictly onto the
    # condensation branch and add cloud/rain before testing repeatability.
    state_sub["Qv"].assign(1.2 * state_sub["Qv"])
    state_sub["Qc"].project(0.0005 * state_sub["h"])
    state_sub["Qr"].project(0.0002 * state_sub["h"])
    initial_data = _field_data(state_sub)

    timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    timestepper.set_coeff(coefficient)
    first, first_sub, _ = model.get_full_var("first", split_x_and_aux=True)
    second, second_sub, _ = model.get_full_var("second", split_x_and_aux=True)

    timestepper.reset_internal_vars()
    timestepper.take_forward_step(
        first,
        first_sub,
        state,
        time,
        parameters["timestepping"]["dt"],
    )
    assert all(np.isfinite(field.dat.data_ro).all() for field in first_sub.values())

    # Confirm that the input itself was not consumed or mutated by the map.
    for name, expected in initial_data.items():
        np.testing.assert_array_equal(state_sub[name].dat.data_ro, expected)

    timestepper.reset_internal_vars()
    timestepper.take_forward_step(
        second,
        second_sub,
        state,
        time,
        parameters["timestepping"]["dt"],
    )
    maximum_difference = 0.0
    maximum_tolerance = 0.0
    for name in first_sub:
        scale = max(1.0, float(np.max(np.abs(first_sub[name].dat.data_ro))))
        tolerance = 64.0 * EPS * scale
        maximum_difference = max(
            maximum_difference,
            float(
                np.max(
                    np.abs(
                        second_sub[name].dat.data_ro
                        - first_sub[name].dat.data_ro
                    )
                )
            ),
        )
        maximum_tolerance = max(maximum_tolerance, tolerance)
        np.testing.assert_allclose(
            second_sub[name].dat.data_ro,
            first_sub[name].dat.data_ro,
            rtol=0.0,
            atol=tolerance,
        )
    record_property(
        "complete_step_repeatability",
        json.dumps(
            {
                "all_fields_finite": True,
                "maximum_absolute_difference": maximum_difference,
                "maximum_absolute_tolerance": maximum_tolerance,
            }
        ),
    )
