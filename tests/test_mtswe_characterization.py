"""Non-gating observations of scientifically unresolved MTSWE behavior."""

from copy import deepcopy
import json

import numpy as np
import pytest
from firedrake import SpatialCoordinate, as_vector, assemble, cos, norm, pi, sin

from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import Euler, get_timestepper


pytestmark = pytest.mark.characterization
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


def _initialized_model(parameters=None):
    if parameters is None:
        parameters = get_parameters(CFG)
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    coefficient, coefficient_sub, _ = model.get_coeff_var("coefficient")
    state, state_sub, _ = model.get_full_var("state", split_x_and_aux=True)
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)
    return parameters, logger, model, coefficient, state, state_sub, time


def _switch_safe_moist_state(state_sub, gravity):
    state_sub["v"].assign(0.0)
    state_sub["h"].assign(750.0)
    state_sub["S"].assign(750.0 * gravity)
    state_sub["Qv"].assign(750.0 * 0.0030)
    state_sub["Qc"].assign(750.0 * 0.0010)
    state_sub["Qr"].assign(750.0 * 0.0002)


def _take_step(integrator, model, state, time, dt, name):
    result, result_sub, _ = model.get_full_var(name, split_x_and_aux=True)
    integrator.reset_internal_vars()
    integrator.take_forward_step(result, result_sub, state, time, dt)
    assert all(np.isfinite(field.dat.data_ro).all() for field in result_sub.values())
    return result, result_sub


def test_active_timestep_limiter_hook_observation(monkeypatch, record_property):
    parameters, logger, model, coefficient, state, state_sub, time = (
        _initialized_model()
    )
    state_sub["Qv"].assign(1.2 * state_sub["Qv"])
    state_sub["Qc"].project(0.0005 * state_sub["h"])
    state_sub["Qr"].project(0.0002 * state_sub["h"])
    limiter = next(
        term for term in model.dynamics.forcing_terms if term.name == "dg1limiter"
    )
    original_post_step = limiter.post_step
    calls = []

    def counted_post_step(statevars):
        calls.append(tuple(sorted(statevars)))
        return original_post_step(statevars)

    monkeypatch.setattr(limiter, "post_step", counted_post_step)
    timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    timestepper.set_coeff(coefficient)
    _take_step(
        timestepper,
        model,
        state,
        time,
        parameters["timestepping"]["dt"],
        "limited_observation",
    )

    observation = "called" if calls else "not-called"
    record_property("limiter_post_step", observation)
    assert observation in {"called", "not-called"}


def test_hamiltonian_topography_initialization_observation(record_property):
    parameters = get_parameters(CFG)
    parameters["initial-conditions"]["name"] = "TC5"
    _, _, model, _, _, _, _ = _initialized_model(parameters)

    dynamics_norm = float(norm(model.dynamics.bottom_topography))
    hamiltonian_norm = float(norm(model.dynamics.hamiltonian.bottom_topography))
    threshold = 256.0 * EPS * max(1.0, dynamics_norm)
    observation = (
        "initialized-nonzero"
        if hamiltonian_norm > threshold
        else "zero-to-roundoff"
    )
    record_property(
        "hamiltonian_topography",
        json.dumps(
            {
                "observation": observation,
                "dynamics_l2": dynamics_norm,
                "hamiltonian_l2": hamiltonian_norm,
            }
        ),
    )
    assert np.isfinite([dynamics_norm, hamiltonian_norm]).all()
    assert dynamics_norm > 0.0
    assert observation in {"initialized-nonzero", "zero-to-roundoff"}


def _moist_increment(configured_dt, applied_dt):
    parameters = get_parameters(CFG)
    parameters["timestepping"]["dt"] = configured_dt
    parameters, logger, model, coefficient, state, state_sub, time = (
        _initialized_model(parameters)
    )
    _switch_safe_moist_state(state_sub, model.initcond.g)
    before = float(assemble(state_sub["Qv"] * model.spaces.dx))
    physics = Euler(
        model,
        logger,
        _serial_solver_parameters(),
        terms=["threewayphysics"],
    )
    physics.set_coeff(coefficient)
    _, result_sub = _take_step(
        physics,
        model,
        state,
        time,
        applied_dt,
        f"physics_configured_{configured_dt}_applied_{applied_dt}",
    )
    after = float(assemble(result_sub["Qv"] * model.spaces.dx))
    return after - before


def test_moist_conversion_applied_to_configured_dt_scaling_observation(
    record_property,
):
    cases = {
        "configured_100_applied_100": _moist_increment(100.0, 100.0),
        "configured_100_applied_50": _moist_increment(100.0, 50.0),
        "configured_50_applied_100": _moist_increment(50.0, 100.0),
    }
    reference = cases["configured_100_applied_100"]
    ratios = {name: increment / reference for name, increment in cases.items()}

    # On this fixed active branch every conversion rate has a configured-dt
    # denominator, while Euler supplies the applied-dt numerator.  Thus the
    # increment is proportional to applied_dt/configured_dt.
    record_property(
        "moist_dt_scaling",
        json.dumps(
            {
                "increments": cases,
                "ratios_to_configured_100_applied_100": ratios,
            }
        ),
    )
    assert np.isfinite(list(cases.values()) + list(ratios.values())).all()
    assert all(increment != 0.0 for increment in cases.values())


def test_isolated_dg_transport_qr_observation(record_property):
    parameters, logger, model, coefficient, state, state_sub, time = (
        _initialized_model()
    )
    x = SpatialCoordinate(model.mesh)
    state_sub["v"].project(as_vector([10.0, 0.0]))
    state_sub["Qr"].project(
        0.2 + 0.05 * cos(2.0 * pi * x[0] / model.initcond.Lx)
    )
    transport = Euler(
        model,
        logger,
        _serial_solver_parameters(),
        terms=["dg1limiter"],
    )
    transport.set_coeff(coefficient)
    _, result_sub = _take_step(
        transport,
        model,
        state,
        time,
        parameters["timestepping"]["dt"],
        "transported_qr",
    )
    initial_norm = float(norm(state_sub["Qr"]))
    change_norm = float(norm(result_sub["Qr"] - state_sub["Qr"]))
    threshold = 256.0 * EPS * max(initial_norm, 1.0)
    observation = "modified" if change_norm > threshold else "unchanged"
    record_property(
        "isolated_qr_transport",
        json.dumps(
            {
                "observation": observation,
                "initial_l2": initial_norm,
                "change_l2": change_norm,
            }
        ),
    )
    assert np.isfinite([initial_norm, change_norm]).all()
    assert observation in {"modified", "unchanged"}


def test_hyperviscosity_fourier_amplification_observation(record_property):
    parameters, logger, model, coefficient, state, state_sub, time = (
        _initialized_model()
    )
    x = SpatialCoordinate(model.mesh)
    mode = sin(2.0 * pi * x[0] / model.initcond.Lx)
    state_sub["v"].assign(0.0)
    state_sub["h"].assign(750.0)
    state_sub["S"].project(750.0 * model.initcond.g + 100.0 * mode)
    mode_mass = float(assemble(mode * mode * model.spaces.dx))
    initial_amplitude = float(assemble(state_sub["S"] * mode * model.spaces.dx))
    initial_amplitude /= mode_mass

    hyperviscosity = Euler(
        model,
        logger,
        _serial_solver_parameters(),
        terms=["hyperviscosity"],
    )
    hyperviscosity.set_coeff(coefficient)
    _, result_sub = _take_step(
        hyperviscosity,
        model,
        state,
        time,
        parameters["timestepping"]["dt"],
        "hyperviscous_mode",
    )
    final_amplitude = float(assemble(result_sub["S"] * mode * model.spaces.dx))
    final_amplitude /= mode_mass
    amplification = final_amplitude / initial_amplitude
    if amplification < 1.0 - 512.0 * EPS:
        observation = "damped"
    elif amplification > 1.0 + 512.0 * EPS:
        observation = "amplified"
    else:
        observation = "neutral-to-roundoff"
    record_property(
        "hyperviscosity_mode",
        json.dumps(
            {
                "observation": observation,
                "initial_amplitude": initial_amplitude,
                "final_amplitude": final_amplitude,
                "amplification": amplification,
            }
        ),
    )
    assert np.isfinite([initial_amplitude, final_amplitude, amplification]).all()
    assert initial_amplitude != 0.0
    assert observation in {"damped", "amplified", "neutral-to-roundoff"}
