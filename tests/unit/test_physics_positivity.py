"""Tests that one moist-physics step keeps cloud water non-negative, through the rain-production cap.

Spatially constant fields make the DG projection of the physics sources exact, so one Euler physics
step (the LieSplit physics stage) reproduces the pointwise update and can be compared with it.
"""

import numpy as np
import pytest

from tests.gradient_helpers import build_model

CFG = 'tests/mtswe-cf-physics.cfg'


def physics_step(qv_over_qsat, qc, qr=0.0):
    """Take one physics stage step from constant fields at h = H0, s = g (so qsat = q0, no topography).

    The vapour mixing ratio is qv_over_qsat * q0. Returns (setup, physics term, old means, new values)
    of Qv, Qc and Qr.
    """
    setup = build_model(CFG)
    model = setup.model
    physics_integrator = setup.timestepper.time_integrators[1]
    assert physics_integrator.terms == ['threewayphysics']
    physics = next(term for term in model.dynamics.forcing_terms if term.name == 'threewayphysics')

    state, state_sub, _ = model.get_x_var('physics_state')
    h0, g, q0 = model.initcond.H0, model.initcond.g, model.initcond.q0
    state_sub['h'].assign(h0)
    state_sub['S'].assign(h0 * g)
    for name, q in (('Qv', qv_over_qsat * q0), ('Qc', qc), ('Qr', qr)):
        state_sub[name].assign(h0 * q)

    x1, x1_sub, _ = model.get_full_var('x1', split_x_and_aux=True)
    physics_integrator.take_forward_step(x1, x1_sub, [state, setup.x0[1]], setup.t0, setup.dt)
    old = {name: float(np.mean(state_sub[name].dat.data_ro)) for name in ('Qv', 'Qc', 'Qr')}
    new = {name: x1_sub[name].dat.data_ro.copy() for name in ('Qv', 'Qc', 'Qr')}
    return setup, physics, old, new


def test_full_evaporation_with_rain_keeps_cloud_water_non_negative():
    """Subsaturated air evaporating all its cloud while qc > qprecip ends with Qc = 0, not < 0."""
    # qsat - qv = q0/2 evaporates more than qc in one step (gamma_v (qsat - qv) > qc), and qc > qprecip
    _, physics, old, new = physics_step(qv_over_qsat=0.5, qc=5.0e-4)
    assert 5.0e-4 > physics.qprecip, 'the cloud water should be above the rain threshold'
    scale = old['Qc']
    assert np.min(new['Qc']) >= -1e-12 * scale, np.min(new['Qc'])
    assert np.allclose(new['Qc'], 0.0, atol=1e-12 * scale)
    assert np.allclose(new['Qr'], old['Qr'], atol=1e-12 * scale), 'no rain should form from evaporated cloud'
    total_old = sum(old.values())
    total_new = sum(float(np.mean(values)) for values in new.values())
    assert total_new == pytest.approx(total_old, rel=1e-12)


def test_rain_is_unchanged_when_cloud_is_not_evaporating():
    """In supersaturated air the cap is inactive: rain production is gamma_r (qc - qprecip) per step."""
    qc = 5.0e-4
    setup, physics, old, new = physics_step(qv_over_qsat=1.2, qc=qc)
    h0 = setup.model.initcond.H0
    expected_rain = h0 * physics.gamma_r * (qc - physics.qprecip)
    assert np.allclose(new['Qr'], old['Qr'] + expected_rain, rtol=1e-10)
    assert np.min(new['Qc']) > 0
