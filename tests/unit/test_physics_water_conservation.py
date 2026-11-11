"""Tests that the moist physics source terms conserve total water (Qv + Qc + Qr) exactly.

The vapour, cloud and rain sources cancel pointwise, so the assembled sources must sum to zero over
the Qv, Qc and Qr blocks, provided all of them use the same quadrature rule. Previously each source
integral got its own estimated quadrature degree, and water leaked once rain switched on in the
CF and LP models (H1 conserved, since its lumped-mass measure is one fixed scheme).
"""

import numpy as np
import pytest
from firedrake import SpatialCoordinate, assemble, cos, pi, sin

from tests.gradient_helpers import build_model


def physics_term(model):
    """Return the threewayphysics forcing term of a model."""
    return next(term for term in model.dynamics.forcing_terms if term.name == 'threewayphysics')


@pytest.mark.parametrize('cfg', ['tests/mtswe-cf-physics.cfg', 'tests/mtswe-lp-physics.cfg'])
def test_physics_sources_conserve_total_water(cfg):
    """Sum of the assembled Qv, Qc, Qr physics sources is zero to round-off, with rain active."""
    setup = build_model(cfg)
    model = setup.model
    physics = physics_term(model)
    x, x_sub, x_split = model.get_x_var('physics_state')
    _, xhats = model.get_x_test_vars()

    # A state that switches every branch inside cells: vapour oscillating around saturation
    # (condensation and evaporation) and cloud water oscillating around qprecip (rain on/off).
    X, Y = SpatialCoordinate(model.mesh)
    L = model.initcond.Lx
    h0, g = model.initcond.H0, model.initcond.g
    qsat0 = model.initcond.q0  # qsat at h = H0, s = g and zero topography
    wave = sin(2 * pi * X / L) * cos(2 * pi * Y / L)
    x_sub['h'].interpolate(h0 + 10.0 * wave)
    x_sub['S'].interpolate((h0 + 10.0 * wave) * g)
    x_sub['Qv'].interpolate(h0 * qsat0 * (1.0 + 0.05 * sin(6 * pi * X / L)))
    x_sub['Qc'].interpolate(h0 * physics.qprecip * (1.0 + 0.8 * sin(6 * pi * Y / L)))
    x_sub['Qr'].interpolate(h0 * physics.qprecip * 0.5)

    sources = assemble(physics.rhs(x_split, setup.t0, None, xhats))
    names = model.get_x_var_list()
    blocks = [sources.dat[names.index(name)].data_ro for name in ('Qv', 'Qc', 'Qr')]
    total = sum(float(np.sum(block)) for block in blocks)   # = int (Sv + Sc + Sr) h dx (partition of unity)
    scale = sum(float(np.sum(np.abs(block))) for block in blocks)

    assert float(np.sum(np.abs(blocks[2]))) > 0.0, 'rain source should be active in this state'
    assert abs(total) <= 1e-12 * scale, (total, scale)
