"""Tests that the auxiliary-variable systems have one equation per aux block.

Two ways an aux block could be left without an equation, making direct (LU) aux solves singular:
- the Hamiltonian declared every variable space as an aux space, although dH/dx only has entries for
  the momentum/velocity and the active densities (CF and LP);
- in a LieSplit, each integrator solves for the whole aux space, but only has equations for the
  aux variables of its own terms; the others now get an identity equation (set to zero).
"""

import numpy as np
import pytest
from firedrake import assemble

from tests.gradient_helpers import build_model


@pytest.mark.parametrize('cfg', ['tests/mtswe-cf.cfg', 'tests/mtswe-lp.cfg'])
def test_hamiltonian_aux_spaces_match_aux_variables(cfg):
    """Every aux variable has exactly one aux space, and a forward step with LU aux solves succeeds."""
    setup = build_model(cfg)
    model = setup.model
    assert len(model.get_aux_spaces()) == len(model.get_aux_var_list())
    assert len(model.dynamics.hamiltonian.get_spacelist()) == len(model.dynamics.hamiltonian.get_aux_vars_list())

    # the gradient-test solver parameters use LU for the aux solves, which fails on empty blocks
    x1, x1_sub, _ = model.get_full_var('x1', split_x_and_aux=True)
    setup.timestepper.take_forward_step(x1, x1_sub, setup.x0, setup.t0, setup.dt)


def test_split_integrators_give_unowned_aux_variables_an_identity_equation():
    """An integrator gets identity equations for exactly the aux variables owned by other stages."""
    setup = build_model('tests/tswe.cfg')  # LieSplit [[model], [hyperviscosity]], aux owned by hyperviscosity
    model = setup.model
    xhat, xhat_subs = model.get_full_test_vars(split_x_and_aux=True)
    model_stage, hyperviscosity_stage = setup.timestepper.time_integrators

    owned = model.get_aux_var_list(terms=hyperviscosity_stage.terms)
    assert owned and set(owned) == set(model.get_aux_var_list())
    # the hyperviscosity stage owns every aux variable, so it needs no identity terms
    assert hyperviscosity_stage.unowned_aux_identity(hyperviscosity_stage.terms, xhat_subs) == 0
    # the model stage owns none, so every aux block gets an identity (mass) equation
    identity = model_stage.unowned_aux_identity(model_stage.terms, xhat_subs)
    assert not (identity == 0)  # same emptiness check timestepping uses for forms
    model_stage.xk[1].assign(1.0)
    blocks = assemble(identity).dat.data_ro
    assert all(np.sum(np.abs(block)) > 0 for block in blocks), 'every aux block should get an equation'

