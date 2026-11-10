"""Tests that RK stage residuals use the stage values of every variable component.

The RK residuals are built in terms of the xk variables and then mapped onto the stage values with
UFL replace(). UFL simplifies a fixed-index component of a split() vector (such as u[1] inside
rot2D(u)) directly to the underlying mixed-function component, which the ListTensor key does not
match. Before add_stage_substitution, those components silently kept the xk values; since xk's aux
part is never assigned, the Lie-Poisson Coriolis term inner(mtest, h f rot2D(u)) evaluated to zero.
"""

import pytest
from firedrake import Function, FunctionSpace, UnitSquareMesh, VectorFunctionSpace, dx, inner, replace, split
from ufl.algorithms import extract_coefficients

from dimswe.timestepping import add_stage_substitution
from dimswe.ufl_helpers import rot2D
from tests.gradient_helpers import build_model


def test_add_stage_substitution_replaces_fixed_index_components():
    """Fixed-index components of a split() vector are replaced along with the vector itself."""
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)
    W = VectorFunctionSpace(mesh, 'DG', 1) * FunctionSpace(mesh, 'DG', 1)
    old, new = Function(W), Function(W)
    old_u, old_h = split(old)
    new_u, new_h = split(new)

    # rot2D indexes u with fixed integers; inner(u, u) uses free indices; h is a scalar subspace
    form = inner(rot2D(old_u), rot2D(old_u)) * dx + inner(old_u, old_u) * old_h * dx
    mapping = {}
    add_stage_substitution(mapping, old_u, new_u)
    add_stage_substitution(mapping, old_h, new_h)
    replaced = replace(form, mapping)

    coefficients = extract_coefficients(replaced)
    assert old not in coefficients
    assert new in coefficients


@pytest.mark.parametrize('cfg', ['tests/mtswe-lp.cfg', 'tests/mtswe-cf.cfg'])
def test_rk_stage_residuals_do_not_depend_on_xk_aux_variables(cfg):
    """Every RK stage right-hand side uses the stage aux variables, never xk's (unassigned) aux part.

    This covers the LP Coriolis term rot2D(u) and the CF term rot2D(F), which index the vector aux
    variables u and F with fixed integers.
    """
    setup = build_model(cfg)
    timestepper = setup.timestepper
    _, _, rhs_Fis, rhs_Ws = timestepper.get_rhs_expr()
    xk_aux = timestepper.xk[1]
    assert xk_aux is not None, 'the test cfg should have aux variables'
    for i, rhs_Fi in enumerate(rhs_Fis):
        assert xk_aux not in extract_coefficients(rhs_Fi), f'stage {i} rhs still uses xk aux variables'
    for i, rhs_Wi in enumerate(rhs_Ws):
        if not (rhs_Wi == 0):  # same check timestepping uses for an empty aux rhs
            assert xk_aux not in extract_coefficients(rhs_Wi), f'stage {i} aux rhs still uses xk aux variables'
