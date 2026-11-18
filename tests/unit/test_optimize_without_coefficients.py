"""Gradient and HVP tests for a model without coefficients.

The workflow's ic-params and ic-field targets run the model with every forcing term's
treat_as_coeffs false (as in mtswe.cfg), so the model has no coefficient space and
model.get_coeff_var returns None. The other tests use tests/tswe_small.cfg, which sets
hyperviscosity.treat_as_coeffs true, so this case was untested. LieSplittingIntegrator used to
assign to its None coefficient-gradient variable, and every first- and second-order evaluation
failed with "'NoneType' object has no attribute 'assign'".

These tests run tests/tswe_small.cfg with treat_as_coeffs false, through the LieSplit timestepper, for
the initial-state and initial-condition-parameter targets.
"""

import numpy as np
import pytest
import yaml

from dimswe.optimize import ParametricInitialCondition, taylor_test

from tests.gradient_helpers import (
    DEFAULT_CFG,
    assert_repeatable,
    assert_taylor_convergence,
    flat,
    reduced_hessian_taylor_test,
    setup_problem,
)

NSTEPS = 3
OBS_STEPS = [1, 3]


@pytest.fixture
def no_coeff_cfg(tmp_path):
    """Write tests/tswe_small.cfg with every treat_as_coeffs flag false and return its path."""
    with open(DEFAULT_CFG, "r") as handle:
        parameters = yaml.safe_load(handle)
    for section in parameters.values():
        if isinstance(section, dict) and "treat_as_coeffs" in section:
            section["treat_as_coeffs"] = False
    path = tmp_path / "tswe_no_coeffs.cfg"
    path.write_text(yaml.safe_dump(parameters), encoding="utf-8")
    return str(path)


def test_model_has_no_coefficients(no_coeff_cfg):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS, cfg=no_coeff_cfg)
    assert not setup.model.has_coeff()
    assert setup.model.get_coeff_size() == 0
    assert setup.coeff is None
    assert setup.parameters["timestepping"]["method"] == "LieSplit"


def test_ic_gradient_taylor_without_coefficients(no_coeff_cfg):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS, cfg=no_coeff_cfg)
    x0 = flat(setup.x0[0])
    direction = np.random.default_rng(30).standard_normal(x0.size)
    make_problem = lambda: setup.optimizer.reduced_problem(x0 * 1.01, opt_type="ics", coeffs0=None,
        scaling_factors=np.abs(x0) + 1.0)
    _, remainders, rates = assert_repeatable(lambda: taylor_test(make_problem(), direction))
    print(remainders, rates)
    assert_taylor_convergence(rates)


def test_ic_hessp_taylor_without_coefficients(no_coeff_cfg):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS, cfg=no_coeff_cfg)
    x0 = flat(setup.x0[0])
    direction = np.random.default_rng(31).standard_normal(x0.size)
    make_problem = lambda: setup.optimizer.reduced_problem(x0 * 1.01, opt_type="ics", coeffs0=None,
        scaling_factors=np.abs(x0) + 1.0)
    _, remainders, rates, change = assert_repeatable(lambda: reduced_hessian_taylor_test(make_problem, direction))
    print(remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)


def test_ic_params_gradient_and_hessp_without_coefficients(no_coeff_cfg):
    setup = setup_problem(NSTEPS, obs_steps=OBS_STEPS, cfg=no_coeff_cfg)
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ["ox", "oy"])
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
    make_problem = lambda: setup.optimizer.reduced_problem(params, opt_type="ic-params", coeffs0=None,
        ic_map=ic_map, scaling_factors=np.array([0.1, 0.1]))
    direction = np.array([1.0, 0.5])

    _, remainders, rates = taylor_test(make_problem(), direction)
    print("gradient", remainders, rates)
    assert_taylor_convergence(rates)

    _, remainders, rates, change = reduced_hessian_taylor_test(make_problem, direction)
    print("hessp", remainders, rates)
    assert_taylor_convergence(rates, remainders=remainders, change=change)
