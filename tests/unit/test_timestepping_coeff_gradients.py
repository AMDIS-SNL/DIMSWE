"""Taylor tests of the adjoint gradient of the final-state L2 loss with respect to coefficients."""

import numpy as np
import pytest

from tests.gradient_helpers import (
    assert_repeatable,
    assert_taylor_convergence,
    flat,
    perturbed_coeffs,
    reduced_taylor_test,
    setup_problem,
)

# hyperviscosity coefficients away from the truth (c0=0.07, s=3.2 in tests/tswe.cfg)
PERTURBED_COEFFS = {'hyperviscosity': {'c0': 0.05, 's': 2.8}}
EPS_LIST = list(1e-1 * 0.5 ** np.arange(0, 8))


@pytest.mark.parametrize('nsteps', [1, 5])
def test_coeff_gradient_vanishes_at_truth(nsteps):
    setup = setup_problem(nsteps)
    problem = setup.optimizer.reduced_problem(flat(setup.coeff), opt_type='coeffs')
    assert np.all(problem.jac(problem.z0) == 0)
    assert problem.fun(problem.z0) == 0


@pytest.mark.parametrize('nsteps', [1, 5])
def test_coeff_gradient_taylor(nsteps):
    setup = setup_problem(nsteps)
    coeffs = perturbed_coeffs(setup, PERTURBED_COEFFS)

    def run_test():
        make_problem = lambda: setup.optimizer.reduced_problem(coeffs, opt_type='coeffs')
        return reduced_taylor_test(make_problem, 0.1 * coeffs, eps_list=EPS_LIST)

    _, remainders, rates = assert_repeatable(run_test)
    print(remainders, rates)
    assert_taylor_convergence(rates)
