"""Taylor tests of the adjoint gradient of the final-state L2 loss with respect to the initial state."""

import numpy as np
import pytest

from tests.gradient_helpers import (
    assert_repeatable,
    assert_taylor_convergence,
    flat,
    perturbed_initial_state,
    reduced_taylor_test,
    setup_problem,
)

# double-vortex offsets away from the truth (ox=oy=0.1 in tests/tswe.cfg)
PERTURBED_IC = {'ox': 0.11, 'oy': 0.11}


@pytest.mark.parametrize('nsteps', [1, 5])
def test_ic_gradient_vanishes_at_truth(nsteps):
    setup = setup_problem(nsteps)
    problem = setup.optimizer.reduced_problem(flat(setup.x0[0]), opt_type='ics', coeffs0=setup.coeff)
    assert np.all(problem.jac(problem.z0) == 0)
    assert problem.fun(problem.z0) == 0


@pytest.mark.parametrize('nsteps', [1, 5])
def test_ic_gradient_taylor(nsteps):
    setup = setup_problem(nsteps)
    x0_new = perturbed_initial_state(setup, PERTURBED_IC)

    def run_test():
        make_problem = lambda: setup.optimizer.reduced_problem(x0_new, opt_type='ics', coeffs0=setup.coeff)
        return reduced_taylor_test(make_problem, 0.05 * x0_new)

    _, remainders, rates = assert_repeatable(run_test)
    print(remainders, rates)
    assert_taylor_convergence(rates)
