"""Shared setup and Taylor-test helpers for the gradient unit tests.

Every gradient test follows the same pattern:

1. build a model, timestepper and initial state with build_model (or a full adjoint problem,
   including a truth trajectory and optimizer, with setup_problem),
2. compute Taylor remainders, with dimswe.optimize.taylor_test for objective gradients or with
   form_taylor_test for UFL derivatives of residual forms,
3. check them with assert_taylor_convergence.

All remainders are first-order (forward) Taylor remainders, |f(x + eps d) - f(x) - eps df(x).d|,
which converge at rate 2 for a correct derivative. The tests are meant to run in serial.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function
from dimswe.optimize import (
    L2Objective,
    Lagrangian_ODEConstrainedOptimization,
    ObservationL2Objective,
    compute_states,
    create_states,
)
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import get_timestepper

from firedrake import action, assemble, replace

DEFAULT_CFG = 'tests/tswe.cfg'

# Direct solves make the forward and adjoint steps consistent to round-off, so Taylor remainders
# converge cleanly. A copy is used so the package-wide defaults are not modified.
SOLVER_PARAMETERS = overall_solver_parameters.copy()
for _key in ['erkstage-f', 'erkstage-aux', 'erkstage-mu', 'erkstage-muaux', 'erk-dlambda', 'erk-grad']:
    SOLVER_PARAMETERS[_key] = {'ksp_type': 'preonly', 'pc_type': 'lu'}


def flat(function):
    """Return a copy of the flattened dof array of a (mixed) Function or Cofunction."""
    return create_flattened_numpy_arr_from_mixed_function(function).copy()


def build_model(cfg=DEFAULT_CFG):
    """Build a model and timestepper from a cfg file and initialize the state and coefficients.

    Returns a namespace with parameters, logger, model, timestepper, dt, the coefficient Function
    (coeff, coeff_sub), and the initial state and time (x0, x0_sub, t0).
    """
    parameters = get_parameters(cfg)
    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    coeff, coeff_sub, _ = model.get_coeff_var('coeff')
    timestepper = get_timestepper(parameters, model, logger, SOLVER_PARAMETERS)
    x0, x0_sub, _ = model.get_full_var('x0', split_x_and_aux=True)
    t0 = model.get_t_var()
    model.initialize(x0_sub, t0)
    model.set_coeffs(parameters, coeff_sub)
    timestepper.set_coeff(coeff)
    return SimpleNamespace(
        parameters=parameters,
        logger=logger,
        model=model,
        timestepper=timestepper,
        dt=parameters['timestepping']['dt'],
        coeff=coeff,
        coeff_sub=coeff_sub,
        x0=x0,
        x0_sub=x0_sub,
        t0=t0,
    )


def setup_problem(nsteps, obs_steps=None, var_weights=None, cfg=DEFAULT_CFG):
    """Build an adjoint optimization problem whose truth is the unperturbed model trajectory.

    The truth is integrated for nsteps from the cfg's initial condition and coefficients. With
    obs_steps None, the loss is the final-state L2Objective; otherwise it is an
    ObservationL2Objective comparing at obs_steps, optionally with per-variable weights.

    Returns the build_model namespace extended with objective and optimizer.
    """
    setup = build_model(cfg)
    xns, xn_subs, tns = create_states(setup.model, nsteps)
    compute_states(xns, xn_subs, tns, setup.model, setup.timestepper, nsteps, setup.dt, setup.x0, setup.t0)
    if obs_steps is None:
        objective = L2Objective([[xns[0], xns[nsteps]]], [[tns[0], tns[nsteps]]], nsteps, setup.model.spaces.dx)
    else:
        data_blocks = [[xns[0]] + [xns[step] for step in obs_steps]]
        objective = ObservationL2Objective(data_blocks, [[setup.t0]], nsteps, obs_steps, setup.model.spaces.dx,
            var_weights=var_weights)
    optimizer = Lagrangian_ODEConstrainedOptimization(setup.model, setup.timestepper, objective, setup.dt)
    optimizer.verbose = False
    setup.objective = objective
    setup.optimizer = optimizer
    return setup


def perturbed_initial_state(setup, ic_changes):
    """Return the flattened initial state for the cfg's initial condition with changed parameters.

    ic_changes maps initial-condition parameter names (for example 'ox') to new values. The
    model's initial condition object is restored to the original parameters afterwards.
    """
    parameters = deepcopy(setup.parameters)
    parameters['initial-conditions'].update(ic_changes)
    x_new, x_new_sub, _ = setup.model.get_full_var('x0_perturbed_ic', split_x_and_aux=True)
    t = setup.model.get_t_var()
    setup.model.initialize(x_new_sub, t, new_params=parameters)
    setup.model.initcond.set_params(setup.parameters)
    return flat(x_new[0])


def perturbed_coeffs(setup, coeff_changes):
    """Return the flattened coefficient vector for the cfg's coefficients with changed values.

    coeff_changes maps forcing-term sections to {name: value}, for example
    {'hyperviscosity': {'c0': 0.05}}.
    """
    parameters = deepcopy(setup.parameters)
    for section, changes in coeff_changes.items():
        parameters[section].update(changes)
    coeff_new, coeff_new_sub, _ = setup.model.get_coeff_var('coeff_perturbed')
    setup.model.set_coeffs(parameters, coeff_new_sub)
    return flat(coeff_new)


def reduced_taylor_test(make_problem, direction, eps_list=None):
    """Taylor-test a ReducedProblem built by make_problem() and return (eps, remainders, rates)."""
    from dimswe.optimize import taylor_test

    return taylor_test(make_problem(), np.asarray(direction, dtype=float), eps_list=eps_list)


def convergence_rates(eps_list, remainders):
    """Return the observed convergence rates between consecutive Taylor remainders."""
    rates = []
    for i in range(len(remainders) - 1):
        if remainders[i] > 0 and remainders[i + 1] > 0:
            rates.append(float(np.log(remainders[i] / remainders[i + 1]) / np.log(eps_list[i] / eps_list[i + 1])))
        else:
            rates.append(float('nan'))
    return np.array(rates)


def random_direction(direction, base=None, scale=0.1, seed=0):
    """Fill direction with a seeded random perturbation and return it.

    Each component of a mixed Function is scaled by scale * max|base component|, so the
    perturbation is small relative to the state it perturbs. Components whose base is zero (or
    all components, when base is None) use a magnitude of scale instead, so the direction is
    never zero; a zero direction would make every Taylor remainder vanish.
    """
    generator = np.random.default_rng(seed)
    base_parts = base.subfunctions if base is not None else [None] * len(direction.subfunctions)
    for part, base_part in zip(direction.subfunctions, base_parts):
        data = part.dat.data
        magnitude = float(np.max(np.abs(base_part.dat.data_ro))) if base_part is not None and base_part.dat.data_ro.size else 0.0
        if magnitude == 0.0:
            magnitude = 1.0
        data[:] = scale * magnitude * generator.standard_normal(data.shape)
    return direction


def form_taylor_test(form, jacobian, x, direction, eps_list=None):
    """Taylor-test the UFL derivative of a residual form with respect to a Function.

    form is a 1-form depending on the Function x, jacobian is derivative(form, x, trial), and
    direction is a Function in the space of x. Returns (eps, remainders, rates, change), where
    remainder = ||F(x + eps d) - F(x) - eps J(x) d|| and change = max_eps ||F(x + eps d) - F(x)||,
    which lets assert_taylor_convergence recognize forms that are linear in x.
    """
    if eps_list is None:
        eps_list = list(1e-2 * 0.5 ** np.arange(0, 8))
    assert np.any(flat(direction) != 0), 'Taylor test direction is zero, so the test cannot detect anything'
    f0 = flat(assemble(form))
    jac_direction = flat(assemble(action(jacobian, direction)))
    remainders = []
    change = 0.0
    for eps in eps_list:
        fp = flat(assemble(replace(form, {x: x + float(eps) * direction})))
        remainders.append(float(np.linalg.norm(fp - f0 - eps * jac_direction)))
        change = max(change, float(np.linalg.norm(fp - f0)))
    return np.array(eps_list), np.array(remainders), convergence_rates(eps_list, remainders), change


def assert_taylor_convergence(rates, remainders=None, change=None, expected_rate=2.0, nrates=3, atol=0.2, linear_tol=1e-10):
    """Assert that the leading Taylor remainder rates are close to expected_rate.

    If remainders and change are given and every remainder is at round-off relative to change,
    the function is linear in the perturbed variable and the derivative is exact, so the rates
    are meaningless and the check passes. The same holds when the function does not change at all
    in a (nonzero) direction and every remainder is zero: the derivative is exactly zero.
    """
    if remainders is not None and change is not None:
        if change == 0 and np.max(remainders) == 0:
            return
        if change > 0 and np.max(remainders) <= linear_tol * change:
            return
    finite = rates[np.isfinite(rates)]
    assert finite.size >= nrates, f'too few finite rates: {rates}'
    assert np.allclose(finite[:nrates], expected_rate, atol=atol), f'rates {rates}, expected ~{expected_rate}'


def assert_repeatable(run_test):
    """Run a Taylor test twice and assert identical results, returning the first.

    run_test() must return (eps, remainders, rates). Repeating the test catches state that leaks
    between forward or adjoint evaluations (for example timestepper internal variables).
    """
    first = run_test()
    second = run_test()
    assert np.allclose(first[1], second[1]), f'remainders differ between runs: {first[1]} vs {second[1]}'
    assert np.allclose(first[2], second[2], equal_nan=True)
    return first
