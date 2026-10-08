"""Certification tests for the isolated Firedrake exact-HVP prototype."""

import numpy as np
import pytest

from firedrake import (
    COMM_SELF,
    Cofunction,
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    UnitIntervalMesh,
    assemble,
    dx,
    grad,
    inner,
    pi,
    sin,
)

from firedrake_hvp_prototype import (
    CLASSICAL_RK4,
    EULER,
    ExplicitRungeKutta,
    WeakStageModel,
    dual_pairing,
    terminal_least_squares_gradient,
    terminal_least_squares_hvp,
    terminal_least_squares_objective,
)


def _values(value):
    return value.dat.data_ro.copy()


def _dual_error(left, right):
    with left.dat.vec_ro as left_vec, right.dat.vec_ro as right_vec:
        difference = left_vec.copy()
        difference.axpy(-1.0, right_vec)
        return difference.norm()


def _field_l2_error(left, right):
    return float(assemble(inner(left - right, left - right) * dx)) ** 0.5


def _reaction_hessian(tableau, nsteps, u0, p, dt, target):
    z = dt * p**2
    if tableau is EULER:
        amplification = 1.0 + z
        amplification_z = 1.0
        amplification_zz = 0.0
    elif tableau is CLASSICAL_RK4:
        amplification = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
        amplification_z = 1.0 + z + z**2 / 2.0 + z**3 / 6.0
        amplification_zz = 1.0 + z + z**2 / 2.0
    else:
        raise ValueError("no scalar oracle for this tableau")
    amplification_p = amplification_z * 2.0 * dt * p
    amplification_pp = (
        amplification_zz * (2.0 * dt * p) ** 2
        + amplification_z * 2.0 * dt
    )
    state = u0 * amplification**nsteps
    state_p = (
        u0 * nsteps * amplification ** (nsteps - 1) * amplification_p
    )
    state_pp = u0 * (
        nsteps
        * (nsteps - 1)
        * amplification ** (nsteps - 2)
        * amplification_p**2
        + nsteps * amplification ** (nsteps - 1) * amplification_pp
    )
    residual = state - target
    return {
        "state": state,
        "objective": 0.5 * residual**2,
        "gradient": residual * state_p,
        "hessian": state_p**2 + residual * state_pp,
    }


@pytest.fixture(scope="module")
def reaction_problem():
    mesh = UnitIntervalMesh(4, comm=COMM_SELF)
    space = FunctionSpace(mesh, "CG", 1)
    model = WeakStageModel(space)
    state = Function(space, name="reaction_state").interpolate(2.0)
    target = Function(space, name="reaction_target").interpolate(1.0)
    zero = Function(space, name="zero_direction")
    return model, state, target, zero


@pytest.fixture(scope="module")
def diffusion_problem():
    mesh = UnitIntervalMesh(6, comm=COMM_SELF)
    space = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)
    bc = DirichletBC(space, 0.0, "on_boundary")
    kappa = 0.08
    model = WeakStageModel(space, kappa=kappa, bcs=[bc])
    state = Function(space, name="diffusion_state").interpolate(sin(pi * x))
    target = Function(space, name="diffusion_target").interpolate(
        0.2 * x * (1.0 - x)
    )
    direction = Function(space, name="diffusion_direction").interpolate(
        x * (1.0 - x) * (1.0 + 0.5 * x)
    )
    return model, state, target, direction, bc, kappa


def test_one_step_euler_reaction_exact_fields_and_scalars(reaction_problem):
    model, state, target, zero = reaction_problem
    timestepper = ExplicitRungeKutta(model, EULER)
    result = terminal_least_squares_hvp(
        timestepper,
        1,
        state,
        3.0,
        0.1,
        target,
        0.5,
        zero,
    )

    assert isinstance(result.terminal_adjoint, Cofunction)
    assert result.terminal_adjoint.function_space() == model.dual_space
    assert _values(result.tangent_steps[0].primal.stages[0].stage_tendency) == pytest.approx(18.0)
    assert _values(result.states[-1]) == pytest.approx(3.8)
    assert _values(result.tangent_steps[0].stages[0].stage_tendency_direction) == pytest.approx(6.0)
    assert _values(result.state_directions[-1]) == pytest.approx(0.6)
    assert result.objective == pytest.approx(3.92, abs=2.0e-14)
    assert result.parameter_gradient == pytest.approx(3.36, abs=2.0e-14)
    assert result.parameter_hvp == pytest.approx(1.28, abs=2.0e-14)
    assert result.parameter_hvp / 0.5 == pytest.approx(2.56, abs=4.0e-14)

    assert float(assemble(result.states[-1] * dx)) == pytest.approx(3.8)
    assert float(assemble(result.state_directions[-1] * dx)) == pytest.approx(0.6)

    reverse = result.reverse_steps[0]
    ordinary_stage = reverse.ordinary.stages[0]
    incremental_stage = reverse.incremental_stages[0]
    assert _values(ordinary_stage.auxiliary) == pytest.approx(0.28)
    assert _values(incremental_stage.incremental_auxiliary) == pytest.approx(0.06)
    assert _values(
        model.l2_riesz_representative(result.initial_state_adjoint)
    ) == pytest.approx(5.32)
    assert _values(
        model.l2_riesz_representative(
            result.initial_incremental_state_adjoint
        )
    ) == pytest.approx(1.98)
    psi = ordinary_stage.auxiliary
    delta_psi = incremental_stage.incremental_auxiliary
    hvp_terms = (
        2.0 * 0.5 * float(assemble(state * psi * dx)),
        2.0 * 3.0 * float(assemble(zero * psi * dx)),
        2.0 * 3.0 * float(assemble(state * delta_psi * dx)),
    )
    assert hvp_terms == pytest.approx((0.56, 0.0, 0.72), abs=2.0e-14)
    assert sum(hvp_terms) == pytest.approx(result.parameter_hvp, abs=2.0e-14)


def test_two_step_euler_reaction_exact_reference(reaction_problem):
    model, state, target, zero = reaction_problem
    result = terminal_least_squares_hvp(
        ExplicitRungeKutta(model, EULER),
        2,
        state,
        3.0,
        0.1,
        target,
        0.5,
        zero,
    )
    assert [_values(value)[0] for value in result.states] == pytest.approx(
        [2.0, 3.8, 7.22]
    )
    assert result.objective == pytest.approx(19.3442, abs=8.0e-14)
    assert result.parameter_gradient == pytest.approx(28.3632, abs=1.0e-13)
    assert result.parameter_hvp == pytest.approx(19.6024, abs=1.0e-13)
    assert result.parameter_hvp / 0.5 == pytest.approx(39.2048, abs=2.0e-13)


@pytest.mark.parametrize("tableau", [EULER, CLASSICAL_RK4])
@pytest.mark.parametrize("nsteps", [1, 2, 5])
def test_reaction_multistep_matches_direct_scalar_hessian(
    reaction_problem,
    tableau,
    nsteps,
):
    model, state, target, zero = reaction_problem
    q = 0.37
    expected = _reaction_hessian(tableau, nsteps, 2.0, 0.8, 0.04, 1.0)
    result = terminal_least_squares_hvp(
        ExplicitRungeKutta(model, tableau),
        nsteps,
        state,
        0.8,
        0.04,
        target,
        q,
        zero,
    )
    assert _values(result.states[-1]) == pytest.approx(expected["state"], abs=2.0e-13)
    assert result.objective == pytest.approx(expected["objective"], abs=2.0e-13)
    assert result.parameter_gradient == pytest.approx(expected["gradient"], abs=3.0e-13)
    assert result.parameter_hvp / q == pytest.approx(expected["hessian"], abs=8.0e-13)


def test_dual_coefficients_riesz_field_and_pairing_are_distinct(reaction_problem):
    model, state, _, _ = reaction_problem
    x, = SpatialCoordinate(state.function_space().mesh())
    representative = Function(model.function_space).interpolate(0.4 + x + x**2)
    derivative_dual = model.mass_map(representative, name="derivative_dual")
    recovered = model.l2_riesz_representative(derivative_dual)
    deterministic_test = Function(model.function_space).interpolate(1.0 - 0.3 * x)

    assert isinstance(derivative_dual, Cofunction)
    assert derivative_dual.function_space() == model.function_space.dual()
    assert not np.allclose(_values(derivative_dual), _values(representative))
    assert _field_l2_error(recovered, representative) < 8.0e-15
    assert dual_pairing(derivative_dual, deterministic_test) == pytest.approx(
        float(assemble(representative * deterministic_test * dx)),
        abs=2.0e-15,
    )

    recovered_dual = model.mass_map(recovered)
    assert _dual_error(recovered_dual, derivative_dual) < 2.0e-15


def test_terminal_dual_pairing_is_objective_derivative(reaction_problem):
    model, state, target, _ = reaction_problem
    x, = SpatialCoordinate(state.function_space().mesh())
    test_direction = Function(model.function_space).interpolate(0.2 + x)
    result = terminal_least_squares_gradient(
        ExplicitRungeKutta(model, EULER),
        1,
        state,
        3.0,
        0.1,
        target,
    )
    residual = Function(model.function_space).assign(result.states[-1] - target)
    assert dual_pairing(result.terminal_adjoint, test_direction) == pytest.approx(
        float(assemble(residual * test_direction * dx)),
        abs=3.0e-15,
    )


def test_ordinary_gradient_matches_centered_objective_differences(reaction_problem):
    model, state, target, _ = reaction_problem
    timestepper = ExplicitRungeKutta(model, CLASSICAL_RK4)
    exact = terminal_least_squares_gradient(
        timestepper, 3, state, 0.8, 0.04, target
    ).parameter_gradient
    epsilons = (4.0e-2, 2.0e-2, 1.0e-2, 5.0e-3)
    errors = []
    for epsilon in epsilons:
        plus = terminal_least_squares_objective(
            timestepper, 3, state, 0.8 + epsilon, 0.04, target
        )
        minus = terminal_least_squares_objective(
            timestepper, 3, state, 0.8 - epsilon, 0.04, target
        )
        errors.append(abs((plus - minus) / (2.0 * epsilon) - exact))
    assert all(errors[i] / errors[i + 1] > 3.8 for i in range(3))
    assert errors[-1] < 4.7e-6


def test_rk4_hvp_centered_gradient_converges_quadratically(reaction_problem):
    model, state, target, zero = reaction_problem
    timestepper = ExplicitRungeKutta(model, CLASSICAL_RK4)
    q = 0.37
    exact = terminal_least_squares_hvp(
        timestepper, 4, state, 0.8, 0.04, target, q, zero
    ).parameter_hvp
    epsilons = (4.0e-2, 2.0e-2, 1.0e-2, 5.0e-3)
    errors = []
    for epsilon in epsilons:
        plus = terminal_least_squares_gradient(
            timestepper, 4, state, 0.8 + epsilon * q, 0.04, target
        ).parameter_gradient
        minus = terminal_least_squares_gradient(
            timestepper, 4, state, 0.8 - epsilon * q, 0.04, target
        ).parameter_gradient
        errors.append(abs((plus - minus) / (2.0 * epsilon) - exact))
    assert all(errors[i] / errors[i + 1] > 3.8 for i in range(3))
    assert errors[-1] < 1.01e-6


def test_inputs_are_unmodified_caches_own_copies_and_runs_repeat(reaction_problem):
    model, _, _, _ = reaction_problem
    state = Function(model.function_space).interpolate(2.0)
    target = Function(model.function_space).interpolate(1.0)
    direction = Function(model.function_space).interpolate(0.25)
    before = tuple(_values(value) for value in (state, target, direction))
    timestepper = ExplicitRungeKutta(model, CLASSICAL_RK4)
    first = terminal_least_squares_hvp(
        timestepper, 3, state, 0.8, 0.04, target, 0.37, direction
    )
    terminal_before = _values(first.terminal_adjoint)
    timestepper.reverse_step(
        first.tangent_steps[-1].primal,
        first.terminal_adjoint,
    )
    assert np.array_equal(_values(first.terminal_adjoint), terminal_before)
    for value, expected in zip((state, target, direction), before):
        assert np.array_equal(_values(value), expected)

    first_step = first.tangent_steps[0]
    cached_state = _values(first_step.primal.state_in)
    cached_direction = _values(first_step.state_direction_in)
    assert first_step.primal.state_in.dat is not state.dat
    assert first_step.state_direction_in.dat is not direction.dat
    assert first.states[1].dat is not first_step.primal.state_out.dat
    state.assign(-7.0)
    direction.assign(9.0)
    assert np.array_equal(_values(first_step.primal.state_in), cached_state)
    assert np.array_equal(_values(first_step.state_direction_in), cached_direction)

    state.interpolate(2.0)
    direction.interpolate(0.25)
    second = terminal_least_squares_hvp(
        timestepper, 3, state, 0.8, 0.04, target, 0.37, direction
    )
    assert first.objective == second.objective
    assert first.parameter_gradient == second.parameter_gradient
    assert first.parameter_hvp == second.parameter_hvp
    assert all(
        np.array_equal(_values(left), _values(right))
        for left, right in zip(first.states, second.states)
    )


def _assembled_dense_operators(space):
    test = TestFunction(space)
    trial = TrialFunction(space)
    mass = assemble(inner(test, trial) * dx, mat_type="aij").M.handle
    stiffness = assemble(
        inner(grad(test), grad(trial)) * dx,
        mat_type="aij",
    ).M.handle
    size = mass.getSize()[0]
    indices = np.arange(size, dtype=np.int32)
    return (
        np.asarray(mass.getValues(indices, indices)),
        np.asarray(stiffness.getValues(indices, indices)),
    )


def _dense_rk_oracle(
    tableau,
    nsteps,
    state,
    target,
    state_direction,
    parameter,
    parameter_direction,
    dt,
    kappa,
    mass,
    stiffness,
    free,
):
    mass_free = mass[np.ix_(free, free)]
    stiffness_free = stiffness[np.ix_(free, free)]
    operator = np.linalg.solve(
        mass_free,
        parameter**2 * mass_free - kappa * stiffness_free,
    )
    state_n = state[free].copy()
    direction_n = state_direction[free].copy()
    # P is du/dp and DP is its derivative in the combined (w, q) direction.
    parameter_tangent_n = np.zeros_like(state_n)
    mixed_tangent_n = np.zeros_like(state_n)
    states = [state_n.copy()]

    for _ in range(nsteps):
        stage_states = []
        stage_tendencies = []
        stage_directions = []
        stage_tendency_directions = []
        stage_parameter_tangents = []
        stage_parameter_tendency_tangents = []
        stage_mixed_tangents = []
        stage_mixed_tendency_tangents = []
        for i in range(tableau.nstages):
            yi = state_n.copy()
            wi = direction_n.copy()
            parameter_i = parameter_tangent_n.copy()
            mixed_i = mixed_tangent_n.copy()
            for j in range(i):
                coefficient = dt * tableau.a[i][j]
                yi += coefficient * stage_tendencies[j]
                wi += coefficient * stage_tendency_directions[j]
                parameter_i += coefficient * stage_parameter_tendency_tangents[j]
                mixed_i += coefficient * stage_mixed_tendency_tangents[j]
            ki = operator @ yi
            vi = operator @ wi + 2.0 * parameter * parameter_direction * yi
            parameter_ki = operator @ parameter_i + 2.0 * parameter * yi
            mixed_ki = (
                operator @ mixed_i
                + 2.0 * parameter * parameter_direction * parameter_i
                + 2.0 * parameter_direction * yi
                + 2.0 * parameter * wi
            )
            stage_states.append(yi)
            stage_tendencies.append(ki)
            stage_directions.append(wi)
            stage_tendency_directions.append(vi)
            stage_parameter_tangents.append(parameter_i)
            stage_parameter_tendency_tangents.append(parameter_ki)
            stage_mixed_tangents.append(mixed_i)
            stage_mixed_tendency_tangents.append(mixed_ki)

        for i in range(tableau.nstages):
            coefficient = dt * tableau.b[i]
            state_n += coefficient * stage_tendencies[i]
            direction_n += coefficient * stage_tendency_directions[i]
            parameter_tangent_n += (
                coefficient * stage_parameter_tendency_tangents[i]
            )
            mixed_tangent_n += coefficient * stage_mixed_tendency_tangents[i]
        states.append(state_n.copy())

    residual = state_n - target[free]
    objective = 0.5 * residual @ mass_free @ residual
    gradient = parameter_tangent_n @ mass_free @ residual
    hvp = (
        mixed_tangent_n @ mass_free @ residual
        + parameter_tangent_n @ mass_free @ direction_n
    )
    return states, direction_n, objective, gradient, hvp


@pytest.mark.parametrize(
    ("tableau", "nsteps"),
    [(EULER, 1), (EULER, 3), (CLASSICAL_RK4, 1), (CLASSICAL_RK4, 3)],
)
def test_reaction_diffusion_matches_tiny_dense_serial_oracle(
    diffusion_problem,
    tableau,
    nsteps,
):
    model, state, target, direction, bc, kappa = diffusion_problem
    mass, stiffness = _assembled_dense_operators(model.function_space)
    free = np.setdiff1d(np.arange(mass.shape[0]), bc.nodes)
    parameter = 0.7
    q = 0.3
    dt = 0.02
    expected = _dense_rk_oracle(
        tableau,
        nsteps,
        _values(state),
        _values(target),
        _values(direction),
        parameter,
        q,
        dt,
        kappa,
        mass,
        stiffness,
        free,
    )
    result = terminal_least_squares_hvp(
        ExplicitRungeKutta(model, tableau),
        nsteps,
        state,
        parameter,
        dt,
        target,
        q,
        direction,
    )

    expected_states, expected_direction, objective, gradient, hvp = expected
    for computed, expected_state in zip(result.states, expected_states):
        assert _values(computed)[free] == pytest.approx(expected_state, abs=2.0e-13)
    assert _values(result.state_directions[-1])[free] == pytest.approx(
        expected_direction,
        abs=2.0e-13,
    )
    assert result.objective == pytest.approx(objective, abs=2.0e-13)
    assert result.parameter_gradient == pytest.approx(gradient, abs=3.0e-13)
    assert result.parameter_hvp == pytest.approx(hvp, abs=5.0e-13)


@pytest.mark.parametrize("tableau", [EULER, CLASSICAL_RK4])
def test_reaction_diffusion_combined_hvp_gradient_difference_converges(
    diffusion_problem,
    tableau,
):
    model, state, target, direction, _, _ = diffusion_problem
    timestepper = ExplicitRungeKutta(model, tableau)
    parameter = 0.7
    q = 0.3
    dt = 0.02
    exact = terminal_least_squares_hvp(
        timestepper, 3, state, parameter, dt, target, q, direction
    ).parameter_hvp
    epsilons = (4.0e-2, 2.0e-2, 1.0e-2, 5.0e-3)
    errors = []
    for epsilon in epsilons:
        state_plus = Function(model.function_space).assign(
            state + epsilon * direction
        )
        state_minus = Function(model.function_space).assign(
            state - epsilon * direction
        )
        plus = terminal_least_squares_gradient(
            timestepper,
            3,
            state_plus,
            parameter + epsilon * q,
            dt,
            target,
        ).parameter_gradient
        minus = terminal_least_squares_gradient(
            timestepper,
            3,
            state_minus,
            parameter - epsilon * q,
            dt,
            target,
        ).parameter_gradient
        errors.append(abs((plus - minus) / (2.0 * epsilon) - exact))
    assert all(errors[i] / errors[i + 1] > 3.8 for i in range(3))
    assert errors[-1] < 7.6e-8


def test_reaction_diffusion_riesz_roundtrip_and_boundary_dual(diffusion_problem):
    model, state, _, _, bc, _ = diffusion_problem
    dual = model.mass_map(state)
    representative = model.l2_riesz_representative(dual)
    recovered = model.mass_map(representative)
    assert np.all(_values(dual)[bc.nodes] == 0.0)
    assert np.all(_values(representative)[bc.nodes] == 0.0)
    assert _dual_error(dual, recovered) < 2.0e-15
    assert _field_l2_error(state, representative) < 2.0e-14
