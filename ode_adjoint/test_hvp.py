from adjoint_timesteppers import Euler, RK4, SSPRK3, SSPRK43
from dynamics import _Dynamics, LogisticEquation, LotkaVolterra
from hvp import (
    OperatorComposition,
    terminal_least_squares_gauss_newton_hvp,
    terminal_least_squares_gradient,
    terminal_least_squares_hvp,
)

import numpy as np
import pytest


class ScalarQuadraticDynamics(_Dynamics):
    """Hand-checkable scalar dynamics ``f(x, p) = p**2*x``."""

    def rhs(self, x, t, params):
        return np.array([params[0]**2*x[0]])

    def jac_x(self, x, t, params):
        return np.array([[params[0]**2]])

    def jacT_x(self, x, t, params):
        return self.jac_x(x, t, params).T

    def jac_params(self, x, t, params):
        return np.array([[2.0*params[0]*x[0]]])

    def jacT_params(self, x, t, params):
        return self.jac_params(x, t, params).T

    def directional_jacT_x_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        return np.array([2.0*params[0]*param_direction[0]*adjoint[0]])

    def directional_jacT_params_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        action = (
            2.0*params[0]*state_direction[0]
            + 2.0*x[0]*param_direction[0]
        )
        return np.array([action*adjoint[0]])

    def get_x_size(self):
        return 1

    def get_param_size(self):
        return 1


class FullyNonlinearDynamics(_Dynamics):
    """Two-state/two-parameter dynamics with all second blocks nonzero."""

    def rhs(self, x, t, params):
        x0, x1 = x
        p0, p1 = params
        return np.array([
            x0**2 + p0*x1 + p0*p1,
            np.sin(x0) + p1*x0*x1 + p0**2,
        ])

    def jac_x(self, x, t, params):
        x0, x1 = x
        p0, p1 = params
        return np.array([
            [2.0*x0, p0],
            [np.cos(x0) + p1*x1, p1*x0],
        ])

    def jacT_x(self, x, t, params):
        return self.jac_x(x, t, params).T

    def jac_params(self, x, t, params):
        x0, x1 = x
        p0, p1 = params
        return np.array([
            [x1 + p1, p0],
            [2.0*p0, x0*x1],
        ])

    def jacT_params(self, x, t, params):
        return self.jac_params(x, t, params).T

    def directional_jacT_x_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        x0, x1 = x
        _, p1 = params
        w0, w1 = state_direction
        q0, q1 = param_direction
        directional_jacobian = np.array([
            [2.0*w0, q0],
            [-np.sin(x0)*w0 + q1*x1 + p1*w1,
             q1*x0 + p1*w0],
        ])
        return directional_jacobian.T.dot(adjoint)

    def directional_jacT_params_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        x0, x1 = x
        w0, w1 = state_direction
        q0, q1 = param_direction
        directional_jacobian = np.array([
            [w1 + q1, q0],
            [2.0*q0, w0*x1 + x0*w1],
        ])
        return directional_jacobian.T.dot(adjoint)

    def get_x_size(self):
        return 2

    def get_param_size(self):
        return 2


def _centered_gradient_errors(
    timestepper, nsteps, params, x0, dt, target, direction,
    state_direction=None, epsilons=None,
):
    if epsilons is None:
        epsilons = 10.0**(-np.arange(1, 9))
    if state_direction is None:
        state_direction = np.zeros_like(x0)
    exact = terminal_least_squares_hvp(
        timestepper, nsteps, params, 0.0, x0, dt, target,
        direction, state_direction,
    )
    exact_vector = np.hstack([exact.hvp_params, exact.hvp_initial_state])
    errors = []
    for epsilon in epsilons:
        plus = terminal_least_squares_gradient(
            timestepper,
            nsteps,
            params + epsilon*direction,
            0.0,
            x0 + epsilon*state_direction,
            dt,
            target,
        )
        minus = terminal_least_squares_gradient(
            timestepper,
            nsteps,
            params - epsilon*direction,
            0.0,
            x0 - epsilon*state_direction,
            dt,
            target,
        )
        difference = np.hstack([
            plus.gradient_params - minus.gradient_params,
            plus.gradient_initial_state - minus.gradient_initial_state,
        ])/(2.0*epsilon)
        errors.append(np.linalg.norm(difference - exact_vector))
    return np.asarray(epsilons), np.asarray(errors)


def _assert_centered_convergence(epsilons, errors):
    rates = np.log(errors[:-1]/errors[1:])/np.log(
        epsilons[:-1]/epsilons[1:]
    )
    assert np.all(rates[:3] > 1.8)
    assert np.min(errors) < 2.0e-8
    assert np.argmin(errors) >= 3


def test_second_directional_interface_against_centered_matrix_difference():
    dynamics = FullyNonlinearDynamics()
    x = np.array([0.7, -0.4])
    params = np.array([1.2, -0.8])
    state_direction = np.array([0.3, -0.25])
    param_direction = np.array([-0.2, 0.45])
    adjoint = np.array([0.6, -1.1])
    exact_x = dynamics.directional_jacT_x_action(
        x, 0.0, params, state_direction, param_direction, adjoint
    )
    exact_params = dynamics.directional_jacT_params_action(
        x, 0.0, params, state_direction, param_direction, adjoint
    )
    errors_x = []
    errors_params = []
    epsilons = 10.0**(-np.arange(1, 8))
    for epsilon in epsilons:
        x_plus = x + epsilon*state_direction
        x_minus = x - epsilon*state_direction
        p_plus = params + epsilon*param_direction
        p_minus = params - epsilon*param_direction
        difference_x = (
            dynamics.jacT_x(x_plus, 0.0, p_plus)
            - dynamics.jacT_x(x_minus, 0.0, p_minus)
        ).dot(adjoint)/(2.0*epsilon)
        difference_params = (
            dynamics.jacT_params(x_plus, 0.0, p_plus)
            - dynamics.jacT_params(x_minus, 0.0, p_minus)
        ).dot(adjoint)/(2.0*epsilon)
        errors_x.append(np.linalg.norm(difference_x - exact_x))
        errors_params.append(np.linalg.norm(difference_params - exact_params))
    assert np.min(errors_x) < 1.0e-9
    assert np.min(errors_params) < 1.0e-9
    assert errors_x[1] < 0.02*errors_x[0]
    # The parameter-Jacobian derivative is affine along this direction, so
    # centered differences are roundoff-limited from the first step onward.
    assert errors_params[-1] < 1.0e-8


@pytest.mark.parametrize(
    "dynamics,x,params,state_direction,param_direction,adjoint",
    [
        (
            LotkaVolterra(),
            np.array([2.0, 1.0]),
            np.array([1.5, 1.2, 0.8, 1.3]),
            np.array([0.1, 0.15]),
            np.array([0.1, 0.2, 0.15, 0.12]),
            np.array([0.6, -1.0]),
        ),
        (
            LogisticEquation(),
            np.array([1.0]),
            np.array([1.5, 1.2]),
            np.array([0.1]),
            np.array([0.1, 0.2]),
            np.array([0.6]),
        ),
    ],
)
def test_existing_dynamics_second_actions(
    dynamics, x, params, state_direction, param_direction, adjoint
):
    exact_x = dynamics.directional_jacT_x_action(
        x, 0.0, params, state_direction, param_direction, adjoint
    )
    exact_params = dynamics.directional_jacT_params_action(
        x, 0.0, params, state_direction, param_direction, adjoint
    )
    errors_x = []
    errors_params = []
    for epsilon in 10.0**(-np.arange(2, 8)):
        x_plus = x + epsilon*state_direction
        x_minus = x - epsilon*state_direction
        p_plus = params + epsilon*param_direction
        p_minus = params - epsilon*param_direction
        difference_x = (
            dynamics.jacT_x(x_plus, 0.0, p_plus)
            - dynamics.jacT_x(x_minus, 0.0, p_minus)
        ).dot(adjoint)/(2.0*epsilon)
        difference_params = (
            dynamics.jacT_params(x_plus, 0.0, p_plus)
            - dynamics.jacT_params(x_minus, 0.0, p_minus)
        ).dot(adjoint)/(2.0*epsilon)
        errors_x.append(np.linalg.norm(exact_x - difference_x))
        errors_params.append(np.linalg.norm(exact_params - difference_params))
    assert np.min(errors_x) < 2.0e-10
    assert np.min(errors_params) < 2.0e-10


def test_scalar_one_step_euler_hand_reference():
    dynamics = ScalarQuadraticDynamics()
    result = terminal_least_squares_hvp(
        Euler(dynamics),
        nsteps=1,
        params=np.array([3.0]),
        t0=0.0,
        x0=np.array([2.0]),
        dt=0.1,
        target=np.array([1.0]),
        param_direction=np.array([0.5]),
    )
    # Independently: x+ = x(1+dt*p^2), x+' = 2*dt*p*x,
    # and x+'' = 2*dt*x for J = 0.5*(x+ - d)^2.
    multiplier = 1.0 + 0.1*3.0**2
    state = 2.0*multiplier
    state_prime = 2.0*0.1*3.0*2.0
    state_second = 2.0*0.1*2.0
    residual = state - 1.0
    gradient = residual*state_prime
    hessian = state_prime**2 + residual*state_second
    assert result.states[-1, 0] == pytest.approx(state)
    assert result.gradient_params[0] == pytest.approx(gradient)
    assert result.hvp_params[0] == pytest.approx(hessian*0.5)
    assert state == pytest.approx(3.8)
    assert gradient == pytest.approx(3.36)
    assert hessian*0.5 == pytest.approx(1.28)
    assert hessian == pytest.approx(2.56)


def test_scalar_two_step_euler_hand_reference():
    dynamics = ScalarQuadraticDynamics()
    result = terminal_least_squares_hvp(
        Euler(dynamics), 2, np.array([3.0]), 0.0, np.array([2.0]),
        0.1, np.array([1.0]), np.array([0.5]),
    )
    multiplier = 1.0 + 0.1*3.0**2
    state = 2.0*multiplier**2
    state_prime = 2.0*2.0*multiplier*(2.0*0.1*3.0)
    state_second = 2.0*(
        2.0*(2.0*0.1*3.0)**2 + 2.0*multiplier*(2.0*0.1)
    )
    residual = state - 1.0
    gradient = residual*state_prime
    hessian = state_prime**2 + residual*state_second
    assert result.states[:, 0] == pytest.approx([2.0, 3.8, 7.22])
    assert result.gradient_params[0] == pytest.approx(28.3632)
    assert result.hvp_params[0] == pytest.approx(19.6024)
    assert gradient == pytest.approx(28.3632)
    assert hessian == pytest.approx(39.2048)


def test_vector_euler_matches_direct_one_step_algebra():
    dynamics = FullyNonlinearDynamics()
    dt = 0.15
    x = np.array([0.7, -0.4])
    params = np.array([1.2, -0.8])
    target = np.array([0.2, -0.1])
    state_direction = np.array([0.25, -0.35])
    param_direction = np.array([0.3, -0.5])
    result = terminal_least_squares_hvp(
        Euler(dynamics), 1, params, 0.0, x, dt, target,
        param_direction, state_direction,
    )

    jac_x = dynamics.jac_x(x, 0.0, params)
    jac_params = dynamics.jac_params(x, 0.0, params)
    state_out = x + dt*dynamics.rhs(x, 0.0, params)
    tangent_out = state_direction + dt*(
        jac_x.dot(state_direction) + jac_params.dot(param_direction)
    )
    residual = state_out - target
    gradient = dt*jac_params.T.dot(residual)
    hvp = dt*(
        jac_params.T.dot(tangent_out)
        + dynamics.directional_jacT_params_action(
            x, 0.0, params, state_direction, param_direction, residual
        )
    )
    assert result.states[-1] == pytest.approx(state_out)
    assert result.tangents[-1] == pytest.approx(tangent_out)
    assert result.gradient_params == pytest.approx(gradient)
    assert result.hvp_params == pytest.approx(hvp)


def _populated_legacy_rk_step():
    dynamics = FullyNonlinearDynamics()
    timestepper = RK4(dynamics)
    tn = 0.3
    dt = 0.08
    state = np.array([0.25, -0.15])
    params = np.array([0.45, -0.3])
    state_out = np.empty_like(state)
    timestepper.take_forward_step(state_out, dt, tn, state, params)
    return timestepper, tn, dt, params


def test_legacy_immediate_adjoint_matches_reverse_step():
    timestepper, tn, dt, params = _populated_legacy_rk_step()
    state_adjoint_out = np.array([0.6, -0.4])
    expected = timestepper.reverse_step(
        timestepper._last_step, state_adjoint_out
    )
    ts_grad = np.empty(params.size)
    delta_lambda = np.empty(state_adjoint_out.size)

    timestepper.take_adjoint_step(
        ts_grad,
        delta_lambda,
        dt,
        tn + dt,
        state_adjoint_out,
        params,
    )

    # The wrapper retains its historical negative parameter-gradient sign and
    # returns only the non-identity state-adjoint increment.
    assert ts_grad == pytest.approx(-expected.parameter_adjoint)
    assert delta_lambda == pytest.approx(
        expected.state_adjoint_in - state_adjoint_out
    )


def test_legacy_adjoint_rejects_mismatched_cached_dt():
    timestepper, tn, dt, params = _populated_legacy_rk_step()
    mismatched_dt = 1.5*dt
    with pytest.raises(ValueError, match="cached RK step dt mismatch"):
        timestepper.take_adjoint_step(
            np.empty(params.size),
            np.empty(2),
            mismatched_dt,
            tn + mismatched_dt,
            np.array([0.6, -0.4]),
            params,
        )


def test_legacy_adjoint_rejects_mismatched_cached_time():
    timestepper, tn, dt, params = _populated_legacy_rk_step()
    with pytest.raises(ValueError, match="cached RK step start-time mismatch"):
        timestepper.take_adjoint_step(
            np.empty(params.size),
            np.empty(2),
            dt,
            tn + dt + 0.25,
            np.array([0.6, -0.4]),
            params,
        )


def test_legacy_adjoint_rejects_mismatched_cached_params():
    timestepper, tn, dt, params = _populated_legacy_rk_step()
    with pytest.raises(ValueError, match="cached RK step parameter mismatch"):
        timestepper.take_adjoint_step(
            np.empty(params.size),
            np.empty(2),
            dt,
            tn + dt,
            np.array([0.6, -0.4]),
            params + np.array([0.1, 0.0]),
        )
    with pytest.raises(ValueError, match=r"params must have shape \(2,\)"):
        timestepper.take_adjoint_step(
            np.empty(params.size),
            np.empty(2),
            dt,
            tn + dt,
            np.array([0.6, -0.4]),
            params[:1],
        )


@pytest.mark.parametrize(
    "stepper_factory",
    [Euler, RK4, lambda dynamics: SSPRK3(dynamics, 3),
     lambda dynamics: SSPRK43(dynamics, 4)],
)
def test_generic_explicit_tableau_tangent_and_hvp_convergence(stepper_factory):
    dynamics = FullyNonlinearDynamics()
    timestepper = stepper_factory(dynamics)
    params = np.array([0.45, -0.3])
    x0 = np.array([0.25, -0.15])
    target = np.array([-0.1, 0.2])
    direction = np.array([0.35, -0.2])
    state_direction = np.array([-0.1, 0.15])
    epsilons, errors = _centered_gradient_errors(
        timestepper, 3, params, x0, 0.08, target, direction,
        state_direction,
    )
    _assert_centered_convergence(epsilons, errors)


def test_one_and_multiple_step_centered_gradient_convergence():
    dynamics = FullyNonlinearDynamics()
    params = np.array([0.5, -0.35])
    x0 = np.array([0.3, -0.2])
    target = np.array([-0.2, 0.15])
    direction = np.array([0.4, -0.25])
    for stepper, nsteps in [(Euler(dynamics), 1), (RK4(dynamics), 7)]:
        epsilons, errors = _centered_gradient_errors(
            stepper, nsteps, params, x0, 0.06, target, direction
        )
        _assert_centered_convergence(epsilons, errors)


def test_multidimensional_parameter_hessian_symmetry():
    dynamics = FullyNonlinearDynamics()
    timestepper = RK4(dynamics)
    params = np.array([0.5, -0.35])
    x0 = np.array([0.3, -0.2])
    target = np.array([-0.2, 0.15])
    directions = [
        np.array([0.4, -0.25]),
        np.array([-0.15, 0.5]),
        np.array([0.7, 0.2]),
    ]
    products = [
        terminal_least_squares_hvp(
            timestepper, 6, params, 0.0, x0, 0.06, target, direction
        ).hvp_params
        for direction in directions
    ]
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            left = directions[i].dot(products[j])
            right = directions[j].dot(products[i])
            assert abs(left - right) < 2.0e-13


def test_exact_and_gauss_newton_hvp_distinction():
    dynamics = FullyNonlinearDynamics()
    timestepper = RK4(dynamics)
    params = np.array([0.5, -0.35])
    x0 = np.array([0.3, -0.2])
    direction = np.array([0.4, -0.25])
    states, _ = timestepper.compute_state(4, params, 0.0, x0, 0.06)
    zero_residual_target = states[-1].copy()
    exact_zero = terminal_least_squares_hvp(
        timestepper, 4, params, 0.0, x0, 0.06,
        zero_residual_target, direction,
    ).hvp_params
    gauss_newton_zero = terminal_least_squares_gauss_newton_hvp(
        timestepper, 4, params, 0.0, x0, 0.06,
        zero_residual_target, direction,
    )
    assert exact_zero == pytest.approx(gauss_newton_zero, abs=2.0e-14)

    nonzero_target = zero_residual_target + np.array([0.4, -0.3])
    exact_nonzero = terminal_least_squares_hvp(
        timestepper, 4, params, 0.0, x0, 0.06,
        nonzero_target, direction,
    ).hvp_params
    gauss_newton_nonzero = terminal_least_squares_gauss_newton_hvp(
        timestepper, 4, params, 0.0, x0, 0.06,
        nonzero_target, direction,
    )
    assert np.linalg.norm(exact_nonzero - gauss_newton_nonzero) > 1.0e-3


class LinearParameterMap:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def forward(self, x, params):
        return x + self.matrix.dot(params)

    def jac_x_action(self, x, params, direction):
        return direction.copy()

    def jac_params_action(self, x, params, direction):
        return self.matrix.dot(direction)

    def jacT_x_action(self, x, params, adjoint):
        return adjoint.copy()

    def jacT_params_action(self, x, params, adjoint):
        return self.matrix.T.dot(adjoint)

    def directional_jacT_x_action(
        self, x, params, state_direction, param_direction, adjoint
    ):
        return np.zeros_like(x)

    def directional_jacT_params_action(
        self, x, params, state_direction, param_direction, adjoint
    ):
        return np.zeros_like(params)


class QuadraticStateMap:
    def __init__(self, coefficient):
        self.coefficient = np.asarray(coefficient)

    def forward(self, x, params):
        return x + self.coefficient*x**2

    def jac_x_action(self, x, params, direction):
        return (1.0 + 2.0*self.coefficient*x)*direction

    def jac_params_action(self, x, params, direction):
        return np.zeros_like(x)

    def jacT_x_action(self, x, params, adjoint):
        return (1.0 + 2.0*self.coefficient*x)*adjoint

    def jacT_params_action(self, x, params, adjoint):
        return np.zeros_like(params)

    def directional_jacT_x_action(
        self, x, params, state_direction, param_direction, adjoint
    ):
        return 2.0*self.coefficient*state_direction*adjoint

    def directional_jacT_params_action(
        self, x, params, state_direction, param_direction, adjoint
    ):
        return np.zeros_like(params)


class IdentityStateMap(QuadraticStateMap):
    def __init__(self, size):
        super().__init__(np.zeros(size))


def test_two_child_composition_transports_through_parameter_independent_child():
    matrix = np.array([[1.0, 0.2], [-0.3, 0.7]])
    x0 = np.array([0.2, -0.1])
    params = np.array([0.5, -0.4])
    target = np.array([0.1, 0.3])
    direction = np.array([0.6, -0.25])
    parameter_child = LinearParameterMap(matrix)
    state_child = QuadraticStateMap(np.array([0.4, -0.2]))
    full = OperatorComposition([parameter_child, state_child])
    result = full.terminal_least_squares_hvp(
        x0, params, target, direction
    )
    gradient_result = full.terminal_least_squares_gradient(
        x0, params, target
    )
    assert state_child.jac_params_action(
        result.states[1], params, direction
    ) == pytest.approx(np.zeros(2))
    assert result.value == pytest.approx(0.619633765)
    assert result.gradient_params == pytest.approx(
        [1.33016001, -0.55024612]
    )
    assert result.hvp_params == pytest.approx(
        [1.721557, -0.14764624]
    )
    assert gradient_result.value == pytest.approx(result.value)
    assert gradient_result.gradient_params == pytest.approx(
        [1.33016001, -0.55024612]
    )
    assert gradient_result.gradient_initial_state == pytest.approx(
        result.gradient_initial_state
    )
    for gradient_state, hvp_state in zip(
        gradient_result.states, result.states
    ):
        assert gradient_state == pytest.approx(hvp_state)

    identity = OperatorComposition([parameter_child, IdentityStateMap(2)])
    identity_result = identity.terminal_least_squares_hvp(
        x0, params, target, direction
    )
    assert identity_result.gradient_params == pytest.approx([0.769, -0.477])
    assert identity_result.hvp_params == pytest.approx([0.6565, -0.1385])
    assert not np.allclose(
        result.gradient_params, identity_result.gradient_params
    )
    assert not np.allclose(result.hvp_params, identity_result.hvp_params)

    epsilons = 10.0**(-np.arange(1, 8))
    errors = []
    for epsilon in epsilons:
        plus = full.terminal_least_squares_gradient(
            x0, params + epsilon*direction, target
        ).gradient_params
        minus = full.terminal_least_squares_gradient(
            x0, params - epsilon*direction, target
        ).gradient_params
        errors.append(np.linalg.norm(
            (plus - minus)/(2.0*epsilon) - result.hvp_params
        ))
    _assert_centered_convergence(epsilons, np.asarray(errors))


def test_shapes_dtype_input_mutation_and_repeatability():
    dynamics = FullyNonlinearDynamics()
    timestepper = RK4(dynamics)
    params = np.array([0.5, -0.35], dtype=np.float32)
    x0 = np.array([0.3, -0.2], dtype=np.float32)
    target = np.array([-0.2, 0.15], dtype=np.float32)
    direction = np.array([0.4, -0.25], dtype=np.float32)
    originals = [array.copy() for array in (params, x0, target, direction)]
    first = terminal_least_squares_hvp(
        timestepper, 5, params, 0.0, x0, 0.06, target, direction
    )
    second = terminal_least_squares_hvp(
        timestepper, 5, params, 0.0, x0, 0.06, target, direction
    )
    assert first.states.shape == (6, 2)
    assert first.tangents.shape == (6, 2)
    assert first.gradient_params.shape == (2,)
    assert first.hvp_params.shape == (2,)
    assert np.issubdtype(first.hvp_params.dtype, np.floating)
    assert first.hvp_params == pytest.approx(second.hvp_params, abs=0.0)
    for array, original in zip((params, x0, target, direction), originals):
        assert np.array_equal(array, original)
    with pytest.raises(ValueError, match="param_direction"):
        terminal_least_squares_hvp(
            timestepper, 1, params, 0.0, x0, 0.06, target,
            np.array([1.0]),
        )

    integer_result = terminal_least_squares_hvp(
        Euler(ScalarQuadraticDynamics()), 1, np.array([3]), 0,
        np.array([2]), 1, np.array([1]), np.array([1]),
    )
    assert np.issubdtype(integer_result.states.dtype, np.floating)
    assert np.issubdtype(integer_result.hvp_params.dtype, np.floating)
