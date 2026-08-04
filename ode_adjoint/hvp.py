"""Exact discrete Hessian-vector products for the NumPy ODE prototype."""

from dataclasses import dataclass

import numpy as np


def _inexact_result_type(*values):
    dtype = np.result_type(*values)
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.dtype(float)
    return dtype


@dataclass(frozen=True)
class TerminalLeastSquaresGradient:
    """Value and first derivatives of a terminal least-squares objective."""

    value: float
    states: np.ndarray
    gradient_params: np.ndarray
    gradient_initial_state: np.ndarray


@dataclass(frozen=True)
class TerminalLeastSquaresHVP:
    """Exact directional second derivatives and their forward trajectory."""

    value: float
    states: np.ndarray
    tangents: np.ndarray
    gradient_params: np.ndarray
    gradient_initial_state: np.ndarray
    hvp_params: np.ndarray
    hvp_initial_state: np.ndarray


@dataclass(frozen=True)
class CompositionGradient:
    """Value and ordinary-adjoint derivatives of a composed operator."""

    value: float
    states: tuple
    gradient_params: np.ndarray
    gradient_initial_state: np.ndarray


@dataclass(frozen=True)
class CompositionHVP:
    """Terminal least-squares derivatives for a composed operator."""

    value: float
    states: tuple
    tangents: tuple
    gradient_params: np.ndarray
    gradient_initial_state: np.ndarray
    hvp_params: np.ndarray
    hvp_initial_state: np.ndarray


def _vector(name, value, size, dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    return array


def _check_nsteps(nsteps):
    if not isinstance(nsteps, (int, np.integer)) or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")


def terminal_least_squares_gradient(
    timestepper, nsteps, params, t0, x0, dt, target
):
    """Compute the exact discrete gradient using the ordinary RK adjoint.

    The initial state is treated as an independent input.  Consequently this
    returns gradients with respect to both the constant parameter vector and
    ``x0`` even when only the parameter gradient is needed by an application.
    """
    _check_nsteps(nsteps)
    nx = timestepper.dynamics.get_x_size()
    nparams = timestepper.dynamics.get_param_size()
    x0 = _vector("x0", x0, nx)
    params = _vector("params", params, nparams)
    dtype = _inexact_result_type(
        x0.dtype, params.dtype, np.asarray(dt).dtype
    )
    target = _vector("target", target, nx, dtype=dtype)

    states = [np.asarray(x0, dtype=dtype).copy()]
    steps = []
    time = float(t0)
    for _ in range(nsteps):
        step = timestepper.forward_step_data(
            dt, time, states[-1], params
        )
        steps.append(step)
        states.append(step.state_out.copy())
        time += dt

    residual = states[-1] - target
    value = 0.5*float(np.dot(residual, residual))
    state_adjoint = residual.copy()
    gradient_params = np.zeros(nparams, dtype=dtype)
    for step in reversed(steps):
        reverse = timestepper.reverse_step(step, state_adjoint)
        gradient_params += reverse.parameter_adjoint
        state_adjoint = reverse.state_adjoint_in

    return TerminalLeastSquaresGradient(
        value=value,
        states=np.stack(states),
        gradient_params=gradient_params,
        gradient_initial_state=state_adjoint,
    )


def terminal_least_squares_hvp(
    timestepper,
    nsteps,
    params,
    t0,
    x0,
    dt,
    target,
    param_direction,
    state_direction=None,
):
    """Compute the exact discrete HVP for a terminal least-squares objective.

    ``param_direction`` has shape ``(nparams,)``.  ``state_direction`` has
    shape ``(nx,)`` and defaults to zero (fixed initial state).  The returned
    two HVP vectors are the parameter and initial-state blocks of the full
    Hessian applied to this combined direction.
    """
    _check_nsteps(nsteps)
    nx = timestepper.dynamics.get_x_size()
    nparams = timestepper.dynamics.get_param_size()
    x0 = _vector("x0", x0, nx)
    params = _vector("params", params, nparams)
    dtype = _inexact_result_type(
        x0.dtype, params.dtype, np.asarray(dt).dtype
    )
    target = _vector("target", target, nx, dtype=dtype)
    param_direction = _vector(
        "param_direction", param_direction, nparams, dtype=dtype
    ).copy()
    if state_direction is None:
        state_direction = np.zeros(nx, dtype=dtype)
    state_direction = _vector(
        "state_direction", state_direction, nx, dtype=dtype
    ).copy()

    states = [np.asarray(x0, dtype=dtype).copy()]
    tangents = [state_direction.copy()]
    tangent_steps = []
    time = float(t0)
    for _ in range(nsteps):
        tangent_step = timestepper.linearize_step(
            dt,
            time,
            states[-1],
            params,
            tangents[-1],
            param_direction,
        )
        tangent_steps.append(tangent_step)
        states.append(tangent_step.primal.state_out.copy())
        tangents.append(tangent_step.state_direction_out.copy())
        time += dt

    residual = states[-1] - target
    value = 0.5*float(np.dot(residual, residual))
    state_adjoint = residual.copy()
    incremental_state_adjoint = tangents[-1].copy()
    gradient_params = np.zeros(nparams, dtype=dtype)
    hvp_params = np.zeros(nparams, dtype=dtype)
    for tangent_step in reversed(tangent_steps):
        reverse = timestepper.reverse_hvp_step(
            tangent_step, state_adjoint, incremental_state_adjoint
        )
        gradient_params += reverse.parameter_adjoint
        hvp_params += reverse.parameter_hvp
        state_adjoint = reverse.state_adjoint_in
        incremental_state_adjoint = reverse.incremental_state_adjoint_in

    return TerminalLeastSquaresHVP(
        value=value,
        states=np.stack(states),
        tangents=np.stack(tangents),
        gradient_params=gradient_params,
        gradient_initial_state=state_adjoint,
        hvp_params=hvp_params,
        hvp_initial_state=incremental_state_adjoint,
    )


def terminal_least_squares_gauss_newton_hvp(
    timestepper,
    nsteps,
    params,
    t0,
    x0,
    dt,
    target,
    param_direction,
    state_direction=None,
):
    """Return the parameter block of ``F'(p).T F'(p)`` times a direction.

    This is deliberately distinct from :func:`terminal_least_squares_hvp`.
    It applies the ordinary adjoint to the terminal tangent and omits all
    residual-weighted model-curvature terms.
    """
    _check_nsteps(nsteps)
    nx = timestepper.dynamics.get_x_size()
    nparams = timestepper.dynamics.get_param_size()
    x0 = _vector("x0", x0, nx)
    params = _vector("params", params, nparams)
    dtype = _inexact_result_type(
        x0.dtype, params.dtype, np.asarray(dt).dtype
    )
    _vector("target", target, nx, dtype=dtype)
    param_direction = _vector(
        "param_direction", param_direction, nparams, dtype=dtype
    ).copy()
    if state_direction is None:
        state_direction = np.zeros(nx, dtype=dtype)
    state_direction = _vector(
        "state_direction", state_direction, nx, dtype=dtype
    ).copy()

    state = np.asarray(x0, dtype=dtype).copy()
    tangent = state_direction
    tangent_steps = []
    time = float(t0)
    for _ in range(nsteps):
        tangent_step = timestepper.linearize_step(
            dt, time, state, params, tangent, param_direction
        )
        tangent_steps.append(tangent_step)
        state = tangent_step.primal.state_out
        tangent = tangent_step.state_direction_out
        time += dt

    state_adjoint = tangent.copy()
    hvp_params = np.zeros(nparams, dtype=dtype)
    for tangent_step in reversed(tangent_steps):
        reverse = timestepper.reverse_step(
            tangent_step.primal, state_adjoint
        )
        hvp_params += reverse.parameter_adjoint
        state_adjoint = reverse.state_adjoint_in
    return hvp_params


class OperatorComposition:
    """Small exact-HVP prototype for ``Phi_k(...Phi_1(x, p), p)``.

    Each child supplies ``forward``, Jacobian actions and transpose actions,
    and the same two contracted directional-transpose actions used by the ODE
    dynamics interface.  Child methods omit time and use arguments ``(x, p)``.
    """

    def __init__(self, children):
        self.children = tuple(children)
        if not self.children:
            raise ValueError("an operator composition needs at least one child")

    def terminal_least_squares_gradient(self, x0, params, target):
        """Evaluate value and gradient using only ordinary adjoint actions."""
        x0 = np.asarray(x0)
        params = np.asarray(params)
        if x0.ndim != 1 or params.ndim != 1:
            raise ValueError("x0 and params must be one-dimensional")
        dtype = _inexact_result_type(x0.dtype, params.dtype)
        target = _vector("target", target, x0.size, dtype=dtype)

        states = [np.asarray(x0, dtype=dtype).copy()]
        for child in self.children:
            state = states[-1]
            state_out = _vector(
                "child state output",
                child.forward(state, params),
                state.size,
                dtype=dtype,
            ).copy()
            states.append(state_out)

        residual = states[-1] - target
        value = 0.5*float(np.dot(residual, residual))
        state_adjoint = residual.copy()
        gradient_params = np.zeros(params.size, dtype=dtype)
        for index in range(len(self.children) - 1, -1, -1):
            child = self.children[index]
            state = states[index]
            gradient_params += _vector(
                "child parameter adjoint",
                child.jacT_params_action(state, params, state_adjoint),
                params.size,
                dtype=dtype,
            )
            state_adjoint = _vector(
                "child state adjoint",
                child.jacT_x_action(state, params, state_adjoint),
                state.size,
                dtype=dtype,
            )

        return CompositionGradient(
            value=value,
            states=tuple(states),
            gradient_params=gradient_params,
            gradient_initial_state=state_adjoint,
        )

    def terminal_least_squares_hvp(
        self, x0, params, target, param_direction, state_direction=None
    ):
        """Evaluate a terminal least-squares exact HVP through all children."""
        x0 = np.asarray(x0)
        params = np.asarray(params)
        if x0.ndim != 1 or params.ndim != 1:
            raise ValueError("x0 and params must be one-dimensional")
        dtype = _inexact_result_type(x0.dtype, params.dtype)
        target = _vector("target", target, x0.size, dtype=dtype)
        param_direction = _vector(
            "param_direction", param_direction, params.size, dtype=dtype
        ).copy()
        if state_direction is None:
            state_direction = np.zeros(x0.size, dtype=dtype)
        state_direction = _vector(
            "state_direction", state_direction, x0.size, dtype=dtype
        ).copy()

        states = [np.asarray(x0, dtype=dtype).copy()]
        tangents = [state_direction]
        for child in self.children:
            state = states[-1]
            tangent = tangents[-1]
            state_out = _vector(
                "child state output",
                child.forward(state, params),
                state.size,
                dtype=dtype,
            ).copy()
            tangent_out = _vector(
                "child tangent output",
                child.jac_x_action(state, params, tangent)
                + child.jac_params_action(state, params, param_direction),
                state.size,
                dtype=dtype,
            ).copy()
            states.append(state_out)
            tangents.append(tangent_out)

        residual = states[-1] - target
        value = 0.5*float(np.dot(residual, residual))
        state_adjoint = residual.copy()
        incremental_state_adjoint = tangents[-1].copy()
        gradient_params = np.zeros(params.size, dtype=dtype)
        hvp_params = np.zeros(params.size, dtype=dtype)

        for index in range(len(self.children) - 1, -1, -1):
            child = self.children[index]
            state = states[index]
            tangent = tangents[index]
            gradient_params += _vector(
                "child parameter adjoint",
                child.jacT_params_action(state, params, state_adjoint),
                params.size,
                dtype=dtype,
            )
            hvp_params += _vector(
                "child parameter HVP",
                child.jacT_params_action(
                    state, params, incremental_state_adjoint
                )
                + child.directional_jacT_params_action(
                    state,
                    params,
                    tangent,
                    param_direction,
                    state_adjoint,
                ),
                params.size,
                dtype=dtype,
            )
            incremental_state_adjoint = _vector(
                "child incremental state adjoint",
                child.jacT_x_action(
                    state, params, incremental_state_adjoint
                )
                + child.directional_jacT_x_action(
                    state,
                    params,
                    tangent,
                    param_direction,
                    state_adjoint,
                ),
                state.size,
                dtype=dtype,
            )
            state_adjoint = _vector(
                "child state adjoint",
                child.jacT_x_action(state, params, state_adjoint),
                state.size,
                dtype=dtype,
            )

        return CompositionHVP(
            value=value,
            states=tuple(states),
            tangents=tuple(tangents),
            gradient_params=gradient_params,
            gradient_initial_state=state_adjoint,
            hvp_params=hvp_params,
            hvp_initial_state=incremental_state_adjoint,
        )
