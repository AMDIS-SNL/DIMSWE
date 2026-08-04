import numpy as np
import scipy as sp
from dataclasses import dataclass


def _inexact_result_type(*values):
    dtype = np.result_type(*values)
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.dtype(float)
    return dtype


def _floating_values_agree(left, right):
    """Compare cached floating values at a small roundoff-level tolerance."""
    dtype = _inexact_result_type(np.asarray(left).dtype, np.asarray(right).dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    tolerance = 8.0*np.finfo(real_dtype).eps
    return np.allclose(
        left, right, rtol=tolerance, atol=tolerance, equal_nan=False
    )


@dataclass(frozen=True)
class RKStepData:
    """Immutable-by-convention primal data for one explicit RK step."""

    t: float
    dt: float
    state_in: np.ndarray
    state_out: np.ndarray
    params: np.ndarray
    stage_states: tuple
    stage_rhs: tuple


@dataclass(frozen=True)
class RKTangentStepData:
    """Primal and directional data for one explicit RK step."""

    primal: RKStepData
    state_direction_in: np.ndarray
    state_direction_out: np.ndarray
    param_direction: np.ndarray
    stage_state_directions: tuple
    stage_rhs_directions: tuple


@dataclass(frozen=True)
class RKAdjointStepResult:
    """Ordinary reverse result for one step."""

    state_adjoint_in: np.ndarray
    parameter_adjoint: np.ndarray
    stage_rhs_adjoints: tuple
    stage_state_adjoints: tuple


@dataclass(frozen=True)
class RKHVPAdjointStepResult:
    """Ordinary and incremental reverse result for one step."""

    state_adjoint_in: np.ndarray
    incremental_state_adjoint_in: np.ndarray
    parameter_adjoint: np.ndarray
    parameter_hvp: np.ndarray
    stage_rhs_adjoints: tuple
    incremental_stage_rhs_adjoints: tuple
    stage_state_adjoints: tuple
    incremental_stage_state_adjoints: tuple


class _TimeStepper():

    def compute_state(self, nsteps, params, t0, x0, dt):

        t = np.zeros(nsteps+1)
        xn = np.zeros((nsteps+1,self.dynamics.get_x_size()))
        t[0] = t0
        xn[0,:] = x0

        for n in range(nsteps):
            self.take_forward_step(xn[n+1,:], dt, t[n], xn[n,:], params)
            t[n+1] = t[n] + dt
        return xn, t

    def compute_state_block(self, nblocks, nsteps, t0, x0, dt, params):
        xns = []
        tns = []
        steps = []
        for k in range(nblocks):
            if (k==0):
                xn, t = self.compute_state(nsteps, params, t0, x0, dt)
            else:
                xn, t = self.compute_state(nsteps, params, t[-1], xn[-1,:], dt)
            xns.append(xn[::nsteps,:])
            tns.append(t[::nsteps])
            steps.append(nsteps)
        return xns, steps, tns

class _GeneralRK(_TimeStepper):
    def __init__(self, dynamics, A, c, b, nstages):
        self.A = A
        self.c = c
        self.b = b
        self.nstages = nstages
        self.dynamics = dynamics
        self.Fi = []
        self.ti = []
        self.mui = []
        self.Yi = []
        self.li = []
        self._last_step = None
        for i in range(self.nstages):
            self.Fi.append(np.zeros(dynamics.get_x_size()))
            self.Yi.append(np.zeros(dynamics.get_x_size()))
            self.li.append(np.zeros(dynamics.get_x_size()))
            self.ti.append(0.)
            self.mui.append(np.zeros(dynamics.get_x_size()))

    def _vector(self, name, value, size, dtype=None):
        array = np.asarray(value, dtype=dtype)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must have a numeric dtype")
        return array

    def _state_vector(self, name, value, dtype=None):
        return self._vector(name, value, self.dynamics.get_x_size(), dtype=dtype)

    def _param_vector(self, name, value, dtype=None):
        return self._vector(name, value, self.dynamics.get_param_size(), dtype=dtype)

    def forward_step_data(self, dt, tn, xn, params):
        """Evaluate one step and return copied primal stage data.

        The method does not mutate ``xn`` or ``params``.  It is the reusable
        primal operation used by Euler, RK4, and every other explicit tableau
        represented by this class.
        """
        xn = self._state_vector("xn", xn)
        params = self._param_vector("params", params)
        dtype = _inexact_result_type(
            xn.dtype, params.dtype, np.asarray(dt).dtype
        )
        state = np.asarray(xn, dtype=dtype).copy()
        parameters = np.asarray(params, dtype=dtype).copy()
        stage_states = []
        stage_rhs = []

        for i in range(self.nstages):
            stage_state = state.copy()
            for j in range(i):
                stage_state += dt*self.A[i, j]*stage_rhs[j]
            stage_time = tn + self.c[i]*dt
            rhs = self._state_vector(
                f"rhs at stage {i}",
                self.dynamics.rhs(stage_state, stage_time, parameters),
                dtype=dtype,
            ).copy()
            stage_states.append(stage_state)
            stage_rhs.append(rhs)

        state_out = state.copy()
        for i in range(self.nstages):
            state_out += dt*self.b[i]*stage_rhs[i]

        return RKStepData(
            t=float(tn),
            dt=float(dt),
            state_in=state,
            state_out=state_out,
            params=parameters,
            stage_states=tuple(stage_states),
            stage_rhs=tuple(stage_rhs),
        )

    def linearize_step(self, dt, tn, xn, params, state_direction, param_direction):
        """Evaluate one step and its forward tangent in ``(w, q)``.

        ``state_direction`` has shape ``(nx,)`` and ``param_direction`` has
        shape ``(nparams,)``.  The returned state and all stage tangents have
        shape ``(nx,)``.
        """
        primal = self.forward_step_data(dt, tn, xn, params)
        dtype = primal.state_out.dtype
        state_direction = self._state_vector(
            "state_direction", state_direction, dtype=dtype
        ).copy()
        param_direction = self._param_vector(
            "param_direction", param_direction, dtype=dtype
        ).copy()
        stage_state_directions = []
        stage_rhs_directions = []

        for i in range(self.nstages):
            stage_direction = state_direction.copy()
            for j in range(i):
                stage_direction += (
                    primal.dt*self.A[i, j]*stage_rhs_directions[j]
                )
            stage_time = primal.t + self.c[i]*primal.dt
            rhs_direction = (
                self.dynamics.jac_x_action(
                    primal.stage_states[i], stage_time, primal.params,
                    stage_direction,
                )
                + self.dynamics.jac_params_action(
                    primal.stage_states[i], stage_time, primal.params,
                    param_direction,
                )
            )
            rhs_direction = self._state_vector(
                f"rhs tangent at stage {i}", rhs_direction, dtype=dtype
            ).copy()
            stage_state_directions.append(stage_direction)
            stage_rhs_directions.append(rhs_direction)

        state_direction_out = state_direction.copy()
        for i in range(self.nstages):
            state_direction_out += (
                primal.dt*self.b[i]*stage_rhs_directions[i]
            )

        return RKTangentStepData(
            primal=primal,
            state_direction_in=state_direction,
            state_direction_out=state_direction_out,
            param_direction=param_direction,
            stage_state_directions=tuple(stage_state_directions),
            stage_rhs_directions=tuple(stage_rhs_directions),
        )

    def reverse_step(self, step, state_adjoint_out):
        """Apply the exact discrete transpose of a cached RK step."""
        if not isinstance(step, RKStepData):
            raise TypeError("step must be RKStepData")
        dtype = step.state_out.dtype
        state_adjoint_out = self._state_vector(
            "state_adjoint_out", state_adjoint_out, dtype=dtype
        )
        stage_rhs_adjoints = [np.zeros_like(state_adjoint_out)
                              for _ in range(self.nstages)]
        stage_state_adjoints = [np.zeros_like(state_adjoint_out)
                                for _ in range(self.nstages)]

        for i in range(self.nstages - 1, -1, -1):
            stage_rhs_adjoints[i][:] = step.dt*self.b[i]*state_adjoint_out
            for j in range(i + 1, self.nstages):
                stage_rhs_adjoints[i] += (
                    step.dt*self.A[j, i]*stage_state_adjoints[j]
                )
            stage_time = step.t + self.c[i]*step.dt
            stage_state_adjoints[i][:] = self._state_vector(
                f"state adjoint at stage {i}",
                self.dynamics.jacT_x_action(
                    step.stage_states[i], stage_time, step.params,
                    stage_rhs_adjoints[i],
                ),
                dtype=dtype,
            )

        state_adjoint_in = state_adjoint_out.copy()
        parameter_adjoint = np.zeros(
            self.dynamics.get_param_size(), dtype=dtype
        )
        for i in range(self.nstages):
            state_adjoint_in += stage_state_adjoints[i]
            stage_time = step.t + self.c[i]*step.dt
            parameter_adjoint += self._param_vector(
                f"parameter adjoint at stage {i}",
                self.dynamics.jacT_params_action(
                    step.stage_states[i], stage_time, step.params,
                    stage_rhs_adjoints[i],
                ),
                dtype=dtype,
            )

        return RKAdjointStepResult(
            state_adjoint_in=state_adjoint_in,
            parameter_adjoint=parameter_adjoint,
            stage_rhs_adjoints=tuple(stage_rhs_adjoints),
            stage_state_adjoints=tuple(stage_state_adjoints),
        )

    def reverse_hvp_step(self, tangent_step, state_adjoint_out,
                         incremental_state_adjoint_out):
        """Apply the ordinary and incremental reverse RK stage graph."""
        if not isinstance(tangent_step, RKTangentStepData):
            raise TypeError("tangent_step must be RKTangentStepData")
        step = tangent_step.primal
        ordinary = self.reverse_step(step, state_adjoint_out)
        dtype = step.state_out.dtype
        state_adjoint_out = self._state_vector(
            "state_adjoint_out", state_adjoint_out, dtype=dtype
        )
        incremental_state_adjoint_out = self._state_vector(
            "incremental_state_adjoint_out",
            incremental_state_adjoint_out,
            dtype=dtype,
        )
        incremental_stage_rhs_adjoints = [
            np.zeros_like(incremental_state_adjoint_out)
            for _ in range(self.nstages)
        ]
        incremental_stage_state_adjoints = [
            np.zeros_like(incremental_state_adjoint_out)
            for _ in range(self.nstages)
        ]

        for i in range(self.nstages - 1, -1, -1):
            incremental_stage_rhs_adjoints[i][:] = (
                step.dt*self.b[i]*incremental_state_adjoint_out
            )
            for j in range(i + 1, self.nstages):
                incremental_stage_rhs_adjoints[i] += (
                    step.dt*self.A[j, i]
                    * incremental_stage_state_adjoints[j]
                )
            stage_time = step.t + self.c[i]*step.dt
            incremental_stage_state_adjoints[i][:] = self._state_vector(
                f"incremental state adjoint at stage {i}",
                self.dynamics.jacT_x_action(
                    step.stage_states[i], stage_time, step.params,
                    incremental_stage_rhs_adjoints[i],
                )
                + self.dynamics.directional_jacT_x_action(
                    step.stage_states[i], stage_time, step.params,
                    tangent_step.stage_state_directions[i],
                    tangent_step.param_direction,
                    ordinary.stage_rhs_adjoints[i],
                ),
                dtype=dtype,
            )

        incremental_state_adjoint_in = (
            incremental_state_adjoint_out.copy()
        )
        parameter_hvp = np.zeros(
            self.dynamics.get_param_size(), dtype=dtype
        )
        for i in range(self.nstages):
            incremental_state_adjoint_in += (
                incremental_stage_state_adjoints[i]
            )
            stage_time = step.t + self.c[i]*step.dt
            parameter_hvp += self._param_vector(
                f"parameter HVP at stage {i}",
                self.dynamics.jacT_params_action(
                    step.stage_states[i], stage_time, step.params,
                    incremental_stage_rhs_adjoints[i],
                )
                + self.dynamics.directional_jacT_params_action(
                    step.stage_states[i], stage_time, step.params,
                    tangent_step.stage_state_directions[i],
                    tangent_step.param_direction,
                    ordinary.stage_rhs_adjoints[i],
                ),
                dtype=dtype,
            )

        return RKHVPAdjointStepResult(
            state_adjoint_in=ordinary.state_adjoint_in,
            incremental_state_adjoint_in=incremental_state_adjoint_in,
            parameter_adjoint=ordinary.parameter_adjoint,
            parameter_hvp=parameter_hvp,
            stage_rhs_adjoints=ordinary.stage_rhs_adjoints,
            incremental_stage_rhs_adjoints=tuple(
                incremental_stage_rhs_adjoints
            ),
            stage_state_adjoints=ordinary.stage_state_adjoints,
            incremental_stage_state_adjoints=tuple(
                incremental_stage_state_adjoints
            ),
        )

#THIS IS EXPLICIT ONLY...
    def take_forward_step(self, xnp1, dt, tn, xn, params):
        step = self.forward_step_data(dt, tn, xn, params)
        self._last_step = step
        for i in range(self.nstages):
            self.Yi[i][:] = step.stage_states[i]
            self.Fi[i][:] = step.stage_rhs[i]
            self.ti[i] = tn + self.c[i]*dt
        xnp1[:] = step.state_out

#THIS IS EXPLICIT ONLY...
    def take_adjoint_step(self, ts_grad, delta_lambda, dt, tnp1, lambda_np1, params):
        tn = tnp1 - dt
        if self._last_step is None:
            raise RuntimeError("take_forward_step must precede take_adjoint_step")

        dt_array = np.asarray(dt)
        tnp1_array = np.asarray(tnp1)
        tn_array = np.asarray(tn)
        if dt_array.shape != ():
            raise ValueError(f"dt must be scalar, got shape {dt_array.shape}")
        if tnp1_array.shape != ():
            raise ValueError(
                f"tnp1 must be scalar, got shape {tnp1_array.shape}"
            )
        if tn_array.shape != ():
            raise ValueError(
                f"tnp1 - dt must be scalar, got shape {tn_array.shape}"
            )
        supplied_dt = float(dt_array)
        supplied_tn = float(tn_array)
        supplied_params = self._param_vector("params", params)
        cached = self._last_step

        if not _floating_values_agree(cached.dt, supplied_dt):
            raise ValueError(
                "cached RK step dt mismatch: "
                f"cached {cached.dt!r}, supplied {supplied_dt!r}"
            )
        if not _floating_values_agree(cached.t, supplied_tn):
            raise ValueError(
                "cached RK step start-time mismatch: "
                f"cached {cached.t!r}, supplied tnp1 - dt {supplied_tn!r}"
            )
        if not _floating_values_agree(cached.params, supplied_params):
            raise ValueError(
                "cached RK step parameter mismatch: supplied params do not "
                "agree with the parameters used for the cached forward step"
            )

        reverse = self.reverse_step(cached, lambda_np1)
        for i in range(self.nstages):
            self.mui[i][:] = reverse.stage_rhs_adjoints[i]
            self.li[i][:] = reverse.stage_state_adjoints[i]
        # Preserve the historical optimizer convention: it evolves the
        # negative residual adjoint and expects a sign-flipped parameter term.
        ts_grad[:] = -reverse.parameter_adjoint
        delta_lambda[:] = reverse.state_adjoint_in - lambda_np1

class Euler(_GeneralRK):
    def __init__(self, dynamics):
        A = np.array([[0.0,],])
        b = np.array([1.0,])
        c = np.array([0.0,])
        _GeneralRK.__init__(self, dynamics, A, c, b, 1)

class RK4(_GeneralRK):
    def __init__(self, dynamics):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        b = np.array([1./6., 1./3., 1./3., 1./6.])
        c = np.array([0.0, 0.5, 0.5, 1.0])
        _GeneralRK.__init__(self, dynamics, A, c, b, 4)

class SSPRK3(_GeneralRK):
    def __init__(self, dynamics, nstages):
        A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        b = np.array([1./6., 1./6., 2./3.])
        c = np.array([0.0, 1.0, 0.5])
        _GeneralRK.__init__(self, dynamics, A, c, b, 3)

class SSPRK43(_GeneralRK):
    def __init__(self, dynamics, nstages):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0/6.0, 1.0/6.0, 1.0/6.0, 0.0]])
        b = np.array([1./6., 1./6., 1./6., 3./6.])
        c = np.array([0.0, 0.5, 1.0, 0.5])
        _GeneralRK.__init__(self, dynamics, A, c, b, 4)
