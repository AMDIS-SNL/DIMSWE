import numpy as np
import scipy as sp

class _Dynamics():
    def __init__(self):
        pass

    def jac_x_action(self, x, t, params, direction):
        """Return ``f_x @ direction`` with state-shaped output ``(nx,)``."""
        return self.jac_x(x, t, params).dot(direction)

    def jac_params_action(self, x, t, params, direction):
        """Return ``f_p @ direction`` with state-shaped output ``(nx,)``."""
        return self.jac_params(x, t, params).dot(direction)

    def jacT_x_action(self, x, t, params, adjoint):
        """Return ``f_x.T @ adjoint`` with output shape ``(nx,)``."""
        return self.jacT_x(x, t, params).dot(adjoint)

    def jacT_params_action(self, x, t, params, adjoint):
        """Return ``f_p.T @ adjoint`` with output shape ``(nparams,)``."""
        return self.jacT_params(x, t, params).dot(adjoint)

    def directional_jacT_x_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        """Apply the directional derivative of ``f_x.T`` to ``adjoint``.

        This is ``(f_xx[state_direction] + f_xp[param_direction]).T``
        applied to an ``(nx,)`` adjoint.  The result has shape ``(nx,)``.
        Implementations form only this contracted action, not a rank-3 tensor.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide second derivative actions"
        )

    def directional_jacT_params_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        """Apply the directional derivative of ``f_p.T`` to ``adjoint``.

        This is ``(f_px[state_direction] + f_pp[param_direction]).T``
        applied to an ``(nx,)`` adjoint.  The result has shape ``(nparams,)``.
        Implementations form only this contracted action, not a rank-3 tensor.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide second derivative actions"
        )

class LotkaVolterra(_Dynamics):

    def rhs(self, x, t, params):
        x1 = params[0]*x[0] - params[1]*x[0]*x[1]
        x2 = params[2]*x[0]*x[1] - params[3]*x[1]
        return np.array([x1,x2])

    def jac_x(self, x, t, params):
        return np.array([[params[0]-params[1]*x[1],-params[1]*x[0]],[params[2]*x[1],params[2]*x[0]-params[3]]])

    def jacT_x(self, x, t, params):
        return np.array([[params[0]-params[1]*x[1],params[2]*x[1]],[-params[1]*x[0],params[2]*x[0]-params[3]]])
        #self.jac_x(x,t,params).T

    def jac_params(self, x, t, params):
        return np.array([[x[0],-x[0]*x[1],0,0],[0,0,x[0]*x[1],-x[1]]])

    def jacT_params(self, x, t, params):
        #return self.jac_params(x,t,params).T
        return np.array([[x[0],0],[-x[0]*x[1], 0],[0,x[0]*x[1]],[0,-x[1]]])

    def directional_jacT_x_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        w0, w1 = state_direction
        q0, q1, q2, q3 = param_direction
        d_jac_x = np.array([
            [q0 - q1*x[1] - params[1]*w1, -q1*x[0] - params[1]*w0],
            [q2*x[1] + params[2]*w1, q2*x[0] + params[2]*w0 - q3],
        ])
        return d_jac_x.T.dot(adjoint)

    def directional_jacT_params_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        w0, w1 = state_direction
        product_direction = w0*x[1] + x[0]*w1
        d_jac_params = np.array([
            [w0, -product_direction, 0.0, 0.0],
            [0.0, 0.0, product_direction, -w1],
        ])
        return d_jac_params.T.dot(adjoint)

    def get_x_size(self):
        return 2

    def get_param_size(self):
        return 4

    def get_param_bounds(self):
        return (1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)

    def get_ic_bounds(self):
        return (1e-6, None), (1e-6, None)


class LogisticEquation(_Dynamics):

    def rhs(self, x, t, params):
        x1 = params[0]*x[0] * (1. - x[0]/params[1])
        return np.array([x1,])

    def jac_x(self, x, t, params):
        return np.array([[params[0]*(1.-2.*x[0]/params[1]),]])

    def jacT_x(self, x, t, params):
        return np.array([[params[0]*(1.-2.*x[0]/params[1]),]])
        #self.jac_x(x,t,params).T

    def jac_params(self, x, t, params):
        return np.array([[x[0] - x[0]*x[0]/params[1],params[0]*x[0]*x[0]/params[1]/params[1]]])

    def jacT_params(self, x, t, params):
        return self.jac_params(x,t,params).T
        #np.array([[x[0],0],[-x[0]*x[1], 0],[0,x[0]*x[1]],[0,-x[1]]])

    def directional_jacT_x_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        state = x[0]
        growth, capacity = params
        w = state_direction[0]
        q_growth, q_capacity = param_direction
        d_jac_x = (
            q_growth*(1.0 - 2.0*state/capacity)
            + growth*(-2.0*w/capacity + 2.0*state*q_capacity/capacity**2)
        )
        return np.array([d_jac_x*adjoint[0]])

    def directional_jacT_params_action(
        self, x, t, params, state_direction, param_direction, adjoint
    ):
        state = x[0]
        growth, capacity = params
        w = state_direction[0]
        q_growth, q_capacity = param_direction
        d_jac_params = np.array([[
            w*(1.0 - 2.0*state/capacity)
            + state**2*q_capacity/capacity**2,
            q_growth*state**2/capacity**2
            + growth*(2.0*state*w/capacity**2
                      - 2.0*state**2*q_capacity/capacity**3),
        ]])
        return d_jac_params.T.dot(adjoint)

    def get_x_size(self):
        return 1

    def get_param_size(self):
        return 2

    def get_param_bounds(self):
        return (1e-6, None), (1e-6, None)
