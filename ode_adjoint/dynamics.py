import numpy as np
import scipy as sp

class _Dynamics():
    def __init__(self):
        pass

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

    def get_x_size(self):
        return 1

    def get_param_size(self):
        return 2

    def get_param_bounds(self):
        return (1e-6, None), (1e-6, None)
