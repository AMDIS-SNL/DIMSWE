
from .operators import ForcingBase
from firedrake import exp, Function, FunctionSpace, inner
import ufl
import numpy as np

def qsat(h, s, B, q0, H0, g):
    return q0 * H0 / (h + B) * exp(20.*(1.-s/g))

class ThreeWayPhysics(ForcingBase):
    def __init__(self, parameters, vars, spaces, initcond):
        self.vars = vars
        self.spaces = spaces
        self.name = 'threewayphysics'

        self.g = initcond.g
        self.q0 = initcond.q0
        self.H0 = initcond.H0
        self.dt = parameters['timestepping']['dt']
        #self.L = 10
        #self.beta2 = self.g * self.L
        #self.gamma_r = 10.**(-3.)
        self.treat_as_coeffs = parameters['threewayphysics']['treat_as_coeffs']
        if not self.treat_as_coeffs:
            self.gamma_r = parameters['threewayphysics']['gamma_r']
            self.qprecip = parameters['threewayphysics']['qprecip']
            self.L = parameters['threewayphysics']['L']

        self.tau_v = self.dt
        self.tau_r = self.dt
        #self.qprecip = 10.**(-4.)



        if not self.spaces is None:
            self.dx = spaces.dx
            self.B = Function(FunctionSpace(self.spaces.mesh, 'CG', self.spaces.order, variant="spectral"), name='topo')
            if self.treat_as_coeffs:
                self.gamma_r_space = FunctionSpace(self.spaces.mesh, 'R', 0)
                self.qprecip_space = FunctionSpace(self.spaces.mesh, 'R', 0)
                self.L_space = FunctionSpace(self.spaces.mesh, 'R', 0)

    def has_coeff(self):
        return self.treat_as_coeffs

    def set_coeffs(self, parameters, coeff):
        gamma_r = parameters['gamma_r']
        qprecip = parameters['qprecip']
        L = parameters['L']

        coeff['gamma_r'].assign(gamma_r)
        coeff['qprecip'].assign(qprecip)
        coeff['L'].assign(L)

#WHAT ARE SOME REASONABLE BOUNDS HERE?
    def get_coeff_bounds(self):
        return np.array([1e-6, 1e-6, 1e-6]), np.array([np.inf, np.inf, np.inf])

    def get_coeff_scaling_factors(self):
        return np.array([.001, .0001, 10.])

    def get_coeff(self):
        return [['gamma_r', self.gamma_r_space], ['qprecip', self.qprecip_space], ['L', self.L_space]]

    def initialize(self, varexpr):
        self.B.interpolate(varexpr['bottom_topography'])

    def rhs(self, xvars, t, coeff, xhats):
        h = xvars['h']
        S = xvars['S']
        Qv = xvars['Qv']
        Qr = xvars['Qr']
        Qc = xvars['Qc']
        qv = Qv / h
        qc = Qc / h
        qr = Qr / h
        s = S/h

        if self.treat_as_coeffs:
            beta2 = self.g * coeff['L']
        else:
            beta2 = self.g * self.L
        q_sat = qsat(h, s, self.B, self.q0, self.H0, self.g)
        gamma_v = 1./(1. + q_sat * 20. * beta2 / self.g)
        Dqv = ufl.max_value(0.0, gamma_v*(qv-q_sat)/self.tau_v)
        Dqc = ufl.min_value(qc/self.dt, ufl.max_value(0., gamma_v * (q_sat - qv)/self.tau_v))
        if self.treat_as_coeffs:
            Dqr = ufl.max_value(0.0, coeff['gamma_r'] * (qc-coeff['qprecip'])/self.tau_r)
        else:
            Dqr = ufl.max_value(0.0, self.gamma_r * (qc-self.qprecip)/self.tau_r)
        Sv = Dqc - Dqv
        Sr = Dqr
        Sc = Dqv - Dqc - Dqr
        Sb = Dqc - Dqv
        Qvhat = xhats['Qv']
        Qrhat = xhats['Qr']
        Qchat = xhats['Qc']
        Shat = xhats['S']
        expr = 0.
        expr = expr - inner(Qvhat, h*Sv)*self.dx
        expr = expr - inner(Qrhat, h*Sr)*self.dx
        expr = expr - inner(Qchat, h*Sc)*self.dx
        expr = expr - inner(Shat, h*beta2*Sb)*self.dx
        return expr


#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
