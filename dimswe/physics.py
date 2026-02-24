
from .operators import ForcingBase
from firedrake import exp, Function, FunctionSpace, inner
import ufl

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
        self.L = 10
        self.beta2 = self.g * self.L
        self.gamma_r = 10.**(-3.)

        self.tau_v = self.dt
        self.tau_r = self.dt
        self.qprecip = 10.**(-4.)

        if not self.spaces is None:
            self.dx = spaces.dx
            self.B = Function(FunctionSpace(self.spaces.mesh, 'CG', self.spaces.order, variant="spectral"))

    def initialize(self, varexpr):
        self.B.interpolate(varexpr['bottom_topography'])

    def rhs(self, qvars, xhats):
        h = qvars['h']
        S = qvars['S']
        Qv = qvars['Qv']
        Qr = qvars['Qr']
        Qc = qvars['Qc']
        qv = Qv / h
        qc = Qc / h
        qr = Qr / h
        s = S/h

        q_sat = qsat(h, s, self.B, self.q0, self.H0, self.g)
        gamma_v = 1./(1. + q_sat * 20. * self.beta2 / self.g)
        Dqv = ufl.max_value(0.0, gamma_v*(qv-q_sat)/self.tau_v)
        Dqc = ufl.min_value(qc/self.dt, ufl.max_value(0., gamma_v * (q_sat - qv)/self.tau_v))
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
        expr = expr - inner(Shat, h*self.beta2*Sb)*self.dx
        return expr


#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
