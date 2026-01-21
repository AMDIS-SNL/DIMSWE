
from .operators import ForcingBase
from firedrake import exp
import ufl

def qsat(h, s, B, q0, H0, g):
    return q0 * H0 / (h + B) * exp(20.*(1.-s/g))
    
class ThreeWayPhysics(ForcingBase):
    def __init__(self, parameters, vars, spaces):
        self.vars = vars
        self.spaces = spaces
        self.name = 'threewayphysics'
        
        self.g = 1.0
        self.q0 = 1.0
        self.H0 = 1.0
        self.B = 1.0
        self.dt = 1.0
#GET G FROM INITIAL CONDITION
#ALSO MANY OTHERS SUCH AS H, ETC.
        self.L = 10
        self.beta2 = self.g * self.L
        self.gamma_r = 10.**(-3.)
        
        self.tau_v = self.dt
        self.tau_r = self.dt
        self.qprecip = 10.**(-4.)
        

    def rhs(self, qvars, xhats):
        h = qvars['h']
        S = qvars['S']
        Qv = qvars['Qv']
        Qr = qvars['Qr']
        Qc = qvars['Qc']
        qv = Qv / h
        qc = Qc / h
        qr = Qr / h       
        s = h/S
        
        q_sat = qsat(h, s, self.B, self.q0, self.H0, self.g)
        gamma_v = 1./(1. + qsat * 20. * self.beta2 / self.g)
        Dqv = 0.0 #ufl.Max(0.0, (qv-qsat)/self.tau_v)
        Dqc = 0.0 #ufl.Min(qc/self.dt, ufl.Max(0., gamma_v * (qsat - qv)/self.tau_v))
        Dqr = 0.0 #ufl.Max(0.0, self.gamma_r * (qc-self.qprecip)/self.tau_r)
        Sv = Dqc - Dqv
        Sr = Dqr
        Sc = Dqv - Dqc - Dqr
        Sb = Dqc - Dqv
        Qvhat = xhats['Qv']
        Qrhat = xhats['Qr']
        Qchat = xhats['Qc']
        Shat = xhats['S']
        expr = 0.
        expr = expr - inner(Qvhat, dens*Sv)*self.dx
        expr = expr - inner(Qrhat, dens*Sr)*self.dx
        expr = expr - inner(Qchat, dens*Sc)*self.dx
        expr = expr - inner(Shat, dens*self.beta2*Sb)*self.dx
        return expr
 
        
#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
