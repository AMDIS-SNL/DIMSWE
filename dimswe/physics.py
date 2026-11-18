
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
            self.physics_dx = self.get_physics_measure(parameters)
            self.B = Function(FunctionSpace(self.spaces.mesh, 'CG', self.spaces.order, variant="spectral"), name='topo')
            if self.treat_as_coeffs:
                self.gamma_r_space = FunctionSpace(self.spaces.mesh, 'R', 0)
                self.qprecip_space = FunctionSpace(self.spaces.mesh, 'R', 0)
                self.L_space = FunctionSpace(self.spaces.mesh, 'R', 0)

    def get_physics_measure(self, parameters):
        """Return the single quadrature measure used for all of the physics source terms.

        The vapour, cloud, rain (and entropy) sources cancel pointwise (Sv + Sc + Sr = 0), so total
        water is conserved only if every source integral uses the same quadrature rule. With the
        default dx, UFL estimates a quadrature degree per integral; the rain source Sr = Dqr does
        not involve qsat and gets a lower degree than Sc, which contains -Dqr, so the kinked rain
        term is integrated differently in the Qr and Qc equations and water leaks once it rains.
        A lumped-mass measure is already one fixed (GLL) scheme and is used as is; otherwise a fixed
        degree is attached, set by threewayphysics.quadrature_degree (default 2 * order + 2).
        """
        if parameters['spatial-discretization']['lump_mass']:
            return self.dx
        default_degree = 2 * self.spaces.order + 2
        degree = parameters['threewayphysics'].get('quadrature_degree', default_degree)
        return self.dx(degree=degree)

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
        sl = S/h

        if self.treat_as_coeffs:
            beta2 = self.g * coeff['L']
        else:
            beta2 = self.g * self.L
        q_sat = qsat(h, sl, self.B, self.q0, self.H0, self.g)
        gamma_v = 1./(1. + q_sat * 20. * beta2 / self.g)
        Dqv = ufl.max_value(0.0, gamma_v*(qv-q_sat)/self.tau_v)
        Dqc = ufl.min_value(qc/self.dt, ufl.max_value(0., gamma_v * (q_sat - qv)/self.tau_v))
        if self.treat_as_coeffs:
            Dqr = ufl.max_value(0.0, coeff['gamma_r'] * (qc-coeff['qprecip'])/self.tau_r)
        else:
            Dqr = ufl.max_value(0.0, self.gamma_r * (qc-self.qprecip)/self.tau_r)
        # Cap rain production by the cloud water left after evaporation, so that one Euler step of
        # length dt keeps qc >= 0 pointwise: qc + dt*(Dqv - Dqc - Dqr) >= qc - dt*Dqc - dt*Dqr >= 0.
        # Without it, subsaturated air that evaporates all its cloud in one step (dt*Dqc = qc) while
        # qc > qprecip would end with qc = -gamma_r*(qc - qprecip) < 0. Condensation and evaporation
        # are mutually exclusive, so the cap is only active in that case.
        Dqr = ufl.min_value(Dqr, ufl.max_value(0.0, qc/self.dt - Dqc))
        Sv = Dqc - Dqv
        Sr = Dqr
        Sc = Dqv - Dqc - Dqr
        Sb = Dqc - Dqv
        Qvhat = xhats['Qv']
        Qrhat = xhats['Qr']
        Qchat = xhats['Qc']
        Shat = xhats['S']
        # one integrand on one fixed measure, so the cancelling sources share a quadrature rule
        integrand = inner(Qvhat, h*Sv) + inner(Qrhat, h*Sr) + inner(Qchat, h*Sc) + inner(Shat, h*beta2*Sb)
        return -integrand*self.physics_dx


#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
