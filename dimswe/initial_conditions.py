from firedrake import as_vector, Constant, SpatialCoordinate, exp, pi, sin, cos, sqrt
import ufl
from .physics import qsat
import scipy as sp

#WE SHOULD REALLY SPLIT THIS INTO INITIAL CONDITION ROUTINES FOR VARIOUS "MODELS"
#AND ACTUALLY CREATE VARIOUS MODELS CLASSES IE MAXWELL, ADV DENS, EULER MAXWELL, ETC.
#THIS WOULD SIMPLIFY HOW DYNAMICS WORKS...

def get_initial_condition(parameters):
    if parameters['initial-conditions']['name'] == 'gaussian': return Gaussian(parameters)
    elif parameters['initial-conditions']['name'] == 'doublevortex': return DoubleVortex(parameters)
    elif parameters['initial-conditions']['name'] == 'TC2': return TC2(parameters)
    elif parameters['initial-conditions']['name'] == 'TC5': return TC5(parameters)
    elif parameters['initial-conditions']['name'] == 'galewsky': return Galewsky(parameters)
    elif parameters['initial-conditions']['name'] == 'densitywave': return DensityWave(parameters)

    elif parameters['initial-conditions']['name'] == 'planewave': return PlaneWave(parameters)

#ADD LOTS HERE!!!
    elif parameters['initial-conditions']['name'] == 'RP1': return RiemannProblem1(parameters)
    elif parameters['initial-conditions']['name'] == 'RP2': return RiemannProblem2(parameters)
    elif parameters['initial-conditions']['name'] == 'RP3': return RiemannProblem3(parameters)
    elif parameters['initial-conditions']['name'] == 'ModifiedSod': return ModifiedSod(params)
    elif parameters['initial-conditions']['name'] == 'ToroTest3': return ToroTest3(params)
    elif parameters['initial-conditions']['name'] == 'StationaryContact': return StationaryContact(params)
    elif parameters['initial-conditions']['name'] == 'SlowShock': return SlowShock(params)
    elif parameters['initial-conditions']['name'] == 'PeakProblem': return PeakProblem(params)
    elif parameters['initial-conditions']['name'] == 'LeBlanc': return LeBlanc(params)
    elif parameters['initial-conditions']['name'] == 'StreamCollision': return StreamCollision(params)
    elif parameters['initial-conditions']['name'] == 'RCVCR': return RCVCR(params)
    elif parameters['initial-conditions']['name'] == 'VaccumExpansionRight': return VaccumExpansionRight(params)
    elif parameters['initial-conditions']['name'] == 'VaccumExpansionLeft': return VaccumExpansionLeft(params)
    elif parameters['initial-conditions']['name'] == 'ToroTest4': return ToroTest4(params)
    else:
        raise ValueError('unknown initial condition ' + parameters['initial-condition']['name'])
    return None

class IC():
    def set_thermo(self, thermo):
        self.thermo = thermo

class PlaneWave(IC):
    def __init__(self, parameters):
        self.parameters = parameters
        self.Lx = 1.0
        self.Ly = 1.0
        self.Lz = 1.0
        self.xc = 0.5
        self.xc = 0.5
        self.xc = 0.5

        self.c = sp.constants.c
        self.epsilon0 = sp.constants.epsilon_0
        self.k = 2. * pi
        self.w = self.k * self.c
        self.alpha_x = 0.0
        self.alpha_y = 0.0
        self.E0x = 1000.0
        self.E0y = 1000.0

        self.const_state = {}

    def get_value(self, mesh, t):
        xs = SpatialCoordinate(mesh)
        initcond = {}
        Ex = self.E0x * cos(self.k*xs[2] - self.w*t + self.alpha_x)
        Ey = self.E0y * cos(self.k*xs[2] - self.w*t + self.alpha_y)
        Ez = 0.
        initcond['D'] = as_vector([Ex,Ey,Ez]) * self.epsilon0
        initcond['B'] = as_vector([-Ey,Ex,Ez]) / self.c
        return initcond

class IC1D(IC):

    def gaussian_tracer(self, h, x, t):
        s = 0.1 * (1. + 0.05 * exp(-((x[0]-self.xc)*(x[0]-self.xc))/(1./9*0.5*0.5*self.Lx*self.Lx)))
        return s * h

    def square_tracer(self, h, x, t):
        xscaled = (x[0] - self.xc + self.Lx/2.)/self.Lx
        return ufl.conditional(ufl.gt(xscaled,2./3.), 0.0, ufl.conditional(ufl.lt(xscaled,1./3.), 0.0, 0.1*h))

    def set_tracers(self, x, t, initcond):
        for i,name in enumerate(self.parameters['model']['tracer_names']):
            if self.parameters['initial-conditions']['tracer_init_conds'][i] == 'block':
                initcond[name] = self.square_tracer(initcond['h'], x, t)
            elif self.parameters['initial-conditions']['tracer_init_conds'][i] == 'gaussian':
                initcond[name] = self.gaussian_tracer(initcond['h'], x, t)
            else:
                raise ValueError('unknown tracer initial condition ' + self.parameters['initial-conditions']['tracer_init_conds'][i])

        for i,name in enumerate(self.parameters['model']['dg_tracer_names']):
            if self.parameters['initial-conditions']['dg_tracer_init_conds'][i] == 'block':
                initcond[name] = self.square_tracer(initcond['h'], x, t)
            elif self.parameters['initial-conditions']['dg_tracer_init_conds'][i] == 'gaussian':
                initcond[name] = self.gaussian_tracer(initcond['h'], x, t)
            else:
                raise ValueError('unknown dg tracer initial condition ' + self.parameters['initial-conditions']['dg_tracer_init_conds'][i])


class RiemannProblem(IC1D):
    def __init__(self, parameters):
        self.Lx = 1.0
        self.xc = 0.0
        self.parameters = parameters
        self.gamma = 1.4
        self.Cv = 1.0

    def get_value(self, mesh, t):
        xs = SpatialCoordinate(mesh)
        initcond = {}
        initcond['rho'] = ufl.conditional(ufl.le(xs[0], self.xc_discont), self.rhol, self.rhor)
        initcond['v'] = ufl.conditional(ufl.le(xs[0], self.xc_discont), self.ul, self.ur)
        p = ufl.conditional(ufl.lt(xs[0], self.xc_discont), self.pl, self.pr)
        eta = self.thermo.get_eta(initcond['rho'], p)
        initcond['m'] = ufl.conditional(ufl.le(xs[0], self.xc_discont), self.rhol * self.ul, self.rhor * self.ur)
        initcond['S'] = initcond['rho'] * eta
        return initcond

class Gaussian(IC1D):
    def __init__(self, parameters):
        self.xc = 0.5
        self.Lx = 1.0
        self.parameters = parameters
        self.g = 9.80616
        self.H0 = 750.0
        self.dh = 75.0
        self.sigmax = 3./40. * self.Lx
        self.xc1 = 0.5 * self.Lx
        self.c = 0.05
        self.a = 1./3.
        self.D = 0.5 * self.Lx

    def get_value(self, mesh, t):
        xs = SpatialCoordinate(mesh)
        initcond = {}
        xprime1 = self.Lx / (pi * self.sigmax) * sin(pi / self.Lx * (xs[0] - self.xc1))
        initcond['h'] = self.H0 + self.dh * exp(-0.5 * (xprime1 * xprime1))
        initcond['v'] = Constant(0.0)
        initcond['m'] = Constant(0.0)
        s = self.g * (1. + self.c * exp(-((xs[0]-self.xc)*(xs[0]-self.xc))/(self.a*self.a*self.D*self.D)))
        initcond['S'] = s * initcond['h']
        self.set_tracers(xs, t, initcond)
        return initcond


#Sod Shock Tube
class RiemannProblem1(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 1.0
        self.rhor = 0.125
        self.ur =  0.0
        self.pr = 0.1

#Toro Test 5
#DOES THIS HAVE ANOTHER NAME?
class RiemannProblem2(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0 #-0.2
        self.rhol = 5.99924
        self.ul = 19.5975
        self.pl = 460.894
        self.rhor = 5.99242
        self.ur =  -6.19633
        self.pr = 46.095

#Enfield123 ie VaccumExpansion
class RiemannProblem3(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = -2.0
        self.pl = 0.4
        self.rhor = 1.0
        self.ur =  2.0
        self.pr = 0.4

class ModifiedSod(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.75
        self.pl = 1.0
        self.rhor = 0.125
        self.ur =  0.0
        self.pr = 0.1


#T = 0.012
class ToroTest3(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 1000.0
        self.rhor = 1.0
        self.ur =  0.0
        self.pr = 0.1


#T = 0.035
class ToroTest4(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 0.01
        self.rhor = 1.0
        self.ur =  0.0
        self.pr = 100.0


#T = 0.75
class VaccumExpansionLeft(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 0.0
        self.ul = 0.0
        self.pl = 0.0
        self.rhor = 1.0
        self.ur =  0.0
        self.pr = 1.0


#T = 0.75
class VaccumExpansionRight(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 1.0
        self.rhor = 0.0
        self.ur =  0.0
        self.pr = 0.0


#T = 0.75
class RCVCR(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = -4.0
        self.pl = 0.4
        self.rhor = 1.0
        self.ur =  4.0
        self.pr = 0.4



#T = 0.8
class StreamCollision(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 2.0
        self.pl = 0.1
        self.rhor = 1.0
        self.ur =  -2.0
        self.pr = 0.1


#T = 0.5
class LeBlanc(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = (2. / 3.)*1.e-1
        self.rhor = 1.e-3
        self.ur =  0.0
        self.pr = (2. / 3.)*1.e-10


#T = 3.9e-3
class PeakProblem(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 0.1261192
        self.ul = 8.9047029
        self.pl = 782.92899
        self.rhor = 6.591493
        self.ur =  2.2654207
        self.pr = 3.1544874


#T = 2.
class SlowShock(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 3.857143
        self.ul = -0.810631
        self.pl = 10.33333
        self.rhor = 1.0
        self.ur =  -3.44
        self.pr = 1.0

#T = 0.012
class StationaryContact(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = -19.59745
        self.pl = 1.e3
        self.rhor = 1.0
        self.ur = 19.59745
        self.pr = 1.e-2


#DENSITY WAVE
#LOTS OF OTHER 1D ONES!!!




class IC2D(IC):
    def __init__(self, parameters):
        self.parameters = parameters

    def gaussian_tracer(self, h, xs, t):
        s = 0.1 * (1. + 0.05 * exp(-((xs[0]-self.xc)*(xs[0]-self.xc) + (xs[1]-self.yc)*(xs[1]-self.yc))/(1./9.*0.5*0.5*self.Lx*self.Ly)))
        return s * h

    def square_tracer(self, h, xs, t):
        xscaled = (xs[0] - self.xc + self.Lx/2.)/self.Lx
        yscaled = (xs[1] - self.yc + self.Ly/2.)/self.Ly
        return ufl.conditional(ufl.gt(xscaled,2./3.), 0.0, ufl.conditional(ufl.lt(xscaled,1./3.), 0.0, ufl.conditional(ufl.gt(yscaled,2./3.), 0.0, ufl.conditional(ufl.lt(yscaled,1./3.), 0.0, 0.1*h))))

    def set_tracers(self, x, t, initcond):
        for i,name in enumerate(self.parameters['model']['tracer_names']):
            if self.parameters['initial-conditions']['tracer_init_conds'][i] == 'block':
                initcond[name] = self.square_tracer(initcond['h'], x, t)
            elif self.parameters['initial-conditions']['tracer_init_conds'][i] == 'gaussian':
                initcond[name] = self.gaussian_tracer(initcond['h'], x, t)
            else:
                raise ValueError('unknown tracer initial condition ' + self.parameters['initial-conditions']['tracer_init_conds'][i])

        for i,name in enumerate(self.parameters['model']['dg_tracer_names']):
            if self.parameters['initial-conditions']['dg_tracer_init_conds'][i] == 'block':
                initcond[name] = self.square_tracer(initcond['h'], x, t)
            elif self.parameters['initial-conditions']['dg_tracer_init_conds'][i] == 'gaussian':
                initcond[name] = self.gaussian_tracer(initcond['h'], x, t)
            else:
                raise ValueError('unknown dg tracer initial condition ' + self.parameters['initial-conditions']['dg_tracer_init_conds'][i])


class DensityWave(IC2D):
    def __init__(self, params):
        IC2D.__init__(self, params)

        self.coriolis = 5.0
        self.g = 5.0
        self.bottom_topography = 0.0

        self.Lx = 1.0
        self.Ly = 1.0
        self.xc = 0.5 * self.Lx
        self.yc = 0.5 * self.Ly

        self.const_state = {}
        self.const_state['h'] = 1.0
        self.const_state['S'] = self.g
        self.const_state['u'] = 0.

    def get_value(self, mesh, t):
        xs = SpatialCoordinate(mesh)

        initcond = {}
        initcond['h'] = 1. + 1/(4.*pi) * self.coriolis / self.g * sin(4. * pi * xs[1])
        u = 0.0
        v = sin(2. * pi* xs[0])
        initcond['v'] = as_vector([u,v])
        initcond['m'] = initcond['h'] * initcond['v']
        initcond['S'] = initcond['h'] * self.g
        initcond['coriolis'] = self.coriolis
        initcond['bottom_topography'] = self.bottom_topography
        return initcond

class DoubleVortex(IC2D):
    def __init__(self, params):
        IC2D.__init__(self, params)

        self.Lx = 5000. * 1000.
        self.Ly = 5000. * 1000.
        self.xc = 0.5 * self.Lx
        self.yc = 0.5 * self.Ly
        self.g = 9.80616
        self.H0 =  750.0
        self.ox = 0.1 #0.1
        self.oy = 0.1 #0.1
        self.sigmax = 3./40.*self.Lx #3.
        self.sigmay = 3./40.*self.Lx #3.
        self.dh = 75.0
        self.xc1 = (0.5-self.ox) * self.Lx
        self.yc1 = (0.5-self.oy) * self.Ly
        self.xc2 = (0.5+self.ox) * self.Lx
        self.yc2 = (0.5+self.oy) * self.Ly
        self.U0 = 0. #10. #10.
        self.c = 0.05
        self.a = 1./3.
        self.D = 0.5 * self.Lx

        self.zeta = 0.0 #10^-3, 0.02
        self.q0 = 0.002

        self.const_state = {}
        self.const_state['h'] = self.H0
        self.const_state['S'] = self.g * self.H0
        self.const_state['u'] = 0.
        self.const_state['m'] = 0.

    def get_value(self, mesh, t):
        xs = SpatialCoordinate(mesh)
        xprime1 = self.Lx / (pi * self.sigmax) * sin(pi / self.Lx * (xs[0] - self.xc1))
        yprime1 = self.Ly / (pi * self.sigmay) * sin(pi / self.Ly * (xs[1] - self.yc1))
        xprime2 = self.Lx / (pi * self.sigmax) * sin(pi / self.Lx * (xs[0] - self.xc2))
        yprime2 = self.Ly / (pi * self.sigmay) * sin(pi / self.Ly * (xs[1] - self.yc2))
        xprimeprime1 = self.Lx / (2.0 * pi * self.sigmax) * sin(2 * pi / self.Lx * (xs[0] - self.xc1))
        yprimeprime1 = self.Ly / (2.0 * pi * self.sigmay) * sin(2 * pi / self.Ly * (xs[1] - self.yc1))
        xprimeprime2 = self.Lx / (2.0 * pi * self.sigmax) * sin(2 * pi / self.Lx * (xs[0] - self.xc2))
        yprimeprime2 = self.Ly / (2.0 * pi * self.sigmay) * sin(2 * pi / self.Ly * (xs[1] - self.yc2))
        coriolis = 0.00006147
        initcond = {}
        initcond['h'] = self.H0 - self.dh * (exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) + exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)) - 4. * pi * self.sigmax * self.sigmay / self.Lx / self.Ly)
        u = - self.g * self.dh / coriolis / self.sigmay * (yprimeprime1 * exp(-0.5*(xprime1 * xprime1 + yprime1 * yprime1)) + yprimeprime2 * exp(-0.5*(xprime2 * xprime2 + yprime2 * yprime2)))
        v = self.g * self.dh / coriolis / self.sigmax * (xprimeprime1 * exp(-0.5*(xprime1 * xprime1 + yprime1 * yprime1)) + xprimeprime2 * exp(-0.5*(xprime2 * xprime2 + yprime2 * yprime2)))
        initcond['v'] = as_vector([u,v])
        s = self.g * (1. + self.c * exp(-((xs[0]-self.xc)*(xs[0]-self.xc) + (xs[1]-self.yc)*(xs[1]-self.yc))/(self.a*self.a*self.D*self.D)))
        initcond['m'] = initcond['h'] * initcond['v']
        initcond['S'] = initcond['h'] * s
        initcond['coriolis'] = coriolis
        initcond['bottom_topography'] = 0.0
        initcond['Qv'] = initcond['h'] * (1. - self.zeta) * qsat(initcond['h'], s, initcond['bottom_topography'], self.q0, self.H0, self.g)
        initcond['Qr'] = 0.0
        initcond['Qc'] = 0.0
        self.set_tracers(xs, t, initcond)
        return initcond


class TC2(IC2D):
    def __init__(self, params):
        IC2D.__init__(self, params)

        self.a = 6371120.0
        self.Lx = 2.* pi * self.a
        self.Ly = 2.* pi * self.a
        self.xc = 0.5 * self.Lx
        self.yc = 0.5 * self.Ly
        self.g = 9.80616
        self.u0 = 20.
        self.c = 0.05
        self.H0 = 5960.
        self.f = 0.00006147
        self.zeta = 0.0 #10^-3, 0.02
        self.q0 = 0.007

    def get_value(self, mesh, t, initcond=None):
        xs = SpatialCoordinate(mesh)
        if initcond is None:
            initcond = {}
            initcond['bottom_topography'] = 0.0
        initcond['h'] = self.H0 - self.a * self.f * self.u0 / self.g * sin(xs[1]/self.a) - initcond['bottom_topography']
        u = self.u0 * cos(xs[1] / self.a)
        v = 0.0
        initcond['v'] = as_vector([u,v])
        initcond['m'] = initcond['h'] * initcond['v']
        s = self.g * (1. + self.c * self.H0 * self.H0 / (initcond['h'] * initcond['h']))
        initcond['S'] = initcond['h'] * s
        initcond['coriolis'] = self.f
        initcond['Qv'] = initcond['h'] * (1. - self.zeta) * qsat(initcond['h'], s, initcond['bottom_topography'], self.q0, self.H0, self.g)
        initcond['Qr'] = 0.0
        initcond['Qc'] = 0.0
        self.set_tracers(xs, t, initcond)
        return initcond

class TC5(TC2):

    def __init__(self, params):
        TC2.__init__(self, params)
        self.h0 = 2000.
        self.R = pi/9. * self.a
        self.xm = self.Lx / 3. #3.*pi/2. * self.a
        self.ym = self.Ly * 2. /3. #pi/6. * self.a

    def get_value(self, mesh, t):
        xs = SpatialCoordinate(mesh)
        initcond = {}
        dist = sqrt((xs[0] - self.xm) * (xs[0] - self.xm) + (xs[1] - self.ym) * (xs[1] - self.ym))
        initcond['bottom_topography'] = self.h0 * (1. - 1./self.R * ufl.min_value(self.R, dist))
        initcond = TC2.get_value(self, mesh, t, initcond=initcond)
        return initcond

#EVENTUALLY ADD AND FIX THIS
class Galewsky(IC2D):
    def __init__(self, params):
        IC2D.__init__(self, params)
