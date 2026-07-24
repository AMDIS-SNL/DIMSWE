
from math import pow
from .operators import ForcingBase
from firedrake import inner, grad, TestFunction, TrialFunction, FunctionSpace, Constant, VectorFunctionSpace, as_vector
import numpy as np

#THIS IS PRETTY SPECIFIC TO CG METHODS
#ALSO 2ND ORDER HYPERVISCOSITY ONLY
class Hyperviscosity(ForcingBase):
    def __init__(self, parameters, vars, spaces):
        self.varlist = ['v',] + vars.active_density_names

        self.spaces = spaces

        self.name = 'hyperviscosity'
        #self.mu_space = None
        self.c0_space = None
        self.s_space = None
        self.treat_as_coeffs = parameters['hyperviscosity']['treat_as_coeffs']
        if not self.treat_as_coeffs:
            self.c0 = parameters['hyperviscosity']['c0']
            self.s = parameters['hyperviscosity']['s']

        if not spaces is None:
#IDEALLY HERE WE USE TENSOR HV- THEN WE CAN AVOID THE VERY HACKY SPACES.DX STUFF
#CAN PROBABLY TIE TO CFL CONDITION STUFF ALSO
            self.dx = spaces.dx
            #self.param_space = VectorFunctionSpace(self.spaces.mesh, 'R', 0, dim=2)
            if self.treat_as_coeffs:
                self.c0_space = FunctionSpace(self.spaces.mesh, 'R', 0)
                self.s_space = FunctionSpace(self.spaces.mesh, 'R', 0)
            #self.mu_space = FunctionSpace(self.spaces.mesh, 'CG', self.spaces.order, variant="spectral")
            self.spacelist = vars.get_spacelist()[:len(self.varlist)]
            self.factor = float(max(self.spaces.mesh.dx/self.spaces.order, self.spaces.mesh.dy/self.spaces.order))

    def has_coeff(self):
        return self.treat_as_coeffs

    def get_coeff_scaling_factors(self):
        return np.array([3.2, 0.07])

    def set_coeffs(self, parameters, coeff):
        c0 = parameters['c0']
        s = parameters['s']
        #coeff['mu'].assign(c0 * pow(max(self.spaces.mesh.dx/self.spaces.order, self.spaces.mesh.dy/self.spaces.order), s))
        coeff['c0'].assign(c0)
        coeff['s'].assign(s)
        #coeff['param'].assign(as_vector([c0,s]))

    def get_coeff(self):
        #return [['mu', self.mu_space],]
        return [['s', self.s_space], ['c0', self.c0_space]]
        #return [['param', self.param_space],]

    def get_coeff_bounds(self):
        return np.array([2., 0.01]), np.array([4.,2.])
        #return np.array([1e-6, 1e-6]), np.array([np.inf, np.inf])

        #return sp.optimize.Bounds(lb=,rb=,keep_feasible=True)

    def get_spacelist(self):
        return self.spacelist

    def get_aux_vars_list(self):
        auxlist = []
        for varname in self.varlist:
            auxlist.append('Q_' + varname)
        return auxlist

    def compute_aux_expressions(self, xvars, t, coeff, xhats, expressions):
        for varname, varspace in zip(self.varlist, self.spacelist):
            varhat = xhats['Q_' + varname]
            var = xvars[varname]
            expressions['Q_' + varname] = [inner(varhat, xvars['Q_' + varname])*self.dx, -inner(grad(varhat), grad(var))*self.dx]

    def rhs(self, xvars, t, coeff, xhats):
        expr = 0
        for varname in self.varlist:
            qvar = xvars['Q_' + varname]
            varhat = xhats[varname]
            #expr = expr + inner(-coeff['mu'] * grad(varhat), grad(qvar))*self.dx
            #print(coeff['param'])
            if self.treat_as_coeffs:
                expr = expr + inner(-coeff['c0'] * self.factor**coeff['s'] * grad(varhat), grad(qvar))*self.dx
            else:
                expr = expr + inner(-self.c0 * self.factor**self.s * grad(varhat), grad(qvar))*self.dx
            #expr = expr + inner(-coeff['param'][0] * self.factor**coeff['param'][1] * grad(varhat), grad(qvar))*self.dx
        return expr

#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
