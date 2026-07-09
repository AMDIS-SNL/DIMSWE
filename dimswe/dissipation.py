
from math import pow
from .operators import ForcingBase
from firedrake import Function, inner, grad, TestFunction, TrialFunction, FunctionSpace, Constant

#THIS IS PRETTY SPECIFIC TO CG METHODS
#ALSO 2ND ORDER HYPERVISCOSITY ONLY
class Hyperviscosity(ForcingBase):
    def __init__(self, parameters, vars, spaces):
        self.varlist = ['v',] + vars.active_density_names

        self.spaces = spaces
        self.c0 = parameters['hyperviscosity']['c0']
        self.s = parameters['hyperviscosity']['s']

        self.name = 'hyperviscosity'
        self.mu_space = None

        if not spaces is None:
#IDEALLY HERE WE USE TENSOR HV- THEN WE CAN AVOID THE VERY HACKY SPACES.DX STUFF
#CAN PROBABLY TIE TO CFL CONDITION STUFF ALSO
            self.dx = spaces.dx
            self.mu_space = FunctionSpace(self.spaces.mesh, 'CG', self.spaces.order, variant="spectral")
            self.spacelist = vars.get_spacelist()[:len(self.varlist)]

    def has_coeff(self):
        return True

    def set_default_coeffs(self, coeff):
        coeff['mu'].assign(self.c0 * pow(max(self.spaces.mesh.dx/self.spaces.order, self.spaces.mesh.dy/self.spaces.order), self.s))

    def get_coeff(self):
        return ['mu', self.mu_space]

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
            #expressions['Q_' + varname] = inner(varhat, xvars['Q_' + varname])*self.dx + inner(grad(varhat), grad(var))*self.dx

    def rhs(self, xvars, t, coeff, xhats):
        expr = 0
        for varname in self.varlist:
            qvar = xvars['Q_' + varname]
            varhat = xhats[varname]
            expr = expr + inner(-coeff['mu'] * grad(varhat), grad(qvar))*self.dx
        return expr

#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
