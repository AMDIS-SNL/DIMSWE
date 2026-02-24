
from math import pow
from .operators import ForcingBase
from firedrake import Function, inner, grad, TestFunction, TrialFunction, FunctionSpace

#THIS IS PRETTY SPECIFIC TO CG METHODS
#ALSO 2ND ORDER HYPERVISCOSITY ONLY
class Hyperviscosity(ForcingBase):
    def __init__(self, parameters, vars, spaces):
        self.varlist = ['v',] + vars.active_density_names

        self.spaces = spaces
        self.c0 = parameters['hyperviscosity']['c0']
        self.s = parameters['hyperviscosity']['s']

        self.name = 'hyperviscosity'

        if not spaces is None:
            self.dx = spaces.dx
            self.coeff = Function(FunctionSpace(self.spaces.mesh, 'CG', self.spaces.order, variant="spectral"))
            self.coeff.assign(self.c0 * pow(max(spaces.mesh.dx/spaces.order, spaces.mesh.dy/spaces.order), self.s))
            self.spacelist = vars.spacelist[:len(self.varlist)]
    def get_aux_vars(self, vars):
        for varname, varspace in zip(self.varlist, self.spacelist):
            vars['Q_' + varname] = Function(varspace)

    def get_aux_vars_list(self):
        auxlist = []
        for varname in self.varlist:
            auxlist.append('Q_' + varname)
        return auxlist

    def compute_q_expressions(self, vars, expressions):
        for varname, varspace in zip(self.varlist, self.spacelist):
            varhat = TestFunction(varspace)
            vartrial = TrialFunction(varspace)
            var = vars[varname]
            expressions['Q_' + varname] = [inner(varhat, vartrial)*self.dx, -inner(grad(varhat), grad(var))*self.dx]

    def rhs(self, qvars, xhats):
        expr = 0
        for varname in self.varlist:
            qvar = qvars['Q_' + varname]
            varhat = xhats[varname]
            expr = expr + inner(-self.coeff * grad(varhat), grad(qvar))*self.dx
        return expr

#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
