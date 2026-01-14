from math import pow
from firedrake import (
    Function,
    TestFunction,
    TrialFunction,
    inner,
    grad,
    dx,
)


class Hyperviscosity:
    def __init__(self, parameters, vars, spaces):
        self.vars = vars
        self.spaces = spaces
        self.eps = parameters["c0"]
        self.s = parameters["s"]
        self.coeff = pow(self.eps, self.s)

    def initialize(self, varexpr):
        pass

    def get_aux_vars(self, vars):
        for varname, varspace in zip(
            self.vars.varlist, self.vars.spacelist, strict=False
        ):
            vars["q_" + varname] = Function(varspace)

    def get_aux_vars_list(self):
        auxlist = []
        for varname in self.var.varlist:
            auxlist.append("q_" + varname)
        return auxlist

    def compute_q_expressions(self, vars, expressions):
        for varname, varspace in zip(
            self.vars.varlist, self.vars.spacelist, strict=False
        ):
            varhat = TestFunction(varspace)
            vartrial = TrialFunction(varspace)
            var = vars["q_" + varname]
            # SIGNS? COEFF?
            expressions["q_" + varname] = [
                inner(varhat, vartrial) * dx,
                inner(grad(varhat), grad(var)) * dx,
            ]

    # THIS IS PRETTY SPECIFIC TO CG METHODS
    # ALSO 2ND ORDER ONLY
    def rhs(self, qvars, xhats):
        expr = 0
        for varname in self.var.varlist:
            qvar = qvars["q_" + varname]
            varhat = xhats[varname]
            # SIGNS?
            expr = expr + inner(self.coeff * grad(varhat), grad(qvar)) * dx
        return expr

    def linear_rhs(self, const_state, xstar, xhats):
        return self.rhs(xstar, xhats)
