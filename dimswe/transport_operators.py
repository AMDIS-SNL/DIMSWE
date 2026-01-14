from firedrake import (
    inner,
    dx,
    dS,
    grad,
    dot,
    sign,
)
from firedrake import VertexBasedLimiter


def SVLieDerivative(degree, dim, u, a, ahat, alpha_s, n, order):
    # 0-forms
    if degree == 0:
        return
    # volume forms
    elif degree == dim:
        alpha = alpha_s * sign(dot(u("+"), n("+")))
        # MISSING BOUNDARY TERMS- ds
        atilde = 0.5 * ((1.0 + alpha) * a("+") + (1.0 - alpha) * a("-"))
        expr = (
            (ahat("+") * inner(u("+"), n("+")) + ahat("-") * inner(u("-"), n("-")))
            * atilde
            * dS
        )
        if order > 1:
            rhs_expr = rhs_expr - inner(grad(ahat), a * u) * dx
        return expr
    # PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    # 1-forms in 2D
    elif degree == 1 and dim == 2:
        return
    # 1-forms in 3D
    elif degree == 1 and dim == 3:
        return
    # 2-forms in 3D
    elif degree == 2 and dim == 3:
        return


# MISSING LOTS OF THESE EXPRESSIONS...
def VVLieDerivative(degree, dim, u, a, ahat):
    # 0-forms
    if degree == 0:
        return
    # volume forms
    elif degree == dim:
        return
    # PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    # 1-forms in 2D
    elif degree == 1 and dim == 2:
        return
    # 1-forms in 3D
    elif degree == 1 and dim == 3:
        return
    # 2-forms in 3D
    elif degree == 2 and dim == 3:
        return


# MISSING LOTS OF THESE EXPRESSIONS...
def CVLieDerivative(degree, dim, u, a, ahat):
    # 0-forms
    if degree == 0:
        return
    # volume forms
    elif degree == dim:
        return
    # PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    # 1-forms in 2D
    elif degree == 1 and dim == 2:
        return
    # 1-forms in 3D
    elif degree == 1 and dim == 3:
        return
    # 2-forms in 3D
    elif degree == 2 and dim == 3:
        return


# EVENTUALLY ADD SOME TENSOR-VALUED BUNDLES ALSO? Unclear...
class DG1LimiterTransport:
    def __init__(self, parameters, vars, spaces):
        self.vars = vars
        self.spaces = spaces
        self.limiter = VertexBasedLimiter(DG1SPACE)

    def initialize(self, varexpr):
        pass

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def post_step(self, state):
        for varname in self.vars.dg_density_names:
            field = state[varname]  # WRONG
            self.limiter.apply(field)

    # THIS IS PRETTY SPECIFIC TO CG METHODS
    # ALSO 2ND ORDER ONLY
    def rhs(self, qvars, xhats):

        n = self.spaces.n
        alpha = self.alpha_s * sign(dot(v("+"), n("+")))
        total_dens_avg = 0.5 * (total_dens("+") + total_dens("-"))
        for dens_name in self.dg_density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_vars["B_" + dens_name]
            dens = qvars[dens_name]
            # MISSING BOUNDARY TERMS- ds
            denstilde = 0.5 * ((1.0 + alpha) * dens("+") + (1.0 - alpha) * dens("-"))
            rhs_expr = (
                rhs_expr
                + (
                    denstest("+") * inner(F("+"), n("+"))
                    + denstest("-") * inner(F("-"), n("-"))
                )
                * denstilde
                / total_dens_avg
                * dS
            )
            rhs_expr = (
                rhs_expr
                - (
                    Bdens("+") * inner(vtest("+"), n("+"))
                    + Bdens("-") * inner(vtest("-"), n("-"))
                )
                * denstilde
                / total_dens_avg
                * dS
            )
            if self.spaces.order > 1:
                rhs_expr = rhs_expr + inner(grad(Bdens), dens / total_dens * vtest) * dx
                rhs_expr = rhs_expr - inner(grad(denstest), dens / total_dens * F) * dx
        return rhs_expr

    def linear_rhs(self, const_state, xstar, xhats):
        return self.rhs(xstar, xhats)
