from firedrake import inner, sign, dot, grad, div
from firedrake import VertexBasedLimiter, FunctionSpace
from .operators import ForcingBase


#             denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
#             rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), n('+')) + denstest('-')*inner(F('-'), n('-')))*denstilde/total_dens_avg*self.dS
#             rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), n('+')) + Bdens('-')*inner(vtest('-'), n('-')))*denstilde/total_dens_avg*self.dS
#             if self.spaces.order >1:
#                 rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*self.dx
#                 rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx

def SVLieDerivative(degree, dim, u, a, ahat, afac, alpha_s, n, order, dx, dS):
    #0-forms
    if degree == 0:
        raise NotImplementedError
    #volume forms
    elif degree == dim:
        alpha = alpha_s * sign(dot(u('+'),n('+')))
#MISSING BOUNDARY TERMS- ds
        atilde = 0.5 * ((1. + alpha) * a('+') + (1. - alpha)*a('-'))
        expr = (ahat('+')*inner(u('+'), n('+')) + ahat('-')*inner(u('-'), n('-')))*atilde/afac*dS
        if order > 1:
            rhs_expr = rhs_expr - inner(grad(ahat), a/ afac * u)*dx
        return expr
#PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    #1-forms in 2D
    elif degree == 1 and dim == 2:
        raise NotImplementedError
    #1-forms in 3D
    elif degree == 1 and dim == 3:
        raise NotImplementedError
    #2-forms in 3D
    elif degree == 2 and dim == 3:
        raise NotImplementedError

#MISSING LOTS OF THESE EXPRESSIONS...
def VVLieDerivative(degree, dim, u, a, ahat, dx):
    #0-forms
    if degree == 0:
        raise NotImplementedError
    #volume forms
    elif degree == dim:
        raise NotImplementedError
#PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    #1-forms in 2D
    elif degree == 1 and dim == 2:
        raise NotImplementedError
    #1-forms in 3D
    elif degree == 1 and dim == 3:
        raise NotImplementedError
    #2-forms in 3D
    elif degree == 2 and dim == 3:
        raise NotImplementedError

#MISSING LOTS OF THESE EXPRESSIONS...
def CVLieDerivative(degree, dim, u, a, ahat, dx):
    #0-forms
    if degree == 0:
        raise NotImplementedError
    #volume forms
    elif degree == dim:
        raise NotImplementedError
#PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    #1-forms in 2D
    elif degree == 1 and dim == 2:
        raise NotImplementedError
    #1-forms in 3D
    elif degree == 1 and dim == 3:
        raise NotImplementedError
    #2-forms in 3D
    elif degree == 2 and dim == 3:
        raise NotImplementedError
#
# #EVENTUALLY ADD SOME TENSOR-VALUED BUNDLES ALSO? Unclear...
#
# #ADD THIS EVENTUALLY
# class LieDerivTransport(ForcingBase):
#     def __init__(self, parameters, vars, spaces):
#         self.vars = vars
#         self.spaces = spaces
# #THIS NEEDS TO BE MODIFIED- SHOULD REALLY JUST BE SOME SET OF TRACERS
#         self.inactive_advected_vars = vars.inactive_advected_vars
#         self.name = 'lietransport'
#
#     def rhs(self, xvars, xhats):
#         return 0
#
#     def linear_rhs(self, const_state, xvars, xhats):
#         raise NotImplementedError
#
# class CGTransport(ForcingBase):
#     def __init__(self, parameters, vars, spaces):
#         self.vars = vars
#         self.spaces = spaces
# #THIS NEEDS TO BE MODIFIED- SHOULD REALLY JUST BE SOME SET OF TRACERS
#         self.density_names = vars.cg_inactive_density_names
#         self.name = 'cgtransport'
#         self.use_split_form = parameters['spatial-discretization']['use_split_form']
#
#         if not spaces is None:
#             self.dx = spaces.dx
#
#     def rhs(self, xvars, t, coeff, qvars, xhats):
#         v = xvars['v']
#         rhs_expr = 0
#         for dens_name in self.density_names:
#             denstest = xhats[dens_name]
#             dens = xvars[dens_name]
#             if self.use_split_form[dens_name]:
#                 rhs_expr = rhs_expr + inner(denstest, 0.5 *( div(dens*v) + dot(grad(dens),v) + dens * div(v)))*self.dx
#             else:
#                 rhs_expr = rhs_expr + inner(denstest, div(dens*v))*self.dx
#         return rhs_expr
#
# #THIS IS BROKEN!
#     def linear_rhs(self, const_state, xvars, xhats):
#         raise NotImplementedError



class DG1LimiterTransport(ForcingBase):
    def __init__(self, parameters, vars, spaces):
        self.vars = vars
        self.spaces = spaces

        self.dg_density_names = vars.dg_inactive_density_names
        self.alpha_s = parameters['spatial-discretization']['alpha_s']
        self.name = 'dg1limiter'

        if not spaces is None:
            self.limiter = VertexBasedLimiter(FunctionSpace(self.spaces.mesh, 'DG', 1, variant="spectral"))
            self.dx = spaces.dx
            self.dS = spaces.dS
            self.ds = spaces.ds

    def post_step(self, statevars):
        for varname in self.dg_density_names:
            field = statevars[varname]
            self.limiter.apply(field)


    def rhs(self, xvars, t, coeff, xhats):
        v = xvars['v']
        n = self.spaces.n

        rhs_expr = 0
        n = self.spaces.n
        alpha = self.alpha_s * sign(dot(v('+'),n('+')))
        for dens_name in self.dg_density_names:
            denstest = xhats[dens_name]
            dens = xvars[dens_name]
#MISSING BOUNDARY TERMS- ds
            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(v('+'), n('+')) + denstest('-')*inner(v('-'), n('-')))*denstilde*self.dS
            rhs_expr = rhs_expr - inner(grad(denstest), dens * v   )*self.dx
        return rhs_expr



#THIS IS BROKEN!
    def linear_rhs(self, const_state, xvars, xhats):
        raise NotImplementedError
