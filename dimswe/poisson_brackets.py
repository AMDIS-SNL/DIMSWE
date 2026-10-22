from firedrake import derivative, TestFunction, inner, TrialFunction, curl, as_vector, Function, grad, div, dot, sign
from .ufl_helpers import skewgrad, curl2D, rot2D
from .transport_operators import SVLieDerivative, VVLieDerivative, CVLieDerivative
from .operators import BracketBase

class PoissonBracket(BracketBase):
    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return self.rhs(const_state, dfdx_linear_vars, xhats)

class LiePoisson_AdvectedQuantities_Bracket(PoissonBracket):
    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.vars = vars
        self.advected_quantity_names = vars.advected_quantity_names
        self.advected_quantity_dhdx_names = vars.dhdx_var_list[1:]
        self.advected_quantity_bundles = vars.advected_quantity_bundles
        self.advected_quantity_degrees = vars.advected_quantity_degrees
        self.inactive_advected_quantity_names = vars.inactive_advected_quantity_names
        self.inactive_advected_quantity_bundles = vars.inactive_advected_quantity_bundles
        self.inactive_advected_quantity_degrees = vars.inactive_advected_quantity_degrees
        self.dim = parameters['mesh']['dim']
        self.total_density_func = vars.get_total_density_expr

#THIS STUFF SHOULD REALLY BE CONFIGURABLE PER ADVECTED QUANTITY I THINK..
        self.alpha_s = parameters['spatial-discretization']['alpha_s']

        if not spaces is None:
            if self.dim == 2:
                self.coriolis = Function(spaces.CG)

            elif self.dim == 3:
                self.coriolis = Function(spaces.CGV) #PRETTY UNCLEAR ACTUALLY- MAYBE H(curl) or even H(div)?

            self.dx = spaces.dx
            self.dS = spaces.dS
            self.ds = spaces.ds
            self.n = spaces.n
            self.order = spaces.order

    def initialize(self, varexpr):
        if self.dim == 2 or self.dim == 3:
            self.coriolis.interpolate(varexpr['coriolis'])

    def rhs(self, xvars, t, coeff, xhats):

        m = xvars['m']
        u = xvars['u']
        mtest = xhats['m']
        total_dens = self.total_density_func(xvars)

        rhs_expr = CVLieDerivative(self.dim, self.dim, u, m, mtest, 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)

#WHY IS SV LIE DERIVATIVE FAILING?
#ACTUALLY, DOES IT FAIL?
        #
        # alpha = self.alpha_s * sign(dot(u('+'),self.n('+')))
        # for densname, dhdx_name in zip(self.advected_quantity_names, self.advected_quantity_dhdx_names):
        #     dens = xvars[densname]
        #     Bdens = xvars[dhdx_name]
        #     denstest = xhats[densname]
        #     denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
        #     rhs_expr = rhs_expr + (denstest('+')*inner(u('+'), self.n('+')) + denstest('-')*inner(u('-'), self.n('-')))*denstilde*self.dS
        #     rhs_expr = rhs_expr - (Bdens('+')*inner(mtest('+'), self.n('+')) + Bdens('-')*inner(mtest('-'), self.n('-')))*denstilde*self.dS
        #     if self.spaces.order >1:
        #         rhs_expr = rhs_expr + inner(grad(Bdens   ), dens * mtest)*self.dx
        #         rhs_expr = rhs_expr - inner(grad(denstest), dens * u   )*self.dx
        #
        # for dens in self.inactive_advected_quantity_names:
        #     dens = xvars[densname]
        #     denstest = xhats[densname]
        #     denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
        #     rhs_expr = rhs_expr + (denstest('+')*inner(u('+'), self.n('+')) + denstest('-')*inner(u('-'), self.n('-')))*denstilde*self.dS
        #     if self.spaces.order >1:
        #         rhs_expr = rhs_expr - inner(grad(denstest), dens * u   )*self.dx


        for name, dhdx_name, bundle, degree in zip(self.advected_quantity_names, self.advected_quantity_dhdx_names, self.advected_quantity_bundles, self.advected_quantity_degrees):
            if bundle == 'S':
                rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, u, xvars[name], xhats[name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
                rhs_expr = rhs_expr - SVLieDerivative(degree, self.dim, mtest, xvars[name], xvars[dhdx_name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'VV':
                rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, u, xvars[name], xhats[name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
                rhs_expr = rhs_expr - VVLieDerivative(degree, self.dim, mtest, xvars[name], xvars[dhdx_name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'CV':
                rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, u, xvars[name], xhats[name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
                rhs_expr = rhs_expr - CVLieDerivative(degree, self.dim, mtest, xvars[name], xvars[dhdx_name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)

        for name, bundle, degree in zip(self.inactive_advected_quantity_names, self.inactive_advected_quantity_bundles, self.inactive_advected_quantity_degrees):
            if bundle == 'S':
                rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, u, xvars[name], xhats[name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'VV':
                rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, u, xvars[name], xhats[name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'CV':
                rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, u, xvars[name], xhats[name], 1, 1, u, self.alpha_s, self.n, self.order, self.dx, self.dS)

        if self.dim == 2:
            rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*self.dx
        elif self.dim == 3:
            raise NotImplementedError()
        return rhs_expr

class CurlForm_AdvectedQuantities_Bracket(PoissonBracket):
    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.vars = vars
        self.advected_quantity_names = vars.advected_quantity_names
        self.advected_quantity_dhdx_names = vars.dhdx_var_list[1:]
        self.advected_quantity_bundles = vars.advected_quantity_bundles
        self.advected_quantity_degrees = vars.advected_quantity_degrees
        self.inactive_advected_quantity_names = vars.inactive_advected_quantity_names
        self.inactive_advected_quantity_bundles = vars.inactive_advected_quantity_bundles
        self.inactive_advected_quantity_degrees = vars.inactive_advected_quantity_degrees
        self.dim = parameters['mesh']['dim']
        self.total_density_func = vars.get_total_density_expr

        self.upwind_v = parameters['spatial-discretization']['upwind_v']
        if not self.upwind_v:
            raise NotImplementedError('Auxiliary variables with non-constant lhs (ex. q in curl-form bracet) are not properly implemented in timestepping')
#THIS STUFF SHOULD REALLY BE CONFIGURABLE PER ADVECTED QUANTITY I THINK..
        self.alpha_s = parameters['spatial-discretization']['alpha_s']

        if not spaces is None:
            if self.dim == 2:
                self.coriolis = Function(spaces.CG)
                self.q_space = self.spaces.CG

            elif self.dim == 3:
                self.coriolis = Function(spaces.CGV) #PRETTY UNCLEAR ACTUALLY- MAYBE H(curl) or even H(div)?
                self.q_space = self.spaces.CGV

            self.dx = spaces.dx
            self.dS = spaces.dS
            self.ds = spaces.ds
            self.n = spaces.n
            self.order = spaces.order


    def initialize(self, varexpr):
        if self.dim == 2 or self.dim == 3:
            self.coriolis.interpolate(varexpr['coriolis'])

    def get_aux_vars_list(self):
        if self.dim >= 2 and not self.upwind_v:
            return ['q',]
        else:
#WHAT SHOULD ACTUALLY BE DONE HERE?
            return []

    def get_spacelist(self):
        if self.dim >= 2 and not self.upwind_v:
            return [self.q_space,]
        else:
#WHAT SHOULD ACTUALLY BE DONE HERE?
            return []

    def compute_q_expressions(self, xvars, t, coeff, xhats, expressions):
        if not self.upwind_v:
            v = xvars['v']
            q = xvars['q']
            total_dens = self.total_density_func(xvars)
            qhat = xhats['q']
    #MISSING BOUNDARY TERMS...
            if self.dim == 2:
                expressions['q'] = [inner(qhat, total_dens * q)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            elif self.dim == 3:
                expressions['q'] = [inner(qhat, total_dens * q)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]

    def rhs(self, xvars, t, coeff, xhats):
        v = xvars['v']
        F = xvars['F']
        vtest = xhats['v']
        total_dens = self.total_density_func(xvars)

#MISSING BOUNDARY TERMS- ds
        rhs_expr = 0.0
        if self.dim == 2:
            if self.upwind_v:
                alpha = self.alpha_s * sign(dot(v('+'),self.n('+')))
#PROBABLY WRAP THIS UP IN ONE OF THE LIE DERIVATIVE CLASSES! OR SOME SORT OF INTERIOR PRODUCT...
#TO DO THIS "RIGHT" SHOULD ACTUALLY COMPUTE FLOW VELOCITY U I THINK IE FOLLOWING GOLO PAPER!
                nperp = rot2D(self.n)
                vtilde = 0.5 * ((1. + alpha) * v('+') + (1. - alpha)*v('-'))
                Fperp = rot2D(F)
                rhs_expr = rhs_expr + inner(vtest, self.coriolis / total_dens * Fperp)*self.dx
                Fpart = inner(vtest,Fperp)/total_dens
                rhs_expr = rhs_expr - inner(skewgrad(Fpart), v)*self.dx
                jump_term = Fpart('+')*nperp('+') + Fpart('-')*nperp('-')
                rhs_expr = rhs_expr + inner(jump_term,vtilde)*self.dS
            else:
                q = xvars['q']
                rhs_expr = rhs_expr + inner(vtest, q * rot2D(F))*self.dx #q has coriolis in it!
        elif self.dim == 3:
            raise NotImplementedError
        #print(F)
        #print(Fperp)
        #rhs_expr = inner(vtest, self.coriolis / total_dens * Fperp)*self.dx

#WHY IS SV LIE DERIVATIVE FAILING?
#ONLY FOR ORDER >1, SO SOMETHING IS WEIRD WITH THE GRADIENT STUFF...
        total_dens_avg = 0.5 * (total_dens('+') + total_dens('-'))


        # alpha = self.alpha_s * sign(dot(v('+'),self.n('+')))
        # for densname, dhdx_name in zip(self.advected_quantity_names, self.advected_quantity_dhdx_names):
        #     dens = xvars[densname]
        #     Bdens = xvars[dhdx_name]
        #     denstest = xhats[densname]
        #     denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
        #     rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), self.n('+')) + denstest('-')*inner(F('-'), self.n('-')))*denstilde/total_dens_avg*self.dS
        #     rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), self.n('+')) + Bdens('-')*inner(vtest('-'), self.n('-')))*denstilde/total_dens_avg*self.dS
        #     if self.spaces.order >1:
        #         rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*self.dx
        #         rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx
        #

        # for dens in self.inactive_advected_quantity_names:
        #     dens = xvars[densname]
        #     denstest = xhats[densname]
        #     denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
        #     rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), self.n('+')) + denstest('-')*inner(F('-'), self.n('-')))*denstilde/total_dens_avg*self.dS
        #     if self.spaces.order >1:
        #         rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx
        #

        for name, dhdx_name, bundle, degree in zip(self.advected_quantity_names, self.advected_quantity_dhdx_names, self.advected_quantity_bundles, self.advected_quantity_degrees):
            if bundle == 'S':
                rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
                rhs_expr = rhs_expr - SVLieDerivative(degree, self.dim, vtest, xvars[name], xvars[dhdx_name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'VV':
                rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
                rhs_expr = rhs_expr - VVLieDerivative(degree, self.dim, vtest, xvars[name], xvars[dhdx_name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'CV':
                rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
                rhs_expr = rhs_expr - CVLieDerivative(degree, self.dim, vtest, xvars[name], xvars[dhdx_name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)

        for name, bundle, degree in zip(self.inactive_advected_quantity_names, self.inactive_advected_quantity_bundles, self.inactive_advected_quantity_degrees):
            if bundle == 'S':
                rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'VV':
                rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
            elif bundle == 'CV':
                rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, total_dens, v, self.alpha_s, self.n, self.order, self.dx, self.dS)
        return rhs_expr



#RESTORE THESE EVENTUALLY

# #THIS IS MISSING BOUNDARY CONDITION STUFF!
# class MaxwellBracket(PoissonBracket):
#     def __init__(self, spaces):
#         self.spaces = spaces
#         if not self.spaces is None:
#             self.dx = spaces.dx
#
# #THIS IS MISSING BOUNDARY CONDITION STUFF!
#     def rhs(self, qvars, dfdx_vars, xhats):
#         E, H = dfdx_vars['E'], dfdx_vars['H']
#         Dhat, Bhat = xhats['D'], xhats['B']
#         rhs_expr = inner(Bhat, curl(E))*self.dx - inner(curl(Dhat), H)*self.dx
#         return rhs_expr
#
#     def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
#         return self.rhs(None, dfdx_linear_vars, xhats)
#
# class EulerMaxwellCouplingBracket_LP(PoissonBracket):
#     def __init__(self, spaces):
#         self.spaces = spaces
#
#
# class ScalarWaveBracket(PoissonBracket):
#     def __init__(self, spaces):
#         self.spaces = spaces
