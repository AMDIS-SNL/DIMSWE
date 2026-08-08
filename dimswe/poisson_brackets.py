from firedrake import derivative, TestFunction, inner, TrialFunction, curl, as_vector, Function, grad, div, dot, sign
from .ufl_helpers import skewgrad, curl2D, rot2D
from .transport_operators import SVLieDerivative, VVLieDerivative, CVLieDerivative
from .operators import BracketBase

class PoissonBracket(BracketBase):
    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return self.rhs(const_state, dfdx_linear_vars, xhats)

# class LiePoisson_AdvectedQuantities_Bracket(PoissonBracket):
#     def __init__(self, spaces, vars, parameters):
#         self.spaces = spaces
#         self.advected_quantity_names = vars.advected_quantity_names
#         self.advected_quantity_bundle = vars.advected_quantity_bundle
#         self.advected_quantity_degree = vars.advected_quantity_degree
#         self.testvars = {}
#         self.trialvars = {}
#         self.alpha_s = parameters['alpha_s']
#         self.dim = parameters['dim']
#         self.total_density_func = vars.get_total_density_expr
#
#         if not spaces is None:
#             if self.dim == 2:
#                 self.coriolis = Function(spaces.CG)
#             elif self.dim == 3:
#                 self.coriolis = Function(spaces.CGV) #PRETTY UNCLEAR ACTUALLY- MAYBE H(curl) or even H(div)?
#             self.dx = spaces.dx
#             self.dS = spaces.dS
#             self.ds = spaces.ds
#
#     def initialize(self, varexpr):
#         if self.dim == 2 or self.dim == 3:
#             self.coriolis.interpolate(varexpr['coriolis'])
#
#     def rhs(self, qvars, dfdx_vars, xhats):
#         m = qvars['m']
#         u = dfdx_vars['u']
#         mtest = xhats['m']
#         total_dens = self.total_density_func(qvars)
#
#         rhs_expr = CVLieDerivative(3, self.dim, u, m, mtest)
#
#         for name, bundle, degree in zip(self.advected_quantity_names, self.advected_quantity_bundle, self.advected_quantity_degree):
#             if bundle == 'S':
#                 rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, u, qvars[name], xhats[name])
#                 rhs_expr = rhs_expr - SVLieDerivative(degree, self.dim, mtest, qvars[name], dfdx_vars[name])
#             elif bundle == 'VV':
#                 rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, u, qvars[name], xhats[name])
#                 rhs_expr = rhs_expr - VVLieDerivative(degree, self.dim, mtest, qvars[name], dfdx_vars[name])
#             elif bundle == 'CV':
#                 rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, u, qvars[name], xhats[name])
#                 rhs_expr = rhs_expr - CVLieDerivative(degree, self.dim, mtest, qvars[name], dfdx_vars[name])
#
#         if self.dim == 2:
#             rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*self.dx
#         elif self.dim == 3:
#             ERROR
#         return rhs_expr

class CurlForm_AdvectedQuantities_Bracket(PoissonBracket):
    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.vars = vars
        self.advected_quantity_names = vars.advected_quantity_names
        self.advected_quantity_dhdx_names = vars.dhdx_var_list[1:]
        self.advected_quantity_bundle = vars.advected_quantity_bundle
        self.advected_quantity_degree = vars.advected_quantity_degree
        self.inactive_advected_quantity_names = vars.inactive_advected_quantity_names
        self.inactive_advected_quantity_bundle = vars.inactive_advected_quantity_bundle
        self.inactive_advected_quantity_degree = vars.inactive_advected_quantity_degree
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

#             if self.dim == 2:
#                 self.testvars['q'] = TestFunction(self.spaces.CG)
#                 self.trialvars['q'] = TrialFunction(self.spaces.CG)
# ##THESE ARE ACTUALLY SPECIFIC TO HDIV VARIANT, NOT GENERAL ENOUGH FOR H1 in 3D.
# #probably not doing H1 in 3D, so maybe okay?
#             elif self.dim == 3:
#                 self.testvars['q'] = TestFunction(self.spaces.Hcurl)
#                 self.trialvars['q'] = TrialFunction(self.spaces.Hcurl)


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
                rhs_expr = rhs_expr + inner(vtest, self.coriolis / total_dens * rot2D(F))*self.dx
                vtilde = 0.5 * ((1. + alpha) * v('+') + (1. - alpha)*v('-'))
                Fperp = rot2D(F)
                rhs_expr = rhs_expr + inner(vtest, self.coriolis / total_dens * Fperp)*self.dx
                Fpart = inner(vtest,Fperp)/total_dens
                rhs_expr = rhs_expr - inner(skewgrad(Fpart), v)*self.dx
                jump_term = Fpart('+')*nperp('+') + Fpart('-')*nperp('-')
                rhs_expr = rhs_expr + inner(jump_term,vtilde)*self.dS
            else:
                q = xvars['q']
                rhs_expr = inner(vtest, q * rot2D(F))*self.dx #q has coriolis in it!
        elif self.dim == 3:
            raise NotImplementedError

#WHY IS SV LIE DERIVATIVE FAILING?
#MAYBE IT IS IN ALPHA CALCS- DEPENDING ON V VS F?

        total_dens_avg = 0.5 * (total_dens('+') + total_dens('-'))
        alpha = self.alpha_s * sign(dot(v('+'),self.n('+')))
        for densname, dhdx_name in zip(self.advected_quantity_names, self.advected_quantity_dhdx_names):
            dens = xvars[densname]
            Bdens = xvars[dhdx_name]
            denstest = xhats[densname]
            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), self.n('+')) + denstest('-')*inner(F('-'), self.n('-')))*denstilde/total_dens_avg*self.dS
            rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), self.n('+')) + Bdens('-')*inner(vtest('-'), self.n('-')))*denstilde/total_dens_avg*self.dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*self.dx
                rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx

        for dens in self.inactive_advected_quantity_names:
            dens = xvars[densname]
            denstest = xhats[densname]
            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), self.n('+')) + denstest('-')*inner(F('-'), self.n('-')))*denstilde/total_dens_avg*self.dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx


        # for name, dhdx_name, bundle, degree in zip(self.advected_quantity_names, self.advected_quantity_dhdx_names, self.advected_quantity_bundle, self.advected_quantity_degree):
        #     if bundle == 'S':
        #         rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #         rhs_expr = rhs_expr - SVLieDerivative(degree, self.dim, vtest, xvars[name], xvars[dhdx_name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #     elif bundle == 'VV':
        #         rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #         rhs_expr = rhs_expr - VVLieDerivative(degree, self.dim, vtest, xvars[name], xvars[dhdx_name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #     elif bundle == 'CV':
        #         rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #         rhs_expr = rhs_expr - CVLieDerivative(degree, self.dim, vtest, xvars[name], xvars[dhdx_name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #
        # for name, bundle, degree in zip(self.inactive_advected_quantity_names, self.inactive_advected_quantity_bundle, self.inactive_advected_quantity_degree):
        #     if bundle == 'S':
        #         rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #     elif bundle == 'VV':
        #         rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        #     elif bundle == 'CV':
        #         rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, F, xvars[name], xhats[name], total_dens_avg, self.alpha_s, self.n, self.order, self.dx, self.dS)
        return rhs_expr

#
#
# class LiePoisson_AdvectedDensities_Bracket(PoissonBracket):
#
#     def __init__(self, spaces, vars, parameters):
#         self.spaces = spaces
#         self.density_names = vars.density_names
#         self.testvars = {}
#         self.trialvars = {}
#         self.alpha_s = parameters['alpha_s']
#         self.dim = parameters['dim']
#
#         if not spaces is None:
#             self.coriolis = Function(spaces.CG)
#             self.dx = spaces.dx
#             self.dS = spaces.dS
#             self.ds = spaces.ds
#
#     def initialize(self, varexpr):
#         self.coriolis.interpolate(varexpr['coriolis'])
#
#     def rhs(self, qvars, dfdx_vars, xhats):
#         m = qvars['m']
#         u = dfdx_vars['u']
#         mtest = xhats['m']
#         n = self.spaces.n
#
#         alpha = self.alpha_s * sign(dot(u('+'),n('+')))
#
# #FIX THIS STUFF UP!
#         mtilde = 0.5 * ((1. + alpha) * m('+') + (1. - alpha)*m('-'))
#         rhs_expr = (mtest('+')*inner(u('+'), n('+')) + mtest('-')*inner(u('-'), n('-')))*mtilde*self.dS
#         rhs_expr = rhs_expr - (u('+')*inner(mtest('+'), n('+')) + u('-')*inner(mtest('-'), n('-')))*mtilde*self.dS
#
# #FIX THIS STUFF UP!
# #super unclear if this is the correct notation
# #probably need some sort of tensor product type thing for u*m, etc.
# #could write in terms of coordinates pretty easy, so maybe do that?
#         if self.spaces.order >1:
#             rhs_expr = rhs_expr - inner(grad(mtest), outer(u,m))*self.dx
#             rhs_expr = rhs_expr + inner(grad(u), outer(mtest,m))*self.dx
#
#         if self.dim == 1:
#             ERROR
#         elif self.dim == 2:
#             rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*self.dx
#         elif self.dim == 3:
#             ERROR
#
#         for dens_name in self.density_names:
#             denstest = xhats[dens_name]
#             Bdens = dfdx_vars['B_' + dens_name]
#             dens = qvars[dens_name]
#
# #MISSING BOUNDARY TERMS- ds
#             denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
#             rhs_expr = rhs_expr + (denstest('+')*inner(u('+'), n('+')) + denstest('-')*inner(u('-'), n('-')))*denstilde*self.dS
#             rhs_expr = rhs_expr - (Bdens('+')*inner(mtest('+'), n('+')) + Bdens('-')*inner(mtest('-'), n('-')))*denstilde*self.dS
#             if self.spaces.order >1:
#                 rhs_expr = rhs_expr + inner(grad(Bdens   ), dens * mtest)*self.dx
#                 rhs_expr = rhs_expr - inner(grad(denstest), dens * u   )*self.dx
#         return rhs_expr
#
# #THIS SHOULD BE CAPABLE OF USING THE SIMPLE LINEARIZED VERSION?
#     def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
#         m = const_state['m']
#         u = dfdx_linear_vars['u']
#         mtest = xhats['m']
#         n = self.spaces.n
#         total_dens = self.total_density_func(const_state)
#
#
#         rhs_expr = (mtest('+')*inner(u('+'), n('+')) + mtest('-')*inner(u('-'), n('-')))*m*self.dS
#         rhs_expr = rhs_expr - (u('+')*inner(mtest('+'), n('+')) + u('-')*inner(mtest('-'), n('-')))*m*self.dS
#
#         if self.dim == 1:
#             ERROR
#         elif self.dim == 2:
#             rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*self.dx
#         elif self.dim == 3:
#             ERROR
#
#         if self.spaces.order >1:
# #GRAD OR NABLA GRAD HERE?
#             rhs_expr = rhs_expr - inner(grad(mtest), outer(u,m))*self.dx
#             rhs_expr = rhs_expr + inner(grad(u), outer(mtest,m))*self.dx
#
#         for dens_name in self.density_names:
#             denstest = xhats[dens_name]
#             Bdens = dfdx_linear_vars['B_' + dens_name]
#             dens = const_state[dens_name]
#
# #MISSING BOUNDARY TERMS- ds
#             denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
#             rhs_expr = rhs_expr + (denstest('+')*inner(u('+'), n('+')) + denstest('-')*inner(u('-'), n('-')))*dens*self.dS
#             rhs_expr = rhs_expr - (Bdens('+')*inner(mtest('+'), n('+')) + Bdens('-')*inner(mtest('-'), n('-')))*dens*self.dS
#             if self.spaces.order >1:
#                 rhs_expr = rhs_expr + inner(grad(Bdens   ), dens * mtest)*self.dx
#                 rhs_expr = rhs_expr - inner(grad(denstest), dens * u   )*self.dx
#         return rhs_expr
#
# class CurlForm_AdvectedDensities_Bracket_Base(PoissonBracket):
#     def __init__(self, spaces, vars, parameters):
#         self.spaces = spaces
#         self.vars = vars
#         self.active_density_names = vars.active_density_names
#         self.dim = parameters['dim']
#         self.total_density_func = vars.get_total_density_expr
#         self.testvars = {}
#         self.trialvars = {}
#         self.alpha_s = parameters['alpha_s']
#         self.upwind_v = parameters['upwind_v']
#         #NOT CURRENTLY USED
#         self.upwind_total_dens = parameters['upwind_total_dens']
#         self.use_split_form = parameters['use_split_form']
#
#
#         if not spaces is None:
#             self.coriolis = Function(spaces.CG)
#             self.dx = spaces.dx
#             self.dS = spaces.dS
#             self.ds = spaces.ds
#
#         if self.spaces is not None and not self.upwind_v:
#             if self.dim == 2:
#                 self.testvars['q'] = TestFunction(self.spaces.CG)
#                 self.trialvars['q'] = TrialFunction(self.spaces.CG)
# ##THESE ARE ACTUALLY SPECIFIC TO HDIV VARIANT, NOT GENERAL ENOUGH FOR H1 in 3D.
# #probably not doing H1 in 3D, so maybe okay?
#             elif self.dim == 3:
#                 self.testvars['q'] = TestFunction(self.spaces.Hcurl)
#                 self.trialvars['q'] = TrialFunction(self.spaces.Hcurl)
#
#     def initialize(self, varexpr):
#         self.coriolis.interpolate(varexpr['coriolis'])
#
#
#     def get_aux_vars_list(self):
#         if self.dim >= 2 and not self.upwind_v:
#             return ['q',]
#         else:
#             return []
#
#
#     def compute_q_expressions(self, vars, expressions):
#         if not self.upwind_v:
#             v = vars['v']
#             total_dens = self.total_density_func(vars)
#             qhat = self.testvars['q']
#             qtrial = self.trialvars['q']
#     #MISSING BOUNDARY TERMS...
#             if self.dim == 2:
#                 expressions['q'] = [inner(qhat, total_dens * qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             elif self.dim == 3:
#                 expressions['q'] = [inner(qhat, total_dens * qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#
#
#
# #SWAP THESE TO USE LIE DERIVATIVE FUNCTIONS
# class CurlForm_AdvectedDensities_Bracket(CurlForm_AdvectedDensities_Bracket_Base):
#
#     def get_aux_vars(self, vars):
#         if self.dim == 2 and not self.upwind_v:
#             vars['q'] = Function(self.spaces.CG, name='q')
#         elif self.dim == 3 and not self.upwind_v:
#             vars['q'] = Function(self.spaces.Hcurl, name='q')
#
#     def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
#         vtest = xhats['v']
#         F = dfdx_linear_vars['F']
#         total_dens = self.total_density_func(const_state)
#         n = self.spaces.n
#
#         if self.dim == 1:
#             ERROR
#         elif self.dim == 2:
#             rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*self.dx
#         elif self.dim == 3:
#             ERROR
#
#         for dens_name in self.active_density_names:
#             denstest = xhats[dens_name]
#             Bdens = dfdx_linear_vars['B_' + dens_name]
#             dens = const_state[dens_name]
#             rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), n('+')) + denstest('-')*inner(F('-'), n('-')))*dens/total_dens*self.dS
#             rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), n('+')) + Bdens('-')*inner(vtest('-'), n('-')))*dens/total_dens*self.dS
#             if self.spaces.order >1:
#                 rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx
#                 rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*self.dx
#         return rhs_expr
#
#     def rhs(self, qvars, dfdx_vars, xhats):
#         v = qvars['v']
#         F = dfdx_vars['F']
#         total_dens = self.total_density_func(qvars)
#         vtest = xhats['v']
#         n = self.spaces.n
#
#         alpha = self.alpha_s * sign(dot(v('+'),n('+')))
#         total_dens_avg = 0.5 * (total_dens('+') + total_dens('-'))
#
# #MISSING BOUNDARY TERMS- ds
#         if self.dim == 1:
#             ERROR
#         elif self.dim == 2:
#             if self.upwind_v:
# #TO DO THIS "RIGHT" SHOULD ACTUALLY COMPUTE FLOW VELOCITY U I THINK IE FOLLOWING GOLO PAPER!
#                 nperp = rot2D(n)
#                 vtilde = 0.5 * ((1. + alpha) * v('+') + (1. - alpha)*v('-'))
#                 Fperp = rot2D(F)
#                 rhs_expr = inner(vtest, self.coriolis / total_dens * Fperp)*self.dx
#                 Fpart = inner(vtest,Fperp)/total_dens
#                 rhs_expr = rhs_expr - inner(skewgrad(Fpart), v)*self.dx
#                 jump_term = Fpart('+')*nperp('+') + Fpart('-')*nperp('-')
#                 rhs_expr = rhs_expr + inner(jump_term,vtilde)*self.dS
#             else:
#                 q = qvars['q']
#                 rhs_expr = inner(vtest, q * rot2D(F))*self.dx #q has coriolis in it!
#         elif self.dim == 3:
#             ERROR
#
# #TO DO THIS "RIGHT" SHOULD ACTUALLY COMPUTE FLOW VELOCITY U I THINK IE FOLLOWING GOLO PAPER!
#
#         for dens_name in self.active_density_names:
#             denstest = xhats[dens_name]
#             Bdens = dfdx_vars['B_' + dens_name]
#             dens = qvars[dens_name]
#
# #MISSING BOUNDARY TERMS- ds
#
#             denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
#             rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), n('+')) + denstest('-')*inner(F('-'), n('-')))*denstilde/total_dens_avg*self.dS
#             rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), n('+')) + Bdens('-')*inner(vtest('-'), n('-')))*denstilde/total_dens_avg*self.dS
#             if self.spaces.order >1:
#                 rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*self.dx
#                 rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*self.dx
#         return rhs_expr
#

# class CurlForm_AdvectedDensities_Bracket_H1(CurlForm_AdvectedDensities_Bracket_Base):
#
#     def rhs(self, qvars, dfdx_vars, xhats):
#         v = qvars['v']
#         F = dfdx_vars['F']
#         total_dens = self.total_density_func(qvars)
#         vtest = xhats['v']
#
#
#         if self.dim == 1:
#             ERROR
#         elif self.dim == 2:
#             rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*self.dx
#             rhs_expr = rhs_expr + inner(vtest, curl2D(v) / total_dens * rot2D(F))*self.dx
#         elif self.dim == 3:
#             ERROR
#
#
#         for dens_name in self.active_density_names:
#             denstest = xhats[dens_name]
#             Bdens = dfdx_vars['B_' + dens_name]
#             dens = qvars[dens_name]
#             if self.use_split_form[dens_name]:
#                 rhs_expr = rhs_expr + inner(denstest, 0.5 *( div(dens/total_dens*F) + dot(grad(dens/total_dens),F) + dens/total_dens * div(F)))*self.dx
#                 rhs_expr = rhs_expr + inner(vtest, 0.5 *( dens/total_dens*grad(Bdens) + grad(Bdens*dens/total_dens)))*self.dx + 0.5 * inner(dens/total_dens, div(vtest*Bdens))*self.dx
#             else:
#                 rhs_expr = rhs_expr + inner(denstest, div(dens/total_dens*F))*self.dx
#                 rhs_expr = rhs_expr + inner(vtest, dens/total_dens*grad(Bdens))*self.dx
#
#         return rhs_expr
#
# #SHOULD BE CAPABLE OF USING THE SIMPLIFIED VERSION, I THINK...
#     def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
#         F = dfdx_linear_vars['F']
#         total_dens = self.total_density_func(const_state)
#         vtest = xhats['v']
#
# #THIS IS ALL 2D SPECIFIC, EVENTUALLY GENERALIZE TO 3D?
#
#         rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*self.dx
#
#         for dens_name in self.active_density_names:
#             denstest = xhats[dens_name]
#             Bdens = dfdx_linear_vars['B_' + dens_name]
#             dens = const_state[dens_name]
#             #don't need split form here for approximate Jacobian
#             rhs_expr = rhs_expr + inner(denstest, div(dens/total_dens*F))*self.dx
#             rhs_expr = rhs_expr + inner(vtest, dens/total_dens*grad(Bdens))*self.dx
#         return rhs_expr





#
#
# #WE CAN ELIMINATE THIS WITH THE APPROPRIATE GENERALIZATION OF LP AND CF BRACKETS!!!
# class MHDBracket_LP(PoissonBracket):
#     def __init__(self, spaces):
#         self.spaces = spaces
#
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
