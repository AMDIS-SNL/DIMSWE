from firedrake import derivative, TestFunction, inner, TrialFunction, dx, curl, as_vector, Function, ds, dS, grad, div, dot, sign
from ufl_helpers import skewgrad, curl2D, rot2D
from transport_operators import SVLieDerivative, VVLieDerivative, CVLieDerivative


class PoissonBracket():
    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

class LiePoisson_AdvectedQuantities_Bracket(PoissonBracket):
    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.advected_quantity_names = vars.advected_quantity_names
        self.advected_quantity_bundle = vars.advected_quantity_bundle
        self.advected_quantity_degree = vars.advected_quantity_degree
        self.testvars = {}
        self.trialvars = {}
        self.alpha_s = parameters['alpha_s']
        self.dim = parameters['dim']
        self.total_density_func = vars.get_total_density_expr

        if not spaces is None:
            if self.dim == 2:
                self.coriolis = Function(spaces.CG)
            elif self.dim == 3:
                self.coriolis = Function(spaces.CGV) #PRETTY UNCLEAR ACTUALLY- MAYBE H(curl) or even H(div)?


    def initialize(self, varexpr):
        if self.dim == 2 or self.dim == 3:
            self.coriolis.interpolate(varexpr['coriolis'])

    def rhs(self, qvars, dfdx_vars, xhats):
        m = qvars['m']
        u = dfdx_vars['u']
        mtest = xhats['m']
        total_dens = self.total_density_func(qvars)

        rhs_expr = CVLieDerivative(3, self.dim, u, m, mtest)

        for name, bundle, degree in zip(self.advected_quantity_names, self.advected_quantity_bundle, self.advected_quantity_degree):
            if bundle == 'S':
                rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, u, qvars[name], xhats[name])
                rhs_expr = rhs_expr - SVLieDerivative(degree, self.dim, mtest, qvars[name], dfdx_vars[name])
            elif bundle == 'VV':
                rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, u, qvars[name], xhats[name])
                rhs_expr = rhs_expr - VVLieDerivative(degree, self.dim, mtest, qvars[name], dfdx_vars[name])
            elif bundle == 'CV':
                rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, u, qvars[name], xhats[name])
                rhs_expr = rhs_expr - CVLieDerivative(degree, self.dim, mtest, qvars[name], dfdx_vars[name])

        if self.dim == 2:
            rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*dx
        elif self.dim == 3:
            ERROR
        return rhs_expr

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return self.rhs(const_state, dfdx_linear_vars, xhats)

class CurlForm_AdvectedQuantities_Bracket(PoissonBracket):
    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.vars = vars
        self.advected_quantity_names = vars.advected_quantity_names
        self.advected_quantity_bundle = vars.advected_quantity_bundle
        self.advected_quantity_degree = vars.advected_quantity_degree
        self.dim = parameters['dim']
        self.total_density_func = vars.get_total_density_expr
        self.testvars = {}
        self.trialvars = {}
        self.upwind_v = parameters['upwind_v']
#THIS STUFF SHOULD REALLY BE CONFIGURABLE PER ADVECTED QUANTITY I THINK..
        self.alpha_s = parameters['alpha_s']

        if not spaces is None:
            if self.dim == 2:
                self.coriolis = Function(spaces.CG)
            elif self.dim == 3:
                self.coriolis = Function(spaces.CGV) #PRETTY UNCLEAR ACTUALLY- MAYBE H(curl) or even H(div)?

            if self.dim == 2:
                self.testvars['q'] = TestFunction(self.spaces.CG)
                self.trialvars['q'] = TrialFunction(self.spaces.CG)
##THESE ARE ACTUALLY SPECIFIC TO HDIV VARIANT, NOT GENERAL ENOUGH FOR H1 in 3D.
#probably not doing H1 in 3D, so maybe okay?
            elif self.dim == 3:
                self.testvars['q'] = TestFunction(self.spaces.Hcurl)
                self.trialvars['q'] = TrialFunction(self.spaces.Hcurl)


    def initialize(self, varexpr):
        if self.dim == 2 or self.dim == 3:
            self.coriolis.interpolate(varexpr['coriolis'])

    def get_aux_vars_list(self):
        if self.dim >= 2 and not self.upwind_v:
            return ['q',]
        else:
            return []

    def compute_q_expressions(self, vars, expressions):
        if not self.upwind_v:
            v = vars['v']
            total_dens = self.total_density_func(vars)
            qhat = self.testvars['q']
            qtrial = self.trialvars['q']
    #MISSING BOUNDARY TERMS...
            if self.dim == 2:
                expressions['q'] = [inner(qhat, total_dens * qtrial)*dx, inner(-skewgrad(qhat), v)*dx + inner(qhat, self.coriolis)*dx]
            elif self.dim == 3:
                expressions['q'] = [inner(qhat, total_dens * qtrial)*dx, inner(-curl(qhat), v)*dx + inner(qhat, self.coriolis)*dx]

    def rhs(self, qvars, dfdx_vars, xhats):
        v = qvars['v']
        F = dfdx_vars['F']
        vtest = xhats['v']
        total_dens = self.total_density_func(qvars)

#MISSING BOUNDARY TERMS- ds
        rhs_expr = 0.0
        if self.dim == 2:
            if self.upwind_v:
#PROBABLY WRAP THIS UP IN ONE OF THE LIE DERIVATIVE CLASSES! OR SOME SORT OF INTERIOR PRODUCT...
#TO DO THIS "RIGHT" SHOULD ACTUALLY COMPUTE FLOW VELOCITY U I THINK IE FOLLOWING GOLO PAPER!
                nperp = rot2D(n)
                rhs_expr = rhs_expr + inner(vtest, self.coriolis / total_dens * rot2D(F))*dx
                vtilde = 0.5 * ((1. + alpha) * v('+') + (1. - alpha)*v('-'))
                Fperp = rot2D(F)
                rhs_expr = rhs_expr + inner(vtest, self.coriolis / total_dens * Fperp)*dx
                Fpart = inner(vtest,Fperp)/total_dens
                rhs_expr = rhs_expr - inner(skewgrad(Fpart), v)*dx
                jump_term = Fpart('+')*nperp('+') + Fpart('-')*nperp('-')
                rhs_expr = rhs_expr + inner(jump_term,vtilde)*dS
            else:
                q = qvars['q']
                rhs_expr = inner(vtest, q * rot2D(F))*dx #q has coriolis in it!
        elif self.dim == 3:
            ERROR

        for name, bundle, degree in zip(self.advected_quantity_names, self.advected_quantity_bundle, self.advected_quantity_degree):
            if bundle == 'S':
                rhs_expr = rhs_expr + SVLieDerivative(degree, self.dim, F/total_dens, const_state[name], xhats[name])
                rhs_expr = rhs_expr - SVLieDerivative(degree, self.dim, vtest/total_dens, const_state[name], dfdx_vars[name])
            elif bundle == 'VV':
                rhs_expr = rhs_expr + VVLieDerivative(degree, self.dim, F/total_dens, const_state[name], xhats[name])
                rhs_expr = rhs_expr - VVLieDerivative(degree, self.dim, vtest/total_dens, const_state[name], dfdx_vars[name])
            elif bundle == 'CV':
                rhs_expr = rhs_expr + CVLieDerivative(degree, self.dim, F/total_dens, const_state[name], xhats[name])
                rhs_expr = rhs_expr - CVLieDerivative(degree, self.dim, vtest/total_dens, const_state[name], dfdx_vars[name])


    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return self.rhs(const_state, dfdx_linear_vars, xhats)






class LiePoisson_AdvectedDensities_Bracket(PoissonBracket):

    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.density_names = vars.density_names
        self.testvars = {}
        self.trialvars = {}
        self.alpha_s = parameters['alpha_s']
        self.dim = parameters['dim']

        if not spaces is None:
            self.coriolis = Function(spaces.CG)


    def initialize(self, varexpr):
        self.coriolis.interpolate(varexpr['coriolis'])

    def rhs(self, qvars, dfdx_vars, xhats):
        m = qvars['m']
        u = dfdx_vars['u']
        mtest = xhats['m']
        n = self.spaces.n

        alpha = self.alpha_s * sign(dot(u('+'),n('+')))

#FIX THIS STUFF UP!
        mtilde = 0.5 * ((1. + alpha) * m('+') + (1. - alpha)*m('-'))
        rhs_expr = (mtest('+')*inner(u('+'), n('+')) + mtest('-')*inner(u('-'), n('-')))*mtilde*dS
        rhs_expr = rhs_expr - (u('+')*inner(mtest('+'), n('+')) + u('-')*inner(mtest('-'), n('-')))*mtilde*dS

#FIX THIS STUFF UP!
#super unclear if this is the correct notation
#probably need some sort of tensor product type thing for u*m, etc.
#could write in terms of coordinates pretty easy, so maybe do that?
        if self.spaces.order >1:
            rhs_expr = rhs_expr - inner(grad(mtest), outer(u,m))*dx
            rhs_expr = rhs_expr + inner(grad(u), outer(mtest,m))*dx

        if self.dim == 1:
            ERROR
        elif self.dim == 2:
            rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*dx
        elif self.dim == 3:
            ERROR

        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_vars['B_' + dens_name]
            dens = qvars[dens_name]

#MISSING BOUNDARY TERMS- ds
            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(u('+'), n('+')) + denstest('-')*inner(u('-'), n('-')))*denstilde*dS
            rhs_expr = rhs_expr - (Bdens('+')*inner(mtest('+'), n('+')) + Bdens('-')*inner(mtest('-'), n('-')))*denstilde*dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr + inner(grad(Bdens   ), dens * mtest)*dx
                rhs_expr = rhs_expr - inner(grad(denstest), dens * u   )*dx
        return rhs_expr

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        m = const_state['m']
        u = dfdx_linear_vars['u']
        mtest = xhats['m']
        n = self.spaces.n
        total_dens = self.total_density_func(const_state)


        rhs_expr = (mtest('+')*inner(u('+'), n('+')) + mtest('-')*inner(u('-'), n('-')))*m*dS
        rhs_expr = rhs_expr - (u('+')*inner(mtest('+'), n('+')) + u('-')*inner(mtest('-'), n('-')))*m*dS

        if self.dim == 1:
            ERROR
        elif self.dim == 2:
            rhs_expr = rhs_expr + inner(mtest, total_dens*self.coriolis*rot2D(u))*dx
        elif self.dim == 3:
            ERROR

        if self.spaces.order >1:
#GRAD OR NABLA GRAD HERE?
            rhs_expr = rhs_expr - inner(grad(mtest), outer(u,m))*dx
            rhs_expr = rhs_expr + inner(grad(u), outer(mtest,m))*dx

        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_linear_vars['B_' + dens_name]
            dens = const_state[dens_name]

#MISSING BOUNDARY TERMS- ds
            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(u('+'), n('+')) + denstest('-')*inner(u('-'), n('-')))*dens*dS
            rhs_expr = rhs_expr - (Bdens('+')*inner(mtest('+'), n('+')) + Bdens('-')*inner(mtest('-'), n('-')))*dens*dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr + inner(grad(Bdens   ), dens * mtest)*dx
                rhs_expr = rhs_expr - inner(grad(denstest), dens * u   )*dx
        return rhs_expr

class CurlForm_AdvectedDensities_Bracket_Base(PoissonBracket):
    def __init__(self, spaces, vars, parameters):
        self.spaces = spaces
        self.vars = vars
        self.density_names = vars.density_names
        self.dg_density_names = vars.dg_density_names
        self.dim = parameters['dim']
        self.total_density_func = vars.get_total_density_expr
        self.testvars = {}
        self.trialvars = {}
        self.alpha_s = parameters['alpha_s']
        self.upwind_v = parameters['upwind_v']
        #NOT CURRENTLY USED
        self.upwind_total_dens = parameters['upwind_total_dens']
        self.use_split_form = parameters['use_split_form']


        if not spaces is None:
            self.coriolis = Function(spaces.CG)

        if self.spaces is not None and not self.upwind_v:
            if self.dim == 2:
                self.testvars['q'] = TestFunction(self.spaces.CG)
                self.trialvars['q'] = TrialFunction(self.spaces.CG)
##THESE ARE ACTUALLY SPECIFIC TO HDIV VARIANT, NOT GENERAL ENOUGH FOR H1 in 3D.
#probably not doing H1 in 3D, so maybe okay?
            elif self.dim == 3:
                self.testvars['q'] = TestFunction(self.spaces.Hcurl)
                self.trialvars['q'] = TrialFunction(self.spaces.Hcurl)

    def initialize(self, varexpr):
        self.coriolis.interpolate(varexpr['coriolis'])


    def get_aux_vars_list(self):
        if self.dim >= 2 and not self.upwind_v:
            return ['q',]
        else:
            return []

    def compute_q_expressions(self, vars, expressions):
        if not self.upwind_v:
            v = vars['v']
            total_dens = self.total_density_func(vars)
            qhat = self.testvars['q']
            qtrial = self.trialvars['q']
    #MISSING BOUNDARY TERMS...
            if self.dim == 2:
                expressions['q'] = [inner(qhat, total_dens * qtrial)*dx, inner(-skewgrad(qhat), v)*dx + inner(qhat, self.coriolis)*dx]
            elif self.dim == 3:
                expressions['q'] = [inner(qhat, total_dens * qtrial)*dx, inner(-curl(qhat), v)*dx + inner(qhat, self.coriolis)*dx]


#SWAP THESE TO USE LIE DERIVATIVE FUNCTIONS
class CurlForm_AdvectedDensities_Bracket(CurlForm_AdvectedDensities_Bracket_Base):

    def get_aux_vars(self, vars):
        if self.dim == 2 and not self.upwind_v:
            vars['q'] = Function(self.spaces.CG, name='q')
        elif self.dim == 3 and not self.upwind_v:
            vars['q'] = Function(self.spaces.Hcurl, name='q')

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        vtest = xhats['v']
        F = dfdx_linear_vars['F']
        total_dens = self.total_density_func(const_state)
        n = self.spaces.n

        if self.dim == 1:
            ERROR
        elif self.dim == 2:
            rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*dx
        elif self.dim == 3:
            ERROR

        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_linear_vars['B_' + dens_name]
            dens = const_state[dens_name]
            rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), n('+')) + denstest('-')*inner(F('-'), n('-')))*dens/total_dens*dS
            rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), n('+')) + Bdens('-')*inner(vtest('-'), n('-')))*dens/total_dens*dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*dx
                rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*dx
        return rhs_expr

    def rhs(self, qvars, dfdx_vars, xhats):
        v = qvars['v']
        F = dfdx_vars['F']
        total_dens = self.total_density_func(qvars)
        vtest = xhats['v']
        n = self.spaces.n

        alpha = self.alpha_s * sign(dot(v('+'),n('+')))
        total_dens_avg = 0.5 * (total_dens('+') + total_dens('-'))

#MISSING BOUNDARY TERMS- ds
        if self.dim == 1:
            ERROR
        elif self.dim == 2:
            if self.upwind_v:
#TO DO THIS "RIGHT" SHOULD ACTUALLY COMPUTE FLOW VELOCITY U I THINK IE FOLLOWING GOLO PAPER!
                nperp = rot2D(n)
                vtilde = 0.5 * ((1. + alpha) * v('+') + (1. - alpha)*v('-'))
                Fperp = rot2D(F)
                rhs_expr = inner(vtest, self.coriolis / total_dens * Fperp)*dx
                Fpart = inner(vtest,Fperp)/total_dens
                rhs_expr = rhs_expr - inner(skewgrad(Fpart), v)*dx
                jump_term = Fpart('+')*nperp('+') + Fpart('-')*nperp('-')
                rhs_expr = rhs_expr + inner(jump_term,vtilde)*dS
            else:
                q = qvars['q']
                rhs_expr = inner(vtest, q * rot2D(F))*dx #q has coriolis in it!
        elif self.dim == 3:
            ERROR

#TO DO THIS "RIGHT" SHOULD ACTUALLY COMPUTE FLOW VELOCITY U I THINK IE FOLLOWING GOLO PAPER!

        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_vars['B_' + dens_name]
            dens = qvars[dens_name]

#MISSING BOUNDARY TERMS- ds

            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), n('+')) + denstest('-')*inner(F('-'), n('-')))*denstilde/total_dens_avg*dS
            rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), n('+')) + Bdens('-')*inner(vtest('-'), n('-')))*denstilde/total_dens_avg*dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*dx
                rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*dx
        return rhs_expr


class CurlForm_AdvectedDensities_Bracket_H1(CurlForm_AdvectedDensities_Bracket_Base):

    def rhs(self, qvars, dfdx_vars, xhats):
        v = qvars['v']
        F = dfdx_vars['F']
        total_dens = self.total_density_func(qvars)
        vtest = xhats['v']


        if self.dim == 1:
            ERROR
        elif self.dim == 2:
            rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*dx
            rhs_expr = rhs_expr + inner(vtest, curl2D(v) / total_dens * rot2D(F))*dx
        elif self.dim == 3:
            ERROR


        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_vars['B_' + dens_name]
            dens = qvars[dens_name]
            if self.use_split_form[dens_name]:
                rhs_expr = rhs_expr + inner(denstest, 0.5 *( div(dens/total_dens*F) + dot(grad(dens/total_dens),F) + dens/total_dens * div(F)))*dx
                rhs_expr = rhs_expr + inner(vtest, 0.5 *( dens/total_dens*grad(Bdens) + grad(Bdens*dens/total_dens)))*dx + 0.5 * inner(dens/total_dens, div(vtest*Bdens))*dx
            else:
                rhs_expr = rhs_expr + inner(denstest, div(dens/total_dens*F))*dx
                rhs_expr = rhs_expr + inner(vtest, dens/total_dens*grad(Bdens))*dx

        n = self.spaces.n
        alpha = self.alpha_s * sign(dot(v('+'),n('+')))
        total_dens_avg = 0.5 * (total_dens('+') + total_dens('-'))
        for dens_name in self.dg_density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_vars['B_' + dens_name]
            dens = qvars[dens_name]
#MISSING BOUNDARY TERMS- ds
            denstilde = 0.5 * ((1. + alpha) * dens('+') + (1. - alpha)*dens('-'))
            rhs_expr = rhs_expr + (denstest('+')*inner(F('+'), n('+')) + denstest('-')*inner(F('-'), n('-')))*denstilde/total_dens_avg*dS
            rhs_expr = rhs_expr - (Bdens('+')*inner(vtest('+'), n('+')) + Bdens('-')*inner(vtest('-'), n('-')))*denstilde/total_dens_avg*dS
            if self.spaces.order >1:
                rhs_expr = rhs_expr + inner(grad(Bdens   ), dens/total_dens * vtest)*dx
                rhs_expr = rhs_expr - inner(grad(denstest), dens/total_dens * F   )*dx
        return rhs_expr

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        F = dfdx_linear_vars['F']
        total_dens = self.total_density_func(const_state)
        vtest = xhats['v']

#THIS IS ALL 2D SPECIFIC, EVENTUALLY GENERALIZE TO 3D?

        rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*dx

        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_linear_vars['B_' + dens_name]
            dens = const_state[dens_name]
            #don't need split form here for approximate Jacobian
            rhs_expr = rhs_expr + inner(denstest, div(dens/total_dens*F))*dx
            rhs_expr = rhs_expr + inner(vtest, dens/total_dens*grad(Bdens))*dx
        return rhs_expr






#WE CAN ELIMINATE THIS WITH THE APPROPRIATE GENERALIZATION OF LP AND CF BRACKETS!!!
class MHDBracket_LP():
    def __init__(self, spaces):
        self.spaces = spaces

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def rhs(self, xn, xnp1, qvars, dfdx_vars):
        pass

    def initialize(self, varexpr):
        pass

class MaxwellBracket():
    def __init__(self, spaces):
        self.spaces = spaces

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def rhs(self, xn, xnp1, qvars, dfdx_vars):
        pass

    def initialize(self, varexpr):
        pass

class EulerMaxwellCouplingBracket_LP():
    def __init__(self, spaces):
        self.spaces = spaces

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def rhs(self, xn, xnp1, qvars, dfdx_vars):
        pass

    def initialize(self, varexpr):
        pass

class ScalarWaveBracket():
    def __init__(self, spaces):
        self.spaces = spaces

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def rhs(self, xn, xnp1, qvars, dfdx_vars):
        pass

    def initialize(self, varexpr):
        pass
