#from .poisson_brackets import LiePoisson_AdvectedDensities_Bracket, CurlForm_AdvectedDensities_Bracket, EulerMaxwellCouplingBracket_LP, MHDBracket_LP, ScalarWaveBracket, MaxwellBracket
#from .metric_brackets import ThermodynamicallyCompatibleViscousRegularization_CF, ThermodynamicallyCompatibleViscousRegularization_LP
#from .hamiltonians import CompressibleEuler_Hamiltonian_CF, CompressibleEuler_Hamiltonian_LP
#from .hamiltonians import Maxwell_Hamiltonian, EulerMaxwell_Hamiltonian_LP, ScalarWave_Hamiltonian, MHD_Hamiltonian_LP, get_thermo
from .hamiltonians import ThermalShallowWater_Hamiltonian_CF
#ThermalShallowWater_Hamiltonian_LP
#from .hamiltonians import MoistThermalShallowWater_Hamiltonian_CF, MoistThermalShallowWater_Hamiltonian_LP, MoistThermalShallowWater_Hamiltonian_CF_H1
#from .hamiltonians import IdealGasThermo_Entropy, IdealGasThermo_PotTemp
#from .transport_operators import SVLieDerivative, VVLieDerivative, CVLieDerivative

from firedrake import Function, inner, div, grad, dot, Constant, split, TrialFunction, TrialFunctions, TestFunction, TestFunctions, MixedFunctionSpace
from .ufl_helpers import skewgrad, curl2D, rot2D

import scipy as sp

class Dynamics():

    def get_x_var(self, varname):

        if not (self.xspace is None):
            x = Function(self.xspace, name=varname)
            x_sub = {}
            x_split = {}
            splitx = split(x)
            for i,var in enumerate(self.xvarlist):
                x_sub[var] = x.sub(i)
                x_split[var] = splitx[i]

            return x, x_sub, x_split

        else:
            return None, {}, {}

    def get_full_var(self, varname, split_x_and_aux=False):

        if not (self.fullspace is None):
            if split_x_and_aux:
                x = Function(self.xspace, name=varname)
                x_sub = {}
                x_split = {}
                splitx = split(x)
                for i,var in enumerate(self.xvarlist):
                    x_sub[var] = x.sub(i)
                    x_split[var] = splitx[i]
                aux = Function(self.auxspace, name=varname)
                aux_sub = {}
                aux_split = {}
                splitaux = split(aux)
                for i,var in enumerate(self.auxvarlist):
                    aux_sub[var] = aux.sub(i)
                    aux_split[var] = splitaux[i]
                return [x,aux], x_sub | aux_sub, x_split | aux_split
            else:
                full = Function(self.fullspace, name=varname)
                full_sub = {}
                full_split = {}
                splitfull = split(full)
                for i,var in enumerate(self.fullvarlist):
                    full_sub[var] = full.sub(i)
                    full_split[var] = splitfull[i]
                return [full,], full_sub, full_split

        else:
            return None, {}, {}

    def get_aux_var(self, varname):
        if not (self.auxspace is None):
            aux = Function(self.auxspace, name=varname)
            aux_sub = {}
            aux_split = {}
            splitaux = split(aux)
            for i,var in enumerate(self.auxvarlist):
                aux_sub[var] = aux.sub(i)
                aux_split[var] = splitaux[i]

            return aux, aux_sub, aux_split

        else:
            return None, {}, {}


#THIS IGNORES THAT THERE MIGHT BE PARAMETERS IN DYNAMICS (ex. through the Hamiltonian, etc.)
#THIS IS AN EASY EXTENSION AS NEEDED...
    def get_coeff_var(self, varname):

        if not (self.coeffspace is None):
            coeff = Function(self.coeffspace, name=varname)
            coeff_trial = TrialFunction(self.coeffspace)
            coeff_sub = {}
            coeff_split = {}
            splitcoeff = split(coeff)
            for i,var in enumerate(self.coefflist):
                coeff_sub[var] = coeff.sub(i)
                coeff_split[var] = splitcoeff[i]

            return coeff, coeff_sub, coeff_split

        else:
            return None, {}, {}, None

    def get_coeff_test_trial_vars(self):
        if not (self.coeffspace is None):
            coeff_trial = TrialFunction(self.coeffspace)
            coeff_trials = TrialFunctions(self.coeffspace)
            coeff_test = TestFunction(self.coeffspace)
            coeff_tests = TestFunctions(self.coeffspace)
            coeff_trial_subs = {}
            coeff_test_subs = {}
            for i,var in enumerate(self.coefflist):
                coeff_trial_subs[var] = coeff_trials[i]
                coeff_test_subs[var] = coeff_tests[i]
            return coeff_test, coeff_test_subs, coeff_trial, coeff_trial_subs
        else:
            return None, {}, None, {}

    def get_x_spaces(self,):
        return self.xspacelist

    def get_full_spaces(self,):
        return self.fullspacelist

    def get_aux_spaces(self):
        return self.auxspacelist

    def get_coeff_spaces(self):
        return self.coeffspacelist

    def get_x_var_list(self):
        return self.xvarlist

    def get_full_var_list(self):
        return self.fullvarlist

    def get_coeff_list(self):
        return self.coefflist

    def get_dfdx_aux_var_list(self, terms='all'):
        return []

    def compute_dfdx_expressions(self, x, t, coeff, xhats, terms='all'):
        return {}

    def get_q_aux_var_list(self, terms='all'):
        return []

    def compute_q_expressions(self, x, t, coeff, xhats, terms='all'):
        return {}

    def get_aux_var_list(self, terms='all'):
        return []

    def compute_aux_expressions(self, x, t, coeff, xhats, terms='all'):
        return {}

    def get_x_test_vars(self):
        xhat = TestFunction(self.xspace)
        xhats = TestFunctions(self.xspace)
        xhat_subs = {}
        for i,var in enumerate(self.xvarlist):
            xhat_subs[var] = xhats[i]
        return xhat, xhat_subs

    def get_x_trial_vars(self):
        xtrial = TrialFunction(self.xspace)
        xtrials = TrialFunctions(self.xspace)
        xtrial_subs = {}
        for i,var in enumerate(self.xvarlist):
            xtrial_subs[var] = xtrials[i]
        return xtrial, xtrial_subs

    def get_full_test_vars(self, split_x_and_aux=False):
        if split_x_and_aux:
            xhat = TestFunction(self.xspace)
            xhats = TestFunctions(self.xspace)
            xhat_subs = {}
            for i,var in enumerate(self.xvarlist):
                xhat_subs[var] = xhats[i]
            auxhat = TestFunction(self.auxspace)
            auxhats = TestFunctions(self.auxspace)
            auxhat_subs = {}
            for i,var in enumerate(self.auxvarlist):
                auxhat_subs[var] = auxhats[i]
            return [xhat, auxhat], xhat_subs | auxhat_subs
        else:
            xhat = TestFunction(self.fullspace)
            xhats = TestFunctions(self.fullspace)
            xhat_subs = {}
            for i,var in enumerate(self.fullvarlist):
                xhat_subs[var] = xhats[i]
            return [xhat,], xhat_subs

    def get_full_trial_vars(self, split_x_and_aux=False):
        if split_x_and_aux:
            xtrial = TrialFunction(self.xspace)
            xtrials = TrialFunctions(self.xspace)
            xtrial_subs = {}
            for i,var in enumerate(self.xvarlist):
                xtrial_subs[var] = xtrials[i]
            auxtrial = TrialFunction(self.auxspace)
            auxtrials = TrialFunctions(self.auxspace)
            auxtrial_subs = {}
            for i,var in enumerate(self.auxvarlist):
                auxtrial_subs[var] = auxtrials[i]
            return [xtrial,auxtrial], xtrial_subs | auxtrial_subs
        else:
            xtrial = TrialFunction(self.fullspace)
            xtrials = TrialFunctions(self.fullspace)
            xtrial_subs = {}
            for i,var in enumerate(self.fullvarlist):
                xtrial_subs[var] = xtrials[i]
            return [xtrial,], xtrial_subs

    def get_aux_test_vars(self):
        if self.has_aux:
            auxhat = TestFunction(self.auxspace)
            auxhats = TestFunctions(self.auxspace)
            auxhat_subs = {}
            for i,var in enumerate(self.auxvarlist):
                auxhat_subs[var] = auxhats[i]
            return auxhat, auxhat_subs
        else:
            return None, {}

    def get_aux_trial_vars(self):
        if self.has_aux:
            auxtrial = TrialFunction(self.auxspace)
            auxtrials = TrialFunctions(self.auxspace)
            auxtrial_subs = {}
            for i,var in enumerate(self.auxvarlist):
                auxtrial_subs[var] = auxtrials[i]
            return auxtrial, auxtrial_subs
        else:
            return None, {}

#THIS IGNORES THAT THERE MIGHT BE PARAMETERS IN DYNAMICS (ex. through the Hamiltonian, etc.)
#THIS IS AN EASY EXTENSION AS NEEDED..
    def set_default_coeffs(self, coeff):
        for term in self.forcing_terms:
            term.set_default_coeffs(coeff)



#This is basically going to be LP brackets (or CF, etc.) with prescribed u or F
#instead of diagnostic
#can basically do it as a modified Hamiltonian, actually!!!
#although maybe interaction with q for CF is a little tricky?
# class AdvectionDynamics(Dynamics):
#     def __init__(self, mesh, spaces, initcond, vars, statistics, diagnostics, logger):
#         self.mesh = mesh
#         self.spaces = spaces
#         self.variableset = vars
#         self.diagnostics = diagnostics
#         self.statistics = statistics
#         self.logger = logger
#         self.initcond = initcond
#
#         self.advected_quantities = parameters['advected_quantities']
#         self.form_degrees = parameters['form_degrees'] #0,1,2,3 in 3D, 0,1,2 in 2D (MAYBE NEED TO DISTINGUISH 1-FORMS AND N-1 FORMS HERE?)
#         self.form_bundle = parameters['form_bundles'] #S, VV, CV
#         self.advection_type = parameters['advection_type'] #'CF' 'LP' 'CF-H1'
# #ALSO ADD IN VARIOUS CHOICES OF SPLIT FORM, UPWINDING, ETC.
# #basically here we want to be able to test a large variety of advection choices for different types of initial conditions, ex. gaussian and square, probably the slotted cylinder, etc.
#
# #WITH THESE WE CAN BASICALLY BUILD ALL THE POISSON BRACKETS
# #For Euler-Maxwell there are coupling terms that also need to be added, they are just various interior products though!
#         self.total_dens_func = self.variableset.get_total_density_expr()
#
#     def get_dfdx_aux_var_list(self, terms='all'):
#         if self.advection_type in ['CF', 'CF-H1']:
#             return ['F',]
#         elif self.advection_type == 'LP':
#             return ['u', ]
#
#
#     def get_dfdx_aux_vars(self, terms='all'):
#         if self.advection_type == 'CF':
#             return {'F' : Function(self.spaces.Hdiv, name='F')}
#         elif self.advection_type == 'CF-H1':
#             return {'F' : Function(self.spaces.CGV, name='F')}
#         elif self.advection_type == 'LP':
#             return {'u' : Function(self.spaces.DGV, name='u')}
#
#     def initialize(self, vars, t):
#         varexpr = SOMETHING
#         t.assign(SOMETHING)
#         self.variableset.initialize(varexpr, vars)
#
# #FIX THIS
# #THIS IS A GREAT WAY TO TEST t dependence!
#     def compute_dfdx_expressions(self, x, t, terms='all'):
#         if self.advection_type in ['CF', 'CF-H1']:
#             Fexpr = SOMETHING
#             return {'F' : Fexpr}
#         elif self.advection_type == 'LP':
#             uexrp = SOMETHING
#             return {'u' : uexrp}
#
#     def rhs(self, qvars, dfdx_vars, xhats, t, terms='all'):
#         rhs = 0.0
#         if self.advection_type == 'CF':
#             total_dens = self.total_dens_func(qvars)
#             for quantity, bundle, degree in zip(self.advected_quantities, self.form_bundles, self.form_degrees):
#                 if bundle == 'S':
#                     rhs = rhs + SVLieDerivative(degree, dim, dfdx_vars['F'] / total_dens, qvars[quantity], xhats[quantity])
#                 elif bundle == 'VV':
#                     rhs = rhs + VVLieDerivative(degree, dim, dfdx_vars['F'] / total_dens, qvars[quantity], xhats[quantity])
#                 elif bundle == 'LP':
#                     rhs = rhs + VVLieDerivative(degree, dim, dfdx_vars['F'] / total_dens, qvars[quantity], xhats[quantity])
#         elif self.advection_type == 'CF-H1':
# #FIX THIS- ALL THESE LIE DERIVATIVES ARE GOING TO BE DIFFERENT...
# #ALSO HERE WE HAVE SPLIT FORM VS NON SPLIT FORM CHOICE
#             pass
#         elif self.advection_type == 'LP':
#             for quantity, bundle, degree in zip(self.advected_quantities, self.form_bundles, self.form_degrees):
#                 if bundle == 'S':
#                     rhs = rhs + SVLieDerivative(degree, dim, dfdx_vars['u'], qvars[quantity], xhats[quantity])
#                 elif bundle == 'VV':
#                     rhs = rhs + VVLieDerivative(degree, dim, dfdx_vars['u'], qvars[quantity], xhats[quantity])
#                 elif bundle == 'LP':
#                     rhs = rhs + VVLieDerivative(degree, dim, dfdx_vars['u'], qvars[quantity], xhats[quantity])
#         return rhs
#
#
# #FIX THISTHESE ARE ALL GOING TO BE DIFFERENT...
# #Might actually just be rhs? UNCLEAR!
#     def linear_rhs(self, const_state, xstar, xhats, t, terms='all'):
#         if self.advection_type == 'CF':
#             return 0.0
#         elif self.advection_type == 'LP':
#             return 0.0




class AdvDensCF_H1_Dynamics(Dynamics):
    def __init__(self, parameters, mesh, spaces, vars, hamiltonian, forcing_terms, logger):
        self.mesh = mesh
        self.spaces = spaces
        self.variableset = vars
        self.hamiltonian = hamiltonian
        self.logger = logger
        self.forcing_terms = forcing_terms


        self.density_names = self.variableset.active_density_names
        self.dim = parameters['mesh']['dim']
        self.total_density_func = self.variableset.get_total_density_expr
        self.use_split_form = parameters['spatial-discretization']['use_split_form']

        self.coeffspacelist = []
        self.coefflist = []
        for term in self.forcing_terms:
            if term.has_coeff():
                name, space = term.get_coeff()
                self.coeffspacelist.append(space)
                self.coefflist.append(name)

        self.xspacelist = []
        self.xvarlist = []
        self.auxspacelist = []
        self.auxvarlist = []

        self.xvarlist += self.variableset.get_varlist()
        self.xspacelist += self.variableset.get_spacelist()
        for term in self.forcing_terms:
            self.auxvarlist += term.get_aux_vars_list()
            self.auxspacelist += term.get_spacelist()

        self.fullspacelist = self.xspacelist + self.auxspacelist
        self.fullvarlist = self.xvarlist + self.auxvarlist

        self.has_aux = not (len(self.auxspacelist) == 0)
        self.has_coeff = not (len(self.coeffspacelist) == 0)

        if not spaces is None:
            self.coriolis = Function(spaces.CG)
            self.bottom_topography = Function(spaces.CG)
            #dimension = self.spaces.CG.V.mesh().geometric_dimension()

            self.dx = spaces.dx

            self.xspace = MixedFunctionSpace(self.xspacelist)
            if self.has_coeff:
                self.coeffspace = MixedFunctionSpace(self.coeffspacelist)
            else:
                self.coeffspace = None
            if self.has_aux:
                self.auxspace = MixedFunctionSpace(self.auxspacelist)
            else:
                self.auxspace = None
            self.fullspace = MixedFunctionSpace(self.fullspacelist)

    def initialize(self, x, varexpr):
        self.variableset.initialize(varexpr, x)
        for term in self.forcing_terms:
            term.initialize(varexpr)
        self.coriolis.interpolate(varexpr['coriolis'])
        self.bottom_topography.interpolate(varexpr['bottom_topography'])

    def get_aux_var_list(self, terms='all'):
        aux_var_list = []
        for term in self.forcing_terms:
            if terms == 'all' or term.name in terms:
                aux_var_list = aux_var_list + term.get_aux_vars_list()
        return aux_var_list

    def compute_aux_expressions(self, x, t, coeff, xhats, terms='all'):
        expressions = {}
        for term in self.forcing_terms:
            if terms == 'all' or term.name in terms:
                term.compute_aux_expressions(x, t, coeff, xhats, expressions)
        return expressions




    def _rhs(self, xvars, t, coeff, xhats):

        v = xvars['v']
        total_dens = self.total_density_func(xvars)
        vtest = xhats['v']

        dfdx_expressions = {}
        self.hamiltonian.compute_dfdx_expressions(xvars, t, coeff, dfdx_expressions)
        F = dfdx_expressions['F'][0]

        if self.dim == 1:
            raise NotImplementedError
        elif self.dim == 2:
            rhs_expr = inner(vtest, self.coriolis / total_dens * rot2D(F))*self.dx
            rhs_expr = rhs_expr + inner(vtest, curl2D(v) / total_dens * rot2D(F))*self.dx
        elif self.dim == 3:
            raise NotImplementedError

        for dens_name in self.density_names:
            denstest = xhats[dens_name]
            Bdens = dfdx_expressions['B_' + dens_name][0]
            dens = xvars[dens_name]
            if self.use_split_form[dens_name]:
                rhs_expr = rhs_expr + inner(denstest, 0.5 *( div(dens/total_dens*F) + dot(grad(dens/total_dens),F) + dens/total_dens * div(F)))*self.dx
                rhs_expr = rhs_expr + inner(vtest, 0.5 *( dens/total_dens*grad(Bdens) + grad(Bdens*dens/total_dens)))*self.dx + 0.5 * inner(dens/total_dens, div(vtest*Bdens))*self.dx
            else:
                rhs_expr = rhs_expr + inner(denstest, div(dens/total_dens*F))*self.dx
                rhs_expr = rhs_expr + inner(vtest, dens/total_dens*grad(Bdens))*self.dx

        return rhs_expr

    def rhs(self, xvars, t, coeff, xhats, terms='all'):
        self.logger.output('computing rhs', 1)
        rhs = 0
        if terms == 'all' or 'model' in terms:
            rhs = rhs + self._rhs(xvars, t, coeff, xhats)
        for term in self.forcing_terms:
            if terms == 'all' or term.name in terms:
                rhs = rhs + term.rhs(xvars, t, coeff, xhats)
        return rhs
        self.logger.output('computing rhs', 1)

    def linear_rhs(self, const_state, xstar, xhats, terms='all'):
        raise NotImplementedError

    def post_step(self, statevars, terms='all'):
        for term in self.forcing_terms:
            if terms == 'all' or term.name in terms:
                term.post_step(statevars)





# class MetriplecticDynamics(Dynamics):
#     def __init__(self, mesh, spaces, vars, poisson_brackets, metric_brackets,
#         hamiltonian, entropy, statistics, diagnostics, forcing_terms, logger):
#         self.mesh = mesh
#         self.spaces = spaces
#         self.variableset = vars
#         self.poisson_brackets = poisson_brackets
#         self.metric_brackets = metric_brackets
#         self.hamiltonian = hamiltonian
#         self.entropy = entropy
#         self.diagnostics = diagnostics
#         self.statistics = statistics
#         self.logger = logger
#         self.forcing_terms = forcing_terms
#
# #THIS IS A HORRIBLE HACK FOR MAXWELL
#     def get_max_wavespeed(self):
#         return sp.constants.c
#
#     def initialize(self, vars, t):
#         t.assign(self.initcond.get_t0())
#         varexpr = self.initcond.get_value(self.mesh, t)
#         self.variableset.initialize(varexpr, vars)
#         for bracket in self.poisson_brackets:
#             bracket.initialize(varexpr)
#         for metric_bracket in self.metric_brackets:
#             metric_bracket.initialize(varexpr)
#         for term in self.forcing_terms:
#             term.initialize(varexpr)
#         self.diagnostics.initialize(varexpr)
#         self.statistics.initialize(varexpr)
#
#     def get_q_aux_var_list(self, terms='all'):
#         q_aux_var_list = []
#         if terms == 'all' or 'model' in terms:
#             for bracket in self.poisson_brackets:
#                 q_aux_var_list = q_aux_var_list + bracket.get_aux_vars_list()
#             for metric_bracket in self.metric_brackets:
#                 q_aux_var_list = q_aux_var_list + metric_bracket.get_aux_vars_list()
#         for term in self.forcing_terms:
#             if terms == 'all' or term.name in terms:
#                 q_aux_var_list = q_aux_var_list + term.get_aux_vars_list()
#         return q_aux_var_list
#
#     def get_dfdx_aux_var_list(self, terms='all'):
#         dfdx_aux_var_list = []
#         if terms == 'all' or 'model' in terms:
#             dfdx_aux_var_list = dfdx_aux_var_list + self.hamiltonian.get_aux_vars_list()
#             dfdx_aux_var_list = dfdx_aux_var_list + self.entropy.get_aux_vars_list()
#         return dfdx_aux_var_list
#
#
#     def get_q_aux_vars(self, terms='all'):
#         self.logger.output('creating q aux vars', 1)
#         vars = {}
#         if terms == 'all' or 'model' in terms:
#             for bracket in self.poisson_brackets:
#                 bracket.get_aux_vars(vars)
#             for metric_bracket in self.metric_brackets:
#                 metric_bracket.get_aux_vars(vars)
#         for term in self.forcing_terms:
#             if terms == 'all' or term.name in terms:
#                 term.get_aux_vars(vars)
#         self.logger.output('created q aux vars', 1)
#         return vars
#
#     def get_dfdx_aux_vars(self, terms='all'):
#         self.logger.output('creating dfdx aux vars', 1)
#         vars = {}
#         if terms == 'all' or 'model' in terms:
#             self.hamiltonian.get_aux_vars(vars)
#             self.entropy.get_aux_vars(vars)
#         self.logger.output('created dfdx aux vars', 1)
#         return vars
#
#     def compute_dfdx_expressions(self, x, terms='all'):
#         expressions = {}
#         if terms == 'all' or 'model' in terms:
#             self.hamiltonian.compute_dfdx_expressions(x, expressions)
#             self.entropy.compute_dfdx_expressions(x, expressions)
#         return expressions
#
#     def compute_q_expressions(self, x, terms='all'):
#         expressions = {}
#         if terms == 'all' or 'model' in terms:
#             for bracket in self.poisson_brackets:
#                 bracket.compute_q_expressions(x, expressions)
#             for metric_bracket in self.metric_brackets:
#                 metric_bracket.compute_q_expressions(x, expressions)
#         for term in self.forcing_terms:
#             if terms == 'all' or term.name in terms:
#                 term.compute_q_expressions(x, expressions)
#         return expressions
#
#
# #MIGHT NEED SOME GENERALIZATION FOR HIGHER ORDER EC INTEGRATORS?
#
#     def rhs(self, qvars, dfdx_vars, xhats, terms='all'):
#         self.logger.output('computing rhs', 1)
#         rhs = 0
#         if terms == 'all' or 'model' in terms:
#             for bracket in self.poisson_brackets:
#                 rhs = rhs + bracket.rhs(qvars, dfdx_vars, xhats)
#             for metric_bracket in self.metric_brackets:
#                 rhs = rhs + metric_bracket.rhs(qvars, dfdx_vars, xhats)
#         for term in self.forcing_terms:
#             if terms == 'all' or term.name in terms:
#                 rhs = rhs + term.rhs(qvars, xhats)
#         return rhs
#         self.logger.output('computing rhs', 1)
#
#     def linear_rhs(self, const_state, xstar, xhats, terms='all'):
#         self.logger.output('computing linear rhs', 1)
#         linear_rhs = 0
#         if terms == 'all' or 'model' in terms:
#             dfdx_linear_vars = {}
#             self.hamiltonian.compute_dfdx_linear(const_state, xstar, dfdx_linear_vars)
#             self.entropy.compute_dfdx_linear(const_state, xstar, dfdx_linear_vars)
#             for bracket in self.poisson_brackets:
#                 linear_rhs = linear_rhs + bracket.linear_rhs(const_state, dfdx_linear_vars, xhats)
#             for metric_bracket in self.metric_brackets:
#                 linear_rhs = linear_rhs + metric_bracket.linear_rhs(const_state, dfdx_linear_vars, xhats)
# #NOT SURE EXACTLY HOW TO HANDLE THIS YET
#         for term in self.forcing_terms:
#             if terms == 'all' or term.name in terms:
#                 linear_rhs = linear_rhs + term.linear_rhs(const_state, xstar, xhats)
#         self.logger.output('computed linear rhs', 1)
#         return linear_rhs
#
#     def post_step(self, statevars, terms='all'):
#         if terms == 'all' or 'model' in terms:
#             for bracket in self.poisson_brackets:
#                 bracket.post_step(statevars)
#             for metric_bracket in self.metric_brackets:
#                 bracket.post_step(statevars)
#         for term in self.forcing_terms:
#             if terms == 'all' or term.name in terms:
#                 term.post_step(statevars)
#
# def get_forcing_terms(parameters, vars, spaces, initcond):
#     forcing_terms = []
#     for forcing_term in parameters['model']['forcing_terms']:
#         if forcing_term == 'hyperviscosity':
#             forcing_terms.append(Hyperviscosity(parameters, vars, spaces))
#         elif forcing_term == 'threewayphysics':
#             forcing_terms.append(ThreeWayPhysics(parameters, vars, spaces, initcond))
#         elif forcing_term == 'dg1limiter':
#             forcing_terms.append(DG1LimiterTransport(parameters, vars, spaces))
#         elif forcing_term == 'cgtransport':
#             forcing_terms.append(CGTransport(parameters, vars, spaces))
#         else:
#             raise ValueError("forcing term " + forcing_term + " is unknown")
#     return forcing_terms
#
#
# def get_dynamics(parameters, mesh, spaces, logger, initcond):
#
#     if parameters['model']['class'] == 'metriplectic':
#         if parameters['model']['type'] in ['tswe-lp', 'ce-lp', 'mtswe-lp']:
#             if parameters['model']['type'] == 'tswe-lp':
#                 vars = ThermalShallowWaterVariables_LP(spaces, parameters['model']['tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_LP(vars)
#             if parameters['model']['type'] == 'mtswe-lp':
#                 vars = MoistThermalShallowWaterVariables_LP(spaces, parameters['model']['tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_LP(vars)
#             if parameters['model']['type'] == 'ce-lp':
#                 vars = CompressibleEulerVariables_LP(spaces, parameters['model']['tracer_names'])
#                 thermo = get_thermo(parameters['model']['thermo'])
#                 hamiltonian = CompressibleEuler_Hamiltonian_LP(vars, thermo)
#                 initcond.set_thermo(hamiltonian.thermo)
#                 hamiltonian.thermo.set_thermo_const(initcond)
#             poisson_brackets = [LiePoisson_AdvectedDensities_Bracket(spaces, vars, parameters), ]
#             metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_LP(spaces, vars), ]
#             entropy = SimpleEntropy(spaces, vars)
#             statistics = AdvDensStatistics_LP(spaces, hamiltonian, vars, initcond, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = AdvDensDiagnostics_LP(spaces, hamiltonian, vars, initcond, parameters['mesh']['dim'])
#
#         elif parameters['model']['class'] in ['tswe-cf', 'ce-cf', 'mtswe-cf']:
#             if parameters['model'] == 'tswe-cf':
#                 vars = ThermalShallowWaterVariables_CF(spaces, parameters['model']['tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
#             if parameters['model']['type'] == 'mtswe-cf':
#                 vars = MoistThermalShallowWaterVariables_CF(spaces, parameters['model']['tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
#             if parameters['model']['type'] == 'ce-cf':
#                 vars = CompressibleEulerVariables_CF(spaces, parameters['model']['tracer_names'])
#                 thermo = get_thermo(parameters['model']['thermo'])
#                 hamiltonian = CompressibleEuler_Hamiltonian_CF(vars, thermo)
#                 initcond.set_thermo(hamiltonian.thermo)
#                 hamiltonian.thermo.set_thermo_const(initcond)
#             poisson_brackets = [CurlForm_AdvectedDensities_Bracket(spaces, vars, parameters), ]
#             metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_CF(spaces, vars), ]
#             entropy = SimpleEntropy(spaces, vars)
#             statistics = AdvDensStatistics_CF(spaces, hamiltonian, vars, initcond, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = AdvDensDiagnostics_CF(spaces, hamiltonian, vars, initcond, parameters['mesh']['dim'])
#
#         elif parameters['model']['type'] == 'mhd':
#             vars = MHDVariables_LP(spaces)
#             poisson_brackets = [LiePoisson_AdvectedDensities_Bracket(spaces, vars, parameters), MHDBracket_LP(spaces, vars, parameters)]
#             metric_brackets = [ThermodynamicallyCompatibleViscousRegularization(spaces, vars), ]
#             entropy = SimpleEntropy(spaces, vars)
#             hamiltonian = MHD_Hamiltonian_LP(vars)
#             statistics = MHDStatistics(spaces, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = MHDDiagnostics(spaces)
#             initcond.set_thermo(hamiltonian.thermo)
#             hamiltonian.thermo.set_thermo_const(initcond)
#
#         elif parameters['model']['type'] == 'maxwell':
#             vars = MaxwellVariables(spaces)
#             poisson_brackets = [MaxwellBracket(spaces),]
#             metric_brackets = []
#             entropy = SimpleEntropy(spaces, vars)
#             hamiltonian = Maxwell_Hamiltonian(vars)
#             statistics = MaxwellStatistics(spaces, hamiltonian, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = MaxwellDiagnostics(spaces)
#
#         elif parameters['model']['type'] == 'eulermaxwell':
#             vars = EulerMaxwellVariables_LP(spaces)
#             poisson_brackets = [LiePoisson_AdvectedDensities_Bracket(spaces, vars, parameters), MaxwellBracket(spaces, vars, parameters), EulerMaxwellCouplingBracket_LP(spaces, vars, parameters)]
#             metric_brackets = [ThermodynamicallyCompatibleViscousRegularization(spaces, vars), ]
#             entropy = SimpleEntropy(spaces, vars)
#             hamiltonian = EulerMaxwell_Hamiltonian_LP(spaces)
#             statistics = EulerMaxwellStatistics(spaces, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = EulerMaxwellDiagnostics(spaces)
#             initcond.set_thermo(hamiltonian.thermo)
#             hamiltonian.thermo.set_thermo_const(initcond)
#
#         elif parameters['model']['type'] == 'scalarwave':
#             vars = ScalarWaveVariables(spaces)
#             poisson_brackets = [ScalarWaveBracket(spaces),]
#             metric_brackets = []
#             entropy = None
#             hamiltonian = ScalarWave_Hamiltonian(spaces)
#             statistics = ScalarWaveStatistics(spaces, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = ScalarWaveDiagnostics(spaces)
#
#         else:
#             raise ValueError("model " + parameters['model']['type'] + " is unknown")
#
#         forcing_terms = get_forcing_terms(parameters, vars, spaces, initcond)
#         return MetriplecticDynamics(mesh, spaces, vars, poisson_brackets, metric_brackets,
#             hamiltonian, entropy, statistics, diagnostics, forcing_terms, logger)
#
#     elif parameters['model']['class'] == 'advdens-cf-h1':
#
#         if parameters['model']['type'] in ['tswe-cf-h1', 'mtswe-cf-h1']:
#             if parameters['model']['type'] == 'tswe-cf-h1':
#                 vars = ThermalShallowWaterVariables_CF_H1(spaces, parameters['model']['tracer_names'], parameters['model']['dg_tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
#             if parameters['model']['type'] == 'mtswe-cf-h1':
#                 vars = MoistThermalShallowWaterVariables_CF_H1(spaces, parameters['model']['tracer_names'], parameters['model']['dg_tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
#             #poisson_brackets = [CurlForm_AdvectedDensities_Bracket_H1(spaces, vars, parameters), ]
#     #DOES METRIC BRACKET HAVE TO CHANGE FOR H1 SPACES?
#             #metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_CF(spaces, vars), ]
#             entropy = SimpleEntropy(spaces, vars)
#             statistics = AdvDensStatistics_CF_H1(spaces, hamiltonian, vars, initcond, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
#             diagnostics = AdvDensDiagnostics_CF_H1(spaces, hamiltonian, vars, initcond, parameters['mesh']['dim'])
#
#         else:
#             raise ValueError("model " + parameters['model']['type'] + " is unknown")
#
#         forcing_terms = get_forcing_terms(parameters, vars, spaces, initcond)
#         return AdvDensCF_H1_Dynamics(parameters, mesh, spaces, vars, hamiltonian, entropy, statistics, diagnostics, forcing_terms, logger, initcond)
#
#     elif parameters['modeltype']['class'] == 'advection':
#         raise NotImplementedError
#
#     else:
#         raise ValueError("modeltype " + parameters['modeltype']['class'] + " is unknown")
