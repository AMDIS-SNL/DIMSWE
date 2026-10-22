from firedrake import Function, MixedFunctionSpace, TestFunction, TestFunctions, TrialFunction
from firedrake import project, FunctionSpace


def get_diff_form_space(spaces, bundle, degree, dim, nm1):
    if bundle == 'S':
        if degree == 0:
            return spaces.CG
        elif degree == dim:
            return spaces.DG
        elif degree == 1 and dim == 2 and nm1:
            return spaces.Hdiv
        elif degree == 1 and dim == 2 and not nm1:
            return spaces.Hcurl
        elif degree == 1 and dim == 3:
            return spaces.Hcurl
        elif degree == 2 and dim == 3:
            return spaces.Hdiv
    elif bundle in ['VV', 'CV']:
        if degree == 0:
            return spaces.CGV
        elif degree == dim:
            return spaces.DGV
        elif degree == 1 and dim == 2 and nm1:
            return spaces.HdivV
        elif degree == 1 and dim == 2 and not nm1:
            return spaces.HcurlV
        elif degree == 1 and dim == 3:
            return spaces.HcurlV
        elif degree == 2 and dim == 3:
            return spaces.HdivV
    else:
        raise NotImplementedError('bundle ' + bundle + ' is not implemented')

class VariablesBase():

    def get_varlist(self):
        return self.varlist

    def get_spacelist(self):
        return self.spacelist

#THIS SHOULD BE DONE VIA A FIXED LINEAR SYSTEM I THINK!
    def initialize(self, varexpr, vars):
        for var in self.varlist:
            if not varexpr[var] == 0:
                vars[var].project(varexpr[var])

    def get_dhdx_var_list(self):
        return self.dhdx_var_list




class DiffFormVariablesBase(VariablesBase):
    def __init__(self, spaces, variablelist, bundlelist, degreelist, dim):
        self.spaces = spaces
        self.varlist = variablelist
        self.bundlelist = bundlelist
        self.degreelist = degreelist
        self.dim = dim

        if not spaces is None:
            self.spacelist = []
            for bundle, degree in zip(self.bundlelist, self.degreelist):
#NEED TO HANDLE 1-FORMS VS N-1 FORMS in 2D PROPERLY!
#RIGHT NOW IT DEFAULTS TO HDIV IE N-1 FORMS
                self.spacelist.append(get_diff_form_space(spaces, bundle, degree, dim, True))


class LPVariablesBase(DiffFormVariablesBase):
    def __init__(self, spaces, dim,
        advected_quantity_names, advected_quantity_bundles, advected_quantity_degrees,
        advected_quantity_dhdx_var_list, advected_quantity_dsdx_var_list,
        inactive_advected_quantity_names, inactive_advected_quantity_bundles, inactive_advected_quantity_degrees):
        variablelist = ['m',] + advected_quantity_names + inactive_advected_quantity_names
        bundlelist = ['CV',] + advected_quantity_bundles + inactive_advected_quantity_bundles
        degreelist = [dim,] + advected_quantity_degrees + inactive_advected_quantity_degrees
        self.advected_quantity_names = advected_quantity_names
        self.advected_quantity_bundles = advected_quantity_bundles
        self.advected_quantity_degrees = advected_quantity_degrees
        self.dhdx_var_list = ['u'] + advected_quantity_dhdx_var_list
        self.dsdx_var_list = ['dSdm'] + advected_quantity_dsdx_var_list
        self.inactive_advected_quantity_names = inactive_advected_quantity_names
        self.inactive_advected_quantity_bundles = inactive_advected_quantity_bundles
        self.inactive_advected_quantity_degrees = inactive_advected_quantity_degrees
        DiffFormVariablesBase.__init__(self, spaces, variablelist, bundlelist, degreelist, dim)

class CFVariablesBase(DiffFormVariablesBase):
    def __init__(self, spaces, dim,
        advected_quantity_names, advected_quantity_bundles, advected_quantity_degrees,
        advected_quantity_dhdx_var_list, advected_quantity_dsdx_var_list,
        inactive_advected_quantity_names, inactive_advected_quantity_bundles, inactive_advected_quantity_degrees):
        variablelist = ['v',] + advected_quantity_names + inactive_advected_quantity_names
        bundlelist = ['S',] + advected_quantity_bundles + inactive_advected_quantity_bundles
#NEED TO HANDLE PROPERLY 1-FORMS VS N-1 FORMS in 2D!
#THE DEFAULT WORKS HERE, BUT IT SHOULD BE DONE PROPERLY
        degreelist = [dim-1,] + advected_quantity_degrees + inactive_advected_quantity_degrees
        self.advected_quantity_names = advected_quantity_names
        self.advected_quantity_bundles = advected_quantity_bundles
        self.advected_quantity_degrees = advected_quantity_degrees
        self.dhdx_var_list = ['F'] + advected_quantity_dhdx_var_list
        self.dsdx_var_list = ['dSdv'] + advected_quantity_dsdx_var_list
        self.inactive_advected_quantity_names = inactive_advected_quantity_names
        self.inactive_advected_quantity_bundles = inactive_advected_quantity_bundles
        self.inactive_advected_quantity_degrees = inactive_advected_quantity_degrees
        DiffFormVariablesBase.__init__(self, spaces, variablelist, bundlelist, degreelist, dim)



class AdvDensVariables_CF(CFVariablesBase):
     def __init__(self, spaces, active_density_names, inactive_density_names, dim):
         self.density_names = active_density_names + inactive_density_names

         advected_quantity_bundles = ['S',] * len(active_density_names)
         advected_quantity_degrees = [dim, ] * len(active_density_names)
         inactive_advected_quantity_bundles = ['S',] * len(inactive_density_names)
         inactive_advected_quantity_degrees = [dim, ] * len(inactive_density_names)
         advected_quantity_dhdx_var_list = ['B_' + dens for dens in active_density_names]
         advected_quantity_dsdx_var_list = ['dSd' + dens for dens in active_density_names]

         CFVariablesBase.__init__(self, spaces, dim,
         active_density_names, advected_quantity_bundles, advected_quantity_degrees, advected_quantity_dhdx_var_list, advected_quantity_dsdx_var_list,
         inactive_density_names, inactive_advected_quantity_bundles, inactive_advected_quantity_degrees)

class AdvDensVariables_LP(LPVariablesBase):
     def __init__(self, spaces, active_density_names, inactive_density_names, dim):
         self.density_names = active_density_names + inactive_density_names

         advected_quantity_bundles = ['S',] * len(active_density_names)
         advected_quantity_degrees = [dim, ] * len(active_density_names)
         inactive_advected_quantity_bundles = ['S',] * len(inactive_density_names)
         inactive_advected_quantity_degrees = [dim, ] * len(inactive_density_names)
         advected_quantity_dhdx_var_list = ['B_' + dens for dens in active_density_names]
         advected_quantity_dsdx_var_list = ['dSd' + dens for dens in active_density_names]

         LPVariablesBase.__init__(self, spaces, dim,
         active_density_names, advected_quantity_bundles, advected_quantity_degrees, advected_quantity_dhdx_var_list, advected_quantity_dsdx_var_list,
         inactive_density_names, inactive_advected_quantity_bundles, inactive_advected_quantity_degrees)


#THIS IS WHAT WE NEED TO DO SOON-ISH

#DO LOTS OF MERGING HERE
#basically we have CF, LP and CF-H1 models, each with some set of advected quantities
#with a base differential forms variables class that assigns correct spaces to forms of a given bundle and degree, based on dimension
#for CF-H1 we can "ignore" the form degree and just treat "everything" (maybe modulo some DG advection stuff?) as a 0-form!

#ie variables are just a set of differential forms
#then add some specialized logic for various examples ie TSWE, mTSWE, CE, MCE, MHD, Euler-Maxwell, Maxwell!
#YES THIS IS THE WAY..





# class AdvectionVariables(VariablesBase):
#     def __init__(self, spaces, SOMETHING):
#
#         if not spaces is None:
#             self.spacelist = []
#             self.spacelist.append(SOMETHING)
#
# #CF/CF-H1: v, dens
# #LP: m, dens
# #lots of upwinding choices, also split form
# #for v, have q-form, centered and upwinded
# #eventually add in positive-definite filters, etc.
#
#     def initialize(self, varexpr, vars):
#         SOMETHING
#         #if not varexpr['v']==0:
#         #    vars.sub(0).project(varexpr['v'])
#         #for i,dens in enumerate(self.density_names):
#         #    vars.sub(i+1).interpolate(varexpr[dens])
#


class AdvDensVariables_CF_H1(VariablesBase):
    def __init__(self, spaces, active_density_names, cg_inactive_density_names, dg_inactive_density_names, dim):
        self.spaces = spaces
        self.density_names = active_density_names + cg_inactive_density_names + dg_inactive_density_names
        self.active_density_names = active_density_names
        self.inactive_density_names = cg_inactive_density_names + dg_inactive_density_names
        self.dg_inactive_density_names = dg_inactive_density_names
        self.cg_inactive_density_names = cg_inactive_density_names

        if not spaces is None:
            self.spacelist = [self.spaces.CGV, ]
            for dens in self.active_density_names + self.cg_inactive_density_names:
                self.spacelist.append(self.spaces.CG)
            for dens in self.dg_inactive_density_names:
#EVENTUALLY MAKE THIS TUNABLE, IFF THE SLOPE LIMITER EVER GETS GENERALIZED...
                self.spacelist.append(FunctionSpace(self.spaces.mesh, 'DG', 1, variant="spectral"))

        self.varlist = ['v',]
        for dens in self.density_names:
            self.varlist.append(dens)

        self.dhdx_var_list = ['F',]
        for dens in self.active_density_names:
            self.dhdx_var_list.append('B_' + dens)




# class AdvDensVariables_CF_Base(VariablesBase):
#     def __init__(self, spaces, active_density_names, inactive_density_names, dim):
#         self.spaces = spaces
#         self.density_names = active_density_names + inactive_density_names
#         self.active_density_names = active_density_names
#         self.inactive_density_names = inactive_density_names
#
#         self.advected_quantity_names = self.active_density_names
#         self.advected_quantity_bundle = ['S',] * len(self.active_density_names)
#         self.advected_quantity_degree = [dim, ] * len(self.active_density_names)
#         self.inactive_advected_quantity_names = self.inactive_density_names
#         self.inactive_advected_quantity_bundle = ['S',] * len(self.inactive_density_names)
#         self.inactive_advected_quantity_degree = [dim, ] * len(self.inactive_density_names)
#
#         self.varlist = ['v',]
#         for dens in self.density_names:
#             self.varlist.append(dens)
#
#         self.dhdx_var_list = ['F',]
#         for dens in self.active_density_names:
#             self.dhdx_var_list.append('B_' + dens)




# class AdvDensVariables_CF(AdvDensVariables_CF_Base):
#     def __init__(self, spaces, active_density_names, inactive_density_names, dg_inactive_density_names, dim):
#         AdvDensVariables_CF_Base.__init__(self, spaces, active_density_names, inactive_density_names + dg_inactive_density_names, dim)
#
#         self.dg_inactive_density_names = dg_inactive_density_names
#
#         if not spaces is None:
#             self.spacelist = [self.spaces.Hdiv, ]
#             for dens in self.active_density_names + self.inactive_density_names:
#                 self.spacelist.append(self.spaces.DG)
#             for dens in self.dg_inactive_density_names:
# #EVENTUALLY MAKE THIS TUNABLE, IFF THE SLOPE LIMITER EVER GETS GENERALIZED...
#                 self.spacelist.append(FunctionSpace(self.spaces.mesh, 'DG', 1, variant="spectral"))
#
#
# class AdvDensVariables_LP(VariablesBase):
#     def __init__(self, spaces, active_density_names, inactive_density_names, dg_inactive_density_names, dim):
#         self.spaces = spaces
#
#         self.density_names = active_density_names + inactive_density_names + dg_inactive_density_names
#         self.active_density_names = active_density_names
#         self.inactive_density_names = inactive_density_names
#         self.dg_inactive_density_names = dg_inactive_density_names
#         self.dim = dim
#
#         self.advected_quantity_names = self.active_density_names
#         self.advected_quantity_bundle = ['S',] * len(self.active_density_names)
#         self.advected_quantity_degree = [dim, ] * len(self.active_density_names)
#         self.inactive_advected_quantity_names = self.inactive_density_names
#         self.inactive_advected_quantity_bundle = ['S',] * len(self.inactive_density_names)
#         self.inactive_advected_quantity_degree = [dim, ] * len(self.inactive_density_names)
#
#         self.varlist = ['m',]
#         for dens in self.density_names:
#             self.varlist.append(dens)
#
#         self.dhdx_var_list = ['u',]
#         for dens in self.active_density_names:
#             self.dhdx_var_list.append('B_' + dens)
#
#         if not spaces is None:
#
#             self.spacelist = [self.spaces.DGV, ]
#             for dens in self.active_density_names + self.inactive_density_names:
#                 self.spacelist.append(self.spaces.DG)
#             for dens in self.dg_inactive_density_names:
# #EVENTUALLY MAKE THIS TUNABLE, IFF THE SLOPE LIMITER EVER GETS GENERALIZED...
#                 self.spacelist.append(FunctionSpace(self.spaces.mesh, 'DG', 1, variant="spectral"))

###################


class ThermalShallowWaterBase():

    def get_total_density_expr(self, vars):
        return vars['h']


class ThermalShallowWaterVariables_CF(AdvDensVariables_CF, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_CF.__init__(self, spaces, ['h', 'S'], tracer_names, 2)
        self.entropy_name = 'S'


class ThermalShallowWaterVariables_CF_H1(AdvDensVariables_CF_H1, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names, dg_tracer_names=[]):
        AdvDensVariables_CF_H1.__init__(self, spaces, ['h', 'S'], tracer_names, dg_tracer_names, 2)
        self.entropy_name = 'S'

class ThermalShallowWaterVariables_LP(AdvDensVariables_LP, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_LP.__init__(self, spaces, ['h', 'S'], tracer_names, 2)
        self.entropy_name = 'S'


class MoistThermalShallowWaterVariables_CF(AdvDensVariables_CF, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names, dg_tracer_names=[]):
        AdvDensVariables_CF.__init__(self, spaces, ['h', 'S'], tracer_names + ['Qv', 'Qc', 'Qr'], 2)
        self.entropy_name = 'S'


class MoistThermalShallowWaterVariables_CF_H1(AdvDensVariables_CF_H1, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names, dg_tracer_names=[]):
        AdvDensVariables_CF_H1.__init__(self, spaces, ['h', 'S'], tracer_names, ['Qv', 'Qc', 'Qr'] + dg_tracer_names, 2)
        self.entropy_name = 'S'

class MoistThermalShallowWaterVariables_LP(AdvDensVariables_LP, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names, dg_tracer_names=[]):
        AdvDensVariables_LP.__init__(self, spaces, ['h', 'S'], tracer_names + ['Qv', 'Qc', 'Qr'], 2)
        self.entropy_name = 'S'


#
#
# class CompressibleEulerBase():
#
#     def get_total_density_expr(self, vars):
#         return vars['rho']
#
# class CompressibleEulerVariables_CF(AdvDensVariables_CF, CompressibleEulerBase):
#     def __init__(self, spaces, tracer_names):
#         AdvDensVariables_CF.__init__(self, spaces, ['rho', 'S'], tracer_names)
#         self.entropy_name = 'S'
#
# class CompressibleEulerVariables_LP(AdvDensVariables_LP, CompressibleEulerBase):
#     def __init__(self, spaces, tracer_names):
#         AdvDensVariables_LP.__init__(self, spaces, ['rho', 'S'], tracer_names)
#         self.entropy_name = 'S'
#
#
#
#
# class MHDVariables_LP(VariablesBase):
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.varlist = ['m', 'rho', 'S', 'B']
#         self.dhdx_var_list = ['u', 'B_rho', 'B_S', 'H']
#
#         if not spaces is None:
#             self.spacelist = [self.spaces.DGV, self.spaces.DG, self.spaces.DG, self.spaces.Hdiv]
#             self.mixedspace = MixedFunctionSpace(self.spacelist)
#
# class MaxwellVariables(VariablesBase):
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.varlist = ['B', 'D']
#         self.dhdx_var_list = ['H', 'E']
#         self.entropy_name = None
#         if not spaces is None:
#             self.spacelist = [self.spaces.Hdiv, self.spaces.Hcurl]
#             self.mixedspace = MixedFunctionSpace(self.spacelist)
#
#     def initialize(self, varexpr, vars):
#         for i,var in enumerate(self.varlist):
#             if not varexpr[var]==0:
#                 vars.sub(i).project(varexpr[var])
#
# class EulerMaxwellVariables_LP(VariablesBase):
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.varlist = ['m', 'rho', 'S', 'B', 'D']
#         self.dhdx_var_list = ['u', 'B_rho', 'B_S', 'H', 'E']
#
#         if not spaces is None:
#             self.spacelist = [self.spaces.DGV, self.spaces.DG, self.spaces.DG, self.spaces.Hdiv, self.spaces.Hcurl]
#             self.mixedspace = MixedFunctionSpace(self.spacelist)
#
# class ScalarWaveVariables(VariablesBase):
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.varlist = ['h', 'v']
#         self.dhdx_var_list = ['B', 'F']
#
#         if not spaces is None:
#             self.spacelist = [self.spaces.DG, self.spaces.Hdiv]
#             self.mixedspace = MixedFunctionSpace(self.spacelist)
