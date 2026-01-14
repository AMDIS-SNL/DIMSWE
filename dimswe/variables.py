from firedrake import (
    Function,
    FunctionSpace,
    MixedFunctionSpace,
    TestFunction,
    TestFunctions,
    TrialFunction,
)


class VariablesBase:

    def get_vars(self, varname):
        return Function(self.mixedspace, name=varname)

    def get_test_var(self):
        return TestFunction(self.mixedspace)

    def get_trial_var(self):
        return TrialFunction(self.mixedspace)

    def get_test_vars(self):
        xhats = TestFunctions(self.mixedspace)
        xhat_subs = {}
        for i, var in enumerate(self.varlist):
            xhat_subs[var] = xhats[i]
        return xhat_subs


def get_space(spaces, bundle, degree, dim, nm1):
    if bundle == "S":
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
    if bundle in ["VV", "CV"]:
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


class DiffFormVariablesBase(VariablesBase):
    def __init__(self, spaces, variablelist, bundlelist, degreelist, dim):
        self.spaces = spaces
        self.variablelist = variablelist
        self.bundlelist = bundlelist
        self.degreelist = degreelist
        self.dim = dim

        if not spaces is None:
            self.spacelist = []
            for _var, bundle, degree in zip(
                self.variablelist, self.bundlelist, self.degreelist, strict=False
            ):
                # NEED TO HANDLE 1-FORMS VS N-1 FORMS in 2D!
                self.spacelist.append(get_space(spaces, bundle, degree, dim))
            self.mixedspace = MixedFunctionSpace(self.spacelist)

    def initialize(self, varexpr, vars):
        for i, var in enumerate(self.variablelist):
            if not varexpr[var] == 0:
                vars.sub(i).project(varexpr[dens])


class LPVariablesBase(DiffFormVariablesBase):
    def __init__(
        self,
        spaces,
        advected_quantity_names,
        advected_quantity_bundles,
        advected_quantity_degrees,
        dim,
    ):
        variablelist = [
            "m",
        ] + advected_quantity_names
        bundlelist = [
            "CV",
        ] + advected_quantity_bundle
        degreelist = [
            dim,
        ] + advected_quantity_degrees
        DiffFormVariablesBase.__init__(
            self, spaces, variablelist, bundlelist, degreelist, dim
        )


class CFVariablesBase(DiffFormVariablesBase):
    def __init__(
        self,
        spaces,
        advected_quantity_names,
        advected_quantity_bundles,
        advected_quantity_degrees,
        dim,
    ):
        variablelist = [
            "v",
        ] + advected_quantity_names
        bundlelist = [
            "S",
        ] + advected_quantity_bundle
        # NEED TO HANDLE 1-FORMS VS N-1 FORMS in 2D!
        degreelist = [
            dim - 1,
        ] + advected_quantity_degrees
        DiffFormVariablesBase.__init__(
            self, spaces, variablelist, bundlelist, degreelist, dim
        )


class CFH1VariablesBase(DiffFormVariablesBase):
    def __init__(
        self,
        spaces,
        advected_quantity_names,
        advected_quantity_bundles,
        advected_quantity_degrees,
        dim,
    ):
        variablelist = [
            "v",
        ] + advected_quantity_names
        bundlelist = [
            "VV",
        ] + advected_quantity_bundle
        # BROKEN
        degreelist = [
            0,
        ] + advected_quantity_degrees
        DiffFormVariablesBase.__init__(
            self, spaces, variablelist, bundlelist, degreelist, dim
        )


# DO LOTS OF MERGING HERE
# basically we have CF, LP and CF-H1 models, each with some set of advected quantities
# with a base differential forms variables class that assigns correct spaces to forms of a given bundle and degree, based on dimension
# for CF-H1 we can "ignore" the form degree and just treat "everything" (maybe modulo some DG advection stuff?) as a 0-form!

# ie variables are just a set of differential forms
# then add some specialized logic for various examples ie TSWE, mTSWE, CE, MCE, MHD, Euler-Maxwell, Maxwell!
# YES THIS IS THE WAY..


class AdvectionVariables(VariablesBase):
    def __init__(self, spaces, SOMETHING):

        if not spaces is None:
            self.spacelist = []
            self.spacelist.append(SOMETHING)
            self.mixedspace = MixedFunctionSpace(self.spacelist)

    # CF/CF-H1: v, dens
    # LP: m, dens
    # lots of upwinding choices, also split form
    # for v, have q-form, centered and upwinded
    # eventually add in positive-definite filters, etc.

    def initialize(self, varexpr, vars):
        SOMETHING
        # if not varexpr['v']==0:
        #    vars.sub(0).project(varexpr['v'])
        # for i,dens in enumerate(self.density_names):
        #    vars.sub(i+1).interpolate(varexpr[dens])

    # NEED TO ADD DSDX VARLIST ALSO!!!
    def get_total_density_expr(self):
        return XXX


class AdvDensVariables_CF_Base(VariablesBase):
    def __init__(self, spaces, density_names):
        self.spaces = spaces
        self.density_names = density_names

        self.varlist = [
            "v",
        ]
        for dens in self.density_names:
            self.varlist.append(dens)

        self.dhdx_var_list = [
            "F",
        ]
        for dens in self.density_names:
            self.dhdx_var_list.append("B_" + dens)

    def initialize(self, varexpr, vars):
        if not varexpr["v"] == 0:
            vars.sub(0).project(varexpr["v"])
        for i, dens in enumerate(self.density_names):
            vars.sub(i + 1).interpolate(varexpr[dens])


class AdvDensVariables_CF_H1(AdvDensVariables_CF_Base):
    def __init__(self, spaces, density_names, dg_density_names):
        AdvDensVariables_CF_Base.__init__(self, spaces, density_names)

        self.dg_density_names = dg_density_names

        for dens in self.dg_density_names:
            self.varlist.append(dens)

        for dens in self.dg_density_names:
            self.dhdx_var_list.append("B_" + dens)

        if not spaces is None:
            self.spacelist = [
                self.spaces.CGV,
            ]
            for dens in self.density_names:
                self.spacelist.append(self.spaces.CG)
            for dens in self.dg_density_names:
                # EVENTUALLY MAKE THIS TUNABLE, IFF THE SLOPE LIMITER EVER GETS GENERALIZED...
                self.spacelist.append(FunctionSpace(self.spaces.mesh, "DG", 1))
            self.mixedspace = MixedFunctionSpace(self.spacelist)

    def initialize(self, varexpr, vars):
        AdvDensVariables_CF_Base.initialize(self, varexpr, vars)
        for i, dens in enumerate(self.dg_density_names):
            vars.sub(i + 1 + len(self.density_names)).interpolate(varexpr[dens])


class AdvDensVariables_CF(AdvDensVariables_CF_Base):
    def __init__(self, spaces, density_names):
        AdvDensVariables_CF_Base.__init__(self, spaces, density_names)

        if not spaces is None:
            self.spacelist = [
                self.spaces.Hdiv,
            ]
            for dens in self.density_names:
                self.spacelist.append(self.spaces.DG)
            self.mixedspace = MixedFunctionSpace(self.spacelist)


class AdvDensVariables_LP(VariablesBase):
    def __init__(self, spaces, density_names):
        self.spaces = spaces
        self.density_names = density_names

        self.varlist = [
            "m",
        ]
        for dens in self.density_names:
            self.varlist.append(dens)

        self.dhdx_var_list = [
            "u",
        ]
        for dens in self.density_names:
            self.dhdx_var_list.append("B_" + dens)

        if not spaces is None:

            self.spacelist = [
                self.spaces.DGV,
            ]
            for dens in self.density_names:
                self.spacelist.append(self.spaces.DG)
            self.mixedspace = MixedFunctionSpace(self.spacelist)

    def initialize(self, varexpr, vars):
        vars.sub(0).interpolate(varexpr["m"])
        for i, dens in enumerate(self.density_names):
            vars.sub(i + 1).interpolate(varexpr[dens])


###################


class ThermalShallowWaterBase:

    def get_total_density_expr(self, vars):
        return vars["h"]


class ThermalShallowWaterVariables_CF(AdvDensVariables_CF, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_CF.__init__(self, spaces, ["h", "S"] + tracer_names)
        self.entropy_name = "S"
        self.tracer_names = tracer_names


class ThermalShallowWaterVariables_CF_H1(
    AdvDensVariables_CF_H1, ThermalShallowWaterBase
):
    def __init__(self, spaces, tracer_names, dg_tracer_names):
        AdvDensVariables_CF_H1.__init__(
            self, spaces, ["h", "S"] + tracer_names, dg_tracer_names
        )
        self.entropy_name = "S"
        self.tracer_names = tracer_names
        self.dg_tracer_names = dg_tracer_names


class ThermalShallowWaterVariables_LP(AdvDensVariables_LP, ThermalShallowWaterBase):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_LP.__init__(self, spaces, ["h", "S"] + tracer_names)
        self.entropy_name = "S"
        self.tracer_names = tracer_names


class MoistThermalShallowWaterVariables_CF(
    AdvDensVariables_CF, ThermalShallowWaterBase
):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_CF.__init__(
            self, spaces, ["h", "S", "Qv", "Qc", "Qr"] + tracer_names
        )
        self.entropy_name = "S"
        self.tracer_names = tracer_names


class MoistThermalShallowWaterVariables_CF_H1(
    AdvDensVariables_CF_H1, ThermalShallowWaterBase
):
    def __init__(self, spaces, tracer_names, dg_tracer_names=[]):
        AdvDensVariables_CF_H1.__init__(
            self,
            spaces,
            ["h", "S"] + tracer_names,
            ["Qv", "Qc", "Qr"] + dg_tracer_names,
        )
        self.entropy_name = "S"
        self.tracer_names = tracer_names
        self.dg_tracer_names = dg_tracer_names


class MoistThermalShallowWaterVariables_LP(
    AdvDensVariables_LP, ThermalShallowWaterBase
):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_LP.__init__(
            self, spaces, ["h", "S", "Qv", "Qc", "Qr"] + tracer_names
        )
        self.entropy_name = "S"
        self.tracer_names = tracer_names


class CompressibleEulerBase:

    def get_total_density_expr(self, vars):
        return vars["rho"]


class CompressibleEulerVariables_CF(AdvDensVariables_CF, CompressibleEulerBase):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_CF.__init__(self, spaces, ["rho", "S"] + tracer_names)
        self.entropy_name = "S"
        self.tracer_names = tracer_names


class CompressibleEulerVariables_LP(AdvDensVariables_LP, CompressibleEulerBase):
    def __init__(self, spaces, tracer_names):
        AdvDensVariables_LP.__init__(self, spaces, ["rho", "S"] + tracer_names)
        self.entropy_name = "S"
        self.tracer_names = tracer_names


class MHDVariables_LP(VariablesBase):
    def __init__(self, spaces):
        self.spaces = spaces
        self.varlist = ["m", "rho", "S", "B"]
        self.dhdx_var_list = ["u", "B_rho", "B_S", "H"]

        if not spaces is None:
            self.spacelist = [
                self.spaces.DGV,
                self.spaces.DG,
                self.spaces.DG,
                self.spaces.Hdiv,
            ]
            self.mixedspace = MixedFunctionSpace(self.spacelist)


class MaxwellVariables(VariablesBase):
    def __init__(self, spaces):
        self.spaces = spaces
        self.varlist = ["B", "D"]
        self.dhdx_var_list = ["H", "E"]

        if not spaces is None:
            self.spacelist = [self.spaces.Hdiv, self.spaces.Hcurl]
            self.mixedspace = MixedFunctionSpace(self.spacelist)


class EulerMaxwellVariables_LP(VariablesBase):
    def __init__(self, spaces):
        self.spaces = spaces
        self.varlist = ["m", "rho", "S", "B", "D"]
        self.dhdx_var_list = ["u", "B_rho", "B_S", "H", "E"]

        if not spaces is None:
            self.spacelist = [
                self.spaces.DGV,
                self.spaces.DG,
                self.spaces.DG,
                self.spaces.Hdiv,
                self.spaces.Hcurl,
            ]
            self.mixedspace = MixedFunctionSpace(self.spacelist)


class ScalarWaveVariables(VariablesBase):
    def __init__(self, spaces):
        self.spaces = spaces
        self.varlist = ["h", "v"]
        self.dhdx_var_list = ["B", "F"]

        if not spaces is None:
            self.spacelist = [self.spaces.DG, self.spaces.Hdiv]
            self.mixedspace = MixedFunctionSpace(self.spacelist)
