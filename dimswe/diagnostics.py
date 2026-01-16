from firedrake import Function, LinearVariationalProblem, LinearVariationalSolver
from firedrake import FunctionSpace, TestFunction, TrialFunction, inner, curl
from .ufl_helpers import skewgrad, curl2D
from .parameters import overall_solver_parameters

#Advected density diagnostics

#nD
#specific densities ie density / total

#2D/3D: pv, eta, zeta, v = m / total_dens, m, u
#note that u in terms of m or v is Hamiltonian dependent, and also depends on LP vs CF

#hamiltonian and variableset specific: temperature, pressure, specific entropy, probably some magnetic ones

#SWAP TO THIS!
#class Diagnostic


#LOTS IS SHARED- COMBINE IT!
class AdvDensDiagnostics():
    def __init__(self, spaces, hamiltonian, variableset, dim):
        self.spaces = spaces
        self.hamiltonian = hamiltonian
        self.variableset = variableset
        self.density_names = variableset.density_names
        self.dim = dim

#THIS SHOULD COME FROM INITIAL CONDITION
#REALLY ACTUALLY THIS SHOULD SOME SORT OF CONSTANT FUNCTION CREATED BY INITIAL CONDITIONS
#THAT IS THEN SHARED ACROSS POISSON BRACKETS, ETC.
#SAME TREATMENT FOR HS, GEOPOTENTIAL, ETC.
        self.coriolis = 0.00006147

        self.testvars = {}
        self.trialvars = {}
        self.var_list = []
        if self.dim >= 2:
            self.var_list.append('q')
            self.var_list.append('eta')
            self.var_list.append('zeta')
        for dens in self.density_names:
            self.var_list.append(dens + '_l')

        if not self.spaces is None:
            self.dx = spaces.dx
            self.ds = spaces.ds

            self.vars = {}
            if self.dim ==2:
                self.vars['q'] = Function(self.spaces.CG)
                self.vars['eta'] = Function(self.spaces.CG)
                self.vars['zeta'] = Function(self.spaces.CG)
                self.testvars['q'] = TestFunction(self.spaces.CG)
                self.trialvars['q'] = TrialFunction(self.spaces.CG)
#THIS MIGHT BE bracket-specific, unclear...
            if self.dim ==3:
                self.vars['q'] = Function(self.spaces.Hcurl)
                self.vars['eta'] = Function(self.spaces.Hcurl)
                self.vars['zeta'] = Function(self.spaces.Hcurl)
                self.testvars['q'] = TestFunction(self.spaces.Hcurl)
                self.trialvars['q'] = TrialFunction(self.spaces.Hcurl)
            for dens in self.density_names:
                varspace = self.variableset.spacelist[self.variableset.varlist.index(dens)]
                self.vars[dens + '_l'] = Function(varspace)

#FOR PERFORMANCE SHOULD PROBABLY CREATE AN INTERPOLATOR AND CALL IT?
#ALSO UNCLEAR IF INTERPOLATE IS BEST, OR IF SOME SORT OF LINEAR SYTEM SHOULD BE USED?

    def compute(self):
        for dens in self.density_names:
            self.vars[dens + '_l'].interpolate(self.xn[dens] / self.total_dens)
        if self.dim >=2:
            self.pv_solver.solve()
            self.eta_solver.solve()
            self.zeta_solver.solve()

class AdvDensDiagnostics_CF_H1(AdvDensDiagnostics):
    # def __init__(self, spaces, hamiltonian, vars, dim):
    #     AdvDensDiagnostics.__init__(self, spaces, hamiltonian, vars, dim)
    #
    #     self.dg_density_names = vars.dg_density_names
    #     for dens in self.dg_density_names:
    #         self.var_list.append(dens + '_l')
    #
    #     if not self.spaces is None:
    #         for dens in self.dg_density_names:
    #             self.vars[dens + '_l'] = Function(FunctionSpace(self.spaces.mesh, 'DG', 1))
    #
    # def compute(self):
    #     AdvDensDiagnostics.compute(self)
    #     for dens in self.dg_density_names:
    #         self.vars[dens + '_l'].interpolate(self.xn[dens] / self.total_dens)

    def create(self, xn):
        self.xn = xn
        self.total_dens = self.hamiltonian.vars.get_total_density_expr(self.xn)

        v = xn['v']
        qhat = self.testvars['q']
        qtrial = self.trialvars['q']

        if self.dim == 2:
            pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(qhat, curl2D(v))*self.dx + inner(qhat, self.coriolis)*self.dx]
            eta_expr = [inner(qhat, qtrial)*self.dx, inner(qhat, curl2D(v))*self.dx + inner(qhat, self.coriolis)*self.dx]
            zeta_expr = [inner(qhat, qtrial)*self.dx, inner(qhat, curl2D(v))*self.dx]
        if self.dim == 3:
            pass
        if self.dim >=2:
            pv_problem = LinearVariationalProblem(pv_expr[0], pv_expr[1], self.vars['q'])
            eta_problem = LinearVariationalProblem(eta_expr[0], eta_expr[1], self.vars['eta'])
            zeta_problem = LinearVariationalProblem(zeta_expr[0], zeta_expr[1], self.vars['zeta'])
            self.pv_solver = LinearVariationalSolver(pv_problem, solver_parameters=overall_solver_parameters['qdiag'], options_prefix='qdiag')
            self.eta_solver = LinearVariationalSolver(eta_problem, solver_parameters=overall_solver_parameters['etadiag'], options_prefix='etadiag')
            self.zeta_solver = LinearVariationalSolver(zeta_problem, solver_parameters=overall_solver_parameters['zetadiag'], options_prefix='zetadiag')


class AdvDensDiagnostics_CF(AdvDensDiagnostics):

    def create(self, xn):
        self.xn = xn
        self.total_dens = self.hamiltonian.vars.get_total_density_expr(self.xn)

        v = xn['v']
        qhat = self.testvars['q']
        qtrial = self.trialvars['q']

#MISSING BOUNDARY TERMS...
        if self.dim == 2:
            pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            eta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx]
        if self.dim == 3:
            pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            eta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx]
        if self.dim >=2:
            pv_problem = LinearVariationalProblem(pv_expr[0], pv_expr[1], self.vars['q'])
            eta_problem = LinearVariationalProblem(eta_expr[0], eta_expr[1], self.vars['eta'])
            zeta_problem = LinearVariationalProblem(zeta_expr[0], zeta_expr[1], self.vars['zeta'])
            self.pv_solver = LinearVariationalSolver(pv_problem, solver_parameters=overall_solver_parameters['qdiag'], options_prefix='qdiag')
            self.eta_solver = LinearVariationalSolver(eta_problem, solver_parameters=overall_solver_parameters['etadiag'], options_prefix='etadiag')
            self.zeta_solver = LinearVariationalSolver(zeta_problem, solver_parameters=overall_solver_parameters['zetadiag'], options_prefix='zetadiag')



#ADD M, U, V
class AdvDensDiagnostics_LP(AdvDensDiagnostics):
    def create(self, xn):
        self.xn = xn
        total_dens = self.hamiltonian.vars.get_total_density_expr(self.xn)

        m = xn['m']
        v = m / self.total_dens
        qhat = self.testvars['q']
        qtrial = self.trialvars['q']


        if self.dim == 2:
            pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            eta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx]
        if self.dim == 3:
            pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            eta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
            zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx]
        if self.dim >=2:
            pv_problem = LinearVariationalProblem(pv_expr[0], pv_expr[1], self.vars['q'])
            eta_problem = LinearVariationalProblem(eta_expr[0], eta_expr[1], self.vars['eta'])
            zeta_problem = LinearVariationalProblem(zeta_expr[0], zeta_expr[1], self.vars['zeta'])
            self.pv_solver = LinearVariationalSolver(pv_problem, solver_parameters=overall_solver_parameters['qdiag'], options_prefix='qdiag')
            self.eta_solver = LinearVariationalSolver(eta_problem, solver_parameters=overall_solver_parameters['etadiag'], options_prefix='etadiag')
            self.zeta_solver = LinearVariationalSolver(zeta_problem, solver_parameters=overall_solver_parameters['zetadiag'], options_prefix='zetadiag')

class MaxwellDiagnostics():
    def __init__(self, spaces):
        self.spaces = spaces

class EulerMaxwellDiagnostics():
    def __init__(self, spaces):
        self.spaces = spaces


class ScalarWaveDiagnostics():
    def __init__(self, spaces):
        self.spaces = spaces

class MHDDiagnostics():
    def __init__(self, spaces):
        self.spaces = spaces
