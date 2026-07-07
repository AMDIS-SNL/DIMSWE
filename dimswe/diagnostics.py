from firedrake import Function, LinearVariationalProblem, LinearVariationalSolver
from firedrake import FunctionSpace, TestFunction, TrialFunction, inner, curl, grad, div
from .ufl_helpers import skewgrad, curl2D
from .parameters import overall_solver_parameters
from .physics import qsat

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
    def __init__(self, spaces, hamiltonian, variableset, initcond, dim):
        self.spaces = spaces
        self.hamiltonian = hamiltonian
        self.variableset = variableset
        self.density_names = variableset.density_names
        self.dim = dim
        self.initcond = initcond

        if not spaces is None:
            self.bottom_topography = Function(spaces.CG)
            if self.dim == 2:
                self.coriolis = Function(spaces.CG)
            elif self.dim == 3:
                self.coriolis = Function(spaces.CGV) #PRETTY UNCLEAR ACTUALLY- MAYBE H(curl) or even H(div)?

        self.testvars = {}
        self.trialvars = {}
        self.var_list = []
        if self.dim >= 2:
            self.var_list.append('q')
            self.var_list.append('eta')
            self.var_list.append('zeta')
        for dens in self.density_names:
            self.var_list.append(dens + '_l')
        self.var_list.append('coriolis')
        self.var_list.append('bottom_topography')

        if 'Qv' in variableset.varlist:
            self.var_list.append('rh')
            self.var_list.append('qsat')

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
            self.vars['coriolis'] = self.coriolis
            self.vars['bottom_topography'] = self.bottom_topography
            if 'Qv' in variableset.varlist:
                self.vars['rh'] = Function(self.spaces.CG)
                self.testvars['rh'] = TestFunction(self.spaces.CG)
                self.trialvars['rh'] = TrialFunction(self.spaces.CG)
                self.vars['qsat'] = Function(self.spaces.CG)
                self.testvars['qsat'] = TestFunction(self.spaces.CG)
                self.trialvars['qsat'] = TrialFunction(self.spaces.CG)

    def initialize(self, varexpr):
        self.coriolis.interpolate(varexpr['coriolis'])
        self.bottom_topography.interpolate(varexpr['bottom_topography'])

#FOR PERFORMANCE SHOULD PROBABLY CREATE AN INTERPOLATOR AND CALL IT?
#ALSO UNCLEAR IF INTERPOLATE IS BEST, OR IF SOME SORT OF LINEAR SYSTEM SHOULD BE USED?

    def compute_diagnostics(self):
        for dens in self.density_names:
            self.vars[dens + '_l'].interpolate(self.xn[dens] / self.total_dens)
        if self.dim >=2:
            self.pv_solver.solve()
            self.eta_solver.solve()
            self.zeta_solver.solve()
        if 'Qv' in self.variableset.varlist:
            self.rh_solver.solve()
            self.qsat_solver.solve()

    def create_diagnostics(self, xn, t, coeff):
        self.xn = xn
        self.t = t
        if 'Qv' in self.variableset.varlist:
            h = xn['h']
            S = xn['S']
            Qv = xn['Qv']
            qv = Qv / h
            s = S / h
            rhhat = self.testvars['rh']
            rhtrial = self.trialvars['rh']
            qsathat = self.testvars['qsat']
            qsattrial = self.trialvars['qsat']
            q_sat = qsat(h, s, self.bottom_topography, self.initcond.q0, self.initcond.H0, self.initcond.g)
            rh_expr = [inner(rhhat, rhtrial)*self.dx, inner(rhhat, qv/q_sat*100.)*self.dx]
            rh_problem = LinearVariationalProblem(rh_expr[0], rh_expr[1], self.vars['rh'])
            self.rh_solver = LinearVariationalSolver(rh_problem, solver_parameters=overall_solver_parameters['rhdiag'], options_prefix='rhdiag')
            qsat_expr = [inner(qsathat, qsattrial)*self.dx, inner(qsathat, q_sat)*self.dx]
            qsat_problem = LinearVariationalProblem(qsat_expr[0], qsat_expr[1], self.vars['qsat'])
            self.qsat_solver = LinearVariationalSolver(qsat_problem, solver_parameters=overall_solver_parameters['qsatdiag'], options_prefix='qsatdiag')


    def get_diagnostics_list(self):
        return self.var_list

    def get_diagnostics(self):
        return self.vars

class AdvDensDiagnostics_CF_H1(AdvDensDiagnostics):

    def create_diagnostics(self, xn, t, coeff):
        AdvDensDiagnostics.create_diagnostics(self, xn, t, coeff)
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

#
# class AdvDensDiagnostics_CF(AdvDensDiagnostics):
#
#     def create(self, xn, t, coeff):
#         AdvDensDiagnostics.create(self, xn, t, coeff)
#         self.total_dens = self.hamiltonian.vars.get_total_density_expr(self.xn)
#
#         v = xn['v']
#         qhat = self.testvars['q']
#         qtrial = self.trialvars['q']
#
# #MISSING BOUNDARY TERMS...
#         if self.dim == 2:
#             pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             eta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx]
#         if self.dim == 3:
#             pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             eta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx]
#         if self.dim >=2:
#             pv_problem = LinearVariationalProblem(pv_expr[0], pv_expr[1], self.vars['q'])
#             eta_problem = LinearVariationalProblem(eta_expr[0], eta_expr[1], self.vars['eta'])
#             zeta_problem = LinearVariationalProblem(zeta_expr[0], zeta_expr[1], self.vars['zeta'])
#             self.pv_solver = LinearVariationalSolver(pv_problem, solver_parameters=overall_solver_parameters['qdiag'], options_prefix='qdiag')
#             self.eta_solver = LinearVariationalSolver(eta_problem, solver_parameters=overall_solver_parameters['etadiag'], options_prefix='etadiag')
#             self.zeta_solver = LinearVariationalSolver(zeta_problem, solver_parameters=overall_solver_parameters['zetadiag'], options_prefix='zetadiag')
#

#
# #ADD M, U, V
# class AdvDensDiagnostics_LP(AdvDensDiagnostics):
#     def create(self, xn, t, coeff):
#         AdvDensDiagnostics.create(self, xn, t, coeff)
#         total_dens = self.hamiltonian.vars.get_total_density_expr(self.xn)
#
#         m = xn['m']
#         v = m / self.total_dens
#         qhat = self.testvars['q']
#         qtrial = self.trialvars['q']
#
#
#         if self.dim == 2:
#             pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             eta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-skewgrad(qhat), v)*self.dx]
#         if self.dim == 3:
#             pv_expr = [inner(qhat, self.total_dens * qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             eta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx + inner(qhat, self.coriolis)*self.dx]
#             zeta_expr = [inner(qhat, qtrial)*self.dx, inner(-curl(qhat), v)*self.dx]
#         if self.dim >=2:
#             pv_problem = LinearVariationalProblem(pv_expr[0], pv_expr[1], self.vars['q'])
#             eta_problem = LinearVariationalProblem(eta_expr[0], eta_expr[1], self.vars['eta'])
#             zeta_problem = LinearVariationalProblem(zeta_expr[0], zeta_expr[1], self.vars['zeta'])
#             self.pv_solver = LinearVariationalSolver(pv_problem, solver_parameters=overall_solver_parameters['qdiag'], options_prefix='qdiag')
#             self.eta_solver = LinearVariationalSolver(eta_problem, solver_parameters=overall_solver_parameters['etadiag'], options_prefix='etadiag')
#             self.zeta_solver = LinearVariationalSolver(zeta_problem, solver_parameters=overall_solver_parameters['zetadiag'], options_prefix='zetadiag')

# #add involution constraints
# class MaxwellDiagnostics():
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.var_list = ['dB', 'dD']
#
#         if not self.spaces is None:
#             self.dx = spaces.dx
#             self.ds = spaces.ds
#
#             self.vars = {}
#             self.vars['dB'] = Function(self.spaces.DG)
#             self.vars['dD'] = Function(self.spaces.CG)
#
#     def initialize(self, varexpr):
#         pass
#
#     def create(self, xn, t, coeff):
#         self.xn = xn
#         self.t = t
#
#         D, B = xn['D'], xn['B']
#         dD, dB = self.vars['dD'], self.vars['dB']
#         Qhat = TestFunction(self.spaces.CG)
#         Qtrial = TrialFunction(self.spaces.CG)
# #MISSING BOUNDARY TERMS
#         dD_expression = -inner(grad(Qhat), D)*self.dx
#         a = inner(Qhat, Qtrial)*self.dx
#         dD_problem = LinearVariationalProblem(a, dD_expression, dD)
#         self.dD_solver = LinearVariationalSolver(dD_problem, solver_parameters=overall_solver_parameters['dD'], options_prefix='dD')
#
#         dBhat = TestFunction(self.spaces.DG)
#         dBtrial = TrialFunction(self.spaces.DG)
#         dB_expression = inner(dBhat, div(B))*self.dx
#         a = inner(dBhat, dBtrial)*self.dx
#         dB_problem = LinearVariationalProblem(a, dB_expression, dB)
#         self.dB_solver = LinearVariationalSolver(dB_problem, solver_parameters=overall_solver_parameters['dB'], options_prefix='dB')
#
#     def compute(self):
#         self.dD_solver.solve()
#         self.dB_solver.solve()
#
# class EulerMaxwellDiagnostics():
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.var_list = []
#     def initialize(self, varexpr):
#         pass
#     def create(self, xn, t):
#         pass
#
#     def compute(self):
#         pass
#
#
# class ScalarWaveDiagnostics():
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.var_list = []
#     def initialize(self, varexpr):
#         pass
#     def create(self, xn, t, coeff):
#         pass
#
#     def compute(self):
#         pass
#
# class MHDDiagnostics():
#     def __init__(self, spaces):
#         self.spaces = spaces
#         self.var_list = []
#     def initialize(self, varexpr):
#         pass
#     def create(self, xn, t, coeff):
#         pass
#
#     def compute(self):
#         pass
