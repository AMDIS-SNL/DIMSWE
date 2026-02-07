import numpy as np
from firedrake import assemble

#Advected density statistics

#nD: total densities, total energy, total momenta (per component)

#2D/3D: total "PV"

#hamiltonian/bracket specific: total entropy, various energy comparments, likely some magnetic ones
#2D swe specific: total potential enstrophy


class AdvDensStatistics():
    def __init__(self, spaces, hamiltonian, vars, initcond, nstat):
        self.spaces = spaces
        self.hamiltonian = hamiltonian
        self.vars = vars
        self.density_names = vars.density_names
        self.initcond = initcond

        self.nstat = nstat
        self.statistics = {}
        self.statistics['total_energy'] = np.zeros(nstat)
        self.statistics['total_density'] = np.zeros(nstat)
        self.statistic_names = ['total_energy', 'total_density']
        for dens in self.density_names:
            self.statistic_names.append('total_' + dens)
            self.statistics['total_' + dens] = np.zeros(nstat)

        if 'Qv' in vars.varlist:
            self.statistic_names.append('total_water')
            self.statistics['total_water'] = np.zeros(nstat)

        if not self.spaces is None:
            self.dx = spaces.dx
            self.ds = spaces.ds

    def initialize(self, varexpr):
        pass

    def create(self, xn):
        self.total_energy_expression = self.hamiltonian.compute_total_energy(xn)*self.dx
        self.total_density_expression = self.hamiltonian.vars.get_total_density_expr(xn)*self.dx
        self.total_density_expressions = {}
        for dens in self.density_names:
            self.total_density_expressions[dens] = xn[dens]*self.dx
        if 'Qv' in self.vars.varlist:
            self.water_expression = (xn['Qv'] + xn['Qr'] + xn['Qc'])*self.dx

    def compute(self, step, stat_step):
        self.statistics['total_energy'][stat_step] = assemble(self.total_energy_expression)
        self.statistics['total_density'][stat_step] = assemble(self.total_density_expression)
        for dens in self.density_names:
            self.statistics['total_' + dens][stat_step] = assemble(self.total_density_expressions[dens])
        if 'Qv' in self.vars.varlist:
            self.statistics['total_water'][stat_step] = assemble(self.water_expression)
#Unclear if there is anything that differs here
#probably for computing PV, etc. there will be
class AdvDensStatistics_LP(AdvDensStatistics):
    pass

class AdvDensStatistics_CF(AdvDensStatistics):
    pass

class AdvDensStatistics_CF_H1(AdvDensStatistics):
    pass


class MaxwellStatistics():
    def __init__(self, spaces, nstat):
        self.spaces = spaces

class EulerMaxwellStatistics():
    def __init__(self, spaces, nstat):
        self.spaces = spaces



class ScalarWaveStatistics():
    def __init__(self, spaces, nstat):
        self.spaces = spaces

class MHDStatistics():
    def __init__(self, spaces, nstat):
        self.spaces = spaces
