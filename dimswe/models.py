from .hamiltonians import ThermalShallowWater_Hamiltonian_CF_H1, Maxwell_Hamiltonian
from .variables import ThermalShallowWaterVariables_CF_H1, MoistThermalShallowWaterVariables_CF_H1, MaxwellVariables
from .statistics import AdvDensStatistics, MaxwellStatistics
from .diagnostics import AdvDensDiagnostics_CF_H1, MaxwellDiagnostics
from .dynamics import MetriplecticDynamics, AdvDensCF_H1_Dynamics
from .initcond import get_advdens_initcond, get_maxwell_initcond
from .meshes import get_mesh_and_spaces
from .poisson_brackets import MaxwellBracket
from .entropies import SimpleEntropy
from .dissipation import Hyperviscosity
from .physics import ThreeWayPhysics
from .transport_operators import CGTransport, DG1LimiterTransport

def get_forcing_terms(parameters, vars, spaces, initcond):
    forcing_terms = []
    for forcing_term in parameters['forcing_terms']:
        if forcing_term == 'hyperviscosity':
            forcing_terms.append(Hyperviscosity(parameters, vars, spaces))
        elif forcing_term == 'threewayphysics':
            forcing_terms.append(ThreeWayPhysics(parameters, vars, spaces, initcond))
        elif forcing_term == 'dg1limiter':
            forcing_terms.append(DG1LimiterTransport(parameters, vars, spaces))
        elif forcing_term == 'cgtransport':
            forcing_terms.append(CGTransport(parameters, vars, spaces))
        else:
            raise ValueError("forcing term " + forcing_term + " is unknown")
    return forcing_terms

class Model():
    pass

class AdvDensH1Model(Model):
    def __init__(self, mesh, spaces, parameters):
        self.initcond = get_advdens_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

        if parameters['hamiltonian'] == 'tswe':
            vars = ThermalShallowWaterVariables_CF_H1(self.spaces, parameters['tracer_names'], parameters['dg_tracer_names'])
            hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
        elif parameters['hamiltonian'] == 'mtswe':
            vars = MoistThermalShallowWaterVariables_CF_H1(self.spaces, parameters['tracer_names'], parameters['dg_tracer_names'])
            hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
        else:
            raise ValueError("model " + parameters['model'] + " is unknown")

        self.statistics = AdvDensStatistics_CF_H1(self.spaces, hamiltonian, vars, self.initcond, parameters['num_steps'] // parameters['stat_freq'] + 1)
        self.diagnostics = AdvDensDiagnostics_CF_H1(self.spaces, hamiltonian, vars, self.initcond, parameters['dim'])
        forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)
        self.dynamics = AdvDensCF_H1_Dynamics(parameters, self.mesh, self.spaces, vars, hamiltonian, forcing_terms, logger)

class MaxwellModel(Model):
    def __init__(self, parameters, logger):
        self.initcond = get_maxwell_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

        vars = MaxwellVariables(self.spaces)
        poisson_brackets = [MaxwellBracket(self.spaces),]
        metric_brackets = []
        entropy = SimpleEntropy(self.spaces, vars)
        hamiltonian = Maxwell_Hamiltonian(vars)
        forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)

        self.statistics = MaxwellStatistics(self.spaces, hamiltonian, parameters['num_steps'] // parameters['stat_freq'] + 1)
        self.diagnostics = MaxwellDiagnostics(self.spaces)
        self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)

class AdvDensModelLP(Model):
    def __init__(self, parameters, logger):
        self.initcond = get_maxwell_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

#MIGHT BE ABLE TO JUST HAVE A SINGLE LP OR CF VARIABLES CLASS?
#Or keep these as is...
        if parameters['hamiltonian'] == 'tswe':
            vars = ThermalShallowWaterVariables_LP(spaces, parameters['tracer_names'])
            hamiltonian = ThermalShallowWater_Hamiltonian_LP(vars)
        elif parameters['hamiltonian'] == 'mtswe':
            vars = ThermalShallowWaterVariables_LP(spaces, parameters['tracer_names'])
            hamiltonian = ThermalShallowWater_Hamiltonian_LP(vars)
        elif parameters['hamiltonian'] == 'ce':
            vars = CompressibleEulerVariables_LP(spaces, parameters['tracer_names'])
            thermo = get_thermo(parameters['thermo'])
            hamiltonian = CompressibleEuler_Hamiltonian_LP(vars, thermo)
#NEED A BETTER WAY TO HANDLE THIS
            initcond.set_thermo(hamiltonian.thermo)
            hamiltonian.thermo.set_thermo_const(initcond)
        elif parameters['hamiltonian'] == 'mhd':
            vars = MHDVariables_LP(spaces, parameters['tracer_names'])
            thermo = get_thermo(parameters['thermo'])
            hamiltonian = MHD_Hamiltonian_LP(vars, thermo)
            initcond.set_thermo(hamiltonian.thermo)
            hamiltonian.thermo.set_thermo_const(initcond)

        poisson_brackets = [LiePoisson_Bracket(spaces, vars, parameters), ]
#FIX THIS STUFF?
        entropy = SimpleEntropy(spaces, vars)
        metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_LP(spaces, vars, entropy), ]

        self.statistics = AdvDensStatistics_LP(spaces, hamiltonian, vars, initcond, parameters['num_steps'] // parameters['stat_freq'] + 1)
        self.diagnostics = AdvDensDiagnostics_LP(spaces, hamiltonian, vars, initcond, parameters['dim'])
        self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)

class AdvDensModelCF(Model):
    def __init__(self, parameters, logger):
        self.initcond = get_maxwell_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

        elif parameters['model'] in ['tswe-cf', 'ce-cf', 'mtswe-cf']:
            if parameters['model'] == 'tswe-cf':
                vars = ThermalShallowWaterVariables_CF(spaces, parameters['tracer_names'])
                hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
            if parameters['model'] == 'mtswe-cf':
                vars = MoistThermalShallowWaterVariables_CF(spaces, parameters['tracer_names'])
                hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
            if parameters['model'] == 'ce-cf':
                vars = CompressibleEulerVariables_CF(spaces, parameters['tracer_names'])
                thermo = get_thermo(parameters['thermo'])
                hamiltonian = CompressibleEuler_Hamiltonian_CF(vars, thermo)
                initcond.set_thermo(hamiltonian.thermo)
                hamiltonian.thermo.set_thermo_const(initcond)
            poisson_brackets = [CurlForm_AdvectedDensities_Bracket(spaces, vars, parameters), ]
            metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_CF(spaces, vars), ]
            entropy = SimpleEntropy(spaces, vars)
            statistics = AdvDensStatistics_CF(spaces, hamiltonian, vars, initcond, parameters['num_steps'] // parameters['stat_freq'] + 1)
            diagnostics = AdvDensDiagnostics_CF(spaces, hamiltonian, vars, initcond, parameters['dim'])

class ScalarWaveModel(Model):
    def __init__(self, parameters, logger):
        self.initcond = get_maxwell_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

        vars = ScalarWaveVariables(self.spaces)
        poisson_brackets = [ScalarWaveBracket(self.spaces),]
        metric_brackets = []
        entropy = SimpleEntropy(self.spaces, vars)
        hamiltonian = ScalarWave_Hamiltonian(vars)
        forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)

        self.statistics = ScalarWaveStatistics(self.spaces, hamiltonian, parameters['num_steps'] // parameters['stat_freq'] + 1)
        self.diagnostics = ScalarWaveDiagnostics(self.spaces)
        self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)

class EulerMaxwellModel(Model):
    def __init__(self, parameters, logger):
        self.initcond = get_maxwell_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

        vars = EulerMaxwellVariables(self.spaces)
        #MORE ARGUMENTS HERE LIKELY!
        poisson_brackets = [LiePoisson_Bracket(self.spaces), MaxwellBracket(self.spaces),]
        metric_brackets = []
        entropy = SimpleEntropy(self.spaces, vars)
        hamiltonian = EulerMaxwell_Hamiltonian(vars)
        forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)

        self.statistics = EulerMaxwellStatistics(self.spaces, hamiltonian, parameters['num_steps'] // parameters['stat_freq'] + 1)
        self.diagnostics = EulerMaxwellDiagnostics(self.spaces)
        self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)
