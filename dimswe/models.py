from .hamiltonians import ThermalShallowWater_Hamiltonian_CF
from .variables import ThermalShallowWaterVariables_CF_H1, MoistThermalShallowWaterVariables_CF_H1
from .statistics import AdvDensStatistics_CF_H1
from .diagnostics import AdvDensDiagnostics_CF_H1
from .dynamics import AdvDensCF_H1_Dynamics
from .initial_conditions import get_advdens_initcond
from .meshes import get_mesh_and_spaces
#from .poisson_brackets import MaxwellBracket
#from .entropies import SimpleEntropy
from .dissipation import Hyperviscosity
from .physics import ThreeWayPhysics
from .transport_operators import DG1LimiterTransport

from firedrake import Constant, inner, assemble

def get_forcing_terms(parameters, vars, spaces, initcond):
    forcing_terms = []
    for forcing_term in parameters['model']['forcing_terms']:
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

    def get_max_wavespeed(self):
        return self.dynamics.get_max_wavespeed(name)

    def get_x_var(self, name):
        return self.dynamics.get_x_var(name)

    def get_full_var(self, name, split_x_and_aux=False):
        return self.dynamics.get_full_var(name, split_x_and_aux=split_x_and_aux)

    def get_aux_var(self, name):
        return self.dynamics.get_aux_var(name)

    def get_coeff_var(self, name):
        return self.dynamics.get_coeff_var(name)

    def get_coeff_test_trial_vars(self):
        return self.dynamics.get_coeff_test_trial_vars()

    def get_x_spaces(self,):
        return self.dynamics.get_x_spaces()

    def get_full_spaces(self):
        return self.dynamics.get_full_spaces()

    def get_aux_spaces(self):
        return self.dynamics.get_aux_spaces()

    def get_coeff_spaces(self):
        return self.dynamics.get_coeff_spaces()

    def get_t_var(self):
        return Constant(0.)

    def has_aux(self):
        return self.dynamics.has_aux

    def has_coeff(self):
        return self.dynamics.has_coeff

    def get_x_test_vars(self):
        return self.dynamics.get_x_test_vars()

    def get_x_trial_vars(self):
        return self.dynamics.get_x_trial_vars()

    def get_full_test_vars(self, split_x_and_aux=False):
        return self.dynamics.get_full_test_vars(split_x_and_aux=split_x_and_aux)

    def get_full_trial_vars(self, split_x_and_aux=False):
        return self.dynamics.get_full_trial_vars(split_x_and_aux=split_x_and_aux)

    def get_aux_test_vars(self):
        return self.dynamics.get_aux_test_vars()

    def get_aux_trial_vars(self):
        return self.dynamics.get_aux_trial_vars()

    def compute_aux_expressions(self, xk_sub, t, coeff_sub, xhats, terms='all'):
        return self.dynamics.compute_aux_expressions(xk_sub, t, coeff_sub, xhats, terms=terms)

    def compute_q_expressions(self, xk_sub, aux, t, coeff_sub, xhats, terms='all'):
        return self.dynamics.compute_q_expressions(xk_sub, t, coeff_sub, xhats, terms=terms)

    def compute_dfdx_expressions(self, xk_sub, aux, t, coeff_sub, xhats, terms='all'):
        return self.dynamics.compute_dfdx_expressions(xk_sub, t, coeff_sub, xhats, terms=terms)

    def rhs(self, xk_split, t, coeff_split, xhat_subs, terms='all'):
        return self.dynamics.rhs(xk_split, t, coeff_split, xhat_subs, terms=terms)

    def get_x_var_list(self):
        return self.dynamics.get_x_var_list()

    def get_full_var_list(self):
        return self.dynamics.get_full_var_list()

    def get_coeff_list(self):
        return self.dynamics.get_coeff_list()

    def get_q_aux_var_list(self, terms='all'):
        return self.dynamics.get_q_aux_var_list(terms=terms)

    def get_dfdx_aux_var_list(self, terms='all'):
        return self.dynamics.get_dfdx_aux_var_list(terms=terms)

    def get_aux_var_list(self, terms='all'):
        return self.dynamics.get_aux_var_list(terms=terms)

    def set_coeffs(self, parameters, coeff_sub):
        self.dynamics.set_coeffs(parameters, coeff_sub)

    def create_diagnostics(self, xn_sub, t, coeff):
        self.diagnostics.create_diagnostics(xn_sub, t, coeff)

    def create_statistics(self, xn_sub, t, coeff):
        self.statistics.create_statistics(xn_sub, t, coeff)

    def compute_diagnostics(self):
        self.diagnostics.compute_diagnostics()

    def compute_statistics(self, step, stat_step):
        self.statistics.compute_statistics(step, stat_step)

    def get_diagnostics_list(self):
        return self.diagnostics.get_diagnostics_list()

    def get_statistics_list(self):
        return self.statistics.get_statistics_list()

    def get_statistics(self):
        return self.statistics.get_statistics()

    def get_diagnostics(self):
        return self.diagnostics.get_diagnostics()

    def initialize(self, xn, t, new_params=None):
        if not(new_params is None):
            self.initcond.set_params(new_params)
        t.assign(self.initcond.get_t0())
        varexpr = self.initcond.get_value(self.mesh, t)
        self.dynamics.initialize(xn, varexpr)
        if self.has_dynamics_statistics:
            self.diagnostics.initialize(varexpr)
            self.statistics.initialize(varexpr)

    def norm(self, v):
        expr = inner(v,v)*self.spaces.dx
        return assemble(expr)**(1./2.)

    #def restart(self, xn, x0, t, t0):
    #    t.assign(t0)
    #    xn[0].assign(x0[0])
        #if len(xn) > 1:
        #    xn[1].assign(x0[1])
#DIAGNOSTICS AND STATISTICS NEED TO BE RESET ALSO
#ALONG WITH ANY MODEL CONSTANTS/COEFFICIENTS...

    def get_x_size(self):
        return self.dynamics.get_x_size()

    def get_coeff_size(self):
        return self.dynamics.get_coeff_size()

class AdvDensH1Model(Model):
    def __init__(self, parameters, logger, has_dynamics_statistics=True):
        self.initcond = get_advdens_initcond(parameters)
        self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)

        if parameters['model']['hamiltonian'] == 'tswe':
            vars = ThermalShallowWaterVariables_CF_H1(self.spaces, parameters['model']['tracer_names'], parameters['model']['dg_tracer_names'])
            hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
        elif parameters['model']['hamiltonian'] == 'mtswe':
            vars = MoistThermalShallowWaterVariables_CF_H1(self.spaces, parameters['model']['tracer_names'], parameters['model']['dg_tracer_names'])
            hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
        else:
            raise ValueError("hamiltonian " + parameters['model']['hamiltonian'] + " is unknown")

        self.has_dynamics_statistics=has_dynamics_statistics
        if has_dynamics_statistics:
            self.statistics = AdvDensStatistics_CF_H1(self.spaces, hamiltonian, vars, self.initcond, parameters['timestepping']['num_steps'] // parameters['output']['stat_freq'] + 1)
            self.diagnostics = AdvDensDiagnostics_CF_H1(self.spaces, hamiltonian, vars, self.initcond, parameters['mesh']['dim'])
        forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)
        self.dynamics = AdvDensCF_H1_Dynamics(parameters, self.mesh, self.spaces, vars, hamiltonian, forcing_terms, logger)

def get_model(parameters, logger, has_dynamics_statistics=True):
    if parameters['model']['type'] == 'advdens-cf-h1':
        return AdvDensH1Model(parameters, logger, has_dynamics_statistics=has_dynamics_statistics)
    elif parameters['model']['type'] == 'metriplectic':
        return MetriplecticModel(parameters, logger, has_dynamics_statistics=has_dynamics_statistics)
    elif parameters['model']['type'] == 'advection':
        return AdvectionModel(parameters, logger, has_dynamics_statistics=has_dynamics_statistics)
    else:
        raise ValueError("model type" + parameters['model']['type'] + " is unknown")
#
# class MaxwellModel(Model):
#     def __init__(self, parameters, logger):
#         self.initcond = get_maxwell_initcond(parameters)
#         self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)
#
#         vars = MaxwellVariables(self.spaces)
#         poisson_brackets = [MaxwellBracket(self.spaces),]
#         metric_brackets = []
#         entropy = SimpleEntropy(self.spaces, vars)
#         hamiltonian = Maxwell_Hamiltonian(vars)
#         forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)
#
#         self.statistics = MaxwellStatistics(self.spaces, hamiltonian, parameters['num_steps'] // parameters['stat_freq'] + 1)
#         self.diagnostics = MaxwellDiagnostics(self.spaces)
#         self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)
#
# class AdvDensModelLP(Model):
#     def __init__(self, parameters, logger):
#         self.initcond = get_maxwell_initcond(parameters)
#         self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)
#
# #MIGHT BE ABLE TO JUST HAVE A SINGLE LP OR CF VARIABLES CLASS?
# #Or keep these as is...
#         if parameters['hamiltonian'] == 'tswe':
#             vars = ThermalShallowWaterVariables_LP(spaces, parameters['tracer_names'])
#             hamiltonian = ThermalShallowWater_Hamiltonian_LP(vars)
#         elif parameters['hamiltonian'] == 'mtswe':
#             vars = ThermalShallowWaterVariables_LP(spaces, parameters['tracer_names'])
#             hamiltonian = ThermalShallowWater_Hamiltonian_LP(vars)
#         elif parameters['hamiltonian'] == 'ce':
#             vars = CompressibleEulerVariables_LP(spaces, parameters['tracer_names'])
#             thermo = get_thermo(parameters['thermo'])
#             hamiltonian = CompressibleEuler_Hamiltonian_LP(vars, thermo)
# #NEED A BETTER WAY TO HANDLE THIS
#             initcond.set_thermo(hamiltonian.thermo)
#             hamiltonian.thermo.set_thermo_const(initcond)
#         elif parameters['hamiltonian'] == 'mhd':
#             vars = MHDVariables_LP(spaces, parameters['tracer_names'])
#             thermo = get_thermo(parameters['thermo'])
#             hamiltonian = MHD_Hamiltonian_LP(vars, thermo)
#             initcond.set_thermo(hamiltonian.thermo)
#             hamiltonian.thermo.set_thermo_const(initcond)
#
#         poisson_brackets = [LiePoisson_Bracket(spaces, vars, parameters), ]
# #FIX THIS STUFF?
#         entropy = SimpleEntropy(spaces, vars)
#         metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_LP(spaces, vars, entropy), ]
#
#         self.statistics = AdvDensStatistics_LP(spaces, hamiltonian, vars, initcond, parameters['num_steps'] // parameters['stat_freq'] + 1)
#         self.diagnostics = AdvDensDiagnostics_LP(spaces, hamiltonian, vars, initcond, parameters['dim'])
#         self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)
#
# class AdvDensModelCF(Model):
#     def __init__(self, parameters, logger):
#         self.initcond = get_maxwell_initcond(parameters)
#         self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)
#
#         elif parameters['model'] in ['tswe-cf', 'ce-cf', 'mtswe-cf']:
#             if parameters['model'] == 'tswe-cf':
#                 vars = ThermalShallowWaterVariables_CF(spaces, parameters['tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
#             if parameters['model'] == 'mtswe-cf':
#                 vars = MoistThermalShallowWaterVariables_CF(spaces, parameters['tracer_names'])
#                 hamiltonian = ThermalShallowWater_Hamiltonian_CF(vars)
#             if parameters['model'] == 'ce-cf':
#                 vars = CompressibleEulerVariables_CF(spaces, parameters['tracer_names'])
#                 thermo = get_thermo(parameters['thermo'])
#                 hamiltonian = CompressibleEuler_Hamiltonian_CF(vars, thermo)
#                 initcond.set_thermo(hamiltonian.thermo)
#                 hamiltonian.thermo.set_thermo_const(initcond)
#             poisson_brackets = [CurlForm_AdvectedDensities_Bracket(spaces, vars, parameters), ]
#             metric_brackets = [ThermodynamicallyCompatibleViscousRegularization_CF(spaces, vars), ]
#             entropy = SimpleEntropy(spaces, vars)
#             statistics = AdvDensStatistics_CF(spaces, hamiltonian, vars, initcond, parameters['num_steps'] // parameters['stat_freq'] + 1)
#             diagnostics = AdvDensDiagnostics_CF(spaces, hamiltonian, vars, initcond, parameters['dim'])
#
# class ScalarWaveModel(Model):
#     def __init__(self, parameters, logger):
#         self.initcond = get_maxwell_initcond(parameters)
#         self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)
#
#         vars = ScalarWaveVariables(self.spaces)
#         poisson_brackets = [ScalarWaveBracket(self.spaces),]
#         metric_brackets = []
#         entropy = SimpleEntropy(self.spaces, vars)
#         hamiltonian = ScalarWave_Hamiltonian(vars)
#         forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)
#
#         self.statistics = ScalarWaveStatistics(self.spaces, hamiltonian, parameters['num_steps'] // parameters['stat_freq'] + 1)
#         self.diagnostics = ScalarWaveDiagnostics(self.spaces)
#         self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)
#
# class EulerMaxwellModel(Model):
#     def __init__(self, parameters, logger):
#         self.initcond = get_maxwell_initcond(parameters)
#         self.mesh, self.spaces = get_mesh_and_spaces(parameters, self.initcond)
#
#         vars = EulerMaxwellVariables(self.spaces)
#         #MORE ARGUMENTS HERE LIKELY!
#         poisson_brackets = [LiePoisson_Bracket(self.spaces), MaxwellBracket(self.spaces),]
#         metric_brackets = []
#         entropy = SimpleEntropy(self.spaces, vars)
#         hamiltonian = EulerMaxwell_Hamiltonian(vars)
#         forcing_terms = get_forcing_terms(parameters, vars, self.spaces, self.initcond)
#
#         self.statistics = EulerMaxwellStatistics(self.spaces, hamiltonian, parameters['num_steps'] // parameters['stat_freq'] + 1)
#         self.diagnostics = EulerMaxwellDiagnostics(self.spaces)
#         self.dynamics =  MetriplecticDynamics(self.mesh, self.spaces, vars, poisson_brackets, metric_brackets, hamiltonian, entropy, forcing_terms, logger)
