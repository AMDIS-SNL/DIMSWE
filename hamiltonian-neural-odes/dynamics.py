from poisson_brackets import CanonicalBracket, LiePoissonBracket, LiePoissonSDPBracket
from hamiltonians import HarmonicOscillatorHamiltonian, MultipleHarmonicOscillatorHamiltonian
from hamiltonians import DoubleWellOscillatorHamiltonian, NonlinearOscillatorHamiltonian
from vars import CanonicalPairs, LieAlgebra
from statistics import HarmonicOscillatorStatistics, SO3Statistics, SE3Statistics
from statistics import MultipleHarmonicOscillatorStatistics, DoubleWellOscillatorStatistics, NonlinearOscillatorStatistics

class HamiltonianDynamics():
    def __init__(self, parameters, poissonbracket, hamiltonian, vars, statistics, initcond):
        self.poissonbracket = poissonbracket
        self.hamiltonian = hamiltonian
        self.parameters = parameters
        self.vars = vars
        self.initcond = initcond
        self.statistics = statistics

    def create_statistics(self, nsteps):
        self.statistics.create_statistics(nsteps)

    def compute_statistics(self, i, xn):
        self.statistics.compute_statistics(i, xn)

    def get_statistics(self):
        return self.statistics.get_statistics()

    def get_state(self):
        return self.timestepper.get_state()

    def compute_dhdx(self, dhdx, x):
        self.hamiltonian.compute_dhdx(dhdx, x)

    def compute_rhs(self, rhs, x, dhdx):
        self.poissonbracket.compute_rhs(rhs, x, dhdx)

    def create_rhs(self):
        return self.vars.create_x()

    def create_x(self, size=None):
        if size is None:
            return self.vars.create_x()
        else:
            return self.vars.create_long_x(size)

    def create_dhdx(self):
        return self.hamiltonian.create_dhdx()

def get_dynamics(parameters, initcond):
    if parameters['bracket'] == 'canonical':
        vars = CanonicalPairs(parameters['num_variable_pairs'])
        poissonbracket = CanonicalBracket(parameters['num_variable_pairs'])
        if parameters['hamiltonian'] == 'harmonic-oscillator':
            hamiltonian = HarmonicOscillatorHamiltonian(parameters['omega'])
            statistics = HarmonicOscillatorStatistics(parameters['omega'])
        elif parameters['hamiltonian'] == 'doublewell-oscillator':
            hamiltonian = DoubleWellOscillatorHamiltonian(parameters['omega'])
            statistics = DoubleWellOscillatorStatistics(parameters['omega'])
        elif parameters['hamiltonian'] == 'nonlinear-oscillator':
            hamiltonian = NonlinearOscillatorHamiltonian(parameters['omega'])
            statistics = NonlinearOscillatorStatistics(parameters['omega'])
        elif parameters['hamiltonian'] == 'multiple-harmonic-oscillator':
            hamiltonian = MultipleHarmonicOscillatorHamiltonian(parameters['omegas'])
            statistics = MultipleHarmonicOscillatorStatistics(parameters['omegas'])
        else:
            raise ValueError('unknown hamiltonian ' + parameters['hamiltonian'])
    elif parameters['bracket'] == 'lie-poisson':
        vars = LieAlgebra()
        poissonbracket = LiePoissonBracket(vars)
        if parameters['hamiltonian'] == 'SO3':
            hamiltonian == SO3Hamiltonian()
            statistics = SO3Statistics()
        else:
            raise ValueError('unknown hamiltonian ' + parameters['hamiltonian'])
    elif parameters['bracket'] == 'lie-poisson-spd':
        vars = LieAlgebra()
        poissonbracket = LiePoissonSDPBracket(vars)
        if parameters['hamiltonian'] == 'SE3':
            hamiltonian == SE3Hamiltonian()
            statistics = SE3Statistics()
        else:
            raise ValueError('unknown hamiltonian ' + parameters['hamiltonian'])
    else:
        raise ValueError('unknown bracket ' + parameters['bracket'])
    return HamiltonianDynamics(parameters, poissonbracket, hamiltonian, vars, statistics, initcond)
