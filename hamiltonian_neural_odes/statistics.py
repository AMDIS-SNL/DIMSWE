import numpy as np
from math import exp

class Statistics():
    def get_statistics(self):
        return [['energy', self.energy],]

    def get_statistics_names(self):
        return ['energy',]

class HarmonicOscillatorStatistics(Statistics):
    def __init__(self, omega):
        self.omega = omega

    def create_statistics(self, nsteps):
        self.energy = np.zeros(nsteps)

    def compute_statistics(self, i, x):
        self.energy[i] = self.omega/2.0 * (x[0]*x[0] + x[1]*x[1])

class DoubleWellOscillatorStatistics(Statistics):
    def __init__(self, omega):
        self.omega = omega

    def create_statistics(self, nsteps):
        self.energy = np.zeros(nsteps)

    def compute_statistics(self, i, x):
        self.energy[i] = self.omega/2.0 * (x[0]*x[0] + (1-x[1]*x[1])**2)

class NonlinearOscillatorStatistics(Statistics):
    def __init__(self, omega):
        self.omega = omega

    def create_statistics(self, nsteps):
        self.energy = np.zeros(nsteps)

    def compute_statistics(self, i, x):
        self.energy[i] = self.omega/2.0 * (x[0]*x[0] + exp(x[1]*x[1]))

class MultipleHarmonicOscillatorStatistics(Statistics):
    def __init__(self, omegas):
        self.omegas = omegas
        self.noscillators = self.omegas.shape[0]

    def create_statistics(self, nsteps):
        self.energy = np.zeros(nsteps)
        self.energies = []
        for k in range(self.noscillators):
            self.energies.append(np.zeros(nsteps))

    def compute_statistics(self, i, x):
        self.energy[i] = 0.0
        for k in range(self.noscillators):
            self.energies[k][i] = self.omegas[k]/2.0 * (x[2*k]*x[2*k] + x[2*k+1]*x[2*k+1])
            self.energy[i] += self.energies[k][i]

    def get_statistics(self):
        statistics =  [['energy', self.energy],]
        for k in range(self.noscillators):
            statistics.append(['energy' + str(k), self.energies[k]])
        return statistics

    def get_statistics_names(self):
        statnames = ['energy',]
        for k in range(self.noscillators):
            statnames.append('energy' + str(k))
        return statnames


class SO3Statistics(Statistics):
    def __init__(self):
        pass

    def create_statistics(self, nsteps):
        self.energy = np.zeros(nsteps+1)

    def compute_statistics(self, i, x):
        pass

class SE3Statistics(Statistics):
    def __init__(self):
        pass

    def create_statistics(self, nsteps):
        self.energy = np.zeros(nsteps+1)

    def compute_statistics(self, i, x):
        pass
