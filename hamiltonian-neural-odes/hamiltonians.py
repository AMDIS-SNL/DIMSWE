import numpy as np
from math import exp

class Hamiltonian():
    pass

class ParticleHamiltonian(Hamiltonian):
    def __init__(self):
        pass


class HarmonicOscillatorHamiltonian(Hamiltonian):
    def __init__(self, omega):
        self.omega = omega

    def compute_dhdx(self, dhdx, x):
        dhdx[0] = self.omega * x[0]
        dhdx[1] = self.omega * x[1]

    def create_dhdx(self):
        return np.zeros(2)


class DoubleWellOscillatorHamiltonian(Hamiltonian):
    def __init__(self, omega):
        self.omega = omega

    def compute_dhdx(self, dhdx, x):
        dhdx[0] = self.omega * x[0]
        dhdx[1] = -2.0*self.omega * x[1] * (1. - x[1]*x[1])

    def create_dhdx(self):
        return np.zeros(2)

class NonlinearOscillatorHamiltonian(Hamiltonian):
    def __init__(self, omega):
        self.omega = omega

    def compute_dhdx(self, dhdx, x):
        dhdx[0] = self.omega * x[0]
        dhdx[1] = self.omega * x[1] * exp(x[1]*x[1])

    def create_dhdx(self):
        return np.zeros(2)

class MultipleHarmonicOscillatorHamiltonian(Hamiltonian):
    def __init__(self, omegas):
        self.omegas = omegas
        self.noscillators = self.omegas.shape[0]

    def compute_dhdx(self, dhdx, x):

        for i in range(self.noscillators):
            dhdx[i*2] = self.omegas[i] * x[i*2]
            dhdx[i*2+1] = self.omegas[i] * x[i*2+1]

    def create_dhdx(self):
        return np.zeros(2*self.noscillators)

class SO3Hamiltonian(Hamiltonian):
    def __init__(self):
        pass

class SE3Hamiltonian(Hamiltonian):
    def __init__(self):
        pass
