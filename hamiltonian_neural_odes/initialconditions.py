class InitialCondition():
    def __init__(self,):
        pass


class HarmonicOscillatorIC(InitialCondition):
    def __init__(self, noscillators):
        self.noscillators = noscillators

    def set_initial_condition(self, x):
        for i in range(self.noscillators):
            x[2*i] = 1.0
            x[2*i+1] = 0.0

def get_initcond(parameters):
    if parameters['initcond'] == 'harmonic-oscillator':
        return HarmonicOscillatorIC(parameters['num_variable_pairs'])
    else:
        raise ValueError('unknown initial condition ' + parameters['initcond'])
