import numpy as np

class CanonicalPairs():
    def __init__(self, n):
        self.n = n
        self.dim = 2 * n

    def create_x(self):
        return np.zeros(self.dim)

    def create_long_x(self, nsteps):
        return np.zeros((nsteps, self.dim))

    def variable_names(self):
        varnames = []
        for i in range(self.n):
            varnames.append('q'+str(i))
            varnames.append('p'+str(i))
        return varnames

class LieAlgebra():
    def liebracket():
        pass
#MIGHT BE NICE TO EVENTUALLY ADD COMPOSITION AND EXTENSIONS?

#UNCLEAR EXACTLY THE GENERAL WAY TO DO THIS?
class VectorSpace():
    def diamond():
        pass


#we want to support (p,q), (mu,) and (mu,alpha)
#with possibly multiple p,q pairs!
#this is enough to test a variety of systems I think
#also add some other non-canonical Hamiltonian systems I think?
