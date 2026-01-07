class PoissonBracket():
    def __init__(self):
        pass

class CanonicalBracket(PoissonBracket):
    def __init__(self, n):
        self.n = n

    def compute_rhs(self, rhs, x, dhdx):
        for i in range(self.n):
            rhs[i*2] = dhdx[i*2+1]
            rhs[i*2+1] = -dhdx[i*2]

class LiePoissonBracket(PoissonBracket):
    def __init__(self, liealgebra):
        self.liealgebra = liealgebra

    def compute_rhs(self, rhs, x, dhdx):
        pass



class LiePoissonSDPBracket(PoissonBracket):
    def __init__(self, liealgebra, vectorspace):
        self.liealgebra = liealgebra
        self.vectorspace = vectorspace

    def compute_rhs(self, rhs, x, dhdx):
        pass
