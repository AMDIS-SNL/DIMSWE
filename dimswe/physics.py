
from .operators import ForcingBase


class ThreeWayPhysics(ForcingBase):
    def __init__(self, parameters, vars, spaces):
        self.vars = vars
        self.spaces = spaces
        self.name = 'threewayphysics'

#THIS IS BROKEN!
    def linear_rhs(self, const_state, xstar, xhats):
        raise NotImplementedError
