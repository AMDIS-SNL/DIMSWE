


class ThreeWayPhysics():
    def __init__(self, parameters, vars, spaces):
        self.vars = vars
        self.spaces = spaces

    def initialize(self, varexpr):
        pass
        
    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def rhs(self, qvars, xhats):
        return 0

    def linear_rhs(self, const_state, xstar, xhats):
        return 0
