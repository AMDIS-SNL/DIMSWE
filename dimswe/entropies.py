class EmptyEntropy():
    # def __init__(self, spaces, vars):
    #     self.spaces = spaces
    #     self.vars = vars
    #     self.entropy_name = vars.entropy_name
    def __init__(self):
        pass
        
    def compute_total_entropy():
        pass

    def initialize(self, varexpr):
        pass

    def compute_dfdx_expressions(self, xvars, t, coeff, expressions):
        pass

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        pass

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []
