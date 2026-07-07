class OperatorBase():

    def initialize(self, varexpr):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_aux_expressions(self, x, aux, t, coeff, xhats, expressions):
        pass

    def post_step(self, statevars):
        pass

    def get_coeff(self):
        return None

    def has_coeff(self):
        return False

    def set_default_coeffs(self, coeff):
        pass

    def get_varlist(self):
        return []

    def get_spacelist(self):
        return []

    def rhs(self, xvars, auxvars, xhats):
        return 0.0

#POSSIBLY FIX UP LINEAR RHS CALLING FORMAT?
class BracketBase(OperatorBase):

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return 0.0

class ForcingBase(OperatorBase):

    def linear_rhs(self, const_state, xvars, xhats):
        return 0.0
