class OperatorBase():

    def initialize(self, varexpr):
        pass

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, x, t, coeff, expressions):
        pass

    def post_step(self, statevars):
        pass

    def get_coeff(self):
        return None

    def has_coeff(self):
        return False

    def set_default_coeffs(self, coeff):
        pass

class BracketBase(OperatorBase):
    def rhs(self, qvars, dfdx_vars, xhats):
        return 0.0

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return 0.0


class ForcingBase(OperatorBase):
    def rhs(self, xvars, xhats):
        return 0.0

    def linear_rhs(self, const_state, xvars, xhats):
        return 0.0
