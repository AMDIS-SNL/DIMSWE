from .parameters import overall_solver_parameters

class ThermodynamicallyCompatibleViscousRegularization_Base:
    def __init__(self, spaces, vars, regularization_type='const'):
        self.spaces = spaces
        self.vars = vars
        self.density_names = vars.density_names
        self.entropy_name = vars.entropy_name
        self.regularization_type = regularization_type

    def initialize(self, varexpr):
        pass

    def get_aux_vars(self, vars):
        pass

    def get_aux_vars_list(self):
        return []

    def compute_q_expressions(self, vars, expressions):
        pass

    def rhs(self, qvars, dfdx_vars, xhats):
        return 0.0

    def linear_rhs(self, const_state, dfdx_linear_vars, xhats):
        return 0.0


class ThermodynamicallyCompatibleViscousRegularization_LP(ThermodynamicallyCompatibleViscousRegularization_Base):
    pass


class ThermodynamicallyCompatibleViscousRegularization_CF(ThermodynamicallyCompatibleViscousRegularization_Base):
    pass
