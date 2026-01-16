from .parameters import overall_solver_parameters
from .operators import BracketBase

class ThermodynamicallyCompatibleViscousRegularization_Base(BracketBase):
    def __init__(self, spaces, vars, regularization_type='const'):
        self.spaces = spaces
        self.vars = vars
        self.density_names = vars.density_names
        self.entropy_name = vars.entropy_name
        self.regularization_type = regularization_type

class ThermodynamicallyCompatibleViscousRegularization_LP(ThermodynamicallyCompatibleViscousRegularization_Base):
    pass


class ThermodynamicallyCompatibleViscousRegularization_CF(ThermodynamicallyCompatibleViscousRegularization_Base):
    pass
