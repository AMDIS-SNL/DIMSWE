
from firedrake import Function, exp, ln, TestFunction, dx, inner, derivative, TrialFunction, Constant

#THIS IS THE ABGRALL VERSION!
class IdealGasThermo_Entropy():
    def __init__(self):
        pass

    def set_thermo_const(self, ic):
        self.gamma = ic.gamma
        self.Cv = ic.Cv

    def compute_dudrho(self, rho, eta):
        return pow(rho, self.gamma - 2.0) * exp(eta / self.Cv)

    def compute_dudeta(self, rho, eta):
        return pow(rho, self.gamma - 1.0) * exp(eta / self.Cv) / (self.gamma - 1.0) / self.Cv

    def compute_u(self, rho, eta):
        return pow(rho, self.gamma - 1.0) * exp(eta / self.Cv) / (self.gamma - 1.0)

    def get_eta(self, rho, p):
        return self.Cv * ln(p / pow(rho, self.gamma))

    def get_p(self, rho, eta):
        return pow(rho, self.gamma) * exp(eta / self.Cv)

    def get_T(self, rho, eta):
        return pow(rho, self.gamma - 1.0) * exp(eta / self.Cv) / (self.gamma - 1.0) / self.Cv

class IdealGasThermo_PotTemp():
    def __init__(self):
        pass
    def set_thermo_const(self):
        pass
    def get_eta(self, rho, p):
        return 0


def get_thermo(thermochoice):

    if thermochoice == 'idealgas-entropy':
        return IdealGasThermo_Entropy()
    elif thermochoice == 'idealgas-pottemp':
        return IdealGasThermo_PotTemp()
    else:
        raise ValueError("thermo " + thermochoice + " is unknown")

class Hamiltonian_Base():
    def __init__(self, vars):
        self.vars = vars
        self.testvars = {}
        self.trialvars = {}
        if self.vars.spaces is not None:
            for var, space in zip(self.vars.dhdx_var_list, self.vars.spacelist):
                self.testvars[var] = TestFunction(space)
                self.trialvars[var] = TrialFunction(space)

    def get_aux_vars(self, vars):
        for var, space in zip(self.vars.dhdx_var_list, self.vars.spacelist):
            vars[var] = Function(space, name=var)

    def get_aux_vars_list(self):
        return self.vars.dhdx_var_list

def make_a_L(Lexpr, vartrial, varhat):
    a = inner(varhat, vartrial)*dx
    L = inner(varhat, Lexpr)*dx
    return [a, L]

class AdvDensHamiltonian_AdvectionOnly(Hamiltonian_Base):
    def __init__(self, vars, values):
        Hamiltonian_Base.__init__(self, vars)
        self.values = values

    def compute_dfdx_expressions(self, vars, expressions):
        m = vars['m']
        expressions['u'] = make_a_L(self.values['u'], mtrial, mhat)
        for dens_name in self.vars.density_names:
            denshat = self.testvars[dens_name]
            denstrial = self.trialvars[dens_name]
            expressions['B_' + dens_name] = make_a_L(Constant(0), denstrial, denshat)

#WHAT DO WE ACTUALLY DO HERE?
    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        pass

class ThermalShallowWater_Hamiltonian_Base(Hamiltonian_Base):

    def __init__(self, vars):
        Hamiltonian_Base.__init__(self, vars)
        if not vars.spaces is None:
            self.bottom_topography = Function(vars.spaces.CG)

    def initialize(self, varexpr):
        self.bottom_topography.interpolate(varexpr['bottom_topography'])

class ThermalShallowWater_Hamiltonian_LP(ThermalShallowWater_Hamiltonian_Base):

    def compute_total_energy(self, state):
        m = state['m']
        h = state['h']
        S = state['S']
        return inner(m,m)/(2. * h) + inner(h,S)/2. + inner(h,self.bottom_topography)

    def compute_dfdx_expressions(self, vars, expressions):
        m = vars['m']
        h = vars['h']
        S = vars['S']
        mhat = self.testvars['u']
        hhat = self.testvars['B_h']
        Shat = self.testvars['B_S']
        mtrial = self.trialvars['u']
        htrial = self.trialvars['B_h']
        Strial = self.trialvars['B_S']
        expressions['u'] = make_a_L(m / h, mtrial, mhat)
        expressions['B_h'] = make_a_L(-inner(m,m)/(2.*inner(h,h)) + S/2. + self.bottom_topography, htrial, hhat)
        expressions['B_S'] = make_a_L(h/2., Strial, Shat)
        for tracer_name in self.vars.tracer_names:
            tracerhat = self.testvars['B_' + tracer_name]
            tracertrial = self.trialvars['B_' + tracer_name]
            expressions['B_' + tracer_name] = make_a_L(Constant(0), tracertrial, tracerhat)

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):

        H0 = const_state['h']
        S0 = const_state['S']
        m = xstar['m']
        h = xstar['h']
        S = xstar['S']

        dfdx_linear_vars['u'] = m / H0
        dfdx_linear_vars['B_h'] = (S0 + S)/2. + self.bottom_topography
        dfdx_linear_vars['B_S'] = (H0 + H)/2.
        for tracer_name in self.vars.tracer_names:
            dfdx_linear_vars['B_' + tracer_name] = 0.0

class ThermalShallowWater_Hamiltonian_CF(ThermalShallowWater_Hamiltonian_Base):


    def compute_total_energy(self, state):
        v = state['v']
        h = state['h']
        S = state['S']
        return h*inner(v,v)/2. + inner(h,S)/2. + inner(h,self.bottom_topography)

    def compute_dfdx_expressions(self, vars, expressions):
        v = vars['v']
        h = vars['h']
        S = vars['S']
        vhat = self.testvars['F']
        hhat = self.testvars['B_h']
        Shat = self.testvars['B_S']
        vtrial = self.trialvars['F']
        htrial = self.trialvars['B_h']
        Strial = self.trialvars['B_S']
        expressions['F'] = make_a_L(h*v, vtrial, vhat)
        expressions['B_h'] = make_a_L(inner(v,v)/2. + S/2. + self.bottom_topography, htrial, hhat)
        expressions['B_S'] = make_a_L(h/2., Strial, Shat)
        for tracer_name in self.vars.tracer_names:
            tracerhat = self.testvars['B_' + tracer_name]
            tracertrial = self.trialvars['B_' + tracer_name]
            expressions['B_' + tracer_name] = make_a_L(Constant(0), tracertrial, tracerhat)

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        H0 = const_state['h']
        S0 = const_state['S']
        v = xstar['v']
        h = xstar['h']
        S = xstar['S']

        dfdx_linear_vars['F'] = H0 * v
        dfdx_linear_vars['B_h'] = (S0 + S)/2. + self.bottom_topography
        dfdx_linear_vars['B_S'] = (H0 + h)/2.
        for tracer_name in self.vars.tracer_names:
            dfdx_linear_vars['B_' + tracer_name] = 0.0

class ThermalShallowWater_Hamiltonian_CF_H1(ThermalShallowWater_Hamiltonian_CF):
    def compute_dfdx_expressions(self, vars, expressions):
        ThermalShallowWater_Hamiltonian_CF.compute_dfdx_expressions(self, vars, expressions)

        for tracer_name in self.vars.dg_tracer_names:
            tracerhat = self.testvars['B_' + tracer_name]
            tracertrial = self.trialvars['B_' + tracer_name]
            expressions['B_' + tracer_name] = make_a_L(Constant(0), tracertrial, tracerhat)

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        ThermalShallowWater_Hamiltonian_CF.compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars)
        for tracer_name in self.vars.dg_tracer_names:
            dfdx_linear_vars['B_' + tracer_name] = 0.0

class MoistThermalShallowWater_Hamiltonian_CF_H1(ThermalShallowWater_Hamiltonian_CF_H1):
    def compute_dfdx_expressions(self, vars, expressions):
        ThermalShallowWater_Hamiltonian_CF_H1.compute_dfdx_expressions(self, vars, expressions)
        for varname in ['Qv', 'Qr', 'Qc']:
            varhat = self.testvars['B_' + varname]
            vartrial = self.trialvars['B_' + varname]
            expressions['B_' + varname] = make_a_L(Constant(0), vartrial, varhat)

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        ThermalShallowWater_Hamiltonian_CF_H1.compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars)
        for varname in ['Qv', 'Qr', 'Qc']:
            dfdx_linear_vars['B_' + varname] = 0.0

class MoistThermalShallowWater_Hamiltonian_CF(ThermalShallowWater_Hamiltonian_CF):
    def compute_dfdx_expressions(self, vars, expressions):
        ThermalShallowWater_Hamiltonian_CF.compute_dfdx_expressions(self, vars, expressions)
        for varname in ['Qv', 'Qr', 'Qc']:
            varhat = self.testvars['B_' + varname]
            vartrial = self.trialvars['B_' + varname]
            expressions['B_' + varname] = make_a_L(Constant(0), vartrial, varhat)

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        ThermalShallowWater_Hamiltonian_CF.compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars)
        for varname in ['Qv', 'Qr', 'Qc']:
            dfdx_linear_vars['B_' + varname] = 0.0

class MoistThermalShallowWater_Hamiltonian_LP(ThermalShallowWater_Hamiltonian_LP):
    def compute_dfdx_expressions(self, vars, expressions):
        ThermalShallowWater_Hamiltonian_LP.compute_dfdx_expressions(self, vars, expressions)
        for varname in ['Qv', 'Qr', 'Qc']:
            varhat = self.testvars[varname]
            vartrial = self.trialvars[varname]
            expressions['B_' + varname] = make_a_L(Constant(0), vartrial, varhat)

    def compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars):
        ThermalShallowWater_Hamiltonian_LP.compute_dfdx_linear(self, const_state, xstar, dfdx_linear_vars)
        for varname in ['Qv', 'Qr', 'Qc']:
            dfdx_linear_vars['B_' + varname] = 0.0





class CompressibleEuler_Hamiltonian_Base(Hamiltonian_Base):

    def __init__(self, vars, thermo):
        Hamiltonian_Base.__init__(self, vars)
        self.thermo = thermo
        if not vars.spaces is None:
            self.geopotential = Function(vars.spaces.CG)

    def initialize(self, varexpr):
        self.geopotential.interpolate(varexpr['geopotential'])

class CompressibleEuler_Hamiltonian_LP(CompressibleEuler_Hamiltonian_Base):

#ADD POTENTIAL PART!
#EVENTUALLY MAKE TOTAL DENSITY AND ETA COMPUTATIONS PART OF VARIABLESET I THINK?
    def compute_total_energy(self, state):
        m = state['m']
        rho = state['rho']
        S = state['S']
        eta = S / rho
        int_energy = self.thermo.compute_u(rho, eta)
        return inner(m,m)/(2. * rho) + rho * int_energy

#ADD POTENTIAL PART!
#EVENTUALLY MAKE TOTAL DENSITY AND ETA COMPUTATIONS PART OF VARIABLESET I THINK?
    def compute_dfdx_expressions(self, vars, expressions):
        m = vars['m']
        rho = vars['rho']
        S = vars['S']
        mhat = self.testvars['m']
        hhat = self.testvars['h']
        Shat = self.testvars['S']
        eta = S / rho
        int_energy = self.thermo.compute_u(rho, eta)
        dudeta = self.thermo.compute_dudeta(rho, eta)
        dudrho = self.thermo.compute_dudrho(rho, eta)
#FIX THESE!
        expressions['u'] = m / rho
        expressions['B_rho'] = -inner(m,m)/(2. * inner(rho,rho)) + + u + SOMETHING *dudrho + SOMETHING * dudeta
        expressions['B_S'] = SOMETHING * dudeta
        for tracer in self.vars.tracer_names:
            expressions['B_' + tracer] = 0.0




class CompressibleEuler_Hamiltonian_CF(CompressibleEuler_Hamiltonian_Base):

#ADD POTENTIAL PART!
#EVENTUALLY MAKE TOTAL DENSITY AND ETA COMPUTATIONS PART OF VARIABLESET I THINK?
    def compute_total_energy(self, state):
        v = state['v']
        rho = state['rho']
        S = state['S']
        eta = S / rho
        int_energy = self.thermo.compute_u(rho, eta)
        return rho*inner(v,v)/2. + rho * int_energy

#ADD POTENTIAL PART!
#EVENTUALLY MAKE TOTAL DENSITY AND ETA COMPUTATIONS PART OF VARIABLESET I THINK?
    def compute_dfdx_expressions(self, vars, expressions):
        v = vars['v']
        rho = vars['rho']
        S = vars['S']
        vhat = self.testvars['v']
        hhat = self.testvars['h']
        Shat = self.testvars['S']
        eta = S / rho
        int_energy = self.thermo.compute_u(rho, eta)
        dudeta = self.thermo.compute_dudeta(rho, eta)
        dudrho = self.thermo.compute_dudrho(rho, eta)
#FIX THESE!
        expressions['F'] = rho*v
        expressions['B_rho'] = inner(v,v)/2. + u + SOMETHING *dudrho + SOMETHING * dudeta
        expressions['B_S'] = SOMETHING * dudeta
        for tracer in self.vars.tracer_names:
            expressions['B_' + tracer] = 0.0




class MHD_Hamiltonian_LP(CompressibleEuler_Hamiltonian_Base):

    def compute_total_energy(self, state):
        pass

    def compute_dfdx_expressions(self, vars, expressions):
        pass

    def solve_for_aux_vars(self, vars, aux_vars):
        pass

class Maxwell_Hamiltonian(Hamiltonian_Base):

    def compute_total_energy(self, state):
        pass

    def compute_dfdx_expressions(self, vars, expressions):
        pass

    def solve_for_aux_vars(self, vars, aux_vars):
        pass

class ScalarWave_Hamiltonian(Hamiltonian_Base):

    def compute_total_energy(self, state):
        pass

    def compute_dfdx_expressions(self, vars, expressions):
        pass

    def solve_for_aux_vars(self, vars, aux_vars):
        pass

class EulerMaxwell_Hamiltonian_LP(CompressibleEuler_Hamiltonian_Base):

    def compute_total_energy(self, state):
        pass

    def compute_dfdx_expressions(self, vars, expressions):
        pass

    def solve_for_aux_vars(self, vars, aux_vars):
        pass
