import yaml

def get_parameters(cfgfile):
    with open(cfgfile, 'r') as file:
        parameters = yaml.safe_load(file) # Use safe_load for security
    #print(parameters)
    return parameters

#-nonlinsys_snes_max_it 100
#-nonlinsys_snes_linesearch_type basic
#-nonlinsys_snes_linesearch_orderNO 3
#-nonlinsys_snes_linesearch_damping 1.0

preonly_linear_system = { 'ksp_type': 'preonly', 'pc_type' : 'lu', 'ksp_converged_reason': None,} #'ksp_monitor_true_residual': None}
basic_linear_system = { 'ksp_type': 'cg', 'pc_type' : 'jacobi', 'ksp_converged_reason': None} #'ksp_monitor_true_residual': None}

overall_solver_parameters = {}
overall_solver_parameters['qn'] = {'snes_monitor': None, 'snes_converged_reason': None,
    'snes_stol': 1e-12, 'snes_rtol': 1e-12, 'snes_atol': 1e-12,
    'snes_lag_jacobian': 100, 'snes_lag_preconditioner': 100, 'ksp_converged_reason': None,
    'ksp_type': 'gmres', 'pc_type' : 'ilu', 'snes_max_it': 50,
    'ksp_stol': 1e-25, 'ksp_rtol': 1e-25, 'ksp_atol': 1e-25}
overall_solver_parameters['fixedpoint'] = basic_linear_system
#overall_solver_parameters['q'] = basic_linear_system
#overall_solver_parameters['u'] = basic_linear_system
#overall_solver_parameters['F'] = basic_linear_system
#overall_solver_parameters['B_h'] = basic_linear_system
#overall_solver_parameters['B_S'] = basic_linear_system
overall_solver_parameters['qdiag'] = basic_linear_system
overall_solver_parameters['etadiag'] = basic_linear_system
overall_solver_parameters['zetadiag'] = basic_linear_system
overall_solver_parameters['erkstage-f'] = basic_linear_system
overall_solver_parameters['erkstage-aux'] = basic_linear_system
overall_solver_parameters['erkstage-mu'] = basic_linear_system
overall_solver_parameters['erkstage-muaux'] = basic_linear_system
overall_solver_parameters['erk-dlambda'] = basic_linear_system
overall_solver_parameters['erk-grad'] = basic_linear_system
#overall_solver_parameters['grad'] = basic_linear_system
#overall_solver_parameters['B_T1'] = basic_linear_system
#overall_solver_parameters['B_T2'] = basic_linear_system
#overall_solver_parameters['B_DGT1'] = basic_linear_system
#overall_solver_parameters['B_DGT2'] = basic_linear_system
#overall_solver_parameters['Q_v'] = basic_linear_system
#overall_solver_parameters['Q_h'] = basic_linear_system
#overall_solver_parameters['Q_S'] = basic_linear_system
overall_solver_parameters['rhdiag'] = basic_linear_system
overall_solver_parameters['qsatdiag'] = basic_linear_system
#overall_solver_parameters['E'] = basic_linear_system
#overall_solver_parameters['H'] = basic_linear_system
#overall_solver_parameters['dD'] = basic_linear_system
#overall_solver_parameters['dB'] = basic_linear_system
