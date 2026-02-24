import yaml




#def setdefault_recursively(tgt, default = default_values):
#    for k in default:
#        if isinstance(default[k], dict): # if the current item is a dict,
#            # expand it recursively
#            setdefault_recursively(tgt.setdefault(k, {}), default[k])
#        else:
#            # ... otherwise simply set a default value if it's not set before
#            tgt.setdefault(k, default[k])

#dic = { 'image_name': 'ubuntu', 'components': { 'image_name': 'ubuntu' }

#setdefault_recursively(dic)


def get_parameters(cfgfile):
    with open(cfgfile, 'r') as file:
        parameters = yaml.safe_load(file) # Use safe_load for security
    print(parameters)
    return parameters


#-nonlinsys_snes_max_it 100
#-nonlinsys_snes_linesearch_type basic
#-nonlinsys_snes_linesearch_orderNO 3
#-nonlinsys_snes_linesearch_damping 1.0

preonly_linear_system = { 'ksp_type': 'preonly', 'pc_type' : 'lu', 'ksp_converged_reason': None,} #'ksp_monitor_true_residual': None}
basic_linear_system = { 'ksp_type': 'cg', 'pc_type' : 'ilu', 'ksp_converged_reason': None,} #'ksp_monitor_true_residual': None}

overall_solver_parameters = {}
overall_solver_parameters['qn'] = {'snes_monitor': None, 'snes_converged_reason': None,
    'snes_stol': 1e-12, 'snes_rtol': 1e-12, 'snes_atol': 1e-12,
    'snes_lag_jacobian': 100, 'snes_lag_preconditioner': 100, 'ksp_converged_reason': None,
    'ksp_type': 'gmres', 'pc_type' : 'ilu', 'snes_max_it': 50}
overall_solver_parameters['fixedpoint'] = basic_linear_system
overall_solver_parameters['q'] = basic_linear_system
overall_solver_parameters['u'] = basic_linear_system
overall_solver_parameters['F'] = basic_linear_system
overall_solver_parameters['B_h'] = basic_linear_system
overall_solver_parameters['B_S'] = basic_linear_system
overall_solver_parameters['qdiag'] = basic_linear_system
overall_solver_parameters['etadiag'] = basic_linear_system
overall_solver_parameters['zetadiag'] = basic_linear_system
overall_solver_parameters['rkstage'] = basic_linear_system
overall_solver_parameters['B_T1'] = basic_linear_system
overall_solver_parameters['B_T2'] = basic_linear_system
overall_solver_parameters['B_DGT1'] = basic_linear_system
overall_solver_parameters['B_DGT2'] = basic_linear_system
overall_solver_parameters['Q_v'] = basic_linear_system
overall_solver_parameters['Q_h'] = basic_linear_system
overall_solver_parameters['Q_S'] = basic_linear_system
overall_solver_parameters['rhdiag'] = basic_linear_system
overall_solver_parameters['qsatdiag'] = basic_linear_system
