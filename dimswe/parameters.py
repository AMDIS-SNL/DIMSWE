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
quasi_newton_system = {'snes_monitor': None, 'snes_converged_reason': None,
    'snes_stol': 1e-12, 'snes_rtol': 1e-12, 'snes_atol': 1e-12,
    'snes_lag_jacobian': 100, 'snes_lag_preconditioner': 100, 'ksp_converged_reason': None,
    'ksp_type': 'gmres', 'pc_type' : 'ilu', 'snes_max_it': 50,
    'ksp_stol': 1e-25, 'ksp_rtol': 1e-25, 'ksp_atol': 1e-25}

overall_solver_parameters = {}
overall_solver_parameters['qn'] = quasi_newton_system
overall_solver_parameters['fixedpoint'] = basic_linear_system
overall_solver_parameters['qdiag'] = basic_linear_system
overall_solver_parameters['etadiag'] = basic_linear_system
overall_solver_parameters['zetadiag'] = basic_linear_system
overall_solver_parameters['erkstage-f'] = basic_linear_system
overall_solver_parameters['erkstage-aux'] = basic_linear_system
overall_solver_parameters['erkstage-mu'] = basic_linear_system
overall_solver_parameters['erkstage-muaux'] = basic_linear_system
overall_solver_parameters['erk-dlambda'] = basic_linear_system
overall_solver_parameters['erk-grad'] = basic_linear_system
overall_solver_parameters['rhdiag'] = basic_linear_system
overall_solver_parameters['qsatdiag'] = basic_linear_system
#

# Elliptic (Hodge Laplacian) solvers, see dimswe/elliptic.py and TODO-elliptic §4.6.
# The 'lagrange' solve with harmonic forms is matrix-free (Firedrake cannot assemble monolithic
# matrices with R blocks) and uses dimswe.harmonic_forms.lagrange_solver_parameters instead.
elliptic_direct = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps',
                   'mat_mumps_icntl_14': 200}
# Singular systems ('nullspace' approach): GMRES with the nullspace attached, preconditioned by the
# LU of the regularized non-singular matrix a + eps (u, v) that dimswe.elliptic assembles as pmat
# (not by MUMPS null-pivot detection, which can miss the null pivot in 1D and amplify the nullspace
# direction). Right preconditioning: the stopping test uses the true residual.
elliptic_direct_singular = {'ksp_type': 'gmres', 'ksp_pc_side': 'right', 'ksp_rtol': 1e-12, 'ksp_atol': 1e-14,
                            'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps',
                            'mat_mumps_icntl_14': 200}
# MINRES with a block-diagonal SPD (Riesz map) preconditioner; the fieldsplit options per block are
# added by dimswe.elliptic.elliptic_solver_parameters.
elliptic_minres = {'ksp_type': 'minres', 'ksp_rtol': 1e-12, 'ksp_atol': 1e-14, 'ksp_max_it': 2000}

elliptic_solver_parameters = {
    'elliptic-direct': elliptic_direct,
    'elliptic-direct-singular': elliptic_direct_singular,
    'elliptic-minres': elliptic_minres,
}
