
#EVENTUALLY READ THIS FROM AN INPUT FILE

def get_default_parameters():

    parameters = {}

    parameters['mesh'] = 'rectangle-periodic' #line rectangle
    parameters['nx'] = 50
    parameters['ny'] = 50
    parameters['nz'] = 100
    parameters['diagonal'] = 'crossed' #WHAT ARE THE OPTIONS HERE?
    parameters['simplicial_cells'] = False

#THESE ARE FE SPECIFIC
    parameters['order'] = 3
    parameters['family'] = 'Q'
#MISSING LOTS OF STUFF FOR UPWINDING, ETC.

    parameters['loglevel'] = 100

#OTHER STUFF WILL BE DECLIB SPECIFIC
#FOR EXAMPLE
    #parameters['num_form_quad'] = 3

    parameters['num_steps'] = 1000
    parameters['dt'] = 400
#THESE ARE DYNAMICS TIME STEPS
#200s at nx=100, order=1 for tswe double vortex
#0.0001 at nx=100, order=1 for twse density wave
    parameters['timestep_method'] = 'TimeSplit' #AVF2 TimeStaggered SSPRK TimeSplit
    parameters['num_avf_quad'] = 2
    parameters['output_freq'] = 25
    parameters['stat_freq'] = 1
    parameters['output_aux_vars'] = False #super useful for debugging
    parameters['avf_solver'] = 'qn' #fixedpoint qn
    parameters['kgrk2_name'] = '52' #32 42 52 53
#THE KGRK SCHEMES ARE UNSTABLE WITH nx=50, order=3 for tswe-cf-h1 double vortex at dt=200s
#But RK4 and SSPRK43 are fine
#Unclear what is going on?
#Maybe need hyperviscosity to stabilize?

    parameters['forcing_terms'] = ['dg1limiter', 'hyperviscosity', ] #'threewayphysics'

    parameters['timestepper_split_terms'] = [['model'],  ['hyperviscosity'], ['dg1limiter',], ] #['threewayphysics',]
    #['hyperviscosity',]
    parameters['timestepper_list'] = ['RK4', 'Euler', 'SSPRK43',] #  'Euler'
    parameters['timestepper_substeps'] = [2, 1, 2,] #  1

    parameters['alpha_s'] = 1 #0 = centered, 1 = upwind
    parameters['upwind_v'] = True
#NOT CURRENTLY USED...
    parameters['upwind_total_dens'] = True
#    parameters['use_split_form'] = {'h': False, 'S': True, 'T1': True, 'T2': True, 'DGT1': True, 'DGT2': True}
    parameters['use_split_form'] = {'h': True, 'S': True, 'T1': True, 'T2': True, 'DGT1': False, 'DGT2': False}

    parameters['modeltype'] = 'advdens-cf-h1' #metriplectic advection
    parameters['model'] = 'tswe-cf-h1' # tswe-cf tswe-lp tswe-cf-h1 ce-cf ce-lp mhd maxwell eulermaxwell scalarwave

    parameters['lump_mass'] = True

    parameters['tracer_names'] = [] #['T1', 'T2']
    parameters['dg_tracer_names'] = ['DGT1', 'DGT2'] #['DGT1', 'DGT2']
    parameters['thermo'] = 'idealgas-entropy'
    parameters['tracer_init_conds'] = ['gaussian', 'block'] #['gaussian', 'block']
    parameters['dg_tracer_init_conds'] = ['gaussian', 'block'] #['gaussian', 'block']

    parameters['initialcondition'] = 'doublevortex' #densitywave doublevortex


#WRONG, FIX THEM
    parameters['c0'] = 7e-2
    parameters['s'] = 3.2
    #tswe: doublevortex
    #ce: RP1 RP2 RP3 LOTSMORE
    #maxwell: LOTS HERE
    #eulermaxwell: LOTS ALSO?
    #scalarwave: gaussian block LOTSMORE

    parameters['outfile_name'] = 'sim'
    return parameters

def get_parameters():
    parameters = get_default_parameters()
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
