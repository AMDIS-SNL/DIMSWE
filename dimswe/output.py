from firedrake import CheckpointFile, split, VTKFile

class Output():
    def __init__(self, parameters, dynamics, timestepper, logger):
        self.parameters = parameters
        self.dynamics = dynamics
        self.timestepper = timestepper
        self.logger = logger

        self.chkpoint_name = self.parameters['outfile_name'] + '.h5'
        CheckpointFile(self.parameters['outfile_name'] + '.h5', 'w')

        self.output_aux_vars = self.parameters['output_aux_vars']
        self.plotoutfile = VTKFile(self.parameters['outfile_name'] + '.pvd')

        self.varlist = self.dynamics.variableset.varlist
        self.q_aux_var_list = self.dynamics.get_q_aux_var_list()
        self.dfdx_aux_var_list = self.dynamics.get_dfdx_aux_var_list()

    def output_mesh(self):
        with CheckpointFile(self.chkpoint_name, 'a') as chkpoint_file:
            chkpoint_file.save_mesh(self.dynamics.mesh)


    def output(self, t, step, output_step, stat_step):
        self.logger.output('output at step ' + str(step) + ' and output_step ' + str(output_step) + ' and stat_step ' + str(stat_step), 0)
        with CheckpointFile(self.chkpoint_name, 'a') as chkpoint_file:

            chkpoint_file.save_function(self.timestepper.xn, idx=output_step, name='xn')

            self.vtk_vars = []
            for i,var in enumerate(self.varlist):
                chkpoint_file.save_function(self.timestepper.xn.sub(i), idx=output_step, name=var)
                self.vtk_vars.append(self.timestepper.xn.sub(i))

            if self.output_aux_vars:
                for var in self.q_aux_var_list:
                    chkpoint_file.save_function(self.timestepper.q_aux_vars[var], idx=output_step, name=var)
                    self.vtk_vars.append(self.timestepper.q_aux_vars[var])
                for var in self.dfdx_aux_var_list:
                    chkpoint_file.save_function(self.timestepper.dfdx_aux_vars[var], idx=output_step, name=var)
                    self.vtk_vars.append(self.timestepper.dfdx_aux_vars[var])

            for var in self.dynamics.diagnostics.var_list:
                chkpoint_file.save_function(self.dynamics.diagnostics.vars[var], idx=output_step, name=var)
                self.vtk_vars.append(self.dynamics.diagnostics.vars[var])

            self.plotoutfile.write(*self.vtk_vars, time=t)

            h5file = chkpoint_file.h5pyfile
            for statistic_name in self.dynamics.statistics.statistic_names:
                if statistic_name in h5file:
                    del h5file[statistic_name]
                h5file.create_dataset(statistic_name, data=self.dynamics.statistics.statistics[statistic_name][:stat_step+1])


#EVENTUALLY ADD RESTART CAPABILITIES TO THE CODE...
