from firedrake import CheckpointFile, split, VTKFile

class Output():
    def __init__(self, xn, xn_sub, coeff_sub, parameters, model, logger):
        self.parameters = parameters
        self.model = model
        self.logger = logger
        self.xn = xn
        self.xn_sub = xn_sub
        self.coeff_sub = coeff_sub

        self.chkpoint_name = self.parameters['output']['outfile_name'] + '.h5'
        CheckpointFile(self.chkpoint_name, 'w')

        self.output_aux_vars = self.parameters['output']['output_aux_vars']
        self.plotoutfile = VTKFile(self.parameters['output']['outfile_name'] + '.pvd')

        self.x_var_list = self.model.get_x_var_list()
        self.aux_var_list = self.model.get_aux_var_list()
        self.coeff_list = self.model.get_coeff_list()
        self.diagnostics_list = self.model.get_diagnostics_list()
        self.statistics_list = self.model.get_statistics_list()

        self.statistics = self.model.get_statistics()
        self.diagnostics = self.model.get_diagnostics()

    def output_mesh(self):
        with CheckpointFile(self.chkpoint_name, 'a') as chkpoint_file:
            chkpoint_file.save_mesh(self.model.mesh)


    def output(self, t, step, output_step, stat_step):
        self.logger.output('output at step ' + str(step) + ' and output_step ' + str(output_step) + ' and stat_step ' + str(stat_step), 0)
        with CheckpointFile(self.chkpoint_name, 'a') as chkpoint_file:

            for i in range(len(self.xn)):
                chkpoint_file.save_function(self.xn[i], idx=output_step, name='xn'+str(i))

            self.vtk_vars = []
            for i,var in enumerate(self.x_var_list):
                chkpoint_file.save_function(self.xn_sub[var], idx=output_step, name=var)
                self.vtk_vars.append(self.xn_sub[var])

            self.vtk_vars = []
            for i,var in enumerate(self.coeff_list):
                chkpoint_file.save_function(self.coeff_sub[var], idx=output_step, name=var)
                self.vtk_vars.append(self.coeff_sub[var])

            if self.output_aux_vars:
                for var in self.aux_var_list:
                    chkpoint_file.save_function(self.xn_sub[var], idx=output_step, name=var)
                    self.vtk_vars.append(self.xn_sub[var])

            for var in self.diagnostics_list:
                chkpoint_file.save_function(self.diagnostics[var], idx=output_step, name=var)
                self.vtk_vars.append(self.diagnostics[var])

            self.plotoutfile.write(*self.vtk_vars, time=t)

            h5file = chkpoint_file.h5pyfile
            for statistic_name in self.statistics_list:
                if statistic_name in h5file:
                    del h5file[statistic_name]
                h5file.create_dataset(statistic_name, data=self.statistics[statistic_name][:stat_step+1])


#EVENTUALLY ADD RESTART CAPABILITIES TO THE CODE...
