import h5py as h5

class Output():
    def __init__(self, parameters, timestepper):
        self.parameters = parameters
        self.timestepper = timestepper

        self.ofile = h5.File(parameters['outputname'] + '.hdf5', 'w')

    def output(self,):

        for statistic_name,statistic_data in self.timestepper.get_statistics():
            self.ofile.create_dataset(statistic_name, data=statistic_data)

        for var_name,var_data in self.timestepper.get_state():
            self.ofile.create_dataset(var_name, data=var_data)
        if self.timestepper.implicit:
            for var_name,var_data in self.timestepper.solver.get_convergence_data():
                self.ofile.create_dataset(var_name, data=var_data)
