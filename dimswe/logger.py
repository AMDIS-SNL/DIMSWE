from firedrake.petsc import PETSc

#EVENTUALLY DO SOMETHING MORE CLEVER- CMD LINE VS FILE OUTPUT, ERROR OUTPUT, SINGLE PROCESS, ETC.
class Logger():
    def __init__(self, parameters):
        self.loglevel = parameters['loglevel']
        
    def output(self, message, loglevel):
        if self.loglevel >= loglevel:
            print(message)


#PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)
#PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
#                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
#                comm=COMM_SELF)
