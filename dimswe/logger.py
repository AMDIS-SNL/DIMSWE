from firedrake.petsc import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#EVENTUALLY DO SOMETHING MORE CLEVER- CMD LINE VS FILE OUTPUT, ERROR OUTPUT, SINGLE PROCESS, ETC.
class Logger():
    def __init__(self, parameters):
        self.loglevel = parameters['output']['loglevel']

    def output(self, message, loglevel):
        if (rank == 0) and self.loglevel >= loglevel:
            print(message)

class EmptyLogger():
    def __init__(self):
        pass

    def output(self, message, loglevel):
        pass


#PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)
#PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
#                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
#                comm=COMM_SELF)
