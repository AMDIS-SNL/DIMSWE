source ~/venv-firedrake/bin/activate

if [ "$#" -lt 2 ]; then
  echo "Usage: bash workflows/elliptic_solver_scaling.sh <mpi-ranks> <cfg> --nx N1 N2 ... [--solver S1 S2 ...] [--order P ...] [--contrast C ...]"
  exit 1
fi

MPI_RANKS="$1"
shift

mpiexec -n "$MPI_RANKS" python -m workflows.elliptic_solver_scaling "$@"
