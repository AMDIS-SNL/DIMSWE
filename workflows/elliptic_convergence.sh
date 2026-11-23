source ~/venv-firedrake/bin/activate

if [ "$#" -lt 2 ]; then
  echo "Usage: bash workflows/elliptic_convergence.sh <mpi-ranks> <cfg> --nx N1 N2 ... [--set PATH=VALUE ...]"
  exit 1
fi

MPI_RANKS="$1"
shift

mpiexec -n "$MPI_RANKS" python -m workflows.elliptic_convergence "$@"
