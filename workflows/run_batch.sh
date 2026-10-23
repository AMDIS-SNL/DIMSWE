source ~/venv-firedrake/bin/activate

if [ "$#" -lt 1 ]; then
  echo "Usage: bash workflows/run_batch.sh <mpi-ranks> [run_batch.py args...]"
  exit 1
fi

MPI_RANKS="$1"
shift

mpiexec -n "$MPI_RANKS" python -m workflows.run_batch "$@"
