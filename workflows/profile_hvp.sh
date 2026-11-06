# Profile objective, gradient and Hessian-vector-product evaluations of one optimization job
# (docs/hvp_performance_plan.md, section 2; workflows/profile_hvp.py).
#
# Usage: bash workflows/profile_hvp.sh <mpi-ranks> <optimize-cfg> [--flamegraph] [profile_hvp.py args...]
#
# The truth runs of the spec must already exist (for example from optimizer_baseline.sh). The
# report goes to runs/profiles/<optimization>/<job>/profile.yaml and profile-tree.txt. With
# --flamegraph, PETSc's nested event log is also written, as flamegraph.txt and (if
# ./flamegraph.pl exists, as for run_model.sh) flamegraph.svg next to profile.yaml; the
# profiling regions of dimswe/profiling.py appear in it inside PETSc's own events. Nested PETSc
# logging adds some overhead, so take the timings from a run without --flamegraph.
#
# For example, the steady-state Newton-CG costs of the baseline problem:
#   bash workflows/profile_hvp.sh 4 workflows/examples/optimize_ic_params_newton_cg.cfg --job __final --repeats 2
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash workflows/profile_hvp.sh <mpi-ranks> <optimize-cfg> [--flamegraph] [--job SUBSTRING] [--repeats N] [--seed S] [--output-dir DIR] [--results RESULT.yaml ...]"
  exit 1
fi
MPI_RANKS="$1"
SPEC="$2"
shift 2
FLAMEGRAPH=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--flamegraph" ]; then
    FLAMEGRAPH=1
  else
    ARGS+=("$arg")
  fi
done

source ~/venv-firedrake/bin/activate

if [ "$FLAMEGRAPH" = "0" ]; then
  mpiexec -n "$MPI_RANKS" python -m workflows.profile_hvp "$SPEC" ${ARGS[@]+"${ARGS[@]}"}
  exit 0
fi

# the flamegraph needs an output directory before the run: use --output-dir if given, otherwise
# ask profile_hvp.py (--print-output-dir does not build the model)
OUTPUT_DIR=$(python -m workflows.profile_hvp "$SPEC" ${ARGS[@]+"${ARGS[@]}"} --print-output-dir | tail -n 1)
mkdir -p "$OUTPUT_DIR"
mpiexec -n "$MPI_RANKS" python -m workflows.profile_hvp "$SPEC" ${ARGS[@]+"${ARGS[@]}"} --output-dir "$OUTPUT_DIR" \
  -log_view ":$OUTPUT_DIR/flamegraph.txt:ascii_flamegraph"
if [ -x ./flamegraph.pl ] || [ -f ./flamegraph.pl ]; then
  perl ./flamegraph.pl --countname us "$OUTPUT_DIR/flamegraph.txt" > "$OUTPUT_DIR/flamegraph.svg"
  echo "Wrote $OUTPUT_DIR/flamegraph.svg"
else
  echo "flamegraph.pl not found in $(pwd); render $OUTPUT_DIR/flamegraph.txt with it to get an SVG"
fi
