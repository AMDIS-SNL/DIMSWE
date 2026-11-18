# Optimizer regression baseline (step 19): run one small problem with a first-order and a
# second-order method, compare them, and save or check the baseline.
#
# Usage: bash workflows/optimizer_baseline.sh <mpi-ranks> (--save | --check) [--all-jobs] [compare_optimizers.py args...]
#
# The problem is the double-vortex offset recovery (ic-params, ox and oy) of
# workflows/examples/optimize_ic_params.cfg (L-BFGS-B) and optimize_ic_params_newton_cg.cfg
# (Newton-CG). By default only the final-state loss jobs are run (one per truth run), which keeps
# the run short; --all-jobs also runs the trajectory loss. Truth runs and finished jobs are reused
# (--skip-existing): delete runs/optimizations/doublevortex-params* to rerun the optimizations,
# for example after changing the optimizer code.
#
#   --save   write workflows/baselines/doublevortex-params.yaml from the results
#   --check  compare the results with that baseline and fail on a regression (more evaluations,
#            lost convergence, a much worse final loss); extra arguments such as --count-rtol,
#            --loss-factor or --time-factor are passed to compare_optimizers.py
set -euo pipefail

if [ "$#" -lt 2 ] || { [ "$2" != "--save" ] && [ "$2" != "--check" ]; }; then
  echo "Usage: bash workflows/optimizer_baseline.sh <mpi-ranks> (--save | --check) [--all-jobs] [compare_optimizers.py args...]"
  exit 1
fi
MPI_RANKS="$1"
MODE="$2"
shift 2
ONLY=(--only __final)
if [ "${1:-}" = "--all-jobs" ]; then
  ONLY=()
  shift
fi
BASELINE=workflows/baselines/doublevortex-params.yaml
ROOTS=(runs/optimizations/doublevortex-params runs/optimizations/doublevortex-params-newton-cg)

source ~/venv-firedrake/bin/activate

python -m workflows.generate_cfgs workflows/examples/truth_ic_sweep.cfg
mpiexec -n "$MPI_RANKS" python -m workflows.run_batch --skip-existing \
  --manifest workflows/generated/truth-doublevortex/manifest.yaml
for spec in optimize_ic_params.cfg optimize_ic_params_newton_cg.cfg; do
  mpiexec -n "$MPI_RANKS" python -m workflows.optimize_params --skip-existing "workflows/examples/$spec" "${ONLY[@]}"
done

# compare only the jobs that were run, so a --save after a partial run records exactly those
if [ "$MODE" = "--save" ]; then
  python -m workflows.compare_optimizers "${ROOTS[@]}" "${ONLY[@]}" --save-baseline "$BASELINE" "$@"
else
  python -m workflows.compare_optimizers "${ROOTS[@]}" "${ONLY[@]}" --check-baseline "$BASELINE" "$@"
fi
