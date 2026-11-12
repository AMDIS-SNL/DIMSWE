# Staged verification of the Hessian-vector-product (second-order optimization) feature.
#
# Usage: bash workflows/verify_hessp.sh [--workflow <mpi-ranks>] [--from <stage>]
#
# Runs the step 18 verification sequence in order and stops at the first failing stage:
#
#   1. plumbing      pure-Python tests of ReducedProblem.hessp, optimize() method selection and
#                    the workflow's method tables (no model runs)
#   2. taylor        Firedrake HVP Taylor, symmetry, storage and repeatability tests
#   3. reporting     workflow spec validation, result fields, examples, docs and the comparison
#                    tooling (pure Python)
#   4. end-to-end    one small ic-params optimization with L-BFGS-B and one with Newton-CG, and the
#                    comparison script run on their recorded results
#                    (tests/integration/test_optimizer_end_to_end.py)
#   5. workflow      only with --workflow <mpi-ranks>: the real shell workflow on the double-vortex
#                    example: truth sweep, optimize_ic_params.cfg (L-BFGS-B) and
#                    optimize_ic_params_newton_cg.cfg (Newton-CG), then compare_optimizers.sh on
#                    both roots. Existing truth runs and optimization jobs are reused
#                    (--skip-existing), so delete their directories to rerun from scratch.
#
# --from <stage> (a number or name) skips the earlier stages, for example after fixing a failure.
# The Firedrake stages are meant to run in serial.
set -euo pipefail

WORKFLOW_RANKS=""
FROM=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --workflow) WORKFLOW_RANKS="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "Usage: bash workflows/verify_hessp.sh [--workflow <mpi-ranks>] [--from <stage>]"; exit 1 ;;
  esac
done
case "$FROM" in
  plumbing) FROM=1 ;; taylor) FROM=2 ;; reporting) FROM=3 ;; end-to-end) FROM=4 ;; workflow) FROM=5 ;;
esac

source ~/venv-firedrake/bin/activate

# stage <number> <name>: print a header and return success if the stage should run
stage() {
  if [ "$1" -lt "$FROM" ]; then
    echo "=== stage $1: $2 (skipped by --from) ==="
    return 1
  fi
  echo
  echo "=== stage $1: $2 ==="
}

if stage 1 plumbing; then
  python -m pytest -q tests/unit/test_reduced_problem_hvp.py tests/unit/test_optimizer_methods.py \
    tests/unit/test_workflow_optimizer_tables.py
fi

if stage 2 taylor; then
  python -m pytest -q -s tests/unit/test_timestepping_second_order.py tests/unit/test_optimize_hvp_storage.py \
    tests/unit/test_optimize_hvp.py tests/unit/test_optimize_hvp_repeatability.py \
    tests/unit/test_optimize_without_coefficients.py
  # the first-order gradients must be unaffected by the second-order changes
  python -m pytest -q tests/unit/test_optimize_gradients.py tests/unit/test_timestepping_coeff_gradients.py \
    tests/unit/test_timestepping_ic_gradients.py tests/unit/test_dynamics_gradients.py
fi

if stage 3 reporting; then
  python -m pytest -q tests/integration/test_optimization_workflows.py tests/integration/test_compare_optimizers.py
fi

if stage 4 end-to-end; then
  python -m pytest -q -s tests/integration/test_optimizer_end_to_end.py
fi

if [ -n "$WORKFLOW_RANKS" ] && stage 5 workflow; then
  python -m workflows.generate_cfgs workflows/examples/truth_ic_sweep.cfg
  mpiexec -n "$WORKFLOW_RANKS" python -m workflows.run_batch --skip-existing \
    --manifest workflows/generated/truth-doublevortex/manifest.yaml
  for spec in optimize_ic_params.cfg optimize_ic_params_newton_cg.cfg; do
    mpiexec -n "$WORKFLOW_RANKS" python -m workflows.optimize_params --skip-existing "workflows/examples/$spec"
  done
  # roots from the examples' output_root and optimization_name
  python -m workflows.compare_optimizers runs/optimizations/doublevortex-params \
    runs/optimizations/doublevortex-params-newton-cg
  echo "Check runs/optimizations/optimizer-comparison.csv: L-BFGS-B rows have nhvp 0 and hessp_used False,"
  echo "Newton-CG rows have nhvp > 0 and hessp_used True, and the counts match each job's result.yaml."
elif [ -z "$WORKFLOW_RANKS" ]; then
  echo
  echo "=== stage 5: workflow (not run; pass --workflow <mpi-ranks> to run it) ==="
fi

echo
echo "Verification stages passed."
