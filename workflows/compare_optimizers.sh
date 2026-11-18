# Compare optimization jobs across optimizer methods from their result.yaml files.
#
# Usage: bash workflows/compare_optimizers.sh <job-dir-or-root> [<job-dir-or-root> ...] [compare_optimizers.py args...]
#
# For example, after running both optimize_ic_params.cfg and optimize_ic_params_newton_cg.cfg:
#   bash workflows/compare_optimizers.sh runs/optimizations/doublevortex-params runs/optimizations/doublevortex-params-newton-cg
# Pure Python: no MPI ranks are needed.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash workflows/compare_optimizers.sh <job-dir-or-root> [<job-dir-or-root> ...] [--output-dir DIR] [--no-save] [--only SUBSTRING ...] [--columns COLUMN ...]"
  exit 1
fi

source ~/venv-firedrake/bin/activate
python -m workflows.compare_optimizers "$@"
