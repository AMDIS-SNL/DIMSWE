# Run the full optimization workflow: truth data, optimization, validation, plotting.
#
# Usage: bash workflows/run_pipeline.sh <mpi-ranks> <truth-sweep.cfg> <optimize.cfg> [--skip-truth]
#
# The optimization spec must point at the manifest the truth sweep generates
# (workflows/generated/<sweep-name>/manifest.yaml).
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash workflows/run_pipeline.sh <mpi-ranks> <truth-sweep.cfg> <optimize.cfg> [--skip-truth]"
  exit 1
fi

MPI_RANKS="$1"
TRUTH_CFG="$2"
OPTIMIZE_CFG="$3"
SKIP_TRUTH="${4:-}"

source ~/venv-firedrake/bin/activate

if [ "$SKIP_TRUTH" != "--skip-truth" ]; then
  # step 1: truth data
  python -m workflows.generate_cfgs "$TRUTH_CFG"
  SWEEP_NAME=$(python -c "import sys, yaml; from pathlib import Path; p = Path(sys.argv[1]); print(yaml.safe_load(p.read_text()).get('sweep_name', p.stem))" "$TRUTH_CFG")
  mpiexec -n "$MPI_RANKS" python -m workflows.run_batch --skip-existing --manifest "workflows/generated/$SWEEP_NAME/manifest.yaml"
fi

# step 2: optimization (ic-field jobs require a single rank)
TARGET=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['target'])" "$OPTIMIZE_CFG")
OPT_RANKS="$MPI_RANKS"
if [ "$TARGET" = "ic-field" ]; then
  OPT_RANKS=1
fi
mpiexec -n "$OPT_RANKS" python -m workflows.optimize_params --skip-existing "$OPTIMIZE_CFG"

# step 3: validation and plots
OPT_DIR=$(python -c "
import sys
from pathlib import Path
from workflows.common import load_yaml, resolve_runs_root
spec = load_yaml(sys.argv[1])
print(resolve_runs_root(spec.get('output_root', 'runs/optimizations')) / spec.get('optimization_name', Path(sys.argv[1]).stem))
" "$OPTIMIZE_CFG")
mpiexec -n "$OPT_RANKS" python -m workflows.validate --skip-existing "$OPT_DIR"
python -m workflows.plot_validation --sweep "$OPT_DIR"
