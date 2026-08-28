#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
CONFIGURATION="dimswe/configs/test2b_rain_active_learning.json"
PREPARATION="external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz"

if [[ $# -ne 1 || ! "$1" =~ ^[ABC]$ ]]; then
  echo "usage: $0 A|B|C" >&2
  exit 2
fi
REPRESENTATION="$1"
OUTPUT_ROOT="external-results/test2b-rain-active-learning/production/representation-${REPRESENTATION}"

cd "$REPOSITORY"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
RUNTIME_CACHE="$(mktemp -d /tmp/dimswe-test2b-learning.XXXXXX)"
export PYOP2_CACHE_DIR="$RUNTIME_CACHE/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUNTIME_CACHE/tsfc"
export XDG_CACHE_HOME="$RUNTIME_CACHE/xdg"
export MPLCONFIGDIR="$RUNTIME_CACHE/matplotlib"
export PYTHONPYCACHEPREFIX="$RUNTIME_CACHE/pycache"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$PREPARATION" || ! -f "${PREPARATION%.npz}.json" ]]; then
  echo "certified fixed preparation is missing" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

run_stage() {
  local stage="$1" limit="$2" directory="$3" initial="${4:-}"
  local arguments=(
    -u -m dimswe.test2b_rain_learning_campaign train
    --configuration "$CONFIGURATION" --preparation "$PREPARATION"
    --representation "$REPRESENTATION" --stage "$stage"
    --output-directory "$directory" --iteration-limit "$limit"
  )
  if [[ -n "$initial" ]]; then arguments+=(--initial-parameters "$initial"); fi
  "$PYTHON" "${arguments[@]}"
}

# Independent seed-zero fits expose the M2-X optimizer-basin effect.
run_stage M1 10000 "$OUTPUT_ROOT/m1-seed0-m20-10k"
run_stage M2-X 10000 "$OUTPUT_ROOT/m2x-seed0-m20-10k"
M1_FINAL="$OUTPUT_ROOT/m1-seed0-m20-10k/final_parameters.npz"
run_stage M2-X 5000 "$OUTPUT_ROOT/m1-to-m2x-m20-5k" "$M1_FINAL"

# Deployment-location/recursive curriculum; every stage creates a new ROL
# process with empty L-BFGS history and receives parameters only.
run_stage H1 5000 "$OUTPUT_ROOT/h1-from-m1-m20-5k" "$M1_FINAL"
H1_FINAL="$OUTPUT_ROOT/h1-from-m1-m20-5k/final_parameters.npz"
run_stage H2 20 "$OUTPUT_ROOT/h2-from-h1-m20-20" "$H1_FINAL"
H2_FINAL="$OUTPUT_ROOT/h2-from-h1-m20-20/final_parameters.npz"
run_stage H5 20 "$OUTPUT_ROOT/h5-from-h2-m20-20" "$H2_FINAL"

printf '%s\n' "Test2B representation $REPRESENTATION production ladder complete."
