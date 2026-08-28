#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "BTP" && "$1" != "BTPL" ]]; then
  echo "usage: $0 BTP|BTPL" >&2
  exit 2
fi

VARIANT="$1"
REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
PYTHON="/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python"
CONFIGURATION="dimswe/configs/test2b_rain_active_learning.json"
PREPARATION="external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz"
if [[ "$VARIANT" == "BTP" ]]; then
  RAIN_OUTPUT_PREPARATION="external-results/test2b-rain-active-learning/preparation/representation_btp_output_map.json"
else
  RAIN_OUTPUT_PREPARATION="external-results/test2b-rain-active-learning/preparation/representation_bplus_output_map.json"
fi
OUTPUT_ROOT="external-results/test2b-rain-active-learning/production/representation-${VARIANT}"
STAGE_ROOT="$OUTPUT_ROOT/m1-seed0-m20-10k"

cd "$REPOSITORY"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
RUNTIME_CACHE="$(mktemp -d /tmp/dimswe-test2b-${VARIANT}.XXXXXX)"
export PYOP2_CACHE_DIR="$RUNTIME_CACHE/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUNTIME_CACHE/tsfc"
export XDG_CACHE_HOME="$RUNTIME_CACHE/xdg"
export MPLCONFIGDIR="$RUNTIME_CACHE/matplotlib"
export PYTHONPYCACHEPREFIX="$RUNTIME_CACHE/pycache"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite $OUTPUT_ROOT" >&2
  exit 2
fi
for required in \
  "$PREPARATION" "${PREPARATION%.npz}.json" "$RAIN_OUTPUT_PREPARATION"; do
  if [[ ! -f "$required" ]]; then
    echo "required frozen preparation is missing: $required" >&2
    exit 2
  fi
done

"$PYTHON" -u -m dimswe.test2b_rain_learning_campaign train \
  --configuration "$CONFIGURATION" \
  --preparation "$PREPARATION" \
  --rain-output-preparation "$RAIN_OUTPUT_PREPARATION" \
  --representation "$VARIANT" \
  --stage M1 \
  --output-directory "$STAGE_ROOT" \
  --iteration-limit 10000

printf '%s\n' "Test2B ${VARIANT} M1 production fit complete."
