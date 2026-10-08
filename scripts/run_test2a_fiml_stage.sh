#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {direct-h2|direct-h5|fi-h2|fi-h5|pseudo-h2|pseudo-h5|stage2-h2|stage2-h5}" >&2
  exit 2
fi

STAGE="$1"
SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_fiml_sparse_endpoint_h2_h5.json"
ROOT="$REPOSITORY/external-results/test2a/fiml-sparse-endpoint-h2-h5"
PLAN="$ROOT/preparation/production_plan.json"

cd "$REPOSITORY"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
RUN_CACHE_ROOT="${TEST2A_FIML_CACHE_ROOT:-$(mktemp -d /tmp/dimswe-test2a-fiml.XXXXXX)}"
export PYOP2_CACHE_DIR="$RUN_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUN_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_CACHE_ROOT/pycache"

"$PYTHON" - "$CONFIGURATION" "$PLAN" <<'PY'
import json
import sys
from pathlib import Path
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2a_discrete_training import _file_sha256
from dimswe.test2a_fiml import H1_BASELINE_SHA256, _canonical_json_sha256, load_fiml_configuration
from dimswe.test2a_operator import load_mlp_parameters

configuration = load_fiml_configuration(sys.argv[1])
plan = json.load(open(sys.argv[2], encoding="utf-8"))
assert plan["status"] == "ready"
assert plan["configuration_sha256"] == _canonical_json_sha256(configuration)
parameters, _ = load_mlp_parameters(configuration["baseline"]["parameter_file"])
assert parameter_pytree_sha256(parameters) == H1_BASELINE_SHA256
assert plan["baseline_parameter_pytree_sha256"] == H1_BASELINE_SHA256
assert plan["states_after_80_accessed"] is False
assert plan["budgets"] == {
    "direct": {"2": 100, "5": 100},
    "field_inversion_per_window": {"2": 25, "5": 50},
    "stage2": {"2": 50000, "5": 50000},
}
assert plan["regularization_selection"]["selected_lambdas"] == {"2": 1.0, "5": 0.01}
assert plan["regularization_selection"]["used_true_A"] is False
assert _file_sha256(plan["regularization_selection"]["file"]) == plan["regularization_selection"]["file_sha256"]
print({"event": "fiml_stage_preflight_ok", "configuration_sha256": plan["configuration_sha256"]})
PY

mkdir -p "$ROOT" "$ROOT/logs"
if [[ "${TEST2A_FIML_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "FIML stage preflight complete; production command intentionally not executed"
  exit 0
fi
case "$STAGE" in
  direct-h2|direct-h5)
    HORIZON="${STAGE#direct-h}"
    OUTPUT="$ROOT/direct-endpoint-h$HORIZON"
    if [[ -e "$OUTPUT/fit_result.json" ]]; then exit 0; fi
    "$PYTHON" -u -m dimswe.test2a_fiml train-direct \
      --configuration "$CONFIGURATION" --horizon "$HORIZON" \
      --iterations 100 --output-directory "$OUTPUT" \
      2>&1 | tee -a "$ROOT/logs/$STAGE.log"
    ;;
  fi-h2|fi-h5)
    HORIZON="${STAGE#fi-h}"
    OUTPUT="$ROOT/field-inversion/h$HORIZON"
    if [[ "$HORIZON" == "2" ]]; then ITERATIONS=25; else ITERATIONS=50; fi
    "$PYTHON" -u -m dimswe.test2a_fiml train-field-inversion \
      --configuration "$CONFIGURATION" --horizon "$HORIZON" \
      --iterations "$ITERATIONS" --output-directory "$OUTPUT" \
      2>&1 | tee -a "$ROOT/logs/$STAGE.log"
    ;;
  pseudo-h2|pseudo-h5)
    HORIZON="${STAGE#pseudo-h}"
    CONTROLS="$ROOT/field-inversion/h$HORIZON"
    test -e "$CONTROLS/field_inversion_summary.json"
    OUTPUT="$ROOT/pseudo-labels/h$HORIZON/fiml_pseudo_labels.npz"
    if [[ -e "$OUTPUT" ]]; then
      echo "refusing to overwrite existing pseudo-label dataset" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$OUTPUT")"
    "$PYTHON" -u -m dimswe.test2a_fiml build-pseudo-labels \
      --configuration "$CONFIGURATION" --horizon "$HORIZON" \
      --controls-directory "$CONTROLS" --output "$OUTPUT" \
      2>&1 | tee -a "$ROOT/logs/$STAGE.log"
    ;;
  stage2-h2|stage2-h5)
    HORIZON="${STAGE#stage2-h}"
    DATASET="$ROOT/pseudo-labels/h$HORIZON/fiml_pseudo_labels.npz"
    test -e "$DATASET"
    OUTPUT="$ROOT/stage2/h$HORIZON"
    if [[ -e "$OUTPUT/fit_result.json" ]]; then exit 0; fi
    "$PYTHON" -u -m dimswe.test2a_fiml train-stage2 \
      --configuration "$CONFIGURATION" --horizon "$HORIZON" \
      --dataset "$DATASET" --iterations 50000 --output-directory "$OUTPUT" \
      2>&1 | tee -a "$ROOT/logs/$STAGE.log"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
