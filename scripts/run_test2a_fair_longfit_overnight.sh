#!/bin/bash
set -euo pipefail

REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
VIRTUAL_ENVIRONMENT="/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312"
OUTPUT_ROOT="$REPOSITORY/external-results/test2a/fair-longfit"
OPERATOR_ROOT="$OUTPUT_ROOT/operator-seed0-m20-200k"
DISCRETE_ROOT="$OUTPUT_ROOT/discrete-seed0-m20-200k"
COMPARISON_ROOT="$OUTPUT_ROOT/comparison"
CACHE="$REPOSITORY/external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz"
OPERATOR_CONFIG="$REPOSITORY/dimswe/configs/test2a_fair_operator_200k.json"
DISCRETE_CONFIG="$REPOSITORY/dimswe/configs/test2a_fair_discrete_200k.json"
DATASET="$REPOSITORY/external-results/test2a/dataset/doublevortex_A_operator.npz"
SELECTED="$REPOSITORY/dimswe/configs/test2a_selected_operator.json"
AUTONOMOUS_CONFIG="$REPOSITORY/dimswe/configs/test2a_apriori_autonomous.json"

cd "$REPOSITORY"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"

RUN_CACHE_ROOT="${TEST2A_OVERNIGHT_CACHE_ROOT:-$(mktemp -d /tmp/dimswe-test2a-fair-longfit.XXXXXX)}"
export PYOP2_CACHE_DIR="$RUN_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUN_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_CACHE_ROOT/pycache"

mkdir -p "$OUTPUT_ROOT" "$COMPARISON_ROOT"
for path in "$OPERATOR_ROOT" "$DISCRETE_ROOT"; do
  if [[ -e "$path/fit_result.json" || -e "$path/fit_progress.json" ]]; then
    echo "refusing to overwrite existing primary fit at $path" >&2
    exit 2
  fi
done

python - <<'PY'
from pathlib import Path
from dimswe.test2a_discrete_training import load_fixed_cache
from dimswe.test2a_operator import initialize_mlp, load_selected_configuration, mlp_configuration_from_record
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
cache = load_fixed_cache('external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz')
if not cache.metadata.get('production_oracle_certified', False):
    raise SystemExit('Method-2 cache is not production-oracle certified')
selected = load_selected_configuration('dimswe/configs/test2a_selected_operator.json')
initial = initialize_mlp(mlp_configuration_from_record(selected['model']))
expected = '6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975'
if parameter_pytree_sha256(initial) != expected:
    raise SystemExit('canonical seed-0 fingerprint mismatch')
print({'event': 'overnight_preflight_ok', 'seed_sha256': expected,
       'cache_sha256': cache.metadata['cache_npz_sha256']}, flush=True)
PY

python -u -m dimswe.test2a_fair_longfit train-operator \
  --configuration "$OPERATOR_CONFIG" \
  --output-directory "$OPERATOR_ROOT" \
  2>&1 | tee "$OPERATOR_ROOT.log"

python - <<PY
import json
r=json.load(open('$OPERATOR_ROOT/fit_result.json'))
assert r['status'] == 'complete'
assert r['initialization']['parameter_pytree_sha256'] == '6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975'
PY

python -u -m dimswe.test2a_discrete_training train \
  --configuration "$DISCRETE_CONFIG" \
  --cache "$CACHE" \
  --output-directory "$DISCRETE_ROOT" \
  2>&1 | tee "$DISCRETE_ROOT.log"

python - <<PY
import json
r=json.load(open('$DISCRETE_ROOT/fit_result.json'))
assert r['status'] == 'complete'
assert r['initialization']['parameter_pytree_sha256'] == '6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975'
PY

python -u -m dimswe.test2a_fair_longfit cross-objectives \
  --operator-result "$OPERATOR_ROOT/fit_result.json" \
  --discrete-result "$DISCRETE_ROOT/fit_result.json" \
  --cache "$CACHE" \
  --selected-configuration "$SELECTED" \
  --dataset "$DATASET" \
  --output "$COMPARISON_ROOT/cross_objectives.json"

python -u -m dimswe.test2a_apriori_autonomous \
  --configuration "$AUTONOMOUS_CONFIG" \
  --parameter-file "$OPERATOR_ROOT/final_parameters.npz" \
  --output-directory "$COMPARISON_ROOT/operator-autonomous-training-support"

python -u -m dimswe.test2a_apriori_autonomous \
  --configuration "$AUTONOMOUS_CONFIG" \
  --parameter-file "$DISCRETE_ROOT/final_parameters.npz" \
  --output-directory "$COMPARISON_ROOT/discrete-autonomous-training-support"

python -u -m dimswe.test2a_fair_longfit report \
  --cross-objectives "$COMPARISON_ROOT/cross_objectives.json" \
  --operator-rollout "$COMPARISON_ROOT/operator-autonomous-training-support/rollout_summary.json" \
  --discrete-rollout "$COMPARISON_ROOT/discrete-autonomous-training-support/rollout_summary.json" \
  --output-json "$COMPARISON_ROOT/fair_longfit_comparison.json" \
  --output-markdown "$COMPARISON_ROOT/fair_longfit_comparison.md"

echo '{"event":"fair_longfit_pipeline_complete"}'
