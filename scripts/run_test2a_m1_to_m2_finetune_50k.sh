#!/bin/bash
set -euo pipefail

REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
VIRTUAL_ENVIRONMENT="/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312"
CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_m1_to_m2_finetune_50k.json"
CACHE="$REPOSITORY/external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz"
AUTONOMOUS_CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_apriori_autonomous.json"
OUTPUT_ROOT="$REPOSITORY/external-results/test2a/m1-to-m2-finetune"
FIT_ROOT="$OUTPUT_ROOT/operator-200k-to-discrete-m20-50k"
POSTPROCESS_ROOT="$OUTPUT_ROOT/postprocess"
FIT_LOG="$OUTPUT_ROOT/operator-200k-to-discrete-m20-50k.log"

resume_requested=0
if [[ "${1:-}" == "--resume" ]]; then
  resume_requested=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi

cd "$REPOSITORY"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"

RUN_CACHE_ROOT="${TEST2A_FINETUNE_CACHE_ROOT:-$(mktemp -d /tmp/dimswe-test2a-m1-to-m2.XXXXXX)}"
export PYOP2_CACHE_DIR="$RUN_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUN_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_CACHE_ROOT/pycache"

mkdir -p "$OUTPUT_ROOT"

python - <<'PY'
from dimswe.test2a_discrete_training import (
    load_discrete_training_configuration,
    load_fixed_cache,
    load_training_initial_parameters,
)
from dimswe.test2a_operator import (
    load_selected_configuration,
    mlp_configuration_from_record,
)
from dimswe.test2a_embedded_moist import parameter_pytree_sha256

configuration = load_discrete_training_configuration(
    'dimswe/configs/test2a_m1_to_m2_finetune_50k.json'
)
cache = load_fixed_cache(
    'external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz'
)
selected = load_selected_configuration(configuration['selected_operator_configuration'])
model = mlp_configuration_from_record(selected['model'])
initial = load_training_initial_parameters(configuration, model)
assert parameter_pytree_sha256(initial) == (
    'f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8'
)
assert cache.metadata['cache_npz_sha256'] == (
    'baee2dd3ae8a5e3f9ec16f6883e3583d4ac61281d777c3079b002e611504bacf'
)
print({
    'event': 'm1_to_m2_preflight_ok',
    'initial_parameter_sha256': parameter_pytree_sha256(initial),
    'cache_sha256': cache.metadata['cache_npz_sha256'],
    'source_optimizer_secant_history_reused': False,
}, flush=True)
PY

if [[ -e "$FIT_ROOT/fit_result.json" ]]; then
  if [[ "$resume_requested" -ne 0 ]]; then
    echo "fit is already complete; --resume is invalid" >&2
    exit 2
  fi
  python - <<PY
import json
record = json.load(open('$FIT_ROOT/fit_result.json', encoding='utf-8'))
assert record['status'] == 'complete'
assert record['initialization']['parameter_pytree_sha256'] == (
    'f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8'
)
PY
else
  if [[ -e "$FIT_ROOT/fit_progress.json" && "$resume_requested" -eq 0 ]]; then
    echo "incomplete parameter checkpoint exists; review it and rerun with --resume" >&2
    echo "a parameter-only restart starts with empty L-BFGS secant history" >&2
    exit 2
  fi
  if [[ "$resume_requested" -ne 0 ]]; then
    python -u -m dimswe.test2a_discrete_training train \
      --configuration "$CONFIGURATION" \
      --cache "$CACHE" \
      --output-directory "$FIT_ROOT" \
      --resume \
      2>&1 | tee "$FIT_LOG"
  else
    python -u -m dimswe.test2a_discrete_training train \
      --configuration "$CONFIGURATION" \
      --cache "$CACHE" \
      --output-directory "$FIT_ROOT" \
      2>&1 | tee "$FIT_LOG"
  fi
fi

python -u -m dimswe.test2a_m1_to_m2_finetune prepare-postprocess \
  --configuration "$CONFIGURATION" \
  --cache "$CACHE" \
  --fit-result "$FIT_ROOT/fit_result.json" \
  --output-directory "$POSTPROCESS_ROOT"

python - <<PY | while IFS=$'\t' read -r label parameter_file parameter_sha output_directory; do
import json
manifest = json.load(open('$POSTPROCESS_ROOT/autonomous_manifest.json', encoding='utf-8'))
for entry in manifest['entries']:
    print(entry['label'], entry['parameter_file'], entry['parameter_pytree_sha256'], entry['output_directory'], sep='\t')
PY
  if [[ -e "$output_directory/rollout_summary.json" ]]; then
    python - <<PY
import json
record = json.load(open('$output_directory/rollout_summary.json', encoding='utf-8'))
assert record['status'] == 'complete'
assert record['parameter_provenance']['parameter_pytree_sha256'] == '$parameter_sha'
assert record['deployment_contract']['states_after_80_accessed'] is False
PY
    echo "reusing complete autonomous evaluation for $label"
  else
    python -u -m dimswe.test2a_apriori_autonomous \
      --configuration "$AUTONOMOUS_CONFIGURATION" \
      --parameter-file "$parameter_file" \
      --expected-pytree-sha256 "$parameter_sha" \
      --output-directory "$output_directory"
  fi
done

python -u -m dimswe.test2a_m1_to_m2_finetune report \
  --postprocess-directory "$POSTPROCESS_ROOT" \
  --output-json "$POSTPROCESS_ROOT/m1_to_m2_finetune_report.json" \
  --output-markdown "$POSTPROCESS_ROOT/m1_to_m2_finetune_report.md"

echo '{"event":"m1_to_m2_finetune_pipeline_complete"}'
