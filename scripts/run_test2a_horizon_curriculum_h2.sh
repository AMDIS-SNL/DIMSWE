#!/bin/bash
set -euo pipefail

REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
ROOT="$REPOSITORY/external-results/test2a/horizon-curriculum-h1-h2-h5"
UPSTREAM="$ROOT/h1-from-m1-200k/fit_result.json"
cd "$REPOSITORY"
source "/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/activate"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
if [[ ! -e "$UPSTREAM" ]]; then
  echo "H1 must complete before H2" >&2
  exit 2
fi
read -r INITIAL INITIAL_SHA < <(python - "$UPSTREAM" <<'PY'
import json
import sys
from dimswe.test2a_horizon_curriculum import (
    _canonical_json_sha256,
    load_curriculum_configuration,
    load_h1_cache,
    validate_complete_stage_result,
)

configuration = load_curriculum_configuration(
    "dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"
)
cache = load_h1_cache(
    "external-results/test2a/horizon-curriculum-h1-h2-h5/h1_postprefix_cache.npz",
    configuration,
)
record = json.load(open(sys.argv[1], encoding="utf-8"))
path, fingerprint = validate_complete_stage_result(
    record,
    _canonical_json_sha256(configuration),
    1,
    cache.metadata["cache_npz_sha256"],
    "f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8",
)
print(path, fingerprint)
PY
)

exec bash "$REPOSITORY/scripts/run_test2a_horizon_curriculum_stage.sh" \
  2 "$INITIAL" "$INITIAL_SHA" "H1_fixed_postprefix_M2_Y" \
  "$ROOT/h2-from-h1" "$ROOT/h2-from-h1.log" "${1:-}"
