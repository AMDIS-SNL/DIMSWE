#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 HORIZON INITIAL_PARAMETERS EXPECTED_INITIAL_SHA SOURCE_STAGE OUTPUT_DIRECTORY LOG_FILE [--resume]" >&2
  exit 2
fi

HORIZON="$1"
INITIAL_PARAMETERS="$2"
EXPECTED_INITIAL_SHA="$3"
SOURCE_STAGE="$4"
OUTPUT_DIRECTORY="$5"
LOG_FILE="$6"
RESUME_FLAG="${7:-}"
if [[ "$RESUME_FLAG" != "" && "$RESUME_FLAG" != "--resume" ]]; then
  echo "the only optional argument is --resume" >&2
  exit 2
fi

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"
H1_CACHE="$REPOSITORY/external-results/test2a/horizon-curriculum-h1-h2-h5/h1_postprefix_cache.npz"

cd "$REPOSITORY"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"

RUN_CACHE_ROOT="${TEST2A_CURRICULUM_CACHE_ROOT:-$(mktemp -d /tmp/dimswe-test2a-horizon.XXXXXX)}"
export PYOP2_CACHE_DIR="$RUN_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUN_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_CACHE_ROOT/pycache"

mkdir -p "$OUTPUT_DIRECTORY" "$(dirname "$LOG_FILE")"

"$PYTHON" - "$HORIZON" "$INITIAL_PARAMETERS" "$EXPECTED_INITIAL_SHA" <<'PY'
import sys

from dimswe.test2a_horizon_curriculum import (
    _canonical_json_sha256,
    _load_model_context,
    _load_parameter,
    load_curriculum_configuration,
    load_h1_cache,
)

horizon = int(sys.argv[1])
parameter_file = sys.argv[2]
expected_sha = sys.argv[3]
configuration = load_curriculum_configuration(
    "dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"
)
cache = load_h1_cache(
    "external-results/test2a/horizon-curriculum-h1-h2-h5/h1_postprefix_cache.npz",
    configuration,
)
model_configuration, _, _, _ = _load_model_context(configuration)
_load_parameter(parameter_file, expected_sha, model_configuration)
assert cache.metadata["cache_npz_sha256"] == (
    "b30bb0fd2c919734ca0ecba44e32a2c9bd40b491fb7abce0a0aa011a5ea99b89"
)
assert cache.metadata["common_denominator_sha256"] == (
    "10bda77bf2e003802c560ef1218fe28b17531da6b30e3f97cf22fa04a62d4753"
)
print(
    {
        "event": "horizon_stage_preflight_ok",
        "horizon": horizon,
        "initial_parameter_pytree_sha256": expected_sha,
        "configuration_sha256": _canonical_json_sha256(configuration),
        "h1_cache_npz_sha256": cache.metadata["cache_npz_sha256"],
        "common_denominator_sha256": cache.metadata[
            "common_denominator_sha256"
        ],
        "new_optimizer_process": True,
        "source_optimizer_secant_history_reused": False,
    },
    flush=True,
)
PY

if [[ -e "$OUTPUT_DIRECTORY/fit_result.json" ]]; then
  "$PYTHON" - "$OUTPUT_DIRECTORY/fit_result.json" "$HORIZON" "$EXPECTED_INITIAL_SHA" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
assert record["status"] == "complete"
assert int(record["horizon"]) == int(sys.argv[2])
assert record["initialization"]["source_parameter_pytree_sha256"] == sys.argv[3]
assert record["initialization"]["new_optimizer_process"] is True
assert record["initialization"]["source_optimizer_secant_history_reused"] is False
assert record["states_after_80_accessed"] is False
print({"event": "reusing_complete_horizon_stage", "horizon": int(sys.argv[2])})
PY
  exit 0
fi

if [[ -e "$OUTPUT_DIRECTORY/fit_progress.json" && "$RESUME_FLAG" != "--resume" ]]; then
  echo "incomplete parameter checkpoint exists; review it and rerun with --resume" >&2
  echo "parameter-only restart uses a new process and does not restore L-BFGS secant history" >&2
  exit 2
fi

COMMAND=(
  "$PYTHON" -u -m dimswe.test2a_horizon_curriculum train
  --configuration "$CONFIGURATION"
  --h1-cache "$H1_CACHE"
  --horizon "$HORIZON"
  --initial-parameter-file "$INITIAL_PARAMETERS"
  --expected-initial-sha256 "$EXPECTED_INITIAL_SHA"
  --source-stage "$SOURCE_STAGE"
  --output-directory "$OUTPUT_DIRECTORY"
)
if [[ "$RESUME_FLAG" == "--resume" ]]; then
  COMMAND+=(--resume)
fi
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
