#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
CONFIGURATION="dimswe/configs/test2a_problem_b.json"
PREPARATION="external-results/test2a/problem-b/preparation/problem_b_fixed_data.npz"
RECURSIVE_BENCHMARK="external-results/test2a/problem-b/preparation/problem_b_recursive_benchmark.json"
RECURSIVE_SMOKE="external-results/test2a/problem-b/preparation/problem_b_recursive_smoke.json"
OUTPUT_ROOT="external-results/test2a/problem-b/production"

cd "$REPOSITORY"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
TEST2B_RUNTIME_CACHE="$(mktemp -d /tmp/dimswe-test2b-production.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2B_RUNTIME_CACHE/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2B_RUNTIME_CACHE/tsfc"
export XDG_CACHE_HOME="$TEST2B_RUNTIME_CACHE/xdg"
export MPLCONFIGDIR="$TEST2B_RUNTIME_CACHE/matplotlib"
export PYTHONPYCACHEPREFIX="$TEST2B_RUNTIME_CACHE/pycache"

if [[ ! -f "$PREPARATION" || ! -f "${PREPARATION%.npz}.json" ]]; then
  echo "Problem-B preparation NPZ/JSON pair is missing" >&2
  exit 2
fi
"$PYTHON" - "$RECURSIVE_BENCHMARK" "$RECURSIVE_SMOKE" <<'PY'
import json
import pathlib
import sys
benchmark = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
smoke = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if benchmark.get("status") != "complete" or not {"H2", "H5"}.issubset(
    benchmark.get("timings", {})
):
    raise SystemExit("Problem-B recursive benchmark gate is incomplete")
if smoke.get("status") != "complete" or not {"H2", "H5"}.issubset(
    smoke.get("stages", {})
):
    raise SystemExit("Problem-B recursive smoke gate is incomplete")
for stage in ("H2", "H5"):
    record = smoke["stages"][stage]
    if not record.get("objective_decreased") or record.get("HVP_evaluations") != 0:
        raise SystemExit(f"Problem-B {stage} smoke gate failed")
PY
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing Problem-B production root: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

run_seed_stage() {
  local stage="$1"
  local limit="$2"
  local directory="$3"
  "$PYTHON" -u -m dimswe.test2a_problem_b_campaign train \
    --configuration "$CONFIGURATION" \
    --preparation "$PREPARATION" \
    --stage "$stage" \
    --output-directory "$directory" \
    --iteration-limit "$limit"
}

parameter_sha() {
  "$PYTHON" - "$1" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
if record.get("parameter_pytree_sha256") is None:
    raise SystemExit("parameter sidecar lacks pytree SHA256")
print(record["parameter_pytree_sha256"])
PY
}

run_warm_stage() {
  local stage="$1"
  local limit="$2"
  local directory="$3"
  local source="$4"
  local expected
  expected="$(parameter_sha "$source")"
  "$PYTHON" -u -m dimswe.test2a_problem_b_campaign train \
    --configuration "$CONFIGURATION" \
    --preparation "$PREPARATION" \
    --stage "$stage" \
    --output-directory "$directory" \
    --iteration-limit "$limit" \
    --initial-parameters "$source" \
    --expected-initial-sha256 "$expected"
}

# Matched seed-zero objectives: retain both to expose optimizer-basin effects.
run_seed_stage "M1" 200000 "$OUTPUT_ROOT/m1-seed0-m20-200k"
run_seed_stage "M2-X-independent" 200000 "$OUTPUT_ROOT/m2x-seed0-m20-200k"

M1_FINAL="$OUTPUT_ROOT/m1-seed0-m20-200k/final_parameters.npz"
run_warm_stage "M1-to-M2-X" 50000 \
  "$OUTPUT_ROOT/m1-to-m2x-m20-50k" "$M1_FINAL"

# Dense deployment-location/recursive curriculum.  Every stage is a new ROL
# process and receives parameters only; no L-BFGS secant history is transferred.
run_warm_stage "H1" 50000 "$OUTPUT_ROOT/h1-from-m1" "$M1_FINAL"
H1_FINAL="$OUTPUT_ROOT/h1-from-m1/final_parameters.npz"
run_warm_stage "H2" 100 "$OUTPUT_ROOT/h2-from-h1" "$H1_FINAL"
H2_FINAL="$OUTPUT_ROOT/h2-from-h1/final_parameters.npz"
run_warm_stage "H5" 100 "$OUTPUT_ROOT/h5-from-h2" "$H2_FINAL"

"$PYTHON" -u -m dimswe.test2a_problem_b_campaign postprocess \
  --configuration "$CONFIGURATION" \
  --preparation "$PREPARATION" \
  --output "$OUTPUT_ROOT/problem_b_comparison.json" \
  --artifact "M1=$M1_FINAL" \
  --artifact "M2-X-independent=$OUTPUT_ROOT/m2x-seed0-m20-200k/final_parameters.npz" \
  --artifact "M1-to-M2-X=$OUTPUT_ROOT/m1-to-m2x-m20-50k/final_parameters.npz" \
  --artifact "H1=$H1_FINAL" \
  --artifact "H2=$H2_FINAL" \
  --artifact "H5=$OUTPUT_ROOT/h5-from-h2/final_parameters.npz"

printf '%s\n' "Problem-B five-objective production stages and postprocessing completed."
