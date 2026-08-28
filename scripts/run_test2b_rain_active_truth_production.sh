#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
ENVIRONMENT="/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312"
CONFIGURATION="$REPOSITORY/dimswe/configs/test2b_rain_active_case.json"
OUTPUT_ROOT="$REPOSITORY/external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000"

cd "$REPOSITORY"
source "$ENVIRONMENT/bin/activate"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"

TEST2B_CASE_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2b-rain-case.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2B_CASE_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2B_CASE_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2B_CASE_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2B_CASE_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TEST2B_CASE_CACHE_ROOT/pycache"

python -u -m dimswe.test2b_rain_case_design validate-configuration \
  --configuration "$CONFIGURATION"

python -u -m dimswe.resolved_hidden_c0_driver run \
  --case doublevortex \
  --nx 64 --ny 64 \
  --dt 100 \
  --nsteps 160 \
  --output-stride 1 \
  --c0 0.14 \
  --s 3.2 \
  --moist-backend ufl \
  --seed 0 \
  --initial-moisture-zeta -0.06 \
  --spectral-nx 128 --spectral-ny 128 \
  --high-wavenumber-fraction 0.6666666666666666 \
  --output-directory "$OUTPUT_ROOT"

python -u -m dimswe.test2b_rain_truth_driver audit-run \
  --run "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/rain_activity_audit.json"

python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
audit = json.loads(
    (root / "rain_activity_audit.json").read_text(encoding="utf-8")
)
if metadata.get("status") != "complete" or audit.get("status") != "complete":
    raise SystemExit("Test2B rain-active truth or its audit did not complete")
if metadata["configuration"]["initial_moisture_zeta"] != -0.06:
    raise SystemExit("completed truth has the wrong initial-moisture control")
print(
    json.dumps(
        {
            "truth_status": metadata["status"],
            "rain_audit_status": audit["status"],
            "rain_summary": audit["summary"],
        },
        indent=2,
        sort_keys=True,
    )
)
PY
