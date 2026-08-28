#!/bin/bash
set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
ROOT="$REPOSITORY/external-results/test2a/horizon-curriculum-h1-h2-h5"
CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"
H1_CACHE="$ROOT/h1_postprefix_cache.npz"
AUTONOMOUS_CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_apriori_autonomous.json"
POSTPROCESS="$ROOT/postprocess"
MANIFEST="$POSTPROCESS/stage_boundary_manifest.json"

cd "$REPOSITORY"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
RUN_CACHE_ROOT="${TEST2A_CURRICULUM_CACHE_ROOT:-$(mktemp -d /tmp/dimswe-test2a-horizon-post.XXXXXX)}"
export PYOP2_CACHE_DIR="$RUN_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUN_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_CACHE_ROOT/pycache"
mkdir -p "$POSTPROCESS/cross-objectives" "$POSTPROCESS/autonomous"

"$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

from dimswe.resolved_hidden_c0 import write_json_record
from dimswe.test2a_horizon_curriculum import (
    _canonical_json_sha256,
    load_curriculum_configuration,
    load_h1_cache,
    validate_complete_stage_result,
)

root = Path("external-results/test2a/horizon-curriculum-h1-h2-h5").resolve()
configuration = load_curriculum_configuration(
    "dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"
)
cache = load_h1_cache(root / "h1_postprefix_cache.npz", configuration)
configuration_sha = _canonical_json_sha256(configuration)
initial_file = Path(
    "external-results/test2a/fair-longfit/operator-seed0-m20-200k/final_parameters.npz"
).resolve()
initial_sha = "f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8"
entries = [("M1-200k-initial", initial_file, initial_sha)]
upstream_sha = initial_sha
for horizon, directory, label in (
    (1, "h1-from-m1-200k", "H1-final"),
    (2, "h2-from-h1", "H2-final"),
    (5, "h5-from-h2", "H5-final"),
):
    result_path = root / directory / "fit_result.json"
    if not result_path.exists():
        raise SystemExit(f"missing completed stage: {result_path}")
    record = json.load(result_path.open(encoding="utf-8"))
    parameter_file, parameter_sha = validate_complete_stage_result(
        record,
        configuration_sha,
        horizon,
        cache.metadata["cache_npz_sha256"],
        upstream_sha,
    )
    entries.append((label, Path(parameter_file), parameter_sha))
    upstream_sha = parameter_sha
manifest_entries = []
for label, parameter_file, parameter_sha in entries:
    safe = label.lower().replace("-", "_")
    manifest_entries.append(
        {
            "label": label,
            "parameter_file": str(parameter_file),
            "parameter_pytree_sha256": parameter_sha,
            "cross_objective_file": str(
                root / "postprocess" / "cross-objectives" / f"{safe}.json"
            ),
            "rollout_summary_file": str(
                root / "postprocess" / "autonomous" / safe / "rollout_summary.json"
            ),
        }
    )
write_json_record(
    sys.argv[1],
    {
        "status": "ready",
        "entries": manifest_entries,
        "autonomous_metrics_used_for_selection": False,
        "states_after_80_accessed": False,
    },
)
PY

"$PYTHON" - "$MANIFEST" <<'PY' | while IFS=$'\t' read -r label parameter_file parameter_sha cross_file rollout_file; do
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
for entry in record["entries"]:
    print(
        entry["label"],
        entry["parameter_file"],
        entry["parameter_pytree_sha256"],
        entry["cross_objective_file"],
        entry["rollout_summary_file"],
        sep="\t",
    )
PY
  if [[ ! -e "$cross_file" ]]; then
    "$PYTHON" -u -m dimswe.test2a_horizon_curriculum cross-evaluate \
      --configuration "$CONFIGURATION" \
      --h1-cache "$H1_CACHE" \
      --parameter-file "$parameter_file" \
      --expected-sha256 "$parameter_sha" \
      --output "$cross_file"
  fi
  rollout_directory="$(dirname "$rollout_file")"
  if [[ ! -e "$rollout_file" ]]; then
    "$PYTHON" -u -m dimswe.test2a_apriori_autonomous \
      --configuration "$AUTONOMOUS_CONFIGURATION" \
      --parameter-file "$parameter_file" \
      --expected-pytree-sha256 "$parameter_sha" \
      --output-directory "$rollout_directory"
  fi
done

"$PYTHON" -u -m dimswe.test2a_horizon_curriculum report \
  --manifest "$MANIFEST" \
  --output-json "$POSTPROCESS/horizon_curriculum_report.json" \
  --output-markdown "$POSTPROCESS/horizon_curriculum_report.md"
