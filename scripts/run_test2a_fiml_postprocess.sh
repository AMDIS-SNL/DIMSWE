#!/bin/bash
set -euo pipefail

REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
VIRTUAL_ENVIRONMENT="/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312"
ROOT="$REPOSITORY/external-results/test2a/fiml-sparse-endpoint-h2-h5"
CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_fiml_sparse_endpoint_h2_h5.json"
CURRICULUM_CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"
H1_CACHE="$REPOSITORY/external-results/test2a/horizon-curriculum-h1-h2-h5/h1_postprefix_cache.npz"
AUTONOMOUS_CONFIGURATION="$REPOSITORY/dimswe/configs/test2a_apriori_autonomous.json"
POSTPROCESS="$ROOT/postprocess"
MANIFEST="$POSTPROCESS/network_manifest.tsv"

cd "$REPOSITORY"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPOSITORY"
RUN_CACHE_ROOT="${TEST2A_FIML_CACHE_ROOT:-$(mktemp -d /tmp/dimswe-test2a-fiml-post.XXXXXX)}"
export PYOP2_CACHE_DIR="$RUN_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$RUN_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$RUN_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$RUN_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_CACHE_ROOT/pycache"
mkdir -p "$POSTPROCESS/sparse" "$POSTPROCESS/dense" "$POSTPROCESS/autonomous"

python - "$ROOT" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
entries = [
    (
        "h1-baseline",
        Path("external-results/test2a/horizon-curriculum-h1-h2-h5/h1-from-m1-200k/final_parameters.npz"),
        "ebc49083bda299d91e614adeaeefdda0400ca1e8cfccc95a3b4ba953044f963c",
    )
]
for label, result in (
    ("direct-h2", root / "direct-endpoint-h2/fit_result.json"),
    ("direct-h5", root / "direct-endpoint-h5/fit_result.json"),
    ("fiml-h2", root / "stage2/h2/fit_result.json"),
    ("fiml-h5", root / "stage2/h5/fit_result.json"),
):
    record = json.load(result.open(encoding="utf-8"))
    assert record["status"] == "complete"
    entries.append((label, Path(record["final_parameter_file"]), record["final_parameter_pytree_sha256"]))
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    for label, path, fingerprint in entries:
        stream.write(f"{label}\t{path.resolve()}\t{fingerprint}\n")
PY

while IFS=$'\t' read -r label parameter_file parameter_sha; do
  if [[ ! -e "$POSTPROCESS/sparse/$label.json" ]]; then
    python -u -m dimswe.test2a_fiml cross-evaluate \
      --configuration "$CONFIGURATION" --parameter-file "$parameter_file" \
      --expected-sha256 "$parameter_sha" --output "$POSTPROCESS/sparse/$label.json"
  fi
  if [[ ! -e "$POSTPROCESS/dense/$label.json" ]]; then
    python -u -m dimswe.test2a_horizon_curriculum cross-evaluate \
      --configuration "$CURRICULUM_CONFIGURATION" --h1-cache "$H1_CACHE" \
      --parameter-file "$parameter_file" --expected-sha256 "$parameter_sha" \
      --output "$POSTPROCESS/dense/$label.json"
  fi
  if [[ ! -e "$POSTPROCESS/autonomous/$label/rollout_summary.json" ]]; then
    python -u -m dimswe.test2a_apriori_autonomous \
      --configuration "$AUTONOMOUS_CONFIGURATION" \
      --parameter-file "$parameter_file" --expected-pytree-sha256 "$parameter_sha" \
      --output-directory "$POSTPROCESS/autonomous/$label"
  fi
done < "$MANIFEST"

python -u -m dimswe.test2a_fiml report --root "$ROOT" \
  --output-json "$POSTPROCESS/fiml_sparse_endpoint_report.json" \
  --output-markdown "$POSTPROCESS/fiml_sparse_endpoint_report.md"
