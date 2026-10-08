#!/bin/bash
set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
STAGE="$REPOSITORY/scripts/run_test2a_fiml_stage.sh"
ROOT="$REPOSITORY/external-results/test2a/fiml-sparse-endpoint-h2-h5"

for name in direct-h2 direct-h5 fi-h2 fi-h5 pseudo-h2 pseudo-h5 stage2-h2 stage2-h5; do
  "$STAGE" "$name"
done

for required in \
  "$ROOT/direct-endpoint-h2/fit_result.json" \
  "$ROOT/direct-endpoint-h5/fit_result.json" \
  "$ROOT/field-inversion/h2/field_inversion_summary.json" \
  "$ROOT/field-inversion/h5/field_inversion_summary.json" \
  "$ROOT/stage2/h2/fit_result.json" \
  "$ROOT/stage2/h5/fit_result.json"; do
  test -s "$required"
done

"$REPOSITORY/scripts/run_test2a_fiml_postprocess.sh"
