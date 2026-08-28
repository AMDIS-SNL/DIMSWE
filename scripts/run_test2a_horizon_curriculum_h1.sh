#!/bin/bash
set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
ROOT="$REPOSITORY/external-results/test2a/horizon-curriculum-h1-h2-h5"
INITIAL="$REPOSITORY/external-results/test2a/fair-longfit/operator-seed0-m20-200k/final_parameters.npz"
INITIAL_SHA="f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8"

exec bash "$REPOSITORY/scripts/run_test2a_horizon_curriculum_stage.sh" \
  1 "$INITIAL" "$INITIAL_SHA" "matched_M1_200k" \
  "$ROOT/h1-from-m1-200k" "$ROOT/h1-from-m1-200k.log" "${1:-}"
