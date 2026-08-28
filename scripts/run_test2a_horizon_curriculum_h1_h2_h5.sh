#!/bin/bash
set -euo pipefail

REPOSITORY="/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615"
ROOT="$REPOSITORY/external-results/test2a/horizon-curriculum-h1-h2-h5"
resume_requested=0
if [[ "${1:-}" == "--resume" ]]; then
  resume_requested=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi

run_stage() {
  local name="$1"
  local script="$2"
  local directory="$ROOT/$name"
  if [[ -e "$directory/fit_result.json" ]]; then
    bash "$script"
  elif [[ -e "$directory/fit_progress.json" ]]; then
    if [[ "$resume_requested" -ne 1 ]]; then
      echo "incomplete $name checkpoint exists; rerun wrapper with --resume" >&2
      exit 2
    fi
    bash "$script" --resume
  else
    bash "$script"
  fi
}

run_stage "h1-from-m1-200k" "$REPOSITORY/scripts/run_test2a_horizon_curriculum_h1.sh"
run_stage "h2-from-h1" "$REPOSITORY/scripts/run_test2a_horizon_curriculum_h2.sh"
run_stage "h5-from-h2" "$REPOSITORY/scripts/run_test2a_horizon_curriculum_h5.sh"
bash "$REPOSITORY/scripts/run_test2a_horizon_curriculum_postprocess.sh"

echo '{"event":"test2a_horizon_curriculum_pipeline_complete"}'

