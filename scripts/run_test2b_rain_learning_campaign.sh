#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/reproduction_environment.sh"
cd "$REPOSITORY"
for representation in A B C; do
  bash scripts/run_test2b_rain_learning_representation.sh "$representation"
done

CONFIGURATION="dimswe/configs/test2b_rain_active_learning.json"
PREPARATION="external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz"
OUTPUT_ROOT="external-results/test2b-rain-active-learning/production"
artifacts=()
for representation in A B C; do
  root="$OUTPUT_ROOT/representation-${representation}"
  artifacts+=(
    --artifact "$representation:M1=$root/m1-seed0-m20-10k/final_parameters.npz"
    --artifact "$representation:M2-X-independent=$root/m2x-seed0-m20-10k/final_parameters.npz"
    --artifact "$representation:M1-to-M2-X=$root/m1-to-m2x-m20-5k/final_parameters.npz"
    --artifact "$representation:H1=$root/h1-from-m1-m20-5k/final_parameters.npz"
    --artifact "$representation:H2=$root/h2-from-h1-m20-20/final_parameters.npz"
    --artifact "$representation:H5=$root/h5-from-h2-m20-20/final_parameters.npz"
  )
done
"$PYTHON" -u -m dimswe.test2b_rain_learning_campaign postprocess \
  --configuration "$CONFIGURATION" --preparation "$PREPARATION" \
  --output "$OUTPUT_ROOT/test2b_rain_active_comparison.json" \
  "${artifacts[@]}"
