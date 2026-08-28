#!/usr/bin/env bash
# Shared, machine-independent setup for the frozen campaign entry points.
#
# Source this file from another script. The caller may set:
#   DIMSWE_REPOSITORY           repository root (defaults to this script's parent)
#   DIMSWE_VIRTUAL_ENVIRONMENT  environment directory to activate (optional)
#   DIMSWE_PYTHON               Python executable (defaults to python)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/reproduction_environment.sh from a campaign script" >&2
  exit 2
fi

DIMSWE_SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="${DIMSWE_REPOSITORY:-$(dirname "$DIMSWE_SCRIPT_DIRECTORY")}"

if ! REPOSITORY="$(CDPATH= cd -- "$REPOSITORY" 2>/dev/null && pwd)"; then
  echo "DIMSWE repository root is unavailable: ${DIMSWE_REPOSITORY:-unset}" >&2
  return 2
fi
if [[ ! -d "$REPOSITORY/dimswe" || ! -d "$REPOSITORY/scripts" ]]; then
  echo "DIMSWE repository root is invalid: $REPOSITORY" >&2
  return 2
fi

if [[ -n "${DIMSWE_VIRTUAL_ENVIRONMENT:-}" ]]; then
  if [[ ! -f "$DIMSWE_VIRTUAL_ENVIRONMENT/bin/activate" ]]; then
    echo "DIMSWE virtual environment is invalid: $DIMSWE_VIRTUAL_ENVIRONMENT" >&2
    return 2
  fi
  # shellcheck disable=SC1090
  source "$DIMSWE_VIRTUAL_ENVIRONMENT/bin/activate"
fi

PYTHON="${DIMSWE_PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "DIMSWE Python executable is unavailable: $PYTHON" >&2
  return 2
fi

export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
