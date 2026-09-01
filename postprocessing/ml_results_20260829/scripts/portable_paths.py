"""Portable path contract for the frozen ML-results postprocessors.

The original evaluation-only workspace recorded absolute production paths.
The collaborator snapshot keeps those strings in immutable provenance records,
but runnable scripts resolve inputs from this repository by default. Optional
environment variables allow an external artifact mirror without source edits.
"""

from __future__ import annotations

import os
from pathlib import Path


def _configured(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = _configured(
    "DIMSWE_REPOSITORY", PACKAGE_ROOT.parents[1]
)

# These repositories were separate during the accepted evaluation pass. In the
# collaborator snapshot their preserved source and compact artifacts coexist.
REFERENCE_REPOSITORY = _configured(
    "DIMSWE_REFERENCE_REPOSITORY", REPOSITORY_ROOT
)
M1Y_REPOSITORY = _configured(
    "DIMSWE_M1Y_REPOSITORY", REPOSITORY_ROOT
)
AUDIT_ROOT = _configured(
    "DIMSWE_ML_RESULTS_AUDIT_ROOT",
    WORKSTREAM_ROOT / "ml_results_audit_20260829",
)
GROUND_TRUTH_PACKAGE = _configured(
    "DIMSWE_GROUND_TRUTH_PACKAGE",
    WORKSTREAM_ROOT / "ground_truth_figures_20260829",
)
