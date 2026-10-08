"""Portable paths for the accepted deployed-hybrid replay/render pipeline."""

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
REFERENCE_REPOSITORY = _configured(
    "DIMSWE_REFERENCE_REPOSITORY", REPOSITORY_ROOT
)
M1Y_REPOSITORY = _configured(
    "DIMSWE_M1Y_REPOSITORY", REPOSITORY_ROOT
)
GROUND_TRUTH_PACKAGE = _configured(
    "DIMSWE_GROUND_TRUTH_PACKAGE",
    WORKSTREAM_ROOT / "ground_truth_figures_20260829",
)
TRUTH_MAP_CACHE = (
    GROUND_TRUTH_PACKAGE
    / "outputs/ground_truth_figures_20260829/data/test2b_truth_maps.npz"
)
