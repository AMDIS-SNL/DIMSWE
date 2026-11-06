"""Checks that the shared test configuration stays small and otherwise matches tests/tswe.cfg.

The Firedrake tests build their problems from tests/gradient_helpers.DEFAULT_CFG, which is
tests/tswe_small.cfg: tests/tswe.cfg with an 8x8 instead of a 50x50 mesh, so the full suite runs
in reasonable time. These tests read the YAML only, so they need no Firedrake.
"""

from pathlib import Path

import yaml

TESTS_DIR = Path(__file__).resolve().parents[1]
SMALL_CFG = TESTS_DIR / "tswe_small.cfg"
FULL_CFG = TESTS_DIR / "tswe.cfg"
# upper bound on mesh cells for the shared test problem
MAX_CELLS = 100


def load(path):
    """Load a cfg file as YAML."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_gradient_helpers_use_the_small_config():
    source = (TESTS_DIR / "gradient_helpers.py").read_text(encoding="utf-8")
    assert "DEFAULT_CFG = 'tests/tswe_small.cfg'" in source


def test_small_config_has_a_small_mesh():
    mesh = load(SMALL_CFG)["mesh"]
    assert mesh["nx"] * mesh["ny"] <= MAX_CELLS


def test_small_config_differs_from_tswe_only_in_the_mesh_size():
    small, full = load(SMALL_CFG), load(FULL_CFG)
    for parameters in (small, full):
        parameters["mesh"].pop("nx")
        parameters["mesh"].pop("ny")
    assert small == full
