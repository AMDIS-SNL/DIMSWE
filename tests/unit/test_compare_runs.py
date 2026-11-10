"""Unit tests of the pure-Python option and colour-limit logic in workflows.compare_runs."""

import numpy as np
import pytest
import yaml

from workflows.common import WORKFLOWS_DIR
from workflows.compare_runs import combine_limits, grid_axis_points, merge_limits, resolve_comparison_options


def test_resolve_comparison_options_defaults_to_first_run():
    """Without a reference key, differences are taken against the first run."""
    options = resolve_comparison_options({}, ["CF", "LP"])
    assert options == {"reference": "CF", "differences": True, "samples_per_cell": 4}


def test_resolve_comparison_options_uses_explicit_reference():
    """An explicit reference label is used as given."""
    options = resolve_comparison_options({"reference": "LP"}, ["CF", "LP"])
    assert options["reference"] == "LP"


def test_resolve_comparison_options_disables_differences_for_one_run():
    """A single run has nothing to difference against."""
    options = resolve_comparison_options({}, ["CF"])
    assert options["differences"] is False


def test_resolve_comparison_options_respects_differences_false():
    """differences: false turns the difference panels off."""
    options = resolve_comparison_options({"differences": False}, ["CF", "LP"])
    assert options["differences"] is False


@pytest.mark.parametrize(
    "compare_cfg, labels",
    [
        ({"reference": "H1"}, ["CF", "LP"]),
        ({"differences": "yes"}, ["CF", "LP"]),
        ({}, ["CF", "CF"]),
        ({"samples_per_cell": 0}, ["CF", "LP"]),
        ({"samples_per_cell": 2.5}, ["CF", "LP"]),
    ],
)
def test_resolve_comparison_options_rejects_invalid_options(compare_cfg, labels):
    """Unknown references, non-boolean flags, duplicate labels and bad sample counts are errors."""
    with pytest.raises(ValueError):
        resolve_comparison_options(compare_cfg, labels)


def test_combine_limits_shares_range_and_symmetrizes_differences():
    """The value range spans all runs; the difference limit is the largest absolute difference."""
    (low, high), limit = combine_limits(
        [np.array([1.0, 3.0]), np.array([0.5, 2.0])],
        [np.array([-0.2, 0.1]), np.array([0.05, -0.4])],
    )
    assert (low, high) == (0.5, 3.0)
    assert limit == pytest.approx(0.4)


def test_combine_limits_handles_degenerate_inputs():
    """Constant fields and zero differences still give valid, non-empty colour ranges."""
    (low, high), limit = combine_limits([np.array([2.0, 2.0])], [np.zeros(3)])
    assert high > low == 2.0
    assert limit == 1.0
    assert combine_limits([], []) == ((0.0, 1.0), 1.0)


def test_merge_limits_covers_both_inputs():
    """Merged limits contain both value ranges and the larger difference limit."""
    merged = merge_limits(((0.0, 1.0), 0.1), ((-1.0, 0.5), 0.3))
    assert merged == ((-1.0, 1.0), 0.3)
    assert merge_limits(None, ((0.0, 1.0), 0.1)) == ((0.0, 1.0), 0.1)


def test_example_compare_mtswe_cfg_compares_cf_lp_and_h1():
    """The CF/LP/H1 example config is valid and takes differences against CF."""
    data = yaml.safe_load((WORKFLOWS_DIR / "examples" / "compare_mtswe.cfg").read_text(encoding="utf-8"))
    labels = [run["label"] for run in data["runs"]]
    assert labels == ["CF", "LP", "H1"]
    options = resolve_comparison_options(data, labels)
    assert options["reference"] == "CF" and options["differences"] is True


def test_combine_limits_ignores_nan_samples():
    """Sample points outside the mesh (NaN) do not affect the difference limit."""
    _, limit = combine_limits([np.array([0.0, 1.0])], [np.array([[np.nan, -0.25], [0.1, np.nan]])])
    assert limit == pytest.approx(0.25)


def test_grid_axis_points_avoid_cell_edges():
    """Sample points are evenly spaced and never fall on a cell edge of the uniform mesh."""
    points = grid_axis_points(0.0, 5.0, ncells=5, samples_per_cell=4)
    assert len(points) == 20
    assert np.allclose(np.diff(points), 0.25)
    offsets_in_cell = points - np.floor(points)
    assert np.all(offsets_in_cell > 1e-8) and np.all(offsets_in_cell < 1.0 - 1e-8)


def test_example_compare_mtswe_physics_cfg_compares_moisture_against_cf():
    """The physics example config compares CF, LP and H1 moist runs, with differences against CF."""
    data = yaml.safe_load((WORKFLOWS_DIR / "examples" / "compare_mtswe_physics.cfg").read_text(encoding="utf-8"))
    labels = [run["label"] for run in data["runs"]]
    assert labels == ["CF", "LP", "H1"]
    assert all(run["run_dir"].endswith("-physics") for run in data["runs"])
    options = resolve_comparison_options(data, labels)
    assert options["reference"] == "CF" and options["differences"] is True
    assert {"Qv", "Qc", "Qr"} <= set(data["quantities"]["fields"])
    assert "total_water" in data["quantities"]["statistics"]
