"""Tests for the nested timing regions of dimswe/profiling.py (HVP performance plan, section 2).

The module is loaded through workflows.profile_hvp.profiling_module, which does not import the
dimswe package, so these tests need neither Firedrake nor PETSc. A fake clock that advances by
one second per reading makes the recorded times deterministic.
"""

import pytest

from workflows.profile_hvp import profiling_module

profiling = profiling_module()


class FakeClock:
    """A clock that advances by a fixed step on every reading."""

    def __init__(self, step=1.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


@pytest.fixture
def table(monkeypatch):
    """Enable timing with a fake clock and an empty table; restore the module state afterwards."""
    monkeypatch.setattr(profiling, "_clock", FakeClock())
    was_enabled = profiling.is_enabled()
    profiling.reset()
    profiling.enable()
    yield profiling
    profiling.enable(was_enabled)
    profiling.reset()


def test_regions_record_nothing_when_disabled(table):
    table.enable(False)
    with table.region("outer"):
        with table.region("inner"):
            pass
    assert table.snapshot() == {}


def test_nested_regions_are_keyed_by_path_and_count_calls(table):
    for _ in range(3):
        with table.region("sweep"):
            with table.region("stages"):
                pass
    snap = table.snapshot()
    assert list(snap) == ["sweep", "sweep/stages"]
    assert snap["sweep"]["calls"] == 3 and snap["sweep/stages"]["calls"] == 3
    # each region reads the clock twice; the inner readings lie inside the outer ones
    assert snap["sweep/stages"]["seconds"] == pytest.approx(3.0)
    assert snap["sweep"]["seconds"] == pytest.approx(9.0)


def test_table_is_in_first_entry_order(table):
    with table.region("a"):
        with table.region("recompute"):
            pass
        with table.region("adjoints"):
            with table.region("stages"):
                pass
        with table.region("recompute"):
            pass
    assert list(table.snapshot()) == ["a", "a/recompute", "a/adjoints", "a/adjoints/stages"]


def test_same_name_in_different_parents_is_kept_apart(table):
    with table.region("jac"):
        with table.region("forward stages"):
            pass
    with table.region("hessp"):
        with table.region("forward stages"):
            pass
    snap = table.snapshot()
    assert "jac/forward stages" in snap and "hessp/forward stages" in snap


def test_self_seconds_subtract_direct_children(table):
    with table.region("outer"):
        with table.region("first"):
            pass
        with table.region("second"):
            with table.region("deep"):
                pass
    snap = table.snapshot()
    exclusive = table.self_seconds(snap)
    assert exclusive["outer"] == pytest.approx(snap["outer"]["seconds"] - snap["outer/first"]["seconds"]
                                               - snap["outer/second"]["seconds"])
    assert exclusive["outer/second"] == pytest.approx(snap["outer/second"]["seconds"] - snap["outer/second/deep"]["seconds"])
    assert exclusive["outer/first"] == pytest.approx(snap["outer/first"]["seconds"])


def test_totals_by_name_sum_over_call_sites_and_respect_under(table):
    for top in ("jac", "hessp"):
        with table.region(top):
            for label in ("0:RK4(model)", "1:Euler(hyperviscosity)"):
                with table.region(label):
                    with table.region("forward stages"):
                        pass
    snap = table.snapshot()
    everywhere = table.totals_by_name(snap)
    assert everywhere["forward stages"]["calls"] == 4
    under_hessp = table.totals_by_name(snap, under="hessp")
    assert under_hessp["forward stages"]["calls"] == 2
    assert "hessp" not in under_hessp and "jac" not in under_hessp
    # sorted by decreasing time
    seconds = [entry["seconds"] for entry in under_hessp.values()]
    assert seconds == sorted(seconds, reverse=True)


def test_timed_decorator_times_calls_and_keeps_the_function(table):
    @table.timed("work")
    def work(value):
        """Return value doubled."""
        return 2 * value

    assert work(3) == 6 and work(4) == 8
    assert work.__name__ == "work" and work.__doc__ == "Return value doubled."
    assert table.snapshot()["work"]["calls"] == 2


def test_regions_are_closed_when_the_body_raises(table):
    with pytest.raises(RuntimeError):
        with table.region("outer"):
            with table.region("failing"):
                raise RuntimeError("solver failed")
    with table.region("after"):
        pass
    snap = table.snapshot()
    assert snap["outer/failing"]["calls"] == 1
    # the stack was unwound, so the next region is top-level again
    assert "after" in snap


def test_reset_inside_a_region_keeps_the_open_region(table):
    with table.region("outer"):
        table.reset()
        with table.region("inner"):
            pass
    snap = table.snapshot()
    assert snap["outer"]["calls"] == 1 and snap["outer/inner"]["calls"] == 1


def test_format_tree_indents_and_hides_small_regions(table, monkeypatch):
    monkeypatch.setattr(profiling, "_clock", FakeClock(step=1.0))
    with table.region("hessp"):
        with table.region("sweep"):
            with table.region("soa stages"):
                pass
    snap = table.snapshot()
    # make one child negligible: 1% of the top-level region
    snap["hessp/tiny"] = {"calls": 1, "seconds": 0.01 * snap["hessp"]["seconds"]}
    text = table.format_tree(snap, min_fraction=0.05)
    lines = text.splitlines()
    assert lines[0].split()[:2] == ["region", "calls"]
    assert lines[1].startswith("hessp")
    assert lines[2].startswith("  sweep") and lines[3].startswith("    soa stages")
    assert "tiny" not in text
    assert "tiny" in table.format_tree(snap, min_fraction=0.0)
