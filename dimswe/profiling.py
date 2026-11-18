"""Lightweight nested timing regions for profiling the optimizer and the timesteppers.

Used by docs/hvp_performance_plan.md (section 2) to find where objective, gradient and
Hessian-vector-product evaluations spend their time. Code is instrumented with

    with region("forward stages"):
        ...

Every region does two things:

- it opens a PETSc log event of the same name, if petsc4py has already been imported (by
  Firedrake). With ``-log_view :flamegraph.txt:ascii_flamegraph`` the regions then appear, nested,
  in the PETSc flamegraph next to PETSc's own events (KSPSolve, assembly, ...);
- if timing is enabled (``enable()``), it adds its wall time and call count to an in-process
  table keyed by the path of enclosing regions, e.g. ``("hessp", "second-order adjoint sweep",
  "substep adjoints", "0:RK4(model)", "soa stages")``. This works without ``-log_view`` and lets a
  driver separate setup from steady-state costs by resetting the table between evaluations.

Timing is disabled by default, so instrumented code only pays for a PETSc event begin/end (or
nothing, without petsc4py) when no profile is requested. Times are wall-clock seconds on the
calling rank; under MPI the collective solves keep the ranks in step, so rank 0 is representative.

This module deliberately does not import petsc4py itself: importing it before Firedrake would
initialize PETSc without the command-line options (such as -log_view).
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
import sys
import time
from typing import Any, Callable, Iterator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# separator used when region paths are written as strings (region names must not contain it)
PATH_SEPARATOR = "/"

_enabled = False
# names of the currently open regions, outermost first
_stack: list[str] = []
# path tuple -> [calls, inclusive seconds]
_table: dict[tuple[str, ...], list[float]] = {}
# region name -> PETSc.Log.Event, created once per name
_petsc_events: dict[str, object] = {}
# wall clock used for the timing table (replaceable in tests)
_clock: Callable[[], float] = time.perf_counter


def enable(flag: bool = True) -> None:
    """Turn the in-process timing table on (or off with flag=False)."""
    global _enabled
    _enabled = bool(flag)


def is_enabled() -> bool:
    """Return True if regions are currently recorded in the timing table."""
    return _enabled


def now() -> float:
    """Return the current reading of the clock used by the timing table (seconds)."""
    return _clock()


def reset() -> None:
    """Clear the timing table. Regions open at the time are added again when they exit."""
    _table.clear()


def _petsc_event(name: str):
    """Return the PETSc log event for a region name, or None if petsc4py is not loaded yet."""
    event = _petsc_events.get(name)
    if event is not None:
        return event
    petsc = sys.modules.get("petsc4py.PETSc")
    if petsc is None:
        return None
    event = petsc.Log.Event(name)
    _petsc_events[name] = event
    return event


@contextmanager
def region(name: str) -> Iterator[None]:
    """Time the enclosed block as a region nested inside the currently open regions.

    The name should describe the work (for example "forward stages") rather than the call site,
    so the same work is summed over call sites by totals_by_name. It must not contain
    PATH_SEPARATOR.
    """
    event = _petsc_event(name)
    if event is not None:
        event.begin()
    if not _enabled:
        try:
            yield
        finally:
            if event is not None:
                event.end()
        return
    _stack.append(name)
    path = tuple(_stack)
    # create the entry on the way in, so the table is ordered by first entry (parents before
    # children, siblings in execution order)
    entry = _table.setdefault(path, [0, 0.0])
    start = _clock()
    try:
        yield
    finally:
        elapsed = _clock() - start
        # after a reset() inside this region the entry is no longer in the table; add it again
        entry = _table.setdefault(path, [0, 0.0])
        entry[0] += 1
        entry[1] += elapsed
        _stack.pop()
        if event is not None:
            event.end()


def timed(name: str) -> Callable[[F], F]:
    """Decorator form of region: time every call of the decorated function as region name."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with region(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def snapshot() -> dict[str, dict[str, float]]:
    """Return a copy of the timing table as {"a/b/c": {"calls": n, "seconds": s}}.

    Seconds are inclusive (they contain the nested regions). Entries are in order of first
    entry, so parents come before their children and siblings appear in execution order. Regions
    that are still open are left out.
    """
    return {
        PATH_SEPARATOR.join(path): {"calls": int(calls), "seconds": float(seconds)}
        for path, (calls, seconds) in _table.items()
        if calls > 0
    }


def _split(path: str) -> tuple[str, ...]:
    """Split a snapshot key back into its region names."""
    return tuple(path.split(PATH_SEPARATOR))


def self_seconds(table: dict[str, dict[str, float]]) -> dict[str, float]:
    """Return each region's exclusive time: its seconds minus those of its direct children.

    The exclusive time of a region is the work done in it outside any nested region, for example
    UFL expression building and numpy copies around the instrumented solves.
    """
    result = {path: entry["seconds"] for path, entry in table.items()}
    for path, entry in table.items():
        parts = _split(path)
        if len(parts) > 1:
            parent = PATH_SEPARATOR.join(parts[:-1])
            if parent in result:
                result[parent] -= entry["seconds"]
    return result


def totals_by_name(table: dict[str, dict[str, float]], under: str | None = None) -> dict[str, dict[str, float]]:
    """Sum calls and inclusive seconds of all regions with the same (last) name.

    If under is given, only regions strictly inside the region path under (a snapshot key, such
    as "hessp") are counted. Region names never nest inside themselves in the instrumented code,
    so the inclusive sums do not double count. Returns {name: {"calls": n, "seconds": s}}, sorted
    by decreasing seconds.
    """
    prefix = None if under is None else _split(under)
    totals: dict[str, dict[str, float]] = {}
    for path, entry in table.items():
        parts = _split(path)
        if prefix is not None and (len(parts) <= len(prefix) or parts[:len(prefix)] != prefix):
            continue
        total = totals.setdefault(parts[-1], {"calls": 0, "seconds": 0.0})
        total["calls"] += entry["calls"]
        total["seconds"] += entry["seconds"]
    return dict(sorted(totals.items(), key=lambda item: -item[1]["seconds"]))


def format_tree(table: dict[str, dict[str, float]], min_fraction: float = 0.0) -> str:
    """Format a snapshot as an indented tree with calls, inclusive and exclusive seconds.

    Each line shows the region's share of its top-level region. Regions below min_fraction of
    their top-level region are omitted (with their children), to keep large profiles readable.
    """
    exclusive = self_seconds(table)
    lines = [f"{'region':<72} {'calls':>7} {'seconds':>10} {'self':>10} {'share':>7}"]
    hidden: list[tuple[str, ...]] = []
    for path, entry in table.items():
        parts = _split(path)
        if any(parts[:len(h)] == h for h in hidden):
            continue
        top = table.get(parts[0], entry)["seconds"]
        share = entry["seconds"] / top if top > 0 else 0.0
        if len(parts) > 1 and share < min_fraction:
            hidden.append(parts)
            continue
        label = "  " * (len(parts) - 1) + parts[-1]
        lines.append(f"{label:<72} {entry['calls']:>7d} {entry['seconds']:>10.3f} {exclusive[path]:>10.3f} {share:>7.1%}")
    return "\n".join(lines)
