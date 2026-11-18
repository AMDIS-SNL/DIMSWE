"""Pure-Python tests of the optimizer comparison tooling in workflows/compare_optimizers.py.

Fake optimization roots are written to tmp_path with result.yaml files in the layout written by
optimize_params.run_job, so no model runs are needed. Two roots solve the same problems, one with
L-BFGS-B and one with Newton-CG, like the paired examples in workflows/examples/.
"""

import math
from types import SimpleNamespace

import pytest

from workflows.common import dump_yaml, load_yaml, read_csv, to_builtin
from workflows.compare_optimizers import (
    COMPARISON_COLUMNS,
    COMPARISON_CSV,
    COMPARISON_YAML,
    TABLE_COLUMNS,
    collect_comparison_rows,
    comparison_row,
    format_comparison_table,
    write_comparison,
)
from workflows.optimize_params import DEFAULT_OPTIMIZER, optimizer_summary


def write_job(root, job_name, method, counts, final_loss, initial_loss=2.0, seconds=10.0, nit=5,
              success=True, hessp_used=None, target="ic-params"):
    """Write <root>/<job_name>/result.yaml for a finished job and return the job directory.

    counts is (nfev, njev, nhvp). The optimizer section is built with optimizer_summary, exactly
    as run_job builds it.
    """
    nfev, njev, nhvp = counts
    if hessp_used is None:
        hessp_used = nhvp > 0
    spec = {**DEFAULT_OPTIMIZER, "method": method}
    scipy_result = SimpleNamespace(success=success, status=0 if success else 1, message="done", nit=nit,
                                   hessp_used=hessp_used)
    optimizer = optimizer_summary(spec, scipy_result, SimpleNamespace(nfev=nfev, njev=njev, nhvp=nhvp), gtol=1.0e-8)
    result = {
        "job_name": job_name, "target": target, "loss_label": job_name.split("__")[-1],
        "truth_runs": [job_name.split("__")[0]], "optimizer": optimizer,
        "initial_loss": initial_loss, "final_loss": final_loss, "optimize_seconds": seconds,
        "parameters": {"initial-conditions.ox": {"initial": 0.1, "optimal": 0.11, "true": 0.11, "rel_error": 1.0e-6}},
    }
    job_dir = root / job_name
    dump_yaml(to_builtin(result), job_dir / "result.yaml")
    return job_dir


@pytest.fixture
def paired_roots(tmp_path):
    """Two optimization roots solving the same two problems with L-BFGS-B and Newton-CG."""
    first = tmp_path / "doublevortex-params"
    second = tmp_path / "doublevortex-params-newton-cg"
    write_job(first, "ox-0.1__final", "L-BFGS-B", (14, 14, 0), final_loss=2.0e-9, nit=12, seconds=30.0)
    write_job(first, "ox-0.1__trajectory-4", "L-BFGS-B", (20, 20, 0), final_loss=5.0e-8, nit=18, seconds=40.0)
    write_job(second, "ox-0.1__final", "Newton-CG", (6, 6, 15), final_loss=1.0e-9, nit=5, seconds=25.0)
    write_job(second, "ox-0.1__trajectory-4", "Newton-CG", (7, 7, 18), final_loss=1.0e-7, nit=6, seconds=50.0)
    return first, second


def test_rows_pair_methods_by_problem(paired_roots):
    rows = collect_comparison_rows([str(root) for root in paired_roots])
    assert [(row["problem"], row["method"]) for row in rows] == [
        ("ic-params:ox-0.1__final", "L-BFGS-B"),
        ("ic-params:ox-0.1__final", "Newton-CG"),
        ("ic-params:ox-0.1__trajectory-4", "L-BFGS-B"),
        ("ic-params:ox-0.1__trajectory-4", "Newton-CG"),
    ]
    assert all(row["nmethods"] == 2 for row in rows)
    assert [row["optimization"] for row in rows[:2]] == ["doublevortex-params", "doublevortex-params-newton-cg"]


def test_rows_record_the_required_metrics(paired_roots):
    rows = collect_comparison_rows([str(paired_roots[1])])
    row = next(row for row in rows if row["job"] == "ox-0.1__final")
    # step 15's minimum: method, optimization steps, function, Jacobian and HVP calls, final loss, status
    assert row["method"] == "Newton-CG"
    assert (row["nit"], row["nfev"], row["njev"], row["nhvp"]) == (5, 6, 6, 15)
    assert row["final_loss"] == 1.0e-9
    assert row["success"] is True and row["status"] == 0 and row["message"] == "done"
    assert row["hessp_used"] is True
    assert row["loss_reduction"] == pytest.approx(0.5e-9)
    assert row["optimize_seconds"] == 25.0
    assert row["initial-conditions.ox__rel_error"] == 1.0e-6
    assert list(row)[: len(COMPARISON_COLUMNS)] == list(COMPARISON_COLUMNS)


def test_final_loss_is_compared_with_the_best_of_each_problem(paired_roots):
    rows = {(row["job"], row["method"]): row for row in collect_comparison_rows([str(root) for root in paired_roots])}
    assert rows[("ox-0.1__final", "Newton-CG")]["final_loss_vs_best"] == 1.0
    assert rows[("ox-0.1__final", "L-BFGS-B")]["final_loss_vs_best"] == pytest.approx(2.0)
    assert rows[("ox-0.1__trajectory-4", "L-BFGS-B")]["final_loss_vs_best"] == 1.0
    assert rows[("ox-0.1__trajectory-4", "Newton-CG")]["final_loss_vs_best"] == pytest.approx(2.0)


def test_job_directories_and_roots_can_be_mixed_without_duplicates(paired_roots):
    first, second = paired_roots
    rows = collect_comparison_rows([str(first), str(first / "ox-0.1__final"), str(second / "ox-0.1__final")])
    assert len(rows) == 3
    assert sum(row["problem"] == "ic-params:ox-0.1__final" for row in rows) == 2


def test_jobs_of_different_targets_are_not_compared(tmp_path):
    write_job(tmp_path / "a", "l-6__final", "L-BFGS-B", (5, 5, 0), final_loss=1.0, target="coeffs")
    write_job(tmp_path / "b", "l-6__final", "trust-constr", (5, 5, 9), final_loss=2.0, target="ic-params")
    rows = collect_comparison_rows([str(tmp_path / "a"), str(tmp_path / "b")])
    assert len({row["problem"] for row in rows}) == 2
    assert all(row["nmethods"] == 1 and row["final_loss_vs_best"] == 1.0 for row in rows)


def test_results_written_before_hvp_support_get_defaults(tmp_path):
    result = {
        "job_name": "l-6__final", "target": "coeffs", "loss_label": "final", "truth_runs": ["l-6"],
        "optimizer": {"success": True, "status": 0, "message": "CONVERGENCE", "nit": 9, "nfev": 11, "njev": 11},
        "initial_loss": 1.0, "final_loss": 1.0e-6,
    }
    row = comparison_row(tmp_path / "old-root" / "l-6__final", result)
    assert (row["method"], row["nhvp"], row["hessp_used"]) == ("", 0, False)
    assert math.isnan(row["optimize_seconds"])
    assert row["optimization"] == "old-root"


def test_nonfinite_and_zero_losses_do_not_break_the_ratios(tmp_path):
    write_job(tmp_path / "a", "j__final", "L-BFGS-B", (5, 5, 0), final_loss=float("nan"), initial_loss=0.0)
    write_job(tmp_path / "b", "j__final", "Newton-CG", (5, 5, 5), final_loss=0.0)
    rows = {row["method"]: row for row in collect_comparison_rows([str(tmp_path / "a"), str(tmp_path / "b")])}
    assert math.isnan(rows["L-BFGS-B"]["loss_reduction"])
    # the best loss is zero, so no ratio to it is meaningful
    assert math.isnan(rows["L-BFGS-B"]["final_loss_vs_best"]) and math.isnan(rows["Newton-CG"]["final_loss_vs_best"])


def test_write_comparison_saves_csv_and_yaml(paired_roots, tmp_path):
    rows = collect_comparison_rows([str(root) for root in paired_roots])
    csv_path, yaml_path = write_comparison(rows, tmp_path / "comparison")
    assert (csv_path.name, yaml_path.name) == (COMPARISON_CSV, COMPARISON_YAML)

    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(COMPARISON_COLUMNS) + ["initial-conditions.ox__rel_error"]
    loaded = read_csv(csv_path)
    assert [row["method"] for row in loaded] == [row["method"] for row in rows]
    assert [row["nhvp"] for row in loaded] == [0.0, 15.0, 0.0, 18.0]

    summary = load_yaml(yaml_path)
    final = summary["problems"]["ic-params:ox-0.1__final"]
    assert [job["method"] for job in final["jobs"]] == ["L-BFGS-B", "Newton-CG"]
    assert final["best"] == {
        "lowest_final_loss": ["doublevortex-params-newton-cg/Newton-CG"],
        "fewest_gradients": ["doublevortex-params-newton-cg/Newton-CG"],
        "shortest_time": ["doublevortex-params-newton-cg/Newton-CG"],
    }
    trajectory = summary["problems"]["ic-params:ox-0.1__trajectory-4"]
    assert trajectory["best"]["lowest_final_loss"] == ["doublevortex-params/L-BFGS-B"]
    assert trajectory["best"]["shortest_time"] == ["doublevortex-params/L-BFGS-B"]


def test_write_comparison_handles_jobs_with_different_parameters(tmp_path):
    write_job(tmp_path / "a", "j__final", "L-BFGS-B", (5, 5, 0), final_loss=1.0)
    rows = collect_comparison_rows([str(tmp_path / "a")])
    rows.append({**rows[0], "job": "k__final", "problem": "ic-params:k__final", "threewayphysics.L__rel_error": 0.1})
    del rows[1]["initial-conditions.ox__rel_error"]
    csv_path, _ = write_comparison(rows, tmp_path)
    loaded = read_csv(csv_path)
    assert loaded[0]["threewayphysics.L__rel_error"] == "" and loaded[1]["initial-conditions.ox__rel_error"] == ""


def test_format_comparison_table_aligns_columns(paired_roots):
    rows = collect_comparison_rows([str(root) for root in paired_roots])
    lines = format_comparison_table(rows).splitlines()
    assert lines[0].split() == list(TABLE_COLUMNS)
    assert set(lines[1]) <= {"-", " "}
    assert len(lines) == 2 + len(rows)
    # every line has the same column starts as the header
    starts = [lines[0].index(column) for column in TABLE_COLUMNS]
    for line in lines[2:]:
        assert all(line[start - 1] == " " for start in starts[1:])
    assert "Newton-CG" in lines[3] and "1.000e-09" in lines[3]


def test_missing_results_raise_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No optimization results"):
        collect_comparison_rows([str(tmp_path)])


def test_run_job_records_optimizer_wall_time():
    """run_job needs Firedrake, so check by inspection that result.yaml gets optimize_seconds."""
    from workflows.common import WORKFLOWS_DIR

    source = (WORKFLOWS_DIR / "optimize_params.py").read_text(encoding="utf-8")
    run_job_source = source[source.index("def run_job("):source.index("def main(")]
    assert "start_time = time.perf_counter()" in run_job_source
    assert '"optimize_seconds": optimize_seconds' in run_job_source


# Step 16: the command-line entry point and its shell wrapper.


def test_main_prints_the_table_and_saves_next_to_the_roots(paired_roots, tmp_path, capsys):
    from workflows.compare_optimizers import main

    csv_path, yaml_path = main([str(root) for root in paired_roots])
    # the default output directory is the common parent of the two sibling roots
    assert csv_path == tmp_path.resolve() / COMPARISON_CSV
    assert yaml_path.exists() and len(read_csv(csv_path)) == 4
    out = capsys.readouterr().out
    assert "Optimizer comparison: 4 job(s), 2 problem(s)" in out
    assert out.count("Newton-CG") == 2 and out.count("L-BFGS-B") == 2
    assert str(csv_path) in out


def test_main_with_a_single_root_saves_into_it(paired_roots):
    from workflows.compare_optimizers import main

    csv_path, _ = main([str(paired_roots[0])])
    assert csv_path.parent == paired_roots[0].resolve()


def test_main_options(paired_roots, tmp_path, capsys):
    from workflows.compare_optimizers import main

    roots = [str(root) for root in paired_roots]
    # --no-save writes nothing
    assert main(roots + ["--no-save"]) is None
    assert not (tmp_path / COMPARISON_CSV).exists()
    capsys.readouterr()

    # --only filters jobs by '<optimization>/<job>', --output-dir moves the files, --columns picks the table
    out_dir = tmp_path / "elsewhere"
    csv_path, _ = main(roots + ["--only", "__final", "--output-dir", str(out_dir), "--columns", "problem", "method", "nhvp"])
    assert csv_path == out_dir.resolve() / COMPARISON_CSV
    assert {row["job"] for row in read_csv(csv_path)} == {"ox-0.1__final"}
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["problem", "method", "nhvp"]

    # parameter-error columns can be printed too
    main(roots + ["--no-save", "--columns", "method", "initial-conditions.ox__rel_error"])


@pytest.mark.parametrize("args, message", [
    (["--columns", "not-a-column"], "Unknown --columns"),
    (["--only", "no-such-job"], "No jobs match"),
])
def test_main_rejects_bad_options(paired_roots, args, message):
    from workflows.compare_optimizers import main

    with pytest.raises(ValueError, match=message):
        main([str(paired_roots[0]), "--no-save", *args])


def test_compare_optimizers_wrapper_runs_the_module_without_mpi():
    from workflows.common import WORKFLOWS_DIR

    wrapper = (WORKFLOWS_DIR / "compare_optimizers.sh").read_text(encoding="utf-8")
    assert 'python -m workflows.compare_optimizers "$@"' in wrapper
    assert "mpiexec" not in wrapper
    assert "source ~/venv-firedrake/bin/activate" in wrapper


def test_verification_script_runs_existing_tests_and_examples():
    """workflows/verify_hessp.sh (step 18) must only reference test files and examples that exist."""
    import re

    from workflows.common import PROJECT_ROOT, WORKFLOWS_DIR

    script = (WORKFLOWS_DIR / "verify_hessp.sh").read_text(encoding="utf-8")
    test_files = set(re.findall(r"tests/(?:unit|integration)/test_\w+\.py", script))
    assert "tests/integration/test_optimizer_end_to_end.py" in test_files
    for path in sorted(test_files):
        assert (PROJECT_ROOT / path).exists(), path
    for name in ("optimize_ic_params.cfg", "optimize_ic_params_newton_cg.cfg", "truth_ic_sweep.cfg"):
        assert (WORKFLOWS_DIR / "examples" / name).exists() and name in script
    # every HVP test file is run by the script
    hvp_tests = {f"tests/unit/{path.name}" for path in (PROJECT_ROOT / "tests" / "unit").glob("test_*hvp*.py")}
    assert hvp_tests <= test_files


# Step 19: regression baselines.


def test_baseline_round_trip_passes(paired_roots, tmp_path, capsys):
    from workflows.compare_optimizers import main

    roots = [str(root) for root in paired_roots]
    baseline_path = tmp_path / "baselines" / "pair.yaml"
    main(roots + ["--no-save", "--save-baseline", str(baseline_path)])
    baseline = load_yaml(baseline_path)
    assert set(baseline["jobs"]) == {
        "doublevortex-params/ox-0.1__final", "doublevortex-params/ox-0.1__trajectory-4",
        "doublevortex-params-newton-cg/ox-0.1__final", "doublevortex-params-newton-cg/ox-0.1__trajectory-4",
    }
    entry = baseline["jobs"]["doublevortex-params-newton-cg/ox-0.1__final"]
    assert (entry["method"], entry["nit"], entry["nfev"], entry["njev"], entry["nhvp"]) == ("Newton-CG", 5, 6, 6, 15)
    assert entry["final_loss"] == 1.0e-9 and entry["optimize_seconds"] == 25.0
    capsys.readouterr()

    # the same results pass the check
    main(roots + ["--no-save", "--check-baseline", str(baseline_path)])
    out = capsys.readouterr().out
    assert "passed: 4 baseline job(s)" in out and "REGRESSION" not in out


def check(rows, baseline, **kwargs):
    """Run check_against_baseline and return (regressions, notes)."""
    from workflows.compare_optimizers import check_against_baseline

    return check_against_baseline(rows, baseline, **kwargs)


def test_baseline_check_flags_regressions_and_notes_improvements(paired_roots, tmp_path):
    from workflows.compare_optimizers import baseline_from_rows

    first, second = paired_roots
    baseline = baseline_from_rows(collect_comparison_rows([str(first), str(second)]))

    # Newton-CG on the final loss: more HVPs, lost convergence, a much worse loss
    write_job(second, "ox-0.1__final", "Newton-CG", (6, 6, 40), final_loss=1.0e-6, nit=5, seconds=25.0, success=False)
    # L-BFGS-B on the final loss: fewer evaluations and a lower loss
    write_job(first, "ox-0.1__final", "L-BFGS-B", (10, 10, 0), final_loss=1.0e-9, nit=9, seconds=60.0)
    regressions, notes = check(collect_comparison_rows([str(first), str(second)]), baseline)
    text = "\n".join(regressions)
    assert len(regressions) == 3, regressions
    assert "converged in the baseline but not now" in text
    assert "nhvp 40 exceeds the baseline 15" in text
    assert "final_loss 1.000e-06 is worse than 10 x" in text
    assert all("doublevortex-params-newton-cg/ox-0.1__final" in line for line in regressions)
    assert any("nfev 10 is below the baseline 14" in note for note in notes)
    assert any("final_loss 1.000e-09 is below the baseline 2.000e-09" in note for note in notes)
    # wall time is reported, but only checked with time_factor
    assert any("optimize_seconds 2.00 x the baseline" in note for note in notes)
    regressions, _ = check(collect_comparison_rows([str(first), str(second)]), baseline, time_factor=1.5)
    assert any("optimize_seconds 60.0 is 2.00 x the baseline" in line for line in regressions)


def test_baseline_count_tolerance(paired_roots):
    from workflows.compare_optimizers import baseline_from_rows

    first, second = paired_roots
    baseline = baseline_from_rows(collect_comparison_rows([str(second)]))
    # 15 HVPs in the baseline: up to 15 * 1.25 + 1 = 19.75 are allowed by default
    write_job(second, "ox-0.1__final", "Newton-CG", (6, 6, 19), final_loss=1.0e-9, nit=5, seconds=25.0)
    assert check(collect_comparison_rows([str(second)]), baseline)[0] == []
    write_job(second, "ox-0.1__final", "Newton-CG", (6, 6, 20), final_loss=1.0e-9, nit=5, seconds=25.0)
    assert len(check(collect_comparison_rows([str(second)]), baseline)[0]) == 1
    assert check(collect_comparison_rows([str(second)]), baseline, count_rtol=0.5)[0] == []


def test_baseline_check_flags_missing_and_changed_jobs(paired_roots):
    from workflows.compare_optimizers import baseline_from_rows

    first, second = paired_roots
    baseline = baseline_from_rows(collect_comparison_rows([str(first), str(second)]))
    write_job(second, "ox-0.1__final", "trust-ncg", (6, 6, 15), final_loss=1.0e-9)
    write_job(second, "ox-0.2__final", "Newton-CG", (6, 6, 15), final_loss=1.0e-9)
    regressions, notes = check(collect_comparison_rows([str(second)]), baseline)
    text = "\n".join(regressions)
    assert "doublevortex-params/ox-0.1__final: missing" in text
    assert "doublevortex-params/ox-0.1__trajectory-4: missing" in text
    assert "method changed from the baseline" in text
    assert any("ox-0.2__final: not in the baseline" in note for note in notes)


def test_failed_baseline_check_exits_with_status_1(paired_roots, tmp_path, capsys):
    from workflows.compare_optimizers import main

    first, second = paired_roots
    baseline_path = tmp_path / "pair.yaml"
    main([str(first), str(second), "--no-save", "--save-baseline", str(baseline_path)])
    write_job(second, "ox-0.1__final", "Newton-CG", (6, 6, 40), final_loss=1.0e-9, nit=5)
    with pytest.raises(SystemExit) as exit_info:
        main([str(first), str(second), "--no-save", "--check-baseline", str(baseline_path)])
    assert exit_info.value.code == 1
    assert "REGRESSION: doublevortex-params-newton-cg/ox-0.1__final: nhvp 40" in capsys.readouterr().out


def test_baseline_script_runs_both_methods_and_checks_the_baseline():
    from workflows.common import WORKFLOWS_DIR

    script = (WORKFLOWS_DIR / "optimizer_baseline.sh").read_text(encoding="utf-8")
    for text in ("optimize_ic_params.cfg", "optimize_ic_params_newton_cg.cfg", "--save-baseline", "--check-baseline",
                 "workflows/baselines/doublevortex-params.yaml", "--skip-existing"):
        assert text in script, text
