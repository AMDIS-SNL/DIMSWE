"""Tests for the HVP profiling driver workflows/profile_hvp.py (HVP performance plan, section 2).

Most tests use a fake problem that opens the same profiling regions, in the same nesting, as
Lagrangian_ODEConstrainedOptimization with a two-integrator LieSplit timestepper that reuses
the recomputed stages in its substep adjoint steps (plan item A3). A clock that
only advances when the fake does "work" makes every time exact, so the summary arithmetic
(setup costs, ratios, stage passes, per-region breakdown, cost per stage pass, optimizer
estimate) is checked exactly. The CLI is run on the fake problem as well. The last test runs the
real gradient and HVP of tests/tswe_small.cfg (Firedrake) and checks the stage-pass counts of the cost
model in docs/hvp_performance_plan.md against the instrumented code.
"""

from pathlib import Path

import numpy as np
import pytest

from workflows import profile_hvp
from workflows.common import load_yaml
from workflows.profile_hvp import (
    COST_MODEL_STAGE_PASSES,
    COST_MODEL_RATIOS_RECOMPUTE,
    evaluation_plan,
    main,
    optimizer_estimate,
    parse_args,
    profile_evaluations,
    profiling_module,
    summarize,
    timestepper_layout,
)

profiling = profiling_module()
region = profiling.region

NSTEPS = 2
LABELS = ("0:RK4(model)", "1:Euler(hyperviscosity)")
NSUBSTEPS = len(LABELS)
# fake cost of one pass of each kind of work, in clock seconds
COST = {
    "forward stages": 1.0, "tangent stages": 2.0, "adjoint stages": 2.0, "soa stages": 4.0,
    "setup second order": 10.0, "setup hvp": 5.0,
    # restoring stored stages is a few assigns; free here, so stage costs stay exact
    "restore stages": 0.0,
}


class WorkClock:
    """A clock that only advances when work is done, so all recorded times are exact."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def work(self, name):
        """Do one unit of the named work inside its profiling region."""
        with region(name):
            self.now += COST[name]


class FakeProblem:
    """Opens the profiling regions of the real obj/jac/hessp for a two-integrator LieSplit step."""

    def __init__(self, clock):
        self.clock = clock
        self.z0 = np.zeros(2)
        self.hvp_ready = False
        self.second_order_ready = False

    def to_full(self, z):
        return np.asarray(z, dtype=float)

    def direction_to_full(self, p):
        return np.asarray(p, dtype=float)

    def substeps(self, kinds):
        """One time step: each sub-integrator does the given stage passes once."""
        for label in LABELS:
            with region(label):
                for kind in kinds:
                    self.clock.work(kind)

    def full_fun(self, x):
        with region("obj"):
            with region("forward pass"):
                for _ in range(NSTEPS):
                    self.substeps(["forward stages"])
        return 0.0

    def full_jac(self, x):
        with region("jac"):
            with region("forward pass"):
                for _ in range(NSTEPS):
                    self.substeps(["forward stages"])
            with region("adjoint sweep"):
                for _ in range(NSTEPS):
                    with region("substep recompute"):
                        self.substeps(["forward stages"])
                    with region("substep adjoints"):
                        self.substeps(["restore stages", "adjoint stages"])
        return np.zeros(2)

    def full_hessp(self, x, v):
        with region("hessp"):
            if not self.hvp_ready:
                self.clock.work("setup hvp")
                self.hvp_ready = True
            with region("forward+tangent pass"):
                for _ in range(NSTEPS):
                    for label in LABELS:
                        with region(label):
                            if not self.second_order_ready:
                                self.clock.work("setup second order")
                            self.clock.work("forward stages")
                            self.clock.work("tangent stages")
                    self.second_order_ready = True
            with region("second-order adjoint sweep"):
                for _ in range(NSTEPS):
                    with region("substep recompute"):
                        self.substeps(["forward stages", "tangent stages"])
                    with region("substep adjoints"):
                        self.substeps(["restore stages", "adjoint stages", "soa stages"])
        return np.zeros(2)


@pytest.fixture
def clock(monkeypatch):
    """Install a WorkClock as the profiling clock, with a clean timing table."""
    work_clock = WorkClock()
    monkeypatch.setattr(profiling, "_clock", work_clock)
    was_enabled = profiling.is_enabled()
    profiling.reset()
    yield work_clock
    profiling.enable(was_enabled)
    profiling.reset()


@pytest.fixture
def records(clock):
    return profile_evaluations(FakeProblem(clock), repeats=1, seed=3)


# exact fake costs, per call: obj NSTEPS * 2 substeps * forward; see FakeProblem
OBJ = NSTEPS * NSUBSTEPS * 1.0
JAC = NSTEPS * NSUBSTEPS * (1.0 + 1.0 + 2.0)
HESSP = NSTEPS * NSUBSTEPS * ((1.0 + 2.0) * 2 + 2.0 + 4.0)
HESSP_SETUP = COST["setup hvp"] + NSUBSTEPS * COST["setup second order"]


def test_evaluation_plan_has_first_and_steady_calls_and_a_second_direction():
    assert evaluation_plan(1) == [
        ("obj-1", "obj", 0), ("obj-2", "obj", 0), ("jac-1", "jac", 0), ("jac-2", "jac", 0),
        ("hessp-1", "hessp", 1), ("hessp-2", "hessp", 1), ("hessp-3", "hessp", 2),
    ]
    assert [label for label, _, _ in evaluation_plan(2)][-4:] == ["hessp-1", "hessp-2", "hessp-3", "hessp-4"]


def test_profile_evaluations_records_one_table_per_call(records):
    assert [record["label"] for record in records] == [label for label, _, _ in evaluation_plan(1)]
    for record in records:
        # every table is rooted at the call's own label, so the reset between calls worked
        assert all(path.split("/")[0] == record["label"] for path in record["regions"])
    seconds = {record["label"]: record["seconds"] for record in records}
    assert seconds["obj-1"] == seconds["obj-2"] == OBJ
    assert seconds["jac-2"] == JAC
    assert seconds["hessp-1"] == HESSP + HESSP_SETUP
    assert seconds["hessp-2"] == seconds["hessp-3"] == HESSP
    assert any(path.endswith("setup second order") for path in records[4]["regions"])
    assert not any(path.endswith("setup second order") for path in records[5]["regions"])


def test_profile_evaluations_leaves_timing_disabled(clock):
    profile_evaluations(FakeProblem(clock), repeats=1)
    assert not profiling.is_enabled()


def test_summary_separates_setup_from_steady_state_and_compares_with_the_cost_model(records):
    summary = summarize(records, NSTEPS, NSUBSTEPS)
    assert summary["seconds"]["obj"]["setup"] == 0.0
    assert summary["seconds"]["hessp"]["first"] == HESSP + HESSP_SETUP
    assert summary["seconds"]["hessp"]["steady"] == HESSP
    assert summary["seconds"]["hessp"]["setup"] == HESSP_SETUP
    assert summary["hessp_setup_construction_seconds"] == HESSP_SETUP
    assert summary["ratios"]["jac/obj"] == pytest.approx(JAC / OBJ)
    assert summary["ratios"]["hessp/jac"] == pytest.approx(HESSP / JAC)
    assert summary["cost_model_ratios"]["hessp/jac"] == 2.0
    # the fake has the call structure of the real LieSplit code, so it matches the model exactly
    for kind, passes in summary["stage_passes_per_substep"].items():
        for stage_kind, entry in passes.items():
            assert entry["measured"] == entry["model"] == COST_MODEL_STAGE_PASSES[kind].get(stage_kind, 0), (kind, stage_kind)


def test_breakdown_accounts_for_every_second(records):
    summary = summarize(records, NSTEPS, NSUBSTEPS)
    hessp = summary["breakdown"]["hessp"]
    assert hessp["soa stages"]["seconds"] == NSTEPS * NSUBSTEPS * 4.0
    assert hessp["tangent stages"]["calls"] == 2 * NSTEPS * NSUBSTEPS
    assert hessp["restore stages"]["calls"] == NSTEPS * NSUBSTEPS
    # the fake does no work outside the stage regions
    assert hessp["other"]["seconds"] == pytest.approx(0.0)
    assert sum(entry["share"] for entry in hessp.values()) == pytest.approx(1.0)


def test_stage_costs_are_per_sub_integrator_and_relative_to_a_forward_pass(records):
    costs = summarize(records, NSTEPS, NSUBSTEPS)["stage_costs"]["hessp"]
    assert set(costs) == set(LABELS)
    for label in LABELS:
        assert costs[label]["forward stages"]["seconds_per_pass"] == 1.0
        assert costs[label]["tangent stages"]["vs_forward"] == 2.0
        assert costs[label]["soa stages"]["vs_forward"] == 4.0
        assert costs[label]["soa stages"]["calls"] == NSTEPS


def test_optimizer_estimate_combines_counts_with_steady_costs(records):
    summary = summarize(records, NSTEPS, NSUBSTEPS)
    result = {"job_name": "ox__final", "optimize_seconds": 600.0,
              "optimizer": {"method": "Newton-CG", "nit": 5, "nfev": 5, "njev": 5, "nhvp": 7}}
    estimate = optimizer_estimate(result, summary["seconds"])
    # the first obj and jac happen before run_job starts the optimizer clock
    expected = 4 * OBJ + 4 * JAC + 7 * HESSP + HESSP_SETUP
    assert estimate["estimate_seconds"] == pytest.approx(expected)
    assert estimate["hvp_share"] == pytest.approx((7 * HESSP + HESSP_SETUP) / expected)
    assert estimate["estimate_vs_recorded"] == pytest.approx(expected / 600.0)
    assert estimate["hvps_per_iteration"] == pytest.approx(7 / 5)
    first_order = optimizer_estimate({"optimizer": {"method": "L-BFGS-B", "nit": 6, "nfev": 8, "njev": 8, "nhvp": 0}},
                                     summary["seconds"])
    assert first_order["estimate_parts"]["hessp setup"] == 0.0 and first_order["hvp_share"] == 0.0
    assert first_order["estimate_vs_recorded"] is None


def test_recompute_cost_model_has_one_more_stage_pass(records):
    summary = summarize(records, NSTEPS, NSUBSTEPS, recompute_stages=True)
    passes = summary["stage_passes_per_substep"]
    assert passes["jac"]["forward stages"]["model"] == 3.0
    assert passes["hessp"]["tangent stages"]["model"] == 3.0
    assert summary["cost_model_ratios"]["jac/obj"] == 4.0
    assert summarize(records, NSTEPS, NSUBSTEPS)["cost_model_ratios"]["jac/obj"] == 3.0


def test_timestepper_layout_counts_lie_split_substeps():
    class SplitReusing:
        subcycle_list = [2, 1, 2, 1]
        reuse_stages = True

    class SplitRecomputing:
        subcycle_list = [2, 1]
        reuse_stages = False

    class Single:
        pass

    assert timestepper_layout(SplitReusing()) == (6, False)
    assert timestepper_layout(SplitRecomputing()) == (3, True)
    assert timestepper_layout(Single()) == (1, False)


def test_parse_args_passes_petsc_options_through():
    args = parse_args(["spec.cfg", "--job", "__final", "-log_view", ":flame.txt:ascii_flamegraph", "-ksp_view"])
    assert args.optimize_cfg == "spec.cfg" and args.job == "__final"
    assert args.petsc_options == ["-log_view", ":flame.txt:ascii_flamegraph", "-ksp_view"]
    with pytest.raises(SystemExit):
        parse_args(["spec.cfg", "--repeats", "0"])
    with pytest.raises(SystemExit):
        parse_args(["spec.cfg", "other.cfg"])


@pytest.fixture
def fake_job(monkeypatch, clock, tmp_path):
    """Replace job selection and model building in profile_hvp with the fake problem."""
    spec = {"optimization_name": "fake-params-newton-cg", "target": "ic-params", "optimizer": {"method": "Newton-CG"}}
    job = {"name": "fake__final", "job_dir": tmp_path / "optimizations" / "fake__final"}
    monkeypatch.setattr(profile_hvp, "select_job", lambda spec_path, job_filter: (spec, job))
    monkeypatch.setattr(profile_hvp, "build_job_problem",
                        lambda spec, job, recompute_stages=False: (FakeProblem(clock), NSTEPS, NSUBSTEPS,
                                                                   recompute_stages))
    return spec, job


def test_cli_writes_profile_yaml_and_tree(fake_job, tmp_path, capsys):
    _, job = fake_job
    job["job_dir"].mkdir(parents=True)
    from workflows.common import dump_yaml

    dump_yaml({"job_name": "fake__final", "optimize_seconds": 100.0,
               "optimizer": {"method": "Newton-CG", "nit": 5, "nfev": 5, "njev": 5, "nhvp": 7}},
              job["job_dir"] / "result.yaml")
    output_dir = tmp_path / "profile"
    profile = main(["spec.cfg", "--output-dir", str(output_dir), "-log_view", ":x.txt:ascii_flamegraph"])
    out = capsys.readouterr().out
    assert "Seconds per evaluation" in out and "Stage passes per time step" in out
    assert "== hessp-1" in out and "Optimizer run fake__final" in out

    written = load_yaml(output_dir / "profile.yaml")
    assert written["job_name"] == "fake__final" and written["method"] == "Newton-CG"
    assert written["petsc_options"] == ["-log_view", ":x.txt:ascii_flamegraph"]
    assert written["nsteps"] == NSTEPS and written["nsubsteps"] == NSUBSTEPS and written["recompute_stages"] is False
    assert [record["label"] for record in written["evaluations"]] == [label for label, _, _ in evaluation_plan(1)]
    assert written["summary"]["seconds"]["hessp"]["steady"] == HESSP
    # the job's own result.yaml is used by default
    assert written["summary"]["optimizer"][0]["nhvp"] == 7
    assert profile["summary"]["ratios"] == written["summary"]["ratios"]
    tree = (output_dir / "profile-tree.txt").read_text(encoding="utf-8")
    assert "second-order adjoint sweep" in tree and "soa stages" in tree


def test_cli_default_output_dir_and_dry_run(fake_job, capsys):
    assert main(["spec.cfg", "--print-output-dir"]) is None
    out = capsys.readouterr().out.strip().splitlines()
    assert Path(out[-1]).parts[-3:] == ("profiles", "fake-params-newton-cg", "fake__final")

    # the recompute A/B run does not overwrite the default profile
    assert main(["spec.cfg", "--print-output-dir", "--recompute-stages"]) is None
    out = capsys.readouterr().out.strip().splitlines()
    assert Path(out[-1]).parts[-1] == "fake__final__recompute"

    assert main(["spec.cfg", "--dry-run", "--repeats", "2"]) is None
    out = capsys.readouterr().out
    assert "hessp-4" in out and "Seconds per evaluation" not in out


def test_cli_recompute_stages_selects_the_recompute_cost_model(fake_job, tmp_path, capsys):
    profile = main(["spec.cfg", "--output-dir", str(tmp_path / "recompute"), "--recompute-stages"])
    capsys.readouterr()
    assert profile["recompute_stages"] is True
    assert profile["summary"]["cost_model_ratios"] == COST_MODEL_RATIOS_RECOMPUTE


def test_profile_script_wraps_the_driver_and_the_flamegraph():
    script = (Path(__file__).resolve().parents[2] / "workflows" / "profile_hvp.sh").read_text(encoding="utf-8")
    assert "python -m workflows.profile_hvp" in script
    assert "ascii_flamegraph" in script and "flamegraph.pl" in script
    assert "--print-output-dir" in script


def test_real_gradient_and_hvp_match_the_stage_pass_cost_model():
    """Run obj, jac and hessp on tests/tswe_small.cfg and count the instrumented stage passes."""
    pytest.importorskip("firedrake")
    from dimswe.optimize import ParametricInitialCondition
    from tests.gradient_helpers import setup_problem

    nsteps = 3
    setup = setup_problem(nsteps, obs_steps=[1, 3])
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ["ox", "oy"])
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
    problem = setup.optimizer.reduced_problem(params, opt_type="ic-params", coeffs0=setup.coeff, ic_map=ic_map,
                                              scaling_factors=np.array([0.1, 0.1]))
    nsubsteps, recompute_stages = timestepper_layout(setup.timestepper)
    # tests/tswe_small.cfg: LieSplit of RK4 x 2 and Euler x 1, reusing stages by default
    assert not recompute_stages and nsubsteps == 3

    records = profile_evaluations(problem, repeats=1, seed=5)
    summary = summarize(records, nsteps, nsubsteps, recompute_stages=recompute_stages)
    print(profile_hvp.format_summary(summary))
    for kind, passes in summary["stage_passes_per_substep"].items():
        for stage_kind, entry in passes.items():
            assert entry["measured"] == entry["model"], (kind, stage_kind, entry)
    first, steady = records[4], records[5]
    assert any(path.endswith("setup second order") for path in first["regions"])
    assert not any(path.endswith("setup second order") for path in steady["regions"])
    assert set(summary["stage_costs"]["hessp"]) == set(setup.timestepper.profile_labels)
    assert all(record["seconds"] > 0 for record in records)
    assert summary["breakdown"]["hessp"]["other"]["share"] < 1.0
