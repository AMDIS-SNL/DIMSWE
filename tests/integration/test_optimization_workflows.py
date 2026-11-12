import pytest
import yaml

from workflows.common import (
    PROJECT_ROOT,
    WORKFLOWS_DIR,
    build_observation_schedule,
    check_paths_exist,
    classify_parameter,
    expand_parameter_specs,
    expand_value_spec,
    loss_label,
    summarize_series,
    validate_target_and_losses,
)
from workflows.generate_cfgs import classify_swept_parameters, validate_sweep_paths
from workflows.optimize_params import normalize_parameter_specs

WORKFLOWS = WORKFLOWS_DIR


def test_expand_value_spec_linspace_and_geomspace():
    assert expand_value_spec({"linspace": [0.0008, 0.0012, 3]}) == [0.0008, 0.001, 0.0012]
    assert expand_value_spec({"geomspace": [1.0, 100.0, 3]}) == [1.0, 10.0, 100.0]
    assert expand_value_spec([1, 2]) == [1, 2]
    assert expand_value_spec(5) == [5]


def test_expand_value_spec_random_is_seeded_and_in_range():
    spec = {"random": {"low": 0.08, "high": 0.12, "n": 4, "seed": 7}}
    first = expand_value_spec(spec)
    assert first == expand_value_spec(spec)
    assert len(first) == 4
    assert all(0.08 <= value <= 0.12 for value in first)


@pytest.mark.parametrize(
    "spec",
    [
        {"linspace": [0.0, 1.0]},
        {"geomspace": [0.0, 1.0, 3]},
        {"random": {"low": 0.0, "high": 1.0, "n": 2}},
        {"linspace": [0, 1, 2], "values": [1]},
    ],
)
def test_expand_value_spec_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        expand_value_spec(spec)


def test_expand_parameter_specs_expands_every_entry():
    expanded = expand_parameter_specs({"a.b": {"linspace": [0, 1, 2]}, "a.c": [3]})
    assert expanded == {"a.b": [0.0, 1.0], "a.c": [3]}


def test_classify_parameter_tags_ic_and_coefficients():
    parameters = {
        "initial-conditions": {"ox": 0.1},
        "hyperviscosity": {"c0": 0.07, "treat_as_coeffs": False},
        "mesh": {"nx": 10},
    }
    assert classify_parameter("initial-conditions.ox", parameters) == "ic"
    assert classify_parameter("hyperviscosity.c0", parameters) == "coeff"
    assert classify_parameter("hyperviscosity.treat_as_coeffs", parameters) == "other"
    assert classify_parameter("mesh.nx", parameters) == "other"


def test_strict_paths_reject_typos():
    template = {"hyperviscosity": {"c0": 0.07, "treat_as_coeffs": False}}
    with pytest.raises(KeyError):
        check_paths_exist(template, ["hyperviscosity.cO"], "test")
    sweep = {"sweep_name": "s", "parameters": {"hyperviscosity.cO": [1.0]}}
    with pytest.raises(KeyError):
        validate_sweep_paths(sweep, template)
    validate_sweep_paths({**sweep, "strict_paths": False}, template)


def test_classify_swept_parameters_includes_tied_sets():
    template = {"initial-conditions": {"ox": 0.1, "oy": 0.1}, "hyperviscosity": {"c0": 1.0, "treat_as_coeffs": False}}
    sweep = {
        "parameters": {"hyperviscosity.c0": [1.0]},
        "tied_parameter_sets": [{"initial-conditions.ox": 0.1, "initial-conditions.oy": 0.1}],
    }
    assert classify_swept_parameters(sweep, template) == {
        "hyperviscosity.c0": "coeff",
        "initial-conditions.ox": "ic",
        "initial-conditions.oy": "ic",
    }


def test_final_schedule_compares_once_at_window_end():
    schedule = build_observation_schedule({"type": "final"}, 0, 20, 2, 31)
    assert schedule["block_start_indices"] == [0]
    assert schedule["block_steps"] == 20
    assert schedule["obs_steps"] == [20]
    assert schedule["block_obs_indices"] == [[10]]


def test_windowed_schedule_restarts_every_window():
    schedule = build_observation_schedule({"type": "windowed", "window_steps": 4}, 0, 20, 2, 31)
    assert schedule["block_start_indices"] == [0, 2, 4, 6, 8]
    assert schedule["block_steps"] == 4
    assert schedule["obs_steps"] == [4]
    assert schedule["block_obs_indices"] == [[2], [4], [6], [8], [10]]


def test_trajectory_schedule_compares_every_obs_every_steps():
    schedule = build_observation_schedule({"type": "trajectory", "obs_every": 4}, 3, 16, 2, 31)
    assert schedule["block_start_indices"] == [3]
    assert schedule["obs_steps"] == [4, 8, 12, 16]
    assert schedule["block_obs_indices"] == [[5, 7, 9, 11]]


@pytest.mark.parametrize(
    "loss, start, num_steps",
    [
        ({"type": "final"}, 0, 21),
        ({"type": "final"}, 0, 80),
        ({"type": "windowed", "window_steps": 3}, 0, 12),
        ({"type": "windowed", "window_steps": 8}, 0, 12),
        ({"type": "trajectory", "obs_every": 3}, 0, 12),
        ({"type": "unknown"}, 0, 4),
    ],
)
def test_schedule_rejects_inconsistent_windows(loss, start, num_steps):
    with pytest.raises(ValueError):
        build_observation_schedule(loss, start, num_steps, 2, 31)


def test_windowed_loss_is_rejected_for_initial_condition_targets():
    with pytest.raises(ValueError):
        validate_target_and_losses("ic-field", [{"type": "windowed", "window_steps": 2}], "per-run")
    with pytest.raises(ValueError):
        validate_target_and_losses("ic-params", [{"type": "windowed", "window_steps": 2}], "per-run")
    validate_target_and_losses("coeffs", [{"type": "windowed", "window_steps": 2}], "per-run")


def test_joint_mode_is_only_for_coefficients():
    with pytest.raises(ValueError):
        validate_target_and_losses("ic-params", [{"type": "final"}], "joint")
    validate_target_and_losses("coeffs", [{"type": "final"}], "joint")


def test_duplicate_loss_labels_are_rejected():
    with pytest.raises(ValueError):
        validate_target_and_losses("coeffs", [{"type": "final"}, {"type": "final"}], "per-run")
    assert loss_label({"type": "windowed", "window_steps": 4}) == "windowed-4"
    assert loss_label({"type": "trajectory", "obs_every": 2}) == "trajectory-2"


def test_normalize_parameter_specs_handles_fixed_shorthand_and_bounds():
    normalized = normalize_parameter_specs(
        "coeffs",
        {"a.b": {"fixed": 3.2}, "a.c": 0.5, "a.d": {"initial": 1.0, "bounds": [None, 2.0]}},
    )
    assert normalized["a.b"] == {"initial": 3.2, "bounds": None, "scale": None, "fixed": True}
    assert normalized["a.c"]["initial"] == 0.5 and normalized["a.c"]["fixed"] is False
    assert normalized["a.d"]["bounds"] == [float("-inf"), 2.0]
    with pytest.raises(ValueError):
        normalize_parameter_specs("ic-params", {"initial-conditions.ox": {"initial": 0.1, "bounds": [0, 1]}})
    with pytest.raises(ValueError):
        normalize_parameter_specs("coeffs", {"a.b": {"initial": 1.0, "scale": 0.0}})


def test_summarize_series_splits_training_and_forecast():
    summary = summarize_series([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, float("nan"), 0.5], 1.0)
    assert summary["final"] == 0.5
    assert summary["max"] == 2.0
    assert summary["train_max"] == 2.0
    assert summary["forecast_max"] == 0.5


@pytest.mark.parametrize(
    "name",
    ["truth_coeff_sweep.cfg", "truth_ic_sweep.cfg", "optimize_coeff.cfg", "optimize_ic_params.cfg", "optimize_ic_field.cfg",
     "optimize_coeff_trust_constr.cfg", "optimize_ic_params_newton_cg.cfg"],
)
def test_example_optimization_cfgs_are_valid_yaml(name):
    data = yaml.safe_load((WORKFLOWS / "examples" / name).read_text(encoding="utf-8"))
    assert data


@pytest.mark.parametrize(
    "name",
    ["optimize_coeff.cfg", "optimize_ic_params.cfg", "optimize_ic_field.cfg",
     "optimize_coeff_trust_constr.cfg", "optimize_ic_params_newton_cg.cfg"],
)
def test_example_optimization_specs_pass_validation(name):
    from workflows.optimize_params import load_spec

    spec = load_spec(WORKFLOWS / "examples" / name)
    assert spec["target"] in ("coeffs", "ic-params", "ic-field")


@pytest.mark.parametrize("name", ["optimize_params", "validate"])
def test_optimization_wrappers_use_explicit_mpi_ranks(name):
    wrapper = (WORKFLOWS / f"{name}.sh").read_text(encoding="utf-8")
    assert 'MPI_RANKS="$1"' in wrapper
    assert f'mpiexec -n "$MPI_RANKS" python -m workflows.{name} "$@"' in wrapper


def test_ic_field_target_is_guarded_to_serial():
    source = (WORKFLOWS / "optimize_params.py").read_text(encoding="utf-8")
    assert 'spec["target"] == "ic-field" and MPI.COMM_WORLD.Get_size() > 1' in source


def test_ic_params_gradient_uses_ufl_derivative_not_finite_differences():
    source = (PROJECT_ROOT / "dimswe" / "optimize.py").read_text(encoding="utf-8")
    assert "ufl.derivative(expr, func, direction)" in source
    assert "ic_map.backward(self.jac(" in source


def test_run_batch_writes_manifest_runs_where_the_manifest_says(tmp_path):
    """run_batch.py must write truth runs to the manifest's run_directory, where optimize_params reads them."""
    import argparse

    pytest.importorskip("mpi4py")
    from workflows import run_batch

    cfg = tmp_path / "a.cfg"
    cfg.write_text(yaml.safe_dump({"output": {}}), encoding="utf-8")
    run_directory = tmp_path / "truth" / "a"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"runs_root": str(tmp_path / "truth"),
             "runs": [{"run_name": "a", "cfg_path": str(cfg), "run_directory": str(run_directory)}]}
        ),
        encoding="utf-8",
    )
    specs = run_batch.build_run_specifications(argparse.Namespace(manifest=str(manifest), cfgs=[], runs_root=None))
    assert specs[0]["run_directory"] == run_directory
    run_info, _ = run_batch.prepare_run(cfg, "a", specs[0]["run_directory"], {}, "test")
    assert run_info["parameters"]["output"]["outfile_name"] == str(run_directory / "sim")

    moved = run_batch.build_run_specifications(
        argparse.Namespace(manifest=str(manifest), cfgs=[], runs_root=str(tmp_path / "elsewhere"))
    )
    assert moved[0]["run_directory"] == tmp_path / "elsewhere" / "a"


def test_relative_gtol_scales_with_initial_gradient():
    from workflows.optimize_params import optimizer_gtol

    report = {"grad_max": 3.0e-7}
    assert optimizer_gtol({"gtol": 1.0e-6, "tolerance": "relative"}, report) == pytest.approx(3.0e-13)
    assert optimizer_gtol({"gtol": 1.0e-6, "tolerance": "absolute"}, report) == 1.0e-6
    assert optimizer_gtol({"gtol": 1.0e-6}, report) == pytest.approx(3.0e-13)
    with pytest.raises(ValueError):
        optimizer_gtol({"gtol": 1.0e-6, "tolerance": "other"}, report)


def test_relative_gtol_does_not_stop_on_a_small_but_real_gradient():
    """With an absolute gtol, L-BFGS-B stops at iteration 0 when the starting gradient is small."""
    np = pytest.importorskip("numpy")
    scipy_optimize = pytest.importorskip("scipy.optimize")
    from workflows.optimize_params import optimizer_gtol

    # loss ~1.48 that varies weakly with the parameter: initial gradient ~1e-7, optimum far away
    target, weight = 5.0, 1.0e-8
    fun = lambda z: 1.48 + weight * float(np.sum((z - target) ** 2))
    jac = lambda z: 2.0 * weight * (z - target)
    z0 = np.zeros(1)
    report = {"grad_max": float(np.max(np.abs(jac(z0))))}

    absolute = scipy_optimize.minimize(fun, z0, jac=jac, method="L-BFGS-B",
        options={"gtol": optimizer_gtol({"gtol": 1.0e-6, "tolerance": "absolute"}, report), "ftol": 0.0})
    assert absolute.nit == 0

    relative = scipy_optimize.minimize(fun, z0, jac=jac, method="L-BFGS-B",
        options={"gtol": optimizer_gtol({"gtol": 1.0e-6, "tolerance": "relative"}, report), "ftol": 0.0})
    assert relative.nit > 0
    assert relative.x[0] == pytest.approx(target, rel=1e-3)


def test_sweep_plots_skip_fields_without_positive_errors(tmp_path, capsys):
    """Fields whose errors are all zero or NaN must be skipped, not crash the log-scale chart."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from workflows.common import write_csv
    from workflows import plot_validation

    rows = []
    for run in ["l-6", "l-9"]:
        for loss in ["final", "windowed-4"]:
            rows.append({
                "job": f"{run}__{loss}", "case": "optimal", "truth_run": run, "loss": loss,
                "threewayphysics.L__true": 6.0, "threewayphysics.L__value": 6.1,
                "h__rel_l2__final": 1.0e-4, "Qr__rel_l2__final": float("nan"), "Qc__rel_l2__final": 0.0,
            })
    write_csv(tmp_path / "summary.csv", rows)
    plot_validation.apply_style()
    plot_validation.render_sweep(tmp_path)

    written = {path.name for path in (tmp_path / "plots").iterdir()}
    assert "errors-by-loss__h.png" in written
    assert "errors-by-loss__Qc.png" not in written and "errors-by-loss__Qr.png" not in written
    assert "Skipped errors-by-loss plots for ['Qc', 'Qr']" in capsys.readouterr().out


def test_default_plot_fields_include_moisture_and_skip_fixed_fields():
    from workflows.plot_validation import default_fields

    available = ["Qc", "Qc_l", "Qr", "Qv", "Qv_l", "S", "bottom_topography", "coriolis", "eta", "h", "qsat", "rh", "v"]
    groups = {"fields": {"prognostic": ["v", "h", "S", "Qv", "Qc", "Qr"], "moisture": ["Qv_l", "Qc_l", "rh", "qsat"],
                         "diagnostic": ["eta"]}}
    assert default_fields(groups, available) == ["v", "h", "S", "Qv", "Qc", "Qr", "Qv_l", "Qc_l", "rh", "qsat"]
    # validations written before field groups were recorded: everything except fixed fields
    fallback = default_fields({}, available)
    assert {"Qv", "Qc", "Qr", "qsat", "rh"} <= set(fallback)
    assert "coriolis" not in fallback and "bottom_topography" not in fallback


def optimizer_spec_with(**changes):
    """Return the default optimizer spec with changes applied, as load_spec would build it."""
    from workflows.optimize_params import DEFAULT_OPTIMIZER

    return {**DEFAULT_OPTIMIZER, **changes}


def test_default_optimizer_spec_is_unchanged_first_order():
    from workflows.optimize_params import DEFAULT_OPTIMIZER, validate_optimizer_spec

    assert DEFAULT_OPTIMIZER["method"] == "L-BFGS-B"
    assert DEFAULT_OPTIMIZER["use_hessp"] is None
    assert DEFAULT_OPTIMIZER["use_bounds"] is True
    assert DEFAULT_OPTIMIZER["xtol"] is None
    # null: stages when the timestepper supports it, else states (dimswe.optimize.resolve_checkpoint)
    assert DEFAULT_OPTIMIZER["checkpoint"] is None
    for target in ("coeffs", "ic-params", "ic-field"):
        validate_optimizer_spec(optimizer_spec_with(), target, {})


@pytest.mark.parametrize(
    "changes, target",
    [
        ({"method": "trust-constr"}, "coeffs"),
        ({"method": "trust-constr", "use_hessp": True, "xtol": 1.0e-10}, "coeffs"),
        ({"method": "Newton-CG", "use_bounds": False}, "coeffs"),
        ({"method": "Newton-CG", "xtol": 1.0e-8}, "ic-params"),
        ({"method": "trust-ncg"}, "ic-field"),
        ({"method": "trust-krylov", "use_hessp": True}, "ic-params"),
        ({"method": "newton-cg", "use_hessp": False}, "ic-field"),  # finite differences of the gradient
        ({"method": "CG"}, "coeffs"),  # first-order methods keep scipy's warn-and-ignore bounds behaviour
        ({"method": "TNC"}, "coeffs"),
        ({"method": "Newton-CG", "checkpoint": "stages"}, "ic-params"),  # stage checkpoint (A1/A2)
        ({"checkpoint": "stages"}, "coeffs"),  # also speeds up first-order gradients
        ({"checkpoint": "states"}, "ic-params"),  # the recompute path, for memory-bound windows
        ({"mass_inverse": "factor"}, "ic-params"),  # LU-factored mass matrices, as before
        ({"mass_inverse": "cellwise"}, "coeffs"),
    ],
)
def test_second_order_optimizer_specs_are_accepted(changes, target):
    from workflows.optimize_params import validate_optimizer_spec

    validate_optimizer_spec(optimizer_spec_with(**changes), target, {})


@pytest.mark.parametrize(
    "changes, target, parameters, message",
    [
        ({"method": "not-a-method"}, "coeffs", {}, "Unknown optimizer.method"),
        ({"checkpoint": "everything"}, "coeffs", {}, "optimizer.checkpoint"),
        ({"mass_inverse": "lumped"}, "coeffs", {}, "optimizer.mass_inverse"),
        ({"method": "dogleg"}, "ic-params", {}, "full Hessian"),
        ({"method": "trust-exact"}, "ic-field", {}, "full Hessian"),
        ({"method": "Newton-CG"}, "coeffs", {}, "cannot handle bounds"),
        ({"method": "trust-ncg"}, "coeffs", {}, "cannot handle bounds"),
        ({"method": "L-BFGS-B", "use_hessp": True}, "coeffs", {}, "does not use Hessian-vector products"),
        ({"method": "trust-krylov", "use_hessp": False}, "ic-params", {}, "requires Hessian-vector products"),
        ({"method": "trust-constr", "use_hessp": "yes"}, "coeffs", {}, "use_hessp must be"),
        ({"method": "Newton-CG", "use_bounds": "no"}, "coeffs", {}, "use_bounds must be"),
        ({"method": "Newton-CG", "use_bounds": False}, "coeffs", {"a.b": {"bounds": [0.0, 1.0]}}, "would be ignored"),
        ({"method": "Newton-CG", "xtol": -1.0}, "ic-params", {}, "positive number"),
        ({"method": "L-BFGS-B", "xtol": 1.0e-8}, "coeffs", {}, "xtol is only used by"),
    ],
)
def test_invalid_optimizer_specs_are_rejected(changes, target, parameters, message):
    from workflows.optimize_params import validate_optimizer_spec

    with pytest.raises(ValueError, match=message):
        validate_optimizer_spec(optimizer_spec_with(**changes), target, parameters)


def write_example_with_optimizer(tmp_path, name, optimizer_changes):
    """Copy an example optimization spec to tmp_path with changed optimizer settings."""
    data = yaml.safe_load((WORKFLOWS / "examples" / name).read_text(encoding="utf-8"))
    data["optimizer"] = {**(data.get("optimizer") or {}), **optimizer_changes}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_spec_accepts_trust_constr_for_coeffs(tmp_path):
    from workflows.optimize_params import load_spec

    spec = load_spec(write_example_with_optimizer(tmp_path, "optimize_coeff.cfg", {"method": "trust-constr", "xtol": 1.0e-10}))
    assert spec["optimizer"]["method"] == "trust-constr"
    assert spec["optimizer"]["use_bounds"] is True


def test_load_spec_rejects_newton_cg_for_bounded_coeffs(tmp_path):
    from workflows.optimize_params import load_spec

    # the example sets per-parameter bounds, so use_bounds: false is rejected too
    with pytest.raises(ValueError, match="cannot handle bounds"):
        load_spec(write_example_with_optimizer(tmp_path, "optimize_coeff.cfg", {"method": "Newton-CG"}))
    with pytest.raises(ValueError, match="would be ignored"):
        load_spec(write_example_with_optimizer(tmp_path, "optimize_coeff.cfg", {"method": "Newton-CG", "use_bounds": False}))


@pytest.mark.parametrize("name", ["optimize_ic_params.cfg", "optimize_ic_field.cfg"])
def test_load_spec_accepts_newton_cg_for_unbounded_targets(tmp_path, name):
    from workflows.optimize_params import load_spec

    spec = load_spec(write_example_with_optimizer(tmp_path, name, {"method": "Newton-CG"}))
    assert spec["optimizer"]["method"] == "Newton-CG"
    assert spec["optimizer"]["use_hessp"] is None


def test_optimizer_summary_records_method_counts_and_status():
    from types import SimpleNamespace

    from workflows.optimize_params import optimizer_summary

    spec = optimizer_spec_with(method="trust-constr", xtol=1.0e-10)
    result = SimpleNamespace(success=True, status=1, message="`gtol` termination condition is satisfied.", nit=7, hessp_used=True)
    problem = SimpleNamespace(nfev=9, njev=8, nhvp=23)
    summary = optimizer_summary(spec, result, problem, gtol=3.0e-12)
    assert summary["method"] == "trust-constr"
    assert summary["xtol"] == 1.0e-10
    assert summary["use_hessp"] is None and summary["use_bounds"] is True
    assert (summary["nit"], summary["nfev"], summary["njev"], summary["nhvp"]) == (7, 9, 8, 23)
    assert summary["hessp_used"] is True
    assert summary["success"] is True and summary["status"] == 1
    assert summary["gtol_used"] == 3.0e-12


def test_optimizer_summary_is_backward_compatible_for_first_order_runs():
    from types import SimpleNamespace

    from workflows.optimize_params import optimizer_summary

    # results without HVP information (first-order runs, or problems without an nhvp counter)
    result = SimpleNamespace(success=False, status=1, message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", nit=100)
    problem = SimpleNamespace(nfev=120, njev=120)
    summary = optimizer_summary(optimizer_spec_with(), result, problem, gtol=1.0e-6)
    assert summary["method"] == "L-BFGS-B"
    assert summary["nhvp"] == 0 and summary["hessp_used"] is False
    # the fields validate.py and plot_validation.py read are unchanged
    for key in ("success", "status", "message", "nit", "nfev", "njev", "gtol_used"):
        assert key in summary


def test_history_appends_cumulative_counters_after_existing_columns(tmp_path):
    from types import SimpleNamespace

    from workflows.common import write_csv
    from workflows.optimize_params import HISTORY_COUNTERS, History

    problem = SimpleNamespace(nfev=1, njev=1, nhvp=0)
    history = History(["a.b", "a.c"], problem=problem)
    history(0, [1.0, 2.0], 5.0, 0.5)
    problem.nfev, problem.njev, problem.nhvp = 2, 2, 6
    history(1, [1.5, 2.5], 1.0, 0.1)
    assert list(history.rows[0]) == ["iteration", "loss", "grad_norm", "a.b", "a.c", *HISTORY_COUNTERS]
    assert [row["nhvp"] for row in history.rows] == [0, 6]
    assert [row["nfev"] for row in history.rows] == [1, 2]

    write_csv(tmp_path / "history.csv", history.rows)
    header = (tmp_path / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "iteration,loss,grad_norm,a.b,a.c,nfev,njev,nhvp"


def test_history_without_problem_keeps_previous_columns():
    from workflows.optimize_params import History

    history = History(None)
    history(0, [0.0], None, None)
    assert list(history.rows[0]) == ["iteration", "loss", "grad_norm"]


# Step 13: workflow-level reporting of second-order runs. These follow a result from
# optimizer_summary through result.yaml to validate.summary_row and summary.csv, without running
# the model.


def test_bounds_rejection_message_names_the_method_target_and_remedies():
    from workflows.optimize_params import validate_optimizer_spec

    with pytest.raises(ValueError) as error:
        validate_optimizer_spec(optimizer_spec_with(method="Newton-CG"), "coeffs", {})
    message = str(error.value)
    # the user's spelling of the method, the reason, and each way out are all in the message
    for text in ("Newton-CG", "coeffs target", "trust-constr", "L-BFGS-B", "optimizer.use_bounds: false"):
        assert text in message


def test_run_job_forwards_second_order_options_to_optimize():
    """run_job needs Firedrake, so check by inspection that the new optimizer options reach optimize()."""
    source = (WORKFLOWS / "optimize_params.py").read_text(encoding="utf-8")
    run_job_source = source[source.index("def run_job("):source.index("def main(")]
    for text in ('use_hessp=optimizer_spec["use_hessp"]', 'use_bounds=bool(optimizer_spec["use_bounds"])',
                 'optimizer_spec["xtol"]', "History(names, problem=problem)",
                 '"optimizer": optimizer_summary(optimizer_spec, result, problem, gtol)'):
        assert text in run_job_source, text


def second_order_result(**optimizer_changes):
    """Return a minimal result.yaml mapping, as run_job writes it, for a trust-constr coeffs run."""
    from types import SimpleNamespace

    from workflows.optimize_params import optimizer_summary

    spec = optimizer_spec_with(method="trust-constr", xtol=1.0e-10)
    scipy_result = SimpleNamespace(success=True, status=1, message="`gtol` termination condition is satisfied.",
                                   nit=6, hessp_used=True)
    optimizer = optimizer_summary(spec, scipy_result, SimpleNamespace(nfev=8, njev=7, nhvp=19), gtol=1.0e-12)
    optimizer.update(optimizer_changes)
    return {
        "job_name": "l-6__final", "target": "coeffs", "loss_label": "final",
        "initial_loss": 2.0, "final_loss": 1.0e-9, "optimizer": optimizer,
        "parameters": {"hyperviscosity.c0": {"initial": 0.05, "optimal": 0.0699, "true": 0.07, "free": True}},
    }


def test_second_order_optimizer_summary_survives_result_yaml(tmp_path):
    np = pytest.importorskip("numpy")
    from types import SimpleNamespace

    from workflows.common import dump_yaml, load_yaml, to_builtin
    from workflows.optimize_params import optimizer_summary

    # scipy and the counters may hand back numpy scalars; result.yaml must still hold builtins
    scipy_result = SimpleNamespace(success=np.bool_(True), status=np.int64(1), message="ok", nit=np.int64(4),
                                   hessp_used=np.bool_(True))
    problem = SimpleNamespace(nfev=np.int64(5), njev=np.int64(5), nhvp=np.int64(11))
    summary = optimizer_summary(optimizer_spec_with(method="Newton-CG", xtol=1.0e-9), scipy_result, problem, gtol=1.0e-8)
    dump_yaml(to_builtin({"optimizer": summary}), tmp_path / "result.yaml")
    loaded = load_yaml(tmp_path / "result.yaml")["optimizer"]
    assert loaded["method"] == "Newton-CG" and loaded["xtol"] == 1.0e-9
    assert (loaded["nit"], loaded["nfev"], loaded["njev"], loaded["nhvp"]) == (4, 5, 5, 11)
    assert loaded["hessp_used"] is True and loaded["use_hessp"] is None and loaded["use_bounds"] is True


def test_validation_summary_row_records_method_and_hvp_counts():
    from workflows.validate import summary_row

    row = summary_row(second_order_result(), {"case": "optimal", "truth_name": "l-6"}, {"errors": {}}, [])
    assert row["method"] == "trust-constr"
    assert (row["nit"], row["nfev"], row["njev"], row["nhvp"]) == (6, 8, 7, 19)
    assert row["hessp_used"] is True
    # the existing leading columns keep their order; the new ones follow nfev
    columns = list(row)
    assert columns[:10] == ["job", "case", "truth_run", "target", "loss", "initial_loss", "final_loss",
                            "optimizer_success", "nit", "nfev"]
    assert columns[10:14] == ["method", "njev", "nhvp", "hessp_used"]
    assert row["hyperviscosity.c0__value"] == 0.0699


def test_validation_summary_row_accepts_results_written_before_hvp_support():
    from workflows.validate import summary_row

    result = second_order_result()
    result["optimizer"] = {"success": True, "status": 0, "message": "CONVERGENCE", "nit": 12, "nfev": 15}
    row = summary_row(result, {"case": "initial-guess", "truth_name": "l-6"}, {"errors": {}}, [])
    assert (row["method"], row["njev"], row["nhvp"], row["hessp_used"]) == ("", 0, 0, False)
    assert row["hyperviscosity.c0__value"] == 0.05  # the initial-guess case reports the initial value


def test_summary_csv_compares_first_and_second_order_runs(tmp_path):
    from types import SimpleNamespace

    from workflows.common import read_csv, write_csv
    from workflows.optimize_params import optimizer_summary
    from workflows.validate import summary_row

    first_order = second_order_result()
    first_order["job_name"] = "l-6__final__lbfgsb"
    first_order["optimizer"] = optimizer_summary(
        optimizer_spec_with(), SimpleNamespace(success=True, status=0, message="CONVERGENCE", nit=25),
        SimpleNamespace(nfev=31, njev=31), gtol=1.0e-12)
    plan = {"case": "optimal", "truth_name": "l-6"}
    rows = [summary_row(result, plan, {"errors": {}}, []) for result in (first_order, second_order_result())]
    write_csv(tmp_path / "summary.csv", rows)
    loaded = {row["job"]: row for row in read_csv(tmp_path / "summary.csv")}

    assert loaded["l-6__final__lbfgsb"]["method"] == "L-BFGS-B"
    assert loaded["l-6__final__lbfgsb"]["nhvp"] == 0.0
    assert loaded["l-6__final__lbfgsb"]["hessp_used"] == "False"
    assert loaded["l-6__final"]["method"] == "trust-constr"
    assert (loaded["l-6__final"]["njev"], loaded["l-6__final"]["nhvp"]) == (7.0, 19.0)
    assert loaded["l-6__final"]["hessp_used"] == "True"


# Step 14: first- versus second-order example pairs. Each second-order example solves exactly the
# problem of its first-order counterpart, so their outputs can be compared job by job.
EXAMPLE_METHOD_PAIRS = [
    ("optimize_coeff.cfg", "optimize_coeff_trust_constr.cfg", "trust-constr"),
    ("optimize_ic_params.cfg", "optimize_ic_params_newton_cg.cfg", "Newton-CG"),
]


@pytest.mark.parametrize("first_order, second_order, method", EXAMPLE_METHOD_PAIRS)
def test_second_order_examples_match_their_first_order_problem(first_order, second_order, method):
    first = yaml.safe_load((WORKFLOWS / "examples" / first_order).read_text(encoding="utf-8"))
    second = yaml.safe_load((WORKFLOWS / "examples" / second_order).read_text(encoding="utf-8"))
    assert first["optimizer"]["method"] == "L-BFGS-B"
    assert second["optimizer"]["method"] == method
    # distinct output roots, so the two runs do not overwrite each other's job directories
    assert first["optimization_name"] != second["optimization_name"]
    assert first.get("output_root") == second.get("output_root")
    # the optimization problem itself is identical
    problem_keys = (set(first) | set(second)) - {"optimizer", "optimization_name"}
    for key in sorted(problem_keys):
        assert first.get(key) == second.get(key), key
    # the shared stopping settings are identical too, so only the method differs
    for key in ("gtol", "ftol"):
        assert first["optimizer"].get(key) == second["optimizer"].get(key), key


@pytest.mark.parametrize("first_order, second_order, method", EXAMPLE_METHOD_PAIRS)
def test_second_order_examples_resolve_to_hessian_vector_products(first_order, second_order, method):
    from workflows.optimize_params import HESSP_METHODS, load_spec, target_has_bounds

    first = load_spec(WORKFLOWS / "examples" / first_order)["optimizer"]
    second_spec = load_spec(WORKFLOWS / "examples" / second_order)
    second = second_spec["optimizer"]
    assert first["method"].upper() not in HESSP_METHODS and first["use_hessp"] is None
    assert second["method"].upper() in HESSP_METHODS and second["use_hessp"] is True
    assert second["xtol"] is not None
    if target_has_bounds(second_spec["target"], second["use_bounds"]):
        # only trust-constr handles the coefficient bounds of the coeffs target
        assert second["method"] == "trust-constr"


def test_second_order_examples_are_listed_in_the_workflow_readme():
    readme = (WORKFLOWS / "README.md").read_text(encoding="utf-8")
    for _, second_order, _ in EXAMPLE_METHOD_PAIRS:
        assert f"examples/{second_order}" in readme


# Step 17: documentation of the second-order capability.


def test_readme_no_longer_claims_first_order_only():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "only a Jacobian" not in readme
    assert "forward-over-adjoint" in readme and "2.11" in readme and "petsc-opt.pdf" in readme


def test_workflow_readme_documents_second_order_methods_and_reporting():
    from workflows.optimize_params import DEFAULT_OPTIMIZER, HESSP_METHODS

    readme = (WORKFLOWS / "README.md").read_text(encoding="utf-8")
    assert "### Optimizer methods" in readme
    assert "forward-over-adjoint" in readme and "equation 2.11 of `petsc-opt.pdf`" in readme
    # every HVP method, as spelled in scipy, and the rejected full-Hessian methods
    for method in ("Newton-CG", "trust-ncg", "trust-krylov", "trust-constr", "dogleg", "trust-exact"):
        assert f"`{method}`" in readme, method
    assert len(HESSP_METHODS) == 4
    # every optimizer spec key is shown in the spec example
    for key in DEFAULT_OPTIMIZER:
        assert f"  {key}:" in readme, key
    # the new result fields and the comparison script
    for text in ("`nhvp`", "`hessp_used`", "`optimize_seconds`", "compare_optimizers.sh", "optimizer-comparison.csv"):
        assert text in readme, text
