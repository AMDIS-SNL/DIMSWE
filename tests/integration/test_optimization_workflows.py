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
    ["truth_coeff_sweep.cfg", "truth_ic_sweep.cfg", "optimize_coeff.cfg", "optimize_ic_params.cfg", "optimize_ic_field.cfg"],
)
def test_example_optimization_cfgs_are_valid_yaml(name):
    data = yaml.safe_load((WORKFLOWS / "examples" / name).read_text(encoding="utf-8"))
    assert data


@pytest.mark.parametrize(
    "name",
    ["optimize_coeff.cfg", "optimize_ic_params.cfg", "optimize_ic_field.cfg"],
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
