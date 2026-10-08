"""Post-process completed resolved hidden-c0 pilot runs without timestepping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .resolved_hidden_c0 import (
    STATE_FIELDS,
    LateTimeGrowthConfiguration,
    late_time_growth_indicator,
    read_json_record,
    write_json_record,
)


def _metadata(run_directory: str | Path) -> tuple[Path, dict[str, Any]]:
    directory = Path(run_directory).resolve()
    metadata = read_json_record(directory / "metadata.json")
    if metadata.get("status") != "complete":
        raise RuntimeError(f"pilot run {directory} is not complete")
    return directory, metadata


def _diagnostic_by_step(metadata):
    return {int(value["step"]): value for value in metadata["diagnostics"]}


def _state_path(directory: Path, step: int) -> Path:
    return directory / "restart" / f"step_{step:08d}.npy"


def _spectrum_path(directory: Path, step: int) -> Path:
    return directory / "spectra" / f"step_{step:08d}.npz"


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / "checkpoints" / f"step_{step:08d}.h5"


def _load_state(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise FloatingPointError(f"invalid restart state {path}")
    return values


def _mass_analysis_template(directory, metadata, step):
    # Firedrake remains a lazy analysis-only dependency.  No timestepper or
    # solver model is built and no saved run is mutated.
    from firedrake import CheckpointFile, dx

    from .meshes import gauss_lobatto_legendre_cube_rule

    checkpoint_path = _checkpoint_path(directory, step)
    with CheckpointFile(str(checkpoint_path), "r") as checkpoint:
        mesh = checkpoint.load_mesh()
        template = checkpoint.load_function(mesh, "mixed_state", idx=0)
    degree = int(metadata["finite_element_spaces"]["h"].split("(")[1].split(")")[0])
    rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=degree)
    return template, dx(scheme=rule)


def _set_state(function, values):
    from .numpy_helpers import set_mixed_function_from_flattened_array

    set_mixed_function_from_flattened_array(function, values)


def _mass_separations(template, measure, left_values, right_values):
    from firedrake import assemble, inner

    left = template.copy(deepcopy=True)
    right = template.copy(deepcopy=True)
    _set_state(left, left_values)
    _set_state(right, right_values)
    result = {}
    residual = left.copy(deepcopy=True)
    with residual.dat.vec as residual_vec, right.dat.vec_ro as right_vec:
        residual_vec.axpy(-1.0, right_vec)
    numerator = float(assemble(inner(residual, residual) * measure))
    denominator = float(assemble(inner(right, right) * measure))
    mixed = float(
        np.sqrt(numerator / max(denominator, np.finfo(float).tiny))
    )
    for index, name in enumerate(STATE_FIELDS):
        field_residual = left.sub(index) - right.sub(index)
        field_numerator = float(
            assemble(inner(field_residual, field_residual) * measure)
        )
        field_denominator = float(
            assemble(inner(right.sub(index), right.sub(index)) * measure)
        )
        result[name] = float(
            np.sqrt(
                field_numerator
                / max(field_denominator, np.finfo(float).tiny)
            )
        )
    return mixed, result


def _relative_history(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return np.abs(a - b) / np.maximum(np.abs(b), np.finfo(float).tiny)


def _spectrum_mismatch(left_path, right_path):
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        a = np.asarray(left["shell_energy_sum"], dtype=np.float64)
        b = np.asarray(right["shell_energy_sum"], dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("paired spectra use different shell grids")
        mismatch = np.linalg.norm(a - b) / max(
            np.linalg.norm(b), np.finfo(float).tiny
        )
        return {
            "relative_shell_energy_l2": float(mismatch),
            "c0_a_high_wavenumber_fraction": float(
                left["high_wavenumber_fraction"]
            ),
            "c0_b_high_wavenumber_fraction": float(
                right["high_wavenumber_fraction"]
            ),
        }


def _growth_heuristics(times, kinetic, enstrophy, hyperviscosity, high_k, config):
    common = {
        "baseline_fraction": config.baseline_fraction,
        "tail_fraction": config.tail_fraction,
    }
    return {
        "kinetic_energy": late_time_growth_indicator(
            times,
            kinetic,
            growth_factor=config.kinetic_energy_factor,
            absolute_floor=config.absolute_floor,
            **common,
        ),
        "projected_enstrophy": late_time_growth_indicator(
            times,
            enstrophy,
            growth_factor=config.projected_enstrophy_factor,
            absolute_floor=config.absolute_floor,
            **common,
        ),
        "hyperviscosity_tendency_mass_norm": late_time_growth_indicator(
            times,
            hyperviscosity,
            growth_factor=config.hyperviscosity_tendency_factor,
            absolute_floor=config.absolute_floor,
            **common,
        ),
        "velocity_high_wavenumber_energy_fraction": late_time_growth_indicator(
            times,
            high_k,
            growth_factor=config.high_wavenumber_fraction_factor,
            absolute_floor=config.high_wavenumber_absolute_floor,
            **common,
        ),
    }


def _growth_classification(all_finite, diagnostics):
    if not all_finite:
        return "nonfinite trajectory"
    if any(
        value["suspicious_late_time_growth"] for value in diagnostics.values()
    ):
        return "suspicious late-time growth detected"
    return "no suspicious late-time growth detected; stability is not proved"


def compare_completed_pilot_runs(
    run_a: str | Path,
    run_b: str | Path,
    *,
    separation_threshold: float = 1.0e-10,
    high_wavenumber_threshold: float = 1.0e-6,
    growth_configuration: LateTimeGrowthConfiguration = (
        LateTimeGrowthConfiguration()
    ),
):
    """Compare an identical-physics c0 pair using the deployed mass metric.

    Thresholds produce diagnostic flags only.  The routine intentionally does
    not select a flow, resolution, timestep, or duration.
    """
    directory_a, metadata_a = _metadata(run_a)
    directory_b, metadata_b = _metadata(run_b)
    if metadata_a["paired_non_c0_physics_sha256"] != metadata_b[
        "paired_non_c0_physics_sha256"
    ]:
        raise ValueError("pilot runs differ in physics beyond c0")
    c0_a = float(metadata_a["physical_parameters"]["c0"])
    c0_b = float(metadata_b["physical_parameters"]["c0"])
    if c0_a == c0_b:
        raise ValueError("pilot comparison requires distinct c0 values")
    steps_a = tuple(int(value) for value in metadata_a["completed_output_steps"])
    steps_b = tuple(int(value) for value in metadata_b["completed_output_steps"])
    if steps_a != steps_b:
        raise ValueError("pilot runs do not share output steps")
    template, measure = _mass_analysis_template(directory_a, metadata_a, steps_a[0])
    diagnostics_a = _diagnostic_by_step(metadata_a)
    diagnostics_b = _diagnostic_by_step(metadata_b)
    mixed = []
    fieldwise = {name: [] for name in STATE_FIELDS}
    spectral = []
    for step in steps_a:
        state_a = _load_state(_state_path(directory_a, step))
        state_b = _load_state(_state_path(directory_b, step))
        separation, fields = _mass_separations(
            template, measure, state_a, state_b
        )
        mixed.append(separation)
        for name in STATE_FIELDS:
            fieldwise[name].append(fields[name])
        spectral.append(
            _spectrum_mismatch(
                _spectrum_path(directory_a, step),
                _spectrum_path(directory_b, step),
            )
        )
    times = np.array(
        [float(diagnostics_a[step]["time"]) for step in steps_a],
        dtype=np.float64,
    )
    kinetic_a = np.array(
        [diagnostics_a[step]["kinetic_energy"] for step in steps_a]
    )
    kinetic_b = np.array(
        [diagnostics_b[step]["kinetic_energy"] for step in steps_a]
    )
    enstrophy_a = np.array(
        [diagnostics_a[step]["projected_enstrophy"] for step in steps_a]
    )
    enstrophy_b = np.array(
        [diagnostics_b[step]["projected_enstrophy"] for step in steps_a]
    )
    hyper_a = np.array(
        [
            diagnostics_a[step]["hyperviscosity_tendency_mass_norm"]
            for step in steps_a
        ]
    )
    hyper_b = np.array(
        [
            diagnostics_b[step]["hyperviscosity_tendency_mass_norm"]
            for step in steps_a
        ]
    )
    high_a = np.array(
        [item["c0_a_high_wavenumber_fraction"] for item in spectral]
    )
    high_b = np.array(
        [item["c0_b_high_wavenumber_fraction"] for item in spectral]
    )
    mixed_array = np.asarray(mixed)
    above = np.flatnonzero(mixed_array > separation_threshold)
    finite_a = all(
        diagnostics_a[step]["all_state_coefficients_finite"]
        for step in steps_a
    ) and bool(
        np.all(
            np.isfinite(
                np.concatenate(
                    (kinetic_a, enstrophy_a, hyper_a, high_a)
                )
            )
        )
    )
    finite_b = all(
        diagnostics_b[step]["all_state_coefficients_finite"]
        for step in steps_a
    ) and bool(
        np.all(
            np.isfinite(
                np.concatenate((kinetic_b, enstrophy_b, hyper_b, high_b))
            )
        )
    )
    all_finite = finite_a and finite_b and bool(np.all(np.isfinite(mixed_array)))
    growth_a = _growth_heuristics(
        times, kinetic_a, enstrophy_a, hyper_a, high_a, growth_configuration
    )
    growth_b = _growth_heuristics(
        times, kinetic_b, enstrophy_b, hyper_b, high_b, growth_configuration
    )
    classification_a = _growth_classification(finite_a, growth_a)
    classification_b = _growth_classification(finite_b, growth_b)
    growth_warning = any(
        value["suspicious_late_time_growth"]
        for value in tuple(growth_a.values()) + tuple(growth_b.values())
    )
    return {
        "analysis": "Test 1B-0 paired resolved hidden-c0 diagnostics",
        "run_a": str(directory_a),
        "run_b": str(directory_b),
        "case": metadata_a["case"],
        "nx": metadata_a["mesh"]["nx"],
        "ny": metadata_a["mesh"]["ny"],
        "c0_a": c0_a,
        "c0_b": c0_b,
        "steps": steps_a,
        "times": times.tolist(),
        "mixed_state_mass_separation": mixed_array.tolist(),
        "fieldwise_mass_separation": fieldwise,
        "kinetic_energy": {
            "c0_a": kinetic_a.tolist(),
            "c0_b": kinetic_b.tolist(),
            "relative_mismatch": _relative_history(
                kinetic_a, kinetic_b
            ).tolist(),
        },
        "projected_enstrophy": {
            "c0_a": enstrophy_a.tolist(),
            "c0_b": enstrophy_b.tolist(),
            "relative_mismatch": _relative_history(
                enstrophy_a, enstrophy_b
            ).tolist(),
        },
        "hyperviscosity_tendency_mass_norm": {
            "c0_a": hyper_a.tolist(),
            "c0_b": hyper_b.tolist(),
            "relative_mismatch": _relative_history(hyper_a, hyper_b).tolist(),
        },
        "velocity_high_wavenumber_energy_fraction": {
            "c0_a": high_a.tolist(),
            "c0_b": high_b.tolist(),
        },
        "spectral_mismatch": spectral,
        "finite_state_check": {
            "all_states_finite": all_finite,
            "all_states_finite_c0_a": finite_a,
            "all_states_finite_c0_b": finite_b,
            "minimum_height_coefficient_c0_a": min(
                diagnostics_a[step]["minimum_height_coefficient"] for step in steps_a
            ),
            "minimum_height_coefficient_c0_b": min(
                diagnostics_b[step]["minimum_height_coefficient"] for step in steps_a
            ),
            "semantic_contract": (
                "finite coefficients and positive height do not establish "
                "numerical stability"
            ),
        },
        "numerical_stability_heuristics": {
            "kind": (
                "data-dependent late-time growth warnings; neither a necessary "
                "nor sufficient proof of instability"
            ),
            "configuration": growth_configuration.__dict__,
            "c0_a": growth_a,
            "c0_b": growth_b,
            "classification_c0_a": classification_a,
            "classification_c0_b": classification_b,
            "any_suspicious_late_time_growth": growth_warning,
            "exact_child_audit_required": (
                "interpret alongside dimswe.hyperviscosity_stability; finite-state "
                "and growth heuristics do not replace the Euler spectral bound"
            ),
        },
        "diagnostic_flags_not_case_selection": {
            "separation_threshold": separation_threshold,
            "separation_beyond_threshold": bool(above.size),
            "first_separation_time": (
                None if not above.size else float(times[int(above[0])])
            ),
            "maximum_mixed_separation": float(np.max(mixed_array)),
            "high_wavenumber_threshold": high_wavenumber_threshold,
            "high_wavenumber_threshold_exceeded": bool(
                max(float(np.max(high_a)), float(np.max(high_b)))
                > high_wavenumber_threshold
            ),
            "high_wavenumber_interpretation": (
                "threshold exceedance alone is not evidence of physically "
                "populated high modes, especially when growth is suspicious or "
                "exact Euler-child stability has not been certified"
            ),
            "nonzero_hyperviscosity_proxy": bool(
                max(float(np.max(hyper_a)), float(np.max(hyper_b)))
                > np.finfo(float).tiny
            ),
            "finite_state_check_passed": all_finite,
            "numerical_stability_heuristic_warning": growth_warning,
        },
        "selection_authorized": False,
    }


def compare_resolution_summaries(summaries):
    """Collect resolution trends without declaring scientific adequacy."""
    records = []
    for path in summaries:
        summary = read_json_record(path)
        flags = summary["diagnostic_flags_not_case_selection"]
        records.append(
            {
                "summary": str(Path(path).resolve()),
                "case": summary["case"],
                "nx": summary["nx"],
                "ny": summary["ny"],
                "c0_a": summary["c0_a"],
                "c0_b": summary["c0_b"],
                "maximum_mixed_separation": flags["maximum_mixed_separation"],
                "first_separation_time": flags["first_separation_time"],
                "maximum_high_wavenumber_fraction": max(
                    summary["velocity_high_wavenumber_energy_fraction"]["c0_a"]
                    + summary["velocity_high_wavenumber_energy_fraction"]["c0_b"]
                ),
                "finite_state_check_passed": flags["finite_state_check_passed"],
                "numerical_stability_heuristic_warning": flags[
                    "numerical_stability_heuristic_warning"
                ],
            }
        )
    records.sort(key=lambda value: (value["case"], value["nx"], value["ny"]))
    return {
        "analysis": "Test 1B-0 resolution trend",
        "records": records,
        "selection_authorized": False,
        "required_human_review": (
            "inspect separation onset, high-mode history, exact-child bound, "
            "growth warnings, finite status, and resolution trend before "
            "selecting Test 1B"
        ),
    }


def plot_paired_summary(summary: Mapping[str, Any], directory: str | Path):
    """Optionally plot an already computed summary; no solver is imported."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("optional plots require matplotlib") from exc
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    time = np.asarray(summary["times"])
    figures = (
        (
            "mixed_separation",
            (summary["mixed_state_mass_separation"],),
            ("mixed state",),
            "normalized separation",
        ),
        (
            "kinetic_energy",
            (
                summary["kinetic_energy"]["c0_a"],
                summary["kinetic_energy"]["c0_b"],
            ),
            (f"c0={summary['c0_a']}", f"c0={summary['c0_b']}"),
            "kinetic energy",
        ),
        (
            "high_wavenumber_fraction",
            (
                summary["velocity_high_wavenumber_energy_fraction"]["c0_a"],
                summary["velocity_high_wavenumber_energy_fraction"]["c0_b"],
            ),
            (f"c0={summary['c0_a']}", f"c0={summary['c0_b']}"),
            "high-wavenumber energy fraction",
        ),
    )
    paths = []
    for name, histories, labels, ylabel in figures:
        figure, axis = plt.subplots()
        for history, label in zip(histories, labels):
            axis.plot(time, history, label=label)
        axis.set_xlabel("time")
        axis.set_ylabel(ylabel)
        if len(histories) > 1:
            axis.legend()
        axis.grid(True, alpha=0.25)
        destination = output / f"{name}.png"
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(destination))
    return tuple(paths)


def _parser():
    parser = argparse.ArgumentParser(description="Analyze completed Test 1B-0 pilots")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("run_a")
    pair.add_argument("run_b")
    pair.add_argument("--output", required=True)
    pair.add_argument("--plot-directory")
    pair.add_argument("--separation-threshold", type=float, default=1.0e-10)
    pair.add_argument("--high-wavenumber-threshold", type=float, default=1.0e-6)
    defaults = LateTimeGrowthConfiguration()
    pair.add_argument(
        "--growth-baseline-fraction",
        type=float,
        default=defaults.baseline_fraction,
    )
    pair.add_argument(
        "--growth-tail-fraction", type=float, default=defaults.tail_fraction
    )
    pair.add_argument(
        "--kinetic-energy-growth-factor",
        type=float,
        default=defaults.kinetic_energy_factor,
    )
    pair.add_argument(
        "--enstrophy-growth-factor",
        type=float,
        default=defaults.projected_enstrophy_factor,
    )
    pair.add_argument(
        "--hyperviscosity-growth-factor",
        type=float,
        default=defaults.hyperviscosity_tendency_factor,
    )
    pair.add_argument(
        "--high-wavenumber-growth-factor",
        type=float,
        default=defaults.high_wavenumber_fraction_factor,
    )
    resolution = subparsers.add_parser("resolutions")
    resolution.add_argument("summaries", nargs="+")
    resolution.add_argument("--output", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "pair":
        growth = LateTimeGrowthConfiguration(
            baseline_fraction=arguments.growth_baseline_fraction,
            tail_fraction=arguments.growth_tail_fraction,
            kinetic_energy_factor=arguments.kinetic_energy_growth_factor,
            projected_enstrophy_factor=arguments.enstrophy_growth_factor,
            hyperviscosity_tendency_factor=(
                arguments.hyperviscosity_growth_factor
            ),
            high_wavenumber_fraction_factor=(
                arguments.high_wavenumber_growth_factor
            ),
        )
        summary = compare_completed_pilot_runs(
            arguments.run_a,
            arguments.run_b,
            separation_threshold=arguments.separation_threshold,
            high_wavenumber_threshold=arguments.high_wavenumber_threshold,
            growth_configuration=growth,
        )
        write_json_record(arguments.output, summary)
        if arguments.plot_directory:
            plot_paired_summary(summary, arguments.plot_directory)
    else:
        summary = compare_resolution_summaries(arguments.summaries)
        write_json_record(arguments.output, summary)
    print(json.dumps({"output": str(Path(arguments.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "compare_completed_pilot_runs",
    "compare_resolution_summaries",
    "plot_paired_summary",
)
