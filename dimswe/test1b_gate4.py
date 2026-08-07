"""Pure aggregation and lightweight plotting for completed Test-1B Gate 4."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from .learned_physics.objectives import TrainingMode
from .resolved_hidden_c0 import read_json_record, write_json_record


STRATEGIES = tuple(mode.value for mode in TrainingMode)


def _completed_gate4_record(strategy, record):
    if not isinstance(record, Mapping):
        raise TypeError(f"Gate-4 record for {strategy} must be a mapping")
    if record.get("status") != "complete":
        raise ValueError(f"Gate-4 record for {strategy} is not complete")
    if record.get("fitted_training_mode") != strategy:
        raise ValueError(f"Gate-4 record for {strategy} has the wrong fit mode")
    provenance = record.get("fit_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("success") is not True:
        raise ValueError(f"Gate-4 record for {strategy} lacks a successful fit")
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"Gate-4 record for {strategy} lacks evaluation data")
    certification = evaluation.get("gate4_certification")
    if not isinstance(certification, Mapping) or certification.get("passed") is not True:
        raise ValueError(f"Gate-4 record for {strategy} did not pass certification")
    return provenance, evaluation


def aggregate_gate4_records(records):
    """Return a deterministic compact table plus the three plotting histories."""
    supplied = set(records)
    expected = set(STRATEGIES)
    if supplied != expected:
        raise ValueError(
            "Gate-4 aggregation requires exactly "
            f"{STRATEGIES}; missing={sorted(expected - supplied)}, "
            f"extra={sorted(supplied - expected)}"
        )
    rows = []
    plot_data = {}
    for strategy in STRATEGIES:
        provenance, evaluation = _completed_gate4_record(
            strategy, records[strategy]
        )
        mixed = evaluation["heldout_autonomous_trajectory_error"]
        kinetic = evaluation["kinetic_energy_mismatch"]
        enstrophy = evaluation["projected_enstrophy_mismatch"]
        rows.append(
            {
                "strategy": strategy,
                "starting_c0": provenance["starting_c0"],
                "learned_c0": provenance["recovered_c0"],
                "maximum_mixed_state_error": mixed["maximum"],
                "final_mixed_state_error": mixed["final"],
                "maximum_kinetic_energy_absolute_mismatch": kinetic[
                    "maximum_absolute_mismatch"
                ],
                "final_kinetic_energy_absolute_mismatch": kinetic[
                    "final_absolute_mismatch"
                ],
                "maximum_kinetic_energy_relative_mismatch": kinetic[
                    "maximum_relative_mismatch"
                ],
                "final_kinetic_energy_relative_mismatch": kinetic[
                    "final_relative_mismatch"
                ],
                "maximum_projected_enstrophy_absolute_mismatch": enstrophy[
                    "maximum_absolute_mismatch"
                ],
                "final_projected_enstrophy_absolute_mismatch": enstrophy[
                    "final_absolute_mismatch"
                ],
                "maximum_projected_enstrophy_relative_mismatch": enstrophy[
                    "maximum_relative_mismatch"
                ],
                "final_projected_enstrophy_relative_mismatch": enstrophy[
                    "final_relative_mismatch"
                ],
                "complete_solver_steps": evaluation[
                    "heldout_deployment_contract"
                ]["complete_production_steps"],
                "wall_time_seconds": evaluation["wall_time_seconds"],
                "passed": True,
            }
        )
        plot_data[strategy] = {
            "times": mixed["times"],
            "mixed_state_relative_mass_norm_error": mixed[
                "relative_mass_norm_error"
            ],
            "kinetic_energy_truth": kinetic["truth"],
            "kinetic_energy_deployed": kinetic["predicted"],
            "projected_enstrophy_truth": enstrophy["truth"],
            "projected_enstrophy_deployed": enstrophy["predicted"],
        }
    return {
        "benchmark": "Test 1B",
        "gate": "post-fit autonomous held-out deployment",
        "status": "complete",
        "strategies": tuple(rows),
        "plot_data": plot_data,
        "interpretation": (
            "deterministic end-to-end workflow certification, not difficult "
            "extrapolation or machine-learning generalization"
        ),
    }


def plot_gate4_summary(summary, output_directory):
    """Write three simple diagnostic figures from an aggregate JSON record."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    plot_data = summary["plot_data"]

    figure, axis = plt.subplots()
    for strategy in STRATEGIES:
        values = plot_data[strategy]
        axis.plot(
            values["times"],
            values["mixed_state_relative_mass_norm_error"],
            label=strategy,
        )
    axis.set(xlabel="time", ylabel="held-out mixed-state relative mass error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "gate4_mixed_state_error.png", dpi=150)
    plt.close(figure)

    for key, ylabel, filename in (
        ("kinetic_energy", "kinetic energy", "gate4_kinetic_energy.png"),
        (
            "projected_enstrophy",
            "projected enstrophy",
            "gate4_projected_enstrophy.png",
        ),
    ):
        figure, axis = plt.subplots()
        first = plot_data[STRATEGIES[0]]
        axis.plot(first["times"], first[f"{key}_truth"], label="truth")
        for strategy in STRATEGIES:
            values = plot_data[strategy]
            axis.plot(
                values["times"],
                values[f"{key}_deployed"],
                label=strategy,
            )
        axis.set(xlabel="time", ylabel=ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / filename, dpi=150)
        plt.close(figure)


def _parse_result_arguments(values):
    records = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--result must have STRATEGY=PATH form")
        strategy, path = value.split("=", 1)
        if strategy in records:
            raise ValueError(f"duplicate Gate-4 strategy {strategy}")
        records[strategy] = read_json_record(path)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="completed Gate-4 record in STRATEGY=PATH form",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-directory")
    arguments = parser.parse_args(argv)
    summary = aggregate_gate4_records(_parse_result_arguments(arguments.result))
    write_json_record(arguments.output, summary)
    if arguments.plot_directory:
        plot_gate4_summary(summary, arguments.plot_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "STRATEGIES",
    "aggregate_gate4_records",
    "plot_gate4_summary",
)
