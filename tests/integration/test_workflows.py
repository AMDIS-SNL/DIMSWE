from pathlib import Path

import pytest
import yaml

from workflows.common import (
    derive_schedule_overrides,
    expand_parameter_grid,
    expand_tied_parameter_sets,
    merge_parameter_overrides,
)
from workflows.generate_cfgs import apply_overrides


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ({}, [{}]),
        (
            {"mesh.nx": [25, 50], "spatial-discretization.order": [2, 3]},
            [
                {"mesh.nx": 25, "spatial-discretization.order": 2},
                {"mesh.nx": 25, "spatial-discretization.order": 3},
                {"mesh.nx": 50, "spatial-discretization.order": 2},
                {"mesh.nx": 50, "spatial-discretization.order": 3},
            ],
        ),
    ],
)
def test_expand_parameter_grid(parameters, expected):
    assert expand_parameter_grid(parameters) == expected


def test_apply_overrides_updates_nested_paths():
    parameters = {"mesh": {"nx": 10}, "output": {"outfile_name": "sim"}}
    apply_overrides(parameters, {"mesh.nx": 50, "output.outfile_name": "runs/test/sim"})
    assert parameters["mesh"]["nx"] == 50
    assert parameters["output"]["outfile_name"] == "runs/test/sim"


def test_expand_tied_parameter_sets_returns_one_axis_of_grouped_values():
    tied_sets = [
        {"mesh.nx": 25, "mesh.ny": 25},
        {"mesh.nx": 50, "mesh.ny": 50},
    ]
    assert expand_tied_parameter_sets(tied_sets) == tied_sets


def test_merge_parameter_overrides_rejects_conflicts():
    with pytest.raises(ValueError, match="Conflicting override"):
        merge_parameter_overrides({"mesh.nx": 25}, {"mesh.nx": 50})


def test_merge_parameter_overrides_accepts_independent_and_tied_values():
    merged = merge_parameter_overrides(
        {"spatial-discretization.order": 3},
        {"mesh.nx": 50, "mesh.ny": 50},
    )
    assert merged == {
        "spatial-discretization.order": 3,
        "mesh.nx": 50,
        "mesh.ny": 50,
    }


def test_derive_schedule_overrides_scales_dt_steps_and_output_frequency():
    derived = derive_schedule_overrides(
        {
            "mesh.nx": 25,
            "mesh.ny": 25,
            "timestepping.dt": 400,
            "timestepping.num_steps": 1000,
            "output.output_freq": 50,
        },
        {"mesh.nx": 50, "mesh.ny": 50},
        ["mesh.nx", "mesh.ny"],
    )
    assert derived == {
        "timestepping.dt": 200.0,
        "timestepping.num_steps": 2000,
        "output.output_freq": 100,
    }


def test_derive_schedule_overrides_rejects_noninteger_refinement_ratio():
    with pytest.raises(ValueError, match="integer ratio"):
        derive_schedule_overrides(
            {
                "mesh.nx": 25,
                "mesh.ny": 25,
                "timestepping.dt": 400,
                "timestepping.num_steps": 1000,
                "output.output_freq": 50,
            },
            {"mesh.nx": 40, "mesh.ny": 40},
            ["mesh.nx", "mesh.ny"],
        )


def test_example_compare_cfg_is_valid_yaml():
    compare_cfg = Path("/projects/DIMSWE/workflows/examples/compare.cfg")
    data = yaml.safe_load(compare_cfg.read_text(encoding="utf-8"))
    assert "runs" in data
    assert "quantities" in data


def test_run_batch_wrapper_uses_explicit_mpi_ranks():
    wrapper = Path("/projects/DIMSWE/workflows/run_batch.sh").read_text(encoding="utf-8")
    assert 'MPI_RANKS="$1"' in wrapper
    assert 'mpiexec -n "$MPI_RANKS" python -m workflows.run_batch "$@"' in wrapper


def test_run_batch_uses_safe_error_collection_for_python_strings():
    run_batch = Path("/projects/DIMSWE/workflows/run_batch.py").read_text(encoding="utf-8")
    assert "def collect_error_text" in run_batch
    assert "COMM.allgather(local_error_text)" in run_batch
    assert "COMM.allreduce(error_text, op=MPI.MAX)" not in run_batch


def test_compare_runs_uses_horizontal_layout_and_funcanimation():
    compare_runs = Path("/projects/DIMSWE/workflows/compare_runs.py").read_text(encoding="utf-8")
    assert "width_ratios" in compare_runs
    assert "cax=colorbar_axis" in compare_runs
    assert "animation.FuncAnimation" in compare_runs
    assert "update(0)" in compare_runs
    assert "ArtistAnimation" not in compare_runs
