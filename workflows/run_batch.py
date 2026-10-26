"""Run a manual list of cfg files or a generated sweep manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI

from workflows.common import (
    build_output_basename,
    build_run_directory,
    deep_copy_parameters,
    dump_yaml,
    ensure_output_sections,
    load_yaml,
    resolve_path,
    resolve_runs_root,
    timestamp_utc,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfgs", nargs="*", help="Simulation cfg files to run")
    parser.add_argument("--manifest", help="Path to a generated sweep manifest")
    parser.add_argument(
        "--runs-root",
        help="Runs root directory. Defaults to runs/ for cfg files; manifest runs default to the "
        "run_directory recorded in the manifest (the sweep's output_root), and are moved under this "
        "root only when it is given explicitly",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with existing sim.h5 output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed run")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a generated sweep manifest."""
    return load_yaml(path)


def manifest_run_directory(run: dict[str, Any], manifest: dict[str, Any], runs_root_override: Path | None) -> Path:
    """Return where a manifest run is written.

    The manifest records each run's directory (under the sweep's output_root), and downstream
    workflows such as optimize_params.py read the runs from there, so it is used unless an
    explicit --runs-root moves the runs. Manifests from before run_directory was recorded fall
    back to the manifest's runs_root.
    """
    if runs_root_override is not None:
        return build_run_directory(runs_root_override, run["run_name"])
    if run.get("run_directory"):
        return resolve_path(run["run_directory"])
    return build_run_directory(resolve_runs_root(manifest.get("runs_root")), run["run_name"])


def build_run_specifications(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build a normalized list of runs, each with the directory it is written to."""
    runs_root_override = resolve_runs_root(args.runs_root) if args.runs_root else None
    run_specs: list[dict[str, Any]] = []
    if args.manifest:
        manifest_path = resolve_path(args.manifest)
        manifest = load_manifest(manifest_path)
        for run in manifest.get("runs", []):
            run_specs.append(
                {
                    "cfg_path": resolve_path(run["cfg_path"]),
                    "run_name": run["run_name"],
                    "run_directory": manifest_run_directory(run, manifest, runs_root_override),
                    "parameter_values": run.get("parameter_values", {}),
                    "source": str(manifest_path),
                }
            )
        if runs_root_override is not None:
            root_print(
                f"Note: --runs-root moves manifest runs to {runs_root_override}; workflows that read "
                "this manifest (for example optimize_params.py) will not find them there, so list the "
                "run directories under truth.runs instead"
            )
    for cfg in args.cfgs:
        cfg_path = resolve_path(cfg)
        run_specs.append(
            {
                "cfg_path": cfg_path,
                "run_name": cfg_path.stem,
                "run_directory": build_run_directory(resolve_runs_root(args.runs_root), cfg_path.stem),
                "parameter_values": {},
                "source": str(cfg_path),
            }
        )
    return run_specs


def is_rank_zero() -> bool:
    """Return whether this process is the communicator root."""
    return RANK == 0


def root_print(message: str) -> None:
    """Print only from communicator root to avoid duplicate workflow logs."""
    if is_rank_zero():
        print(message)


def broadcast_value(value: Any) -> Any:
    """Broadcast a Python value from rank 0 to every rank."""
    return COMM.bcast(value, root=0)


def write_run_metadata(run_directory: Path, metadata: dict[str, Any]) -> None:
    """Write run metadata alongside the resolved cfg."""
    dump_yaml(metadata, run_directory / "run-metadata.yaml")


def prepare_run(
    cfg_path: Path,
    run_name: str,
    run_directory: Path,
    parameter_values: dict[str, Any],
    source: str,
) -> tuple[dict[str, Any], Path]:
    """Load and patch parameters for a concrete run written to run_directory."""
    parameters = deep_copy_parameters(load_yaml(cfg_path))
    ensure_output_sections(parameters)
    output_basename = build_output_basename(run_directory)
    parameters["output"]["outfile_name"] = str(output_basename)
    metadata = {
        "created_at": timestamp_utc(),
        "original_cfg": str(cfg_path),
        "source": source,
        "run_name": run_name,
        "run_directory": str(run_directory),
        "output_basename": str(output_basename),
        "parameter_values": parameter_values,
        "mpi": {"world_size": WORLD_SIZE, "root_rank": 0},
    }
    return {"parameters": parameters, "metadata": metadata}, run_directory


def prepare_execution_decision(checkpoint_path: Path, skip_existing: bool, fail_fast_stop: bool) -> dict[str, Any]:
    """Create a run decision on rank 0 and broadcast it to all ranks."""
    if is_rank_zero():
        if fail_fast_stop:
            decision = {"action": "stop", "reason": "fail-fast"}
        elif skip_existing and checkpoint_path.exists():
            decision = {"action": "skip", "reason": str(checkpoint_path)}
        else:
            decision = {"action": "execute", "reason": None}
    else:
        decision = None
    return broadcast_value(decision)


def execute_run(run_info: dict[str, Any], run_directory: Path, dry_run: bool) -> None:
    """Write provenance files and optionally execute the simulation."""
    parameters = run_info["parameters"]
    metadata = run_info["metadata"]
    root_print(f"Run: {metadata['run_name']}")
    root_print(f"  cfg: {metadata['original_cfg']}")
    root_print(f"  out: {metadata['output_basename']}")
    if is_rank_zero():
        run_directory.mkdir(parents=True, exist_ok=True)
        dump_yaml(parameters, run_directory / "run.cfg")
        write_run_metadata(run_directory, metadata)
    COMM.barrier()
    if dry_run:
        return
    from run_model import run_model

    run_model(parameters)


def collect_error_text(local_error_text: str | None) -> str | None:
    """Collect the first non-empty error string across all MPI ranks."""
    error_texts = COMM.allgather(local_error_text)
    for error_text in error_texts:
        if error_text:
            return error_text
    return None


def main() -> None:
    """Execute all requested runs."""
    args = parse_args()
    root_print(f"Starting batch run on {WORLD_SIZE} MPI rank(s)")
    run_specs = build_run_specifications(args)

    if not run_specs:
        raise ValueError("Please provide cfg paths or --manifest")

    failures = []
    stop_after_failure = False
    for run_spec in run_specs:
        run_info, run_directory = prepare_run(
            run_spec["cfg_path"],
            run_spec["run_name"],
            run_spec["run_directory"],
            run_spec["parameter_values"],
            run_spec["source"],
        )
        checkpoint_path = Path(run_info["parameters"]["output"]["outfile_name"] + ".h5")
        decision = prepare_execution_decision(checkpoint_path, args.skip_existing, stop_after_failure)
        if decision["action"] == "stop":
            break
        if decision["action"] == "skip":
            root_print(f"Skipping existing run {run_spec['run_name']}: {decision['reason']}")
            COMM.barrier()
            continue
        try:
            execute_run(run_info, run_directory, args.dry_run)
        except Exception as exc:  # pragma: no cover - defensive error reporting
            error_text = str(exc)
        else:
            error_text = None
        error_text = collect_error_text(error_text)
        if error_text is not None:
            failures.append((run_spec["run_name"], error_text))
            root_print(f"Run failed: {run_spec['run_name']}: {error_text}")
            stop_after_failure = args.fail_fast
            COMM.barrier()
            continue
        COMM.barrier()

    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
    main()
