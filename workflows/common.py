"""Shared helpers for workflow configuration, naming, and file layout."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import itertools
import math
import random
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
RUNS_DIR = PROJECT_ROOT / "runs"
GENERATED_DIR = WORKFLOWS_DIR / "generated"
VECTOR_VARIABLES = {"m", "v", "u", "F"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def dump_yaml(data: Any, path: str | Path) -> None:
    """Write YAML data to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_copy_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Copy nested parameter dictionaries."""
    return deepcopy(parameters)


def get_by_dotted_path(data: dict[str, Any], dotted_path: str) -> Any:
    """Get a nested value from a mapping using dotted syntax."""
    current = data
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Unknown configuration path '{dotted_path}'")
        current = current[key]
    return current


def set_by_dotted_path(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested value in a mapping using dotted syntax."""
    keys = dotted_path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def flatten_named_values(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping using underscores for naming/template use."""
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_named_values(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def format_value_for_name(value: Any) -> str:
    """Normalize a value into a filesystem-friendly slug fragment."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value:g}".replace("-", "neg-")
        return text.replace(".", "p")
    text = str(value).strip().lower().replace(" ", "-")
    text = text.replace(".", "p")
    text = text.replace("/", "-")
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def default_run_name(parameter_values: dict[str, Any]) -> str:
    """Create a deterministic run name from sweep parameter values."""
    parts = []
    for key, value in sorted(parameter_values.items()):
        name_key = key.replace(".", "_")
        parts.append(f"{name_key}-{format_value_for_name(value)}")
    return "__".join(parts)


def render_run_name(template: str | None, parameter_values: dict[str, Any], index: int) -> str:
    """Render a run name from either a template or a default pattern."""
    if not template:
        return default_run_name(parameter_values) or f"run-{index:03d}"
    flattened = {
        key.replace(".", "_"): format_value_for_name(value)
        for key, value in parameter_values.items()
    }
    flattened["index"] = index
    return template.format(**flattened)


def resolve_runs_root(path: str | None = None) -> Path:
    """Resolve the root directory for workflow run outputs."""
    if path is None:
        return RUNS_DIR
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a possibly relative path against an optional base directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (Path(base_dir) / candidate).resolve()
    return (PROJECT_ROOT / candidate).resolve()


def ensure_output_sections(parameters: dict[str, Any]) -> None:
    """Ensure required output sections exist before patching paths."""
    parameters.setdefault("output", {})
    parameters["output"].setdefault("loglevel", 0)
    parameters["output"].setdefault("output_freq", 1)
    parameters["output"].setdefault("stat_freq", 1)
    parameters["output"].setdefault("output_aux_vars", False)


def build_run_directory(runs_root: Path, run_name: str) -> Path:
    """Return the run directory path for a run name."""
    return runs_root / run_name


def build_output_basename(run_directory: Path) -> Path:
    """Return the output basename used by the core simulation code."""
    return run_directory / "sim"


def timestamp_utc() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def expand_parameter_grid(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a sweep parameter grid into concrete combinations."""
    if not parameters:
        return [{}]
    keys = list(parameters.keys())
    value_lists = [parameters[key] for key in keys]
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values, strict=True)))
    return combinations


def expand_tied_parameter_sets(tied_parameter_sets: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Expand tied parameter sets that should vary together as one sweep axis."""
    if not tied_parameter_sets:
        return [{}]
    expanded_sets = []
    for index, tied_set in enumerate(tied_parameter_sets, start=1):
        if not isinstance(tied_set, dict):
            raise ValueError(f"Tied parameter set {index} must be a mapping")
        expanded_sets.append(dict(tied_set))
    return expanded_sets


def merge_parameter_overrides(*override_groups: dict[str, Any]) -> dict[str, Any]:
    """Merge override groups and reject conflicting assignments."""
    merged: dict[str, Any] = {}
    for override_group in override_groups:
        for key, value in override_group.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"Conflicting override for '{key}': {merged[key]!r} vs {value!r}"
                )
            merged[key] = value
    return merged


def compute_refinement_scale(
    reference_overrides: dict[str, Any],
    target_overrides: dict[str, Any],
    refinement_keys: list[str],
) -> int:
    """Compute an integer refinement scale from selected dotted-path keys."""
    if not refinement_keys:
        raise ValueError("refinement_scaling.refinement_keys must not be empty")
    scales = []
    for key in refinement_keys:
        if key not in reference_overrides:
            raise ValueError(f"Reference refinement set is missing '{key}'")
        if key not in target_overrides:
            raise ValueError(f"Target refinement set is missing '{key}'")
        reference_value = reference_overrides[key]
        target_value = target_overrides[key]
        if not isinstance(reference_value, int) or not isinstance(target_value, int):
            raise ValueError(f"Refinement key '{key}' must use integer values")
        if reference_value <= 0 or target_value <= 0:
            raise ValueError(f"Refinement key '{key}' must be positive")
        if target_value % reference_value != 0:
            raise ValueError(
                f"Refinement key '{key}' does not scale by an integer ratio: "
                f"{target_value} / {reference_value}"
            )
        scales.append(target_value // reference_value)
    scale = scales[0]
    if any(candidate != scale for candidate in scales[1:]):
        raise ValueError("Refinement keys do not agree on a common integer refinement ratio")
    return scale


def derive_schedule_overrides(
    reference_overrides: dict[str, Any],
    target_overrides: dict[str, Any],
    refinement_keys: list[str],
) -> dict[str, Any]:
    """Derive dt, num_steps, and output_freq for a refinement set."""
    scale = compute_refinement_scale(reference_overrides, target_overrides, refinement_keys)
    dt_key = "timestepping.dt"
    num_steps_key = "timestepping.num_steps"
    output_freq_key = "output.output_freq"
    required_keys = [dt_key, num_steps_key, output_freq_key]
    for key in required_keys:
        if key not in reference_overrides:
            raise ValueError(f"Reference refinement set is missing '{key}'")
    reference_dt = reference_overrides[dt_key]
    reference_num_steps = reference_overrides[num_steps_key]
    reference_output_freq = reference_overrides[output_freq_key]
    if reference_dt <= 0 or reference_num_steps <= 0 or reference_output_freq <= 0:
        raise ValueError("Reference timestep and output settings must be positive")
    if reference_dt % scale != 0:
        raise ValueError(
            f"Reference dt {reference_dt!r} is not divisible by refinement scale {scale}"
        )
    return {
        dt_key: reference_dt / scale,
        num_steps_key: reference_num_steps * scale,
        output_freq_key: reference_output_freq * scale,
    }


def quantity_categories(model: Any, output_aux_vars: bool) -> dict[str, list[str]]:
    """Return the available quantity names by category for a model."""
    categories = {
        "x_vars": list(model.get_x_var_list()),
        "diagnostics": list(model.get_diagnostics_list()),
        "statistics": list(model.get_statistics_list()),
    }
    categories["aux_vars"] = list(model.get_aux_var_list()) if output_aux_vars else []
    return categories


def is_vector_quantity(name: str) -> bool:
    """Classify a quantity using the existing vector naming convention."""
    return name in VECTOR_VARIABLES


VALUE_SPEC_KINDS = ("values", "linspace", "geomspace", "random")


def _check_count(count: Any, kind: str) -> int:
    """Validate the number of samples requested by a value spec."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"'{kind}' value spec needs a positive integer count, got {count!r}")
    return count


def expand_value_spec(spec: Any) -> list[Any]:
    """Expand one sweep parameter specification into a concrete list of values.

    Supported forms:
    - a list of literal values: ``[0.05, 0.07]``
    - ``{values: [...]}``: the same as a list
    - ``{linspace: [low, high, n]}``: n evenly spaced values including both ends
    - ``{geomspace: [low, high, n]}``: n geometrically spaced values (low, high > 0)
    - ``{random: {low, high, n, seed, log}}``: n uniform samples from a seeded generator,
      sampled in log space when ``log`` is true. The seed is required so sweeps are reproducible.
    - a scalar: a single value
    """
    if isinstance(spec, list):
        return list(spec)
    if not isinstance(spec, dict):
        return [spec]
    if len(spec) != 1 or next(iter(spec)) not in VALUE_SPEC_KINDS:
        raise ValueError(
            f"Value spec {spec!r} must be a list or a mapping with exactly one of {VALUE_SPEC_KINDS}"
        )
    kind, args = next(iter(spec.items()))
    if kind == "values":
        if not isinstance(args, list):
            raise ValueError("'values' value spec must be a list")
        return list(args)
    if kind in ("linspace", "geomspace"):
        if not isinstance(args, list) or len(args) != 3:
            raise ValueError(f"'{kind}' value spec must be [low, high, n]")
        low, high, count = float(args[0]), float(args[1]), _check_count(args[2], kind)
        if count == 1:
            return [low]
        if kind == "linspace":
            return [_clean_float(low + (high - low) * i / (count - 1)) for i in range(count)]
        if low <= 0 or high <= 0:
            raise ValueError("'geomspace' value spec needs positive low and high")
        ratio = (high / low) ** (1.0 / (count - 1))
        return [_clean_float(low * ratio**i) for i in range(count)]
    # random sampling
    if not isinstance(args, dict):
        raise ValueError("'random' value spec must be a mapping with low, high, n and seed")
    missing = [key for key in ("low", "high", "n", "seed") if key not in args]
    if missing:
        raise ValueError(f"'random' value spec is missing {missing}")
    low, high = float(args["low"]), float(args["high"])
    count = _check_count(args["n"], "random")
    generator = random.Random(args["seed"])
    if args.get("log", False):
        if low <= 0 or high <= 0:
            raise ValueError("log-uniform 'random' value spec needs positive low and high")
        return [_clean_float(math.exp(generator.uniform(math.log(low), math.log(high)))) for _ in range(count)]
    return [_clean_float(generator.uniform(low, high)) for _ in range(count)]


def _clean_float(value: float) -> float:
    """Round away floating-point noise (for example 0.07500000000000001) from generated values."""
    return float(f"{value:.12g}")


def expand_parameter_specs(parameters: dict[str, Any]) -> dict[str, list[Any]]:
    """Expand every sweep parameter value spec into a list of values."""
    return {key: expand_value_spec(spec) for key, spec in parameters.items()}


def has_dotted_path(data: dict[str, Any], dotted_path: str) -> bool:
    """Return whether a dotted path already exists in a nested mapping."""
    try:
        get_by_dotted_path(data, dotted_path)
    except KeyError:
        return False
    return True


def check_paths_exist(data: dict[str, Any], dotted_paths: list[str], context: str) -> None:
    """Raise if any dotted path is missing from data, which usually indicates a typo."""
    missing = [path for path in dotted_paths if not has_dotted_path(data, path)]
    if missing:
        raise KeyError(f"{context}: unknown configuration path(s) {missing}")


def classify_parameter(dotted_path: str, parameters: dict[str, Any]) -> str:
    """Classify a dotted path as an initial-condition, coefficient, or other parameter.

    Coefficients are entries of a forcing-term section that has a ``treat_as_coeffs`` flag (for
    example ``hyperviscosity.c0``); initial-condition parameters live under ``initial-conditions``.
    """
    keys = dotted_path.split(".")
    if keys[0] == "initial-conditions" and len(keys) == 2:
        return "ic"
    section = parameters.get(keys[0])
    if (
        len(keys) == 2
        and isinstance(section, dict)
        and "treat_as_coeffs" in section
        and keys[1] != "treat_as_coeffs"
    ):
        return "coeff"
    return "other"


def is_mpi_root() -> bool:
    """Return whether this process is rank 0, without requiring mpi4py for serial use."""
    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover - serial fallback
        return True
    return MPI.COMM_WORLD.Get_rank() == 0


def root_print(message: str) -> None:
    """Print only from MPI rank 0."""
    if is_mpi_root():
        print(message, flush=True)


def load_run_directory(run_dir: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the resolved cfg, metadata, and checkpoint path of an existing run directory."""
    run_dir = resolve_path(run_dir, base_dir)
    cfg_path = run_dir / "run.cfg"
    checkpoint = run_dir / "sim.h5"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run directory {run_dir} has no run.cfg")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Run directory {run_dir} has no sim.h5")
    metadata_path = run_dir / "run-metadata.yaml"
    metadata = load_yaml(metadata_path) if metadata_path.exists() else {}
    return {
        "run_name": metadata.get("run_name", run_dir.name),
        "run_dir": run_dir,
        "cfg_path": cfg_path,
        "checkpoint": checkpoint,
        "parameters": load_yaml(cfg_path),
        "parameter_values": metadata.get("parameter_values", {}),
    }


def resolve_truth_runs(truth_spec: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    """Resolve truth runs from a manifest and/or an explicit list of run directories."""
    runs: list[dict[str, Any]] = []
    if "manifest" in truth_spec:
        manifest_path = resolve_path(truth_spec["manifest"], base_dir)
        manifest = load_yaml(manifest_path)
        for run in manifest.get("runs", []):
            runs.append(load_run_directory(run["run_directory"]))
    for entry in truth_spec.get("runs", []):
        run_dir = entry["run_dir"] if isinstance(entry, dict) else entry
        runs.append(load_run_directory(run_dir, base_dir))
    if not runs:
        raise ValueError("truth must list a manifest and/or runs")
    return runs


def build_observation_schedule(
    loss: dict[str, Any],
    start_index: int,
    num_steps: int,
    output_freq: int,
    available_outputs: int,
) -> dict[str, Any]:
    """Describe the data blocks and observations for one loss over one training window.

    The window starts at truth output index ``start_index`` and covers ``num_steps`` model steps.
    Returns a mapping with
    - ``block_start_indices``: truth output index where each block starts (and is initialized from)
    - ``block_steps``: model steps per block
    - ``obs_steps``: observation steps within a block (1..block_steps)
    - ``block_obs_indices``: truth output indices compared, per block

    Loss types:
    - ``final``: one block over the window, compared only at its end
    - ``windowed``: blocks of ``window_steps`` steps, each restarted from the truth and compared
      at its end
    - ``trajectory``: one block over the window, compared every ``obs_every`` steps
    """
    if output_freq <= 0:
        raise ValueError("output_freq must be positive")
    if start_index < 0:
        raise ValueError("window.start_index must be non-negative")
    if num_steps <= 0 or num_steps % output_freq != 0:
        raise ValueError(
            f"window.num_steps ({num_steps}) must be a positive multiple of the truth output_freq ({output_freq})"
        )
    loss_type = loss.get("type")
    if loss_type == "final":
        block_steps, nblocks, obs_steps = num_steps, 1, [num_steps]
    elif loss_type == "windowed":
        window_steps = loss.get("window_steps")
        if not isinstance(window_steps, int) or window_steps <= 0:
            raise ValueError("windowed loss needs a positive integer window_steps")
        if window_steps % output_freq != 0 or num_steps % window_steps != 0:
            raise ValueError(
                "windowed loss window_steps must be a multiple of output_freq and divide window.num_steps"
            )
        block_steps, nblocks, obs_steps = window_steps, num_steps // window_steps, [window_steps]
    elif loss_type == "trajectory":
        obs_every = loss.get("obs_every")
        if not isinstance(obs_every, int) or obs_every <= 0:
            raise ValueError("trajectory loss needs a positive integer obs_every")
        if obs_every % output_freq != 0 or num_steps % obs_every != 0:
            raise ValueError(
                "trajectory loss obs_every must be a multiple of output_freq and divide window.num_steps"
            )
        block_steps, nblocks = num_steps, 1
        obs_steps = list(range(obs_every, num_steps + 1, obs_every))
    else:
        raise ValueError(f"Unknown loss type {loss_type!r}; expected final, windowed or trajectory")

    outputs_per_block = block_steps // output_freq
    block_start_indices = [start_index + k * outputs_per_block for k in range(nblocks)]
    block_obs_indices = [
        [block_start + step // output_freq for step in obs_steps] for block_start in block_start_indices
    ]
    last_index = block_obs_indices[-1][-1]
    if last_index >= available_outputs:
        raise ValueError(
            f"Training window needs truth output index {last_index}, but the truth run only has "
            f"{available_outputs} outputs"
        )
    return {
        "type": loss_type,
        "block_start_indices": block_start_indices,
        "block_steps": block_steps,
        "obs_steps": obs_steps,
        "block_obs_indices": block_obs_indices,
    }


def loss_label(loss: dict[str, Any]) -> str:
    """Return a short, filesystem-friendly label for a loss specification."""
    if "label" in loss:
        return format_value_for_name(loss["label"])
    loss_type = loss.get("type", "loss")
    if loss_type == "windowed":
        return f"windowed-{loss.get('window_steps')}"
    if loss_type == "trajectory":
        return f"trajectory-{loss.get('obs_every')}"
    return str(loss_type)


TARGET_LOSS_COMPATIBILITY = {
    "coeffs": {"final", "windowed", "trajectory"},
    "ic-field": {"final", "trajectory"},
    "ic-params": {"final", "trajectory"},
}


def validate_target_and_losses(target: str, losses: list[dict[str, Any]], truth_mode: str) -> None:
    """Reject unsupported combinations of optimization target, loss types, and truth mode."""
    if target not in TARGET_LOSS_COMPATIBILITY:
        raise ValueError(f"Unknown target {target!r}; expected one of {sorted(TARGET_LOSS_COMPATIBILITY)}")
    if not losses:
        raise ValueError("At least one loss must be listed")
    for loss in losses:
        if loss.get("type") not in TARGET_LOSS_COMPATIBILITY[target]:
            raise ValueError(
                f"Loss type {loss.get('type')!r} is not supported for target {target!r}; windowed "
                "losses restart every window from the truth, so they cannot identify an initial condition"
            )
    if truth_mode not in ("per-run", "joint"):
        raise ValueError("truth.mode must be 'per-run' or 'joint'")
    if truth_mode == "joint" and target != "coeffs":
        raise ValueError("truth.mode 'joint' is only supported for the coeffs target")
    labels = [loss_label(loss) for loss in losses]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Loss labels must be unique, got {labels}; add a 'label' to distinguish them")


def relative_error(error: float, reference: float) -> float:
    """Return error / reference, or NaN when the reference is zero."""
    return error / reference if reference != 0 else float("nan")


def summarize_series(times: list[float], values: list[float], split_time: float | None) -> dict[str, float]:
    """Summarize an error time series: final, max, mean, and max within/after the training window."""
    finite = [(t, v) for t, v in zip(times, values, strict=True) if not math.isnan(v)]
    if not finite:
        return {"final": float("nan"), "max": float("nan"), "mean": float("nan")}
    summary = {
        "final": finite[-1][1],
        "max": max(v for _, v in finite),
        "mean": sum(v for _, v in finite) / len(finite),
    }
    if split_time is not None:
        train = [v for t, v in finite if t <= split_time]
        forecast = [v for t, v in finite if t > split_time]
        summary["train_max"] = max(train) if train else float("nan")
        summary["forecast_max"] = max(forecast) if forecast else float("nan")
    return summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat mappings as CSV, using the union of keys in first-seen order."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV written by write_csv, converting numeric fields to float where possible."""
    import csv

    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def to_builtin(value: Any) -> Any:
    """Convert numpy scalars/arrays (and nested containers of them) to YAML-safe builtins."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def find_job_dirs(targets: list[str]) -> list[Path]:
    """Expand targets into optimization job directories (those containing result.yaml).

    Each target is either a job directory or an optimization root whose subdirectories are jobs.
    """
    job_dirs: list[Path] = []
    for target in targets:
        path = resolve_path(target)
        if (path / "result.yaml").exists():
            job_dirs.append(path)
            continue
        found = sorted(candidate.parent for candidate in path.glob("*/result.yaml"))
        if not found:
            raise FileNotFoundError(f"No optimization results (result.yaml) found under {path}")
        job_dirs.extend(found)
    return job_dirs
