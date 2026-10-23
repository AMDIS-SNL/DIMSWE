"""Shared helpers for workflow configuration, naming, and file layout."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import itertools
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
