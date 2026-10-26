"""Generate concrete simulation cfg files from a sweep template."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.common import (
    GENERATED_DIR,
    build_output_basename,
    build_run_directory,
    check_paths_exist,
    classify_parameter,
    deep_copy_parameters,
    derive_schedule_overrides,
    dump_yaml,
    ensure_output_sections,
    expand_parameter_grid,
    expand_parameter_specs,
    expand_tied_parameter_sets,
    load_yaml,
    merge_parameter_overrides,
    render_run_name,
    resolve_path,
    resolve_runs_root,
    set_by_dotted_path,
    timestamp_utc,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for config generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_cfg", help="Path to the workflow sweep YAML file")
    parser.add_argument(
        "--output-dir",
        help="Optional directory for generated cfgs. Defaults to workflows/generated/<sweep-name>",
    )
    return parser.parse_args()


def load_sweep_spec(path: Path) -> dict[str, Any]:
    """Load the sweep specification YAML."""
    return load_yaml(path)


def apply_overrides(parameters: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply dotted-path overrides to a parameters dictionary."""
    for dotted_path, value in overrides.items():
        set_by_dotted_path(parameters, dotted_path, value)


def build_manifest(
    sweep_name: str,
    sweep_spec: dict[str, Any],
    generated_cfg_dir: Path,
    runs_root: Path,
    generated_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the manifest describing generated configurations."""
    return {
        "sweep_name": sweep_name,
        "created_at": timestamp_utc(),
        "template_cfg": str(sweep_spec["template_cfg"]),
        "sweep_cfg": str(sweep_spec["_source_path"]),
        "generated_cfg_dir": str(generated_cfg_dir),
        "runs_root": str(runs_root),
        "parameters": sweep_spec.get("parameters", {}),
        "expanded_parameters": expand_parameter_specs(sweep_spec.get("parameters", {})),
        "parameter_kinds": sweep_spec.get("_parameter_kinds", {}),
        "tied_parameter_sets": sweep_spec.get("tied_parameter_sets", []),
        "refinement_scaling": sweep_spec.get("refinement_scaling"),
        "fixed": sweep_spec.get("fixed", {}),
        "runs": generated_runs,
    }


def get_reference_refinement_set(sweep_spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return the reference refinement set configuration, if present."""
    return sweep_spec.get("refinement_scaling")


def collect_swept_paths(sweep_spec: dict[str, Any]) -> list[str]:
    """Return every dotted path that the sweep varies or fixes."""
    paths = list(sweep_spec.get("parameters", {}).keys())
    for tied_set in sweep_spec.get("tied_parameter_sets") or []:
        paths.extend(key for key in tied_set if key not in paths)
    paths.extend(key for key in sweep_spec.get("fixed", {}) if key not in paths)
    return paths


def classify_swept_parameters(sweep_spec: dict[str, Any], template_parameters: dict[str, Any]) -> dict[str, str]:
    """Tag each varied path as an initial-condition ('ic'), coefficient ('coeff') or 'other' parameter."""
    varied = list(sweep_spec.get("parameters", {}).keys())
    for tied_set in sweep_spec.get("tied_parameter_sets") or []:
        varied.extend(key for key in tied_set if key not in varied)
    return {path: classify_parameter(path, template_parameters) for path in varied}


def validate_sweep_paths(sweep_spec: dict[str, Any], template_parameters: dict[str, Any]) -> None:
    """Reject paths missing from the template unless strict_paths is disabled.

    The output section is exempt because ensure_output_sections fills it in.
    """
    if not sweep_spec.get("strict_paths", True):
        return
    paths = [path for path in collect_swept_paths(sweep_spec) if not path.startswith("output.")]
    check_paths_exist(template_parameters, paths, f"sweep '{sweep_spec.get('sweep_name')}'")


def main() -> None:
    """Generate all concrete cfgs for a sweep specification."""
    args = parse_args()
    sweep_cfg_path = resolve_path(args.sweep_cfg)
    sweep_spec = load_sweep_spec(sweep_cfg_path)
    sweep_spec["_source_path"] = str(sweep_cfg_path)

    sweep_name = sweep_spec.get("sweep_name", sweep_cfg_path.stem)
    template_cfg_path = resolve_path(sweep_spec["template_cfg"], sweep_cfg_path.parent)
    template_parameters = load_yaml(template_cfg_path)
    runs_root = resolve_runs_root(sweep_spec.get("output_root"))
    validate_sweep_paths(sweep_spec, template_parameters)
    sweep_spec["_parameter_kinds"] = classify_swept_parameters(sweep_spec, template_parameters)

    if args.output_dir:
        generated_cfg_dir = resolve_path(args.output_dir)
    else:
        generated_cfg_dir = GENERATED_DIR / sweep_name
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    combinations = expand_parameter_grid(expand_parameter_specs(sweep_spec.get("parameters", {})))
    tied_parameter_sets = expand_tied_parameter_sets(sweep_spec.get("tied_parameter_sets"))
    fixed_overrides = sweep_spec.get("fixed", {})
    run_name_template = sweep_spec.get("run_name_template")
    refinement_scaling = get_reference_refinement_set(sweep_spec)
    generated_runs: list[dict[str, Any]] = []

    run_index = 1
    for parameter_values in combinations:
        for tied_parameter_values in tied_parameter_sets:
            merged_parameter_values = merge_parameter_overrides(parameter_values, tied_parameter_values)
            if refinement_scaling is not None:
                reference_overrides = refinement_scaling["reference_overrides"]
                refinement_keys = refinement_scaling["refinement_keys"]
                schedule_overrides = derive_schedule_overrides(
                    reference_overrides,
                    merged_parameter_values,
                    refinement_keys,
                )
                merged_parameter_values = merge_parameter_overrides(
                    merged_parameter_values,
                    schedule_overrides,
                )
            parameters = deep_copy_parameters(template_parameters)
            ensure_output_sections(parameters)
            apply_overrides(parameters, fixed_overrides)
            apply_overrides(parameters, merged_parameter_values)

            run_name = render_run_name(run_name_template, merged_parameter_values, run_index)
            run_directory = build_run_directory(runs_root, run_name)
            output_basename = build_output_basename(run_directory)
            parameters["output"]["outfile_name"] = str(output_basename)

            cfg_path = generated_cfg_dir / f"{run_name}.cfg"
            dump_yaml(parameters, cfg_path)
            generated_runs.append(
                {
                    "index": run_index,
                    "run_name": run_name,
                    "cfg_path": str(cfg_path),
                    "run_directory": str(run_directory),
                    "output_basename": str(output_basename),
                    "parameter_values": merged_parameter_values,
                    "tied_parameter_values": tied_parameter_values,
                }
            )
            run_index += 1

    manifest = build_manifest(sweep_name, sweep_spec, generated_cfg_dir, runs_root, generated_runs)
    manifest_path = generated_cfg_dir / "manifest.yaml"
    dump_yaml(manifest, manifest_path)

    print(f"Generated {len(generated_runs)} cfg file(s) in {generated_cfg_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
