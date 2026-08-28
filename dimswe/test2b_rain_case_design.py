"""Pure postprocessing contracts for the Test2B rain-active case design.

This module reads completed bounded rain-audit JSON files.  It never imports
Firedrake, advances a state, or modifies the analytical moist laws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def proposed_sustained_interval(
    records,
    *,
    duration: float = 1000.0,
    minimum_mean_active_fraction: float = 1.0e-4,
):
    """Return the first interval satisfying the proposed sustain criterion.

    This is a documented case-design criterion, not a repository-wide
    canonical definition.  Every saved state in the interval must have a
    physically meaningful local rain rate and positive domain-integrated rain
    production.  The interval must span at least ``duration``, have the stated
    mean active fraction, and accumulate positive rain mass above a float64
    total-water floor.
    """
    rows = tuple(records)
    if len(rows) < 2:
        return None
    floor = 128.0 * np.finfo(np.float64).eps * abs(
        float(rows[0]["total_water_mass"])
    )
    for start in range(len(rows)):
        fractions = []
        for stop in range(start, len(rows)):
            row = rows[stop]
            if not (
                float(row["physically_meaningful_R_fraction"]) > 0.0
                and float(row.get("rain_source_mass_rate", 0.0)) > 0.0
            ):
                break
            fractions.append(float(row["physically_meaningful_R_fraction"]))
            elapsed = float(row["time"]) - float(rows[start]["time"])
            rain_gain = float(row["rain_water_mass"]) - float(
                rows[start]["rain_water_mass"]
            )
            if (
                elapsed >= duration
                and np.mean(fractions) >= minimum_mean_active_fraction
                and rain_gain > floor
            ):
                return {
                    "start_step": int(rows[start]["step"]),
                    "start_time": float(rows[start]["time"]),
                    "certification_step": int(row["step"]),
                    "certification_time": float(row["time"]),
                    "duration": elapsed,
                    "saved_states": int(stop - start + 1),
                    "mean_physically_meaningful_R_fraction": float(
                        np.mean(fractions)
                    ),
                    "rain_mass_gain": rain_gain,
                    "rain_mass_float64_floor": floor,
                }
    return None


def summarize_candidate(audit: Mapping[str, Any], *, zeta: float):
    if audit.get("status") != "complete":
        raise ValueError("case-design summary requires a complete rain audit")
    records = tuple(audit["records"])
    if not records:
        raise ValueError("rain audit contains no records")
    summary = dict(audit["summary"])
    threshold = float(summary["rain_threshold_specific_Qc"])
    sustained = proposed_sustained_interval(records)
    first_qc = next(
        (
            {"step": int(row["step"]), "time": float(row["time"])}
            for row in records
            if float(row["specific_Qc_maximum"]) > threshold
        ),
        None,
    )
    max_fraction = max(
        float(row["physically_meaningful_R_fraction"]) for row in records
    )
    if sustained is not None:
        classification = "RAIN_ACTIVE_SUSTAINED_UNDER_PROPOSED_CRITERION"
    elif summary["first_physically_meaningful_R_step"] is not None:
        classification = "TRANSIENT_RAIN"
    elif float(summary["threshold_fraction_reached"]) >= 0.8:
        classification = "NEAR_ONSET_DRY"
    else:
        classification = "DRY"
    rain_masses = np.asarray(
        [row["rain_water_mass"] for row in records], dtype=np.float64
    )
    water_floor = 128.0 * np.finfo(np.float64).eps * max(
        abs(float(rain_masses[-1])),
        abs(float(summary["initial_total_water_mass"])),
    )
    minimum_fields = {
        name.removeprefix("minimum_").removesuffix("_GLL"): float(
            min(row[name] for row in records)
        )
        for name in (
            "minimum_h_GLL",
            "minimum_Qv_GLL",
            "minimum_Qc_GLL",
            "minimum_Qr_GLL",
        )
    }
    maximum_fields = {
        name.removeprefix("maximum_").removesuffix("_GLL"): float(
            max(row[name] for row in records)
        )
        for name in (
            "maximum_h_GLL",
            "maximum_Qv_GLL",
            "maximum_Qc_GLL",
            "maximum_Qr_GLL",
        )
    }
    initial = records[0]
    final = records[-1]
    return {
        "initial_moisture_zeta": float(zeta),
        "initial_boundary_saturation_ratio": float(1.0 - zeta),
        "classification": classification,
        "initial_postprefix_saturation_ratio_range": [
            float(initial["saturation_ratio_minimum"]),
            float(initial["saturation_ratio_maximum"]),
        ],
        "maximum_specific_Qc": float(summary["maximum_specific_Qc"]),
        "maximum_specific_Qc_over_qprecip": float(
            summary["threshold_fraction_reached"]
        ),
        "maximum_specific_Qc_step": int(summary["maximum_specific_Qc_step"]),
        "maximum_specific_Qc_time": float(summary["maximum_specific_Qc_time"]),
        "first_specific_Qc_above_qprecip": first_qc,
        "first_exact_nonzero_R_step": summary["first_exact_nonzero_R_step"],
        "first_physically_meaningful_R_step": summary[
            "first_physically_meaningful_R_step"
        ],
        "maximum_R": float(summary["maximum_R"]),
        "maximum_spatial_R_rms": float(summary["maximum_R_rms"]),
        "space_time_R_rms": float(
            np.sqrt(np.mean([float(row["R_rms"]) ** 2 for row in records]))
        ),
        "maximum_physically_meaningful_R_fraction": max_fraction,
        "maximum_rain_water_mass": float(summary["maximum_rain_water_mass"]),
        "final_rain_water_mass": float(final["rain_water_mass"]),
        "time_integrated_rain_source_mass": float(
            summary["time_integrated_rain_source_mass"]
        ),
        "rain_water_mass_nondecreasing_with_float64_allowance": bool(
            np.all(np.diff(rain_masses) >= -water_floor)
        ),
        "proposed_sustained_interval": sustained,
        "relative_maximum_total_water_drift": float(
            summary["relative_maximum_total_water_drift"]
        ),
        "maximum_source_invariant_residuals": dict(
            audit["maximum_source_invariant_residuals"]
        ),
        "minimum_postprefix_GLL_fields": minimum_fields,
        "maximum_postprefix_GLL_fields": maximum_fields,
        "kinetic_energy_initial": float(initial["kinetic_energy"]),
        "kinetic_energy_final": float(final["kinetic_energy"]),
        "kinetic_energy_relative_change": float(
            final["kinetic_energy"] / initial["kinetic_energy"] - 1.0
        ),
        "projected_enstrophy_initial": float(initial["projected_enstrophy"]),
        "projected_enstrophy_final": float(final["projected_enstrophy"]),
        "projected_enstrophy_relative_change": float(
            final["projected_enstrophy"] / initial["projected_enstrophy"]
            - 1.0
        ),
        "all_saved_states_finite": bool(
            all(row["all_state_coefficients_finite"] for row in records)
        ),
    }


def build_design_summary(candidates, *, output: str | Path):
    rows = []
    for name, zeta, audit_path in candidates:
        path = Path(audit_path).resolve()
        audit = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "audit_path": str(path),
                "audit_sha256": _sha256(path),
                "diagnostics": summarize_candidate(audit, zeta=zeta),
            }
        )
    result = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "purpose": "bounded Test2B physical rain-active case-design sweep",
        "control": {
            "name": "initial_moisture_zeta",
            "definition": "Qv(t=0)=h*(1-zeta)*qsat; Qc(t=0)=Qr(t=0)=0",
            "physical_meaning": (
                "signed initial saturation deficit; negative is uniform "
                "fractional supersaturation"
            ),
        },
        "proposed_sustained_rain_criterion": {
            "canonical_status": "PROPOSED_FOR_THIS_CASE_DESIGN_NOT_GLOBAL_CANON",
            "minimum_continuous_duration": 1000.0,
            "every_saved_state_requires_physically_meaningful_R": True,
            "every_saved_state_requires_positive_integrated_rain_production": True,
            "minimum_mean_active_GLL_fraction": 1.0e-4,
            "requires_positive_rain_mass_gain_above_128eps_total_water": True,
        },
        "candidates": rows,
    }
    _write_json(Path(output), result)
    return result


def validate_production_configuration(document: Mapping[str, Any]):
    if int(document.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported rain-active case configuration format")
    selected = document["selected_case"]
    required = {
        "nx": 64,
        "ny": 64,
        "dt": 100.0,
        "nsteps": 160,
        "output_stride": 1,
        "c0": 0.14,
        "s": 3.2,
        "initial_moisture_zeta": -0.06,
        "qprecip": 1.0e-4,
        "gamma_r": 1.0e-3,
    }
    changed = {
        key: (selected.get(key), value)
        for key, value in required.items()
        if selected.get(key) != value
    }
    if changed:
        raise ValueError(f"rain-active production contract changed: {changed}")
    if float(selected["final_time"]) != float(selected["dt"]) * int(
        selected["nsteps"]
    ):
        raise ValueError("rain-active production time grid is inconsistent")
    if (
        selected.get("initial_boundary_saturation_ratio") != 1.06
        or selected.get("initial_Qc") != 0.0
        or selected.get("initial_Qr") != 0.0
    ):
        raise ValueError("selected initial moisture contract changed")
    if (
        selected.get("case") != "doublevortex"
        or selected.get("moist_backend") != "ufl"
        or selected.get("spectral_nx") != 128
        or selected.get("spectral_ny") != 128
    ):
        raise ValueError("selected production discretization changed")
    for law in ("analytical_A", "analytical_R"):
        if document["frozen_laws"].get(law) != "unchanged":
            raise ValueError(f"{law} was not frozen")
    return {
        "status": "valid",
        "configuration_sha256": hashlib.sha256(
            json.dumps(
                document, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME,ZETA,AUDIT_JSON",
    )
    summarize.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate-configuration")
    validate.add_argument("--configuration", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "summarize":
        candidates = []
        for value in arguments.candidate:
            name, zeta, path = value.split(",", 2)
            candidates.append((name, float(zeta), path))
        result = build_design_summary(candidates, output=arguments.output)
    else:
        document = json.loads(
            Path(arguments.configuration).read_text(encoding="utf-8")
        )
        result = validate_production_configuration(document)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "build_design_summary",
    "proposed_sustained_interval",
    "summarize_candidate",
    "validate_production_configuration",
)
