"""Pure contracts and summaries for the Test2B rain-active truth preparation.

The production driver remains :mod:`dimswe.resolved_hidden_c0_driver`.  This
module deliberately imports neither Firedrake nor JAX so configuration and
summary tests cannot advance a DIMSWE state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = 1
RAIN_THRESHOLD = 1.0e-4
CANONICAL_FLOAT64_MULTIPLIER = 64.0
CANONICAL_PHYSICAL_INCREMENT_THRESHOLD = 1.0e-12


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RainActivityContract:
    """Existing Test2A onset tolerances plus an explicitly unapproved sustain rule."""

    float64_scale_multiplier: float = CANONICAL_FLOAT64_MULTIPLIER
    physical_increment_relative_threshold: float = (
        CANONICAL_PHYSICAL_INCREMENT_THRESHOLD
    )
    sustained_duration: float = 1000.0
    minimum_consecutive_saved_states: int = 10
    minimum_space_time_fraction: float = 1.0e-4

    def __post_init__(self):
        _finite(
            "float64_scale_multiplier",
            self.float64_scale_multiplier,
            positive=True,
        )
        _finite(
            "physical_increment_relative_threshold",
            self.physical_increment_relative_threshold,
            positive=True,
        )
        _finite("sustained_duration", self.sustained_duration, positive=True)
        if self.minimum_consecutive_saved_states < 1:
            raise ValueError("minimum_consecutive_saved_states must be positive")
        fraction = _finite(
            "minimum_space_time_fraction",
            self.minimum_space_time_fraction,
            positive=True,
        )
        if fraction > 1.0:
            raise ValueError("minimum_space_time_fraction must not exceed one")


def analytical_rain_rate(qc, *, gamma_r: float, qprecip: float, dt: float):
    """Return the exact repository rain law from specific cloud water ``qc``."""
    cloud = np.asarray(qc, dtype=np.float64)
    if not np.all(np.isfinite(cloud)):
        raise ValueError("qc must be finite")
    gamma = _finite("gamma_r", gamma_r, positive=True)
    threshold = _finite("qprecip", qprecip, positive=True)
    step = _finite("dt", dt, positive=True)
    return np.maximum(0.0, gamma * (cloud - threshold) / step)


def source_invariant_residuals(source: Mapping[str, Any], beta2: float):
    """Diagnose, but do not impose, the analytical moist-source identities."""
    values = {
        name: np.asarray(source[name], dtype=np.float64)
        for name in ("S", "Qv", "Qc", "Qr")
    }
    shape = values["S"].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("source arrays must share one shape")
    water = values["Qv"] + values["Qc"] + values["Qr"]
    thermo = values["S"] - _finite("beta2", beta2) * values["Qv"]
    return {
        "water_maximum_absolute": float(np.max(np.abs(water))),
        "water_rms": float(np.sqrt(np.mean(water * water))),
        "S_minus_beta2_Qv_maximum_absolute": float(
            np.max(np.abs(thermo))
        ),
        "S_minus_beta2_Qv_rms": float(np.sqrt(np.mean(thermo * thermo))),
    }


def summarize_activity_records(records, *, qprecip=RAIN_THRESHOLD):
    """Summarize ordered per-time diagnostics without inventing rain activity."""
    rows = tuple(records)
    if not rows:
        raise ValueError("rain history must be nonempty")
    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    times = np.asarray([row["time"] for row in rows], dtype=np.float64)
    if np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("rain history must be strictly ordered")
    qc_max = np.asarray([row["specific_Qc_maximum"] for row in rows])
    r_max = np.asarray([row["R_maximum_absolute"] for row in rows])
    r_rms = np.asarray([row["R_rms"] for row in rows])
    water = np.asarray([row["total_water_mass"] for row in rows])
    rain_mass = np.asarray([row["rain_water_mass"] for row in rows])
    rain_source_mass_rate = np.asarray(
        [row.get("rain_source_mass_rate", 0.0) for row in rows],
        dtype=np.float64,
    )
    exact = np.flatnonzero(r_max > 0.0)
    meaningful = np.flatnonzero(
        [row["physically_meaningful_R_fraction"] > 0.0 for row in rows]
    )
    maximum_index = int(np.argmax(qc_max))
    water_drift = water - water[0]
    return {
        "number_of_records": len(rows),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first_time": float(times[0]),
        "last_time": float(times[-1]),
        "rain_threshold_specific_Qc": float(qprecip),
        "maximum_specific_Qc": float(qc_max[maximum_index]),
        "maximum_specific_Qc_step": int(steps[maximum_index]),
        "maximum_specific_Qc_time": float(times[maximum_index]),
        "threshold_fraction_reached": float(qc_max[maximum_index] / qprecip),
        "threshold_margin": float(qprecip - qc_max[maximum_index]),
        "maximum_R": float(np.max(r_max)),
        "maximum_R_rms": float(np.max(r_rms)),
        "first_exact_nonzero_R_step": (
            None if exact.size == 0 else int(steps[exact[0]])
        ),
        "first_physically_meaningful_R_step": (
            None if meaningful.size == 0 else int(steps[meaningful[0]])
        ),
        "maximum_rain_water_mass": float(np.max(rain_mass)),
        "maximum_rain_source_mass_rate": float(
            np.max(rain_source_mass_rate)
        ),
        "time_integrated_rain_source_mass": float(
            sum(
                float(row.get("rain_source_mass_increment", 0.0))
                for row in rows
                if row.get("applied_to_saved_trajectory", False)
            )
        ),
        "initial_total_water_mass": float(water[0]),
        "maximum_absolute_total_water_drift": float(
            np.max(np.abs(water_drift))
        ),
        "relative_maximum_total_water_drift": float(
            np.max(np.abs(water_drift)) / max(abs(water[0]), np.finfo(float).tiny)
        ),
        "sustained_rain_active": None,
        "sustained_rain_status": (
            "not_classified: candidate sustained criterion requires explicit "
            "scientific approval"
        ),
    }


def validate_configuration(document: Mapping[str, Any]):
    """Validate the frozen physical-refinement preparation contract."""
    if int(document.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported Test2B truth configuration format")
    identity = document["scientific_identity"]
    candidate = document["production_candidate"]
    if identity["case"] != "doublevortex":
        raise ValueError("Test2B truth must retain the double-vortex case")
    if tuple(identity["domain_m"]) != (5.0e6, 5.0e6):
        raise ValueError("Test2B physical domain changed")
    expected = {
        "nx": 64,
        "ny": 64,
        "dt": 100.0,
        "output_stride": 1,
        "c0": 0.14,
        "s": 3.2,
        "moist_backend": "ufl",
        "seed": 0,
        "spectral_nx": 128,
        "spectral_ny": 128,
    }
    changed = {
        key: (candidate.get(key), value)
        for key, value in expected.items()
        if candidate.get(key) != value
    }
    if changed:
        raise ValueError(f"Test2B refinement contract changed: {changed}")
    if candidate["first_segment_nsteps"] * candidate["dt"] != candidate[
        "first_segment_final_time"
    ]:
        raise ValueError("first segment time grid is inconsistent")
    activity = document["rain_activity"]
    if activity["float64_scale_multiplier"] != CANONICAL_FLOAT64_MULTIPLIER:
        raise ValueError("float64 rain tolerance changed")
    if (
        activity["physical_increment_relative_threshold"]
        != CANONICAL_PHYSICAL_INCREMENT_THRESHOLD
    ):
        raise ValueError("physical rain tolerance changed")
    return {
        "status": "valid",
        "configuration_sha256": canonical_json_sha256(document),
        "sustained_rain_criterion_approved": False,
    }


def load_configuration(path: str | Path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return document, validate_configuration(document)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-configuration",))
    parser.add_argument("--configuration", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    _, result = load_configuration(arguments.configuration)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "RainActivityContract",
    "analytical_rain_rate",
    "canonical_json_sha256",
    "load_configuration",
    "source_invariant_residuals",
    "summarize_activity_records",
    "validate_configuration",
)
