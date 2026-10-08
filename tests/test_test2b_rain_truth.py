import json
from pathlib import Path

import numpy as np
import pytest

from dimswe.test2b_rain_truth import (
    RainActivityContract,
    analytical_rain_rate,
    load_configuration,
    source_invariant_residuals,
    summarize_activity_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = ROOT / "dimswe" / "configs" / "test2b_rain_active_truth.json"
RUNNER = ROOT / "scripts" / "run_test2b_rain_truth_segment1.sh"


def test_frozen_test2b_refinement_configuration():
    document, validation = load_configuration(CONFIGURATION)
    candidate = document["production_candidate"]
    assert validation["status"] == "valid"
    assert validation["sustained_rain_criterion_approved"] is False
    assert (candidate["nx"], candidate["ny"]) == (64, 64)
    assert candidate["dt"] == 100.0
    assert candidate["output_stride"] == 1
    assert candidate["c0"] == 0.14
    assert candidate["first_segment_nsteps"] == 160
    assert len(validation["configuration_sha256"]) == 64
    json.dumps(document, allow_nan=False)


def test_exact_rain_threshold_law():
    qc = np.array([0.0, 1.0e-4, 1.1e-4], dtype=np.float64)
    rain = analytical_rain_rate(
        qc, gamma_r=1.0e-3, qprecip=1.0e-4, dt=50.0
    )
    np.testing.assert_array_equal(rain[:2], 0.0)
    assert rain[2] == pytest.approx(2.0e-10, rel=2.0e-15)


def test_manual_runner_matches_frozen_time_grid():
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        "--nx 64 --ny 64",
        "--dt 100",
        "--nsteps 160",
        "--output-stride 1",
        "--c0 0.14",
        "--s 3.2",
        "--moist-backend ufl",
        "--spectral-nx 128 --spectral-ny 128",
    ):
        assert token in text


def test_structural_source_diagnostic_is_not_a_projection():
    source = {
        "S": np.array([98.0, 2.0]),
        "Qv": np.array([1.0, 1.0]),
        "Qc": np.array([-0.5, -0.5]),
        "Qr": np.array([-0.5, 0.25]),
    }
    original = {name: value.copy() for name, value in source.items()}
    result = source_invariant_residuals(source, 98.0)
    assert result["water_maximum_absolute"] == 0.75
    assert result["S_minus_beta2_Qv_maximum_absolute"] == 96.0
    for name in source:
        np.testing.assert_array_equal(source[name], original[name])


def test_rain_summary_keeps_sustained_classification_unfrozen():
    rows = []
    for step, qc, rain in ((0, 0.0, 0.0), (1, 0.5e-4, 0.0), (2, 1.1e-4, 1e-10)):
        rows.append(
            {
                "step": step,
                "time": 50.0 * step,
                "specific_Qc_maximum": qc,
                "R_maximum_absolute": rain,
                "R_rms": rain / 2.0,
                "physically_meaningful_R_fraction": float(rain > 0.0),
                "total_water_mass": 10.0 + 1.0e-14 * step,
                "rain_water_mass": float(step),
            }
        )
    summary = summarize_activity_records(rows)
    assert summary["first_exact_nonzero_R_step"] == 2
    assert summary["first_physically_meaningful_R_step"] == 2
    assert summary["sustained_rain_active"] is None
    assert "requires explicit" in summary["sustained_rain_status"]


def test_activity_contract_rejects_invalid_sustained_fraction():
    with pytest.raises(ValueError):
        RainActivityContract(minimum_space_time_fraction=1.1)
