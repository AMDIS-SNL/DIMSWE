import json
from pathlib import Path

import pytest

from dimswe.resolved_hidden_c0 import ResolvedPilotConfiguration
from dimswe.resolved_hidden_c0_driver import resolved_hidden_c0_parameters
from dimswe.test2b_rain_case_design import (
    proposed_sustained_interval,
    validate_production_configuration,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = ROOT / "dimswe" / "configs" / "test2b_rain_active_case.json"
RUNNER = ROOT / "scripts" / "run_test2b_rain_active_truth_production.sh"


def test_selected_rain_active_configuration_is_frozen():
    document = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    result = validate_production_configuration(document)
    selected = document["selected_case"]
    assert result["status"] == "valid"
    assert selected["initial_moisture_zeta"] == -0.06
    assert selected["initial_Qc"] == 0.0
    assert selected["initial_Qr"] == 0.0
    assert selected["qprecip"] == 1.0e-4
    assert selected["gamma_r"] == 1.0e-3
    assert len(result["configuration_sha256"]) == 64


def test_initial_moisture_control_preserves_historical_default():
    assert ResolvedPilotConfiguration().initial_moisture_zeta == 0.0
    assert ResolvedPilotConfiguration(
        initial_moisture_zeta=-0.06
    ).initial_moisture_zeta == -0.06
    with pytest.raises(ValueError):
        ResolvedPilotConfiguration(initial_moisture_zeta=1.0)


def test_initial_moisture_control_reaches_production_initial_condition():
    configuration = ResolvedPilotConfiguration(initial_moisture_zeta=-0.06)
    parameters = resolved_hidden_c0_parameters(configuration)
    assert parameters["initial-conditions"]["zeta"] == -0.06
    assert parameters["threewayphysics"]["qprecip"] == 1.0e-4
    assert parameters["threewayphysics"]["gamma_r"] == 1.0e-3


def test_sustained_rain_requires_physical_duration_not_one_crossing():
    rows = []
    for step in range(12):
        rows.append(
            {
                "step": step,
                "time": 100.0 * step,
                "total_water_mass": 1.0e6,
                "rain_water_mass": float(step),
                "rain_source_mass_rate": 0.01,
                "physically_meaningful_R_fraction": 0.01,
            }
        )
    assert proposed_sustained_interval(rows[:10]) is None
    result = proposed_sustained_interval(rows[:11])
    assert result["start_step"] == 0
    assert result["certification_step"] == 10
    assert result["duration"] == 1000.0
    rows[5]["rain_source_mass_rate"] = 0.0
    assert proposed_sustained_interval(rows[:11]) is None


def test_manual_runner_uses_selected_physics_without_rain_law_override():
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        "--nx 64 --ny 64",
        "--dt 100",
        "--nsteps 160",
        "--output-stride 1",
        "--c0 0.14",
        "--s 3.2",
        "--initial-moisture-zeta -0.06",
        "--spectral-nx 128 --spectral-ny 128",
    ):
        assert token in text
    assert "--qprecip" not in text
    assert "--gamma-r" not in text
