from pathlib import Path

import numpy as np
import pytest

from dimswe.test2a_horizon_curriculum import (
    TARGET_STEPS,
    load_curriculum_configuration,
    production_windows,
    validate_complete_stage_result,
    validate_stage_progress_record,
)
from dimswe.test2a_trajectory import GlobalMixedMassMetric


CONFIGURATION = "dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json"


class _VectorContext:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def axpy(self, coefficient, other):
        self.owner.values += float(coefficient) * other.owner.values

    def scale(self, coefficient):
        self.owner.values *= float(coefficient)


class _Dat:
    def __init__(self, values):
        self.values = values
        self.vec = _VectorContext(self)
        self.vec_ro = _VectorContext(self)


class _FakeState:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64).copy()
        self.dat = _Dat(self.values)

    def copy(self, deepcopy=True):
        assert deepcopy
        return _FakeState(self.values)

    def rename(self, name):
        self.name = str(name)


class _FakeHelper:
    def state_mass_map(self, value, name):
        del name
        return _FakeState(value.values)

    def dual_pairing(self, dual, primal):
        return float(np.dot(dual.values, primal.values))


@pytest.mark.parametrize(
    ("horizon", "expected_starts"),
    (
        (1, tuple(range(80))),
        (2, tuple(range(0, 80, 2))),
        (5, tuple(range(0, 80, 5))),
    ),
)
def test_nonoverlapping_schedules_cover_every_target_once(horizon, expected_starts):
    windows = production_windows(horizon)
    assert tuple(window.start_step for window in windows) == expected_starts
    targets = tuple(step for window in windows for step in window.target_steps)
    assert targets == TARGET_STEPS
    assert len(set(targets)) == 80
    assert all(window.loss_mode.value == "accumulated" for window in windows)
    assert all(window.weights == (1.0,) * horizon for window in windows)


def test_selected_configuration_freezes_common_loss_and_new_optimizer_history():
    record = load_curriculum_configuration(CONFIGURATION)
    assert record["loss"]["target_normalization"] == "none"
    assert record["loss"]["common_denominator_identical_across_horizons"]
    assert record["optimizer"]["maximum_secant_storage"] == 20
    assert record["optimizer"]["new_process_and_empty_secant_history_each_stage"]
    assert not record["optimizer"]["parameter_checkpoint_resume_restores_secant_history"]
    assert record["truth"]["state_indices"] == [0, 80]
    assert record["truth"]["states_after_80_forbidden"]


def test_global_metric_has_no_hidden_half_or_per_target_normalization():
    helper = _FakeHelper()
    metric = GlobalMixedMassMetric(helper, 10.0, denominator_sha256="a" * 64)
    state = _FakeState([3.0, 5.0])
    target = _FakeState([1.0, 2.0])
    value, derivative = metric.value_and_dual(
        state, target, target_step=17, weight=1.0, name="fake"
    )
    residual = np.array([2.0, 3.0])
    assert value == pytest.approx(np.dot(residual, residual) / 10.0)
    np.testing.assert_allclose(derivative.values, 2.0 * residual / 10.0)
    assert metric.record()["target_normalization"].startswith("one common")


def test_global_metric_rejects_nonpositive_denominator_and_bad_fingerprint():
    with pytest.raises(ValueError, match="positive"):
        GlobalMixedMassMetric(_FakeHelper(), 0.0, denominator_sha256="a" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        GlobalMixedMassMetric(_FakeHelper(), 1.0, denominator_sha256="short")


def test_production_outputs_are_new_and_do_not_alias_historical_roots():
    record = load_curriculum_configuration(CONFIGURATION)
    root = Path(record["output_root"])
    assert root.parts[-1] == "horizon-curriculum-h1-h2-h5"
    assert "fair-longfit" not in root.parts
    assert "m1-to-m2-finetune" not in root.parts


def test_parameter_only_resume_rejects_incompatible_progress():
    record = {
        "status": "in_progress",
        "configuration_sha256": "c" * 64,
        "horizon": 2,
        "h1_cache_npz_sha256": "h" * 64,
        "last_checkpoint_accepted_iteration": 17,
    }
    assert validate_stage_progress_record(record, "c" * 64, 2, "h" * 64) == 17
    with pytest.raises(ValueError, match="configuration"):
        validate_stage_progress_record(record, "x" * 64, 2, "h" * 64)
    with pytest.raises(ValueError, match="stage"):
        validate_stage_progress_record(record, "c" * 64, 5, "h" * 64)
    with pytest.raises(ValueError, match="cache"):
        validate_stage_progress_record(record, "c" * 64, 2, "x" * 64)


def test_stage_transition_requires_complete_fingerprinted_fresh_optimizer():
    source_sha = "s" * 64
    record = {
        "status": "complete",
        "configuration_sha256": "c" * 64,
        "horizon": 1,
        "h1_cache_npz_sha256": "h" * 64,
        "initialization": {
            "source_parameter_pytree_sha256": source_sha,
            "new_optimizer_process": True,
            "source_optimizer_secant_history_reused": False,
        },
        "final_parameter_file": "/tmp/final_parameters.npz",
        "final_parameter_pytree_sha256": "f" * 64,
    }
    assert validate_complete_stage_result(
        record, "c" * 64, 1, "h" * 64, source_sha
    ) == ("/tmp/final_parameters.npz", "f" * 64)
    record["initialization"]["source_optimizer_secant_history_reused"] = True
    with pytest.raises(ValueError, match="history"):
        validate_complete_stage_result(
            record, "c" * 64, 1, "h" * 64, source_sha
        )
