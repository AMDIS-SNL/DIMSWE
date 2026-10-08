import numpy as np

from dimswe.test2a_problem_b_signed_water_budget import (
    _classification,
    summarize_source,
    summarize_state_drift,
)


def test_signed_state_drift_separates_truth_floor_and_model_bias():
    steps = np.arange(4)
    times = 100.0 * steps
    truth = np.asarray((1000.0, 1000.25, 999.75, 1000.0))
    model = np.asarray((1000.0, 1002.0, 997.0, 1004.0))
    record = summarize_state_drift(model, truth, steps, times, 1000.0)
    assert record["final_signed_truth_relative_drift"] == 4.0
    assert record["final_relative_signed_truth_relative_drift"] == 0.004
    assert record["maximum_positive_truth_relative_drift"] == {
        "value": 4.0,
        "step": 3,
        "time": 300.0,
    }
    assert record["most_negative_truth_relative_drift"] == {
        "value": -2.75,
        "step": 2,
        "time": 200.0,
    }


def test_signed_source_summary_does_not_replace_signed_sum_with_rms():
    record = summarize_source(
        np.asarray((2.0, -3.0, 4.0)),
        np.asarray((0, 1, 2)),
        np.asarray((0.0, 100.0, 200.0)),
        100.0,
    )
    assert record["final_applied_source_integral"] == 4.0
    assert record["time_integrated_signed_source_defect"] == 300.0
    assert record["time_integrated_absolute_source_defect"] == 900.0
    assert record["maximum_positive_applied_source_integral"]["step"] == 2
    assert record["most_negative_applied_source_integral"]["step"] == 1

    one_sided = summarize_source(
        np.asarray((2.0, 3.0)),
        np.asarray((0, 1)),
        np.asarray((0.0, 100.0)),
        100.0,
    )
    assert one_sided["most_negative_applied_source_integral"] is None


def test_creation_destruction_and_rain_classification_are_separate():
    assert _classification(4.0, 1.0, 0.0) == {
        "water_budget": "artificial_net_water_creation",
        "rain_partition": "Qr_partition_at_numerical_floor",
    }
    assert _classification(-4.0, 1.0, 5.0) == {
        "water_budget": "artificial_net_water_destruction",
        "rain_partition": "spurious_Qr_partition_present",
    }
