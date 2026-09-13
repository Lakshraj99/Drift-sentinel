import numpy as np

from driftsentinel.evaluation import classification_metrics, early_warning_metrics, threshold_sweep


def test_lead_time_uses_first_valid_pre_drift_warning():
    batches = np.arange(30)
    alerts = np.isin(batches, [7, 9, 20])
    metrics = early_warning_metrics(batches, alerts, [10], warning_window=5)
    assert metrics["mean_lead_time"] == 3
    assert metrics["detection_coverage"] == 1


def test_false_alarm_batch_and_episode_rates_are_distinct():
    batches = np.arange(20)
    alerts = np.isin(batches, [1, 2, 3, 8])
    metrics = early_warning_metrics(batches, alerts, [10], warning_window=3, cooldown=1)
    assert metrics["batch_far"] == 3 / 17
    assert metrics["alert_episodes"] == 2
    assert metrics["event_far"] == 0.5


def test_missed_drift_rate():
    batches = np.arange(30)
    metrics = early_warning_metrics(batches, np.zeros(30, dtype=bool), [10, 20], warning_window=5)
    assert metrics["missed_drift_rate"] == 1


def test_threshold_sweep_has_one_row_per_threshold():
    result = threshold_sweep(np.arange(10), np.linspace(0, 1, 10), [9], 3, 1, [0.2, 0.8])
    assert result["threshold"].tolist() == [0.2, 0.8]


def test_warning_eligibility_exactly_matches_five_batch_target_horizon():
    batches = np.arange(20)
    too_early = early_warning_metrics(batches, batches == 4, [10], warning_window=5)
    boundary = early_warning_metrics(batches, batches == 5, [10], warning_window=5)
    assert too_early["detected_events"] == 0  # onset - 6
    assert too_early["missed_drift_rate"] == 1
    assert boundary["detected_events"] == 1  # onset - 5
    assert boundary["mean_lead_time"] == 5


def test_one_alert_episode_cannot_match_two_drift_events():
    batches = np.arange(20)
    metrics = early_warning_metrics(batches, batches == 8, [10, 12], warning_window=5)
    assert metrics["detected_events"] == 1
    assert metrics["detection_coverage"] == 0.5
    assert metrics["total_adaptations"] == 1
    assert metrics["true_event_related_adaptations"] == 1
    assert metrics["false_adaptations"] == 0


def test_adaptation_cost_uses_deduplicated_alert_episodes():
    batches = np.arange(20)
    alerts = np.isin(batches, [1, 2, 3, 5, 9])
    metrics = early_warning_metrics(batches, alerts, [10], warning_window=5, cooldown=1)
    assert metrics["total_adaptations"] == 3
    assert metrics["true_event_related_adaptations"] == 1
    assert metrics["false_adaptations"] == 2
    assert metrics["adaptation_precision"] == 1 / 3
    assert metrics["adaptations_per_100_batches"] == 15


def test_binary_baseline_scores_have_no_pr_auc():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], continuous_scores=False)
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["roc_auc"])
