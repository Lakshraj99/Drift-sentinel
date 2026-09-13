"""
baseline_detectors.py
----------------------
Wraps River's reactive drift detectors (ADWIN, DDM, KSWIN) so they can
be fed one instance-level correctness signal (0/1, was the online
model's prediction correct) at a time, and log the batch index at
which each one fires an alarm.

These logs become the reactive baselines that Member 3 compares
DriftSentinel's lead-time against.
"""

from river import drift


class BaselineDetectorSuite:
    def __init__(self):
        self.detectors = {
            "ADWIN": drift.ADWIN(),
            "DDM": drift.binary.DDM(),
            "KSWIN": drift.KSWIN(),
        }
        # each entry: list of batch indices where that detector fired
        self.alarms = {name: [] for name in self.detectors}

    def update(self, error: int, batch_idx: int):
        """
        error: 1 if the online model's prediction was wrong, else 0. DDM is
            defined on an error indicator, so all baselines monitor the same
            degradation-oriented stream.
        batch_idx: which batch this instance belongs to (for logging).
        """
        for name, det in self.detectors.items():
            det.update(error)
            if det.drift_detected:
                self.alarms[name].append(batch_idx)

    def alarm_batches(self):
        """Return dict[detector_name] -> sorted list of batch indices with an alarm."""
        return {k: sorted(set(v)) for k, v in self.alarms.items()}
