# Reproducibility

## Environment

The final experiment was reproduced with Python 3.11 and the exact direct
dependency versions pinned in `requirements.txt`: River 0.21.2, NumPy 1.26.4,
pandas 2.3.3, scikit-learn 1.9.1, PyTorch 2.2.2, Matplotlib 3.11.2,
PyArrow 25.0.1, PyYAML 6.0.3, tqdm 4.70.1, and pytest 9.1.1. The package
metadata also declares every runtime dependency, including River.

Create an isolated environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -e .
pytest -q
python scripts/run_pipeline.py reproduce --quick
python scripts/run_pipeline.py reproduce
```

The full run must follow the quick run because the full run is the final result
producer. Each run removes stale generated checkpoints, predictions, training
logs, recovery traces, and figures before writing its own output matrix.

## Fixed experiment specification

All scientific settings are recorded in `configs/default.yaml`.

- Random seeds: 11, 22, and 33; quick mode uses seed 11 only.
- Input sequence length: 10 batches for GRU, LSTM, and Transformer.
- Logistic Regression input: the scaled current batch vector only.
- Target: one at batch `t` exactly when an onset is in `(t, t+5]`.
- Valid warning window: `[onset-5, onset)`; onset-minus-six is invalid and
  onset-minus-five is valid.
- Global chronological split: 55% train, 15% validation, 30% test. SEA uses
  60%/20%/20% so its ten events divide 6/2/2.
- Scaling: `StandardScaler` fitted to training rows only. Constant training
  features are removed.
- Split lookback: a validation or test target may use up to nine immediately
  preceding rows from an earlier split. Its final input row and target belong
  to the destination split; all context indices are at or before that target
  row, no future input enters `X`, and the scaler remains train-only. This is
  why a 119-row validation split can yield 119 sequences.
- Threshold: chosen only on validation F1 from 0.05 through 0.95 in 0.05
  increments; test labels are never used to choose it.
- Neural training: hidden size 48, one layer, dropout 0.15, batch size 32,
  AdamW learning rate 0.001, weight decay 0.0001, weighted BCE-with-logits,
  gradient clipping, at most 35 epochs, and patience 6.
- Logistic Regression: `liblinear`, `class_weight="balanced"`, C=1.0, and
  maximum 1,000 iterations.
- Alert episodes: consecutive alerts separated by at most five batches count
  as one episode.
- Recovery: alert-triggered Hoeffding-tree reset followed by warm start on only
  the latest 300 already-labelled instances. Recovery is the first five-batch
  rolling accuracy at least 95% of the five-batch pre-drift baseline, before
  the next onset.

## Deterministic SEA streams

Both streams contain event instances 3,000, 6,000, 9,000, 12,000, 15,000,
18,000, 21,000, 24,000, 27,000, and 29,500 in a 30,000-instance stream.
After the 300-instance reference and 50-instance batching, their evaluated
batch onsets are 54, 114, 174, 234, 294, 354, 414, 474, 534, and 584.
Concept variants are `0,2,1,3,0,2,1,3,0,2,1`; concept generator seeds start
at 700. The transition RNG seeds are 991 (abrupt) and 992 (gradual), and the
gradual transition width is 500 instances. These values live under
`synthetic_sea` in the configuration and are used by raw-stream replay.

## INSECTS provenance guard

`INSECTS-abrupt_balanced_norm.csv` is retrieved from the documented GitHub
mirror, not represented as an official River host. Every replay validates:

- source URL: `https://raw.githubusercontent.com/durga256/OnlineLearning_ML/master/INSECTS-abrupt_balanced_norm.csv`
- SHA-256: `e4819251b250a6fc1bf2a3798bbb7a1cbd2aff81d2ea34252a904c5266272fff`
- rows: 52,848
- features: 33
- classes: 6

A mismatch in either checksum or parsed metadata raises an error before replay.
The values agree with the expected River INSECTS abrupt-balanced metadata.

## Expected outputs

The final run creates the following committed summaries:

- `results/metrics/model_performance.csv`
- `results/metrics/early_warning_metrics.csv`
- `results/metrics/model_diagnostics.csv`
- `results/metrics/threshold_tradeoff.csv`
- `results/metrics/recovery_events.csv`
- `results/tables/class_event_distribution.csv`
- `results/tables/baseline_comparison.csv`
- `results/tables/per_dataset_results.csv`
- `results/tables/per_dataset_seed_summary.csv`
- `results/tables/macro_dataset_summary.csv`
- `results/tables/pooled_event_results.csv`
- `results/tables/adaptation_summary.csv`
- PNG and PDF figures under `results/figures/`

Checkpoints and detailed predictions are reproducible generated artifacts and
are ignored by Git. The Phase 1 batch-level files under `outputs/` are tracked
inputs; only the two regenerated SEA files differ from the original Phase 1
handoff.
