# Methodology audit

## Horizon alignment

The former target used five future batches while event evaluation accepted ten.
This was inconsistent. All target, lead-time, coverage, false-alarm, missed-
event, and threshold-sweep calls now receive the single configured
`lead_horizon=5`. Regression tests verify `onset-6` is invalid and `onset-5` is
valid. The largest lead in final result and threshold files is five.

## Split viability

Both SEA streams were regenerated with ten deterministic episodes. Each has
30/10/10 positive sequences and 6/2/2 events in chronological
train/validation/test. INSECTS has 10/5/10 positives and 2/1/2 events. Elec2
has zero throughout; it is marked `evaluation_valid=false`, and no neural model
is trained or ranked for it.

## Transformer checklist

- Training class counts and `pos_weight` are saved in `model_diagnostics.csv`.
- Validation/test probability quantiles and positive fractions are saved.
- Every checkpoint is loaded into a fresh instance; metadata is verified.
- Training curves contain logits-based weighted BCE and validation loss.
- Tensor dimensions are recorded; all sequences are fixed length, so no mask is
  required.
- Sinusoidal encoding broadcasts `[1, sequence, hidden]` over the batch.
- The classifier emits logits; sigmoid appears only in prediction.

No genuine architectural or evaluation bug was identified. INSECTS probability
collapse and early stopping at epoch 1, plus large seed-to-seed threshold
variation on SEA, explain the weak result.

## Baseline areas

The Phase 1 artifacts contain binary alarm flags only. Previous code passed
those flags to average precision, producing a mathematically calculable but
misleading “PR-AUC”. Baseline ROC-AUC and PR-AUC are now N/A. Their point
precision, recall, F1, lead, FAR, and missed-event metrics remain.

## Variance

The old aggregate standard deviation mixed datasets and seeds. It was removed.
The replacement `per_dataset_seed_summary.csv` computes sample standard
deviation over seeds within one dataset/model. Macro and pooled summaries are
separate and carry no seed-uncertainty notation.

## Recovery

The former confidence proxy was not retraining recovery and has been removed.
The new raw replay performs actual classifier reset/warm-start adaptation with
past-only data and evaluates rolling batch accuracy. Recovery is unavailable
for Elec2 because no event onsets exist, not because its raw River stream is
unavailable.

## Integrity checks

No future feature, detector flag, target, onset, or distance column enters the
models. Scaling remains train-only, splits remain chronological, thresholds
remain validation-only, alert episodes cannot match two events, and no result
was manually changed.

## Final baseline and adaptation-cost extension

Current-batch Logistic Regression now follows the same chronological,
train-only preprocessing and validation-only threshold protocol. It provides a
direct control for whether the preceding nine batches add information.
Adaptation counts derive from the same deduplicated one-to-one alert matcher:
one episode is one reset, and unmatched episodes are false adaptations. A
split-boundary regression test verifies that cross-boundary lookback is
strictly historical and never changes scaler fitting or target ownership.
