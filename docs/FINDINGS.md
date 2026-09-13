# Findings

## Final scientific result

The current-batch Logistic Regression baseline changes the interpretation but
does not overturn the corrected methodology. On the equal-dataset macro view,
GRU has higher F1 (0.1345 vs 0.1068), PR-AUC (0.1039 vs 0.0909), and warning
coverage (0.3889 vs 0), with lower batch FAR (0.4972 vs 0.6652). Temporal
history therefore added predictive and operational benefit under this protocol.

The benefit is limited rather than decisive. Pooled F1 slightly favours
Logistic Regression (0.1423 vs 0.1358), showing that current rolling statistics
already contain classification signal and that aggregation choice matters.
GRU's practical advantage is event usefulness: it matched seven of 18 pooled
event trials, while Logistic Regression matched none because its alert episodes
started outside the strict five-batch windows.

## Benefit versus adaptation cost

GRU's macro warning coverage of 0.3889 requires FAR 0.4972 and 3.64
adaptations per 100 test batches. Sixty of its 67 pooled reset episodes are
false adaptations; pooled adaptation precision is only 0.1045. Logistic
Regression makes 3.44 pooled adaptations per 100 batches with zero matched
event-related adaptations. DDM is far more conservative at 0.90 pooled resets
per 100 batches and FAR 0.0076, but covers only 0.1667 of events.

GRU recovers in 3.06 batches on the macro mean when recovery is observed, with
recovery coverage 0.5556. This cannot be described as simply better: the reset
policy is expensive and poorly targeted. Logistic Regression's six-batch
observed recovery and 0.1667 coverage arise despite no valid warning episodes,
so they reflect post-onset/unmatched alert adaptation rather than successful
early warning.

## Transformer conclusion

No implementation bug was found. Checkpoints are restored into fresh models;
metadata matches; inputs are `[batch,10,features]`; fixed-length inputs require
no padding mask; sinusoidal positions have the correct dimensions; training
receives logits in weighted BCE-with-logits; and sigmoid is applied once at
inference.

The data are severely sparse: SEA training has 30 positives and 317 negatives
(`pos_weight=10.57`), while INSECTS has 10 and 558 (`pos_weight=55.8`). All
INSECTS Transformer checkpoints select epoch 1 and median test probabilities
are about 0.029–0.042. Synthetic selected thresholds range from 0.20 to 0.60
and positive fractions from 0 to 0.975. The Transformer genuinely
underperformed in this small-event, imbalanced setting; it was not tuned after
test inspection to create a better result.

## Defensible conclusion

DriftSentinel demonstrates that temporal context can improve strict pre-onset
event coverage over a current-vector classifier, particularly with a GRU.
However, useful coverage currently comes with high false-alarm and reset cost,
and classical detectors remain much more conservative. The work supports
feasibility and a measurable temporal signal, not universal superiority or
deployment readiness without further calibration and more independent events.
