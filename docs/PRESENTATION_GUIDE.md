# Presentation Guide

## Suggested story

### Problem and motivation

Concept drift means the relationship between input data and the desired output
changes over time. A classifier that was once accurate can therefore become
stale. Reactive detectors such as ADWIN, DDM, and KSWIN look for statistical or
error changes after evidence accumulates. DriftSentinel asks a different
question: can recent batch-level behaviour predict an onset up to five batches
before it occurs?

### Three phases and architecture

Phase 1 replays each stream through a predict-then-learn Hoeffding tree and
extracts rolling feature shifts, variance shifts, PSI, confidence, confidence
trend, and binary detector alarms. Phase 2 constructs leakage-safe temporal
examples and trains GRU, LSTM, and Transformer predictors. A current-batch
Logistic Regression baseline tests whether sequence history adds value. Phase 3
selects thresholds on validation data, measures test classification and event
warnings, and replays the raw stream with alert-triggered adaptation.

The system flow is: raw stream → online classifier and rolling statistics →
current-vector or ten-batch predictor → alert episode → optional classifier
reset and 300-instance past-only warm start → event and recovery evaluation.

### Datasets and proposal mismatch

The original proposal listed Electricity, Airlines, Forest CoverType, and
USENET. The actual Phase 1 repository contained Elec2, INSECTS
abrupt-balanced, and SEA. The final benchmark truthfully uses Elec2, the
verified INSECTS mirror, regenerated abrupt SEA, and an added regenerated
gradual SEA. It does not claim that Airlines, CoverType, or USENET were tested.
Elec2 lacks authoritative event onsets, so it is audited but excluded from
meaningful event-labelled model evaluation.

### Target and leakage protection

At batch `t`, the target is one only if an onset lies in `(t,t+5]`. A valid
warning must therefore lie in `[onset-5,onset)`. Chronological splits preserve
time order, scaling is trained only on training rows, thresholds are selected
only on validation labels, and labels and detector flags are excluded from
inputs. Validation/test sequences may borrow preceding rows as historical
context, but never future rows; their target and final input row belong to the
destination split.

### Metrics and tradeoff

Classification metrics include precision, recall, F1, ROC-AUC, and PR-AUC.
PR-AUC is N/A for ADWIN/DDM/KSWIN because their Phase 1 output contains only
binary flags, not a genuine continuous score. Event metrics include lead time,
warning coverage, batch and episode false-alarm rates, and missed-drift rate.
Adaptation metrics count deduplicated alert episodes, resets per 100 evaluated
batches, false resets, event-related resets, and adaptation precision.

Do not present fast recovery alone as success. A method may recover quickly
because it resets often; warning coverage must be discussed together with
false alarms and adaptations per 100 batches.

### Final result to present

On the equal-dataset macro view, GRU reaches F1 0.1345, PR-AUC 0.1039,
coverage 0.3889, FAR 0.4972, and 3.64 adaptations per 100 batches. Current-batch
Logistic Regression reaches F1 0.1068, PR-AUC 0.0909, zero valid event
coverage, FAR 0.6652, and 2.36 adaptations per 100 batches. GRU therefore adds
some temporal/event benefit, but not a production-quality operating point.
Pooled F1 slightly favours Logistic Regression, 0.1423 against 0.1358, so do
not describe the deep model as universally better.

DDM covers 0.1667 of events with FAR 0.0083 and 0.88 macro adaptations per 100
batches. It is much more conservative than GRU. GRU's observed recovery is
faster—3.06 batches with 0.5556 recovery coverage—but 60 of 67 pooled resets
are false adaptations. The honest result is a coverage-versus-cost tradeoff.

### Transformer finding

The Transformer implementation passed the checkpoint, tensor shape, positional
encoding, masking, and logits/sigmoid audit. Its weak result is associated with
very sparse positive supervision and unstable operating thresholds, not an
identified implementation bug. The test protocol was not tuned to rescue it.

### Limitations and future work

There are few independent real drift events, SEA episodes share a generator
family, INSECTS positions are mapped from published instance locations, and
Elec2 cannot support event scoring. Three seeds describe optimization
variation, not statistical significance. Future work should add more
authoritatively labelled streams, calibrate alerts, test uncertainty-aware
thresholds, compare richer simple baselines, and validate delayed-label and
compute constraints in a deployed stream processor.

## 60-second explanation

DriftSentinel is an early-warning research prototype for concept drift. It
summarizes a live data stream into rolling statistics, then asks whether a drift
onset will occur within the next five batches. I compare three temporal models—
GRU, LSTM, and Transformer—with current-batch Logistic Regression and reactive
ADWIN, DDM, and KSWIN flags. The protocol is chronological: scaling uses train
data only, thresholds use validation only, and the five-batch target exactly
matches the valid warning window. I also replay the raw SEA and verified
INSECTS streams: every alert episode triggers a real Hoeffding-tree reset and a
warm start from the latest 300 labelled instances. This lets me report not only
warning coverage and recovery, but also false alarms and resets per 100
batches. The core result is a benefit-cost tradeoff, not a claim that deep
learning universally beats classical detectors.

## Two-minute explanation

Concept drift can degrade a deployed classifier because the mapping from
features to labels changes over time. Classical detectors are useful, but they
are usually reactive: they need evidence from changed data or errors. The goal
of DriftSentinel is to investigate whether recent system behaviour contains a
predictive signature before a known onset.

Phase 1 replays Elec2, INSECTS abrupt-balanced, and deterministic abrupt and
gradual SEA streams through an online Hoeffding tree. Every batch produces
rolling shift, PSI, and confidence features plus ADWIN, DDM, and KSWIN alarms.
Phase 2 creates a target that is positive exactly when drift starts in the next
five batches. GRU, LSTM, and Transformer receive ten-batch sequences; Logistic
Regression receives only the current batch vector, giving a direct test of the
value of temporal modelling. Phase 3 selects every supervised threshold on
validation data and evaluates untouched chronological test data.

The event protocol is deliberately strict. A warning is valid only from five
batches before onset up to, but not including, onset. An alert six batches
early receives no credit, and one alert episode cannot detect two events.
Binary detector flags have no comparable PR-AUC. For adaptation, the raw
streams are replayed using only labels available at that time. An alert episode
resets the online classifier and warm-starts it from the latest 300 instances;
recovery is judged against 95% of pre-drift rolling accuracy. Reset frequency
and false adaptations are reported because frequent resets can make recovery
look deceptively fast. The final conclusion should be read from the regenerated
Logistic-versus-GRU result and the coverage-versus-cost curve, under limitations
of sparse event truth and a small number of seeds.
