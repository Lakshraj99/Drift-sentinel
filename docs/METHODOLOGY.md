# Methodology

## Batch features

For numeric feature `j`, Phase 1 compares current batch `C_t` with the most
recent reference window `R_t`. It records mean shift
`mean(C_tj) - mean(R_tj)` and the analogous variance shift. Population
Stability Index is `PSI = sum_b (c_b - r_b) log(c_b / r_b)`, where bins are
reference quantiles and proportions receive a small numerical epsilon. Mean
online-classifier confidence and its within-batch least-squares slope capture
model behaviour. The reference then advances using past observations only.

## Targets, splits, and sequences

Given onset set `D` and horizon `N`, target `y_t=1` iff some `d in D` satisfies
`0 < d-t <= N`. Onset and post-onset rows are never early-warning positives.
The code also retains the next onset and its distance for evaluation.

The configured horizon is `N=5`, and the event evaluator uses the identical
eligibility interval `[d-5, d)`. A warning at `d-5` is valid; one at `d-6` is a
false alarm. The evaluation horizon is derived from `lead_horizon`, not a
separate setting.

Each input is `[x_(t-W+1), ..., x_t]`; the target is `y_t`. Dataset-specific
rows are split chronologically before sequence partitioning. A standard scaler
is fitted only on training rows, then applied unchanged. Constant training
features, ordering metadata, labels, and ADWIN/DDM/KSWIN flags are excluded.

The first validation or test sequence may include preceding rows from the
immediately earlier split as historical context. Its last row and target are in
the destination split, and every context index is less than or equal to the
target index. Thus a 119-row destination split can have 119 sequences even when
`W=10`: the nine-row lookback is past-only, no future observation enters `X`,
and the scaler is still fitted exclusively on training rows.

## Models and training

The GRU and LSTM use a 48-dimensional hidden state and classify the final
recurrent output. The Transformer projects features to 48 dimensions, adds
fixed sinusoidal positions, uses one four-head pre-normalized encoder block with
a 96-unit feed-forward layer, and classifies the last token. Dropout is 0.15.

Training uses AdamW, batch size 32, maximum 35 epochs, global gradient norm 1,
and validation-loss early stopping with patience 6. BCE-with-logits receives
`negative_count / positive_count` as positive weight when positives exist.
Seeds initialize Python, NumPy, and PyTorch; test rows are never used to fit
weights, scaling, or early stopping.

Logistic Regression is the non-sequential supervised baseline. It receives
only `x_t`, the final scaled vector in each example, uses balanced class weights
with C=1 and `liblinear`, and follows the identical chronological split,
train-only scaler, validation-threshold, and untouched-test protocol. It
therefore isolates the incremental value of the preceding nine batches.

## Reactive baselines

ADWIN, DDM, and KSWIN consume the Phase 1 online classifier's instance-level
error indicator. Their batch flags are evaluated exactly as recorded and never used
as sequence-model inputs. An alarm at or after onset does not qualify as an
early warning.

## Metrics

For each event, valid warnings are episode starts in `[d-5, d)`. Lead time is
`d - first_valid_warning`; coverage is warned events over
all eligible events; missed-drift rate is one minus coverage.

Batch FAR is alerted eligible non-warning batches divided by eligible
non-warning batches. Consecutive alerts separated by at most the five-batch
cooldown form one episode. Event FAR is unmatched alert episodes divided by all
alert episodes. This prevents a persistent alert from being counted as many
independent false alarms. A warning episode can match at most one drift event.

Every alert episode is also one adaptation trigger. `total_adaptations` counts
these deduplicated episodes and `adaptations_per_100_batches` is 100 times that
count divided by evaluated test batches. An adaptation is event-related only
when the same one-to-one matcher assigns its alert episode to a true event in
`[d-5,d)`; every unmatched episode is a `false_adaptation`.
`adaptation_precision` is event-related adaptations divided by all adaptations.

Threshold trade-offs sweep model probabilities from 0.05 through 0.95 and
recompute lead time, both FARs, and misses without choosing a favorable point
after seeing test labels.

## Alert-triggered accuracy recovery

Recovery is computed by raw-stream replay for both deterministic SEA streams
and the checksum-verified INSECTS stream. One Hoeffding tree is maintained per
alert policy. At the end of an alert episode's batch, the classifier is reset
and warm-started on the most recent 300 labelled instances, all of which are
available at trigger time. Evaluation then continues predict-before-learn.

The pre-drift baseline is mean batch accuracy over the five batches immediately
before onset. Recovery occurs when five-batch rolling accuracy first reaches
95% of that baseline after the adaptation trigger and before the next event.
`accuracy_recovery_batches` is recovery batch minus onset; recovery coverage is
the fraction of events for which both a trigger and recovery are observed.
Elec2 can be replayed through River, but it has no authoritative onsets, so no
recovery metric is reported for it.

## Aggregation and variance

`per_dataset_seed_summary.csv` reports mean and standard deviation over the
three random seeds separately for each supervised model and dataset. The macro table
first averages seeds within each dataset, then gives an unweighted mean across
datasets without presenting that between-dataset spread as seed uncertainty.
The pooled table sums confusion/event counts; neural pooled rows treat each
seed-run as a replicated prediction experiment. These three views answer
different questions and are never combined into one `±` value.

Binary ADWIN/DDM/KSWIN alarm flags have no continuous score or threshold curve.
Their precision, recall, F1, FAR, lead time, and missed-event rate remain valid,
but ROC-AUC and PR-AUC are recorded as N/A and are not compared with neural
probability-based areas.
