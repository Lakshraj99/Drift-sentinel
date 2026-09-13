# Phase 1 audit

Audit date: 2026-09-12. Repository commit inspected: `8e45c38`.

## Repository and branch inventory

The remote contains only `main` (and `origin/HEAD -> origin/main`) and no tags.
The sole commit contains the Phase 1 implementation, so `main` is the most
complete Phase 1 branch. Phase 2 and Phase 3 are developed on
`laksh/phase2-phase3`; no Phase 1 artifact is overwritten.

## Available datasets and generated files

| File | Rows (batches) | Role | Exact drift truth |
|---|---:|---|---|
| `outputs/synth_sea_abrupt.csv` | 594 | controlled recurring abrupt SEA concepts | yes, 10 onsets |
| `outputs/synth_sea_gradual.csv` | 594 | controlled recurring gradual SEA concepts | yes, 10 onsets |
| `outputs/insects_abrupt_balanced.csv` | 1,050 | real INSECTS abrupt-balanced stream | published change points are supplied in Phase 2 configuration |
| `outputs/elec2.csv` | 900 | real Electricity stream | no authoritative event onsets |

An obsolete root-level duplicate of the earlier SEA file was removed during
finalization. Phase 2 consumes only the four explicitly configured files under
`outputs/`.

## Dataset-plan mismatch and final benchmark choice

The original proposal named **Electricity, Airlines, Forest CoverType, and
USENET**. The checked-in Phase 1 repository instead contained **Elec2, INSECTS
abrupt-balanced, and one SEA stream**. Airlines, CoverType, and USENET were not
present on any remote branch. The corrected benchmark therefore uses Elec2,
INSECTS abrupt-balanced, recurring abrupt SEA, and recurring gradual SEA. These
are reported under their real names; they are not claimed to be the proposal's
original four datasets.

The two SEA files were regenerated because the earlier single/two-event streams
left chronological splits without positive examples. Both now contain ten
deterministic change centers at batches 54, 114, 174, 234, 294, 354, 414, 474,
534, and 584. With the configured 60/20/20 split, six events fall in training,
two in validation, and two in test.

## Preprocessing and rolling features

`build_dataset.py` performs online predict-then-learn with River's
Hoeffding-tree classifier. `RollingStatTracker` first accumulates 300 reference
instances and subsequently emits one row per 50 instances. After every batch,
the reference advances and retains its most recent 300 instances.

For each numeric source feature the output contains:

- current mean minus reference mean;
- current variance minus reference variance; and
- Population Stability Index (PSI), using reference quantiles as bin edges.

Every row also contains mean classifier confidence, the within-batch linear
confidence slope, batch size, and a monotone batch identifier.

## Labels and baseline detectors

For controlled SEA, `ground_truth_drift_label=1` on the five batches immediately
preceding each configured onset. This column is therefore an *advance-warning
label*, not an indicator that the current batch is in drift.
No positive ground-truth labels were written for Electricity or INSECTS.

ADWIN, DDM, and KSWIN originally received instance-level classifier correctness
and their alarms were collapsed to batch flags (`adwin_alarm`, `ddm_alarm`,
`kswin_alarm`). DDM is defined over the error indicator, so this was a genuine
direction bug. This branch changes the shared detector input to `1-correct` and
regenerates the outputs. These reactive flags remain excluded from neural model
inputs. River's detectors are instantiated with defaults.

The historical River INSECTS URL returned HTTP 404 during regeneration. To
avoid changing teammate feature values, only detector flags were replayed from
the same 52,848-row abrupt-balanced sequence in the public
`durga256/OnlineLearning_ML` mirror; the original rolling-feature cells were
preserved. Elec2 and both SEA variants were replayed directly from River.

### INSECTS provenance verification

- Mirror URL: `https://raw.githubusercontent.com/durga256/OnlineLearning_ML/master/INSECTS-abrupt_balanced_norm.csv`
- SHA-256: `e4819251b250a6fc1bf2a3798bbb7a1cbd2aff81d2ea34252a904c5266272fff`
- Observed: 52,848 rows, 33 numeric features, 6 classes (8,808 rows each)
- Expected River metadata: 52,848 samples, 33 features, 6 classes, variant
  `abrupt_balanced`

The dimensions and class cardinality match River's documented metadata. The
original USP endpoint redirects to a missing resource, so the mirror plus fixed
checksum is the reproducible source used for detector and recovery replay.

## Dataset contract

Each CSV has one row per batch and the following logical schema:

- feature columns: `*_mean_shift`, `*_var_shift`, `*_psi`,
  `mean_confidence`, `confidence_trend`, and `batch_size`;
- ordering metadata: `batch_id`;
- label: `ground_truth_drift_label`; and
- detector metadata: `adwin_alarm`, `ddm_alarm`, `kswin_alarm`.

Feature dimensionality varies by source dataset. Phase 2 therefore trains one
model per dataset rather than combining incompatible tensors.

## Known issues and Phase 2 handling

1. The checked-in data do not cover the proposal's four named datasets.
2. Only controlled SEA has exact ground truth. The ten recurring events improve
   split validity but are generated from one family and are not ten independent
   real-world datasets.
3. Electricity has no event annotations. It is retained for loading and
   classification diagnostics, but event metrics are marked as unavailable
   unless an explicitly configured surrogate-event policy is enabled.
4. INSECTS event positions are not stored in its CSV. Phase 2 keeps the CSV
   immutable and supplies documented batch onsets through configuration.
5. `batch_size` is constant and is removed by train-only preprocessing.
6. PSI can be high when current observations fall outside reference-derived
   edges, and rolling references can mask very slow drift.
7. The confidence signal is produced by an evolving online classifier, so it
   describes both stream change and classifier adaptation.
8. Phase 1 stores no per-batch classifier accuracy. Phase 3 therefore replays
   the verified raw streams and explicitly simulates alert-triggered reset and
   warm-start adaptation; it does not infer recovery from confidence.

The Phase 2 loader validates ordering, duplicates, finite values, dtypes, and
constant features; fits transformations on training data only; excludes all
labels and detector flags from inputs; and never randomly shuffles temporal
splits.
