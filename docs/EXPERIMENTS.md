# Experiments

## Final protocol

The final run evaluates current-batch Logistic Regression and ten-batch GRU,
LSTM, and Transformer predictors with seeds 11, 22, and 33. The target and
warning horizon are both exactly five batches. Neural training uses weighted
BCE-with-logits; Logistic Regression uses balanced class weights. Every model
uses train-only scaling, validation-only threshold selection, restored
checkpoints, and untouched chronological test data.

Elec2 has no positive target or authoritative onset in any split. Its rows are
loaded and audited, but all seven methods are marked `evaluation_valid=false`
rather than treating all-negative evaluation as useful drift prediction.

## Class and event distribution

| Dataset | Split | Rows | Sequences | Positive | Negative | Events |
|---|---|---:|---:|---:|---:|---:|
| SEA abrupt | train | 356 | 347 | 30 | 317 | 6 |
| SEA abrupt | validation | 119 | 119 | 10 | 109 | 2 |
| SEA abrupt | test | 119 | 119 | 10 | 109 | 2 |
| SEA gradual | train | 356 | 347 | 30 | 317 | 6 |
| SEA gradual | validation | 119 | 119 | 10 | 109 | 2 |
| SEA gradual | test | 119 | 119 | 10 | 109 | 2 |
| INSECTS | train | 577 | 568 | 10 | 558 | 2 |
| INSECTS | validation | 158 | 158 | 5 | 153 | 1 |
| INSECTS | test | 315 | 315 | 10 | 305 | 2 |
| Elec2 | train | 495 | 486 | 0 | 486 | 0 |
| Elec2 | validation | 135 | 135 | 0 | 135 | 0 |
| Elec2 | test | 270 | 270 | 0 | 270 | 0 |

Validation and test sequences may use immediately preceding rows as past-only
lookback. Their target and final input row are in the destination split, no
future row enters the input, and the scaler is still fitted on training rows
only. This explains 119 validation rows producing 119 sequences.

## Macro dataset means

The table first averages random seeds within each dataset and then weights the
three event-labelled datasets equally. It has no `±`, because between-dataset
variation is not random-seed variation. Recovery means include recovered event
rows; recovery coverage reports how often a recovery was observed. Binary
detectors have PR-AUC N/A.

| Method | F1 | PR-AUC | Lead | Coverage | Batch FAR | Miss | Recovery | Rec. coverage | Adapt./100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | 0.1345 | 0.1039 | 1.83 | 0.3889 | 0.4972 | 0.6111 | 3.06 | 0.5556 | 3.6415 |
| Logistic Regression | 0.1068 | 0.0909 | 0.00 | 0.0000 | 0.6652 | 1.0000 | 6.00 | 0.1667 | 2.3592 |
| LSTM | 0.1222 | 0.0843 | 0.44 | 0.0556 | 0.7199 | 0.9444 | 14.50 | 0.2222 | 1.6890 |
| Transformer | 0.0797 | 0.1020 | 0.11 | 0.0556 | 0.3086 | 0.9444 | 4.75 | 0.1667 | 1.3425 |
| ADWIN | 0.0370 | N/A | 1.33 | 0.1667 | 0.0199 | 0.8333 | 20.17 | 0.6667 | 1.9670 |
| DDM | 0.0513 | N/A | 0.67 | 0.1667 | 0.0083 | 0.8333 | 31.50 | 0.6667 | 0.8777 |
| KSWIN | 0.0000 | N/A | 0.00 | 0.0000 | 0.0087 | 1.0000 | 11.00 | 0.1667 | 0.8466 |

## Within-dataset random-seed variation

Every `±` below is mean ± sample standard deviation over the three seeds for
one supervised model on one dataset. Deterministic Logistic Regression has zero
seed variation under this solver and data. Baseline detectors have one fixed
run and therefore no seed standard deviation.

| Dataset | Model | F1 | PR-AUC | Coverage | Batch FAR | Adapt./100 |
|---|---|---:|---:|---:|---:|---:|
| INSECTS | GRU | 0.0538 ± 0.0255 | 0.0468 ± 0.0190 | 0.5000 ± 0.0000 | 0.2317 ± 0.1710 | 4.7619 ± 1.1446 |
| INSECTS | Logistic Regression | 0.0000 ± 0.0000 | 0.0479 ± 0.0000 | 0.0000 ± 0.0000 | 0.0689 ± 0.0000 | 5.3968 ± 0.0000 |
| INSECTS | LSTM | 0.0593 ± 0.0517 | 0.0746 ± 0.0161 | 0.1667 ± 0.2887 | 0.1749 ± 0.1675 | 3.3862 ± 2.7000 |
| INSECTS | Transformer | 0.0000 ± 0.0000 | 0.0251 ± 0.0027 | 0.0000 ± 0.0000 | 0.0022 ± 0.0038 | 0.1058 ± 0.1833 |
| SEA abrupt | GRU | 0.2016 ± 0.0519 | 0.1444 ± 0.0451 | 0.3333 ± 0.2887 | 0.7034 ± 0.3270 | 2.5210 ± 2.2233 |
| SEA abrupt | Logistic Regression | 0.1653 ± 0.0000 | 0.1293 ± 0.0000 | 0.0000 ± 0.0000 | 0.9266 ± 0.0000 | 0.8403 ± 0.0000 |
| SEA abrupt | LSTM | 0.1563 ± 0.0021 | 0.1001 ± 0.0208 | 0.0000 ± 0.0000 | 0.9908 ± 0.0159 | 0.8403 ± 0.0000 |
| SEA abrupt | Transformer | 0.1196 ± 0.1056 | 0.1309 ± 0.0178 | 0.0000 ± 0.0000 | 0.4862 ± 0.4862 | 1.1204 ± 1.2836 |
| SEA gradual | GRU | 0.1480 ± 0.0112 | 0.1203 ± 0.0461 | 0.3333 ± 0.5774 | 0.5566 ± 0.4360 | 3.6415 ± 2.5673 |
| SEA gradual | Logistic Regression | 0.1550 ± 0.0000 | 0.0956 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8403 ± 0.0000 |
| SEA gradual | LSTM | 0.1510 ± 0.0070 | 0.0781 ± 0.0085 | 0.0000 ± 0.0000 | 0.9939 ± 0.0106 | 0.8403 ± 0.0000 |
| SEA gradual | Transformer | 0.1196 ± 0.1056 | 0.1499 ± 0.0463 | 0.1667 ± 0.2887 | 0.4373 ± 0.4876 | 2.8011 ± 2.7013 |

## Pooled result

Pooled confusion and event counts are a separate view. Supervised methods have
18 event trials because six test events are repeated across three seeds;
detectors have six. Pooled F1 weights datasets by row count and is nonlinear.

| Method | F1 | PR-AUC | Lead | Coverage | Batch FAR | Miss | Adapt./100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1423 | 0.0755 | 0.00 | 0.0000 | 0.4417 | 1.0000 | 3.4358 |
| GRU | 0.1358 | 0.0780 | 2.57 | 0.3889 | 0.3977 | 0.6111 | 4.0386 |
| LSTM | 0.1387 | 0.0832 | 4.00 | 0.0556 | 0.5156 | 0.9444 | 2.2905 |
| Transformer | 0.1502 | 0.0900 | 1.00 | 0.0556 | 0.1938 | 0.9444 | 0.9042 |
| ADWIN | 0.0476 | N/A | 4.00 | 0.1667 | 0.0210 | 0.8333 | 2.1700 |
| DDM | 0.0571 | N/A | 2.00 | 0.1667 | 0.0076 | 0.8333 | 0.9042 |
| KSWIN | 0.0000 | N/A | 0.00 | 0.0000 | 0.0153 | 1.0000 | 1.4467 |

## Adaptation cost

GRU generates 3.64 resets per 100 batches on the macro view, versus 2.36 for
Logistic Regression, 1.97 for ADWIN, and 0.88 for DDM. Only 10.4% of pooled GRU
adaptations and none of the Logistic adaptations match a valid warning event.
Logistic Regression's observed six-batch recovery is therefore not evidence of
useful early warning: its recovery comes from resets not matched to valid
pre-onset warnings. Likewise, rapid recovery must always be read with recovery
coverage and adaptation precision.

## Output inventory

All reported values are generated from code. Detailed per-run metrics live in
`results/metrics/`; dataset, seed, macro, pooled, and adaptation summaries live
in `results/tables/`; the final plots are under `results/figures/`.
