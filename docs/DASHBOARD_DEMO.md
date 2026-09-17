# Dashboard Demo Guide

## Launch

From the repository root, in the same environment used for the project:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

The dashboard is offline-ready. Startup reads tracked CSV/YAML artifacts only;
it does not download a dataset, retrain a model, select a new threshold, or
recompute the final experiment.

## 3–5 minute faculty walkthrough

### 1. Frame the research question — Overview (40 seconds)

Open **Overview**. State that DriftSentinel predicts whether a recorded concept
drift onset will occur in the next five batches. Point to the scientific contract:
the target horizon and valid warning window are both exactly five batches.

Use the architecture strip to explain the flow from stream, to rolling features,
to four supervised predictors and three reactive detector baselines, and finally
to event scoring and causal adaptation replay. Mention that Elec2 is audited but
not event-scored because it has no authoritative onsets.

### 2. Make the event rule concrete — Drift Timeline (60 seconds)

Open **Drift Timeline** with SEA abrupt, GRU, seed 11. The shaded bands are the
only eligible warning intervals. Dashed lines are true onsets; teal diamonds are
successful warning episodes; red crosses are false episodes; amber triangles are
missed events.

Emphasize that alerts are collapsed into episodes and matched one-to-one. A
warning at onset−5 is valid, while onset−6 is false. Switch to DDM briefly to
show how a binary detector appears as flags rather than a probability curve.

### 3. Present the balanced result — Model Comparison (60 seconds)

Open **Model Comparison** on the macro view. Show F1, warning coverage, and batch
FAR. GRU has the strongest macro early-warning balance, but its false-alarm cost
is high. Switch to the pooled view: the F1 ordering changes, which demonstrates
why the aggregation level is explicit rather than silently blended.

Point out that ADWIN, DDM, and KSWIN have N/A PR-AUC. They expose binary alarm
flags only, so their area metrics are not presented as comparable with model
probabilities. Expand seed variation only if asked; every ± is within one dataset
over seeds 11, 22, and 33.

### 4. Show the operational trade-off — Threshold and Recovery (60 seconds)

Open **Threshold Trade-off**. Move the explored threshold and describe the
coverage/FAR/miss relationship. The cyan line remains the validation-selected
threshold; the test sweep is retrospective sensitivity analysis, not retuning.

Open **Recovery & Adaptation**. Explain that each alert episode triggers one
Hoeffding-tree reset and a warm start using only the latest 300 already-labelled
instances. Fast recovery must be read together with coverage, reset frequency,
false resets, and adaptation precision.

### 5. Close with the defensible claim — Research Findings (30 seconds)

End on **Research Findings**: temporal context provides measurable strict
pre-onset signal, especially for GRU, but does not establish universal superiority
or deployment readiness. The Transformer passed implementation checks and
genuinely underperformed under sparse, imbalanced supervision.

## Suggested responses to common questions

- **Why is Elec2 missing from event charts?** It has no verified event onsets and
  zero positive targets, so scoring it would fabricate ground truth.
- **Why is detector PR-AUC N/A?** Only binary alarm flags were recorded; there is
  no continuous score or genuine detector threshold curve.
- **Did the dashboard tune anything?** No. It visualizes committed outputs. The
  reported threshold was selected on validation data by the pipeline.
- **Does low recovery time mean the method is best?** No. Recovery is conditional
  on observed recovery and must be paired with recovery coverage and reset cost.

## Offline fallback

If a live browser is unavailable, present the committed static figures in
`results/figures/` in this order:

1. `model_comparison.png`
2. `drift_timeline_synth_sea_abrupt.png`
3. `lead_time_vs_false_alarm.png`
4. `adaptation_cost_comparison.png`

Then use `README.md` for the macro table and `docs/FINDINGS.md` for the conclusion.
All are generated from the same finalized artifacts used by the dashboard.
