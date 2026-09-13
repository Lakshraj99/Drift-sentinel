# Viva Questions and Answers

1. **What is concept drift?**  A change over time in the relationship the model
   is trying to learn, commonly expressed as a change in `P(y|x)`.

2. **Data drift vs concept drift?**  Data drift changes the input distribution
   `P(x)`; concept drift changes `P(y|x)`. Data drift can occur without harming
   predictions, while concept drift can invalidate the decision rule.

3. **Why ADWIN?**  It adaptively compares subwindows and detects statistically
   significant mean changes without requiring a fixed window length.

4. **Why DDM?**  It monitors online prediction error and its standard deviation,
   making it a natural reactive performance-drift baseline.

5. **Why KSWIN?**  It uses a Kolmogorov–Smirnov comparison between recent and
   older samples, adding a non-parametric distribution-shift baseline.

6. **What is PSI?**  Population Stability Index summarizes how much a binned
   feature distribution differs from a reference distribution.

7. **Why GRU?**  A GRU models ordered history with fewer gates and parameters
   than an LSTM, which is attractive for small streaming datasets.

8. **Why LSTM?**  It is a standard gated recurrent ablation with explicit cell
   state, useful for testing whether a different memory mechanism helps.

9. **Why Transformer?**  Self-attention can model interactions across all ten
   recent batches without recurrence, testing a different temporal inductive
   bias.

10. **Why Logistic Regression?**  It is a transparent non-sequential baseline.
    Comparing it with GRU shows whether history adds value beyond current
    rolling statistics.

11. **Why sequence length 10?**  It gives the temporal models twice the
    five-batch forecast horizon of recent context while remaining compact. It
    is a fixed design choice, not selected on test performance.

12. **Why horizon 5?**  Five batches is the declared operational lead horizon
    from Phase 1 and provides a small actionable warning interval.

13. **How is the target constructed?**  `target(t)=1` exactly when any recorded
    onset is in `(t,t+5]`; the onset batch itself is negative.

14. **How do you avoid leakage?**  Inputs exclude labels and detector flags,
    splits are chronological, scaling fits train rows only, sequence context is
    past-only, and threshold selection uses validation only.

15. **Why chronological splitting?**  Random splits would expose training to
    future stream regimes and overstate deployment performance.

16. **What is weighted BCE?**  Binary cross-entropy that increases the loss
    contribution of rare positive examples during neural training.

17. **What is `pos_weight`?**  Here it is training negatives divided by training
    positives. It is calculated from the training split only.

18. **Why class imbalance?**  Only five batches before each event are positive;
    most normal stream batches are negative.

19. **What is PR-AUC?**  Area under the precision-recall curve over a continuous
    score. It is informative when positives are rare.

20. **Why is baseline PR-AUC N/A?**  Phase 1 retained only binary detector alarm
    flags. One operating point is not a comparable probability-ranking curve.

21. **What is warning lead time?**  Onset batch minus the first matched alert
    episode in the valid five-batch window.

22. **What is warning coverage?**  The fraction of true drift events with one
    matched valid pre-onset alert episode.

23. **What is FAR?**  Batch FAR is alerts outside all valid warning windows
    divided by eligible batches. Event FAR is unmatched alert episodes divided
    by all alert episodes.

24. **What is missed drift rate?**  One minus warning coverage.

25. **How does adaptation simulation work?**  Raw streams are replayed
    predict-then-learn. Each alert episode resets the Hoeffding tree and trains
    the replacement on the latest already-labelled instances only.

26. **Why 300 warm-start instances?**  It matches the Phase 1 reference size
    and gives a fixed, predeclared amount of recent past evidence to every
    policy.

27. **Why is adaptation frequency important?**  Frequent resets consume compute
    and may erase useful knowledge; they can also make measured recovery look
    fast even when alerts are poorly targeted.

28. **Why is Elec2 excluded?**  It has no authoritative event onsets here and
    therefore zero positive future targets. Training or event scoring it as a
    meaningful drift predictor would be invalid.

29. **Why synthetic SEA?**  It supplies exact, reproducible onsets and enough
    events in every chronological split to test event methodology.

30. **Why INSECTS?**  It provides a real recurring-drift stream with published
    change positions. The replay file is guarded by checksum and metadata.

31. **Why did Transformer underperform?**  No code bug was found. Positive
    supervision is sparse, imbalance is severe, and the compact model's
    thresholds and probabilities were unstable across seeds.

32. **Does GRU really beat simpler ML?**  The answer must use the final
    Logistic Regression comparison. A small F1 difference alone is not enough;
    coverage, FAR, adaptation cost, and seed variability must agree.

33. **Can DriftSentinel replace ADWIN/DDM?**  No. It is a predictive prototype
    that could complement reactive detectors; it has not established robust
    superiority or production reliability.

34. **What are the main limitations?**  Few independent labelled events,
    correlated SEA episodes, no Elec2 onset truth, position-mapped INSECTS
    events, only three optimization seeds, and simulated rather than deployed
    adaptation.

35. **How would you deploy this system?**  Compute the same rolling features in
    a stream processor, apply the frozen scaler and calibrated predictor,
    deduplicate alert episodes, gate retraining by cost and confidence, log
    delayed labels, monitor calibration, and retain a reactive fallback.
