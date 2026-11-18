# Phase 3b: Pre-registered confirmation run

Status: **complete**. Script: `phase3b_holdout_confirm.py` (step 13).
Pre-registration: `docs/planning/phase3b/phase3_preregistration.md`. Output:
`results/phase3b/phase3b_holdout_confirm.{txt,json}`.

Updated 2026-10-01 with the confirmation run on the rebuilt, independent data.
Every figure below is literal output from that run, or from the descriptive
check named next to it. An earlier run on the original, smaller data used an
evaluation set that was not independent. It is superseded, and what it reported
is recorded in the pre-registration.

## Purpose

Phase 3's rank target was first chosen after seeing it win an ablation. A
p-value computed after that kind of choice overstates the evidence. Phase 3b
fixes the method in advance, writes it down, and runs **one** pre-specified
test on data the model never trained on.

## The pre-registered method

- **Model**: one `RandomForestRegressor` over (features + algorithm one-hot).
- **Target**: per-row rank. **Prediction**: the lowest predicted rank.
- **Validation**: leave-one-function-out, applied across the two files: to
  predict function F, training uses only rows of the other 23 functions.
- **Test**: one two-sided Wilcoxon signed-rank test, selector vs SBS, over the
  held-out functions, alpha = 0.05. No further model, target or
  hyperparameter search.
- **Interpretation, fixed in advance**: p < 0.05 **and** a positive gap closed
  means the Phase 3 result replicates. Anything else means it does not, and
  that is to be reported plainly.

## The data

| | Training | Evaluation |
|---|---|---|
| File | `results/phase2c/phase2c_matrix.json` | `results/phase2/phase2_matrix.json` |
| Rows | 720 = 24 functions × dims 5/10/20 × seeds 42–51 | 72 = 24 functions × dims 5/10/20, seed 100 |
| Runs per cell | 5 | 15 |

The data differs from what the pre-registration first described, so the change
was logged in its append-only section and committed (`132a7db`) **before**
this run. That entry records the rebuilt matrices, the code changes and what was
already known at the time. The method itself did not change.

Two consequences of the new data, both stated in that entry:

1. Each function has 3 evaluation rows, one per dimension. They are not
   independent, so the test averages each function's rows first and pairs the
   24 function means.
2. Training labels are medians of 5 runs.

## Result

```
  split check     : OK -- eval seed 100 not in training seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

  eval rows       : 72   algorithms: 22   seed: 100
  eval dims       : [5, 10, 20]   functions: 24
  paired test N   : 24 function means (rows aggregated first)
  seeds recorded  : RF_SEED=20260916, n_estimators=300
  scipy/sklearn   : 1.16.3 / 1.9.0

  MEAN RANK on held-out rows (1 = best of 22, lower is better)
  ------------------------------------------------------------
  VBS (oracle)              1.2014
  Selector (RF)             5.1389
  SBS (fold-aware)          5.2222
  Random (analytic)        11.5000

  SBS->VBS gap closed : +2.1%

  statistic=np.float64(88.0)  p=np.float64(0.5255810541761173)
  significant at alpha=0.05: NO
  selector better on 13, worse on 7, tied on 4
```

The headline figures were recomputed independently from the 72 per-row entries
in the JSON and matched. Every fold trained on 690 rows (720 minus the 30 rows
of the held-out function). The output records `"independent_holdout": true`.

## Verdict: the Phase 3 result does not replicate

| Pre-registered condition | Result | Met? |
|---|---|---|
| Gap closed is positive | +2.1% | yes |
| p < 0.05 | p = 0.526 | **no** |

Both conditions are required, so by the rule fixed in advance **the Phase 3
result does not replicate.** The selector's mean rank (5.1389) is 0.08 better
than SBS's (5.2222). That gap is small, and a difference this size could easily
arise by chance.

## Why it lands at the SBS level (descriptive only)

The figures below come from the per-row entries in
`results/phase3b/phase3b_holdout_confirm.json`. They describe what the selector did.
They are **not** part of the pre-registered analysis, carry no significance
test, and must not be read as evidence of an effect.

```
SBS picks: {'MPA': 69, 'EO': 3}
selector differs from SBS on 49 of 72 rows; better 28, worse 19, equal 2
mean gain when better 4.71 ranks; mean loss when worse 6.63 ranks
```

The selector does not just copy SBS. It makes its own pick on 49 of 72 rows and
beats SBS on more of them (28) than it loses (19). But a loss costs more than a
win gains (6.63 ranks against 4.71 on average), so the two nearly cancel.

By dimension:

```
  d5  selector 5.333  sbs 3.750  (24 rows)
  d10 selector 5.083  sbs 5.292  (24 rows)
  d20 selector 5.000  sbs 6.625  (24 rows)
```

Here SBS is almost always MPA. Phase 2 found that MPA is strongest at d5 and
weakens as dimension grows, which fits SBS being hard to beat at d5 and easy to
beat at d20. This is a subgroup pattern noticed after the fact. Testing it
fairly would need new data.

## What this means

On broad, independent data, the learned selector does not do better than always
running the single best solver (MPA or EO). It agrees with every other analysis
of the rebuilt data:

| Analysis | Result |
|---|---|
| Phase 2c LOPO, normalised target, 22 forests | −22.9%, p=0.356 |
| Phase 3 step 11, registered method inside the training matrix | −0.3%, p=0.677 |
| Phase 3 step 12, all 8 ablation cells | best +0.8%, none significant |
| **Phase 3b, registered test, independent data** | **+2.1%, p=0.526** |

At the same time, the features identify the best algorithm almost perfectly
when the function has been seen (in-sample ceiling +96.0% on the training
matrix, +97.7% on the evaluation matrix). The knowledge is there, but it does
not transfer to an unseen function.

## Caveats

1. **"Not significant" does not mean "proven zero."** With 24 functions the test
   cannot rule out a small benefit. But the measured benefit, 0.08 of a rank, is
   too small to matter in practice, and the other analyses agree.
2. **The cause of the original result cannot be pinned down exactly.** It may
   come from choosing the target after seeing results, from the narrowness of
   the original data, or from both. Either way, it does not hold on broader
   data.
3. **Training labels are noisy.** At 5 runs, about 1 row in 6 has a
   best-algorithm label that is off by more than 1% (see the Phase 2c report).

## Reproducing

```bash
python phase3b_holdout_confirm.py
```

About 15 minutes, with progress printed per fold. It is fully seeded
(`RF_SEED=20260916`, `n_estimators=300`) and refuses to run if the training and
evaluation seeds overlap.

## Related

- `docs/planning/phase3b/phase3_preregistration.md`: the method fixed in advance, and the
  2026-09-25 entry logged before this run
- [`phase3_report.md`](phase3_report.md): the selector and ablation on the
  training matrix
- [`phase2c_report.md`](phase2c_report.md): the training data
- [`phase2_report.md`](phase2_report.md): the evaluation data
