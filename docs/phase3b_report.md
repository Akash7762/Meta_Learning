# Phase 3b — Pre-Registered Confirmation Run

Status: **complete**. Script: `phase3b_holdout_confirm.py`.
Pre-registration: `results/phase3_preregistration.md` (written **before** any
confirmation code existed; amended only in its append-only section).
Output: `results/phase3b_holdout_confirm.{txt,json}`.

## Purpose

The Phase 3 headline (+38.2%, p=0.0149) used a rank target chosen **after**
seeing it win an 8-cell ablation. That is post-hoc selection, and the p-value
does not account for the search that produced it. Phase 3b fixes the method in
advance and runs **one** pre-specified test.

## Pre-registered method

- **Model**: one `RandomForestRegressor` over (features + algorithm one-hot).
- **Target**: per-row rank.
- **CV**: leave-one-**function**-out.
- **Train**: `results/phase2c_matrix.json` — 96 rows, `n_runs=3`.
- **Eval**: `results/phase2_matrix.json` — 24 rows, `n_runs=15`.
- **One** test, `scipy.stats.wilcoxon`, paired, α=0.05, no further search.

**Justification for the rank target**, stated independently of which encoding
won the ablation: (1) robustness to 3-run median noise — a rank changes only if
noise reorders two algorithms, a higher bar than perturbing a value; (2) scale
invariance — fitness spans ~1e-2 to 1e9 across functions and F5 is negative.

## Compatibility check

| | phase2 | phase2c |
|---|---|---|
| algorithms | 22 | 22 — identical sets |
| feature schema | 9 keys | identical |
| cell key | `BBOB_F1_Sphere\|ABC` | `BBOB_F1_Sphere\|s42\|d10\|ABC` |
| dim / seed | `problem_seed=42` | `seeds=[42,43,44,45]`, `dims=[10]` |
| `n_runs` | 15 | 3 |

Schemas are compatible. A different problem surfaced:

```
phase2 functions also present in phase2c as s42|d10: 24 of 24
feature vectors bit-identical: 24 of 24
```

## The evaluation set is NOT independent

Phase 2 was built with `problem_seed=42`; Phase 2c includes
`pseudo_instance_seed=42`. **All 24 evaluation rows are re-measurements of
(function, seed=42, dim=10) configurations already in the training file**, with
bit-identical feature vectors. They differ only in `n_runs` (15 vs 3).

A straight train-on-96 / test-on-24 split would leak every evaluation row into
training. Leave-one-function-out is therefore applied **across the two files**:
to predict function F, training uses only Phase 2c rows where function ≠ F.
This was already the pre-specified rule, so the method is unchanged.

## Result

```
  MEAN RANK on held-out functions (1 = best of 22, lower is better)
  ------------------------------------------------------------
  VBS (oracle)              1.2500
  Selector (RF)             3.7083
  SBS (fold-aware)          6.4583
  Random (analytic)        11.5000

  SBS->VBS gap closed : +52.8%

  statistic=np.float64(24.0)  p=np.float64(0.007182896385181838)
  significant at alpha=0.05: YES
  selector better on 15, worse on 3, tied on 6
```

## Does Phase 3 replicate?

**Yes, on the two things this run can test.** The effect is larger here
(+52.8% vs +38.2%) with a smaller p-value (0.0072 vs 0.0149). Direction,
magnitude and significance all hold when targets come from 15-run medians
instead of 3-run medians.

That addresses the noise concern directly: the Phase 3 result was **not** an
artifact of 3-run measurement noise in the evaluation targets.

## What this does NOT establish

This run tests:

1. generalisation to an **unseen function**, and
2. robustness to **15-run vs 3-run** targets.

It does **not** resolve the post-hoc selection risk from the 8-cell ablation.
The rank target was still chosen after seeing it win, and this run reuses that
choice rather than testing it. Resolving it requires problem configurations
absent from Phase 2c — new seeds or new dimensions — which do not exist in this
project's data.

**Honest statement of the finding:** *the selector's advantage replicates on
unseen functions and survives higher-precision targets; the target-encoding
choice remains unvalidated.*

## Reproducing

```bash
python phase3b_holdout_confirm.py
```

Under a minute. Fully seeded (`RF_SEED=20260916`, `n_estimators=300`).

## Related

- [`phase3_report.md`](phase3_report.md) — the result being confirmed
- [`phase2c_report.md`](phase2c_report.md) — the training data
- [`phase2_report.md`](phase2_report.md) — the evaluation data (authoritative)
