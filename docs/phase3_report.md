# Phase 3: Learned per-instance algorithm selector

Status: **complete**. Scripts: `phase3_selector_validation.py` (step 11) and
`phase3_ablation.py` (step 12). Output: `results/phase3_selector.{txt,json}`,
`results/phase3_ablation.json`. Confirmation run: [`phase3b_report.md`](phase3b_report.md).

Updated 2026-10-01 for the rebuilt training matrix. Every figure below is
literal output from those scripts. An earlier version of this report described
results on an older, smaller matrix (96 rows, dim 10 only). Those results are
superseded. What was claimed on that data is recorded in
`docs/phase3_preregistration.md`.

## Purpose

`phase2_lopo.py` uses a normalised-score target and 22 separate per-algorithm
forests. Phase 3 asks whether a different design does better on the same
training data: a **single forest** over (landscape features + algorithm
one-hot), trained to predict each algorithm's **rank**. It reads cached data
only; no optimiser runs.

## Setup

| Parameter | Value |
|---|---|
| Data | `results/phase2c_matrix.json`: 720 rows, 22 algorithms, `n_runs=5`, dims 5/10/20, seeds 42–51 |
| Model | one `RandomForestRegressor` over (features + algorithm one-hot), `n_estimators=300`, `RF_SEED=20260916` |
| Target | per-row rank of each algorithm (1 = best of 22) |
| Prediction | the algorithm with the lowest predicted rank |
| Validation | leave-one-function-out, 24 folds; all 30 rows of a function held out together |
| scipy / sklearn | 1.16.3 / 1.9.0 |

Scores are averaged as per-row **ranks**, not raw fitness. Raw best-fitness
values differ by many orders of magnitude across functions, so a raw average
would be decided by a few large-scale functions. Rank is scale-free.

## Step 11: the selector

```
  MEAN RANK over held-out rows (1 = best of 22, lower is better)
  ------------------------------------------------------------
  VBS (oracle)              1.2083
  Selector (RF)             5.5333
  SBS (fold-aware)          5.5208
  Random (analytic)        11.5000

  SBS->VBS gap closed by selector : -0.3%
  mean norm score  selector=0.0787  sbs=0.0683

  statistic=np.float64(103.5)  p=np.float64(0.6765882259671506)
  significant at alpha=0.05: NO
  selector better on 13 folds, worse on 8, tied on 3

  HEADROOM: SBS already equals VBS on 216 of 720 rows (30%)
            (on every row of a function: 0 of 24 functions)
```

**The selector ties SBS.** Its mean rank is 5.5333 against SBS's 5.5208, and
the Wilcoxon test over the 24 function means gives p=0.677.

The headroom is measured per row: SBS already picks the oracle's algorithm on
216 of 720 rows. The second line counts functions where SBS matches the oracle
on all 30 rows at once, which almost never happens with 30 rows per function.
Quote the per-row figure.

## Step 12: target × architecture ablation

To find out whether the design choices matter, `phase3_ablation.py` tries all
four combinations of target (rank or normalised score) and architecture (one
forest or 22 per-algorithm forests). Everything else is held constant:
`n_estimators=300`, seed 20260916, no feature preprocessing, the same folds and
the same SBS. Each cell is scored on both scales so that no design is judged
only on the measure it was trained for.

Gap closed vs SBS (Wilcoxon p over 24 function means), **rank scale**:

| | One forest | 22 per-algorithm forests |
|---|---|---|
| Rank target | −0.3% (p=0.6766) | −1.6% (p=0.8212) |
| Normalised target | +0.8% (p=0.8228) | −16.6% (p=0.2183) |

**Normalised scale**:

| | One forest | 22 per-algorithm forests |
|---|---|---|
| Rank target | −15.2% (p=0.5202) | −13.6% (p=0.2046) |
| Normalised target | −3.3% (p=0.7369) | −27.6% (p=0.1773) |

Control, using `phase2_lopo.py`'s own settings (`n_estimators=200`, seed
20260915, signed-log1p and standardised features), normalised target, 22
forests: **−22.9%, p=0.355959**.

Two cross-checks pass. The `rank | one forest` cell reproduces step 11 exactly
(−0.3%, p=0.6766), and the control reproduces the Phase 2c LOPO RF-regression
result exactly (−22.9%, p=0.356). The three scripts agree with one another.

## Findings

1. **No design beats SBS.** All 8 cells (4 designs × 2 scoring scales) are near
   zero or negative, and none is significant. The best is +0.8% (p=0.8228). So
   the negative result does not come from one unlucky design choice.
2. **The target encoding does not decide the outcome on this data.** With one
   forest, rank and normalised targets give −0.3% and +0.8%. The one clearly bad
   design is the normalised target with 22 separate forests (−16.6% on the rank
   scale), which is the design `phase2_lopo.py` uses. An earlier version of this
   report concluded that the target dominates. That does not hold on the
   rebuilt data.
3. **When the selector misses, it misses badly.** The rank-trained selectors
   are near zero on the rank scale (−0.3%, −1.6%) but well below zero on the
   normalised scale (−15.2%, −13.6%). They usually pick an algorithm that
   finishes near the top, but their wrong picks cost much more fitness than
   always choosing SBS would.

## Limitations

1. **Training labels are medians of 5 runs.** At 5 runs, about 1 row in 6 has a
   best-algorithm label that is off by more than 1% (see the Phase 2c report).
2. **This is evaluation inside the training matrix.** Each held-out function is
   unseen, but it shares the matrix's seeds and settings. The independent test
   is Phase 3b.
3. **The rank target was originally chosen after seeing it win an ablation.**
   That is why Phase 3b was pre-registered before it ran.

## Reproducing

```bash
python phase3_selector_validation.py   # a few minutes
python phase3_ablation.py              # about 23 minutes (measured per fold)
```

Both fit their forests once per fold. That gives results identical to fitting
once per row, which was verified when the change was made.
