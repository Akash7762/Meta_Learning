# Phase 3 — Learned Per-Instance Algorithm Selector

Status: **complete**. Scripts: `phase3_selector_validation.py`,
`phase3_ablation.py`. Output: `results/phase3_selector.{txt,json}`,
`results/phase3_ablation.json`.

Confirmation run: [`phase3b_report.md`](phase3b_report.md).

## Purpose

Phases 2 and 2c both found that no learned selector beats the SBS. Phase 3 asks
whether that conclusion survives a different **target encoding** and
**model architecture**, on the same cached data. Reads cached JSON only; no
optimiser is invoked and `build_algorithm` is never called.

## Setup

| Parameter | Value |
|---|---|
| Data | `results/phase2c_matrix.json` — 96 rows, `n_runs=3`, dim 10 |
| Algorithms | 22 |
| Model | one `RandomForestRegressor` over (features + algorithm one-hot) |
| Target | per-row **rank** of each algorithm (1 = best of 22) |
| Prediction | argmin of predicted rank |
| CV | leave-one-**function**-out, 24 folds |
| Seeds | `RF_SEED=20260916`, `n_estimators=300` |
| scipy / sklearn | 1.16.3 / 1.9.0 |

**Aggregation.** Scores are aggregated by per-row **rank**, not raw fitness.
Raw best-fitness spans roughly −2e2 (F5) to 1e9 (F12) across functions, so a
mean over raw values would be decided entirely by a few large-scale functions.
Rank is scale-free and directly comparable across rows.

**Dimension is a live variable** throughout — row keys, feature vector, fold
grouping and the JSON schema. Appending dims 5 and 20 is a data change, not a
code change.

## Result

```
  MEAN RANK over held-out rows (1 = best of 22, lower is better)
  ------------------------------------------------------------
  VBS (oracle)              1.2552
  Selector (RF)             4.4948
  SBS (fold-aware)          6.4948
  Random (analytic)        11.5000

  SBS->VBS gap closed by selector : +38.2%
  mean norm score  selector=0.0525  sbs=0.0934

  statistic=np.float64(34.5)  p=np.float64(0.014875680809294889)
  significant at alpha=0.05: YES
  selector better on 14 folds, worse on 5, tied on 5

  HEADROOM: SBS already equals VBS on 2 of 24 functions
```

**The selector beats SBS, significantly.** This contradicts the Phase 2 and 2c
conclusions on the same data, which required explanation.

## The ablation — why the earlier runs disagreed

`phase2_lopo.py` reported **+14.2%, not significant** on identical data.
`phase3_ablation.py` isolates the cause across a 2×2, holding `n_estimators`,
seed and feature preprocessing constant so only target and architecture vary.

**Gap closed % (Wilcoxon p), scored on RANK scale:**

| | single + one-hot | 22 per-algo RFs |
|---|---|---|
| target=rank | **+38.2% (p=0.0149)** | +32.0% (p=0.0521) |
| target=norm | +13.7% (p=0.3082) | +12.3% (p=0.3623) |

**Gap closed % (Wilcoxon p), scored on NORM scale:**

| | single + one-hot | 22 per-algo RFs |
|---|---|---|
| target=rank | +43.8% (p=0.0641) | +39.0% (p=0.2322) |
| target=norm | +15.7% (p=0.6874) | +20.1% (p=0.3066) |

Control reproducing `phase2_lopo.py`'s own knobs (`trees=200`,
`seed=20260915`, `prep=True`): **+14.2%, p=0.434756** — matching the original
figure exactly.

### Which factor dominates

**The regression target, decisively. Architecture is near-irrelevant.**

- Switching target `norm` → `rank` moves the result from ~+13% to ~+35% in
  *both* architectures — roughly a 3× swing.
- Switching architecture moves it by 1–6 points.

The pattern holds on both scoring scales, so it is not an artifact of scoring
on the metric the model was trained on.

**Mechanism (hypothesis, not proven here).** Each cell is a median of only 3
runs. A min-max normalised target inherits that noise on a continuous,
heavy-tailed scale, where one noisy large-magnitude cell distorts a whole row's
normalisation. Ranks compress each row to 1..22, so a cell must be *misordered*,
not merely mis-valued, to corrupt the target. The clean test would be a rebuild
at `n_runs=15`.

The control line confirms `phase2_lopo.py` was computed correctly — it simply
used the noise-sensitive target. The two scripts do not contradict each other.

## Caveats — recorded, not resolved

1. **Post-hoc selection risk.** The rank target was chosen *after* observing it
   win across the 8-cell ablation. The reported p-value does not account for
   that search. This is the motivation for [Phase 3b](phase3b_report.md).
2. **Fragile significance.** Only 1 of 8 ablation cells clears α=0.05, and its
   norm-scale counterpart is p=0.0641. None would survive a multiplicity
   correction across the eight reported comparisons.
3. **`n_runs=3` targets**, not 15.
4. **dim=10 only**, 96 training rows; the dimension axis is untested.

## Reproducing

```bash
python phase3_selector_validation.py
python phase3_ablation.py
```

The ablation exceeds 10 minutes (the `per_algo` cells fit 2,112 RFs each). It
emits a large volume of sklearn `delayed`/`Parallel` warnings from `n_jobs=-1`
on small fits — harmless, but it can swamp captured stdout; read the numbers
from `results/phase3_ablation.json`.
