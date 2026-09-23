# Phase 3b — Pre-registration

Written **before** any confirmation code was written or run. Not to be revised
afterwards. Any deviation forced by the data is to be reported as a deviation
below the line, not by editing what precedes it.

Date written: 2026-09-17

## Why this document exists

The Phase 3 headline (+38.2% gap closed, Wilcoxon p=0.0149) was obtained with a
rank regression target that was chosen **after** observing it outperform the
normalised-score target across an 8-cell ablation grid. That is post-hoc
selection: the reported p-value does not account for the search that produced
it, and it cannot stand as a confirmatory result. This document fixes the
method in advance of a single confirmation run on data never used for training.

## Method, fixed in advance

- **Model**: one `sklearn.ensemble.RandomForestRegressor` over the concatenation
  of (landscape feature vector, algorithm one-hot). Single model, not 22
  per-algorithm models.
- **Target**: per-row rank of each algorithm's median best-fitness
  (1 = best of 22, 22 = worst).
- **Prediction rule**: argmin of predicted rank over the 22 algorithms.
- **Validation**: leave-one-FUNCTION-out. All rows of a function — every seed,
  every dim — are held out together.
- **Training set**: `results/phase2c_matrix.json` (96 rows, n_runs=3, dim=10).
- **Evaluation set**: `results/phase2_matrix.json` (24 rows, n_runs=15).
- **Dimension**: `dim` remains a live variable in all keys, feature vectors and
  groupings. It currently holds only the value 10.
- **Seeds**: every stochastic component seeded; seeds recorded in the output.

## Justification for the rank target

Stated independently of which encoding won the ablation, on two grounds that
were true before the ablation was run:

1. **Robustness to 3-run median noise.** Each training cell is a median of only
   3 runs. A continuous target inherits that sampling noise directly on its own
   scale. A rank target changes only when noise is large enough to reorder two
   algorithms, which is a strictly higher bar than perturbing a value.
2. **Scale invariance across functions.** Best-fitness values span roughly 1e-2
   to 1e9 across the 24 BBOB functions, and F5 is negative. Any target defined
   on raw or min-max-normalised fitness lets a handful of large-magnitude
   functions dominate the fitted model. Rank is scale-free by construction and
   directly comparable across rows.

## Pre-specified analysis

- **One** statistical test: `scipy.stats.wilcoxon`, paired, selector vs SBS,
  over the held-out functions.
- **alpha = 0.05**, two-sided.
- **No** further model search, target search, hyperparameter search, or
  alternative-metric reporting, regardless of the outcome.
- SBS is computed from training rows only. VBS is the per-row oracle on the
  evaluation set.

## Pre-specified interpretation

- p < 0.05 **and** positive gap closed → the Phase 3 result replicates.
- Otherwise → the Phase 3 result **does not** replicate, and the +38.2% figure
  is to be treated as an artifact of post-hoc target selection. This outcome is
  to be reported plainly and is a legitimate finding.

---

## Deviations and observations recorded after writing the above

(To be appended only. Nothing above this line is edited.)

### 2026-09-17 — evaluation set is not independent

Checked before the confirmation run. `phase2_matrix.json` was built with
`problem_seed = 42`, and `phase2c_matrix.json` includes `pseudo_instance_seed
= 42`. All 24 evaluation rows correspond to (function, seed=42, dim=10)
configurations already present in the training file, with bit-identical
feature vectors (24 of 24 verified by `np.allclose(rtol=1e-12, atol=0)`).
They differ only in `n_runs` (15 vs 3).

Consequence: a straight train-on-all-96 / test-on-24 split would leak every
evaluation row into training. The leave-one-FUNCTION-out rule already
specified above resolves this — to predict function F, training uses only
phase2c rows with function != F — so the pre-specified method is unchanged.

What the run therefore tests: (a) generalisation to an unseen function, and
(b) robustness of the result to 15-run versus 3-run targets. What it does
NOT do: resolve the post-hoc selection risk arising from the 8-cell
ablation. No independent instance set exists in this project's data.
