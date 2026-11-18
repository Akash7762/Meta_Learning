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

### 2026-09-25 — registered method to be re-applied to rebuilt, independent data

Recorded before the registered test was run on the full rebuilt data, and
committed before that run. A subset run used only to verify code is disclosed
below. Nothing above this line has been edited.

**Why the data changed.** The 2026-09-17 entry found the evaluation set was
not independent. Both matrices were since rebuilt, partly to create an
independent evaluation set and partly to add dimensions 5 and 20:

| | As registered (2026-09-17) | Rebuilt |
|---|---|---|
| Training, `phase2c_matrix.json` | 96 rows; dim 10; seeds 42–45; n_runs=3 | 720 rows = 24 functions × dims 5/10/20 × seeds 42–51; n_runs=5; 15,840 cells; built 2026-09-24 in 112,514.4 s |
| Evaluation, `phase2_matrix.json` | 24 rows; dim 10; seed 42; n_runs=15 | 72 rows = 24 functions × dims 5/10/20; seed 100; n_runs=15; 1,584 cells; built 2026-09-23 in 30,420.5 s |

In both rebuilt matrices every cell is complete, every run is finite and
nothing is imputed. Both use feature_seed=20260915 and feature_samples=300,
and both record library versions in `meta.env` (python 3.11.9, numpy 2.4.6,
scipy 1.16.3, mealpy 3.0.2).

**The split is now independent.** Evaluation seed 100 is not among training
seeds 42–51, so no evaluation row re-measures a training configuration. This
supersedes the 2026-09-17 statement that no independent set exists.
`phase3b_holdout_confirm.py` now checks this and refuses to run if the seeds
overlap.

The confirmation run on the original data (2026-09-17: +52.8% gap closed,
p=0.0072, reported in `docs/phase3b_report.md`) stands as the result for that
data. The coming run is a second application of the same registered method to
new data, not a re-analysis of the first.

**The method is unchanged.** Model, rank target, argmin prediction,
leave-one-function-out validation, SBS from training rows only, VBS as the
per-row oracle, one two-sided Wilcoxon at alpha=0.05, and the pre-specified
interpretation all apply exactly as written above. Two consequences of the new
data follow, neither of them a choice:

1. Each held-out function now has 3 evaluation rows, one per dimension. These
   are not independent, so each function's rows are averaged before pairing,
   and the test runs over the 24 held-out functions, as registered. With one
   row per function, as in the original data, this is identical to pairing
   rows.
2. Training cells are medians of 5 runs rather than 3. The justification for
   the rank target (robustness to median noise; scale invariance) applies
   unchanged.

**Code changes to `phase3b_holdout_confirm.py` since the original run**, none
altering the method: per-row dimensions when loading the evaluation matrix;
the seed-overlap guard; `--train-matrix` / `--eval-matrix` arguments; and the
function-mean aggregation in item 1 (commit `fb3f35e`); output files closed
properly (commit `c5ab097`); and fitting the forest once per held-out function
instead of once per evaluation row (committed with this entry). The last is
computational only. Run end-to-end on a subset, the old and new scripts
produced identical output in every field.

**Already known when this was written, disclosed for transparency:**

- The registered method has already been evaluated **within the rebuilt
  training matrix** (leave-one-function-out on `phase2c_matrix.json` alone,
  via `phase3_selector_validation.py`): −0.3% gap closed, p=0.677. Related
  designs on the same matrix: normalised target with 22 per-algorithm forests,
  −22.9% (p=0.356); and a target × architecture 2×2 in which no cell
  significantly beats SBS on either scoring scale (best +0.8%, p=0.823). None
  of these is the registered cross-file test.
- The rebuilt evaluation matrix has been analysed on its own: rankings and the
  SBS–VBS gap (`phase2_analyse_matrix.py`), and within-matrix
  leave-one-function-out with a normalised target (RF-regression +30.8%,
  p=0.053).
- To verify the per-fold change, `phase3b_holdout_confirm.py` was run on a
  subset of both matrices: the first 5 functions in sorted order
  (F10_EllipsoidalRotated to F14_DifferentPowers), selected by name before any
  output was seen. Observed: selector mean rank 4.9333 against SBS 2.9333,
  Wilcoxon statistic 0.0, p=0.25 over those 5 functions. No method choice was
  made from it.
