# Phase 2c: Selector training matrix

Status: **complete**. Script: `phase2c_expand_matrix.py`. Output:
`results/phase2c/phase2c_matrix.json`. Evaluation:
`phase2_lopo.py --matrix results/phase2c/phase2c_matrix.json --out-stem results/phase2c/phase2c_lopo`.

Updated 2026-10-01 for the rebuilt matrix. Every figure below is literal output
from those scripts, or from the checks named next to it, run against the
current files. An earlier version of this report described an older,
smaller matrix (96 rows, dim 10 only). That matrix has been replaced. It is
kept in the backup folder as `phase2c_matrix.96row_n3_legacy.json` (see
`results/MANIFEST.md`) and its figures no longer apply.

> **This matrix is for selector training only.** Its cells are medians of 5
> runs, against 15 in `results/phase2/phase2_matrix.json`, which remains the
> authoritative source for any SBS/VBS figure quoted in the thesis.

## Purpose

A selector needs many training rows. Phase 2c multiplies the rows along the
problem axis, using more seeds and more dimensions, while keeping the same 22
algorithms and the same budget as Phase 2.

## What a "pseudo-instance" is

`aad/problems.py` has no BBOB instance concept. Its `seed` parameter moves the
optimum `xopt` and the rotation matrices `R` and `Q`. That is similar to a
BBOB instance (same function family, different optimum and rotation), but it
is not the official COCO instance scheme: there is no f_opt offset and none of
the COCO per-instance transformations. Rows are therefore called
pseudo-instances, labelled `sNN`, and should not be called BBOB instances in
the thesis.

## Setup as built

| Parameter | Value |
|---|---|
| Functions | 24 |
| Pseudo-instance seeds | 42–51 (10 seeds) |
| Dimensions | 5, 10, 20 |
| Algorithms | 22, the same set as Phase 2 |
| Runs per cell | 5, run seeds 1000–1004 |
| Budget | `max_fe=5000`, `population_size=50` |
| Rows | 24 × 10 × 3 = **720** |
| Cells | 720 × 22 = **15,840** |
| Build time | 112,514.4 s = 31.25 hours |
| Library versions (`meta.env`) | python 3.11.9, numpy 2.4.6, scipy 1.16.3, mealpy 3.0.2 |

Features are extracted separately for every row, so no feature vector is
reused across seeds or dimensions. Rows are keyed `{function}|s{seed}|d{dim}`.

## Integrity

From `results/phase2c/phase2c_matrix.json`:

```
rows=720 cells=15840 build_s=112514.4 (31.25 h)
runs/cell [5] finite [5] errors 0 nonfinite 0
```

Every cell is complete, every run is finite, and nothing is imputed.

## Feature check

`phase2c_expand_matrix.py --verify-only`:

```
  feature                            std             min             max  distinct  status
  ----------------------------------------------------------------------------------------
  dimension                       6.2361               5              20         3  near-degenerate (few distinct values)
  fitness_max                1.94092e+15         29.2152     1.97114e+16       720  ok
  fitness_mean               3.08778e+14        -19.2059     3.48413e+15       720  ok
  fitness_min                7.12086e+12        -885.867     1.29803e+14       720  ok
  fitness_std                3.28948e+14        0.104363      3.1249e+15       718  ok
  fitness_variance               103.053      0.00120609         2527.27       718  ok
  gradient_magnitude         1.23024e+14       0.0309881     1.16204e+15       718  ok
  modality                       2.61487        -11.9175         16.6917       720  ok
  ruggedness                     0.16457        0.194401         1.22686       718  ok

  No constant feature columns. Every feature varies across rows.
```

All 9 features vary. `dimension` is flagged "near-degenerate" only because it
has exactly 3 values, one per dimension built. That is expected, not a defect.

## Independence from the evaluation matrix

The evaluation matrix (`results/phase2/phase2_matrix.json`) uses `problem_seed=100`,
which is not among this matrix's seeds:

```
split: eval seed 100 in train seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] -> False
```

So no evaluation row is a re-measurement of a training row.
`phase3b_holdout_confirm.py` checks this and refuses to run if the seeds
overlap. Both matrices use `feature_seed=20260915` and `feature_samples=300`.

## Selectors on this matrix

`phase2_lopo.py`, leave-one-function-out: all 30 rows of a function (10 seeds ×
3 dimensions) are held out together, giving 24 folds. The selectors predict a
per-row min-max normalised score using 22 separate per-algorithm forests.

| Selector | Mean norm | Median | Mean rank | VBS hits | Gap closed vs SBS |
|---|---|---|---|---|---|
| VBS (oracle) | 0.0000 | 0.0000 | 1.208 | 720 | target |
| SBS (fold-aware) | 0.0683 | 0.0007 | 5.521 | 235 | baseline |
| Random | 0.2279 | 0.0675 | 11.781 | 48 | −233.5% |
| 1-NN | 0.0910 | 0.0094 | 6.922 | 133 | −33.1% |
| 3-NN | 0.0812 | 0.0033 | 6.728 | 151 | −18.8% |
| RF-regression | 0.0840 | 0.0010 | 6.118 | 219 | −22.9% |
| RF in-sample (ceiling) | 0.0028 | 0.0000 | 2.325 | 474 | +96.0% |

Primary test, pairing the 24 function means (Bonferroni threshold for 4
selectors: 0.0125):

```
  1-NN             statistic=np.float64(69.0) p=np.float64(0.019441962242126465) better on 6, worse on 18, tied on 0  *SIGNIFICANT*
  3-NN             statistic=np.float64(77.0) p=np.float64(0.06355166456894228) better on 7, worse on 16, tied on 1
  RF-regression    statistic=np.float64(57.0) p=np.float64(0.3559586988465746) better on 7, worse on 10, tied on 7
```

**Every learned selector does worse than SBS on this matrix.** The 1-NN result
flagged as significant is significant in the wrong direction: 1-NN is worse
than SBS on 18 of 24 functions. It also does not survive the Bonferroni
threshold. The rows are not independent (30 per function), which is why the
test pairs function means rather than the 720 rows.

The in-sample ceiling again reaches +96.0%. The features can identify the best
algorithm when the function has been seen. What fails is transfer to a
function the model has not seen.

SBS already matches the oracle on 235 of 720 rows.

## Limitations

1. **5 runs per cell gives noisy labels.** Using the 15 stored runs per cell in
   `results/phase2/phase2_matrix.json`, the algorithm picked from fewer runs lands
   within 1% of the 15-run best this often (computed 2026-10-01):

   ```
   n_runs=3  pick within 1% of the 15-run best: 76.4%
   n_runs=5  pick within 1% of the 15-run best: 83.3%
   n_runs=10 pick within 1% of the 15-run best: 97.2%
   ```

   At 5 runs, roughly 1 row in 6 carries a "best algorithm" label that is off by
   more than 1%. That noise is in the training labels. The 15-run "truth" is
   itself a median, so the curve is somewhat optimistic past 10 runs.
2. **The extra rows do not add independent evidence.** Rows within one function
   are related, so every test still rests on 24 function means.
3. **F5 LinearSlope barely discriminates.** On `BBOB_F5_LinearSlope|s43|d5`, 18
   of 22 algorithms tie at the exact optimum.

## Conclusions

1. The matrix is complete and clean: 720 rows, 15,840 cells, no errors, no
   missing or non-finite values.
2. All 9 features vary, so the dimension axis now carries signal.
3. The split against the evaluation matrix is independent (seeds 42–51 against
   100).
4. With a normalised-score target and per-algorithm forests, no selector beats
   SBS on this matrix. RF-regression closes −22.9% (p=0.356).
5. The features carry the information (ceiling +96.0%); transfer to unseen
   functions is what fails.

## Reproducing

```bash
python phase2c_expand_matrix.py                # ~31 hours, writes results/phase2c/phase2c_matrix.json
python phase2c_expand_matrix.py --verify-only  # seconds; feature-variance check
python phase2_lopo.py --matrix results/phase2c/phase2c_matrix.json --out-stem results/phase2c/phase2c_lopo
```

The build writes incrementally and resumes safely. It refuses to resume a file
built with a different `n_runs`, `feature_seed` or `feature_samples`, because
those are not part of the cell key and mixing them would corrupt the matrix
silently.

## Related

- [`phase2_report.md`](phase2_report.md): the authoritative evaluation matrix
- [`phase3_report.md`](phase3_report.md): selectors trained on this matrix
- [`phase3b_report.md`](phase3b_report.md): the independent confirmation test
