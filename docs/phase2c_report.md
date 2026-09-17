# Phase 2c — Matrix Expansion Along the Problem Axis

Status: **complete, with two documented deviations from spec.**
Script: `phase2c_expand_matrix.py`. Output: `results/phase2c_matrix.json`.
Evaluation: `phase2_lopo.py --matrix results/phase2c_matrix.json`.

> **This matrix is not authoritative.** Its own metadata states:
> *"SELECTOR TRAINING ONLY. n_runs=3 makes per-cell medians noisier than
> Phase 2. results/phase2_matrix.json (n_runs=15) remains the AUTHORITATIVE
> source."* Any SBS/VBS headline figure for the thesis should cite Phase 2.

## Purpose

Phase 2's negative result — no learned selector beats the SBS — was attributed
to having only 24 training rows for a 22-way choice. Phase 2c multiplies rows
without changing the algorithm set, to test whether the bottleneck was simply
training-set size.

## Ground truth established first

`aad/problems.py` has **no BBOB instance concept**. There is no instance index.
The `seed` parameter varies `xopt` and the rotation matrices; `dim` varies
dimensionality. Rows are therefore labelled **pseudo-instances** (`sNN`), and
the metadata records this explicitly so they are never mistaken for
COCO/BBOB instances.

## Setup as built

| Parameter | Value |
|---|---|
| Functions | 24 |
| Pseudo-instance seeds | 42, 43, 44, 45 |
| Dimensions | **[10]** |
| Algorithms | 22 — identical set to Phase 2 (verified) |
| `n_runs` | **3**, seeds 1000–1002 |
| `max_fe` | 5000, `population_size=50` |
| Row key | `{function}\|s{seed}\|d{dim}` |
| Cell key | `{function}\|s{seed}\|d{dim}\|{algorithm}` |
| Rows | 96 |
| Cells | 2112 |
| Build time | 8508.251 s = 2.36 hours |

Features are re-extracted per row, not reused across pseudo-instances.
Writes are incremental, so a restart skips completed cells.
`results/phase2_matrix.json` is never overwritten.

## Deviations from specification

| Spec | Built | Impact |
|---|---|---|
| `n_runs=15` | **3** | Per-cell medians are noisier. Became the binding limitation on every Phase 3 claim. |
| dims 5, 10, 20 | **[10] only** | The dimension axis is entirely untested; the `dimension` feature is constant. |
| 15 pseudo-instances (planned) | **4** | 96 rows instead of the ~360 target. |

Both deviations are recorded here rather than silently absorbed.

## Integrity

```
cells: 2112  expected 96*22 = 2112
cells with errors or zero finite runs: 0
distinct functions : 24
distinct seeds     : [42, 43, 44, 45]
distinct dims      : [10]
distinct (func,seed,dim) combinations: 96
```

No missing cells, no failed cells, no imputation.

## Feature variance check

```
  dimension                       0.000000  <-- CONSTANT
  fitness_max           1303764352904908.250000
  fitness_mean          213527587125356.656250
  fitness_min           734275224820.737183
  fitness_std           223956232672173.343750
  fitness_variance               17.251027
  gradient_magnitude     89076291606391.625000
  modality                        2.034012
  ruggedness                      0.144403
```

**The `dimension` column has exactly zero variance**, because only dim 10 was
built. This is the structural defect the check was designed to catch. It causes
no harm in practice — `phase2_lopo.py` drops zero-variance columns before
fitting — but it carries no learning signal, and the dimension axis the
expansion was partly meant to test was never exercised.

## Results — LOPO selector evaluation

Leave-one-**function**-out, all 4 rows of a held-out function withheld together.

| Selector | Mean norm | Median | Mean rank | VBS hits | Gap closed vs SBS |
|---|---|---|---|---|---|
| VBS (oracle) | 0.0000 | 0.0000 | 1.255 | 96 | — (target) |
| SBS (fold-aware) | 0.0934 | 0.0152 | 6.495 | 14 | — (baseline) |
| Random | 0.2465 | 0.0786 | 12.344 | 7 | −164.1% |
| 1-NN | 0.0885 | 0.0106 | 7.490 | 16 | +5.2% |
| 3-NN | 0.0816 | 0.0016 | 6.234 | 26 | +12.6% |
| RF-regression | 0.0801 | 0.0005 | 5.734 | 25 | +14.2% |
| RF in-sample (ceiling) | 0.0027 | 0.0000 | 2.984 | 54 | +97.1% |

## The statistical trap — pseudo-replication

Wilcoxon over the **96 raw rows**:

```
  1-NN             statistic=np.float64(1582.0) p=np.float64(0.8553795410111338) better on 38, worse on 42, tied on 16
  3-NN             statistic=np.float64(895.0)  p=np.float64(0.1787250089823328) better on 37, worse on 29, tied on 30
  RF-regression    statistic=np.float64(626.0)  p=np.float64(0.03337829011618602) better on 39, worse on 21, tied on 36
```

RF-regression appears significant at `p = 0.0334`. **It is not.** The 96 rows
are not independent — they are 4 pseudo-instances within each of 24 functions.
Treating clustered rows as independent inflates significance.

Aggregating to 24 function means first, then pairing, gives
**`p = 0.434756`** (reproduced this session as the control cell in
`phase3_ablation.py`). With three selectors compared, a Bonferroni threshold of
0.05/3 = 0.0167 applies as well.

**Correct conclusion: no learned selector beats SBS significantly on Phase 2c
data either.** The Phase 2 negative result replicates at 4× the rows.

## Conclusions

1. Quadrupling training rows (24 → 96) **did not** achieve significance.
2. The ceiling diagnostic again reaches ~97%, so features carry signal; the
   failure remains one of **transfer across functions**.
3. More pseudo-instances of the same 24 functions is therefore unlikely to
   help — they add rows but no new landscape families.
4. `n_runs=3` became the binding limitation on all downstream claims.

## Reproducing

```bash
python phase2c_expand_matrix.py
python phase2_lopo.py --matrix results/phase2c_matrix.json --out-stem results/phase2c_lopo
```

The build takes ~2.4 hours. The LOPO evaluation exceeds 10 minutes.

## Related

- [`phase2_report.md`](phase2_report.md) — the authoritative matrix
- [`phase3_report.md`](phase3_report.md) — what was built on this data
