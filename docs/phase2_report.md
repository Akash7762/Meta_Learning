# Phase 2: Cross-problem performance matrix and algorithm ranking

Updated 2026-10-01 for the rebuilt multi-dimension matrix. Every figure below
is literal output from `phase2_analyse_matrix.py` and `phase2_lopo.py` run
against `results/phase2_matrix.json`. Figures marked *supplementary* were
computed directly from the same file during the 2026-10-01 update and are not
printed by either script. No optimiser was run to produce this report.

An earlier version of this report described an older matrix (seed 42, dim 10
only, 24 rows). That matrix has been replaced and its figures no longer apply.
It survives only in git history.

## 1. Purpose

Phase 1b screened 21 algorithms on a single function. Phase 2 asks the
cross-problem question: does one algorithm win across the whole BBOB suite,
and if not, can a per-problem selector beat always picking the single best one?

Two reference points frame every result:

- **SBS** (single best solver): the algorithm with the best mean rank across
  all problems. Any selector must beat it to be worth using.
- **VBS** (virtual best solver): the per-problem oracle, which always picks the
  best algorithm. It is the ceiling.

## 2. Setup

| Parameter | Value |
|---|---|
| Problems | 24 BBOB functions × dims 5, 10, 20 = **72 rows**, `problem_seed=100` |
| Algorithms | 22 (21 mealpy + the built-in `ea`) |
| Runs per cell | 15, run seeds 1000–1014 |
| Budget | `population_size=50`, `max_fe=5000` (`ea`: `max_generations=100`) |
| Cell value | median best fitness over the 15 runs |
| Matrix | 1584 cells |

The budget is equal in **evaluations**, not generations. Population methods
spend `pop_size` evaluations per generation while single-solution methods such
as SA spend about one, so equal generations would not be a fair comparison.

Seed 100 lies outside the Phase 2c training seeds (42–51), so this matrix can
serve as an independent evaluation set (see Phase 3b).

### Integrity

```
  cells expected            : 1584
  cells present             : 1584
  cells missing             : 0 []
  runs per cell             : [15]
  finite runs per cell      : [15]
  total NaN/Inf runs        : 0
  cells containing NaN/Inf  : 0 []
  cells with errors         : 0 []
  matrix build wall-clock   : 30420.5s = 8.45 hours
  identical problem rows    : 0 []
```

No missing cells, no non-finite runs, nothing imputed. The matrix records the
library versions that built it in `meta.env`: python 3.11.9, numpy 2.4.6,
scipy 1.16.3, mealpy 3.0.2.

## 3. Result 1: the algorithms differ significantly

Friedman test over 22 algorithms × 72 rows:

```
  statistic     : np.float64(720.4668450340773)
  p-value       : np.float64(6.271874289949954e-139)
  significant?  : YES - algorithms do NOT all perform equally
```

So choosing an algorithm per problem could matter in principle.

## 4. Result 2: EO and MPA are jointly the strongest

Mean rank, 1 = best (top 8 of 22):

| Rank | Algorithm | Mean rank | Wins (rank 1) |
|---|---|---|---|
| 1 | **EO** | **4.722** | 9 |
| 2 | **MPA** | **4.847** | 27 |
| 3 | L_SHADE | 6.000 | 5 |
| 4 | SHADE | 6.028 | 0 |
| 5 | JADE | 7.681 | 1 |
| 6 | MVO | 8.514 | 6 |
| 7 | PSO | 8.861 | 11 |
| 8 | RIME | 9.250 | 0 |

The worst three are BBO (16.750), HHO (17.500) and AO (20.417).

EO has the best mean rank, so the analyser names it SBS. But EO and MPA cannot
be told apart (*supplementary*, Wilcoxon on their ranks over the 72 rows):

```
EO vs MPA ranks over 72 rows: statistic=1150.5 p=0.5894
```

MPA wins outright far more often (27 rows against EO's 9). EO is the steadier
performer and MPA the stronger specialist. Report them as jointly strongest,
not as a single winner.

### The best algorithm depends on dimension

*Supplementary*, mean rank by dimension:

| Algorithm | d5 | d10 | d20 |
|---|---|---|---|
| EO | 5.208 | 4.625 | 4.333 |
| MPA | 3.458 | 5.125 | 5.958 |
| L_SHADE | 4.917 | 6.667 | 6.417 |
| SHADE | 4.833 | 6.458 | 6.792 |
| PSO | 7.833 | 6.583 | 12.167 |

MPA is clearly best at d5 and gets worse as dimension grows. EO moves the other
way. PSO falls sharply at d20. A selector could in principle exploit this,
which is why `dimension` is kept as a live feature (its standard deviation
across the 72 rows is 6.236).

## 5. Result 3: how much room a selector has

```
  mean normalised gap       : 0.0604
  median normalised gap     : 0.0062
  max normalised gap        : 0.6095  (on BBOB_F23_Katsuura|s100|d20)
  problems where SBS == VBS : 11 / 72

  rank gap (SBS mean rank - 1): 3.722
  VBS distinct algorithms     : 13 -> ['ABC', 'ARO', 'EO', 'GA', 'GWO', 'HHO', 'JADE', 'L_SHADE', 'MPA', 'MVO', 'PSO', 'SMA', 'ea']
```

The normalised gap maps each row onto [0, 1]: 0 means SBS matched the oracle,
1 means SBS was the worst algorithm on that row. SBS already matches the oracle
on 11 of 72 rows. Thirteen different algorithms are the oracle's pick somewhere,
so there is real room for a selector, but on most rows the margin is small.

## 6. Result 4: selectors within this matrix

`phase2_lopo.py`, leave-one-function-out: all 3 rows of a function are held out
together, giving 24 folds. SBS is recomputed per fold from the training rows
only, so it never sees held-out data. The selectors here predict a per-row
min-max normalised score using 22 separate per-algorithm forests.

| Selector | Mean norm | Median | Mean rank | VBS hits | Gap closed vs SBS |
|---|---|---|---|---|---|
| VBS (oracle) | 0.0000 | 0.0000 | 1.201 | 72 | target |
| SBS (fold-aware) | 0.0706 | 0.0062 | 6.125 | 10 | baseline |
| Random | 0.2351 | 0.0387 | 12.069 | 7 | −233.1% |
| 1-NN | 0.0951 | 0.0032 | 7.403 | 16 | −34.8% |
| 3-NN | 0.0564 | 0.0003 | 4.722 | 28 | +20.1% |
| RF-regression | 0.0489 | 0.0001 | 4.861 | 31 | +30.8% |

The 72 rows are not independent (3 dimensions inside each of 24 functions), so
the primary test averages each function's rows first and pairs the 24 function
means. Bonferroni threshold for 4 selectors: 0.0125.

```
  1-NN             statistic=np.float64(103.0) p=np.float64(0.28709167639284705) better on 10, worse on 13, tied on 1
  3-NN             statistic=np.float64(76.0) p=np.float64(0.059331738977628015) better on 17, worse on 6, tied on 1
  RF-regression    statistic=np.float64(67.0) p=np.float64(0.05339578044263108) better on 17, worse on 5, tied on 2
```

**No selector beats SBS significantly.** RF-regression and 3-NN are ahead on
the mean and win on most functions, but neither reaches p < 0.05. This is
within-matrix evidence only. The confirmatory test, training on Phase 2c and
evaluating here, is Phase 3b.

The SBS here matches the oracle on 10 of 72 rows, against 11 in Section 5. The
difference is that `phase2_lopo.py` recomputes SBS for each fold from the
training rows, while the analyser uses one global SBS.

## 7. Result 5: the features carry signal

An in-sample diagnostic trains the forest on every row, including the held-out
one, so the model may memorise the answer:

| Selector | Mean norm | Mean rank | VBS hits | Gap closed |
|---|---|---|---|---|
| RF in-sample (ceiling) | 0.0016 | 3.049 | 45 | **+97.7%** |

A model allowed to see the answer closes 97.7% of the gap, so the 9 landscape
features are enough to tell the rows apart. When selectors fall short, the
problem is transfer to an unseen function, not missing information. The
ceiling is a diagnostic, not a usable selector, and is left out of the
significance tests.

## 8. Known limitations

1. **One problem seed.** `problem_seed=100` gives one landscape realisation per
   (function, dimension).
2. **24 functions bound the test.** Every significance test pairs at most 24
   independent function means, however many rows each function has.
3. **F5 LinearSlope barely discriminates.** On `BBOB_F5_LinearSlope|s100|d5`,
   17 of 22 algorithms tie at the exact optimum, so that row rewards almost any
   choice.

## 9. Conclusions

1. The algorithms differ significantly across the 72 rows (Friedman
   p ≈ 6.27e-139).
2. EO (mean rank 4.722) and MPA (4.847) are jointly the strongest and cannot be
   told apart (p=0.5894).
3. The best algorithm depends on dimension: MPA leads at d5, EO at d20.
4. SBS matches the oracle on 11 of 72 rows, and 13 algorithms win somewhere,
   so a selector has room to help.
5. Within this matrix, no selector beats SBS significantly, although
   RF-regression (+30.8%, p=0.053) and 3-NN (+20.1%, p=0.059) lean that way.
6. The features identify the best algorithm when the function has been seen
   (ceiling +97.7%).

## 10. Reproducing

```bash
python phase2_build_matrix.py     # ~8.45 hours, writes results/phase2_matrix.json
python phase2_analyse_matrix.py   # seconds; ranking, Friedman, SBS/VBS gap
python phase2_lopo.py             # about 3 minutes; LOPO selector evaluation
```

`results/` is tracked in git.

## 11. Related

- `docs/phase0_decision.md`: the equal-budget protocol
- `results/phase2_lopo.txt`: full LOPO output, including per-row detail
- [`phase3b_report.md`](phase3b_report.md): the independent confirmation test
  that uses this matrix as its evaluation set
