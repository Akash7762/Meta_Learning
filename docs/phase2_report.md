# Phase 2 — Cross-Problem Performance Matrix and Algorithm Ranking

Written 2026-09-17. Every figure below is literal output from
`phase2_analyse_matrix.py` and `phase2_lopo.py` run against
`results/phase2_matrix.json` in that session. No optimiser was invoked to
produce this report; both scripts read the cached matrix only.

## 1. Purpose

Phase 1b screened 21 algorithms on a single function. Phase 2 asks the
cross-problem question: **does any one algorithm dominate across the whole BBOB
suite, and if not, can a per-problem selector do better than always picking the
best single one?**

Two reference points frame every result:

- **SBS** (Single Best Solver) — the algorithm with the best mean rank across
  all problems. The baseline any selector must beat to justify its existence.
- **VBS** (Virtual Best Solver) — the per-problem oracle. The ceiling.

## 2. Experimental setup

| Parameter | Value |
|---|---|
| Problems | 24 BBOB functions, `dim=10`, `problem_seed=42` |
| Algorithms | 22 (21 mealpy + built-in `ea`) |
| Runs per cell | 15, seeds `1000–1014` |
| Budget | `population_size=50`, `max_fe=5000` (`ea`: `max_generations=100`) |
| Cell value | median best-fitness over the 15 runs |
| Matrix | 528 cells |

Equal **evaluation** budget, not equal generations — population methods spend
`pop_size` evaluations per generation while single-solution methods such as SA
spend roughly one, so equal generations would not be a fair comparison.

### Integrity

```
  cells expected            : 528
  cells present             : 528
  cells missing             : 0 []
  runs per cell             : [15]
  finite runs per cell      : [15]
  total NaN/Inf runs        : 0
  cells with errors         : 0 []
  matrix build wall-clock   : 10771.8s = 2.99 hours
  identical problem rows    : 0 []
```

No missing cells, no non-finite runs, no imputation anywhere.

## 3. Result 1 — algorithms differ significantly

Friedman test over 22 algorithms × 24 problems:

```
  statistic     : np.float64(263.37684805049946)
  p-value       : np.float64(8.357102617754152e-44)
  significant?  : YES - algorithms do NOT all perform equally
```

So per-problem selection is not obviously pointless — the algorithms are
genuinely different.

## 4. Result 2 — MPA is the Single Best Solver

Mean rank, 1 = best (top 8 of 22):

| Rank | Algorithm | Mean rank | Wins (rank 1) |
|---|---|---|---|
| 1 | **MPA** | **4.500** | 9 |
| 2 | EO | 5.125 | 3 |
| 3 | SHADE | 6.333 | 0 |
| 4 | L_SHADE | 6.542 | 0 |
| 5 | JADE | 7.750 | 0 |
| 6 | MVO | 7.917 | 0 |
| 7 | PSO | 7.958 | 6 |
| 8 | RIME | 8.708 | 0 |

Worst three: WOA (17.292), HHO (18.792), AO (19.875) — consistent with the
Phase 1b screen.

Note PSO: mean rank only 7.958 but **6 outright wins**. It is a specialist —
excellent on the separable/simple functions (F1, F2, F3, F8, F12, F14), poor
elsewhere. Exactly the profile a selector should be able to exploit.

## 5. Result 3 — the headroom is small

```
  mean normalised gap       : 0.0407
  median normalised gap     : 0.0000
  max normalised gap        : 0.3010  (on BBOB_F20_Schwefel)
  problems where SBS == VBS : 10 / 24
  rank gap (SBS mean rank - 1): 3.500
  VBS distinct algorithms     : 8 -> ['ARO', 'EO', 'GA', 'HHO', 'MPA', 'PSO', 'SMA', 'ea']
```

**This is the most important number in Phase 2.** On 10 of 24 problems, always
picking MPA already matches the oracle — a perfect selector would gain nothing
there. The median normalised gap is exactly 0. The entire opportunity sits in a
handful of problems, chiefly F20 Schwefel (0.3010), F19 GriewankRosenbrock
(0.2202) and F4 BucheRastrigin (0.2039).

Eight distinct algorithms appear as VBS somewhere, so the oracle is not
concentrated — but the margin it wins by is usually negligible.

## 6. Result 4 — no learned selector beats SBS

`phase2_lopo.py`, leave-one-problem-out over 24 folds. SBS is recomputed per
fold from training rows only, so it never sees held-out data.

| Selector | Mean norm | Median | Mean rank | VBS hits | Gap closed vs SBS |
|---|---|---|---|---|---|
| VBS (oracle) | 0.0000 | 0.0000 | 1.250 | 24 | — (target) |
| SBS (fold-aware) | 0.0679 | 0.0001 | 5.292 | 9 | — (baseline) |
| Random | 0.2154 | 0.0534 | 11.167 | 3 | −217.4% |
| 1-NN | 0.0794 | 0.0001 | 6.042 | 10 | −17.0% |
| 3-NN | 0.0567 | 0.0003 | 5.333 | 8 | +16.5% |
| RF-regression | 0.0737 | 0.0003 | 6.125 | 8 | −8.6% |

Wilcoxon signed-rank vs SBS, paired over the 24 problems:

```
  1-NN             statistic=np.float64(40.0) p=np.float64(0.700703412251682)   better on 6, worse on 7, tied on 11
  3-NN             statistic=np.float64(18.0) p=np.float64(1.0)                 better on 4, worse on 4, tied on 16
  RF-regression    statistic=np.float64(3.0)  p=np.float64(0.22491588401596185) better on 1, worse on 4, tied on 19
```

**No selector beats SBS significantly.** Only 3-NN is ahead on the mean at all
(+16.5%), and its p-value is exactly 1.0 — it ties SBS on 16 of 24 problems.
Two selectors are worse than the baseline. This is the Phase 2 negative result.

## 7. Result 5 — the features are not the problem

An in-sample diagnostic (added later, `phase2_lopo.py`) trains the RF on every
row *including the held-out one*, so it may memorise the answer:

| Selector | Mean norm | Mean rank | VBS hits | Gap closed |
|---|---|---|---|---|
| RF in-sample (ceiling) | 0.0016 | 2.708 | 15 | **+97.7%** |

A model allowed to memorise closes 97.7% of the gap. So the 9 landscape
features **do** carry enough signal to identify the right algorithm. The
failure in Section 6 is one of **transfer**, not representation: what the model
learns on 23 functions does not carry to a 24th, unseen function.

This is a diagnostic, not a selector — it is excluded from the significance
testing and cannot be deployed.

## 8. Known limitations

1. **Single dimension.** `dim=10` only. Dims 5 and 20 are a planned expansion.
2. **Single problem seed.** `problem_seed=42`; one landscape realisation per
   function.
3. **24 rows for a 22-way choice.** Nearly as many classes as training
   examples. This motivated Phase 2c.
4. **F5 LinearSlope is non-discriminative.** Its optimum lies on the search
   boundary, so many algorithms tie at `-2.0435e+02`, inflating apparent
   agreement.
5. **The `dimension` feature is constant** at 10 and carries no signal.
   `phase2_lopo.py` drops zero-variance columns, so it does no harm.

## 9. Conclusions

1. Algorithms differ significantly across BBOB (Friedman p ≈ 8.4e-44).
2. **MPA is the SBS** at mean rank 4.500.
3. **The headroom above MPA is small** — mean normalised gap 0.0407, and MPA
   already matches the oracle on 10 of 24 problems.
4. **No learned selector beats MPA significantly** on 24 training rows.
5. The bottleneck is **transfer across functions**, not feature quality — the
   in-sample ceiling reaches 97.7%.

Conclusion 4 is a genuine negative result and is reported as such. Conclusion 5
is what redirected the work: more rows (Phase 2c), rather than richer features,
became the next lever.

## 10. Reproducing

```bash
python phase2_build_matrix.py     # ~3 hours, writes results/phase2_matrix.json
python phase2_analyse_matrix.py   # seconds, ranking + Friedman
python phase2_lopo.py             # LOPO selector evaluation
```

`results/` is gitignored; the matrix is not in version control.

## 11. Related

- `docs/phase0_decision.md` — the equal-budget protocol
- `results/phase1b_screening.txt` — algorithm health screen
- `results/phase2_lopo.txt` — full LOPO output
- `docs/phase3_preregistration.md` — Phase 3b pre-registration
