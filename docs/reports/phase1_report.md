# Phase 1 — Baseline Comparison: PSO vs GA

Status: **complete**. Script: `phase1_baseline_comparison.py`.
Figures below are literal output from a re-run on 2026-09-17. Re-run again
on 2026-10-01: the output was identical. Phase 1 uses its own fixed problem
(`BBOB_F1_Sphere`, dim 10, seed 42), so the later matrix rebuilds do not
affect it.

## Purpose

The smallest defensible experiment: two algorithms, one problem, real
statistics. Establishes that the measurement pipeline produces a statistically
sound comparison before scaling to 22 algorithms × 24 problems.

No meta-learning, no surrogate, no LLM designer.

## Setup

| Parameter | Value |
|---|---|
| Problem | `BBOB_F1_Sphere(dim=10)` — unimodal, separable |
| Algorithms | PSO, GA |
| Config | `population_size=50`, `max_generations=100` |
| Runs | 30 per algorithm |
| Test | `scipy.stats.mannwhitneyu`, two-sided, independent |
| Problem seed | 42 |
| PSO seeds | 1000–1029 |
| GA seeds | 2000–2029 |

**Disjoint seed sets** for the two algorithms, so the samples are genuinely
independent — Mann-Whitney U is not a paired test and assumes independence.

Seeds travel through the config dict: mealpy seeds its own
`np.random.Generator` from the seed passed to `solve()`, so the global numpy
RNG does not control it.

## Results

```
  algorithm   n_runs           mean            std         median            min            max
  ---------------------------------------------------------------------------------------------
  PSO             30   7.278935e-09   1.593999e-08   9.340161e-10   4.843016e-11   7.851993e-08
  GA              30   5.952483e-02   2.921580e-02   5.796531e-02   1.600463e-02   1.402772e-01
```

## Statistical comparison

```
  test                 : scipy.stats.mannwhitneyu (two-sided, independent samples)
  n1 (PSO), n2 (GA)    : 30, 30
  U statistic          : np.float64(0.0)
  p-value              : np.float64(3.019859359162157e-11)
  rank-biserial r      : np.float64(1.0)
  alpha                : 0.05

  significant at alpha=0.05? YES (p = 3.01986e-11 < 0.05)
  direction            : PSO reached lower fitness more often
```

## Interpretation

**PSO beats GA decisively on F1 Sphere.** `U = 0.0` with `r = 1.0` means
complete separation — every one of the 30 PSO runs finished below every one of
the 30 GA runs. The medians differ by roughly seven orders of magnitude
(9.34e-10 vs 5.80e-02).

This is the expected result on a trivial unimodal separable function and is
reported as a **pipeline validation**, not a research finding: it confirms
seeding, budget enforcement and the statistical test all behave correctly.

## Limitations

1. **One problem, two algorithms.** Says nothing about cross-problem behaviour.
2. **An easy problem.** F1 Sphere is the simplest function in BBOB; the result
   does not generalise.
3. **Nothing is persisted.** The script prints only — no JSON or text output is
   written, so these figures require a re-run to reproduce.

## Reproducing

```bash
python phase1_baseline_comparison.py
```

Roughly one minute. Fully seeded, so output is deterministic.
