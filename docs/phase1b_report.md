# Phase 1b — Algorithm Health Screen

Status: **complete**. Script: `phase1b_algorithm_screening.py`.
Output: `results/phase1b_screening.{txt,json}`. Figures below are quoted from
that cached output.

## Purpose

Before building a 22-algorithm performance matrix, establish which algorithms
actually optimise. A crash-based smoke test is insufficient: several mealpy
implementations run without error, return finite values, and never improve.

## Setup

| Parameter | Value |
|---|---|
| Problem | `BBOB_F1_Sphere(dim=10, seed=42)` |
| Algorithms | 21 (mealpy only; `ea` excluded — not seedable) |
| `population_size` | 50 |
| `max_fe` | 5000 (equal evaluation budget) |
| Seeds per algorithm | 1000, 1001, 1002 |

Thresholds on median fitness (Sphere optimum = 0.0):
`excellent < 1e-06`, `moderate < 0.01`, `POOR >= 0.01`.

## Results

```
  algorithm verdict             median           min           max   sec/run
  --------------------------------------------------------------------------
  PSO       excellent       3.8026e-09    1.3677e-09    1.0808e-08      0.37
  EO        excellent       4.0633e-07    5.5781e-08    8.5700e-07      0.48
  SHADE     moderate        2.4651e-06    2.0325e-06    6.7458e-06      1.18
  L_SHADE   moderate        7.3852e-06    3.4526e-06    8.1830e-06      1.12
  JADE      moderate        3.1291e-05    2.9110e-05    7.6058e-05      1.54
  MPA       moderate        5.3832e-05    1.5072e-05    4.0986e-04      0.29
  MVO       moderate        5.2051e-04    3.7159e-04    8.6136e-04      0.76
  RIME      moderate        1.8861e-03    4.9565e-04    3.9716e-03      0.29
  GWO       moderate        2.0159e-03    4.1747e-04    2.8437e-03      0.55
  SMA       moderate        4.8067e-03    2.7706e-03    5.1832e-03      1.98
  ARO       POOR            2.6220e-02    2.0845e-02    2.0170e-01      0.53
  HBA       POOR            2.7588e-02    1.4201e-02    4.4608e-02      0.51
  SA        POOR            3.1864e-02    2.7138e-02    3.2753e-02      9.38
  ABC       POOR            4.4414e-02    2.6994e-02    5.2052e-02      0.68
  GA        POOR            7.5727e-02    3.7303e-02    1.3425e-01      0.85
  BBO       POOR            2.7146e-01    1.0450e-01    3.4986e-01      1.21
  WOA       POOR            2.8133e-01    1.2523e-01    4.8929e-01      0.28
  SCA       POOR            5.6468e-01    5.2920e-01    6.7395e-01      1.12
  GJO       POOR            1.6622e+00    1.3203e+00    2.1983e+00      0.62
  HHO       POOR            8.4488e+00    4.5354e+00    8.5540e+00      0.37
  AO        POOR            2.9659e+01    2.8909e+01    2.9825e+01      0.47
```

```
  excellent (< 1e-06) :  2  ['EO', 'PSO']
  moderate  (< 0.01) :  8  ['GWO', 'JADE', 'L_SHADE', 'MPA', 'MVO', 'RIME', 'SHADE', 'SMA']
  POOR      (>= 0.01): 11  ['ABC', 'AO', 'ARO', 'BBO', 'GA', 'GJO', 'HBA', 'HHO', 'SA', 'SCA', 'WOA']
  FAILED              :  0  []
```

**No algorithm failed outright.** "POOR" does not mean broken — on a trivial
unimodal problem it means the algorithm's default settings do not refine
finely, which is a real property of the algorithm as configured.

## Consequence — three algorithms removed from the registry

Identified by **budget insensitivity**: a 10× evaluation budget does not improve
the result. All three construct and run without error, so a crash-based test
does not catch them.

| Algorithm | Evidence |
|---|---|
| `CMA_ES` | Covariance matrix collapses (‖C‖ 2.97 → 0.093) while step size never adapts (‖σ‖ ≈ 3.0), freezing 3.34 from the optimum. Identical at 5k/20k/50k evaluations. mealpy exposes no parameters beyond `epoch`/`pop_size`, so no adapter-level fix exists. |
| `AOA` | Returns `2.4756e+01` at every budget; never improves from its first epoch. |
| `DE` | Stagnates at ≈`7.3e-01`; 2,000 vs 20,000 evaluations differ only in the fifth decimal. |

`JADE` remains registered — a different class in the same module, unaffected.

Removed from both `MEALPY_REGISTRY` (`aad/mealpy_algorithms.py`) and
`MEALPY_ALGORITHM_KEYS` (`aad/config_space.py`) in commit `f051976`.

This leaves **22 algorithms** (21 mealpy + built-in `ea`) for Phase 2.

## Timing note

`SA` is the runtime outlier at 9.38 s/run — roughly 25× the fastest algorithm —
because it is a single-solution method spending ~1 evaluation per step. It
accounts for a disproportionate share of later matrix build times.

## Limitations

1. **One problem.** Screening on F1 Sphere only; an algorithm poor here may be
   strong on multimodal landscapes. This is a health check, not a ranking.
2. **3 seeds per algorithm.** Enough to detect a frozen optimiser, not enough
   to rank close competitors.
3. **`ea` not screened** — not seedable through the same path.

## Reproducing

```bash
python phase1b_algorithm_screening.py
```
