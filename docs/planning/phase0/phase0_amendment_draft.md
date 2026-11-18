# Phase 0 amendment (DRAFT, not approved)

Drafted 2026-11-12. Nothing has been changed in code or data. Once approved,
this text is appended as a dated section to `docs/planning/phase0/phase0_decision.md`; the
original decision of 2026-09-14 is not edited. Items marked **[decide]** need
the user's choice; the proposed option is listed first.

## Why amend Phase 0

The project goal is now: *give the program a problem; it extracts the
problem's features and recommends the algorithm **and its settings** that
perform best.* The original Phase 0 fixed the question as choosing an
**algorithm** at fixed settings (population 50, 5,000 evaluations). Phases
1 to 4 answered that question, and their results stay valid for it.

## A1. Research question

> **Given the landscape features of an unseen optimisation problem, which
> algorithm, with which settings, should be used to solve it?**

This combines algorithm selection with settings choice. The original
algorithm-only question becomes a sub-question, already answered at
population 50 by Phases 2 to 4.

## A2. Settings that are searched **[decide]**

**Proposed:** population for every algorithm; operators for `ea` only, the
one algorithm that reads them (`aad/evolutionary_algorithm.py:75-77`; mealpy
algorithms ignore them, confirmed by identical same-seed runs).

| Algorithm | Settings | Count |
|---|---|---|
| 21 mealpy algorithms | population {20, 50, 100} | 63 |
| `ea` | population {20, 50, 100} × selection {tournament, roulette} × crossover {sbx, uniform} × mutation {gaussian, polynomial} | 24 |
| **Total per problem** | | **87** |

`max_generations` is not searched; it follows from the budget (A3).

*Alternative:* the README's full grid including generations {50, 100, 200}:
261 settings per problem, as `grid_build_matrix.py` currently builds.

## A3. Budget rule **[decide]**

**Proposed: fair.** Every run gets `max_fe = 5000` objective evaluations,
whatever its settings. Mealpy already supports this. `ea` needs either
`max_fe` support (a code change) or matched generations (249 / 99 / 49 for
population 20 / 50 / 100). Reason: a setting must not win just by spending
more evaluations (Phase 1b: equal generations gave SA about 150 evaluations
against about 5,050 for others).

*Alternative: unequal.* Budget = population × generations, as the README
had; real evaluation counts recorded per run. Settings are then not directly
comparable.

## A4. Operator masking (Phase 0 item 6, open since 2026-09-14)

Implemented as part of this amendment: operator keys exist only when
`algorithm == "ea"`. `EA_Designer` stops mutating them for other algorithms,
and the surrogate's encoding no longer carries them as noise columns.
Needed for A2.

## A5. Landscape features

- **Main set:** pflacco 1.2.2, computed in the separate `.venv-pflacco`
  (commit `34e91b9`; problem instances verified bit-identical to the main
  environment). Feature sets: `ela_meta`, `ela_distribution`, `ela_level`,
  `nbc`, `dispersion`, `information_content`, `pca` (62 features on a
  dim-10 test problem).
- **Sampling:** Sobol, with pflacco's fixed internal seed 50 (it ignores the
  `seed` argument for Sobol). Sample size = the power of 2 nearest to
  50 × dimension: 128 / 256 / 512 / 1,024 points for dimensions 2 / 5 / 10 /
  20. Cost = sample size in evaluations.
- **One method for all:** the same sets, sampling and sizes for every
  training and test problem.
- **Old 9 features:** kept only as a comparison set, documented by what
  they compute (`fitness_variance` = std / |mean|, `modality` = skewness,
  `ruggedness` = a spread statistic of randomly ordered values), with their
  real cost of 600 + 50 × dimension evaluations.

## A6. Baseline and success rule

- **Baseline:** the single best configuration (algorithm + settings with the
  best mean rank on training data, test function left out), at the same
  budget. The per-problem best (VBS) is reported as a reference.
- **Test:** unseen functions (leave-one-function-out); each function's rows
  averaged before testing; one pre-registered Wilcoxon test on a test set
  used once.
- **Success:** p < 0.05 and a positive gap closed, as in Phases 3b and 4.
- **Safety rule in the final program:** if the model is not confident, fall
  back to the baseline configuration.

## A7. Data **[decide: runs, dimension 2]**

- Training problems: seeds 42 to 51; test problems: seed 100 (no overlap,
  guarded by `assert_no_leakage`).
- Runs per cell: **10 for training** (at 5 runs, 83.3% of labels are within
  1% of the true best; at 10 runs, 97.2%), **15 for testing**.
- Dimensions: 5, 10, 20 as in the existing data; **2** only if added to the
  data as well.
- A new settings dataset is needed; `grid_build_matrix.py` is adapted to A2
  and A3 before it runs.

## A8. What stays valid

| Phase | Status after the amendment |
|---|---|
| 1, 1b | valid unchanged |
| 2, 2c | valid as the population-50 slice of the new space (`ea` there used 5,050 evaluations, recorded) |
| 3, 3b, 4 | valid answers to the algorithm-only question |
| 5 | paused; may later propose algorithm + settings |

## A9. Verification

`test_phase0_config_consistency.py` gains checks for operator masking and
the budget rule; all existing checks must still pass.
