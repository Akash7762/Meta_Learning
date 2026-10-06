# Project plan

Written 2026-10-01. This file lists every phase of the project from start to
end: what each one asks, what it produced, and what is still to do. Completed
phases link to their reports, where every figure is traced to a result file.
Planned phases say where the plan comes from and what is not decided yet.

Results quoted here are from the rebuilt data only: the 72-row evaluation
matrix and the 720-row training matrix.

## The research question

From `docs/phase0_decision.md`:

> **Given the landscape features of an unseen optimisation problem, which of the
> mealpy metaheuristics should be selected to solve it?**

The registry now holds 21 mealpy algorithms plus the built-in `ea`, 22 in
total, after Phase 1b removed three that do not optimise.

## Overview

| Phase | Question | Status | Report |
|---|---|---|---|
| 0 | What exactly is the research question, and is the code consistent with it? | done | [phase0_report.md](phase0_report.md) |
| 1 | Does the measurement pipeline give a sound comparison on one problem? | done | [phase1_report.md](phase1_report.md) |
| 1b | Which algorithms actually optimise? | done | [phase1b_report.md](phase1b_report.md) |
| 2 | Does one algorithm win across all problems? Can a selector beat it? | done | [phase2_report.md](phase2_report.md) |
| 2c | Build a larger training matrix for selectors | done | [phase2c_report.md](phase2c_report.md) |
| 3 | Does a rank-target, single-forest selector beat the single best solver? | done | [phase3_report.md](phase3_report.md) |
| 3b | Does that result hold in a pre-registered test on independent data? | done | [phase3b_report.md](phase3b_report.md) |
| 4 | Meta-learner over the full grid | **not started, needs a plan** | — |
| 5 | LLM ablations | **not started, needs a plan** | — |
| Write-up | Thesis chapters from the phase reports | not started | — |

## Completed phases

### Phase 0: scope and configuration

Fixed the research question as algorithm selection, with EA operator tuning
kept as a secondary experiment that pins `algorithm="ea"`. Found that the
default code path silently ran a single hardcoded algorithm, and fixed it.
Set the budget rule every later phase uses: equal **evaluations**
(`max_fe=5000`, `population_size=50`), not equal generations.
Check: `test_phase0_config_consistency.py`.

### Phase 1: baseline comparison

PSO against GA on `BBOB_F1_Sphere` (dim 10, seed 42), 30 seeded runs each,
Mann-Whitney U test. PSO won every run: `U=0.0`, `p=3.02e-11`, `r=1.0`. This
validates the pipeline; it is not a research finding.
Script: `phase1_baseline_comparison.py`.

### Phase 1b: algorithm health screen

Screened 21 algorithms on F1 Sphere: 2 excellent, 8 moderate, 11 poor, 0
failed. Removed `CMA_ES`, `AOA` and `DE` from the registry, because a 10× larger
budget does not improve their results.
Script: `phase1b_algorithm_screening.py`. Output: `results/phase1b_screening.{txt,json}`.

### Phase 2: evaluation matrix and ranking

Built the evaluation matrix: 24 BBOB functions × dims 5, 10, 20 = 72 rows,
seed 100, 22 algorithms, 15 runs per cell, 1584 cells, 8.45 hours.

- The algorithms differ strongly (Friedman p ≈ 6.27e-139).
- EO (mean rank 4.722) and MPA (4.847) are jointly strongest and cannot be told
  apart (p=0.5894).
- The best algorithm depends on dimension: MPA leads at d5, EO at d20.
- The single best solver (SBS) already matches the oracle on 11 of 72 rows.
- Within this matrix, no selector beats SBS significantly.

Scripts: `phase2_build_matrix.py`, `phase2_analyse_matrix.py`, `phase2_lopo.py`.
Output: `results/phase2_matrix.{json,csv}`, `results/phase2_lopo.{txt,json}`.

### Phase 2c: training matrix

Built the selector training matrix: 24 functions × dims 5, 10, 20 × seeds
42–51 = 720 rows, 5 runs per cell, 15,840 cells, 31.25 hours. All 9 features
vary. The seeds do not overlap with the evaluation matrix, so the two form an
independent split. On this matrix, no selector beats SBS (RF-regression
−22.9%, p=0.356). The in-sample ceiling reaches +96.0%, so the features carry
the information, but it does not transfer to unseen functions.
Scripts: `phase2c_expand_matrix.py`, `phase2_lopo.py --matrix results/phase2c_matrix.json`.
Output: `results/phase2c_matrix.json`, `results/phase2c_lopo.{txt,json}`.

### Phase 3: learned selector and ablation

One random forest over (features + algorithm one-hot) trained to predict rank,
tested leave-one-function-out on the training matrix: −0.3% gap closed,
p=0.677, a tie with SBS. A 2×2 ablation of target (rank or normalised score)
against architecture (one forest or 22) found no design that beats SBS; the
best cell is +0.8%.
Scripts: `phase3_selector_validation.py`, `phase3_ablation.py`.
Output: `results/phase3_selector.{txt,json}`, `results/phase3_ablation.json`.

### Phase 3b: pre-registered confirmation

Method fixed in advance in `docs/phase3_preregistration.md`, with the data
change logged before the run. Trained on the 720-row matrix, evaluated on the
independent 72-row matrix: +2.1% gap closed, p=0.526. The pre-registered rule
requires p < 0.05 and a positive gap, so **the Phase 3 result does not
replicate**.
Script: `phase3b_holdout_confirm.py`. Output: `results/phase3b_holdout_confirm.{txt,json}`.

### Main finding so far

From features alone, a selector trained on other functions does not beat
always running the single best solver (MPA or EO) on an unseen function. The
features do identify the best algorithm for functions the model has seen. What
fails is transfer.

## Phases still to do

### Phase 4: meta-learner over the full grid

**Where the plan comes from.** The Phase 3 brief named it "meta-learner over the
full grid", to come after Phase 3 was confirmed. No project file defines it
further.

**Not decided yet: which approach.** The project has two candidates:

1. `EA_Designer` in `aad/meta_learner.py`. It tries candidate algorithms by
   running them on the problem ("Evaluate each by running the EA on the target
   problem N times", from its docstring) and can use a surrogate to screen
   candidates. It does not depend on transfer across functions, so Phase 3's
   result does not rule it out.
2. The global surrogate in `user_model_runner.py` (its own "Phase 4"). It
   predicts how every algorithm will do, much like the Phase 3 selector. Its
   docstring already warns that predictions for algorithms it has not seen are
   "extrapolation, not evidence". Expect it to struggle in the same way.

**Rules that carry over from Phases 0–3b.** These are recommendations drawn
from the earlier results, to be confirmed in the Phase 4 plan.

- **Baseline.** Phase 4 must beat always running MPA or EO.
- **Equal budget.** `EA_Designer` defaults to `population_size=10`,
  `generations=5` and `runs_per_eval=3`, so its first generation alone runs
  at least 30 full algorithm runs, against one run for the baseline. Compare
  both at the same total number of function evaluations.
- **No cross-problem transfer.** Learning within one problem is fine. Reusing
  knowledge from other problems (for example the persistent memory bank or
  surrogate warm-starting, both reverted earlier) is the kind of transfer
  Phase 3b found does not work.
- **Same evaluation discipline.** Leave-one-function-out where training is
  involved, function-mean aggregation before testing, and the 72-row matrix as
  the evaluation set.

**Steps.**

1. Choose between candidates 1 and 2.
2. Write the Phase 4 plan: question, baseline, budget, success rule.
3. Pre-register it, as in Phase 3b, before any result is seen.
4. Run it, write `docs/phase4_report.md`, commit and back up.

### Phase 5: LLM ablations

**Where the plan comes from.** The Phase 3 brief named it "LLM ablations".
`docs/phase0_decision.md` §4 states that `LLMDesigner` "is evaluated as a
*selector* for the primary question in a later phase."

**What exists.** `LLMDesigner` in `aad/llm_designer.py` proposes an algorithm
from a text description of the problem. Its default model is
`gemini-3.6-flash`; without an API key it falls back to rule-based heuristics.

**Not decided yet.** Which comparisons count as the "ablations", which
problems it is tested on, and how its picks are scored. One natural design,
not yet agreed: score the LLM's picks on the same independent 72-row matrix,
against the same SBS baseline, so its result is directly comparable with
Phase 3b. LLM outputs are not seeded the way the other phases are, so the plan
will need a way to make runs repeatable.

**Steps.** Same as Phase 4: plan, pre-register, run, report, commit.

### Write-up

Turn the phase reports into thesis chapters. The reports already carry every
figure with its source, so the write-up should quote them rather than
re-derive numbers. Caveats that must travel with the results are listed in
`docs/README.md`.

## Options considered, not decided

These came up in discussion. None is scheduled.

| Option | What it would do | Cost |
|---|---|---|
| Raise training runs from 5 to 10 | At 5 runs, about 1 training label in 6 is off by more than 1%; at 10 runs that drops to under 3% (see the Phase 2c report) | about 31 hours as a top-up; needs a small code change, because the build currently refuses to resume with a different run count |
| Confidence-gated selector | In Phase 3b the selector won more often than it lost but lost more per miss; a selector that only departs from SBS when confident might help | needs fresh data to test fairly, because the current matrices have been used for the confirmatory test |
| More diverse problem families | Every test is bounded by 24 BBOB functions | new problem suite and new matrices |

## Open technical items

| Item | Status, checked 2026-10-01 |
|---|---|
| Operator-dimension masking (`docs/phase0_decision.md` §7, item 6) | still open: `aad/meta_learner.py` and `aad/config_space.py` do not mask operator keys when `algorithm != "ea"`. Relevant to Phase 4 option 1. |
| `load_dotenv()` at import in `aad/llm_designer.py:25` | still present, so a bare `LLMDesigner()` picks up a key from `.env`. Relevant to Phase 5. |
| `user_model_runner.py` docstring says 24 algorithms | the registry has 21 mealpy algorithms plus `ea`. Check before using it in Phase 4. |
| `pytest` missing, stray `pydantic` import, empty-key bug (Phase 0 §7c) | fixed |

## Rules for every phase

- The method is written down before results are seen. Confirmatory tests are
  pre-registered; any change is logged in the append-only section before the
  run.
- Budgets are equal in function evaluations.
- Rows from the same function are held out together, and averaged before any
  significance test.
- `results/phase2_matrix.json` is the authoritative source for SBS/VBS figures.
  `results/phase2c_matrix.json` is for training only.
- When a phase completes, its report is written from the result files, then
  committed. The backup at `~/meta_learning_backup/` is re-synced after long
  builds (see `results/MANIFEST.md`).

## Run order for the completed pipeline

| # | Command | Needs |
|---|---|---|
| 1 | `python test_phase0_config_consistency.py` | — |
| 2 | `python -m pytest tests/test_aad.py -q` | — |
| 3 | `python phase1_baseline_comparison.py` | — |
| 4 | `python phase1b_algorithm_screening.py` | — |
| 5 | `python phase2_build_matrix.py` | — |
| 6 | `python phase2_analyse_matrix.py` | 5 |
| 7 | `python phase2_lopo.py` | 5 |
| 8 | `python phase2c_expand_matrix.py` | — |
| 9 | `python phase2c_expand_matrix.py --verify-only` | 8 |
| 10 | `python phase2_lopo.py --matrix results/phase2c_matrix.json --out-stem results/phase2c_lopo` | 8 |
| 11 | `python phase3_selector_validation.py` | 8 |
| 12 | `python phase3_ablation.py` | 8 |
| 13 | `python phase3b_holdout_confirm.py` | 5 and 8 |

## A note on numbering

`user_model_runner.py` has its own internal "Phase 0" to "Phase 6", describing
the steps of that one interactive script. Those are unrelated to the research
phases in this file. In particular, its "Phase 4" (global surrogate) and
"Phase 5" (LLM pick) are inputs to the decisions above, not the same thing as
research Phases 4 and 5.
