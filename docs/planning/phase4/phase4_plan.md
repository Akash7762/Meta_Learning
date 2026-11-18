# Phase 4 plan

Drafted and approved 2026-10-06, with the changes listed in the next section.
The binding version of the method is `docs/planning/phase4/phase4_preregistration.md`. Every
dataset used in Phases 0 to 3b stays read-only: Phase 4 reads
`results/phase2/phase2_matrix.json` and `results/phase2c/phase2c_matrix.json` and writes only
new files.

## Changes after approval

Decided 2026-10-06. Where the sections below disagree, this section wins.

1. **Only the existing datasets.** At the user's request, no new seed-200
   matrix is built (step 4 is dropped, and so is code change 2 in section 4).
   The test set is `results/phase2/phase2_matrix.json` (seed 100), which was also the
   Phase 3b test set.
2. **Development moves to the training matrix** (`results/phase2c/phase2c_matrix.json`),
   so the test set is not used to choose anything. That matrix stores 5 runs
   per cell, fewer than the 15-run budget, so a replay can ask for a run that
   is not there. It then gets a used run again, and every such draw is counted
   and reported as "reused". The test stage allows none.
3. **Seeds per row and replay.** Development: 1 replay per row over all 720
   rows, seed 10000 + row. Test: 30 replays per row, seed 4000 + row × 30 +
   replay. This replaces the shared seeds 4000 to 4029 of section 3.4, so rows
   do not all start from the same 30 random sequences.
4. **Two code changes found during step 1**, both in `aad/`:
   - the surrogate can run on 1 thread (`n_jobs=1`). With the default
     `n_jobs=-1`, 50 repeated predictions were not all identical (largest
     difference 1.42e-14), so exact repeatability needs 1 thread;
   - a candidate the surrogate skips reuses the prediction already made,
     instead of asking the unchanged surrogate again. Output was bit-identical
     on 48 replays, and about 22% faster.
5. **Surrogate-off variants are the same method.** With one search key, every
   candidate is a uniform random algorithm. With the surrogate off, every
   candidate is run, so v2 and v4 are both 15 random draws.

## In plain words

Phase 3b tried to **guess** the best algorithm from a problem's features, and
could not beat "always run the single best solver" (SBS: MPA or EO). Phase 4
tries a different idea: **try** algorithms on the problem itself, keep trying
the ones that look good, and keep the best answer found. The test is whether
this beats running the SBS with the same total number of function evaluations.

## 1. The question

> At an equal budget of function evaluations, does the meta-learner
> `EA_Designer`, which spends its budget trying algorithms on the problem,
> find better solutions than running the SBS?

`EA_Designer` (`aad/meta_learner.py`) learns only inside one problem, so it does
not depend on transfer between functions, which is what failed in Phase 3b.

Option 2, the global surrogate in `user_model_runner.py`, is left out. It
predicts performance from features, like the Phase 3 selector, so Phase 3b
already covers it.

## 2. What the code does today

Checked against the source on 2026-10-06.

| Fact | Where | Why it matters |
|---|---|---|
| Each candidate is run `runs_per_eval` times (default 3); no seed is passed | `aad/meta_learner.py:123-133` | Runs cannot be repeated exactly |
| The search space varies `population_size` (20, 50, 100) and `max_generations` (50, 100, 200); mealpy algorithms receive these, not `max_fe` | `aad/config_space.py:58-65`, comment at `:32-35` | Each run gets between 1,000 and 20,000 evaluations. This is the unequal-budget problem that Phase 2's `max_fe=5000` rule fixed (`phase2_build_matrix.py:9-20`) |
| With defaults, up to 180 full runs per problem: 10 configs × 3 runs, then 5 generations × 10 children × 3 runs | `aad/meta_learner.py:203-255` | Far more budget than a single SBS run |
| Landscape features are extracted on every call, 300 samples by default | `aad/meta_learner.py:191`, `aad/landscape_features.py:36` | Inside one problem every candidate gets the same feature values, so they cost evaluations but say nothing about which candidate is better |
| The surrogate's memory (`_mem_X`, `_mem_y`) is never cleared inside `evolve` | `aad/meta_learner.py:90-91, 216-217, 254-255` | Reusing one object across problems would carry knowledge over. Each problem needs a fresh object |
| A candidate the surrogate skips gets `prediction × 1.05`, commented "slightly pessimistic" | `aad/meta_learner.py:247` | Objective values can be negative: the lowest stored value is −1426.27, and 2,956 of the 23,760 runs in `results/phase2/phase2_matrix.json` are below zero. For a negative value, × 1.05 makes the guess look **better**, so a candidate that was never run can become the "best" with a made-up score |
| `evolve` returns the best config and its score; that score is a mean of runs, or a surrogate guess | `aad/meta_learner.py:221-278` | Phase 4 must score only runs that actually happened |
| Operator keys (`selection`, `crossover`, `mutation`) are ignored unless `algorithm == "ea"` | `aad/config_space.py:32-35` | Open item from Phase 0 §7. Not needed if the search is over algorithms only (section 3.1) |

## 3. Design

### 3.1 What the meta-learner searches over

No project file defines "meta-learner over the full grid". Proposed meaning:
all 22 algorithms, on every row of the matrix (24 functions × dims 5, 10, 20).

The search space is **the algorithm only**. Every candidate run uses the same
settings as the matrices (`phase2_build_matrix.py:220-234`):
`population_size=50` and `max_fe=5000`; `ea` uses `max_generations=100` and its
fixed operators, which comes to about 5,050 evaluations (1% more, as recorded
in the matrices). The reasons:

- searching `population_size` or `max_generations` breaks the equal-budget rule;
- operator settings only affect `ea`, and Phase 0 put operator tuning in a
  separate, secondary experiment.

### 3.2 Budget

Each method gets **k = 15 runs** per problem (15 × 5,000 = 75,000 evaluations).

| Method | How it spends the 15 runs |
|---|---|
| Meta-learner | Every run it performs counts. A candidate the surrogate skips costs nothing. If it stops with runs left, the leftover runs go to its current best algorithm |
| SBS baseline | The SBS run 15 times; keep the best |
| Oracle reference | The row's best algorithm (the VBS) run 15 times; keep the best |

The SBS is chosen from the training matrix by mean rank, leaving out the test
function, as in Phase 3b.

Why 15: the evaluation matrices store 15 runs per cell, so any way of
spending 15 runs can be looked up without reusing a stored run (section 3.4).

### 3.3 Score

Per row, the result is the best objective value found in the 15 runs. It is
put on a 0-to-1 scale so rows can be averaged: 0 = the best algorithm's median
on that row, 1 = the worst algorithm's median.

Gap closed = (SBS − meta-learner) / (SBS − oracle), on the mean of that score.

### 3.4 Replay instead of new runs

Every cell of an evaluation matrix stores 15 independent, seeded runs
(`phase2_build_matrix.py:248-294`). `results/phase2/phase2_matrix.json` has no missing
runs. When the meta-learner asks to run algorithm A on row r, the runner gives
it one of the stored runs of cell (r, A) that it has not used yet. That is the
same as running A with a fresh seed, because the stored runs are exactly such
runs.

What this gives:

- no optimiser runs for the meta-learner, the baseline or the oracle;
- exact repeatability;
- the baseline and the oracle come from the same runs the meta-learner draws from.

Each row is replayed R = 30 times with recorded seeds (proposed: 4000 to 4029),
and the 30 results are averaged.

**Cost is not known yet.** Most of the time goes into surrogate fits. One fit
plus one prediction on 15 rows took 0.147 s (measured 2026-10-06). A replay
needs several dozen of these, so Phase 4a could take hours as the code stands
(an inference, not a measurement). Step 2 times one problem first, and I ask
before anything over 2 minutes. If it is too slow, the first thing to try is
the surrogate's thread setting (`aad/surrogate_model.py:50`, `n_jobs=-1`),
after checking bit-for-bit that results do not change.

### 3.5 Which problems

| Set | File | Used for |
|---|---|---|
| Training | `results/phase2c/phase2c_matrix.json`: 720 rows, seeds 42 to 51, 5 runs per cell | Choosing the SBS, leave-one-function-out. Read only |
| Development | `results/phase2/phase2_matrix.json`: 72 rows, seed 100, 15 runs per cell | Phase 4a: checking the pipeline and choosing between the declared variants. Read only |
| Test | **new** `results/phase4_matrix.json`: 72 rows, fresh problem seed 200, 15 runs per cell | Phase 4b: the single pre-registered test |

Why a new test set: the seed-100 matrix was the test set for Phase 3b, and it
has been studied in detail since Phase 2. A fresh seed keeps the Phase 4 test
clean. Building it should take about 8.45 h, the measured time of the Phase 2
build, which has the same size. Seed 200 is not used anywhere else in the
project.

### 3.6 Phase 4a: development (exploratory)

Replay on the seed-100 matrix. Four declared variants, nothing else:

| | surrogate on | surrogate off |
|---|---|---|
| meta-population 5 | variant 1 | variant 2 |
| meta-population 10 (default) | variant 3 | variant 4 |

All use `runs_per_eval=1` and k = 15. The variant with the lowest mean score
over the 24 function means goes to Phase 4b. No claim is made from Phase 4a.

### 3.7 Phase 4b: pre-registered test

The chosen variant is run once on the new matrix.

- **Test:** Wilcoxon signed-rank on 24 function means (the rows of each
  function are averaged first), meta-learner against SBS, two-sided.
- **Success rule:** p < 0.05 **and** a positive gap closed, the same rule as
  Phase 3b. Otherwise the result is "does not beat SBS".
- **Reported, no claims:**
  - results by dimension;
  - functions where it is better, worse or tied;
  - how the 15 runs were spent across algorithms;
  - how often the final pick is the row's best algorithm;
  - how often the leftover rule was used.

## 4. Code changes, after approval only

Every change has a default that keeps today's behaviour, so Phases 0 to 3b
reproduce unchanged. The existing tests must still pass.

1. `aad/meta_learner.py`:
   - an optional evaluator hook in `evaluate_candidate` (replay or live), plus
     a log of every run;
   - an optional run budget in `evolve`: stop mid-generation when it is spent,
     keep going while it is not, with a cap of 50 generations;
   - an optional switch to skip feature extraction;
   - a skipped-candidate penalty that is worse for negative values too, for
     example `prediction + 0.05 × |prediction|`.
2. `phase2_build_matrix.py`: `--problem-seed` and `--out` flags. The defaults
   stay as they are, and it refuses to write to `results/phase2/phase2_matrix.json`
   unless the seed is 100.
3. New `phase4_replay.py`, the replay runner. It:
   - reads functions, dims and algorithms from the matrix's metadata, so `dim`
     stays a live variable;
   - seeds Python `random` and NumPy for every replay and stores the seeds;
   - builds a fresh `EA_Designer` for every replay.
4. Tests for the replay hook and the budget cap.

## 5. Steps and stop points

| # | Step | Stop and ask? |
|---|---|---|
| 0 | You approve this plan | yes |
| 1 | Code changes and tests; show the diff | yes |
| 2 | Phase 4a replay on the seed-100 matrix; time one problem first | if over 2 min |
| 3 | Write `docs/planning/phase4/phase4_preregistration.md` with the frozen design, and commit it **before** the new matrix exists | yes, commit |
| 4 | Build `results/phase4_matrix.json` (about 8.45 h), then back it up | yes |
| 5 | Run Phase 4b once | no |
| 6 | Write `docs/reports/phase4_report.md`, commit, back up | yes, commit |

## 6. Files

**Read only:** `results/phase2/phase2_matrix.json`, `results/phase2c/phase2c_matrix.json`, and
every other Phase 0 to 3b result file.

**New:**

- `results/phase4/phase4a_dev.{json,txt}`
- `results/phase4_matrix.{json,csv}`
- `results/phase4/phase4b_confirm.{json,txt}`
- `docs/planning/phase4/phase4_preregistration.md`
- `docs/reports/phase4_report.md`
- `phase4_replay.py`

## 7. Not in this plan

- Option 2, the global surrogate.
- Learning across problems: memory bank, warm start.
- Searching population size, generations or operators.
- Any change made after seeing the Phase 4b result.

## 8. What to expect

An earlier exploratory simulation on the training matrix tried the top
algorithms once each against SBS restarts at the same budget. The script is
kept outside the project and is not part of Phase 4. That simulation gave
+29.6% gap closed at 2 runs (p=0.2292) but only +0.2% at 5 runs. It informs
this design, and the pre-registration must say so.

Inference, not a result: at 15 runs a large gain is unlikely. "Does not beat
SBS" is a likely outcome, and it is still a finding. It would show that even
trying algorithms on the problem itself does not beat the SBS at an equal
budget.

All tests are bounded by 24 functions, so a small benefit cannot be ruled out
(see `docs/README.md`, caveat 2).

## 9. Decisions needed

1. Option 1 (`EA_Designer`), with option 2 left out. *Recommended.*
2. "Full grid" means 22 algorithms on all 72 rows, searching the algorithm
   only. *Recommended.*
3. Test set: a new seed-200 matrix (about 8.45 h). *Recommended.* Alternative:
   reuse the seed-100 matrix at no cost, but it was the Phase 3b test set.
4. Budget k = 15 runs, with the SBS's best of 15 as the baseline.
5. Optional extra: also compare with the SBS as one long run of 75,000
   evaluations. This needs new runs and would take hours.
