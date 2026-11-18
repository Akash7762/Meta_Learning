# Configuration study plan (DRAFT, not approved)

Drafted 2026-10-09. Nothing in this plan has been run and no code has been
changed. No existing dataset is modified: new runs go into new files.

## In plain words

So far every algorithm ran with one setting: population 50 and, for `ea`,
tournament selection, SBX crossover and Gaussian mutation. This study asks
whether **other settings** do better, and whether the best setting depends on
the problem. It keeps the rule every phase uses: every run gets the same
5,000 function evaluations, so no setting wins just by getting more
evaluations.

## 1. The questions

- **Q1, population.** For each algorithm, is population 50 the best choice,
  or do 20 or 100 do better? Does picking the algorithm **and** its population
  per problem leave more room to gain than picking the algorithm alone?
- **Q2, `ea` operators.** Which selection, crossover and mutation work best
  for `ea`? Does the answer depend on the problem, and does a tuned `ea` catch
  up with the mealpy algorithms?

## 2. What the code allows (checked 2026-10-09)

| Fact | Where | Consequence |
|---|---|---|
| Selection, crossover and mutation are read only by `ea` | `aad/config_space.py:32-35` | for the 21 mealpy algorithms only population can vary |
| Mealpy algorithms get the population as `pop_size` and the budget through `Termination(max_fe=...)` | `aad/mealpy_algorithms.py:212-263` | population 20 or 100 runs at exactly 5,000 evaluations |
| The epoch count is calibrated per (algorithm, population) by measuring the real cost | `aad/mealpy_algorithms.py:293-332` | schedule-based algorithms still anneal correctly at every population |
| `ea` does not read `max_fe`. It evaluates its starting population, then one population per generation | `aad/evolutionary_algorithm.py`; `phase2_build_matrix.py:18-20` | 50 × (100 + 1) = 5,050 evaluations today; other populations need matched generations (section 3.2) |
| `max_generations` changes the budget: population × generations ranges from 1,000 to 20,000 | `aad/config_space.py:58-65` | generations are **not** searched; they follow from the budget |

## 3. Design

### 3.1 Configurations

| Study | Configurations | Count |
|---|---|---|
| A, population | 22 algorithms × population {20, 50, 100}; `ea` with its current operators | 66 |
| B, `ea` operators | `ea` × selection {tournament, roulette} × crossover {SBX, uniform} × mutation {Gaussian, polynomial} × population {20, 50, 100} | 24 |
| Both, without duplicates | the 3 `ea` settings with current operators appear in both | **87** |

The 21 mealpy configurations at population 50 already exist in both matrices
and are reused, read only. That leaves **66 new configurations per problem**:
the 42 mealpy ones at populations 20 and 100, and all 24 `ea` ones (see 3.2
for why `ea` at population 50 is run again).

### 3.2 Equal budget for `ea`

Proposed: set `ea`'s generations so it spends exactly 5,000 evaluations:
5,000 / population − 1, which gives 249 at population 20, 99 at 50 and 49
at 100. No code change is needed, and the count is checked by counting
evaluations in step 1.

The existing `ea` cells used 100 generations (5,050 evaluations, 1% more, as
recorded in the matrices). So all 24 `ea` configurations are run fresh, with
an identical budget.

Alternative: add `max_fe` support to `ea` (a code change), so that every
algorithm stops the same way.

### 3.3 Which problems, and how many runs

| Stage | Rows | Runs per cell | Runs | Time |
|---|---|---|---|---|
| **1, now** | the 72 evaluation rows (24 functions × dims 5, 10, 20, seed 100) | 15 | 72 × 66 × 15 = 71,280 | about 25 h |
| 2, only if stage 1 shows a gain worth chasing | the 720 training rows (seeds 42 to 51) | 5 | 720 × 66 × 5 = 237,600 | about 94 h |

The times scale the measured builds (8.45 h for the evaluation matrix, 31.25
h for the training matrix, both 22 configurations) by 66 / 22 = 3. They are
estimates: population 20 runs more generations, which may cost more Python
overhead per evaluation. Step 1 measures the real time on one row.

Same problem seeds and run seeds as the matrices (run seeds 1000 to 1014), so
the new cells sit on exactly the same problem instances and are comparable
with the reused population-50 cells.

### 3.4 What stage 1 reports

Descriptive, with each function's 3 rows averaged before any comparison, as
in every earlier phase:

- **Headroom.** The per-problem best configuration over all 87 against the
  best algorithm at population 50 (22) and against the SBS. That shows how
  much choosing the population and operators could add on top of choosing the
  algorithm.
- **Best default population** for each algorithm, by mean rank, and whether it
  changes with dimension.
- **`ea`:** the best operator setting overall, how much it beats the current
  setting, and where the best `ea` ranks among the 22 algorithms.

Stage 1 makes no selection claims. If the headroom is real, stage 2 trains a
configuration selector, pre-registered like Phase 3b.

## 4. Code changes, after approval only

1. A new builder script that runs one configuration per cell, with the cell
   key `{function}|s{seed}|d{dim}|{algorithm}|p{population}|{selection}-{crossover}-{mutation}`
   (operators shown as `-` for mealpy). It reuses `run_cell` and the seeding
   from `phase2_build_matrix.py`, writes a crash journal plus an atomic
   snapshot, and resumes after an interruption.
2. A smoke test on one row: every configuration runs, budgets are counted
   (`ea`: exactly 5,000), and the time per run is measured.
3. Tests for the cell keys and the `ea` generation formula.

## 5. Steps and stop points

| # | Step | Stop and ask? |
|---|---|---|
| 0 | You approve this plan | yes |
| 1 | Builder, tests, smoke test on one row with timing; show the diff | yes |
| 2 | Build stage 1, about 25 h, in the background and resumable | yes |
| 3 | Stage 1 analysis and report | yes, commit |
| 4 | Decide on stage 2 | yes |

## 6. Files

**Read only:** `results/phase2/phase2_matrix.json` (reused population-50 cells) and
`results/phase2c/phase2c_matrix.json` (stage 2 only).

**New:** `results/config_matrix_eval.json` (stage 1), its `.jsonl` journal
(local only), the builder script, a test file, and a report.

## 7. Not in this plan

- Searching `max_generations` (it breaks the equal budget).
- Operator settings for mealpy algorithms (mealpy does not use them).
- Mutation and crossover **rates** (the project's configuration space has
  only operator types).
- A full grid of 1,584 nominal combinations: most of it would duplicate
  runs, because mealpy ignores the operator keys.

## 8. Relation to Phase 5

Phase 5 is paused on Gemini overload, with 2 of 480 development calls saved.
This study uses only the CPU and no API, so it can run while Phase 5 waits.

## 9. Decisions needed

1. Where it fits: as **Phase 6**, or another name?
2. Scope: studies A and B (66 new configurations per problem), or only one
   of them?
3. `ea` budget: matched generations (no code change, recommended), or add
   `max_fe` to `ea`?
4. Stage 1 on the 72 existing evaluation rows at 15 runs (about 25 h)?
5. Stage 2 (about 94 h): decide after stage 1, or plan it now?
