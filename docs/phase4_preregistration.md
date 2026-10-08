# Phase 4 pre-registration

Written 2026-10-08, after the development run (Phase 4a) and **before** the
test (Phase 4b). Committed before the test is run. Not to be revised
afterwards: any later change goes below the line at the end, before the run it
affects.

## The question

At an equal budget of 15 runs of 5,000 evaluations per problem, does the
meta-learner `EA_Designer` find better solutions than running the single best
solver (SBS) 15 times?

## Data

| Role | File | md5 |
|---|---|---|
| Choosing the SBS, and development (Phase 4a) | `results/phase2c_matrix.json`: 720 rows = 24 functions × dims 5, 10, 20 × seeds 42 to 51; 5 runs per cell | `fc090989610d07fa37461f880e589c2d` |
| Test (Phase 4b) | `results/phase2_matrix.json`: 72 rows = 24 functions × dims 5, 10, 20; seed 100; 15 runs per cell | `fb4c2a5efdb5d86b51da86e4afb6a2d8` |

No new data is created. Both files are read only, and `phase4_replay.py`
checks their md5 before and after every run.

**The test set is not fresh.** It was the Phase 3b test set, and its results
have been studied since Phase 2. Facts already known about it include that EO
and MPA are the strongest algorithms, and that the SBS rule below picks MPA on
69 rows and EO on 3 (identical to Phase 3b, checked row by row). No Phase 4
setting was chosen on this matrix: the variant was chosen on the training
matrix, and every other setting was fixed in `docs/phase4_plan.md` before any
Phase 4 result existed. A fresh seed-200 matrix was planned for the test. It
was dropped at the user's request on 2026-10-06, so that only existing data is
used.

An exploratory simulation on the training matrix informed the plan: trying
the top algorithms once each against SBS restarts at the same budget gave
+29.6% gap closed at 2 runs (p=0.2292) and +0.2% at 5 runs. Its script is
kept outside the project.

## Method, fixed

- **Meta-learner:** `EA_Designer` (`aad/meta_learner.py`), variant **v3**:
  `population_size=10` and `use_surrogate=True`, which are `EA_Designer`'s own
  defaults. It was chosen in Phase 4a by the rule fixed in the plan: lowest
  mean score over the 24 function means of the training matrix. Settings
  shared by all variants: search space = the 22 algorithms
  (`EXTENDED_CONFIG_SPACE["algorithm"]`), `runs_per_eval=1`,
  `generations=50` (a cap), `max_runs=15`, `use_features=False`,
  `surrogate_n_jobs=1`, and a fresh object for every replay.
- **Runs:** replayed from the matrix. A requested run of algorithm A on row r
  is a stored run of cell (r, A) not yet used in that replay, picked at
  random. The test stops if any stored run would be reused.
- **Budget:** every method gets exactly 15 runs per row, each of 5,000
  evaluations (`ea`: about 5,050, as stored in the matrix). `EA_Designer` must
  spend exactly 15 runs, asserted on every replay. Runs still unspent at the
  generation cap go to its current best algorithm.
- **SBS:** for each test function, the algorithm with the lowest mean rank
  over the training rows of the other 23 functions
  (`phase3b_holdout_confirm.py`, lines 200 to 204). Its result is the best of
  its 15 stored runs on the row.
- **Oracle, a reference only:** the row's lowest-median algorithm, best of its
  15 stored runs.
- **Score per row:** (f − best median) / (worst median − best median), using
  that row's 22 stored medians. The meta-learner's row score is the mean, over
  30 replays, of the best value it found.
- **Gap closed:** (SBS − meta-learner) / (SBS − oracle) × 100, on mean row
  scores.
- **Seeds:** replay j of row i (rows in sorted key order) uses seed
  4000 + i × 30 + j, for Python `random`, NumPy's global RNG and the
  stored-run picker.
- **Code:** `phase4_replay.py --stage test`, at the commit that adds this file.

## The one test

- `scipy.stats.wilcoxon(x, y)` with its defaults (`zero_method='wilcox'`,
  two-sided). x = the meta-learner's 24 function means, y = the SBS's 24
  function means. Each function mean averages that function's 3 rows.
- **Pass:** p < 0.05 **and** gap closed > 0. Otherwise the result is "does not
  beat SBS", reported plainly. That is a legitimate finding.
- Run once. No other variant, budget, setting or metric is tried afterwards,
  whatever the outcome.

## Reported, without claims

The other three variants; results by dimension; functions better, worse or
tied; how often the final pick is the row's best algorithm; how the 15 runs
were spent; how often the leftover rule was used.

## Code changes for Phase 4, and their evidence

| Change | Default behaviour kept? | Evidence |
|---|---|---|
| `evaluator` hook, and a log of every run | yes | the 56 earlier tests pass |
| `max_runs` budget, with the leftover rule | yes | new tests: exact budget, leftover rule |
| `use_features` switch | yes | new test |
| Penalty for a skipped candidate: `× 1.05` if the prediction is ≥ 0, `× 0.95` if below | changed for negative predictions only | the old code made a never-run candidate the elite in 5 of 5 seeds of a negative-valued check; the new code never does |
| `n_jobs` for the surrogate forest | yes, default −1 | with −1, 50 repeated predictions were not all identical (largest difference 1.42e-14); with 1 they were |
| A skipped candidate reuses the prediction already made | yes | bit-identical on 48 replays |

No Phase 0 to 3b script uses `EA_Designer`, so no earlier result is affected.
All 73 tests pass (56 earlier, 17 new), and so do the 18 Phase 0 checks.

## Phase 4a: what the development run showed

Run 2026-10-08: `python phase4_replay.py --stage dev`, all 720 training rows,
1 replay per row, 1496.4 s on 8 workers. Output: `results/phase4a_dev.txt`
and `results/phase4a_dev.json` (md5 `001999de8b2c77ceb2fab82f461c8d1c`).

| Variant | Meta-population | Surrogate | Mean score | d5 | d10 | d20 | Distinct algorithms per replay | Reused draws | Replays that used the leftover rule | Mean generations |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 5 | on | −0.0010 | −0.0034 | 0.0005 | −0.0001 | 6.17 | 2884 of 10800 | 699 of 720 | 48.87 |
| v2 | 5 | off | −0.0102 | −0.0047 | −0.0108 | −0.0151 | 11.03 | 1 of 10800 | 0 | 2.00 |
| **v3** | **10** | **on** | **−0.0116** | −0.0136 | −0.0062 | −0.0150 | 9.47 | 537 of 10800 | 293 of 720 | 22.71 |
| v4 | 10 | off | −0.0101 | −0.0103 | −0.0059 | −0.0143 | 11.06 | 0 of 10800 | 0 | 1.00 |

Score: 0 = the best algorithm's median on the row, 1 = the worst; lower is
better. A negative score means the best of 15 runs beat the best median.

- v3 has the lowest mean score, so v3 goes to the test.
- This comparison is biased against variants that put many runs into one
  algorithm. The training matrix stores only 5 runs per cell, and a reused run
  can never improve a replay's best value. v1 reused 2884 of its 10800 draws,
  v3 537. The rule fixed in advance is applied as written regardless.
- v2 and v4 (surrogate off) score −0.0102 and −0.0101, as expected for two
  copies of the same random method.
- No SBS comparison was made on the training matrix: the SBS's best of 15
  needs 15 stored runs per cell.

## Known limits

- The test set has been used before (above).
- 24 functions bound the test, so a small benefit cannot be ruled out.
- With one search key, `EA_Designer`'s mutation redraws the algorithm
  uniformly, so its children are random draws. What adapts is which children
  the surrogate lets through, and which survive.
- **The surrogate often stops letting new candidates through.** In
  development, v3 reached the 50-generation cap with runs unspent in 293 of
  720 replays (v1: 699 of 720). The leftover rule then spent the remaining
  runs, at most 5 for v3, on its best algorithm so far. A possible reason, not
  tested: the surrogate is retrained on its own penalised guesses for skipped
  candidates, which can push its predictions above the median.

---

## Deviations and observations recorded after writing the above

(To be appended only. Nothing above this line is edited.)
