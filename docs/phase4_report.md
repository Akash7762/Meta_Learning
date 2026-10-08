# Phase 4: Meta-learner against the single best solver, at an equal budget

Status: **complete**. Script: `phase4_replay.py` (steps 14 and 15).
Plan: `docs/phase4_plan.md`. Pre-registration: `docs/phase4_preregistration.md`,
committed in `0953353` before the test was run. Output:
`results/phase4a_dev.{txt,json}` and `results/phase4b_confirm.{txt,json}`.

Every figure below is literal output from those runs, or from the descriptive
check named next to it.

## Purpose

Phase 3b showed that guessing the best algorithm from a problem's features does
not beat always running the single best solver (SBS). Phase 4 tests a
different idea: spend the budget **trying** algorithms on the problem itself,
and keep the best answer found. The method is `EA_Designer`
(`aad/meta_learner.py`), the project's meta-learner.

The question: at an equal budget of 15 runs × 5,000 evaluations per problem,
does `EA_Designer` find better solutions than running the SBS 15 times?

## How it was tested

- **Equal budget.** Every method gets exactly 15 runs per problem, each with
  the matrices' settings: `population_size=50`, `max_fe=5000` (`ea`: about
  5,050). `EA_Designer` must spend exactly 15 runs; this was checked on every
  replay. Runs still unspent at its generation cap of 50 go to its best
  algorithm so far.
- **What it searches.** Only which of the 22 algorithms to run. Searching
  population size or generations would break the equal budget, and the
  operator settings only affect `ea`.
- **No new optimiser runs.** Each matrix cell stores independent seeded runs.
  When `EA_Designer` asks to run algorithm A on a problem, it gets one of A's
  stored runs that it has not used yet. That is a run with a fresh seed,
  because the stored runs are exactly such runs.
- **Baseline and reference.** The SBS is chosen from training data, leaving the
  test function out, exactly as in Phase 3b: MPA on 69 test rows, EO on 3. Its
  result is the best of its 15 stored runs. The oracle, a reference only, is
  the row's best algorithm, also best of 15.
- **Score.** Per row, 0 = the best algorithm's median, 1 = the worst; lower is
  better. A negative score means a best-of-15 result beat the best median.
  Gap closed = (SBS − meta-learner) / (SBS − oracle).
- **The one test.** Wilcoxon signed-rank on the 24 function means, two-sided.
  Pass: p < 0.05 and a positive gap closed.

**Data.** Only the existing matrices, at the user's request, both read only;
their md5 was the same before and after every run:

| Role | File |
|---|---|
| Choosing the SBS; development (Phase 4a) | `results/phase2c_matrix.json`: 720 rows, 5 runs per cell |
| Test (Phase 4b) | `results/phase2_matrix.json`: 72 rows, seed 100, 15 runs per cell |

The test matrix was also the Phase 3b test set, and has been studied since
Phase 2. No Phase 4 setting was chosen on it.

### Fixes made to `EA_Designer` first

| Problem in the old code | Fix | Evidence |
|---|---|---|
| Run budgets varied from 1,000 to 20,000 evaluations per run | fixed 15-run budget, matrix settings per run | new tests |
| A skipped candidate's guessed score was multiplied by 1.05, which makes a **negative** guess look better | 5% worse for both signs | the old code made a never-run candidate the elite in 5 of 5 seeds of a negative-valued check; the new code never does |
| The surrogate's forest used all cores, and repeated predictions differed (up to 1.42e-14) | 1 thread for Phase 4 | identical over 50 repeats |
| Landscape features were extracted (300 evaluations) though they are the same for every candidate of a problem | switched off for Phase 4 | new test |

Each fix keeps the old behaviour by default, apart from the penalty for
negative guesses. No Phase 0 to 3b script uses `EA_Designer`. All 73 tests pass.

## Phase 4a: choosing the variant (exploratory)

Four variants, declared in the plan, were replayed once on each of the 720
training rows (1496.4 s on 8 workers):

| Variant | Meta-population | Surrogate | Mean score | Replays that used the leftover rule |
|---|---|---|---|---|
| v1 | 5 | on | −0.0010 | 699 of 720 |
| v2 | 5 | off | −0.0102 | 0 |
| **v3** | **10** | **on** | **−0.0116** | 293 of 720 |
| v4 | 10 | off | −0.0101 | 0 |

v3, `EA_Designer`'s own defaults, had the lowest score and went to the test.
The training matrix stores only 5 runs per cell, so variants that put many
runs into one algorithm had to reuse stored runs (v1: 2884 of 10800 draws),
which can only hurt them. The rule was fixed in advance and applied as written.

## Phase 4b: the pre-registered test

```
  SBS per row     : {'MPA': 69, 'EO': 3} (training ranks, test function left out)
  wall time       : 4232.7 s, 8 parallel workers

  mean score: 0 = best algorithm median on the row, 1 = worst; lower is better
  method                       all         d5      d10      d20
  oracle (VBS), best of 15 -0.0569    -0.0224  -0.0607  -0.0876
  SBS, best of 15          -0.0323    -0.0085  -0.0576  -0.0309
  v1 (pop 5, surr on)      -0.0171    -0.0053  -0.0233  -0.0228
  v2 (pop 5, surr off)     -0.0193    -0.0072  -0.0273  -0.0233
  v3 (pop 10, surr on)     -0.0210    -0.0068  -0.0290  -0.0273
  v4 (pop 10, surr off)    -0.0205    -0.0061  -0.0270  -0.0284

  PRIMARY: v3 vs SBS, paired over 24 function means
    gap closed      : -46.0%
    Wilcoxon        : statistic 109.0, p = 0.2522493600845337 (two-sided)
    functions       : 10 better / 14 worse / 0 tied
    rule            : p < 0.05 AND gap closed > 0
    VERDICT         : DOES NOT BEAT SBS
```

The figures were recomputed independently from the raw matrices and the stored
replays, without the script's own functions, and matched exactly: largest
difference over all row scores 0, primary gap −46.0369%, W=109.0,
p=0.2522493600845337. The same check found every replay's best value among
real stored runs, exactly 15 runs per replay, and no stored run reused.

## Verdict: `EA_Designer` does not beat the SBS

| Pre-registered condition | Result | Met? |
|---|---|---|
| Gap closed is positive | −46.0% | **no** |
| p < 0.05 | p = 0.252 | **no** |

Neither condition is met. The meta-learner is in fact **worse** than the SBS
on average: it gives up 46.0% of the distance from the SBS to the oracle. That
difference is not significant (p = 0.252; 10 functions better, 14 worse), so
the honest summary is: no evidence that it helps, and some sign that it hurts.

## Why it falls short (descriptive only)

These figures come from `results/phase4b_confirm.{txt,json}`. They describe
what happened and carry no test.

- **It spreads its runs thin.** v3 ran 9.39 different algorithms per replay on
  average, and no algorithm got more than 0.94 of its 15 runs on average (EO).
  The SBS spends all 15 on MPA or EO, the two strongest algorithms in Phase 2.
  A likely reason for the loss, not tested: the best of 15 runs of one strong
  algorithm tends to beat the best of one or two runs from each of several.
- **The surrogate often stopped letting new candidates through.** 849 of 2160
  v3 replays reached the generation cap with runs unspent; the leftover rule
  spent 3483 runs in total on the best algorithm found so far.
- **Its final pick was the row's best algorithm in 26.5% of replays.**
- **The surrogate adds little over random choice.** With the surrogate off,
  every candidate is a random algorithm: v2 and v4 closed −53.2% and −48.2%,
  close to v3's −46.0%. v1 (meta-population 5) did worst, −62.0%.
- **By dimension, the SBS is ahead at every dimension** (scores above: d5,
  d10 and d20).
- **Per function** (v3 minus SBS, from the JSON): the largest wins were F4
  BucheRastrigin (−0.0695), F19 GriewankRosenbrock (−0.0629) and F20 Schwefel
  (−0.0323); the largest losses F11 Discus (+0.1705) and F23 Katsuura
  (+0.1386).
- **A perfect pick would gain**: the oracle scores −0.0569 against the SBS's
  −0.0323. The gain exists, but trying algorithms costs more runs than finding
  the right one saves.

This fits the exploratory simulation recorded in the pre-registration, where
spreading runs over the top algorithms beat SBS restarts by +29.6% at 2 runs
but only +0.2% at 5. At 15 runs the sign has turned. That is a pattern across
two different methods, noticed after the fact, not a tested result.

## What this means

Neither guessing nor trying beats the single best solver at an equal budget:

| Analysis | Result |
|---|---|
| Phase 3 step 11, feature-based selector, inside the training matrix | −0.3%, p=0.677 |
| Phase 3b, feature-based selector, pre-registered | +2.1%, p=0.526 |
| **Phase 4b, `EA_Designer` trying algorithms, pre-registered** | **−46.0%, p=0.252** |

The oracle shows there is something to gain on average. Neither approach
captures it: features do not transfer to an unseen function, and spending the
budget to try algorithms costs more than it returns at 15 runs.

## Caveats

1. **"Not significant" does not mean "proven zero."** With 24 functions the
   test cannot rule out a small effect, and the measured one is negative.
2. **The test set was used before**, for Phase 3b. A fresh seed-200 matrix was
   planned and dropped at the user's request, so that only existing data is
   used.
3. **The variant choice used 5-run training cells.** Reuse biased that choice
   against variants that concentrate runs, v1 most.
4. **One budget was tested.** At 2 runs the exploratory simulation found a
   gain (+29.6%, not significant). Smaller budgets were not tested here.
5. **One meta-learner was tested.** With only the algorithm to search,
   `EA_Designer`'s children are random algorithms filtered by the surrogate.
   Other online methods, such as racing or bandits, were not tested.
6. **Settings per run are fixed.** The replay can only use the stored runs
   (population 50, 5,000 evaluations), so tuning population size or operators
   was outside this test.

## Reproducing

```bash
python phase4_replay.py --stage dev
```

```bash
python phase4_replay.py --stage test
```

The development stage took 1496.4 s and the test 4232.7 s, both on 8
workers. Both are fully seeded, and a replay with the same seed gives
identical output (checked in the tests). The test refuses to
run without the pre-registration and the development result, and stops if
any stored run would be reused or any matrix changes. Tests:
`python -m pytest tests/ -q` (73).

## Related

- [`phase4_plan.md`](phase4_plan.md): the plan, with the changes made after
  approval
- `docs/phase4_preregistration.md`: the method fixed before the test
- [`phase3b_report.md`](phase3b_report.md): the feature-based selector's
  pre-registered test
- [`phase2_report.md`](phase2_report.md): the evaluation data
