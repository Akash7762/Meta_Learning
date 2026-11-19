# v2 project plan, following the original README (DRAFT, for review)

Drafted 2026-11-18 on branch `v2`. The phases follow the components and the
pipeline of the original `README.md`, which is never edited. Round one (v1) is
frozen in the git tag `v1-final` and is reference only: no v1 number counts as
v2 evidence.

## 1. The goal

The README's pipeline:

```
Problem  →  Landscape Features  →  Meta-Learner  →  Best EA Config  →  Solution
                                        ↑
                               Surrogate Model
                               (cheap screening)
```

v2 makes every arrow of this pipeline really work, and tests it honestly:

> Give the program an unseen problem. It extracts the problem's landscape
> features. A surrogate model, trained on many earlier problems, screens the
> candidate configurations. The meta-learner (`EA_Designer`) picks the best
> one, and the program runs it to give the solution. It counts as a success
> only if it beats always using the single best configuration (SBS) at the
> same total budget.

## 2. The README against today's code

From reading the v2 code on 2026-11-18 (nothing was run). Phase 0 confirms or
rules out each gap with evidence.

| README part | File(s) | Works today? | Gap (Phase 0 confirms) |
|---|---|---|---|
| Problem | `problems.py` | yes: the README's 6 problems, plus the 24 BBOB functions | ZDT1 and ZDT2 are optimised as f1 + f2 (one number), so no Pareto front is searched for |
| EA and its operators | `ea_components.py`, `evolutionary_algorithm.py` | yes | a run costs population × (generations + 1) evaluations, so the README's settings get from 1,020 to 20,100 evaluations; no fixed-budget option |
| EA config space | `config_space.py` | yes: `EA_CONFIG_SPACE` is exactly the README's | `EA_Designer` does not use it by default: it uses `EXTENDED_CONFIG_SPACE`, which adds an `algorithm` choice (24 mealpy algorithms + `ea`, from `mealpy_algorithms.py`, a file the README does not list; CMA_ES, AOA and DE were restored on 2026-11-19, though v1 found they do not optimise) |
| Landscape Features | `landscape_features.py` | runs | every sample point is evaluated twice; 3 of the 9 feature names do not match what they compute |
| Surrogate Model | `surrogate_model.py` | yes: RF and GP | the default `n_jobs=-1` is not exactly repeatable |
| Meta-Learner | `meta_learner.py` | runs | the surrogate learns only inside one problem (the runner makes a new `EA_Designer` per problem). There, every configuration has the same features, so **the features carry no information and cannot change the choice** |
| LLM Designer | `llm_designer.py` | rule-based mode works offline | the README says OpenAI; the code uses Gemini; a failed API call silently returns a random rule-based pick |
| Evaluation | `evaluation.py` | yes | returns `convergence_speed` (the README shows `convergence_gen`); a failed run becomes 1e12 with no warning |
| Runner and plots | `experiment_runner.py`, `visualization.py` | runs | runs the 24 BBOB functions by default (the README says 5 problems; those are commented out); no seeds; 2 runs per evaluation; `pareto_*.png` plots 200 random points, not the EA's result |

**The main lesson from this table:** the README's idea only works if the
surrogate has learned from many earlier problems. Inside one problem the
features never change, so they cannot help. That is why v2 builds a
performance dataset over many problems (Phase 4), trains the surrogate on it
(Phase 6), and only then gives it to the meta-learner (Phase 7).

## 3. What v1 taught us

| Lesson from v1 | What v2 does |
|---|---|
| Hidden bugs in defaults and wiring | Phase 0 checks every part first |
| Unequal budgets made comparisons unfair | one counted budget rule (Phases 1 and 2) |
| 5 runs per training cell gave wrong labels about 1 time in 6 | 10 runs for training, 15 for test |
| Home-made features were mislabelled and did not transfer | both feature sets checked (Phase 3) and chosen on training data (Phase 6) |
| The test set was reused three times | one fresh test set, used once per question |
| Neither a feature-based selector nor `EA_Designer` beat the SBS | the same honest test, now with a surrogate that has learned from other problems |

## 4. Phases, in the README's pipeline order

| Phase | README part | Question it answers | Main output | Rough effort |
|---|---|---|---|---|
| **0** | all files | Does every part import, connect and run? Where does the code differ from the README? | health report, README-difference list, defect list with decisions | hours |
| **1** | all | What are the rules? | decision document | discussion |
| **2** | Problem, EA, config space | Do the problems and algorithms follow the rules? | fixed code, algorithm health screen, pilot timing | days |
| **3** | Landscape Features | Are the features correct, cheap and repeatable? | checked feature files for every problem | days |
| **4** | Evaluation | How does each configuration do on each problem? | training and test datasets, built once | about 1 day on the college PC (estimate) |
| **5** | Evaluation | What is the bar to beat? | SBS, VBS, random choice, ranking, top-10 configurations | hours |
| **6** | Surrogate Model | Can it predict performance on a problem it has never seen? | chosen surrogate (RF or GP) and feature set | days |
| **7** | Meta-Learner | Which `EA_Designer` setup works best on training data? | frozen design and its pre-registration | days |
| **8** | whole pipeline | Does it beat the SBS on unseen problems? | the main result, run once | hours |
| 9 *(optional)* | LLM Designer | Does an LLM propose better configurations than rules or chance? | its own pre-registration and result | days |
| **10** | Runner and plots | Does the README's full loop work as a tool? | the README's outputs, made honest | days |
| Write-up | | | thesis chapters from the phase reports | weeks |

### Phase 0: health, connectivity and README check

Plan: [phase0/phase0_plan.md](phase0/phase0_plan.md). Checks C1 to C7
(imports, dependencies, call map, fit between modules, tiny runs, known
defects, test suites) plus **C8, the README check**: every name the README
uses exists, every README code example runs as written (in a temporary
folder), and every gap in section 2 is confirmed or ruled out. No research
result.

### Phase 1: the rules (decision document, before any data)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | What "Best EA Config" means | **A**: the README's space only: `ea` with its operator settings. **B**: the README's pipeline with the algorithm added: 24 mealpy algorithms + `ea`, each with its settings (your stated goal) | **Decided 2026-11-18: B.** The README-only answer stays available: it is the `ea` part of the same data |
| 2 | Budget rule | **a**: the README's own: population × (generations + 1) evaluations, different for each setting. **b**: the same fixed number of evaluations for every run; generations then follow from the budget | b (**open**: see the budget note below). Under a, the "best" setting is partly just the one with the most evaluations (about 20 times more at the extremes), and no comparison is fair |
| 3 | Problems | the README's 6 problems; the 24 BBOB functions (home-made `aad` versions, or the official ones from `ioh`) | **Decided 2026-11-18:** learn and test on the 24 BBOB functions × dimensions 5, 10, 20; the README's 4 single-objective problems as an extra unseen check; ZDT1 and ZDT2 only in the Phase 10 demo (official `ioh` versions or home-made: still open) |
| 4 | Runs and seeds | | 10 runs per training cell, 15 per test cell; separate seed ranges, checked so they never overlap |
| 5 | Features | the README's 9 (after fixing), pflacco, or both | compute both in Phase 3; choose in Phase 6 on training data only |
| 6 | Surrogate | RF or GP (both in the README) | choose in Phase 6 |
| 7 | Statistics and success rule | | function means; Wilcoxon; Bonferroni when there is more than one test; pre-registered |
| 8 | LLM provider | the README's OpenAI, or the code's Gemini | decide in Phase 9 |

Configurations per problem for each choice of 1 and 2 (operator settings
counted only for `ea`):

| | a: README budget | b: fixed budget |
|---|---|---|
| **A**: `ea` only | 72 | 24 |
| **B**: `ea` + 24 mealpy | 288 | 96 |

`ea` = 3 population sizes × 2 selections × 2 crossovers × 2 mutations × 3
generation counts = 72; with a fixed budget the generation count is no longer
a choice, so 24. Each mealpy algorithm has 3 population sizes (× 3 generation
counts under a): 24 × 9 + 72 = 288, or 24 × 3 + 24 = 96.

**Budget note** (your question, 2026-11-18: what if some algorithms only get
good after the budget, say after 10,000 evaluations?). With a fixed budget, the
model learns which configuration is best *at that budget*. An algorithm that
improves only later ranks lower, which is right for that budget; the
recommendation can be wrong only if the tool is later used with a very
different budget. Some algorithms also plan their search by the run's total
length: in mealpy, MPA and EO (v1's two best algorithms) use
`epoch / self.epoch` in their update rules (`mealpy/swarm_based/MPA.py:72`,
`mealpy/physics_based/EO.py:80`), so the same algorithm behaves differently
at different budgets. Options:

1. one fixed budget B; every result is stated as "best at B evaluations";
2. option 1 plus a budget check: each of the 25 algorithms, with its best
   setting at B, rerun on 72 training problems (one instance per function and
   dimension) at about 5 × B, to measure how often the winner changes after B
   (rough cost at B = 10,000: about 34 CPU-hours);
3. the budget as an input: data at several budgets (for example 2,000, 10,000
   and 50,000), and the model answers for the budget the user gives (about 6
   times the data of option 1).

Also to decide: the same budget for every dimension, or one that grows with
dimension (for example 1,000 × dimension).

### Phase 2: problems and algorithms follow the rules

Fix what Phase 0 sent here and build what Phase 1 decided, each with a test:
the chosen problem suite with seeded instances (the README's custom-problem
interface unchanged); `ea` stopping at the evaluation budget; seeds in every
algorithm; operator settings used only by `ea`; the v2 configuration list
defined next to the README's `EA_CONFIG_SPACE` (which stays as it is); failed
runs reported, never replaced. Then an algorithm health screen (drop
algorithms that do not optimise; CMA_ES, AOA and DE, restored on 2026-11-19,
are re-tested here) and a short pilot that measures the real time
per run.

### Phase 3: landscape features

The README's extractor (`landscape_features.py`) after its fixes (each point
evaluated once; names that say what they compute), and pflacco (in
`.venv-pflacco`, already set up), on every problem. Checks: each feature's
cost in evaluations (it counts toward the total budget) and its stability
across sampling seeds. Output: feature files for all training and test
problems (minutes to compute).

### Phase 4: performance data (built once)

Every configuration on every problem, through the README's
`EvaluationFramework`, keeping its metrics (best, mean and std fitness,
convergence, stability, runtime) plus evaluations used and seeds. Runs in
parallel and resumes after an interruption.

With B (decided) and b (recommended):

| Set | Problems | Configurations | Runs | Total runs |
|---|---|---|---|---|
| Training | 24 functions × 3 dimensions × 10 instances = 720 | 96 | 10 | 691,200 |
| Test | 24 × 3 × 1 fresh instance = 72 | 96 | 15 | 103,680 |

About 300 CPU-hours, using v1's measured 1.35 s per 5,000-evaluation run:
about 18 to 27 hours on the college PC with 10 to 12 workers (a rough
estimate; the Phase 2 pilot gives the real number). At 10,000 evaluations per run it
takes about twice as long.

### Phase 5: baselines

Training data only: Friedman ranking of all configurations; the single best
configuration (SBS); the best configuration per problem (VBS, the ceiling);
random choice; how much a perfect choice could gain over the SBS, overall and
by dimension; and the top-10 configurations.

### Phase 6: surrogate model

The README's `SurrogateModel`, trained on the training data: input = the
problem's features + a configuration; output = its performance. Tested by
leaving one function out at a time: on a function it has never seen, does it
rank the configurations well, and how close does its top pick get to the VBS?
RF against GP, and the two feature sets; the best pair is chosen on training
data only. `n_jobs=1`, so results repeat exactly.

### Phase 7: meta-learner

The README's `EA_Designer`, now able to start from the trained surrogate (a
small change in `meta_learner.py`). Variants, all at the same total budget
(feature cost + search runs + final runs), compared on training data by
leaving one function out:

| Variant | What it does |
|---|---|
| README as it is | the surrogate starts empty and learns only inside the problem |
| Warm start | the surrogate starts trained on the training data; `EA_Designer` then searches with a few real runs |
| Surrogate only | the surrogate picks the configuration directly; all runs go to that pick |

The best variant is frozen in a pre-registration, committed before Phase 8.

### Phase 8: confirmation (run once)

The frozen pipeline against the SBS on the fresh test set, at the same total
budget including the feature cost. Wilcoxon on function means and the share of
the SBS-to-VBS gap closed. Run once; reported whatever the result.

### Phase 9 (optional): LLM designer

The README's `LLMDesigner`: an LLM proposes a configuration from the
problem's description and features. First, a failed call must show as a
failure (no silent fallback). Compared with the rule-based designer, random
choice and the SBS at the same budget, under its own pre-registration. Depends
on the API being available (v1 hit repeated overload errors).

### Phase 10: the experiment runner (the README's Quick Start, made honest)

`experiment_runner.py` rebuilt to do what the README promises: the README's
problems plus any custom problem; the full loop (features → surrogate →
`EA_Designer` → best configuration → solution); and the README's outputs
(`experiment_results.json`, `fitness_<Problem>.png`, `meta_<Problem>.png`,
`pareto_ZDT1Problem.png`). Seeds, equal budgets, enough runs, and the SBS
result printed beside every result. For ZDT1 and ZDT2, *decide*: plot the EA's
real final points (honest, but summed objectives give a small cluster, not a
front), or add a true multi-objective EA (extra work, outside the learned
pipeline).

## 5. Rules for every phase

- Plans, decisions and pre-registrations go in `docs/planning/phase<N>/`
  before the phase runs; pre-registrations are never edited once committed.
- Results go in `results/phase<N>/` and are listed in `results/MANIFEST.md`.
- When every step of a phase passes, its report goes in
  `docs/reports/phase<N>_report.md`, with numbers taken only from result
  files; then the phase is committed (after asking) and backed up.
- Seeds everywhere; failures are shown, never silently replaced; long runs can
  resume after an interruption.
- `README.md` is never edited; differences from it are recorded in `docs/`.
