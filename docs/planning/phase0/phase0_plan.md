# Phase 0 plan: codebase health and connectivity (decisions taken, not yet run)

Drafted 2026-11-18 on branch `v2`; decisions D1 to D5 taken by 2026-11-20.
Nothing has been run for Phase 0 yet.
Phase 0 produces **no research result**. It proves the code is sound before
any research starts.

## Goal

Before any experiment, show with evidence that:

1. every module and script **imports** cleanly;
2. every public function, class and method is **connected**: something uses
   it, or it is flagged;
3. the modules **fit together**: shared settings and interfaces agree;
4. every algorithm and every main component **works on a tiny case**,
   respects its budget, and repeats exactly with the same seed;
5. both environments match their **requirement files**;
6. every **known defect** is confirmed or ruled out, and each one gets a
   decision;
7. the code does what the **README** describes: every name it uses exists,
   every code example runs as written, and every difference is listed
   (`README.md` itself is never edited).

## Scope

From a static scan of the v2 code on 2026-11-18 (read-only, nothing executed):

| Part | Files | Notes |
|---|---|---|
| Framework `aad/` | 12 modules | 45 functions, 62 classes, 205 methods across the project; `aad/__init__.py` imports all 11 other modules |
| Scripts | `experiment_runner.py`, `user_model_runner.py`, `test_runner.py`, `test_phase0_config_consistency.py` | all have a `__main__` guard |
| Tests | `tests/test_aad.py` | 56 tests, all passing on v2 |
| External packages imported | numpy, scikit-learn, matplotlib, mealpy, python-dotenv, google-genai, pytest | `requirements.txt` pins 59 packages (with dependencies) |
| Feature environment | `.venv-pflacco` | `requirements-pflacco.txt`, 25 pins |
| README | `README.md` | read only; its names, examples and claims are checked (C8) |

How the modules depend on each other (from the scan):

```
aad/__init__ ──► all 11 modules
evaluation ──► evolutionary_algorithm ──► ea_components, mealpy_algorithms
meta_learner ──► config_space, evolutionary_algorithm, landscape_features, surrogate_model
llm_designer ──► config_space
standalone: config_space, ea_components, landscape_features, mealpy_algorithms,
            problems, surrogate_model, visualization
```

The scan already found **2 names that nothing uses**:
`EvaluationFramework.compare_configs` and `LLMDesigner.describe_for_selection`
(the latter was used only by the v1 Phase 5 script, removed on v2). The scan
matches names only, so Phase 0 replaces it with a real call map (C3).

## The checks

| # | Check | How | Passes when |
|---|---|---|---|
| **C1** | **Imports** | import each of the 12 modules in a fresh Python process (catches hidden order dependencies); check every name in `aad.__all__` exists; compile every script and resolve each of its imports | 12 of 12 import; all exported names exist; every script import resolves |
| **C2** | **Dependencies** | match every third-party import to `requirements.txt`; compare installed versions with the pins in both environments; in `.venv-pflacco`, import pflacco and re-check that the problems are identical to the main environment (the v1 fingerprint test) | every import covered; installed = pinned; fingerprints identical |
| **C3** | **Connections** | build a call map: for every public function, class and method, who refers to it in the code, and whether the test suite actually runs it (recorded while the tests run, with Python's own profiler; no new package) | every item is either used and tested, used but untested (listed), or unused (listed) |
| **C4** | **Fit between modules** | the algorithm list in `config_space` = the mealpy registry + `ea`; `build_algorithm` reaches every algorithm; operator settings are read only by `ea`; all 30 problem classes (24 BBOB + 6 classic) provide `evaluate`, `bounds`, `is_multiobjective`, `name`; the feature extractor returns the same 9 keys for every problem; surrogate input and output shapes agree | every contract holds |
| **C5** | **Tiny working runs** | each of the 25 algorithms on a small BBOB problem with a fixed seed and a small budget, run twice: it finishes, gives a finite result, gives the **identical** result both times, and the counted evaluations (made on the real problem) stay within the budget; plus one tiny run each of feature extraction, `EA_Designer`, `LLMDesigner` (offline), `EvaluationFramework`, `Visualizer`, and `experiment_runner.py` on one tiny problem, all writing only to a temporary folder | 25 of 25 algorithms pass (any budget exception, such as `ea`, is measured and listed); every component runs |
| **C6** | **Known-defect register** | confirm or rule out each item below with evidence (file, line, measured behaviour), and add anything C1–C5 finds | every item has a status and a decision |
| **C7** | **Test suites** | run `tests/` and `test_phase0_config_consistency.py` | all pass |
| **C8** | **README check** | every name the README uses exists in `aad` (its classes, `EA_CONFIG_SPACE`, `meta_convergence`, `generate_description`, `predict_with_uncertainty`, `summarize`); run each README code example as written, in a temporary folder (the minimal example, `EA_Designer`, `LLMDesigner` offline, `SurrogateModel` with `"rf"` and `"gp"`, a custom problem); compare each README claim with the code (section 2 of the [project plan](../project_plan.md)) | every name exists; every example runs; every difference is listed with evidence |

### Known defects to confirm or rule out (C6)

Items 1 to 11 are carried over from v1; items 12 to 14 were found while
writing the README-based plan; items 15 and 16 came up while restoring three
algorithms. Each must be checked on v2, not assumed:

| # | Suspected defect | Where |
|---|---|---|
| 1 | Operator settings are not masked: `EA_Designer` still varies selection/crossover/mutation for mealpy algorithms, and the surrogate encodes them | `aad/meta_learner.py`, `aad/config_space.py` |
| 2 | Every feature sample point is evaluated twice, so features cost 600 + 50 × dimension evaluations, not 300 | `aad/landscape_features.py` |
| 3 | `ea` ignores `max_fe`; its budget is population × (generations + 1) | `aad/evolutionary_algorithm.py` |
| 4 | `EvaluationFramework` replaces failed (NaN/Inf) runs with 1e12, hiding failures | `aad/evaluation.py` |
| 5 | The surrogate's default `n_jobs=-1` gives slightly different predictions from run to run | `aad/surrogate_model.py` |
| 6 | Importing `aad` loads `.env` into the environment as a side effect | `aad/llm_designer.py` |
| 7 | Three feature names mislead: `modality` is a skewness, `fitness_variance` is std / \|mean\|, `ruggedness` ignores neighbouring points | `aad/landscape_features.py` |
| 8 | `experiment_runner.py` uses unequal budgets, unseeded runs, 2 evaluation runs, and labels the random rule-based fallback as the LLM designer | `experiment_runner.py` |
| 9 | Algorithm lists: `user_model_runner.py` says 24 algorithms, true again since CMA_ES, AOA and DE were restored (2026-11-19); confirm that the registry, the duplicate list in `config_space.py` and the LLM names all hold the same 24 | `user_model_runner.py`, `aad/config_space.py`, `aad/llm_designer.py` |
| 10 | The old `LLMDesigner.propose_algorithm` path silently swaps in a random rule-based pick when the API fails | `aad/llm_designer.py` |
| 11 | The 2 unused methods found by the scan | `aad/evaluation.py`, `aad/llm_designer.py` |
| 12 | Multi-objective problems (ZDT1, ZDT2) are optimised as f1 + f2, so no Pareto front is searched for | `aad/evolutionary_algorithm.py`, `aad/mealpy_algorithms.py` |
| 13 | `pareto_<Problem>.png` plots 200 random points, not the EA's result | `experiment_runner.py` |
| 14 | `EA_Designer`'s surrogate learns only inside one problem, where every configuration has the same features, so the features cannot change the choice | `aad/meta_learner.py`, `experiment_runner.py` |
| 15 | CMA_ES, AOA and DE (restored 2026-11-19) did not optimise in v1: 10 times the budget did not improve their result | `aad/mealpy_algorithms.py` |
| 16 | ZDT1 sometimes takes the square root of a negative (or NaN) number (`RuntimeWarning`), seen once in a full test run on 2026-11-19. It did not repeat when the ZDT tests were run on their own, so it probably depends on the random state left by earlier tests; find the test and the input that cause it | `aad/problems.py`, `tests/test_aad.py` |

## Outputs

| File | What it is |
|---|---|
| `phase0_health_check.py` (new) | runs C1 to C8; changes no code; writes only to `results/phase0/` and a temporary folder |
| `results/phase0/health_report.txt` and `.json` | the result of every check, including the README-difference list (C8) |
| `results/phase0/call_map.csv` | every public function, class and method: who uses it, whether the tests run it |
| `docs/reports/phase0_report.md` | the Phase 0 report, written when every check passes |

## Steps and stop points

| # | Step | Stop and ask? |
|---|---|---|
| 0 | You review this plan and decisions D1 to D5 (done 2026-11-20) | yes |
| 1 | Write `phase0_health_check.py`; show you what it does | yes |
| 2 | Run it on the second PC (rough estimate: 10 to 25 minutes; see below) | only if the estimate is over about 2 minutes |
| 3 | Go through the findings together; decide each defect: fix in Phase 0, fix in Phase 2, or leave | yes |
| 4 | Apply the fixes chosen for Phase 0, with tests; re-run until every check passes | yes, before each fix |
| 5 | Write `docs/reports/phase0_report.md`; commit | yes, commit |

## Running it on the second PC (decided 2026-11-20)

Step 2 runs on the second PC. The two Python environments are not in git, so
they are rebuilt there from the pinned files. In PowerShell:

1. Python 3.11 must be installed (`py -0` lists it). Both environments here
   are 3.11.9, and the pflacco pins (numpy 1.24.4) need 3.11.
2. Get the code. First time:
   `git clone https://github.com/Akash7762/Meta_Learning.git`, then
   `cd Meta_Learning` and `git switch v2`. Later: `git switch v2` and `git pull`.
3. Build the environments (the names must be exactly these; the script looks
   for `.venv-pflacco`):
   - `py -3.11 -m venv .venv`
   - `.venv\Scripts\python.exe -m pip install -r requirements.txt`
   - `py -3.11 -m venv .venv-pflacco`
   - `.venv-pflacco\Scripts\python.exe -m pip install -r requirements-pflacco.txt`
4. Run: `.venv\Scripts\python.exe phase0_health_check.py`
   (a subset: `--only C1,C4`; describe without running: `--list`).
5. Send the results back: `git add results/phase0`, then
   `git commit -m "phase0: health check results"` and `git push`. On the first
   PC, `git pull` brings them over for step 3.

No `.env` is needed: the script never calls an LLM API. Without a `.env`,
defect #6 (".env loaded on import") shows NOT CHECKED. Run times and the CPU
count in defect #5 depend on the machine; the other checks do not.

## Decisions for you

| # | Question | Recommendation |
|---|---|---|
| D1 | Should Phase 0 fix what it finds, or only report? | **Decided 2026-11-18: report only.** Fix in Phase 0 only what blocks the checks themselves (for example a broken import); everything else is fixed in Phase 2, once Phase 1 has set the rules |
| D2 | Run `user_model_runner.py` and `test_runner.py` (your reference harnesses), or check them statically only? | **Decided 2026-11-20:** static only (imports, compile, references): they are references, not the pipeline. Run `experiment_runner.py` for real, since it is the final-run file |
| D3 | Keep the v1-era `test_phase0_config_consistency.py`? | **Decided 2026-11-20:** keep it and call it from the new health check, so nothing is duplicated |
| D4 | Check `.venv-pflacco` as well? | **Decided 2026-11-20:** yes: pflacco imports and the problem fingerprint matches |
| D5 | Run the README's code examples exactly as written? (The minimal example runs `EA_Designer` for real, on its default space with all 25 algorithms.) | **Decided 2026-11-20:** yes, in a temporary folder, timed; a difference in behaviour is recorded, not fixed |
