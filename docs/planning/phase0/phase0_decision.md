# Phase 0 — Scope Decision: Algorithm Selection as the Primary Research Question

**Status:** reviewed and APPLIED (items 1-5). Item 6 deferred to Phase 1.
**Date:** 2026-09-14

---

## 1. Primary research question

**Given the landscape features of an unseen optimisation problem, which of the 24
mealpy metaheuristics should be selected to solve it?**

This is the framework's main path. Problem characterisation feeds a selector; the
selector emits an `algorithm` key drawn from `EXTENDED_CONFIG_SPACE["algorithm"]`.

## 2. Canonical config space

`EXTENDED_CONFIG_SPACE` (`aad/config_space.py:49`) is the default and canonical
config space for **all top-level experiment scripts and all library defaults**.
It is the only space that contains an `algorithm` key, and the `algorithm` key is
required by `build_algorithm` (`aad/evolutionary_algorithm.py:204`).

## 3. Status of EA operator tuning

`EA_CONFIG_SPACE` and the selection/crossover/mutation operator tuning remain
supported **only as an internal branch reached when `algorithm == "ea"`**. They are
a secondary experiment, never the top-level default. Concretely:

- `EA_CONFIG_SPACE` must never be a default argument value for any public entry
  point. It may be used explicitly by an operator-tuning experiment that pins
  `algorithm="ea"`.
- Operator keys are meaningless for the other 24 algorithms. `MealpyAlgorithm`
  reads only `algorithm`, `population_size`, `max_generations`
  (`aad/mealpy_algorithms.py:210-220`); it silently ignores
  selection/crossover/mutation.

## 4. LLM designer

`LLMDesigner` is evaluated as a *selector* for the primary question in a later
phase. **Its behaviour is not changed in Phase 0.** Where it currently conflates
the two questions is recorded in section 5 as an observation, not an action.

---

## 5. Audit — state BEFORE the Phase 0 changes

*Historical record. Line numbers refer to the tree as audited; items 1-5 in
section 7 have since been applied, so several of these no longer hold.*

### 5a. `EA_CONFIG_SPACE` vs `EXTENDED_CONFIG_SPACE` references

| File | Line | What it does |
|---|---|---|
| `aad/config_space.py` | 21 | Defines `EA_CONFIG_SPACE` — **no `algorithm` key** |
| `aad/config_space.py` | 49 | Defines `EXTENDED_CONFIG_SPACE` — has `algorithm`: `["ea"] + 24 mealpy keys` |
| `aad/config_space.py` | 78 | `random_config()` defaults to `EA_CONFIG_SPACE` |
| `aad/config_space.py` | 91 | `all_configs()` defaults to `EA_CONFIG_SPACE` |
| `aad/config_space.py` | 119 | `config_to_vector()` defaults to `EA_CONFIG_SPACE` |
| `aad/config_space.py` | 153 | `vector_to_config()` defaults to `EA_CONFIG_SPACE` |
| `aad/llm_designer.py` | 24, 51 | `LLMDesigner.config_space` falls back to `EA_CONFIG_SPACE` |
| `aad/meta_learner.py` | 21, 76 | `EA_Designer.config_space` falls back to `EA_CONFIG_SPACE` |
| `experiment_runner.py` | 52, 198, 232 | Passes `EXTENDED_CONFIG_SPACE` explicitly — already compliant |
| `test_runner.py` | 25, 52, 66, 102, 105 | Passes `EXTENDED_CONFIG_SPACE` explicitly — already compliant |
| `user_model_runner.py` | 34, 98, 225, 310, 340 | Uses `EXTENDED_CONFIG_SPACE`; `_sp()` (line 91) narrows it to one pinned algorithm — already compliant |

### 5b. Entry points constructed without an explicit `algorithm` key

| File | Line | What it does |
|---|---|---|
| `aad/meta_learner.py` | 76 | `EA_Designer()` with no `config_space` → `EA_CONFIG_SPACE` → candidates have no `algorithm` key |
| `aad/meta_learner.py` | 101 | `generate_candidate()` → `random_config(EA_CONFIG_SPACE)` → config with no `algorithm` |
| `aad/meta_learner.py` | 128 | `evaluate_candidate()` → `build_algorithm(cfg)` → **raises `ValueError` on the default path** |
| `aad/llm_designer.py` | 51 | `LLMDesigner()` with no `config_space` → `EA_CONFIG_SPACE`; rule-based path then falls back to `algorithm="ea"` for every problem |
| `tests/test_aad.py` | 412, 422, 430, 437 | Four `EA_Designer(...)` constructions with no `config_space` — currently dead (see section 6) |
| `aad/evaluation.py` | 74 | `build_algorithm(ea_config)` — inherits whatever the caller passed; no guard of its own |

### 5c. Operators computed for non-`"ea"` algorithms

| File | Line | What it does |
|---|---|---|
| `aad/config_space.py` | 52-54 | `EXTENDED_CONFIG_SPACE` carries selection/crossover/mutation for all 25 algorithm values; 24 of them ignore those keys |
| `aad/meta_learner.py` | 101, 105, 110 | Meta-EA samples, mutates and crosses over operator keys regardless of `algorithm` — wastes meta-budget on no-op dimensions for 24/25 algorithms |
| `aad/config_space.py` | 101-133 | `config_to_vector` one-hot encodes all three operator keys for every config, so the surrogate sees 6 always-noise features when `algorithm != "ea"` |
| `aad/mealpy_algorithms.py` | 210-220 | `MealpyAlgorithm.__init__` reads only `algorithm`/`population_size`/`max_generations` — the operator keys are silently dropped |
| `aad/evolutionary_algorithm.py` | 211 | Dead default: `config.get("algorithm", "pso")` is unreachable (the `raise` above it fires first), but preserves the old silent-PSO bug in the source |

---

## 6. Live finding — the default path was broken (now fixed by item 1)

`EA_Designer()` constructed without `config_space` raises at the first candidate
evaluation:

    ValueError: config is missing required key 'algorithm'.

Verified by running `EA_Designer(population_size=2, generations=1,
runs_per_eval=1, use_surrogate=False).evolve(SphereProblem(dim=3))`.

This means the `EA_CONFIG_SPACE` default is not merely inconsistent with the Phase 0
decision — it is dead code that cannot execute. `tests/test_aad.py:412/422/430/437`
exercise exactly this path and therefore currently fail. The three top-level runner
scripts all pass `EXTENDED_CONFIG_SPACE` explicitly, which is why the breakage has
not been visible in experiment runs.

---

## 7. Changes — items 1-5 APPLIED, item 6 deferred

Ordered by dependency. Items 1-5 were applied after review; item 6 is
explicitly held for Phase 1 because it changes results, not just defaults.

1. **`aad/meta_learner.py:76`** — change `self.config_space = config_space or EA_CONFIG_SPACE`
   to default to `EXTENDED_CONFIG_SPACE`; update the import on line 21 and the
   docstring on line 45. *Fixes the broken default path and makes selection the
   library default.*
2. **`aad/llm_designer.py:51`** — change the `or EA_CONFIG_SPACE` fallback to
   `EXTENDED_CONFIG_SPACE`; update import on line 24. *Without this, a bare
   `LLMDesigner()` can only ever propose `algorithm="ea"`, which silently answers
   question (a) while appearing to answer (b).*
3. **`aad/config_space.py:78, 91, 119, 153`** — change the four `config_space is None`
   fallbacks from `EA_CONFIG_SPACE` to `EXTENDED_CONFIG_SPACE`; update docstrings on
   lines 71, 112. *Removes the last implicit route to an `algorithm`-less config.*
4. **`aad/evolutionary_algorithm.py:211`** — delete the unreachable
   `config.get("algorithm", "pso")` default, replacing it with `config["algorithm"]`.
   *Removes the source of the original silent-PSO bug.*
5. **`tests/test_aad.py:412, 422, 430, 437`** — pass `config_space=EXTENDED_CONFIG_SPACE`
   explicitly, or let them inherit the new default once (1) lands. Also
   `tests/test_aad.py:244-255`, which asserts `random_config()` keys equal
   `EA_CONFIG_SPACE` keys, must be re-pointed.
6. **Operator-dimension masking (design work, defer to Phase 1)** —
   `aad/meta_learner.py:105/110` and `aad/config_space.py:101` should skip or mask
   the three operator keys when `algorithm != "ea"`, so the meta-search and the
   surrogate stop spending capacity on dimensions that have no effect. This is the
   only item that changes *results* rather than defaults, so it should be measured,
   not just applied.

No change is required in `experiment_runner.py`, `test_runner.py`, or
`user_model_runner.py` — all three already pass `EXTENDED_CONFIG_SPACE` explicitly
and are compliant with this decision as written.

### 7a. Audit gap found while applying

Section 5b listed `aad/evaluation.py:74` as inheriting an unguarded config, but
missed a second instance: `tests/test_aad.py:387` (`TestEvaluationFramework._cfg`)
hardcodes a config dict with no `algorithm` key, failing three tests. This is the
same class of defect as the §5b entries and was fixed as part of item 5
(`"algorithm": "ea"` added, since those tests exercise the EA operator path).

The failure predates this phase: the `build_algorithm` guard that surfaces it was
already present in the uncommitted working tree at the start of Phase 0, not added
by it.

### 7b. Verification

- `test_phase0_config_consistency.py`: 18 passed, 0 failed. The `EA_Designer()`
  default check flipped from "fails loudly with ValueError (non-compliant)" to
  "default space carries 'algorithm' (compliant)".
- `tests/test_aad.py`: 53 passed, 0 failed, with `TestLLMDesigner` skipped (see
  §7c). `pytest` is not installed in `.venv`, so the suite was run through an
  in-memory stub supplying `approx`, `raises`, and the `tmp_path` fixture.

### 7c. Still open

- `pytest` is absent from `.venv`; `tests/test_aad.py` cannot be run normally.
- `tests/test_aad.py:363` comments `# no API key -> rule-based`, but `load_dotenv()`
  at import means a bare `LLMDesigner()` picks up `.env` and makes live API calls.
  Three tests in `TestLLMDesigner` hit the network and were skipped during
  verification.
- `aad/llm_designer.py:52`: `self._use_llm = self.api_key is not None` treats an
  empty-string key as present, so `GEMINI_API_KEY=` in `.env` yields `_use_llm=True`
  and guaranteed API failure. Latent; not triggered today.
- `aad/llm_designer.py:22`: stray `from pydantic import config` — unused, shadows
  the local `config` name. Left in place as outside the approved scope.

---

## 8. Addendum — 2026-09-15: correction to a diff shown during verification

During Phase 0b verification, a "Part B — Phase-0-only diff (reconstructed)" was
presented for `aad/evolutionary_algorithm.py`. One hunk in that presentation was
shown with its `+`/`-` sides transposed, making it read as though the current code
were:

    algorithm_key = config.get("algorithm", "pso")

That is wrong. The actual current code, verified this date by direct file read, is:

    if "algorithm" not in config:
        raise ValueError(
            "config is missing required key 'algorithm'. "
            "Pass 'ea' for the built-in EvolutionaryAlgorithm or one of "
            "the mealpy keys (see MEALPY_ALGORITHM_KEYS)."
        )
    algorithm_key = config["algorithm"]

The `ValueError` guard is intact and item 4 of section 7 remains correctly applied.

**Cause.** Not a regression in the code, and not a fault in the reverse-apply
reconstruction. That reconstruction was re-run on 2026-09-15 and reproduces the
correct hunk (`-` the old `config.get(..., "pso")`, `+` the current
`config["algorithm"]`). The error was introduced when the diff was re-transcribed
by hand from an earlier message rather than regenerated by re-running the command;
the two sides of that single hunk were swapped in the retyping.

**Consequence.** Presentation only. No code change was required and none was made.

**Practice note.** Diffs are to be regenerated by running the diff command, never
re-pasted from an earlier message.
