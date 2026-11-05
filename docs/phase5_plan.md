# Phase 5 plan (DRAFT, not approved)

Drafted 2026-10-08. Nothing in this plan has been run, no code has been
changed, and no API call has been made. Every dataset used in Phases 0 to 4
stays read-only.

## In plain words

Phases 3b and 4 found that neither a feature-based model nor trying
algorithms on the problem beats always running the single best solver (SBS).
Phase 5 asks a large language model (LLM) instead. We describe a problem to
it, it names one of the 22 algorithms, and we look up how well that algorithm
did in the matrix we already have. The "ablations" change what the LLM is
told (the problem's name, its landscape features, both, or neither), to see
which information, if any, helps.

## 0. First: the API key file is in git

Checked 2026-10-08 without opening the file.

| Fact | Evidence |
|---|---|
| `.env` is tracked by git | `git ls-files .env` lists it |
| It was committed in `bd71fa0` ("Work done upto successfull API key's integration and test_runner") | `git log --all -- .env` |
| No ignore rule covers it. `.gitignore` has `.env/`, which matches a folder, not a file | `git check-ignore -v .env` prints nothing |
| No remote is configured, so it has not been pushed from this copy | `git remote -v` prints nothing |

Before Phase 5 uses the key:

1. Stop tracking the file and ignore it: `git rm --cached .env`, add a line
   `.env` to `.gitignore`, and commit. The file stays on disk.
2. The key stays in the history (`bd71fa0`, on both `main` and
   `exp-runner-tuning`). If this repository will ever be pushed or shared,
   either rewrite that history first, or create a new key in Google AI Studio
   and delete the old one. Only you can do the second.

## 1. The question

> Given a description of an unseen problem, can an LLM pick an algorithm that
> beats always running the SBS? And which part of the description helps it?

`docs/phase0_decision.md` (line 39) already says `LLMDesigner` "is evaluated as
a *selector* for the primary question in a later phase". This is that phase.

## 2. What the code does today

Checked against the source on 2026-10-08.

| Fact | Where | Why it matters |
|---|---|---|
| `load_dotenv()` runs when the module is imported | `aad/llm_designer.py:25` | any script that imports `aad` loads the key into the environment |
| `self.api_key = api_key or os.getenv("GEMINI_API_KEY")` | `aad/llm_designer.py:50` | `LLMDesigner(api_key=None)` still uses the key from `.env`, so offline mode cannot be forced. The tests work around this by deleting the variable (`tests/test_aad.py:363-369`) |
| The LLM is told the function's name, dimension and bounds, and no landscape features | `generate_description`, `aad/llm_designer.py:199-233` | for example "BBOB_F11_Discus optimisation problem. Dimension: 5. Search bounds: [-5.00, 5.00]. …". The name lets the LLM use what it remembers about BBOB instead of the problem's features |
| The prompt always asks for 6 keys (algorithm, population size, three operators, generations), whatever the search space | `aad/llm_designer.py:147-148` | Phase 5 scores the algorithm only, at the matrices' fixed settings |
| No temperature or seed is set | `aad/llm_designer.py:153` | answers can change from call to call |
| On any error it prints a message and silently uses the rule-based pick instead | `aad/llm_designer.py:169-174` | an experiment would mix rule-based picks into "LLM" results without saying so |
| Only the answer's keys are checked, not its values | `aad/llm_designer.py:162-165` | an algorithm that is not registered, such as the removed `CMA_ES`, would pass |
| The rule-based fallback picks by keywords such as "multimodal" or "smooth" | `aad/llm_designer.py:73-81` | none of the 72 test-row descriptions contains any of them, so it picks a random algorithm (an unseeded `random.choice`) every time |
| Default model `"gemini-3.6-flash"` | `aad/llm_designer.py:47` | not confirmed to exist; one test call will tell |
| The installed `google-genai` 2.21.0 supports `temperature`, `seed` ("a best effort to provide the same response") and a JSON response schema | `google.genai.types.GenerateContentConfig` | fixed settings reduce run-to-run differences but do not remove them, so every answer must be stored |

## 3. Design

### 3.1 What the LLM is told: four arms

| Arm | The LLM is told | What it tests |
|---|---|---|
| A, name | function name, dimension, bounds (today's description) | what the LLM remembers about BBOB functions |
| **B, features** | dimension, bounds and the 9 landscape features; no name | the research question: selection from the features of an unseen problem |
| C, both | everything in A and B | whether features add anything to the name |
| D, nothing | dimension and bounds only | the LLM's default favourite |

The 9 features are the ones stored in the matrices (`dimension`,
`fitness_max`, `fitness_mean`, `fitness_min`, `fitness_std`,
`fitness_variance`, `gradient_magnitude`, `modality`, `ruggedness`), the same
inputs the Phase 3 selector used. Every arm gets the same instructions and the
same list of the 22 algorithms, and must answer with one algorithm name. A
test checks that arms B and D contain no function name.

**Primary arm: B.** A truly new problem has no BBOB name, so arms A and C
cannot be the main answer to the research question.

### 3.2 Scoring: the same as Phase 3b

- A pick on a row gets that algorithm's rank on the row (1 = best of 22), from
  the 15-run medians in the evaluation matrix. No optimiser is run.
- Baselines as in Phase 3b: the SBS, chosen from the training matrix with the
  test function left out (MPA on 69 rows, EO on 3); the VBS, the row's best
  algorithm (an oracle); and random choice, rank 11.5 on average.
- Gap closed = (SBS rank − LLM rank) / (SBS rank − VBS rank), on mean ranks.
- Budget: a pick costs no function evaluations, and every method then runs
  one algorithm with 5,000 evaluations, as in Phase 3b. Arms B and C use
  features that cost 300 evaluations to compute. As in Phase 3b, that is not
  charged.

### 3.3 Repeatability

- A fixed model ID, `temperature=0`, a fixed `seed`, and a JSON schema that
  only allows the 22 names.
- K = 3 calls per row and arm. A row's score is the mean rank of its 3 picks,
  and how often they agree is reported.
- Every request and answer is stored: prompt, settings, raw text, parsed pick,
  the model version the API reports, time and token counts. The analysis
  reads only this store, so it can be redone exactly, offline. Calling the API
  again may give different answers; the report will say so.
- An invalid answer is retried up to 2 times. If it is still invalid, it gets
  the worst rank (22) and is counted. Nothing is silently replaced.

### 3.4 Which problems

As in Phase 4, only the existing matrices, both read only:

| Set | File | Used for |
|---|---|---|
| Training | `results/phase2c_matrix.json` | choosing the SBS (leave-one-function-out) |
| Development | 24 training rows: every function at seed 42, dim 10 | checking prompts, parsing, validity and cost. No setting is chosen by score |
| Test | `results/phase2_matrix.json`, 72 rows | the single pre-registered test |

The test matrix was also the test set of Phases 3b and 4. The LLM never sees
any result, and the prompts are fixed before any test call, but the reuse is
disclosed.

### 3.5 The tests

- **Primary:** arm B against the SBS. Wilcoxon signed-rank on the 24 function
  means (each averages its 3 rows), two-sided. Pass: p < 0.05 and a positive
  gap closed, the same rule as Phases 3b and 4.
- **Ablation contrasts,** with Bonferroni-corrected p-values (3 tests, alpha
  0.05/3 each):
  - B against D: do the features help?
  - A against D: does the name help?
  - C against A: do the features add anything to the name?
- **Descriptive, no claims:** every arm against the SBS, results by
  dimension, which algorithms each arm picks, agreement across the 3 calls,
  and invalid answers.

### 3.6 Calls and cost

| Stage | Rows | Arms | Calls per row and arm | Calls |
|---|---|---|---|---|
| Test call | 1 | 1 | 1 | 1 |
| Development | 24 | 4 | 3 | 288 |
| Test | 72 | 4 | 3 | 864 |

Retries come on top. Time and cost depend on your API plan's rate limit and
price, which I do not know. The test call measures the time and tokens per
call, and I give an estimate before each batch.

The prompts go to Google's Gemini API. They contain only BBOB function names
(arms A and C), numbers and algorithm names, and no personal data. The key is
read from `.env` only when the LLM is switched on explicitly, and is never
printed or committed.

## 4. Code changes, after approval only

1. Git: untrack `.env` and fix `.gitignore` (section 0).
2. `aad/llm_designer.py`:
   - no `load_dotenv()` at import. The key is read only when the LLM is
     switched on explicitly, so `api_key=None` really means offline;
   - the prompt asks only for the keys in the search space;
   - `temperature`, `seed` and a JSON schema are passed to the API;
   - the answer is checked against the registered algorithms, and an error
     is returned as an error, never replaced by a rule-based pick;
   - description builders for the four arms.

   `experiment_runner.py` and `test_runner.py` create `LLMDesigner()` without
   a key, so after this change they run offline until given one.
3. New `phase5_llm_select.py`. It builds every prompt from the matrices, makes
   the calls with retries and a rate limit, and stores every call (a crash
   journal plus an atomic snapshot). `--dry-run` prints the prompts and counts
   without calling; `--analyse` scores from the store without calling.
4. Tests: no name in arms B and D; parsing and validation; no network call
   while the LLM is off.

## 5. Steps and stop points

| # | Step | Stop and ask? |
|---|---|---|
| 0 | You approve this plan; `.env` untracked and ignored, and the key replaced if needed | yes |
| 1 | Code changes and tests, all offline; show the diff | yes |
| 2 | One test call: confirms the model, measures time and tokens | yes, it sends one prompt to Google |
| 3 | Development run, 288 calls | yes, with a time and cost estimate |
| 4 | Write `docs/phase5_preregistration.md` (prompts, model version and settings frozen) and commit it before the test | yes, commit |
| 5 | Test run, 864 calls | yes |
| 6 | Analyse offline, write `docs/phase5_report.md`, update the docs, back up | yes, commit |

## 6. Files

**Read only:** both matrices, and every Phase 0 to 4 result.

**New:**

- `results/phase5_llm_calls.json`: every request and answer
- `results/phase5a_dev.{json,txt}`
- `results/phase5b_confirm.{json,txt}`
- `docs/phase5_preregistration.md`, `docs/phase5_report.md`
- `phase5_llm_select.py`, `tests/test_phase5_llm.py`

`results/*.jsonl` is ignored by git, so the store is written as `.json`; its
crash journal (`.jsonl`) stays local.

## 7. Not in this plan

- Letting the LLM choose population size, operators or generations. The
  matrices fix them.
- Showing the LLM any result from the matrices (few-shot examples). That is
  transfer, and would need leave-one-function-out.
- Other model families or providers.
- Changing any prompt after seeing a test answer.

## 8. What to expect

Guesses, not results. Phase 3b's feature-based selector landed near the SBS
(+2.1%, p=0.526), so arm B is likely to land near the SBS or below it. Arms A
and C may do better if the LLM remembers published BBOB results, but that
would not carry over to a truly new problem. "Does not beat SBS" is a likely
outcome, and still a finding.

## 9. Decisions needed

1. Untrack `.env` and fix `.gitignore` now, as a separate commit?
   *Recommended.* Replacing the key is your call.
2. Primary arm B (features only)? *Recommended.* Alternative: arm A, today's
   behaviour.
3. K = 3 calls per row and arm, at `temperature=0`? *Recommended.*
   Alternatives: K = 1 (cheaper) or K = 5.
4. Model: keep `gemini-3.6-flash`, or name another.
5. Score an answer that stays invalid as the worst rank? *Recommended.*
6. Only the existing matrices, as in Phase 4? *Recommended.*
7. Are you OK with about 1,150 API calls in total, plus retries?
