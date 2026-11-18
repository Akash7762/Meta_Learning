"""
phase4_replay.py
----------------
Phase 4: at an equal budget, does the meta-learner EA_Designer, which spends
its budget trying algorithms on the problem itself, find better solutions than
running the single best solver (SBS)?

Plan: docs/planning/phase4/phase4_plan.md. Pre-registration: docs/planning/phase4/phase4_preregistration.md.

DATA -- only the matrices built in Phases 2 and 2c, both read-only.
    results/phase2c/phase2c_matrix.json : training, 720 rows, 5 runs per cell.
        Chooses the SBS, leaving out the test function (as in Phase 3b), and
        is the development set (--stage dev).
    results/phase2/phase2_matrix.json  : evaluation, 72 rows, 15 runs per cell.
        The pre-registered test (--stage test).
    The md5 of each file is recorded before and after every run.

REPLAY, NOT RE-RUN.
    Every cell stores independent seeded runs made with population_size=50
    and max_fe=5000 ("ea": max_generations=100, about 5050 evaluations; see
    phase2_build_matrix.py). When the meta-learner asks for a run of
    algorithm A on row r, it gets a stored run of cell (r, A) not yet used in
    this replay, picked at random. That is a run with a fresh seed, because
    the stored runs are exactly such runs. No optimiser is run. If a cell runs
    out (possible only in dev, which stores 5 runs per cell), a used run is
    drawn again and counted as "reused". The test stage requires zero reuse.

EQUAL BUDGET.
    Every method gets BUDGET_RUNS runs of 5000 evaluations per row.
      meta-learner : every run it performs counts. EA_Designer(max_runs=...)
                     spends exactly the budget; asserted on every replay.
      SBS          : the best of BUDGET_RUNS runs of the SBS.
      oracle       : the best of BUDGET_RUNS runs of the row's best algorithm
                     (lowest median). A reference, not a method.
    The expected best of k runs drawn from n stored runs is computed exactly
    over every k-subset, so no draw is left to chance. With k = n it is the
    best stored run.

SEARCH SPACE -- the algorithm only. Population size, generations and
    operators are not searched: the first two would break the equal budget,
    the operators only affect "ea", and the stored runs fix all three. With
    one key, EA_Designer's mutation (_mutate_config) redraws the algorithm
    uniformly for every child, so children are random draws. What adapts is
    which children the surrogate lets through, and which survive.

SCORE.
    Per row, a value f is put on a 0..1 scale: 0 = the best algorithm's median
    on that row, 1 = the worst. gap closed = (SBS - ML) / (SBS - oracle) on
    mean scores. The test pairs 24 function means (the rows of a function are
    averaged first), as in Phase 3b.

SEEDS.
    Replay j of row i uses seed SEED_BASE[stage] + i * replays + j for Python
    `random` (EA_Designer's sampling and variation), NumPy's global RNG, and
    the generator that picks stored runs. All variants share these seeds. The
    surrogate forest has random_state=42 and runs with n_jobs=1, which is
    bit-for-bit repeatable (aad/surrogate_model.py).

Run:
    .venv\\Scripts\\python.exe phase4_replay.py --stage dev --time-rows 16
    .venv\\Scripts\\python.exe phase4_replay.py --stage dev
    .venv\\Scripts\\python.exe phase4_replay.py --stage test
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import comb
from types import SimpleNamespace

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata, wilcoxon

from aad.config_space import EXTENDED_CONFIG_SPACE
from aad.meta_learner import EA_Designer
from phase3b_holdout_confirm import assert_no_leakage

TRAIN_PATH = "results/phase2c/phase2c_matrix.json"
EVAL_PATH = "results/phase2/phase2_matrix.json"
PREREG_PATH = "docs/planning/phase4/phase4_preregistration.md"
OUT_STEM = {"dev": "results/phase4/phase4a_dev", "test": "results/phase4/phase4b_confirm"}

BUDGET_RUNS = 15          # runs per row for every method, each 5000 evaluations
GEN_CAP = 50              # EA_Designer generations: a cap; the budget ends the run
RUNS_PER_EVAL = 1         # 10 configs x 3 runs would spend 30 runs before adapting
REPLAYS = {"dev": 1, "test": 30}
SEED_BASE = {"dev": 10000, "test": 4000}

# Declared in docs/planning/phase4/phase4_plan.md section 3.6. Nothing else is tried.
VARIANTS = {
    "v1": {"population_size": 5, "use_surrogate": True},
    "v2": {"population_size": 5, "use_surrogate": False},
    "v3": {"population_size": 10, "use_surrogate": True},
    "v4": {"population_size": 10, "use_surrogate": False},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_matrix(path):
    """Rows, functions, dims, algorithms, medians and every stored run.

    Refuses missing cells, missing runs and unequal run counts: nothing is
    imputed. Function and dim come from the row key {function}|s{seed}|d{dim}
    and are checked against the cell's own fields where it has them.
    """
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    algos = sorted(d["meta"]["algorithms"])
    rows = sorted(d["features"])
    funcs, dims, med, runs = [], [], [], []
    for r in rows:
        fn, _, dpart = r.split("|")
        dim = int(dpart[1:])
        med_row, run_row = [], []
        for a in algos:
            c = d["cells"].get(r + "|" + a)
            if c is None:
                sys.exit(f"MISSING CELL {r}|{a} in {path} -- refusing to impute.")
            if any(v is None for v in c["fitnesses"]):
                sys.exit(f"MISSING RUN in {r}|{a} ({path}) -- refusing to impute.")
            if c.get("function", fn) != fn or c.get("dim", dim) != dim:
                sys.exit(f"ROW KEY {r} disagrees with its cell fields in {path}.")
            f = np.asarray(c["fitnesses"], float)
            if not np.isclose(np.median(f), c["median"], rtol=1e-12, atol=0.0):
                sys.exit(f"STORED MEDIAN of {r}|{a} does not match its runs.")
            med_row.append(c["median"])
            run_row.append(f)
        funcs.append(fn)
        dims.append(dim)
        med.append(med_row)
        runs.append(run_row)
    counts = {len(f) for row in runs for f in row}
    if len(counts) != 1:
        sys.exit(f"UNEQUAL RUN COUNTS {sorted(counts)} in {path}.")
    return SimpleNamespace(path=path, meta=d["meta"], rows=rows, algos=algos,
                           funcs=np.array(funcs), dims=np.array(dims),
                           med=np.array(med, float), runs=np.array(runs, float),
                           n_runs=counts.pop())


def sbs_by_function(train, test_funcs):
    """Phase 3b's rule: lowest mean rank over the training rows of the OTHER
    functions (phase3b_holdout_confirm.py, lines 200-204)."""
    ranks = np.vstack([rankdata(m) for m in train.med])
    out = {}
    for f in sorted(set(test_funcs.tolist())):
        tr = np.where(train.funcs != f)[0]              # leave the function out
        if len(tr) == len(train.funcs):
            sys.exit(f"Function {f} absent from the training matrix -- stopping.")
        out[f] = int(np.argmin(ranks[tr].mean(axis=0)))
    return out


def expected_best_of_k(values, k):
    """Exact E[min] of k runs drawn without replacement from `values`.

    The i-th smallest value (0-based) is the minimum of a k-subset with
    probability C(n-1-i, k-1) / C(n, k).
    """
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if not 1 <= k <= n:
        raise ValueError(f"k={k} needs 1 <= k <= n={n}")
    w = np.array([comb(n - 1 - i, k - 1) for i in range(n)], float) / comb(n, k)
    return float(np.dot(w, v))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class ReplayEvaluator:
    """Hands out the stored runs of one matrix row instead of running an
    optimiser. Plugs into EA_Designer(evaluator=...)."""

    def __init__(self, runs_by_algo, rng):
        self.runs = runs_by_algo
        self.unused = {a: list(range(len(v))) for a, v in runs_by_algo.items()}
        self.rng = rng
        self.picks = []                   # (algorithm, stored-run index), in order
        self.reused = 0

    def __call__(self, config, problem=None):
        a = config["algorithm"]
        pool = self.unused[a]
        if pool:
            k = pool.pop(int(self.rng.integers(len(pool))))
        else:                             # cell exhausted: fewer stored runs than asked for
            k = int(self.rng.integers(len(self.runs[a])))
            self.reused += 1
        self.picks.append((a, k))
        return float(self.runs[a][k])


def run_replay(row_runs, algos, space, variant, seed,
               budget=BUDGET_RUNS, gen_cap=GEN_CAP):
    """One EA_Designer replay on one row, at exactly `budget` runs."""
    random.seed(seed)
    np.random.seed(seed)
    ev = ReplayEvaluator(dict(zip(algos, row_runs)), np.random.default_rng(seed))
    designer = EA_Designer(
        config_space=space,
        population_size=variant["population_size"],
        generations=gen_cap,
        runs_per_eval=RUNS_PER_EVAL,
        use_surrogate=variant["use_surrogate"],
        evaluator=ev,
        max_runs=budget,
        use_features=False,
        surrogate_n_jobs=1,
    )
    best_cfg, _ = designer.evolve(SimpleNamespace(name="replay"), verbose=False)
    fits = [r["fitness"] for r in designer.run_log]
    if not designer.runs_used == len(fits) == len(ev.picks) == budget:
        raise RuntimeError(f"UNEQUAL BUDGET: seed {seed} spent "
                           f"{designer.runs_used} runs, not {budget}.")
    return {
        "seed": int(seed),
        "best": float(min(fits)),
        "pick": best_cfg["algorithm"],
        "runs_per_algo": dict(Counter(a for a, _ in ev.picks)),
        "reused": int(ev.reused),
        "leftover": int(designer.leftover_runs),
        "generations": int(len(designer.meta_convergence) - 1),
    }


def replay_row(task, algos, space, replays, seed_base):
    """All variants x replays of one row. Seeds depend on (row, replay) only,
    so the result does not depend on the number of worker processes."""
    i, row_runs = task
    return i, {name: [run_replay(row_runs, algos, space, variant,
                                 seed_base + i * replays + j)
                      for j in range(replays)]
               for name, variant in VARIANTS.items()}


def replay_all(mat, row_ids, replays, seed_base, workers):
    space = {"algorithm": list(EXTENDED_CONFIG_SPACE["algorithm"])}
    if sorted(space["algorithm"]) != mat.algos:
        sys.exit("CONFIG SPACE and matrix algorithms differ -- stopping.")
    fn = partial(replay_row, algos=mat.algos, space=space, replays=replays,
                 seed_base=seed_base)
    tasks = [(int(i), mat.runs[i]) for i in row_ids]
    results, t0 = {}, time.perf_counter()
    step = max(1, len(tasks) // 10)

    def collect(pairs):
        for n_done, (i, out) in enumerate(pairs, 1):
            results[i] = out
            if n_done % step == 0 or n_done == len(tasks):
                print(f"  rows {n_done:>4}/{len(tasks)}  "
                      f"{time.perf_counter() - t0:7.1f} s", flush=True)

    if workers <= 1:
        collect(map(fn, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            collect(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 8))))
    return results, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def scale(mat):
    lo, hi = mat.med.min(axis=1), mat.med.max(axis=1)
    return lo, np.where(hi - lo == 0, 1.0, hi - lo)


def variant_rows(mat, results, v):
    """Per row: mean 0..1 score over the replays, and the replays themselves."""
    lo, span = scale(mat)
    score = np.array([np.mean([(r["best"] - lo[i]) / span[i] for r in results[i][v]])
                      for i in range(len(mat.rows))])
    reps = [r for i in range(len(mat.rows)) for r in results[i][v]]
    return score, reps


def usage(reps):
    return {
        "reused_draws": int(sum(r["reused"] for r in reps)),
        "draws": int(len(reps) * BUDGET_RUNS),
        "replays_with_leftover": int(sum(r["leftover"] > 0 for r in reps)),
        "mean_distinct_algorithms": float(np.mean([len(r["runs_per_algo"]) for r in reps])),
        "mean_generations": float(np.mean([r["generations"] for r in reps])),
    }


def env_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform()}


def write_outputs(stem, lines, payload):
    with open(stem + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("\n".join(lines))
    print(f"\n  wrote {stem}.txt and {stem}.json")


def settings_lines(stage, replays):
    return [
        f"  budget          : {BUDGET_RUNS} runs per row for every method "
        f"(each 5000 evaluations; 'ea' about 5050)",
        f"  replays per row : {replays}  (seed = {SEED_BASE[stage]} + row x "
        f"{replays} + replay)",
        f"  EA_Designer     : search space = the 22 algorithms, runs_per_eval="
        f"{RUNS_PER_EVAL}, generation cap {GEN_CAP},",
        "                    use_features=False, surrogate n_jobs=1, "
        "fresh object per replay",
    ]


def describe(mat):
    seeds = sorted({int(r.split("|")[1][1:]) for r in mat.rows})     # from the row keys
    return (f"{len(mat.rows)} rows = {len(set(mat.funcs.tolist()))} functions x "
            f"dims {sorted(set(mat.dims.tolist()))} x seeds {seeds}; "
            f"{mat.n_runs} stored runs per cell")


def run_dev(args):
    mat = load_matrix(TRAIN_PATH)
    replays = REPLAYS["dev"]
    md5_before = md5(TRAIN_PATH)
    print(f"  dev replay on {TRAIN_PATH}: {describe(mat)}", flush=True)
    results, secs = replay_all(mat, range(len(mat.rows)), replays,
                               SEED_BASE["dev"], args.workers)
    md5_after = md5(TRAIN_PATH)
    if md5_after != md5_before:
        sys.exit("TRAINING MATRIX CHANGED during the run -- stopping.")

    ufuncs = sorted(set(mat.funcs.tolist()))
    udims = sorted(set(mat.dims.tolist()))
    summary = {}
    for v in VARIANTS:
        score, reps = variant_rows(mat, results, v)
        fm = np.array([score[mat.funcs == f].mean() for f in ufuncs])
        summary[v] = {
            **VARIANTS[v],
            "mean_score": float(fm.mean()),
            "by_dim": {str(dm): float(score[mat.dims == dm].mean()) for dm in udims},
            "function_means": dict(zip(ufuncs, fm.tolist())),
            **usage(reps),
        }
    chosen = min(VARIANTS, key=lambda v: (summary[v]["mean_score"], v))

    L = ["=" * 78,
         "  PHASE 4a -- DEVELOPMENT REPLAY (exploratory; chooses the variant for 4b)",
         "=" * 78,
         f"  matrix          : {TRAIN_PATH}  (md5 {md5_before}, unchanged after)",
         f"  rows            : {describe(mat)}",
         *settings_lines("dev", replays),
         f"  wall time       : {secs:.1f} s, {args.workers} parallel workers",
         "",
         "  mean score: 0 = best algorithm median on the row, 1 = worst; lower is better",
         "  variant  pop  surrogate   score   " + "  ".join(f"d{dm:<5}" for dm in udims)
         + "  distinct  reused/draws  leftover  gens",
         ]
    for v in VARIANTS:
        s = summary[v]
        L.append(f"  {v:<7} {s['population_size']:>4}  {str(s['use_surrogate']):<9}"
                 f" {s['mean_score']:7.4f}   "
                 + "  ".join(f"{s['by_dim'][str(dm)]:6.4f}" for dm in udims)
                 + f"  {s['mean_distinct_algorithms']:8.2f}  "
                 f"{s['reused_draws']:>5}/{s['draws']:<6}  "
                 f"{s['replays_with_leftover']:>8}  {s['mean_generations']:5.2f}")
    L += ["",
          f"  CHOSEN FOR PHASE 4b: {chosen} (lowest mean score over "
          f"{len(ufuncs)} function means)",
          "",
          "  NOTES",
          "  - Exploratory. No claim is made from these numbers.",
          f"  - The training matrix stores {mat.n_runs} runs per cell, fewer than "
          f"the budget of {BUDGET_RUNS}.",
          "    A replay that asks for more runs of one algorithm than are stored "
          "gets a used",
          "    run again ('reused'). That can only make a variant look worse, "
          "never better.",
          "  - No SBS comparison here: the SBS's best of 15 needs 15 stored runs "
          "per cell.",
          "  - v2 and v4 differ only in population size with the surrogate off. "
          "Every candidate",
          "    is then a uniform random algorithm and is run, so both are "
          f"{BUDGET_RUNS} random draws.",
          ]
    payload = {
        "meta": {"stage": "dev", "matrix": TRAIN_PATH, "matrix_md5": md5_before,
                 "budget_runs": BUDGET_RUNS, "gen_cap": GEN_CAP,
                 "runs_per_eval": RUNS_PER_EVAL, "replays": replays,
                 "seed_base": SEED_BASE["dev"], "variants": VARIANTS,
                 "workers": args.workers, "wall_seconds": round(secs, 1),
                 "env": env_versions()},
        "chosen_variant": chosen,
        "summary": summary,
        "rows": mat.rows,
        "replays": {mat.rows[i]: results[i] for i in range(len(mat.rows))},
    }
    write_outputs(OUT_STEM["dev"], L, payload)


def run_test(args):
    if not os.path.exists(PREREG_PATH):
        sys.exit(f"Write and commit {PREREG_PATH} before the test is run.")
    dev_path = OUT_STEM["dev"] + ".json"
    if not os.path.exists(dev_path):
        sys.exit("Run --stage dev first: the test uses the variant it chose.")
    with open(dev_path, encoding="utf-8") as fh:
        dev = json.load(fh)
    primary = dev["chosen_variant"]

    assert_no_leakage(TRAIN_PATH, EVAL_PATH)
    md5_before = {p: md5(p) for p in (TRAIN_PATH, EVAL_PATH)}
    if dev["meta"]["matrix_md5"] != md5_before[TRAIN_PATH]:
        sys.exit("TRAINING MATRIX CHANGED since the dev stage -- stopping.")
    train, mat = load_matrix(TRAIN_PATH), load_matrix(EVAL_PATH)
    if train.algos != mat.algos:
        sys.exit("ALGORITHM SETS DIFFER between the matrices -- stopping.")
    if mat.n_runs < BUDGET_RUNS:
        sys.exit(f"Test matrix stores {mat.n_runs} runs per cell; the exact "
                 f"replay needs at least {BUDGET_RUNS}.")
    replays = REPLAYS["test"]
    print(f"  test replay on {EVAL_PATH}: {describe(mat)}", flush=True)
    results, secs = replay_all(mat, range(len(mat.rows)), replays,
                               SEED_BASE["test"], args.workers)
    md5_after = {p: md5(p) for p in (TRAIN_PATH, EVAL_PATH)}
    if md5_after != md5_before:
        sys.exit("A MATRIX CHANGED during the run -- stopping.")

    n = len(mat.rows)
    lo, span = scale(mat)
    sbs = sbs_by_function(train, mat.funcs)
    sbs_j = np.array([sbs[f] for f in mat.funcs])
    vbs_j = mat.med.argmin(axis=1)
    sbs_row = np.array([(expected_best_of_k(mat.runs[i, sbs_j[i]], BUDGET_RUNS) - lo[i])
                        / span[i] for i in range(n)])
    ora_row = np.array([(expected_best_of_k(mat.runs[i, vbs_j[i]], BUDGET_RUNS) - lo[i])
                        / span[i] for i in range(n)])
    ufuncs = sorted(set(mat.funcs.tolist()))
    udims = sorted(set(mat.dims.tolist()))

    def fmean(x):
        return np.array([x[mat.funcs == f].mean() for f in ufuncs])

    def by_dim(x):
        return {str(dm): float(x[mat.dims == dm].mean()) for dm in udims}

    gap = float(sbs_row.mean() - ora_row.mean())
    out, ml_rows = {}, {}
    for v in VARIANTS:
        score, reps = variant_rows(mat, results, v)
        u = usage(reps)
        if u["reused_draws"]:
            sys.exit(f"{v}: {u['reused_draws']} REUSED stored runs in the test "
                     "stage -- the replay would not be exact. Stopping.")
        fm_ml, fm_sbs = fmean(score), fmean(sbs_row)
        d = fm_ml - fm_sbs
        if np.all(d == 0):
            stat, pv = float("nan"), float("nan")
        else:
            stat, pv = wilcoxon(fm_ml, fm_sbs)
        alloc = Counter()
        for r in reps:
            alloc.update(r["runs_per_algo"])
        out[v] = {
            **VARIANTS[v],
            "mean_score": float(score.mean()),
            "by_dim": by_dim(score),
            "gap_closed_pct": float((sbs_row.mean() - score.mean()) / gap * 100.0)
                              if gap > 0 else float("nan"),
            "wilcoxon_statistic": float(stat), "p": float(pv),
            "functions_better": int((d < 0).sum()),
            "functions_worse": int((d > 0).sum()),
            "functions_tied": int((d == 0).sum()),
            "final_pick_is_vbs": float(np.mean(
                [results[i][v][j]["pick"] == mat.algos[vbs_j[i]]
                 for i in range(n) for j in range(replays)])),
            "mean_runs_per_replay_by_algorithm": {
                a: alloc[a] / len(reps) for a in sorted(alloc, key=lambda a: -alloc[a])},
            **u,
        }
        ml_rows[v] = score

    p = out[primary]
    passed = bool(p["p"] < 0.05 and p["gap_closed_pct"] > 0)
    sbs_counts = Counter(mat.algos[j] for j in sbs_j)

    L = ["=" * 78,
         "  PHASE 4b -- PRE-REGISTERED TEST (docs/planning/phase4/phase4_preregistration.md)",
         "=" * 78,
         f"  train (SBS)     : {TRAIN_PATH}  (md5 {md5_before[TRAIN_PATH]})",
         f"  test            : {EVAL_PATH}  (md5 {md5_before[EVAL_PATH]})",
         "                    both unchanged after the run",
         f"  rows            : {describe(mat)}",
         *settings_lines("test", replays),
         f"  primary variant : {primary} (chosen in Phase 4a, {dev_path})",
         f"  SBS per row     : {dict(sbs_counts.most_common())} (training ranks, "
         "test function left out)",
         f"  wall time       : {secs:.1f} s, {args.workers} parallel workers",
         "",
         "  mean score: 0 = best algorithm median on the row, 1 = worst; lower is better",
         f"  {'method':<24}{'all':>8}  " + "".join(f"{'d' + str(dm):>9}" for dm in udims),
         ]

    def score_line(label, total, dims_vals):
        return f"  {label:<24}{total:8.4f}  " + "".join(f"{x:9.4f}" for x in dims_vals)

    L.append(score_line(f"oracle (VBS), best of {BUDGET_RUNS}", ora_row.mean(),
                        by_dim(ora_row).values()))
    L.append(score_line(f"SBS, best of {BUDGET_RUNS}", sbs_row.mean(),
                        by_dim(sbs_row).values()))
    for v in VARIANTS:
        tag = f"{v} (pop {VARIANTS[v]['population_size']}, " \
              f"surr {'on' if VARIANTS[v]['use_surrogate'] else 'off'})"
        L.append(score_line(tag, out[v]["mean_score"], out[v]["by_dim"].values()))
    L += ["",
          f"  PRIMARY: {primary} vs SBS, paired over {len(ufuncs)} function means",
          f"    gap closed      : {p['gap_closed_pct']:+.1f}%",
          f"    Wilcoxon        : statistic {p['wilcoxon_statistic']}, "
          f"p = {p['p']!r} (two-sided)",
          "    signature       : wilcoxon(x, y=None, zero_method='wilcox', "
          "correction=False,",
          "                      alternative='two-sided', method='auto', ...)",
          f"    functions       : {p['functions_better']} better / "
          f"{p['functions_worse']} worse / {p['functions_tied']} tied",
          "    rule            : p < 0.05 AND gap closed > 0",
          f"    VERDICT         : {'BEATS SBS' if passed else 'DOES NOT BEAT SBS'}",
          "",
          "  SECONDARY (descriptive, no claims)",
          "  variant  gap closed        p   b/w/t     pick=VBS  distinct  leftover  gens",
          ]
    for v in VARIANTS:
        o = out[v]
        L.append(f"  {v:<7} {o['gap_closed_pct']:+9.1f}%  {o['p']:8.4f}   "
                 f"{o['functions_better']}/{o['functions_worse']}/{o['functions_tied']:<6}"
                 f" {o['final_pick_is_vbs']:8.3f}  {o['mean_distinct_algorithms']:8.2f}"
                 f"  {o['replays_with_leftover']:>8}  {o['mean_generations']:5.2f}")
    top = list(p["mean_runs_per_replay_by_algorithm"].items())[:8]
    L += ["",
          f"  how {primary} spent its {BUDGET_RUNS} runs (mean per replay, top 8): "
          + ", ".join(f"{a} {x:.2f}" for a, x in top),
          "",
          "  NOTES",
          "  - The test matrix was used before: Phase 3b's test set, and studied "
          "since Phase 2.",
          "  - Only the primary comparison is a test. Every other number is "
          "descriptive.",
          f"  - 24 functions bound the test, so a small benefit cannot be ruled out.",
          ]
    detail = [{"row": mat.rows[i], "function": str(mat.funcs[i]),
               "dim": int(mat.dims[i]), "sbs": mat.algos[sbs_j[i]],
               "vbs": mat.algos[vbs_j[i]], "sbs_score": float(sbs_row[i]),
               "oracle_score": float(ora_row[i]),
               **{f"{v}_score": float(ml_rows[v][i]) for v in VARIANTS}}
              for i in range(n)]
    payload = {
        "meta": {"stage": "test", "train_matrix": TRAIN_PATH, "eval_matrix": EVAL_PATH,
                 "md5": md5_before, "dev_result": dev_path, "dev_md5": md5(dev_path),
                 "preregistration": PREREG_PATH, "preregistration_md5": md5(PREREG_PATH),
                 "budget_runs": BUDGET_RUNS, "gen_cap": GEN_CAP,
                 "runs_per_eval": RUNS_PER_EVAL, "replays": replays,
                 "seed_base": SEED_BASE["test"], "variants": VARIANTS,
                 "workers": args.workers, "wall_seconds": round(secs, 1),
                 "env": env_versions()},
        "primary_variant": primary,
        "verdict": "BEATS SBS" if passed else "DOES NOT BEAT SBS",
        "sbs_mean_score": float(sbs_row.mean()),
        "oracle_mean_score": float(ora_row.mean()),
        "sbs_by_dim": by_dim(sbs_row), "oracle_by_dim": by_dim(ora_row),
        "variants": out,
        "detail": detail,
        "replays": {mat.rows[i]: results[i] for i in range(n)},
    }
    write_outputs(OUT_STEM["test"], L, payload)


def time_rows(args):
    """Time K training rows with this stage's settings. Prints no scores and
    writes nothing, so it cannot leak a result."""
    mat = load_matrix(TRAIN_PATH)
    k = min(args.time_rows, len(mat.rows))
    ids = np.unique(np.linspace(0, len(mat.rows) - 1, k).astype(int))
    replays = REPLAYS[args.stage]
    n_target = len(mat.rows) if args.stage == "dev" else len(load_matrix(EVAL_PATH).rows)
    _, secs = replay_all(mat, ids, replays, SEED_BASE[args.stage], args.workers)
    print(f"  timed {len(ids)} rows x {replays} replays x {len(VARIANTS)} variants: "
          f"{secs:.1f} s with {args.workers} workers")
    print(f"  projected for the {args.stage} stage ({n_target} rows): "
          f"{secs * n_target / len(ids):.0f} s")


def main():
    ap = argparse.ArgumentParser(description="Phase 4 replay: EA_Designer vs SBS "
                                             "at an equal budget.")
    ap.add_argument("--stage", choices=("dev", "test"), required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--time-rows", type=int, default=0,
                    help="time K training rows with this stage's settings and "
                         "print a projection; prints no scores, writes nothing")
    args = ap.parse_args()
    if args.time_rows:
        time_rows(args)
    elif args.stage == "dev":
        run_dev(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
