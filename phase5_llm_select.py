"""
phase5_llm_select.py
--------------------
Phase 5: told about an unseen problem, can an LLM pick an algorithm that beats
always running the single best solver (SBS)? And which information helps it?

Plan: docs/planning/phase5/phase5_plan.md. Pre-registration: docs/planning/phase5/phase5_preregistration.md.

ARMS -- what the LLM is told (LLMDesigner.describe_for_selection):
    A name     : function name, dimension, search box, budget
    B features : the 9 stored landscape features, dimension, search box and
                 budget; no name. THE PRIMARY ARM.
    C both     : A and B together
    D nothing  : dimension, search box and budget only
  Every arm gets the same instructions and the same list of the 22 algorithms.

DATA -- the existing matrices only, read-only; md5 checked before and after.
    dev  : 24 training rows, every function at seed 42 and dim 10
           (results/phase2c/phase2c_matrix.json)
    test : the 72-row evaluation matrix (results/phase2/phase2_matrix.json)
  The features come from the matrices: the inputs the Phase 3 selector used.
  The SBS is chosen from training ranks with the row's function left out
  (Phase 3b's rule). No optimiser is run: a pick is scored by its rank on the
  row, from the stored medians, as in Phase 3b.

CALLS -- K identical requests per row and arm: temperature 0, one fixed seed,
  and a JSON schema that allows only the 22 names. An unusable answer is asked
  again up to MAX_REASK times; if it stays unusable it scores the worst rank.
  A network or API error is retried with back-off; if it persists the run
  stops, and the same command resumes from the journal.

STORE -- every call goes to results/phase5/phase5_calls_<stage>.jsonl (journal, local
  only) and results/phase5/phase5_calls_<stage>.json (snapshot, tracked). --analyse
  reads only the snapshot, so the analysis is exact and makes no API call.

The API key is read only through LLMDesigner.from_env() and is never printed.

Run:
    python phase5_llm_select.py --stage dev --dry-run   # prompts, counts; no call
    python phase5_llm_select.py --test-call              # exactly one call
    python phase5_llm_select.py --stage dev
    python phase5_llm_select.py --stage dev --analyse
    python phase5_llm_select.py --stage test
    python phase5_llm_select.py --stage test --analyse
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.stats import rankdata, wilcoxon

import aad
from aad.config_space import EXTENDED_CONFIG_SPACE
from aad.llm_designer import DEFAULT_MODEL, LLMDesigner

TRAIN_PATH = "results/phase2c/phase2c_matrix.json"
EVAL_PATH = "results/phase2/phase2_matrix.json"
PREREG_PATH = "docs/planning/phase5/phase5_preregistration.md"
STORE = "results/phase5/phase5_calls_{stage}"         # + .jsonl journal / .json snapshot
OUT_STEM = {"dev": "results/phase5/phase5a_dev", "test": "results/phase5/phase5b_confirm"}

ARMS = {"A": {"name": True, "features": False},
        "B": {"name": False, "features": True},
        "C": {"name": True, "features": True},
        "D": {"name": False, "features": False}}
PRIMARY = "B"
CONTRASTS = [("B", "D"), ("A", "D"), ("C", "A")]     # Bonferroni over these 3
K = 5                      # identical requests per row and arm
TEMPERATURE = 0.0
LLM_SEED = 20261008        # one seed for every request
MAX_REASK = 2              # extra asks after an unusable answer
BACKOFF = (10, 30, 60, 120, 300)   # seconds before each transport retry
MIN_INTERVAL = 4.0         # default seconds between call starts
DEV_ROWS = ("s42", "d10")  # dev: every function at seed 42, dim 10


# ---------------------------------------------------------------------------
# Data and prompts
# ---------------------------------------------------------------------------

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_matrix(path):
    """Rows, functions, dims, seeds, sorted algorithms, medians, features."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    algos = sorted(d["meta"]["algorithms"])
    rows = sorted(d["features"])
    med = np.full((len(rows), len(algos)), np.nan)
    for i, r in enumerate(rows):
        for j, a in enumerate(algos):
            c = d["cells"].get(r + "|" + a)
            if c is None or c.get("median") is None:
                sys.exit(f"MISSING CELL {r}|{a} in {path} -- refusing to impute.")
            med[i, j] = c["median"]
    parts = [r.split("|") for r in rows]
    return SimpleNamespace(
        path=path, meta=d["meta"], rows=rows, algos=algos, med=med,
        funcs=np.array([p[0] for p in parts]),
        seeds=np.array([int(p[1][1:]) for p in parts]),
        dims=np.array([int(p[2][1:]) for p in parts]),
        features={r: d["features"][r] for r in rows})


def stage_rows(stage, train, test):
    if stage == "dev":
        keep = [i for i, r in enumerate(train.rows)
                if r.split("|")[1:] == list(DEV_ROWS)]
        return train, keep
    return test, list(range(len(test.rows)))


def selection_space(mat):
    space = {"algorithm": list(EXTENDED_CONFIG_SPACE["algorithm"])}
    if sorted(space["algorithm"]) != mat.algos:
        sys.exit("CONFIG SPACE and matrix algorithms differ -- stopping.")
    return space


def build_prompts(mat, ids, designer):
    """{"row|arm": prompt}. Name and bounds come from the problem object;
    features, budget and sample count from the matrix."""
    meta = mat.meta
    prompts = {}
    for i in ids:
        r = mat.rows[i]
        problem = getattr(aad, mat.funcs[i])(dim=int(mat.dims[i]), seed=int(mat.seeds[i]))
        lb, ub = problem.bounds()
        if len(set(np.round(lb, 12))) != 1 or len(set(np.round(ub, 12))) != 1:
            sys.exit(f"{r}: bounds differ between coordinates -- the prompt "
                     "assumes one search box for all.")
        for arm, show in ARMS.items():
            text = LLMDesigner.describe_for_selection(
                dim=int(mat.dims[i]), lower=float(lb[0]), upper=float(ub[0]),
                budget_fe=meta["max_fe"], population=meta["population_size"],
                name=problem.name if show["name"] else None,
                features=mat.features[r] if show["features"] else None,
                n_samples=meta["feature_samples"])
            prompts[f"{r}|{arm}"] = designer.selection_prompt(text), text
    return prompts


def leaks(prompts, funcs):
    """Arm-B and arm-D prompts that mention any function name or 'BBOB'."""
    names = sorted(set(funcs)) + ["BBOB"]
    short = [n.split("_", 2)[-1] for n in sorted(set(funcs))]   # e.g. "Sphere"
    bad = []
    for key, (prompt, _) in prompts.items():
        if key.rsplit("|", 1)[1] in ("B", "D"):
            if any(n in prompt for n in names + short):
                bad.append(key)
    return bad


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

class Pacer:
    """At least `min_interval` seconds between call starts."""

    def __init__(self, min_interval):
        self.min_interval, self.last = min_interval, -1e9

    def __call__(self):
        wait = self.last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()


def retryable(exc):
    """Rate limits, server errors and network failures are retried; any other
    API error (bad key, unknown model, bad request) stops the run."""
    from google.genai import errors
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429
    return isinstance(exc, OSError) or type(exc).__module__.startswith("httpx")


def call_with_retries(designer, problem_text, pace, sleep=time.sleep):
    last = None
    for n, wait in enumerate((0,) + BACKOFF):
        if wait:
            print(f"    waiting {wait} s before retry {n}/{len(BACKOFF)}", flush=True)
            sleep(wait)
        pace()
        t0 = time.monotonic()
        try:
            rec = designer.select_algorithm(problem_text)
        except Exception as exc:                     # noqa: BLE001
            if not retryable(exc):
                raise
            last = exc
            print(f"    transport error: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            continue
        rec["seconds"] = round(time.monotonic() - t0, 3)
        rec["transport_retries"] = n
        return rec
    raise RuntimeError(f"API still failing after {len(BACKOFF)} retries: {last}")


def run_slot(designer, problem_text, pace, sleep=time.sleep):
    """One (row, arm, k) slot: ask, and ask again while the answer is
    unusable, up to MAX_REASK times. Nothing is replaced."""
    attempts = []
    for attempt in range(1 + MAX_REASK):
        rec = call_with_retries(designer, problem_text, pace, sleep)
        rec["attempt"] = attempt
        attempts.append(rec)
        if rec["valid"]:
            break
    return {"pick": attempts[-1]["algorithm"], "attempts": attempts}


def run_header(stage, model, prompts):
    blob = json.dumps({k: v[0] for k, v in sorted(prompts.items())}, sort_keys=True)
    return {"stage": stage, "model": model, "temperature": TEMPERATURE,
            "seed": LLM_SEED, "K": K, "max_reask": MAX_REASK,
            "prompts_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}


def write_snapshot(path, header, prompts, slots, extra):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"meta": {**header, **extra},
                   "prompts": {k: v[0] for k, v in sorted(prompts.items())},
                   "slots": dict(sorted(slots.items()))}, fh, indent=1)
    os.replace(tmp, path)


def env_versions():
    import google.genai
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "google-genai": google.genai.__version__,
            "platform": platform.platform()}


def run_calls(args, train, test):
    if args.stage == "test":
        if not os.path.exists(PREREG_PATH):
            sys.exit(f"Write and commit {PREREG_PATH} before the test calls.")
        if not os.path.exists(STORE.format(stage="dev") + ".json"):
            sys.exit("Run the dev stage first.")
        with open(STORE.format(stage="dev") + ".json", encoding="utf-8") as fh:
            dev_model = json.load(fh)["meta"]["model"]
        if dev_model != args.model:
            sys.exit(f"MODEL MISMATCH: the dev stage used {dev_model}, this run "
                     f"asks for {args.model}. Pass --model {dev_model}.")
    mat, ids = stage_rows(args.stage, train, test)
    md5_before = {p: md5(p) for p in (TRAIN_PATH, EVAL_PATH)}
    designer = LLMDesigner.from_env(model=args.model, config_space=selection_space(mat),
                                    temperature=TEMPERATURE, seed=LLM_SEED)
    prompts = build_prompts(mat, ids, designer)
    bad = leaks(prompts, mat.funcs)
    if bad:
        sys.exit(f"NAME LEAK in {len(bad)} arm-B/D prompts, e.g. {bad[0]} -- stopping.")
    header = run_header(args.stage, args.model, prompts)
    stem = STORE.format(stage=args.stage)
    os.makedirs(os.path.dirname(stem), exist_ok=True)

    slots = {}
    if os.path.exists(stem + ".jsonl"):                # resume
        with open(stem + ".jsonl", encoding="utf-8") as fh:
            first = json.loads(fh.readline())
            if first != {"header": header}:
                sys.exit(f"REFUSING TO RESUME: {stem}.jsonl was started with other "
                         "settings or prompts. Move it away to start afresh.")
            for line in fh:
                rec = json.loads(line)
                slots[rec["key"]] = rec["slot"]
        print(f"  resuming: {len(slots)} slots already done", flush=True)
    else:
        with open(stem + ".jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": header}) + "\n")

    todo = [(key, k) for key in sorted(prompts) for k in range(K)
            if f"{key}|{k}" not in slots]
    print(f"  {args.stage}: {len(ids)} rows x {len(ARMS)} arms x K={K} = "
          f"{len(prompts) * K} slots; {len(todo)} to do; model {args.model}; "
          f"at least {args.min_interval} s between calls", flush=True)
    pace, t0 = Pacer(args.min_interval), time.monotonic()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for n, (key, k) in enumerate(todo, 1):
        slot = run_slot(designer, prompts[key][1], pace)
        slots[f"{key}|{k}"] = slot
        with open(stem + ".jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": f"{key}|{k}", "slot": slot}) + "\n")
        if n % 20 == 0 or n == len(todo):
            write_snapshot(stem + ".json", header, prompts, slots,
                           {"started": started, "env": env_versions()})
            print(f"  slots {n:>5}/{len(todo)}  {time.monotonic() - t0:8.1f} s", flush=True)
    write_snapshot(stem + ".json", header, prompts, slots,
                   {"started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "env": env_versions()})
    if {p: md5(p) for p in (TRAIN_PATH, EVAL_PATH)} != md5_before:
        sys.exit("A MATRIX CHANGED during the run -- stopping.")
    print(f"  wrote {stem}.json ({len(slots)} slots)")


def dry_run(args, train, test):
    """Every prompt is built and checked; nothing is sent, no key is read."""
    mat, ids = stage_rows(args.stage, train, test)
    designer = LLMDesigner(config_space=selection_space(mat))     # offline
    prompts = build_prompts(mat, ids, designer)
    bad = leaks(prompts, mat.funcs)
    n_slots = len(prompts) * K
    print(f"  {args.stage}: {len(ids)} rows x {len(ARMS)} arms x K={K} = {n_slots} "
          f"calls, at most {n_slots * (1 + MAX_REASK)} with every re-ask")
    print(f"  name leaks in arm-B/D prompts: {len(bad)}")
    r = mat.rows[ids[0]]
    for arm in ARMS:
        print(f"\n----- arm {arm}, row {r} -----\n{prompts[f'{r}|{arm}'][0]}")


def test_call(args, train, test):
    mat, ids = stage_rows("dev", train, test)
    designer = LLMDesigner.from_env(model=args.model, config_space=selection_space(mat),
                                    temperature=TEMPERATURE, seed=LLM_SEED)
    prompts = build_prompts(mat, ids[:1], designer)
    key = f"{mat.rows[ids[0]]}|{PRIMARY}"
    print(f"  one call: model {args.model}, row {key}, request time limit "
          f"{designer.timeout_s:.0f} s; sending ...", flush=True)
    rec = call_with_retries(designer, prompts[key][1], Pacer(0))
    for f in ("raw", "algorithm", "valid", "error", "model_version", "tokens", "seconds"):
        print(f"  {f:<14}: {rec.get(f)}")


# ---------------------------------------------------------------------------
# Analysis (offline)
# ---------------------------------------------------------------------------

def analyse(args, train, test):
    stem = STORE.format(stage=args.stage)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".json", encoding="utf-8") as fh:
        store = json.load(fh)
    mat, ids = stage_rows(args.stage, train, test)
    n_alg = len(mat.algos)
    ranks = np.vstack([rankdata(m) for m in mat.med])
    tr_ranks = np.vstack([rankdata(m) for m in train.med])
    col = {a: j for j, a in enumerate(mat.algos)}

    missing = [f"{mat.rows[i]}|{arm}|{k}" for i in ids for arm in ARMS for k in range(K)
               if f"{mat.rows[i]}|{arm}|{k}" not in store["slots"]]
    if missing:
        sys.exit(f"{len(missing)} slots missing from {stem}.json, e.g. {missing[0]} "
                 "-- refusing to score an incomplete run.")

    sbs = {f: int(np.argmin(tr_ranks[train.funcs != f].mean(axis=0)))
           for f in sorted(set(mat.funcs[ids].tolist()))}
    rows = [mat.rows[i] for i in ids]
    funcs, dims = mat.funcs[ids], mat.dims[ids]
    sbs_r = np.array([ranks[i, sbs[mat.funcs[i]]] for i in ids])
    vbs_r = np.array([ranks[i].min() for i in ids])
    rnd_r = np.array([ranks[i].mean() for i in ids])

    arm_r, picks, invalid, agree, tokens, secs, versions = {}, {}, {}, {}, {}, [], Counter()
    for arm in ARMS:
        per_row, picks[arm], invalid[arm], agree[arm] = [], Counter(), 0, 0
        tokens[arm] = 0
        for i, r in zip(ids, rows):
            got = [store["slots"][f"{r}|{arm}|{k}"] for k in range(K)]
            names = [g["pick"] for g in got]
            invalid[arm] += sum(p is None for p in names)
            agree[arm] += len(set(names)) == 1
            picks[arm].update(p if p is not None else "(invalid)" for p in names)
            per_row.append(np.mean([ranks[i, col[p]] if p is not None else n_alg
                                    for p in names]))
            for g in got:
                for a in g["attempts"]:
                    tokens[arm] += (a.get("tokens") or {}).get("total_token_count") or 0
                    secs.append(a.get("seconds") or 0.0)
                    versions[a.get("model_version")] += 1
        arm_r[arm] = np.array(per_row)

    ufuncs = sorted(set(funcs.tolist()))

    def fmean(x):
        return np.array([x[funcs == f].mean() for f in ufuncs])

    def test_pair(x, y):
        d = fmean(x) - fmean(y)
        if np.all(d == 0):
            return float("nan"), float("nan"), d
        st, p = wilcoxon(fmean(x), fmean(y))
        return float(st), float(p), d

    gap = float(sbs_r.mean() - vbs_r.mean())
    res = {}
    for arm in ARMS:
        st, p, d = test_pair(arm_r[arm], sbs_r)
        res[arm] = {"mean_rank": float(arm_r[arm].mean()),
                    "gap_closed_pct": float((sbs_r.mean() - arm_r[arm].mean()) / gap * 100)
                    if gap > 0 else float("nan"),
                    "wilcoxon_statistic": st, "p": p,
                    "functions_better": int((d < 0).sum()), "functions_worse": int((d > 0).sum()),
                    "functions_tied": int((d == 0).sum()),
                    "invalid_slots": int(invalid[arm]), "rows_all_K_agree": int(agree[arm]),
                    "top_picks": picks[arm].most_common(6), "total_tokens": int(tokens[arm]),
                    "by_dim": {str(dm): float(arm_r[arm][dims == dm].mean())
                               for dm in sorted(set(dims.tolist()))}}
    contrasts = []
    for a1, a2 in CONTRASTS:
        st, p, d = test_pair(arm_r[a1], arm_r[a2])
        contrasts.append({"pair": f"{a1} vs {a2}", "statistic": st, "p": p,
                          "p_bonferroni": min(1.0, p * len(CONTRASTS)) if p == p else p,
                          "mean_rank_difference": float(arm_r[a1].mean() - arm_r[a2].mean())})
    prim = res[PRIMARY]
    passed = bool(prim["p"] < 0.05 and prim["gap_closed_pct"] > 0)

    title = ("PHASE 5a -- DEVELOPMENT CALLS (exploratory; no claim)" if args.stage == "dev"
             else "PHASE 5b -- PRE-REGISTERED TEST (docs/planning/phase5/phase5_preregistration.md)")
    L = ["=" * 78, "  " + title, "=" * 78,
         f"  matrix          : {mat.path}  ({len(ids)} rows; dims "
         f"{sorted(set(dims.tolist()))}; {len(ufuncs)} functions)",
         f"  model           : {store['meta']['model']}; versions reported "
         f"{dict(versions)}",
         f"  settings        : K={store['meta']['K']} identical requests per row and arm, "
         f"temperature {store['meta']['temperature']}, seed {store['meta']['seed']}, "
         f"up to {store['meta']['max_reask']} re-asks",
         f"  SBS per row     : {dict(Counter(mat.algos[sbs[f]] for f in funcs).most_common())}",
         "",
         "  MEAN RANK of the pick (1 = best of 22, lower is better)",
         f"  VBS (oracle)        {vbs_r.mean():7.4f}",
         f"  SBS                 {sbs_r.mean():7.4f}",
         f"  random              {rnd_r.mean():7.4f}"]
    for arm in ARMS:
        o = res[arm]
        L.append(f"  arm {arm} {'(primary)' if arm == PRIMARY else '         '}  "
                 f"{o['mean_rank']:7.4f}   gap closed {o['gap_closed_pct']:+7.1f}%   "
                 f"p={o['p']:.4f}  b/w/t {o['functions_better']}/{o['functions_worse']}/"
                 f"{o['functions_tied']}  invalid {o['invalid_slots']}  "
                 f"all-K-agree {o['rows_all_K_agree']}/{len(ids)}")
    L += ["",
          f"  PRIMARY: arm {PRIMARY} vs SBS, Wilcoxon over {len(ufuncs)} function means",
          f"    gap closed : {prim['gap_closed_pct']:+.1f}%",
          f"    Wilcoxon   : statistic {prim['wilcoxon_statistic']}, p = {prim['p']!r} (two-sided)",
          "    rule       : p < 0.05 AND gap closed > 0",
          f"    VERDICT    : {'BEATS SBS' if passed else 'DOES NOT BEAT SBS'}"
          + ("   (dev stage: exploratory, not a test)" if args.stage == "dev" else ""),
          "",
          "  ABLATION CONTRASTS (Wilcoxon over function means, Bonferroni x3)"]
    for c in contrasts:
        L.append(f"    {c['pair']}: mean rank difference {c['mean_rank_difference']:+.4f}, "
                 f"p = {c['p']:.4f}, Bonferroni p = {c['p_bonferroni']:.4f}")
    L += ["", "  TOP PICKS per arm (counts over all slots)"]
    for arm in ARMS:
        L.append(f"    {arm}: " + ", ".join(f"{a} {n}" for a, n in res[arm]["top_picks"]))
    L += ["", f"  calls: {len(secs)} answers received, mean {np.mean(secs):.2f} s each; "
              f"tokens per arm {dict((a, res[a]['total_tokens']) for a in ARMS)}"]
    payload = {"meta": {"stage": args.stage, "store": stem + ".json",
                        "store_md5": md5(stem + ".json"),
                        "matrices_md5": {p: md5(p) for p in (TRAIN_PATH, EVAL_PATH)},
                        "primary": PRIMARY, "contrasts": CONTRASTS},
               "verdict": "BEATS SBS" if passed else "DOES NOT BEAT SBS",
               "vbs_mean_rank": float(vbs_r.mean()), "sbs_mean_rank": float(sbs_r.mean()),
               "random_mean_rank": float(rnd_r.mean()), "arms": res, "contrasts": contrasts,
               "detail": [{"row": r, "sbs": mat.algos[sbs[f]], "sbs_rank": float(s),
                           "vbs_rank": float(v), **{f"{a}_rank": float(arm_r[a][n])
                                                    for a in ARMS}}
                          for n, (r, f, s, v) in enumerate(zip(rows, funcs, sbs_r, vbs_r))]}
    os.makedirs(os.path.dirname(OUT_STEM[args.stage]), exist_ok=True)
    with open(OUT_STEM[args.stage] + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(OUT_STEM[args.stage] + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("\n".join(L))
    print(f"\n  wrote {OUT_STEM[args.stage]}.txt and .json")


def main():
    ap = argparse.ArgumentParser(description="Phase 5: LLM algorithm selection.")
    ap.add_argument("--stage", choices=("dev", "test"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--test-call", action="store_true")
    mode.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if not args.test_call and args.stage is None:
        ap.error("--stage is required")
    train, test = load_matrix(TRAIN_PATH), load_matrix(EVAL_PATH)
    if train.algos != test.algos:
        sys.exit("ALGORITHM SETS DIFFER between the matrices -- stopping.")
    if args.dry_run:
        dry_run(args, train, test)
    elif args.test_call:
        test_call(args, train, test)
    elif args.analyse:
        analyse(args, train, test)
    else:
        run_calls(args, train, test)


if __name__ == "__main__":
    main()
