"""
phase2_lopo.py
--------------
Leave-One-Problem-Out (LOPO) evaluation of feature-based algorithm selectors.

Reads results/phase2/phase2_matrix.json ONLY. No optimiser is ever invoked, so every
number here is reproducible from that file.

WHAT LOPO MEASURES
------------------
For each fold in turn: hide one function's rows, fit a selector on the rest,
ask the selector which algorithm to use on the hidden problem, then look up
what that algorithm actually scored there. This estimates how the selector
would behave on a problem it has never seen.

NO-LEAKAGE RULES (the things that quietly invalidate a LOPO result)
-------------------------------------------------------------------
1. SBS IS RECOMPUTED PER FOLD. The Single Best Solver is the algorithm with
   the best mean rank over the 23 TRAINING problems only. Using the global SBS
   (the global SBS, chosen with every row visible) would let the baseline peek at
   the test problem and would understate the selector's achievement.
2. THE FEATURE SCALER IS FIT ON TRAINING FOLDS ONLY.
3. Constant features are dropped using the training fold's variance.

SCORING
-------
Raw fitness is not comparable across problems (medians span many orders of
magnitude, and F5/F19/F20 have negative optima). Each problem is therefore
scored on the same normalisation used in phase2_analyse_matrix.py:

    score(a, p) = (M[p, a] - VBS_p) / (worst_p - VBS_p)      in [0, 1]

0 = matched the per-problem oracle, 1 = was the worst algorithm on it.

NOTE ON THIS NORMALISATION: VBS_p and worst_p come from the held-out problem's
own row. That is a property of the EVALUATION METRIC, not information given to
the selector -- the selector sees only landscape features and training rows.
This is the standard "normalised regret" used in the algorithm-selection
literature. The raw selected-algorithm fitness is also reported so nothing
depends on accepting the transform.

FEATURE HANDLING
----------------
The 9 landscape features include `dimension`, which is constant at 10.0 across
every row it would carry no signal (checked at runtime), and several
fitness_* features span
15 orders of magnitude. Features are therefore signed-log1p compressed and
standardised (both fit on the training fold), and zero-variance columns are
dropped.

Run:
    .venv\\Scripts\\python.exe phase2_lopo.py
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

IN_PATH = os.path.join("results", "phase2", "phase2_matrix.json")
OUT_JSON = os.path.join("results", "phase2", "phase2_lopo.json")
OUT_TXT = os.path.join("results", "phase2", "phase2_lopo.txt")

RF_SEED = 20260915          # every stochastic component is seeded
RANDOM_SEED = 20260915
RF_TREES = 200

BAR = "=" * 96


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(path=None, exclude=()):
    """
    Load either matrix format and return rows, algorithms, medians, features
    and a GROUP label per row.

    Two layouts are supported and auto-detected:

      Phase 2  (results/phase2/phase2_matrix.json)
          meta.problems present; cell key "problem|algorithm".
          Each row is its own group, so grouped folds reduce exactly to the
          original leave-one-problem-out behaviour -- the default run
          reproduces the published Phase 2 numbers unchanged.

      Phase 2c (results/phase2c/phase2c_matrix.json)
          meta.functions + meta.pseudo_instance_seeds present; cell key
          "function|sSEED|dDIM|algorithm". The group is the FUNCTION, so all
          pseudo-instances of one function are held out together. Without
          that, holding out F1|s42 would leave F1|s43/44/45 in training --
          near-duplicates of the test row, which leaks the answer.
    """
    path = path or IN_PATH
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    A = d["meta"]["algorithms"]
    C = d["cells"]

    if "problems" in d["meta"]:                      # Phase 2 layout
        P = [p for p in d["meta"]["problems"] if p not in exclude]
        groups = list(P)                             # one row per group
        cell_of = lambda r, a: C[f"{r}|{a}"]
    else:                                            # Phase 2c layout
        P = sorted(d["features"])
        P = [r for r in P if r.split("|")[0] not in exclude]
        groups = [r.split("|")[0] for r in P]        # group == function
        cell_of = lambda r, a: C[f"{r}|{a}"]

    M = np.array([[cell_of(p, a)["median"] for a in A] for p in P], dtype=float)
    if not np.all(np.isfinite(M)):
        print("ERROR: matrix contains non-finite medians; refusing to proceed.")
        sys.exit(1)

    fkeys = sorted(d["features"][P[0]].keys())
    F = np.array([[d["features"][p][k] for k in fkeys] for p in P], dtype=float)
    return d, P, A, M, F, fkeys, groups


def normalise_rows(M):
    """Per-problem min-max: VBS -> 0, worst -> 1."""
    vbs = M.min(axis=1, keepdims=True)
    worst = M.max(axis=1, keepdims=True)
    span = worst - vbs
    span[span == 0] = 1.0
    return (M - vbs) / span


def prep_features(F_train, F_test):
    """signed-log1p compress, drop zero-variance cols, standardise. Fit on train."""
    def slog(X):
        return np.sign(X) * np.log1p(np.abs(X))

    Xtr, Xte = slog(F_train), slog(F_test)
    keep = Xtr.std(axis=0) > 0          # decided on TRAIN only
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    sc = StandardScaler().fit(Xtr)      # fit on TRAIN only
    return sc.transform(Xtr), sc.transform(Xte), keep


# ---------------------------------------------------------------------------
# Selectors. Each returns the index of the chosen algorithm.
# ---------------------------------------------------------------------------

def sel_sbs(train_idx, M, **_):
    """Fold-aware SBS: best mean rank over TRAINING problems only."""
    R = np.vstack([rankdata(M[i, :]) for i in train_idx])
    return int(np.argmin(R.mean(axis=0)))


def sel_random(rng, n_alg, **_):
    return int(rng.integers(0, n_alg))


def sel_knn(k, Xtr, Xte, N_train, **_):
    """k nearest training problems in feature space; pick the algorithm with
    the best mean normalised score across those neighbours."""
    dist = np.linalg.norm(Xtr - Xte[0], axis=1)
    nb = np.argsort(dist)[:k]
    return int(np.argmin(N_train[nb].mean(axis=0)))


def sel_rf(Xtr, Xte, N_train, seed, **_):
    """One regressor per algorithm: features -> normalised score; take argmin."""
    preds = []
    for j in range(N_train.shape[1]):
        rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed)
        rf.fit(Xtr, N_train[:, j])
        preds.append(rf.predict(Xte)[0])
    return int(np.argmin(preds))


# ---------------------------------------------------------------------------

def fit_prep(F_train):
    """First half of prep_features(): fit on TRAIN only, return the transform.

    Identical maths to prep_features() -- signed-log1p, drop columns that are
    constant on the training rows, standardise -- split so a fold's transform
    can be fitted once and applied to each of its test rows.
    """
    def slog(X):
        return np.sign(X) * np.log1p(np.abs(X))

    Xtr = slog(F_train)
    keep = Xtr.std(axis=0) > 0
    Xtr = Xtr[:, keep]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), (keep, sc)


def apply_prep(F_test, prep):
    keep, sc = prep
    return sc.transform((np.sign(F_test) * np.log1p(np.abs(F_test)))[:, keep])


def fit_rf(Xtr, N_train, seed):
    """First half of sel_rf(): the per-algorithm forests, fitted once."""
    models = []
    for j in range(N_train.shape[1]):
        rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed)
        rf.fit(Xtr, N_train[:, j])
        models.append(rf)
    return models


def predict_rf(models, Xte):
    """Second half of sel_rf(): argmin of the predicted normalised score."""
    return int(np.argmin([m.predict(Xte)[0] for m in models]))


def run_lopo(P, A, M, F, groups):
    """
    Leave-one-GROUP-out. Every row whose group matches the held-out group is
    removed from training, so no near-duplicate of the test row survives in
    the training set. With one row per group this is ordinary
    leave-one-problem-out.
    """
    n_prob, n_alg = M.shape
    N = normalise_rows(M)
    ranks = np.vstack([rankdata(M[i, :]) for i in range(n_prob)])

    names = ["VBS (oracle)", "SBS (fold-aware)", "Random", "1-NN", "3-NN", "RF-regression",
             "RF in-sample (ceiling)"]
    picks = {n: [] for n in names}
    rng = np.random.default_rng(RANDOM_SEED)

    # PERFORMANCE. Every row of a held-out function shares an IDENTICAL
    # training set, and the ceiling trains on ALL rows -- identical for every
    # row. A forest with a fixed random_state on identical data is
    # deterministic, so refitting per row rebuilds the same model again and
    # again. At 720 rows / 24 folds that was 31,680 forest fits (~8.8 h); it
    # is now 24x22 + 22 = 550 fits. Results are unchanged -- verified
    # bit-identical against the per-row implementation on phase2_matrix.json.
    #
    # The row loop itself is KEPT, and in row order, because the Random
    # baseline draws from a seeded RNG once per row; reordering would change
    # its picks. Only the current fold's forests are held in memory, so a
    # non-contiguous group ordering stays correct (it merely refits).
    cur_held, fold = None, None
    ceil = None
    n_groups = len(set(groups))
    done = 0

    for i in range(n_prob):
        held = groups[i]
        if held != cur_held:
            train_idx = [t for t in range(n_prob) if groups[t] != held]
            Xtr, prep = fit_prep(F[train_idx])
            N_train = N[train_idx]
            fold = (train_idx, Xtr, prep, N_train, fit_rf(Xtr, N_train, RF_SEED))
            cur_held = held
            done += 1
            print(f"  fold {done:>3}/{n_groups}  held out {held}", flush=True)
        train_idx, Xtr, prep, N_train, rf_models = fold
        Xte = apply_prep(F[i:i + 1], prep)

        picks["VBS (oracle)"].append(int(np.argmin(M[i])))
        picks["SBS (fold-aware)"].append(sel_sbs(train_idx, M))
        picks["Random"].append(sel_random(rng, n_alg))
        picks["1-NN"].append(sel_knn(1, Xtr, Xte, N_train))
        picks["3-NN"].append(sel_knn(3, Xtr, Xte, N_train))
        picks["RF-regression"].append(predict_rf(rf_models, Xte))

        # DIAGNOSTIC, NOT A SELECTOR. Deliberately trains on every row,
        # including the held-out one, so the model is allowed to memorise
        # the answer. It measures whether the features can separate these
        # problems AT ALL. If this stalls near the LOPO score the features
        # carry no usable signal; if it approaches VBS the features are
        # fine and the loss is in generalising across functions.
        # Trained on ALL rows, so it is fitted exactly once.
        if ceil is None:
            Xa_tr, ceil_prep = fit_prep(F)
            ceil = (ceil_prep, fit_rf(Xa_tr, N, RF_SEED))
        picks["RF in-sample (ceiling)"].append(
            predict_rf(ceil[1], apply_prep(F[i:i + 1], ceil[0])))

    return N, ranks, names, picks


def summarise(P, A, M, N, ranks, names, picks):
    out = {}
    for name in names:
        idx = picks[name]
        scores = np.array([N[i, idx[i]] for i in range(len(P))])
        rk = np.array([ranks[i, idx[i]] for i in range(len(P))])
        raw = np.array([M[i, idx[i]] for i in range(len(P))])
        vbs_hits = int(sum(1 for i in range(len(P)) if np.isclose(N[i, idx[i]], 0.0)))
        out[name] = {
            "picks": [A[j] for j in idx],
            "norm_scores": scores.tolist(),
            "raw_fitness": raw.tolist(),
            "ranks": rk.tolist(),
            "mean_norm": float(scores.mean()),
            "median_norm": float(np.median(scores)),
            "mean_rank": float(rk.mean()),
            "vbs_hits": vbs_hits,
        }
    return out


def report(P, A, M, N, names, picks, summ, data, groups=None, source=None):
    L = []
    w = L.append

    w(BAR)
    w("  PHASE 2 - LEAVE-ONE-PROBLEM-OUT (LOPO) SELECTOR EVALUATION")
    w(BAR)
    n_groups = len(set(groups)) if groups else len(P)
    w(f"  source            : {source or IN_PATH} (no optimiser invoked)")
    w(f"  rows              : {len(P)}")
    w(f"  folds (groups)    : {n_groups}")
    w(f"  algorithms        : {len(A)}")
    w(f"  scoring           : per-problem min-max, VBS=0, worst=1")
    w(f"  SBS               : recomputed per fold from the training rows only")
    if n_groups != len(P):
        w(f"  fold rule         : leave-one-GROUP-out -- all "
          f"{len(P)//n_groups} rows of a function held out together")
    w(f"  seeds             : RF={RF_SEED}, random baseline={RANDOM_SEED}")
    w(f"  scipy / sklearn   : {scipy.__version__} / "
      f"{__import__('sklearn').__version__}")
    w("")

    sbs_mean = summ["SBS (fold-aware)"]["mean_norm"]

    w(BAR)
    w("  SUMMARY  (normalised score: 0 = oracle, 1 = worst; lower is better)")
    w(BAR)
    w(f"  {'selector':<20}{'mean norm':>11}{'median':>10}{'mean rank':>11}"
      f"{'VBS hits':>10}{'gap closed vs SBS':>24}")
    w("  " + "-" * 86)
    for name in names:
        s = summ[name]
        if name == "VBS (oracle)":
            gc = "-- (is the target)"
        elif name == "SBS (fold-aware)":
            gc = "-- (is the baseline)"
        else:
            gc = f"{100.0 * (sbs_mean - s['mean_norm']) / sbs_mean:+.1f}%" \
                 if sbs_mean > 0 else "n/a"
        w(f"  {name:<20}{s['mean_norm']:>11.4f}{s['median_norm']:>10.4f}"
          f"{s['mean_rank']:>11.3f}{s['vbs_hits']:>10}{gc:>24}")

    w("")
    w("  'gap closed vs SBS' = (SBS mean - selector mean) / SBS mean.")
    w("  Positive means the selector beat the single best solver; negative means")
    w("  it did worse than just always picking that one algorithm.")

    # --- per-fold detail ---
    w("")
    w(BAR)
    w("  PER-PROBLEM DETAIL")
    w(BAR)
    w(f"  {'problem':<34}{'VBS':>9}{'SBS':>9}{'1-NN':>9}{'3-NN':>9}{'RF':>9}"
      f"{'  |  norm score (SBS / 1-NN / 3-NN / RF)'}")
    w("  " + "-" * 94)
    for i, p in enumerate(P):
        cells = "".join(f"{summ[n]['picks'][i]:>9}" for n in
                        ["VBS (oracle)", "SBS (fold-aware)", "1-NN", "3-NN", "RF-regression"])
        sc = "  |  " + " / ".join(
            f"{summ[n]['norm_scores'][i]:.3f}" for n in
            ["SBS (fold-aware)", "1-NN", "3-NN", "RF-regression"])
        w(f"  {p:<34}{cells}{sc}")

    # --- significance vs SBS ---
    grp = list(groups) if groups else list(P)
    ugrp = sorted(set(grp))
    clustered = len(ugrp) != len(P)
    garr = np.array(grp)
    sbs_scores = np.array(summ["SBS (fold-aware)"]["norm_scores"])
    tested = [n for n in names
              if n not in ("VBS (oracle)", "SBS (fold-aware)", "RF in-sample (ceiling)")]
    bonf = 0.05 / max(len(tested), 1)

    def _fold_means(v):
        return np.array([v[garr == g].mean() for g in ugrp])

    if clustered:
        w("")
        w(BAR)
        w(f"  WILCOXON SIGNED-RANK vs SBS  --  PRIMARY TEST")
        w(f"  paired over {len(ugrp)} FUNCTION MEANS (rows aggregated first)")
        w(BAR)
        w(f"  Each function contributes {len(P)//len(ugrp)} rows that are NOT")
        w(f"  independent, so the rows of a function are averaged before pairing.")
        w(f"  signature: wilcoxon(x, y=None, zero_method='wilcox', correction=False,")
        w(f"             alternative='two-sided', method='auto', ...)")
        w(f"  Bonferroni threshold for {len(tested)} selectors: {bonf:.4f}")
        sbs_f = _fold_means(sbs_scores)
        for name in tested:
            xf = _fold_means(np.array(summ[name]["norm_scores"]))
            d = xf - sbs_f
            if np.all(d == 0):
                w(f"  {name:<16} identical to SBS on every fold; test not applicable")
                continue
            stat, pv = wilcoxon(xf, sbs_f)
            mark = "  *SIGNIFICANT*" if pv < 0.05 else ""
            mark += " (survives Bonferroni)" if pv < bonf else ""
            w(f"  {name:<16} statistic={stat!r:<10} p={pv!r:<24} "
              f"better on {int((d < 0).sum())}, worse on {int((d > 0).sum())}, "
              f"tied on {int((d == 0).sum())}{mark}")

    w("")
    w(BAR)
    head = "SECONDARY -- INFLATED, DO NOT QUOTE" if clustered else "WILCOXON SIGNED-RANK vs SBS"
    w(f"  {head}  (paired over the {len(P)} rows)")
    w(BAR)
    if clustered:
        w(f"  Pairs all {len(P)} rows as if independent. They are not -- see the")
        w(f"  primary test above. Reported only so the inflation is visible.")
    w(f"  signature: wilcoxon(x, y=None, zero_method='wilcox', correction=False,")
    w(f"             alternative='two-sided', method='auto', ...)")
    for name in tested:
        x = np.array(summ[name]["norm_scores"])
        diff = x - sbs_scores
        if np.all(diff == 0):
            w(f"  {name:<16} identical to SBS on every problem; test not applicable")
            continue
        stat, pv = wilcoxon(x, sbs_scores)
        better = int((diff < 0).sum())
        worse = int((diff > 0).sum())
        w(f"  {name:<16} statistic={stat!r:<10} p={pv!r:<24} "
          f"better on {better}, worse on {worse}, "
          f"tied on {len(P) - better - worse}")

    w("")
    w(BAR)
    w("  INTERPRETATION LIMITS")
    w(BAR)
    w(f"  - {len(P)} rows over {n_groups} folds. Each fold trains on the rows of")
    w(f"    the other {n_groups - 1} functions only.")
    if clustered:
        w("")
        w("  *** CLUSTERED OBSERVATIONS -- READ BEFORE QUOTING ANY p-VALUE ***")
        w(f"  Rows are NOT independent: {len(P)//len(ugrp)} rows sit inside each of")
        w(f"  {len(ugrp)} functions. Quote the PRIMARY test (function means), not")
        w(f"  the secondary per-row one -- the latter is reported only to make the")
        w(f"  inflation visible. Bonferroni threshold for {len(tested)} selectors:")
        w(f"  0.05/{len(tested)} = {bonf:.4f}.")
        w("")

    # Every claim below is computed from THIS run's data. An earlier version
    # hardcoded figures from a different matrix (96 rows, p=0.0334/0.4348,
    # "dimension is constant at 10.0"), which silently became false the moment
    # the matrix changed. Never hardcode a number that describes the data.
    w(f"  - {len(P)} rows over {len(ugrp)} folds is little data for a "
      f"{len(A)}-way selection")
    w(f"    problem; treat these numbers as indicative, not converged.")

    fk = sorted(data["features"][P[0]].keys())
    Fall = np.array([[data["features"][r][k] for k in fk] for r in P], float)
    const = [k for k, sd in zip(fk, Fall.std(axis=0)) if sd == 0]
    if const:
        w(f"  - constant feature column(s) {const} carry no signal and are")
        w(f"    dropped by the zero-variance filter, leaving {len(fk)-len(const)} usable.")
    else:
        w(f"  - all {len(fk)} feature columns vary across rows; none dropped.")

    ties = [(p_, int((M[i] == M[i].min()).sum())) for i, p_ in enumerate(P)]
    worst = max(ties, key=lambda t: t[1])
    if worst[1] > 1:
        w(f"  - most-tied row: {worst[0]} has {worst[1]} of {len(A)} algorithms at")
        w(f"    the exact optimum, so that row rewards almost any choice.")

    hits = summ["SBS (fold-aware)"]["vbs_hits"]
    w(f"  - SBS is a strong baseline: it already equals VBS on {hits} of {len(P)}")
    w(f"    rows, bounding the headroom a selector can win.")
    w(BAR)

    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=IN_PATH,
                    help="matrix JSON to evaluate (Phase 2 or Phase 2c layout)")
    ap.add_argument("--out-stem", default=None,
                    help="output stem; defaults to <matrix folder>/<matrix stem>_lopo")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="function names to drop, e.g. BBOB_F5_LinearSlope")
    args = ap.parse_args()

    stem = args.out_stem or os.path.join(
        os.path.dirname(args.matrix) or ".", os.path.splitext(os.path.basename(args.matrix))[0]
        .replace("_matrix", "") + "_lopo")
    out_txt, out_json = stem + ".txt", stem + ".json"

    data, P, A, M, F, fkeys, groups = load(args.matrix, exclude=set(args.exclude))
    n_groups = len(set(groups))
    print(f"  source   : {args.matrix}")
    print(f"  rows     : {len(P)}   groups (folds): {n_groups}")
    if args.exclude:
        print(f"  excluded : {args.exclude}")
    if n_groups != len(P):
        print(f"  fold rule: leave-one-GROUP-out -- all {len(P)//n_groups} rows "
              f"of a function are held out together (prevents near-duplicate leakage)")

    N, ranks, names, picks = run_lopo(P, A, M, F, groups)
    summ = summarise(P, A, M, N, ranks, names, picks)
    text = report(P, A, M, N, names, picks, summ, data, groups, args.matrix)
    print(text)

    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({
            "source": args.matrix,
            "rows": P,
            "groups": groups,
            "n_folds": n_groups,
            "excluded": args.exclude,
            "algorithms": A,
            "feature_keys": fkeys,
            "seeds": {"rf": RF_SEED, "random": RANDOM_SEED},
            "rf_trees": RF_TREES,
            "selectors": summ,
        }, fh, indent=2)
    print(f"\n  saved: {out_txt}\n         {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
