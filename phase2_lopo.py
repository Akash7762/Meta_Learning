"""
phase2_lopo.py
--------------
Leave-One-Problem-Out (LOPO) evaluation of feature-based algorithm selectors.

Reads results/phase2_matrix.json ONLY. No optimiser is ever invoked, so every
number here is reproducible from that file.

WHAT LOPO MEASURES
------------------
For each of the 24 problems in turn: hide it, fit a selector on the other 23,
ask the selector which algorithm to use on the hidden problem, then look up
what that algorithm actually scored there. This estimates how the selector
would behave on a problem it has never seen.

NO-LEAKAGE RULES (the things that quietly invalidate a LOPO result)
-------------------------------------------------------------------
1. SBS IS RECOMPUTED PER FOLD. The Single Best Solver is the algorithm with
   the best mean rank over the 23 TRAINING problems only. Using the global SBS
   (MPA, chosen with all 24 problems visible) would let the baseline peek at
   the test problem and would understate the selector's achievement.
2. THE FEATURE SCALER IS FIT ON TRAINING FOLDS ONLY.
3. Constant features are dropped using the training fold's variance.

SCORING
-------
Raw fitness is not comparable across problems (medians span ~20 orders of
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
all 24 problems and carries no signal, and several fitness_* features spanning
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

IN_PATH = os.path.join("results", "phase2_matrix.json")
OUT_JSON = os.path.join("results", "phase2_lopo.json")
OUT_TXT = os.path.join("results", "phase2_lopo.txt")

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

      Phase 2  (results/phase2_matrix.json)
          meta.problems present; cell key "problem|algorithm".
          Each row is its own group, so grouped folds reduce exactly to the
          original leave-one-problem-out behaviour -- the default run
          reproduces the published Phase 2 numbers unchanged.

      Phase 2c (results/phase2c_matrix.json)
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

    for i in range(n_prob):
        held = groups[i]
        train_idx = [t for t in range(n_prob) if groups[t] != held]
        Xtr, Xte, _ = prep_features(F[train_idx], F[i:i + 1])
        N_train = N[train_idx]

        picks["VBS (oracle)"].append(int(np.argmin(M[i])))
        picks["SBS (fold-aware)"].append(sel_sbs(train_idx, M))
        picks["Random"].append(sel_random(rng, n_alg))
        picks["1-NN"].append(sel_knn(1, Xtr, Xte, N_train))
        picks["3-NN"].append(sel_knn(3, Xtr, Xte, N_train))
        picks["RF-regression"].append(sel_rf(Xtr, Xte, N_train, RF_SEED))

        # DIAGNOSTIC, NOT A SELECTOR. Deliberately trains on every row,
        # including the held-out one, so the model is allowed to memorise
        # the answer. It measures whether the features can separate these
        # problems AT ALL. If this stalls near the LOPO score the features
        # carry no usable signal; if it approaches VBS the features are
        # fine and the loss is in generalising across functions.
        Xa_tr, Xa_te, _ = prep_features(F, F[i:i + 1])
        picks["RF in-sample (ceiling)"].append(sel_rf(Xa_tr, Xa_te, N, RF_SEED))

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
    w("")
    w(BAR)
    w(f"  WILCOXON SIGNED-RANK vs SBS  (paired over the {len(P)} rows)")
    w(BAR)
    w(f"  signature: wilcoxon(x, y=None, zero_method='wilcox', correction=False,")
    w(f"             alternative='two-sided', method='auto', ...)")
    sbs_scores = np.array(summ["SBS (fold-aware)"]["norm_scores"])
    for name in names:
        if name in ("VBS (oracle)", "SBS (fold-aware)", "RF in-sample (ceiling)"):
            continue
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
    if n_groups != len(P):
        w("")
        w("  *** CLUSTERED OBSERVATIONS -- READ BEFORE QUOTING ANY p-VALUE ***")
        w(f"  The Wilcoxon above pairs {len(P)} rows, but they are NOT independent:")
        w(f"  they are {len(P)//n_groups} pseudo-instances within each of {n_groups}")
        w("  functions. Treating clustered rows as independent inflates")
        w("  significance. The honest test averages the rows of each function")
        w("  first and pairs the resulting function means. Measured on this")
        w("  data, RF-regression moves from p=0.0334 (96 rows) to p=0.4348")
        w("  (24 function means) -- i.e. NOT significant. Use the aggregated")
        w("  figure. Three selectors are compared, so a Bonferroni threshold")
        w("  of 0.05/3 = 0.0167 applies as well.")
        w("")
    w(f"  - the original concern stands: this is very")
    w(f"    little data for a 22-way selection problem; treat these numbers as")
    w(f"    indicative, not as a converged estimate.")
    w(f"  - 'dimension' is constant (10.0) across all problems and is dropped")
    w(f"    automatically by the zero-variance filter, leaving 8 usable features.")
    w(f"  - On F5 LinearSlope 13 of 22 algorithms tie at the exact global")
    w(f"    optimum, so that fold rewards almost any choice.")
    w(f"  - SBS is a strong baseline here: it already equals VBS on 10 of 24")
    w(f"    problems, so the headroom a selector can win is small by construction.")
    w(BAR)

    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=IN_PATH,
                    help="matrix JSON to evaluate (Phase 2 or Phase 2c layout)")
    ap.add_argument("--out-stem", default=None,
                    help="output stem; defaults to results/<matrix stem>_lopo")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="function names to drop, e.g. BBOB_F5_LinearSlope")
    args = ap.parse_args()

    stem = args.out_stem or os.path.join(
        "results", os.path.splitext(os.path.basename(args.matrix))[0]
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

    os.makedirs("results", exist_ok=True)
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
