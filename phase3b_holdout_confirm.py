"""
PHASE 3b -- Confirmation run for the Phase 3 selector result.

Method fixed in advance; see results/phase3_preregistration.md. Nothing in
this file deviates from it. Reads cached JSON only; no optimiser is invoked
and build_algorithm is never called.

TRAIN : results/phase2c_matrix.json  (96 rows, n_runs=3,  dim=10)
EVAL  : results/phase2_matrix.json   (24 rows, n_runs=15, dim=10)

THE EVALUATION SET IS NOT INDEPENDENT. phase2 was built with problem_seed=42
and phase2c includes pseudo_instance_seed=42, so all 24 evaluation rows are
re-measurements of (function, seed=42, dim=10) configurations already in the
training file, with bit-identical feature vectors. Leave-one-FUNCTION-out is
therefore applied ACROSS the two files: to predict function F, training uses
only phase2c rows whose function != F. This run tests (a) generalisation to
an unseen function and (b) robustness to 15-run vs 3-run targets. It does
NOT resolve the post-hoc selection risk from the 8-cell ablation.

DIM IS A LIVE VARIABLE: carried in row keys, the feature vector and the fold
grouping throughout. Appending dims 5/20 is a data change, not a code change.
"""
import json
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor

TRAIN_PATH = "results/phase2c_matrix.json"
EVAL_PATH = "results/phase2_matrix.json"
OUT_STEM = "results/phase3b_holdout_confirm"
RF_SEED = 20260916
RF_TREES = 300


def load_train(path):
    """phase2c layout: cells keyed {function}|s{seed}|d{dim}|{algorithm}."""
    d = json.load(open(path, encoding="utf-8"))
    algos = sorted(d["meta"]["algorithms"])
    rows = sorted(d["features"].keys())
    fkeys = sorted(next(iter(d["features"].values())).keys())
    F = np.array([[d["features"][r][k] for k in fkeys] for r in rows], float)
    M = np.full((len(rows), len(algos)), np.nan)
    funcs, dims = [], []
    for i, r in enumerate(rows):
        for j, a in enumerate(algos):
            c = d["cells"].get(r + "|" + a)
            if c is None:
                sys.exit("MISSING TRAIN CELL " + r + "|" + a + " -- refusing to impute.")
            M[i, j] = c["median"]
        c0 = d["cells"][r + "|" + algos[0]]
        funcs.append(c0["function"])
        dims.append(c0["dim"])
    if np.isnan(M).any():
        sys.exit("NaN in training matrix -- refusing to impute.")
    return algos, fkeys, F, M, np.array(funcs), np.array(dims)


def load_eval(path, algos, fkeys):
    """phase2 layout: cells keyed {problem}|{algorithm}; single global dim/seed."""
    d = json.load(open(path, encoding="utf-8"))
    if sorted(d["meta"]["algorithms"]) != algos:
        sys.exit("ALGORITHM SETS DIFFER -- stopping as instructed.")
    ek = sorted(next(iter(d["features"].values())).keys())
    if ek != fkeys:
        sys.exit("FEATURE SCHEMAS DIFFER -- stopping as instructed.")
    dim = d["meta"]["problem_dim"]          # dim stays a live variable
    seed = d["meta"]["problem_seed"]
    rows = sorted(d["features"].keys())
    F = np.array([[d["features"][r][k] for k in fkeys] for r in rows], float)
    M = np.full((len(rows), len(algos)), np.nan)
    for i, r in enumerate(rows):
        for j, a in enumerate(algos):
            c = d["cells"].get(r + "|" + a)
            if c is None:
                sys.exit("MISSING EVAL CELL " + r + "|" + a + " -- refusing to impute.")
            M[i, j] = c["median"]
    if np.isnan(M).any():
        sys.exit("NaN in evaluation matrix -- refusing to impute.")
    return rows, F, M, np.array(rows), dim, seed


def main():
    algos, fkeys, Ftr_all, Mtr, ftr, dtr = load_train(TRAIN_PATH)
    erows, Fte_all, Mte, fte, edim, eseed = load_eval(EVAL_PATH, algos, fkeys)
    n_alg = len(algos)

    ranks_tr = np.vstack([rankdata(Mtr[i]) for i in range(Mtr.shape[0])])
    ranks_te = np.vstack([rankdata(Mte[i]) for i in range(Mte.shape[0])])
    vbs_rank = ranks_te.min(axis=1)

    sel_rank, sbs_rank, detail = [], [], []
    for i, f in enumerate(fte):
        tr = np.where(ftr != f)[0]          # LOFO ACROSS FILES
        if len(tr) == len(ftr):
            sys.exit("Function " + str(f) + " absent from training file -- stopping.")

        sbs_j = int(np.argmin(ranks_tr[tr].mean(axis=0)))    # TRAIN rows only

        Xtr = np.hstack([np.repeat(Ftr_all[tr], n_alg, axis=0),
                         np.tile(np.eye(n_alg), (len(tr), 1))])
        rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED)
        rf.fit(Xtr, ranks_tr[tr].reshape(-1))

        Xte = np.hstack([np.repeat(Fte_all[i:i + 1], n_alg, axis=0), np.eye(n_alg)])
        j = int(np.argmin(rf.predict(Xte)))

        sel_rank.append(ranks_te[i, j])
        sbs_rank.append(ranks_te[i, sbs_j])
        detail.append({"function": str(f), "dim": edim, "seed": eseed,
                       "n_train_rows": int(len(tr)),
                       "selector": algos[j], "sbs": algos[sbs_j],
                       "vbs": algos[int(np.argmin(Mte[i]))],
                       "selector_rank": float(ranks_te[i, j]),
                       "sbs_rank": float(ranks_te[i, sbs_j]),
                       "vbs_rank": float(vbs_rank[i])})

    sel_rank, sbs_rank = np.array(sel_rank), np.array(sbs_rank)
    sel_m, sbs_m, vbs_m = float(sel_rank.mean()), float(sbs_rank.mean()), float(vbs_rank.mean())
    rnd_m = float(np.mean([ranks_te[i].mean() for i in range(len(erows))]))
    gap = sbs_m - vbs_m
    closed = (sbs_m - sel_m) / gap * 100.0 if gap > 0 else float("nan")

    d = sel_rank - sbs_rank
    if np.all(d == 0):
        stat, pv = float("nan"), float("nan")
    else:
        stat, pv = wilcoxon(sel_rank, sbs_rank)          # THE one pre-specified test

    L = []
    w = L.append
    w("=" * 78)
    w("  PHASE 3b -- CONFIRMATION RUN (pre-registered)")
    w("=" * 78)
    w("  train           : " + TRAIN_PATH + "  (n_runs=3)")
    w("  eval            : " + EVAL_PATH + "   (n_runs=15)")
    w("  split           : leave-one-FUNCTION-out ACROSS files")
    w("  eval rows       : {}   algorithms: {}   dim: {}   seed: {}".format(
        len(erows), n_alg, edim, eseed))
    w("  seeds recorded  : RF_SEED={}, n_estimators={}".format(RF_SEED, RF_TREES))
    w("  scipy/sklearn   : {} / {}".format(__import__("scipy").__version__,
                                           __import__("sklearn").__version__))
    w("")
    w("  NOT AN INDEPENDENT HELD-OUT SET. phase2's 24 rows are re-measurements")
    w("  of (function, seed=42, dim=10) configurations also present in phase2c.")
    w("  Tests generalisation to an unseen function, and robustness to 15-run")
    w("  vs 3-run targets. Does NOT resolve the post-hoc selection risk.")
    w("")
    w("  MEAN RANK on held-out functions (1 = best of 22, lower is better)")
    w("  " + "-" * 60)
    w("  {:<22}{:>10.4f}".format("VBS (oracle)", vbs_m))
    w("  {:<22}{:>10.4f}".format("Selector (RF)", sel_m))
    w("  {:<22}{:>10.4f}".format("SBS (fold-aware)", sbs_m))
    w("  {:<22}{:>10.4f}".format("Random (analytic)", rnd_m))
    w("")
    w("  SBS->VBS gap closed : {:+.1f}%".format(closed))
    w("")
    w("  WILCOXON  selector vs SBS, paired over {} functions (one test, alpha=0.05)".format(
        len(erows)))
    w("  signature: wilcoxon(x, y=None, zero_method='wilcox', correction=False,")
    w("             alternative='two-sided', method='auto', *, axis=0, ...)")
    w("  statistic={!r}  p={!r}".format(stat, pv))
    w("  significant at alpha=0.05: {}".format("YES" if (pv == pv and pv < 0.05) else "NO"))
    w("  selector better on {}, worse on {}, tied on {}".format(
        int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())))
    w("=" * 78)

    txt = "\n".join(L)
    print(txt)
    open(OUT_STEM + ".txt", "w", encoding="utf-8").write(txt + "\n")
    json.dump({"detail": detail, "selector_mean_rank": sel_m, "sbs_mean_rank": sbs_m,
               "vbs_mean_rank": vbs_m, "random_mean_rank": rnd_m,
               "gap_closed_pct": closed,
               "wilcoxon_statistic": None if stat != stat else float(stat),
               "wilcoxon_p": None if pv != pv else float(pv),
               "rf_seed": RF_SEED, "rf_trees": RF_TREES,
               "eval_dim": edim, "eval_seed": eseed,
               "independent_holdout": False},
              open(OUT_STEM + ".json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
