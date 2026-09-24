"""
PHASE 3 -- Validate a learned per-instance algorithm selector.

Reads results/phase2c_matrix.json ONLY. Never imports or calls
build_algorithm; no optimiser is invoked anywhere in this file.

DIMENSION IS A LIVE VARIABLE. dim is carried through row keys, the feature
vector, the fold grouping and every structure below, and the dims actually
present are read from the data. Adding or removing a dimension is a data
change, not a code change.

AGGREGATION ACROSS ROWS
Raw best-fitness values differ by orders of magnitude across BBOB functions
(F5 sits near -2e2 while F12 can reach ~1e9), so averaging raw fitness would
let a handful of large-scale functions dominate the mean and silently decide
the winner. We therefore aggregate with the per-row RANK of the chosen
algorithm (1 = best of the 22, 22 = worst), which is scale-free and directly
comparable across rows. A per-row min-max normalised score is reported as a
secondary view of magnitude.
"""
import argparse
import json
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor

MATRIX = "results/phase2c_matrix.json"
RF_SEED = 20260916
RF_TREES = 300


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    algos = sorted(d["meta"]["algorithms"])
    rows = sorted(d["features"].keys())
    fkeys = sorted(next(iter(d["features"].values())).keys())   # includes 'dimension'

    F = np.array([[d["features"][r][k] for k in fkeys] for r in rows], float)
    M = np.full((len(rows), len(algos)), np.nan)
    funcs, seeds, dims = [], [], []
    for i, r in enumerate(rows):
        for j, alg in enumerate(algos):
            c = d["cells"].get(r + "|" + alg)
            if c is None:
                sys.exit("MISSING CELL " + r + "|" + alg + " -- refusing to impute.")
            M[i, j] = c["median"]
        c0 = d["cells"][r + "|" + algos[0]]
        funcs.append(c0["function"])
        seeds.append(c0["pseudo_instance_seed"])
        dims.append(c0["dim"])
    if np.isnan(M).any():
        sys.exit("NaN present in matrix -- refusing to impute.")
    # Fold group = FUNCTION only. All seeds AND all dims of a function are
    # held out together, so no near-duplicate of a test row can train.
    return rows, algos, fkeys, F, M, np.array(funcs), np.array(seeds), np.array(dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--out-stem", default="results/phase3_selector")
    args = ap.parse_args()

    rows, algos, fkeys, F, M, funcs, seeds, dims = load(args.matrix)
    n_row, n_alg = M.shape
    ranks = np.vstack([rankdata(M[i]) for i in range(n_row)])
    vbs_rank = ranks.min(axis=1)

    lo = M.min(axis=1, keepdims=True)
    hi = M.max(axis=1, keepdims=True)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    N = (M - lo) / span

    ufuncs = sorted(set(funcs.tolist()))
    sel_rank, sbs_rank, sel_norm, sbs_norm, out_rows = [], [], [], [], []

    for f in ufuncs:
        te = np.where(funcs == f)[0]
        tr = np.where(funcs != f)[0]

        sbs_j = int(np.argmin(ranks[tr].mean(axis=0)))          # TRAIN rows only

        Xtr = np.hstack([np.repeat(F[tr], n_alg, axis=0),
                         np.tile(np.eye(n_alg), (len(tr), 1))])
        ytr = ranks[tr].reshape(-1)
        rf = RandomForestRegressor(n_estimators=RF_TREES,
                                   random_state=RF_SEED, n_jobs=-1)
        rf.fit(Xtr, ytr)

        for i in te:
            Xte = np.hstack([np.repeat(F[i:i + 1], n_alg, axis=0), np.eye(n_alg)])
            j = int(np.argmin(rf.predict(Xte)))
            sel_rank.append(ranks[i, j])
            sbs_rank.append(ranks[i, sbs_j])
            sel_norm.append(N[i, j])
            sbs_norm.append(N[i, sbs_j])
            out_rows.append({
                "row": rows[i], "function": f, "seed": int(seeds[i]),
                "dim": int(dims[i]), "selector": algos[j], "sbs": algos[sbs_j],
                "vbs": algos[int(np.argmin(M[i]))],
                "selector_rank": float(ranks[i, j]),
                "sbs_rank": float(ranks[i, sbs_j]),
                "vbs_rank": float(vbs_rank[i]),
            })

    sel_rank = np.array(sel_rank)
    sbs_rank = np.array(sbs_rank)
    rnd_rank = float(np.mean([ranks[i].mean() for i in range(n_row)]))   # analytic
    vbs_mean = float(vbs_rank.mean())
    sel_mean = float(sel_rank.mean())
    sbs_mean = float(sbs_rank.mean())
    gap = sbs_mean - vbs_mean
    closed = (sbs_mean - sel_mean) / gap * 100.0 if gap > 0 else float("nan")

    # per-FOLD means -> 24 paired observations, no pseudo-replication
    fold_sel = np.array([np.mean([r["selector_rank"] for r in out_rows if r["function"] == f])
                         for f in ufuncs])
    fold_sbs = np.array([np.mean([r["sbs_rank"] for r in out_rows if r["function"] == f])
                         for f in ufuncs])
    fold_vbs = np.array([np.mean([vbs_rank[i] for i in np.where(funcs == f)[0]])
                         for f in ufuncs])

    diff = fold_sel - fold_sbs
    if np.all(diff == 0):
        stat, pv = float("nan"), float("nan")
    else:
        stat, pv = wilcoxon(fold_sel, fold_sbs)

    # Headroom is measured PER ROW. Comparing per-function AVERAGE ranks
    # breaks once a function has many rows: SBS must match the oracle on
    # every one of them to tie on the average, so the count collapses to
    # ~0 (it read "0 of 24" on the 720-row matrix while SBS actually
    # matched the oracle on hundreds of rows).
    headroom_rows = int(sum(np.isclose(r["sbs_rank"], r["vbs_rank"]) for r in out_rows))
    headroom_funcs = int(np.sum(np.isclose(fold_sbs, fold_vbs)))

    L = []
    w = L.append
    w("=" * 78)
    w("  PHASE 3 -- LEARNED SELECTOR vs SBS  (leave-one-FUNCTION-out)")
    w("=" * 78)
    w("  matrix          : " + args.matrix)
    w("  rows            : {}   algorithms: {}   folds: {}".format(n_row, n_alg, len(ufuncs)))
    w("  seeds           : {}".format(sorted(set(seeds.tolist()))))
    w("  dims present    : {}   (dim kept as a live variable in code)".format(
        sorted(set(dims.tolist()))))
    w("  n_runs per cell : {}  (from matrix meta; phase2_matrix.json is the"
      " authoritative source for headline SBS/VBS figures)".format(
          json.load(open(args.matrix, encoding="utf-8"))["meta"].get("n_runs", "?")))
    w("  seeds recorded  : RF_SEED={}, n_estimators={}".format(RF_SEED, RF_TREES))
    w("  scipy/sklearn   : {} / {}".format(__import__("scipy").__version__,
                                           __import__("sklearn").__version__))
    w("")
    w("  MEAN RANK over held-out rows (1 = best of {}, lower is better)".format(n_alg))
    w("  " + "-" * 60)
    w("  {:<22}{:>10.4f}".format("VBS (oracle)", vbs_mean))
    w("  {:<22}{:>10.4f}".format("Selector (RF)", sel_mean))
    w("  {:<22}{:>10.4f}".format("SBS (fold-aware)", sbs_mean))
    w("  {:<22}{:>10.4f}".format("Random (analytic)", rnd_rank))
    w("")
    w("  SBS->VBS gap closed by selector : {:+.1f}%".format(closed))
    w("  mean norm score  selector={:.4f}  sbs={:.4f}".format(
        float(np.mean(sel_norm)), float(np.mean(sbs_norm))))
    w("")
    w("  WILCOXON  selector vs SBS, paired over the {} fold means".format(len(ufuncs)))
    w("  signature: wilcoxon(x, y=None, zero_method='wilcox', correction=False,")
    w("             alternative='two-sided', method='auto', *, axis=0, ...)")
    w("  statistic={!r}  p={!r}".format(stat, pv))
    w("  significant at alpha=0.05: {}".format(
        "YES" if (pv == pv and pv < 0.05) else "NO"))
    w("  selector better on {} folds, worse on {}, tied on {}".format(
        int((diff < 0).sum()), int((diff > 0).sum()), int((diff == 0).sum())))
    w("")
    w("  HEADROOM: SBS already equals VBS on {} of {} rows ({:.0f}%)".format(
        headroom_rows, n_row, 100.0 * headroom_rows / n_row))
    w("            (on every row of a function: {} of {} functions)".format(
        headroom_funcs, len(ufuncs)))
    w("")
    udims = sorted(set(dims.tolist()))
    # Derived, never asserted: an earlier version hardcoded "dim=10 only
    # ... the dimension axis is UNTESTED", which became false the moment
    # the matrix gained dims 5 and 20.
    w("  SCOPE: dims {}; {} rows over {} functions.".format(
        udims, n_row, len(ufuncs)))
    if len(udims) == 1:
        w("  The dimension axis is UNTESTED -- only dim {} is present.".format(udims[0]))
    w("=" * 78)

    txt = "\n".join(L)
    print(txt)
    with open(args.out_stem + ".txt", "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    out = {
        "per_row": out_rows,
        "fold_selector": fold_sel.tolist(),
        "fold_sbs": fold_sbs.tolist(),
        "fold_vbs": fold_vbs.tolist(),
        "functions": ufuncs,
        "feature_keys": fkeys,
        "rf_seed": RF_SEED,
        "rf_trees": RF_TREES,
        "wilcoxon_statistic": None if stat != stat else float(stat),
        "wilcoxon_p": None if pv != pv else float(pv),
    }
    with open(args.out_stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
