"""
PHASE 3b -- Confirmation run for the Phase 3 selector result.

Method fixed in advance; see docs/phase3_preregistration.md. Nothing in
this file deviates from it. Reads cached JSON only; no optimiser is invoked
and build_algorithm is never called.

TRAIN : results/phase2c_matrix.json  (selector training, n_runs=3)
EVAL  : results/phase2_matrix.json   (held out,          n_runs=15)
Both overridable with --train-matrix / --eval-matrix. Row counts, seeds and
dims are read from the files, never assumed here.

INDEPENDENCE IS CHECKED, NOT ASSUMED. assert_no_leakage() refuses to run if
the eval matrix's problem_seed appears in the training matrix's
pseudo_instance_seeds. This guard exists because it happened: Phase 2 was
originally built at problem_seed=42, which is inside Phase 2c's seed list, so
every evaluation row re-measured a training configuration with a
bit-identical feature vector. Nothing in the code caught it -- it was found
by hand on 2026-09-17. phase2_build_matrix.py now uses PROBLEM_SEED=100.

Leave-one-FUNCTION-out is applied ACROSS the two files regardless: to predict
function F, training uses only rows whose function != F.

--allow-leakage reproduces the pre-2026-09-20 result on a non-independent
split. The result is then valid ONLY for (a) generalisation to an unseen
function and (b) robustness to 15-run vs 3-run targets -- it does not test
held-out configurations. The output labels itself accordingly and sets
"independent_holdout": false in the JSON.

DIM IS A LIVE VARIABLE: carried in row keys, the feature vector and the fold
grouping throughout. Appending dims 5/20 is a data change, not a code change.
"""
import argparse
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
    """Evaluation matrix loader. Handles both Phase 2 layouts.

    new (2026-09-22+) : rows keyed {function}|s{seed}|d{dim}, one row per
                        (function, dim). dim is per-row and carried through.
    old               : rows keyed {problem}, cells {problem}|{algorithm},
                        a single global meta.problem_dim.

    Returns per-row `funcs` (the FUNCTION, not the row key) so that
    leave-one-function-out holds every dimension of a function out together.
    """
    d = json.load(open(path, encoding="utf-8"))
    if sorted(d["meta"]["algorithms"]) != algos:
        sys.exit("ALGORITHM SETS DIFFER -- stopping as instructed.")
    ek = sorted(next(iter(d["features"].values())).keys())
    if ek != fkeys:
        sys.exit("FEATURE SCHEMAS DIFFER -- stopping as instructed.")

    seed = d["meta"]["problem_seed"]
    rows = sorted(d["features"].keys())
    multi_dim = "problem_dim" not in d["meta"]

    funcs, dims = [], []
    for r in rows:
        if multi_dim:
            fn, _, dpart = r.split("|")         # function | sSEED | dDIM
            funcs.append(fn)
            dims.append(int(dpart[1:]))
        else:
            funcs.append(r)                     # row key IS the function
            dims.append(d["meta"]["problem_dim"])

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
    return rows, F, M, np.array(funcs), np.array(dims), seed


def assert_no_leakage(train_path, eval_path):
    """Refuse to run if the evaluation configuration also appears in training.

    Phase 2 originally used problem_seed=42, which is inside Phase 2c's
    PSEUDO_INSTANCE_SEEDS, so every evaluation row was a re-measurement of a
    training row with a bit-identical feature vector. That is not a held-out
    set, and nothing in the code caught it -- it was found by hand. This check
    makes that failure mode impossible to repeat silently.
    """
    tr = json.load(open(train_path, encoding="utf-8"))["meta"]
    ev = json.load(open(eval_path, encoding="utf-8"))["meta"]
    train_seeds = set(tr.get("pseudo_instance_seeds", []))
    eval_seed = ev.get("problem_seed")

    if eval_seed is None:
        sys.exit("EVAL matrix has no 'problem_seed' in meta -- cannot verify "
                 "the split is clean. Refusing to run.")
    if eval_seed in train_seeds:
        sys.exit(
            "REFUSING TO RUN -- LEAKING SPLIT.\n"
            f"  eval  {eval_path} was built with problem_seed={eval_seed}\n"
            f"  train {train_path} includes that seed: {sorted(train_seeds)}\n"
            "  Every evaluation row is then a re-measurement of a training\n"
            "  row, so the result would NOT be a held-out test.\n"
            "  Fix: rebuild the eval matrix with a PROBLEM_SEED outside the\n"
            "  training seed list (phase2_build_matrix.py).")

    # Feature sampling must match or the two matrices are not comparable.
    for key in ("feature_seed", "feature_samples"):
        if tr.get(key) != ev.get(key):
            sys.exit(f"REFUSING TO RUN: {key} differs between train "
                     f"({tr.get(key)}) and eval ({ev.get(key)}). Feature "
                     f"vectors would not be comparable.")

    print(f"  split check     : OK -- eval seed {eval_seed} not in training "
          f"seeds {sorted(train_seeds)}")
    return eval_seed


def main():
    ap = argparse.ArgumentParser(
        description="Phase 3b pre-registered confirmation run.")
    ap.add_argument("--train-matrix", default=TRAIN_PATH,
                    help="selector TRAINING matrix (default: %(default)s)")
    ap.add_argument("--eval-matrix", default=EVAL_PATH,
                    help="held-out EVALUATION matrix (default: %(default)s)")
    ap.add_argument("--out-stem", default=OUT_STEM)
    ap.add_argument("--allow-leakage", action="store_true",
                    help="skip the train/eval seed-overlap check. ONLY for "
                         "reproducing the pre-2026-09-20 result, which was "
                         "computed on a non-independent evaluation set.")
    args = ap.parse_args()

    train_path, eval_path, out_stem = (args.train_matrix, args.eval_matrix,
                                       args.out_stem)

    if args.allow_leakage:
        print("  WARNING: --allow-leakage set; the split is NOT verified "
              "independent.")
    else:
        assert_no_leakage(train_path, eval_path)

    algos, fkeys, Ftr_all, Mtr, ftr, dtr = load_train(train_path)
    erows, Fte_all, Mte, fte, edim, eseed = load_eval(eval_path, algos, fkeys)
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
        detail.append({"function": str(f), "dim": int(edim[i]), "seed": eseed,
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

    # AGGREGATE TO FUNCTION MEANS BEFORE TESTING.
    # With several dimensions per function the rows are NOT independent: they
    # are 3 dims within each of 24 functions. Pairing all rows would inflate
    # significance exactly as it did in Phase 2c, where RF-regression read
    # p=0.0334 over 96 raw rows but p=0.4348 over the 24 function means. The
    # honest N is the number of FUNCTIONS. With one dim per function this
    # collapses to the per-row test, so the single-dim result is unchanged.
    ufuncs = sorted(set(fte.tolist()))
    fold_sel = np.array([sel_rank[fte == f].mean() for f in ufuncs])
    fold_sbs = np.array([sbs_rank[fte == f].mean() for f in ufuncs])

    d = fold_sel - fold_sbs
    if np.all(d == 0):
        stat, pv = float("nan"), float("nan")
    else:
        stat, pv = wilcoxon(fold_sel, fold_sbs)          # THE one pre-specified test

    L = []
    w = L.append
    w("=" * 78)
    w("  PHASE 3b -- CONFIRMATION RUN (pre-registered)")
    w("=" * 78)
    w("  train           : " + train_path)
    w("  eval            : " + eval_path)
    w("  split verified  : " + ("NOT VERIFIED (--allow-leakage)"
                               if args.allow_leakage else
                               "eval seed not in training seeds"))
    w("  split           : leave-one-FUNCTION-out ACROSS files")
    w("  eval rows       : {}   algorithms: {}   seed: {}".format(
        len(erows), n_alg, eseed))
    w("  eval dims       : {}   functions: {}".format(
        sorted(set(edim.tolist())), len(ufuncs)))
    w("  paired test N   : {} function means (rows aggregated first)".format(
        len(ufuncs)))
    w("  seeds recorded  : RF_SEED={}, n_estimators={}".format(RF_SEED, RF_TREES))
    w("  scipy/sklearn   : {} / {}".format(__import__("scipy").__version__,
                                           __import__("sklearn").__version__))
    w("")
    if args.allow_leakage:
        w("  *** NOT AN INDEPENDENT HELD-OUT SET (--allow-leakage). The eval")
        w("  rows re-measure configurations present in training. Tests only")
        w("  generalisation to an unseen function and target-noise")
        w("  robustness. Does NOT resolve the post-hoc selection risk.")
    else:
        w("  Independent split verified: the eval matrix's problem_seed is")
        w("  absent from the training matrix's pseudo_instance_seeds, so no")
        w("  eval row re-measures a training configuration.")
    w("")
    w("  MEAN RANK on held-out rows (1 = best of {}, lower is better)".format(n_alg))
    w("  " + "-" * 60)
    w("  {:<22}{:>10.4f}".format("VBS (oracle)", vbs_m))
    w("  {:<22}{:>10.4f}".format("Selector (RF)", sel_m))
    w("  {:<22}{:>10.4f}".format("SBS (fold-aware)", sbs_m))
    w("  {:<22}{:>10.4f}".format("Random (analytic)", rnd_m))
    w("")
    w("  SBS->VBS gap closed : {:+.1f}%".format(closed))
    w("")
    w("  WILCOXON  selector vs SBS, paired over {} function means "
      "(one test, alpha=0.05)".format(len(ufuncs)))
    w("  signature: wilcoxon(x, y=None, zero_method='wilcox', correction=False,")
    w("             alternative='two-sided', method='auto', *, axis=0, ...)")
    w("  statistic={!r}  p={!r}".format(stat, pv))
    w("  significant at alpha=0.05: {}".format("YES" if (pv == pv and pv < 0.05) else "NO"))
    w("  selector better on {}, worse on {}, tied on {}".format(
        int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())))
    w("=" * 78)

    txt = "\n".join(L)
    print(txt)
    open(out_stem + ".txt", "w", encoding="utf-8").write(txt + "\n")
    json.dump({"detail": detail, "selector_mean_rank": sel_m, "sbs_mean_rank": sbs_m,
               "vbs_mean_rank": vbs_m, "random_mean_rank": rnd_m,
               "gap_closed_pct": closed,
               "wilcoxon_statistic": None if stat != stat else float(stat),
               "wilcoxon_p": None if pv != pv else float(pv),
               "rf_seed": RF_SEED, "rf_trees": RF_TREES,
               "eval_dims": sorted(set(edim.tolist())), "eval_seed": eseed,
               "independent_holdout": not args.allow_leakage},
              open(out_stem + ".json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
