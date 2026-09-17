"""
PHASE 3 ABLATION -- isolate why phase2_lopo.py and phase3_selector_validation.py
disagree on identical data.

Two factors, 2x2:
  TARGET        : 'rank' (phase3) vs 'norm' (phase2_lopo, per-row min-max)
  ARCHITECTURE  : 'single'   -- one RF over (features + algorithm one-hot)
                  'per_algo' -- 22 separate RFs, one per algorithm

Everything else is HELD CONSTANT across all four cells: n_estimators, seed,
feature preprocessing, fold rule, SBS definition, and the scoring code.
Reads cached JSON only. No optimiser is invoked.

Each cell is scored on BOTH scales (rank and norm) so the four cells are
directly comparable and neither scale is cherry-picked.
"""
import json

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor

from phase3_selector_validation import load

TREES = 300
SEED = 20260916
PREP = False          # see prep_features(); flipped by the control check below


def prep_features(Ftr, Fte):
    """phase2_lopo.py's preprocessing: signed-log1p, drop zero-variance, standardise."""
    from sklearn.preprocessing import StandardScaler

    def slog(X):
        return np.sign(X) * np.log1p(np.abs(X))

    Xtr, Xte = slog(Ftr), slog(Fte)
    keep = Xtr.std(axis=0) > 0
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def pick_single(Ftr, Fte, Ytr, n_alg, trees, seed):
    """One RF over (features + algorithm one-hot); argmin over algorithms."""
    Xtr = np.hstack([np.repeat(Ftr, n_alg, axis=0), np.tile(np.eye(n_alg), (len(Ftr), 1))])
    rf = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, Ytr.reshape(-1))
    Xte = np.hstack([np.repeat(Fte, n_alg, axis=0), np.eye(n_alg)])
    return int(np.argmin(rf.predict(Xte)))


def pick_per_algo(Ftr, Fte, Ytr, n_alg, trees, seed):
    """One RF per algorithm mapping features -> that algorithm's target; argmin."""
    preds = []
    for j in range(n_alg):
        rf = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=-1)
        rf.fit(Ftr, Ytr[:, j])
        preds.append(rf.predict(Fte)[0])
    return int(np.argmin(preds))


def run(F, M, ranks, N, funcs, target, arch, trees=TREES, seed=SEED, prep=PREP):
    n_row, n_alg = M.shape
    Y = ranks if target == "rank" else N
    picker = pick_single if arch == "single" else pick_per_algo
    ufuncs = sorted(set(funcs.tolist()))

    sel_r, sbs_r, sel_n, sbs_n, fold_of = [], [], [], [], []
    for f in ufuncs:
        te = np.where(funcs == f)[0]
        tr = np.where(funcs != f)[0]
        sbs_j = int(np.argmin(ranks[tr].mean(axis=0)))      # TRAIN rows only
        for i in te:
            Ftr, Fte = (prep_features(F[tr], F[i:i + 1]) if prep
                        else (F[tr], F[i:i + 1]))
            j = picker(Ftr, Fte, Y[tr], n_alg, trees, seed)
            sel_r.append(ranks[i, j]); sbs_r.append(ranks[i, sbs_j])
            sel_n.append(N[i, j]);     sbs_n.append(N[i, sbs_j])
            fold_of.append(f)

    fold_of = np.array(fold_of)
    out = {}
    for scale, sel, sbs in (("rank", np.array(sel_r), np.array(sbs_r)),
                            ("norm", np.array(sel_n), np.array(sbs_n))):
        vbs = (ranks.min(axis=1) if scale == "rank" else np.zeros(n_row))
        fs = np.array([sel[fold_of == f].mean() for f in ufuncs])
        fb = np.array([sbs[fold_of == f].mean() for f in ufuncs])
        fv = np.array([vbs[np.where(funcs == f)[0]].mean() for f in ufuncs])
        gap = sbs.mean() - vbs.mean()
        closed = (sbs.mean() - sel.mean()) / gap * 100.0 if gap > 0 else float("nan")
        d = fs - fb
        stat, pv = ((float("nan"), float("nan")) if np.all(d == 0)
                    else wilcoxon(fs, fb))
        out[scale] = {"sel_mean": float(sel.mean()), "sbs_mean": float(sbs.mean()),
                      "vbs_mean": float(vbs.mean()), "closed_pct": float(closed),
                      "stat": float(stat), "p": float(pv),
                      "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
                      "tied": int((d == 0).sum())}
    return out


def main():
    rows, algos, fkeys, F, M, funcs, seeds, dims = load("results/phase2c_matrix.json")
    n_row, n_alg = M.shape
    ranks = np.vstack([rankdata(M[i]) for i in range(n_row)])
    lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
    N = (M - lo) / np.where(hi - lo == 0, 1.0, hi - lo)

    print("rows={} algorithms={} folds={} trees={} seed={} prep={}".format(
        n_row, n_alg, len(set(funcs.tolist())), TREES, SEED, PREP))
    print()
    results = {}
    for target in ("rank", "norm"):
        for arch in ("single", "per_algo"):
            r = run(F, M, ranks, N, funcs, target, arch)
            results[target + "|" + arch] = r
            print("target={:<5} arch={:<9} | rank: closed={:+7.1f}% p={:.6f}"
                  "  | norm: closed={:+7.1f}% p={:.6f}".format(
                      target, arch, r["rank"]["closed_pct"], r["rank"]["p"],
                      r["norm"]["closed_pct"], r["norm"]["p"]))

    print()
    print("2x2 TABLE -- gap closed %% (Wilcoxon p), scored on RANK scale")
    print("  {:<12}{:>22}{:>22}".format("", "single+one-hot", "22 per-algo RFs"))
    for target in ("rank", "norm"):
        cells = []
        for arch in ("single", "per_algo"):
            c = results[target + "|" + arch]["rank"]
            cells.append("{:+.1f}% (p={:.4f})".format(c["closed_pct"], c["p"]))
        print("  {:<12}{:>22}{:>22}".format("target=" + target, *cells))

    print()
    print("2x2 TABLE -- gap closed %% (Wilcoxon p), scored on NORM scale")
    print("  {:<12}{:>22}{:>22}".format("", "single+one-hot", "22 per-algo RFs"))
    for target in ("rank", "norm"):
        cells = []
        for arch in ("single", "per_algo"):
            c = results[target + "|" + arch]["norm"]
            cells.append("{:+.1f}% (p={:.4f})".format(c["closed_pct"], c["p"]))
        print("  {:<12}{:>22}{:>22}".format("target=" + target, *cells))

    # CONTROL: do the held-constant knobs matter? Re-run the phase2_lopo-like
    # cell under phase2_lopo's own trees/seed/preprocessing.
    print()
    ctl = run(F, M, ranks, N, funcs, "norm", "per_algo",
              trees=200, seed=20260915, prep=True)
    print("CONTROL norm|per_algo @ trees=200 seed=20260915 prep=True :"
          " norm closed={:+.1f}% p={:.6f}".format(
              ctl["norm"]["closed_pct"], ctl["norm"]["p"]))

    json.dump({"grid": results, "control_phase2_knobs": ctl,
               "trees": TREES, "seed": SEED, "prep": PREP},
              open("results/phase3_ablation.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
