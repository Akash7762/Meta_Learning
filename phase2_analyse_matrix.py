"""
phase2_analyse_matrix.py
------------------------
Offline analysis of the Phase 2 performance matrix.

This script NEVER invokes an optimiser. It reads results/phase2_matrix.json
and nothing else, so every number here is reproducible from that file alone.

Contents
--------
  performance matrix   rows = problems, cols = algorithms, cell = median
                       best-fitness over the 15 seeded runs. Median, not mean:
                       the Phase 1 data was heavily right-skewed and a single
                       bad run distorts the mean.
  ranks                per-problem rank, 1 = best (lowest median; all BBOB
                       functions here are minimised).
  SBS                  Single Best Solver - best mean rank over all problems.
  VBS                  Virtual Best Solver - the per-problem oracle.
  SBS-VBS gap          the interval any selector must close.
  Friedman test        scipy.stats.friedmanchisquare over all algorithms.

AGGREGATING THE SBS-VBS GAP ACROSS PROBLEMS
-------------------------------------------
Raw fitness cannot be averaged across these problems: the medians span many
orders of magnitude (the exact range depends on the matrix) and some problems
(F5, F19, F20) have NEGATIVE optima, which makes ratios and log-ratios
undefined.

So the gap is aggregated two ways, both scale-free:

  1. NORMALISED GAP (primary). Per problem, map medians onto [0, 1] with
         g(x) = (x - VBS) / (worst - VBS)
     where VBS is the best median and worst is the worst median on that
     problem. VBS scores 0, the worst algorithm scores 1. This is well defined
     for negative values, unlike a ratio, and bounded, so one extreme problem
     cannot dominate the average.

  2. RANK GAP (secondary). VBS holds rank 1 by construction, so the gap is
     simply (SBS mean rank - 1). Reported because it is distribution-free and
     immune to the normalisation choice.

Run:
    .venv\\Scripts\\python.exe phase2_analyse_matrix.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import scipy
from scipy.stats import friedmanchisquare, rankdata

IN_PATH = os.path.join("results", "phase2_matrix.json")
CSV_PATH = os.path.join("results", "phase2_matrix.csv")

BAR = "=" * 100


def load():
    if not os.path.exists(IN_PATH):
        print(f"ERROR: {IN_PATH} not found. Run phase2_build_matrix.py first.")
        sys.exit(1)
    with open(IN_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def build_matrix(data):
    """Return (rows, algorithms, M) with M[i, j] = median for row i, algorithm j.

    Two layouts are supported:
      old  -- meta.problems present, cell key "problem|algorithm"
      new  -- per-dimension rows keyed "function|sSEED|dDIM", cell key
              "<row>|algorithm". Row order is taken from meta.functions x
              meta.dimensions so the printed tables stay in function order
              rather than lexicographic (which would put d10 before d5).
    """
    meta = data["meta"]
    if "problems" in meta:                      # pre-2026-09-22 single-dim
        P = meta["problems"]
    else:
        seed = meta["problem_seed"]
        P = [f"{fn}|s{seed}|d{dim}"
             for fn in meta["functions"] for dim in meta["dimensions"]]
    A = data["meta"]["algorithms"]
    C = data["cells"]

    M = np.full((len(P), len(A)), np.nan)
    missing = []
    for i, p in enumerate(P):
        for j, a in enumerate(A):
            cell = C.get(f"{p}|{a}")
            if cell is None or cell["median"] is None:
                missing.append(f"{p}|{a}")
                continue
            M[i, j] = cell["median"]
    return P, A, M, missing


def integrity_report(data, P, A, M, missing):
    print(BAR)
    print("  INTEGRITY")
    print(BAR)
    C = data["cells"]
    n_nonfinite = sum(c["n_nonfinite"] for c in C.values())
    bad_cells = [k for k, c in C.items() if c["n_nonfinite"]]
    errs = [k for k, c in C.items() if c["errors"]]

    print(f"  cells expected            : {len(P) * len(A)}")
    print(f"  cells present             : {len(C)}")
    print(f"  cells missing             : {len(missing)} {missing[:5]}")
    print(f"  runs per cell             : {sorted({c['n_runs_requested'] for c in C.values()})}")
    print(f"  finite runs per cell      : {sorted({c['n_runs_finite'] for c in C.values()})}")
    print(f"  total NaN/Inf runs        : {n_nonfinite}")
    print(f"  cells containing NaN/Inf  : {len(bad_cells)} {bad_cells[:5]}")
    print(f"  cells with errors         : {len(errs)} {errs[:5]}")
    print(f"  matrix build wall-clock   : {data['build_seconds']:.1f}s "
          f"= {data['build_seconds'] / 3600:.2f} hours")
    print(f"  seeds per cell            : {data['meta']['run_seeds']}")
    print(f"  budget                    : population_size="
          f"{data['meta']['population_size']}, max_fe={data['meta']['max_fe']} "
          f"('ea': max_generations={data['meta']['ea_max_generations']})")

    # Duplicate-row check: two problems with identical medians everywhere are
    # almost certainly the same function under two names.
    dupes = []
    for i in range(len(P)):
        for k in range(i + 1, len(P)):
            if np.allclose(M[i], M[k], rtol=0, atol=0, equal_nan=True):
                dupes.append((P[i], P[k]))
    print(f"  identical problem rows    : {len(dupes)} {dupes}")
    if dupes:
        print("    ^ WARNING: identical rows mean those problems are not")
        print("      distinct benchmarks. Investigate before using the matrix.")
    return dupes


def print_matrix(P, A, M, n_runs="?"):
    print()
    print(BAR)
    print(f"  PERFORMANCE MATRIX  (median best-fitness over {n_runs} runs; lower is better)")
    print(BAR)
    chunk = 7
    for s in range(0, len(A), chunk):
        cols = A[s:s + chunk]
        print(f"\n  {'problem':<34}" + "".join(f"{c:>13}" for c in cols))
        print("  " + "-" * (34 + 13 * len(cols)))
        for i, p in enumerate(P):
            row = "".join(f"{M[i, s + j]:>13.4e}" for j in range(len(cols)))
            print(f"  {p:<34}{row}")


def write_csv(P, A, M):
    os.makedirs("results", exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem"] + A)
        for i, p in enumerate(P):
            w.writerow([p] + [repr(float(M[i, j])) for j in range(len(A))])
    print(f"\n  matrix written to {CSV_PATH}")


def analyse(P, A, M):
    n_prob, n_alg = M.shape

    # --- ranks: 1 = best (lowest). Ties get the average rank. ---
    R = np.vstack([rankdata(M[i, :], method="average") for i in range(n_prob)])
    mean_rank = R.mean(axis=0)

    print()
    print(BAR)
    print("  MEAN RANK  (1 = best; averaged over all problems)")
    print(BAR)
    order = np.argsort(mean_rank)
    print(f"  {'rank':<6}{'algorithm':<10}{'mean rank':>12}{'wins (rank 1)':>15}")
    print("  " + "-" * 43)
    wins = (R == 1).sum(axis=0)
    for pos, j in enumerate(order, 1):
        print(f"  {pos:<6}{A[j]:<10}{mean_rank[j]:>12.3f}{wins[j]:>15}")

    sbs_j = int(order[0])
    sbs = A[sbs_j]

    # --- VBS: per-problem oracle ---
    vbs_val = M.min(axis=1)
    vbs_alg = [A[int(j)] for j in M.argmin(axis=1)]
    worst_val = M.max(axis=1)
    sbs_val = M[:, sbs_j]

    print()
    print(BAR)
    print(f"  SBS = {sbs}  (mean rank {mean_rank[sbs_j]:.3f})")
    print(BAR)
    print(f"  {'problem':<34}{'VBS':>13}{'VBS alg':>10}{'SBS':>13}{'norm gap':>11}")
    print("  " + "-" * 81)

    norm_gaps = []
    for i, p in enumerate(P):
        denom = worst_val[i] - vbs_val[i]
        g = 0.0 if denom == 0 else (sbs_val[i] - vbs_val[i]) / denom
        norm_gaps.append(g)
        print(f"  {p:<34}{vbs_val[i]:>13.4e}{vbs_alg[i]:>10}"
              f"{sbs_val[i]:>13.4e}{g:>11.4f}")

    norm_gaps = np.asarray(norm_gaps)

    print()
    print(BAR)
    print("  SBS-to-VBS GAP")
    print(BAR)
    print("  Normalised gap per problem: (SBS - VBS) / (worst - VBS), in [0, 1].")
    print("  0 = SBS matched the oracle on that problem, 1 = SBS was the worst.")
    print()
    print(f"  mean normalised gap       : {norm_gaps.mean():.4f}")
    print(f"  median normalised gap     : {np.median(norm_gaps):.4f}")
    print(f"  max normalised gap        : {norm_gaps.max():.4f}"
          f"  (on {P[int(norm_gaps.argmax())]})")
    print(f"  problems where SBS == VBS : {int((norm_gaps == 0).sum())} / {n_prob}")
    print()
    print(f"  rank gap (SBS mean rank - 1): {mean_rank[sbs_j] - 1:.3f}")
    print(f"  VBS distinct algorithms     : {len(set(vbs_alg))} -> {sorted(set(vbs_alg))}")

    # --- Friedman ---
    # friedmanchisquare(*samples): one array per TREATMENT (algorithm), each
    # holding one observation per BLOCK (problem). It ranks within blocks, so
    # differing scales across problems do not distort it.
    print()
    print(BAR)
    print("  FRIEDMAN TEST")
    print(BAR)
    print(f"  scipy version : {scipy.__version__}")
    print(f"  signature     : friedmanchisquare(*samples, axis=0, "
          f"nan_policy='propagate', keepdims=False)")
    print(f"  input         : {n_alg} samples (algorithms), "
          f"each of length {n_prob} (problems)")
    samples = [M[:, j] for j in range(n_alg)]
    stat, p_value = friedmanchisquare(*samples)
    print(f"  statistic     : {stat!r}")
    print(f"  p-value       : {p_value!r}")
    alpha = 0.05
    sig = p_value < alpha
    print(f"  alpha         : {alpha}")
    print(f"  significant?  : {'YES' if sig else 'NO'} - algorithms "
          f"{'do NOT all perform equally' if sig else 'are indistinguishable'}")

    return sbs, mean_rank[sbs_j], norm_gaps, stat, p_value


def main() -> int:
    global IN_PATH
    ap = argparse.ArgumentParser(description="Analyse a cached matrix.")
    ap.add_argument("--matrix", default=IN_PATH,
                    help="matrix JSON to analyse (default: %(default)s)")
    IN_PATH = ap.parse_args().matrix
    data = load()
    P, A, M, missing = build_matrix(data)

    if missing:
        print(f"WARNING: {len(missing)} cells missing. They are excluded, never imputed.")

    dupes = integrity_report(data, P, A, M, missing)
    print_matrix(P, A, M, data["meta"].get("n_runs", "?"))
    write_csv(P, A, M)
    analyse(P, A, M)

    print()
    print(BAR)
    print("  DONE - no optimiser was invoked; all values read from "
          f"{IN_PATH}")
    print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
