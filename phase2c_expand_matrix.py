"""
phase2c_expand_matrix.py
------------------------
Expand the performance matrix along the PROBLEM axis, to give a feature-based
selector more than 23 training rows.

Row set
-------
    24 BBOB functions  x  PSEUDO_INSTANCE_SEEDS  x  DIMENSIONS
    x 22 algorithms = the cell count

Deliberately NOT written out as numbers here. An earlier version of this
header claimed "5 pseudo-instances x 3 dimensions = 360 rows" while the
constants below built 96 rows, and that contradiction is part of why the
Phase 2c spec deviation went unnoticed. The constants are the single source
of truth; the script prints the actual row/cell count at startup.

Cost scales linearly with n_runs at ~1.35 s per run per cell, measured across
both builds (10771.8s/528 cells at n_runs=15; 8508.3s/2112 cells at n_runs=3).

TERMINOLOGY -- "PSEUDO-INSTANCE", NOT "BBOB INSTANCE"
-----------------------------------------------------
aad/problems.py has no concept of a COCO/BBOB instance: `grep -ric instance
aad/problems.py` returns 0. What the `seed` parameter of _BBOBBase actually
varies is the global optimum `xopt` and the two rotation matrices `R` and `Q`,
both drawn from `np.random.RandomState(seed)`.

That is functionally instance-LIKE -- same function family, different optimum
location and rotation -- but it is NOT the official COCO instance scheme:
there is no f_opt offset and none of the COCO-defined per-instance
transformations. These rows are therefore called pseudo-instances throughout,
and each row records the seed that produced it. Do not describe them as BBOB
instances in the thesis.

BUDGET
------
max_fe=5000, population_size=50 -- identical to results/phase2_matrix.json, so
the two matrices are comparable. max_generations is deliberately NOT used: it
gives SA ~150 evaluations against everyone else's ~5050 (Phase 1b), a 33x
handicap unrelated to algorithm merit.

The built-in "ea" does not read max_fe, so it is given max_generations=100,
which measures at ~5050 evaluations -- 1% above the mealpy cells. Recorded per
cell in "budget_note", as in Phase 2.

STATUS OF THIS FILE
-------------------
This matrix uses fewer runs per cell than Phase 2 (see N_RUNS). It exists for
SELECTOR TRAINING, where many moderately-noisy rows beat few precise ones. results/phase2_matrix.json
(n_runs=15) remains the authoritative source for SBS/VBS headline figures.
Both files state this in their meta block.

DURABILITY
----------
Every completed cell is appended to results/phase2c_matrix.jsonl and flushed
immediately (O(1)). A full structured snapshot of results/phase2c_matrix.json
is written after each completed row and at the end. On restart both are read
and merged, so an interruption loses at most the cell in flight. Writing the
whole JSON after every cell was measured at ~0.21s x 7920 = ~28 minutes of
pure I/O, which this avoids.

Run:
    .venv\\Scripts\\python.exe phase2c_expand_matrix.py --dry-run
    .venv\\Scripts\\python.exe phase2c_expand_matrix.py
    .venv\\Scripts\\python.exe phase2c_expand_matrix.py --verify-only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import aad
from aad.evolutionary_algorithm import build_algorithm
from aad.landscape_features import LandscapeFeatureExtractor
from aad.mealpy_algorithms import MEALPY_REGISTRY

# ---------------------------------------------------------------------------
# Fixed experiment parameters
# ---------------------------------------------------------------------------

DIMENSIONS = [5, 10, 20]
PSEUDO_INSTANCE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   # -> xopt + R + Q per row
POPULATION_SIZE = 50
MAX_FE = 5000
EA_MAX_GENERATIONS = 100
N_RUNS = 5
RUN_SEEDS = [1000 + i for i in range(N_RUNS)]

EA_OPERATORS = {"selection": "tournament", "crossover": "sbx",
                "mutation": "gaussian"}

FEATURE_SAMPLES = 300
FEATURE_SEED = 20260915

OUT_DIR = "results"
OUT_JSON = os.path.join(OUT_DIR, "phase2c_matrix.json")
OUT_JSONL = os.path.join(OUT_DIR, "phase2c_matrix.jsonl")

FUNCTIONS = [n for n in aad.__all__ if n.startswith("BBOB_")]
ALGORITHMS = sorted(MEALPY_REGISTRY.keys()) + ["ea"]

CELL_KEY_FORMAT = "{function}|s{seed}|d{dim}|{algorithm}"
ROW_KEY_FORMAT = "{function}|s{seed}|d{dim}"

BAR = "=" * 86


def row_key(fn, seed, dim):
    return ROW_KEY_FORMAT.format(function=fn, seed=seed, dim=dim)


def cell_key(fn, seed, dim, alg):
    return CELL_KEY_FORMAT.format(function=fn, seed=seed, dim=dim, algorithm=alg)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _env_versions():
    """Library versions that produced this matrix. Seeds alone do not make a
    build reproducible -- a different numpy/scipy/mealpy can yield different
    numbers from an identical seed."""
    import platform
    import numpy
    import scipy
    out = {"python": platform.python_version(),
           "numpy": numpy.__version__,
           "scipy": scipy.__version__}
    try:
        import mealpy
        out["mealpy"] = getattr(mealpy, "__version__", "unknown")
    except Exception:
        out["mealpy"] = "unavailable"
    return out


def fresh_state(functions, seeds, dims, algorithms, n_runs):
    return {
        "meta": {
            # Built from n_runs, never hardcoded: this string goes into the
            # data file and would otherwise mislabel the matrix in the thesis.
            "purpose": ("SELECTOR TRAINING ONLY. n_runs={} makes per-cell "
                        "medians noisier than Phase 2. "
                        "results/phase2_matrix.json (n_runs=15) remains the "
                        "AUTHORITATIVE source for SBS/VBS headline "
                        "figures.".format(n_runs)),
            "n_runs": n_runs,
            "max_fe": MAX_FE,
            "population_size": POPULATION_SIZE,
            "ea_max_generations": EA_MAX_GENERATIONS,
            "ea_operators": EA_OPERATORS,
            "run_seeds": RUN_SEEDS[:n_runs],
            "cell_key_format": CELL_KEY_FORMAT,
            "row_key_format": ROW_KEY_FORMAT,
            "pseudo_instance_note": (
                "Rows labelled sNN are PSEUDO-INSTANCES, not COCO/BBOB "
                "instances. aad/problems.py has no instance concept; the seed "
                "varies xopt and the rotation matrices R and Q only. There is "
                "no f_opt offset and no COCO-defined instance transformation."),
            "pseudo_instance_seeds": seeds,
            "dimensions": dims,
            "functions": functions,
            "algorithms": algorithms,
            "feature_samples": FEATURE_SAMPLES,
            "feature_seed": FEATURE_SEED,
            "authoritative_matrix": "results/phase2_matrix.json",
            "env": _env_versions(),
        },
        "features": {},
        "cells": {},
        "build_seconds": 0.0,
    }


def load_state(functions, seeds, dims, algorithms, n_runs):
    """Load snapshot + replay the append-only log. Never discards data."""
    state = fresh_state(functions, seeds, dims, algorithms, n_runs)
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as fh:
            disk = json.load(fh)

        # n_runs is NOT part of the cell key, so a resume with a different
        # n_runs would skip every existing cell and append new ones at a
        # different precision -- producing a matrix that silently mixes, say,
        # 3-run and 15-run medians with nothing to distinguish them. Refuse.
        disk_runs = disk.get("meta", {}).get("n_runs")
        if disk_runs is not None and disk_runs != n_runs:
            sys.exit("\n".join([
                f"REFUSING TO RESUME: {OUT_JSON} was built with n_runs={disk_runs},",
                f"  but this run is configured for n_runs={n_runs}.",
                "  n_runs is not part of the cell key, so resuming would mix",
                "  medians of different precision in one matrix.",
                f"  Either set N_RUNS={disk_runs}, or build to a NEW filename",
                "  (change OUT_JSON/OUT_JSONL), or delete the existing file."]))

        # Same hazard for the feature sampler: features cached under a
        # different seed/sample count are not comparable with new ones.
        for key, cur in (("feature_seed", FEATURE_SEED),
                         ("feature_samples", FEATURE_SAMPLES)):
            was = disk.get("meta", {}).get(key)
            if was is not None and was != cur:
                sys.exit("\n".join([
                    f"REFUSING TO RESUME: {OUT_JSON} has {key}={was}, this run",
                    f"  uses {cur}. Cached feature rows would not be",
                    "  comparable with new ones."]))

        state["features"].update(disk.get("features", {}))
        state["cells"].update(disk.get("cells", {}))
        state["build_seconds"] = disk.get("build_seconds", 0.0)

    replayed = 0
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # torn final line from a hard kill
                if rec.get("_kind") == "feature":
                    state["features"].setdefault(rec["key"], rec["value"])
                elif rec.get("_kind") == "cell" and rec["key"] not in state["cells"]:
                    state["cells"][rec["key"]] = rec["value"]
                    state["build_seconds"] += rec["value"].get("wall_seconds", 0.0)
                    replayed += 1

    if state["cells"]:
        print(f"  resuming: {len(state['cells'])} cells cached "
              f"({replayed} replayed from the log), "
              f"{len(state['features'])} feature rows")
    return state


def append_log(kind, key, value):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"_kind": kind, "key": key, "value": value}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def save_snapshot(state):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, OUT_JSON)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def make_config(algorithm, seed):
    if algorithm == "ea":
        return {"algorithm": "ea", "population_size": POPULATION_SIZE,
                "max_generations": EA_MAX_GENERATIONS, **EA_OPERATORS}
    return {"algorithm": algorithm, "population_size": POPULATION_SIZE,
            "max_fe": MAX_FE, "seed": seed}


def run_once(algorithm, problem, seed):
    """Raw best fitness. Never sanitised -- non-finite stays visible."""
    if algorithm == "ea":
        np.random.seed(seed)        # "ea" draws from NumPy's global RNG
    _, best = build_algorithm(make_config(algorithm, seed)).run(problem)
    return float(best)


def run_cell(algorithm, problem, fn, seed, dim, n_runs, run_seeds):
    t0 = time.perf_counter()
    raw, errors = [], []
    for s in run_seeds[:n_runs]:
        try:
            raw.append(run_once(algorithm, problem, s))
        except Exception as exc:                       # noqa: BLE001
            raw.append(None)
            errors.append(f"seed={s}: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - t0

    finite = [v for v in raw if v is not None and np.isfinite(v)]
    arr = np.asarray(finite, dtype=float)
    stats = ({"mean": float(np.mean(arr)), "median": float(np.median(arr)),
              "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
              "min": float(np.min(arr)), "max": float(np.max(arr))}
             if len(arr) else
             {k: None for k in ("mean", "median", "std", "min", "max")})

    return {
        "algorithm": algorithm,
        "function": fn,
        "pseudo_instance_seed": seed,
        "dim": dim,
        "fitnesses": [v if (v is not None and np.isfinite(v)) else None for v in raw],
        "run_seeds": run_seeds[:n_runs],
        **stats,
        "n_runs_requested": n_runs,
        "n_runs_finite": len(finite),
        "n_nonfinite": sum(1 for v in raw if v is None or not np.isfinite(v)),
        "errors": errors,
        "wall_seconds": round(elapsed, 3),
        "budget_note": ("max_generations=100 (~5050 evals; +1% vs mealpy cells)"
                        if algorithm == "ea" else f"max_fe={MAX_FE}"),
    }


# ---------------------------------------------------------------------------
# Step 4 -- feature variance verification
# ---------------------------------------------------------------------------

def verify_features(state, near_zero=1e-12):
    feats = state["features"]
    if not feats:
        print("  no features recorded; nothing to verify")
        return False, []

    keys = sorted(next(iter(feats.values())).keys())
    rows = sorted(feats)
    X = np.array([[feats[r][k] for k in keys] for r in rows], dtype=float)

    print(BAR)
    print("  STEP 4 - FEATURE VARIANCE ACROSS ROWS")
    print(BAR)
    print(f"  rows with features : {len(rows)}")
    print(f"  feature columns    : {len(keys)}")
    print()
    print(f"  {'feature':<22}{'std':>16}{'min':>16}{'max':>16}{'distinct':>10}  status")
    print("  " + "-" * 88)

    dead = []
    for j, k in enumerate(keys):
        col = X[:, j]
        sd = float(np.std(col))
        nd = len(np.unique(col))
        if sd <= near_zero or nd == 1:
            status = "CONSTANT - carries no signal"
            dead.append(k)
        elif nd < max(2, len(rows) // 20):
            status = "near-degenerate (few distinct values)"
        else:
            status = "ok"
        print(f"  {k:<22}{sd:>16.6g}{col.min():>16.6g}{col.max():>16.6g}"
              f"{nd:>10}  {status}")

    print()
    if dead:
        print(f"  *** {len(dead)} CONSTANT FEATURE COLUMN(S): {dead}")
        print("  *** A constant column gives the surrogate nothing to learn from.")
        print("  *** Reported before any analysis, as required.")
    else:
        print("  No constant feature columns. Every feature varies across rows.")
    print(BAR)
    return (not dead), dead


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="2 functions x 2 seeds x 2 dims x 3 algorithms x 2 runs")
    ap.add_argument("--verify-only", action="store_true",
                    help="run the Step 4 feature check on the existing file and exit")
    args = ap.parse_args()

    global OUT_JSON, OUT_JSONL
    functions, seeds, dims = FUNCTIONS, PSEUDO_INSTANCE_SEEDS, DIMENSIONS
    algorithms, n_runs, run_seeds = ALGORITHMS, N_RUNS, RUN_SEEDS

    if args.dry_run:
        OUT_JSON = os.path.join(OUT_DIR, "phase2c_matrix_dryrun.json")
        OUT_JSONL = os.path.join(OUT_DIR, "phase2c_matrix_dryrun.jsonl")
        functions, seeds, dims = FUNCTIONS[:2], PSEUDO_INSTANCE_SEEDS[:2], DIMENSIONS[:2]
        algorithms, n_runs = ["PSO", "SA", "ea"], 2
        run_seeds = RUN_SEEDS[:2] if len(RUN_SEEDS) >= 2 else [1000, 1001]

    if args.verify_only:
        if not os.path.exists(OUT_JSON):
            print(f"ERROR: {OUT_JSON} not found."); return 1
        with open(OUT_JSON, encoding="utf-8") as fh:
            verify_features(json.load(fh))
        return 0

    n_rows = len(functions) * len(seeds) * len(dims)
    total = n_rows * len(algorithms)

    print(BAR)
    print("  PHASE 2c - PROBLEM-AXIS EXPANSION" + ("  [DRY RUN]" if args.dry_run else ""))
    print(BAR)
    print(f"  functions          : {len(functions)}")
    print(f"  pseudo-instances   : {len(seeds)}  seeds={seeds}")
    print(f"  dimensions         : {dims}")
    print(f"  rows               : {n_rows}")
    print(f"  algorithms         : {len(algorithms)}")
    print(f"  cells              : {total}")
    print(f"  runs per cell      : {n_runs}  seeds={run_seeds[:n_runs]}")
    print(f"  budget             : population_size={POPULATION_SIZE}, max_fe={MAX_FE}")
    print(f"                       ('ea': max_generations={EA_MAX_GENERATIONS})")
    print(f"  cell key format    : {CELL_KEY_FORMAT}")
    print(f"  output             : {OUT_JSON}")
    print(f"  durability log     : {OUT_JSONL}")
    print(f"  NOTE               : n_runs={n_runs} -> SELECTOR TRAINING ONLY.")
    print(f"                       phase2_matrix.json (n_runs=15) stays "
          f"authoritative for SBS/VBS.")
    print(BAR)

    state = load_state(functions, seeds, dims, algorithms, n_runs)
    extractor = LandscapeFeatureExtractor(n_samples=FEATURE_SAMPLES)

    i = 0
    t_start = time.perf_counter()

    for fn in functions:
        for seed in seeds:
            for dim in dims:
                rk = row_key(fn, seed, dim)
                problem = getattr(aad, fn)(dim=dim, seed=seed)

                # Landscape features: re-extracted per (function, seed, dim).
                # Never reused across seeds or dims.
                if rk not in state["features"]:
                    np.random.seed(FEATURE_SEED)
                    feats = {k: float(v) for k, v in
                             extractor.extract_features(problem).items()}
                    state["features"][rk] = feats
                    append_log("feature", rk, feats)

                for alg in algorithms:
                    i += 1
                    ck = cell_key(fn, seed, dim, alg)
                    if ck in state["cells"]:
                        continue

                    cell = run_cell(alg, problem, fn, seed, dim, n_runs, run_seeds)
                    state["cells"][ck] = cell
                    state["build_seconds"] = round(
                        state["build_seconds"] + cell["wall_seconds"], 3)
                    append_log("cell", ck, cell)

                    med = cell["median"]
                    ms = f"{med:.4e}" if med is not None else "NO FINITE RUN"
                    flag = f"  [{cell['n_nonfinite']} non-finite]" if cell["n_nonfinite"] else ""
                    done = len(state["cells"])
                    print(f"  [{done:>4}/{total}] {fn:<34} s{seed} d{dim:<2} "
                          f"x {alg:<8} -> {ms:>13}  {cell['wall_seconds']:>6.2f}s{flag}")

                save_snapshot(state)      # snapshot after each completed row

    save_snapshot(state)
    wall = time.perf_counter() - t_start

    print(BAR)
    print(f"  rows with features : {len(state['features'])} / {n_rows}")
    print(f"  cells in file      : {len(state['cells'])} / {total}")
    missing = [cell_key(f, s, d, a) for f in functions for s in seeds
               for d in dims for a in algorithms if cell_key(f, s, d, a) not in state["cells"]]
    print(f"  cells MISSING      : {len(missing)} {missing[:5]}")
    nonfin = sum(c["n_nonfinite"] for c in state["cells"].values())
    errs = [k for k, c in state["cells"].items() if c["errors"]]
    print(f"  total NaN/Inf runs : {nonfin}")
    print(f"  cells with errors  : {len(errs)} {errs[:5]}")
    print(f"  this session       : {wall:.1f}s = {wall/3600:.2f}h")
    print(f"  cumulative build   : {state['build_seconds']:.1f}s "
          f"= {state['build_seconds']/3600:.2f}h")
    print(f"  distinct (function, pseudo-instance, dim) combos: "
          f"{len({(c['function'], c['pseudo_instance_seed'], c['dim']) for c in state['cells'].values()})}")
    print(BAR)
    print()
    verify_features(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
