"""
phase2_build_matrix.py
----------------------
Build the cross-problem performance matrix: every (problem, algorithm) pair,
cached to disk so all later analysis is offline and never re-runs an optimiser.

Design decisions, each forced by something verified in the source:

BUDGET -- max_fe=5000, NOT max_generations=100.
    Equal generations is not an equal budget. Population methods spend
    pop_size evaluations per generation; single-solution methods (SA) spend
    about one. Under max_generations=100 SA receives ~150 evaluations against
    everyone else's ~5050 (Phase 1b), which would penalise it in all 24 rows
    for a reason unrelated to its merit. MealpyAlgorithm.max_fe applies a
    mealpy Termination(max_fe=...) and calibrates the epoch count so
    schedule-driven algorithms (GWO, WOA, SCA, MPA) still anneal correctly.

    The built-in "ea" does not read max_fe. It is given max_generations=100,
    which measures at 5050 evaluations -- 1% more than the mealpy cells. This
    is recorded per cell in "budget_note" rather than hidden.

RAW RUNS, NOT EvaluationFramework.
    EvaluationFramework.evaluate() passes every result through
    _sanitize_fitness (aad/evaluation.py:79), which silently rewrites NaN/Inf
    to 1e12 before storing. That would make the required NaN/Inf count
    permanently zero and substitute a fabricated value. build_algorithm() is
    called directly so non-finite results are detected and recorded as such,
    never replaced.

SEEDING.
    mealpy algorithms are seeded through the config dict ("seed" key), which
    MealpyAlgorithm forwards to solve(seed=...) and which also reseeds NumPy's
    global RNG to close the scipy cauchy.rvs leak in JADE/SHADE/L_SHADE.
    "ea" draws entirely from NumPy's global RNG, so np.random.seed() is set
    before its runs -- verified reproducible.

    The same 15 seeds are used for every cell (common random numbers), so
    algorithms are compared on identical problem instances.

Run:
    .venv\\Scripts\\python.exe phase2_build_matrix.py --dry-run
    .venv\\Scripts\\python.exe phase2_build_matrix.py
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

# CHANGED 2026-09-22 from a single dim to a list. The selector is trained
# across dims 5/10/20 (phase2c), so evaluating only at dim 10 would leave the
# dimension axis trained but never tested. Rows are now keyed by dim, exactly
# as in phase2c_expand_matrix.py, and the two matrices share a layout.
PROBLEM_DIMS = [5, 10, 20]
# CHANGED 2026-09-20 from 42 to 100. Seed 42 is inside
# phase2c_expand_matrix.PSEUDO_INSTANCE_SEEDS, which made every Phase 2 row a
# re-measurement of a Phase 2c training row (bit-identical feature vectors),
# so phase2_matrix.json was NOT an independent evaluation set. 100 is outside
# the training seed list, which is what makes a train-on-2c / test-on-2 split
# a real held-out test. Keep this OUTSIDE PSEUDO_INSTANCE_SEEDS forever;
# phase3b_holdout_confirm.py asserts it and will refuse to run otherwise.
PROBLEM_SEED = 100
POPULATION_SIZE = 50
MAX_FE = 5000              # mealpy cells
EA_MAX_GENERATIONS = 100   # "ea" cells (no max_fe support) -> ~5050 evals
N_RUNS = 15
RUN_SEEDS = [1000 + i for i in range(N_RUNS)]

# "ea" operator defaults, taken from EvolutionaryAlgorithm.__init__ defaults
EA_OPERATORS = {"selection": "tournament", "crossover": "sbx",
                "mutation": "gaussian"}

FEATURE_SAMPLES = 300      # LandscapeFeatureExtractor default
FEATURE_SEED = 20260915    # seeds the feature sampling, which is stochastic

OUT_DIR = os.path.join("results", "phase2")
OUT_PATH = os.path.join(OUT_DIR, "phase2_matrix.json")

PROBLEM_NAMES = [n for n in aad.__all__ if n.startswith("BBOB_")]
ALGORITHMS = sorted(MEALPY_REGISTRY.keys()) + ["ea"]

# Shared with phase2c_expand_matrix.py so both matrices have one layout and
# phase2_lopo.py / phase3b_holdout_confirm.py read them the same way.
CELL_KEY_FORMAT = "{function}|s{seed}|d{dim}|{algorithm}"
ROW_KEY_FORMAT = "{function}|s{seed}|d{dim}"


def row_key(fn, seed, dim):
    return ROW_KEY_FORMAT.format(function=fn, seed=seed, dim=dim)


def cell_key(fn, seed, dim, alg):
    return CELL_KEY_FORMAT.format(function=fn, seed=seed, dim=dim, algorithm=alg)

BAR = "=" * 78


# ---------------------------------------------------------------------------
# Persistence
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


def load_state(problems, dims, algorithms, n_runs) -> dict:
    """Load the cache, or start a fresh one. Never silently discards data."""
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
        m = state.get("meta", {})

        # A pre-2026-09-22 matrix keys cells "problem|algorithm" with a single
        # global problem_dim. Its keys can never match the new
        # "function|sSEED|dDIM|algorithm" form, so resuming would recompute
        # everything AND keep the old cells alongside -- one file holding two
        # incompatible layouts, with the stale rows indistinguishable.
        if "problems" in m or "problem_dim" in m:
            sys.exit(
                f"REFUSING TO RESUME: {OUT_PATH} uses the old single-dimension\n"
                f"  layout (meta.problems / meta.problem_dim). This build emits\n"
                f"  per-dimension rows keyed {ROW_KEY_FORMAT!r}.\n"
                f"  Delete or rename {OUT_PATH} and rebuild from scratch.")

        if m.get("problem_seed") != PROBLEM_SEED:
            sys.exit(f"REFUSING TO RESUME: {OUT_PATH} has problem_seed="
                     f"{m.get('problem_seed')}, this run uses {PROBLEM_SEED}.")
        if sorted(m.get("dimensions", [])) != sorted(dims):
            sys.exit(f"REFUSING TO RESUME: {OUT_PATH} has dimensions="
                     f"{m.get('dimensions')}, this run uses {PROBLEM_DIMS}.\n"
                     f"  Adding a dimension is safe -- but do it by editing\n"
                     f"  PROBLEM_DIMS and deleting the file, so meta stays true.")
        if m.get("n_runs") != n_runs:
            sys.exit(f"REFUSING TO RESUME: {OUT_PATH} has n_runs="
                     f"{m.get('n_runs')}, this run uses {n_runs}. Resuming "
                     f"would mix medians of different precision.")

        print(f"  resuming: {len(state.get('cells', {}))} cells already cached")
        return state
    return {
        "meta": {
            # 'problems' is deliberately NOT emitted. phase2_lopo.load()
            # switches on its presence: with it, every row is its own fold,
            # which for a multi-dim matrix would hold out F1|d10 while
            # F1|d5 and F1|d20 stayed in training -- the same function, so
            # the answer leaks. Emitting 'functions' instead selects the
            # grouped-fold branch, where all dims of a function are held out
            # together.
            # What this build ACTUALLY covers, not the module
            # constants -- a --dry-run subset must not claim the
            # full grid, or downstream analysis derives rows that
            # were never computed.
            "functions": list(problems),
            "dimensions": list(dims),
            # Single evaluation seed, kept scalar so
            # phase3b_holdout_confirm.assert_no_leakage() can compare it
            # against phase2c's pseudo_instance_seeds.
            "problem_seed": PROBLEM_SEED,
            "population_size": POPULATION_SIZE,
            "max_fe": MAX_FE,
            "ea_max_generations": EA_MAX_GENERATIONS,
            "ea_operators": EA_OPERATORS,
            "n_runs": n_runs,
            "run_seeds": RUN_SEEDS[:n_runs],
            "feature_samples": FEATURE_SAMPLES,
            "feature_seed": FEATURE_SEED,
            "algorithms": list(algorithms),
            "cell_key_format": CELL_KEY_FORMAT,
            "row_key_format": ROW_KEY_FORMAT,
            # Seeds alone do not make a build reproducible: a different
            # numpy/scipy/mealpy can give different numbers from the same
            # seed. Record what actually produced these cells.
            "env": _env_versions(),
        },
        "features": {},
        "cells": {},
        "build_seconds": 0.0,
    }


def save_state(state: dict) -> None:
    """Atomic-ish write so an interrupt cannot leave a truncated JSON file."""
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, OUT_PATH)


# ---------------------------------------------------------------------------
# One run / one cell
# ---------------------------------------------------------------------------

def make_config(algorithm: str, seed: int) -> dict:
    """Build the config dict for *algorithm*. Only keys the classes read."""
    if algorithm == "ea":
        return {
            "algorithm": "ea",
            "population_size": POPULATION_SIZE,
            "max_generations": EA_MAX_GENERATIONS,
            **EA_OPERATORS,
        }
    return {
        "algorithm": algorithm,
        "population_size": POPULATION_SIZE,
        "max_fe": MAX_FE,
        "seed": seed,
    }


def run_once(algorithm: str, problem, seed: int):
    """
    One seeded run. Returns the RAW best fitness -- no sanitising, so a
    non-finite result stays visible instead of becoming 1e12.
    """
    if algorithm == "ea":
        np.random.seed(seed)      # "ea" draws from NumPy's global RNG
    _, best = build_algorithm(make_config(algorithm, seed)).run(problem)
    return float(best)


def run_cell(algorithm: str, problem, n_runs: int, seeds: list) -> dict:
    """Run one (problem, algorithm) cell. Records every raw value and seed."""
    t0 = time.perf_counter()
    raw, errors = [], []

    for seed in seeds[:n_runs]:
        try:
            raw.append(run_once(algorithm, problem, seed))
        except Exception as exc:                      # noqa: BLE001
            raw.append(None)
            errors.append(f"seed={seed}: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t0

    finite = [v for v in raw if v is not None and np.isfinite(v)]
    n_nonfinite = sum(1 for v in raw if v is None or not np.isfinite(v))
    arr = np.asarray(finite, dtype=float)

    stats = {}
    if len(arr):
        stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    else:
        # No usable value. Recorded as null, never imputed.
        stats = {k: None for k in ("mean", "median", "std", "min", "max")}

    return {
        "algorithm": algorithm,
        "problem": type(problem).__name__,
        # non-finite / failed runs are stored as null, not as a substitute value
        "fitnesses": [v if (v is not None and np.isfinite(v)) else None
                      for v in raw],
        "seeds": seeds[:n_runs],
        **stats,
        "n_runs_requested": n_runs,
        "n_runs_finite": len(finite),
        "n_nonfinite": n_nonfinite,
        "errors": errors,
        "wall_seconds": round(elapsed, 3),
        "budget_note": ("max_generations=100 (~5050 evals; +1% vs mealpy cells)"
                        if algorithm == "ea" else f"max_fe={MAX_FE}"),
    }


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="2 problems x 3 algorithms x 2 runs, to a separate file")
    args = ap.parse_args()

    global OUT_PATH
    problems = PROBLEM_NAMES
    algorithms = ALGORITHMS
    n_runs = N_RUNS

    if args.dry_run:
        OUT_PATH = os.path.join(OUT_DIR, "phase2_matrix_dryrun.json")
        problems = PROBLEM_NAMES[:2]
        algorithms = ["PSO", "SA", "ea"]
        n_runs = 2

    print(BAR)
    print("  PHASE 2 - PERFORMANCE MATRIX BUILD" + ("  [DRY RUN]" if args.dry_run else ""))
    print(BAR)
    print(f"  problems        : {len(problems)}")
    print(f"  algorithms      : {len(algorithms)}")
    print(f"  cells           : {len(problems) * len(algorithms)}")
    print(f"  runs per cell   : {n_runs}")
    print(f"  total runs      : {len(problems) * len(algorithms) * n_runs}")
    print(f"  budget          : population_size={POPULATION_SIZE}, max_fe={MAX_FE}")
    print(f"                    ('ea': max_generations={EA_MAX_GENERATIONS})")
    print(f"  seeds per cell  : {RUN_SEEDS[:n_runs]}")
    print(f"  output          : {OUT_PATH}")
    print(BAR)

    state = load_state(problems, PROBLEM_DIMS, algorithms, n_runs)
    extractor = LandscapeFeatureExtractor(n_samples=FEATURE_SAMPLES)

    total = len(problems) * len(PROBLEM_DIMS) * len(algorithms)
    i = 0
    t_start = time.perf_counter()

    for pname in problems:
      for dim in PROBLEM_DIMS:
        problem = getattr(aad, pname)(dim=dim, seed=PROBLEM_SEED)
        rk = row_key(pname, PROBLEM_SEED, dim)

        # Landscape features: once per (problem, dim), seeded -- sampling is
        # stochastic, and a different dim is a different landscape.
        if rk not in state["features"]:
            np.random.seed(FEATURE_SEED)
            feats = extractor.extract_features(problem)
            state["features"][rk] = {k: float(v) for k, v in feats.items()}
            save_state(state)

        for algorithm in algorithms:
            i += 1
            key = cell_key(pname, PROBLEM_SEED, dim, algorithm)

            if key in state["cells"]:
                c = state["cells"][key]
                med = c["median"]
                ms = f"{med:.4e}" if med is not None else "NO FINITE RUN"
                print(f"  [{i:>4}/{total}] {rk:<40} x {algorithm:<8} "
                      f"-> {ms:>14}  (cached)")
                continue

            cell = run_cell(algorithm, problem, n_runs, RUN_SEEDS)
            cell["function"] = pname
            cell["dim"] = dim
            cell["problem_seed"] = PROBLEM_SEED
            state["cells"][key] = cell
            state["build_seconds"] = round(
                state.get("build_seconds", 0.0) + cell["wall_seconds"], 3)
            save_state(state)          # write after EVERY cell

            med = cell["median"]
            ms = f"{med:.4e}" if med is not None else "NO FINITE RUN"
            flag = f"  [{cell['n_nonfinite']} non-finite]" if cell["n_nonfinite"] else ""
            print(f"  [{i:>4}/{total}] {rk:<40} x {algorithm:<8} "
                  f"-> {ms:>14}  {cell['wall_seconds']:>7.2f}s{flag}")

    wall = time.perf_counter() - t_start
    print(BAR)
    print(f"  rows in file     : {len(state['features'])} "
          f"({len(problems)} functions x {len(PROBLEM_DIMS)} dims)")
    print(f"  cells in file    : {len(state['cells'])}")
    print(f"  this session     : {wall:.1f}s")
    print(f"  cumulative build : {state['build_seconds']:.1f}s")
    print(f"  saved            : {OUT_PATH}")
    print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
