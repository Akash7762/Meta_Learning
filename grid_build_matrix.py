"""
grid_build_matrix.py
--------------------
Configuration-grid dataset: every algorithm under every setting of its
original configuration grid, on many problems, with many runs.

GRID -- read from aad/config_space.py (EA_CONFIG_SPACE), the single source of
truth:
    population_size  [20, 50, 100]
    max_generations  [50, 100, 200]
    selection        [tournament, roulette]   "ea" only
    crossover        [sbx, uniform]           "ea" only
    mutation         [gaussian, polynomial]   "ea" only
  "ea"              : all five keys            -> 72 settings
  21 mealpy methods : population x generations -> 9 settings each. They ignore
                      the three operator keys (aad/config_space.py), so
                      varying those would only repeat identical runs.

BUDGET IS NOT EQUAL, BY DESIGN (user decision, 2026-10-09).
    Every run uses the algorithms' original working model: mealpy gets
    epoch=max_generations and pop_size=population with no max_fe; "ea" gets
    max_generations. The number of objective evaluations therefore differs
    between settings and between algorithms (mealpy SA, for example, spends
    about one evaluation per epoch). Each run's real evaluation count is
    counted and stored in "evaluations". This dataset does not follow the
    equal-budget rule of Phases 2 to 4; compare settings with that in mind.

PROBLEMS -- 24 BBOB functions x DIMENSIONS x PSEUDO_INSTANCE_SEEDS. The seeds
    are pseudo-instances, not COCO instances (see phase2c_expand_matrix.py).
RUNS -- N_RUNS per cell, run seeds 1000 upward: mealpy through the config
    "seed" (which also reseeds NumPy's global RNG), "ea" through
    np.random.seed, exactly as in Phases 2 and 2c.
FEATURES -- per row, LandscapeFeatureExtractor with FEATURE_SAMPLES samples
    after np.random.seed(FEATURE_SEED), as in Phase 2c.

PARALLEL AND SPLITTABLE
    --workers N runs N cells at once. --shard K/N builds only the rows whose
    index % N == K-1, so the build can be split over N computers: run
    --shard 1/N ... --shard N/N, copy every results/config_grid/shard_*.jsonl
    into one results/config_grid folder, then run --merge.

DURABILITY
    Every finished cell and feature row is appended to the shard's journal,
    results/config_grid/shard_KofN.jsonl, and flushed at once. Running the
    same command again resumes; a journal started with other settings is
    refused, never mixed.

The output is large (one cell per problem and setting, each with N_RUNS
values), so results/config_grid/ is kept out of git; back it up by copying.

Run, in this order:
    python grid_build_matrix.py --dry-run          # tiny build; prints timing
    python grid_build_matrix.py --workers 8        # full build (or --shard K/N)
    python grid_build_matrix.py --merge            # one file from all shards
    python grid_build_matrix.py --verify           # completeness and checks
"""

import argparse
import glob
import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import aad
from aad.config_space import EA_CONFIG_SPACE
from aad.evolutionary_algorithm import build_algorithm
from aad.landscape_features import LandscapeFeatureExtractor
from aad.mealpy_algorithms import MEALPY_REGISTRY

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

DIMENSIONS = [2, 5, 10, 20]
PSEUDO_INSTANCE_SEEDS = list(range(42, 52))          # 42 to 51
N_RUNS = 30
RUN_SEEDS = [1000 + i for i in range(N_RUNS)]
FEATURE_SAMPLES = 300
FEATURE_SEED = 20260915

OUT_DIR = os.path.join("results", "config_grid")
MERGED = os.path.join(OUT_DIR, "config_grid_matrix.json")

FUNCTIONS = [n for n in aad.__all__ if n.startswith("BBOB_")]
ALGORITHMS = sorted(MEALPY_REGISTRY) + ["ea"]
OPERATOR_KEYS = ("selection", "crossover", "mutation")
ROW_KEY_FORMAT = "{function}|s{seed}|d{dim}"
CELL_KEY_FORMAT = "{function}|s{seed}|d{dim}|{algorithm}|p{population}|g{generations}|{operators}"
BAR = "=" * 86


def settings_for(algorithm):
    """Every setting of one algorithm's grid, as config fragments."""
    pops = EA_CONFIG_SPACE["population_size"]
    gens = EA_CONFIG_SPACE["max_generations"]
    if algorithm == "ea":
        ops = list(itertools.product(*(EA_CONFIG_SPACE[k] for k in OPERATOR_KEYS)))
        return [{"population_size": p, "max_generations": g, **dict(zip(OPERATOR_KEYS, o))}
                for p in pops for g in gens for o in ops]
    return [{"population_size": p, "max_generations": g} for p in pops for g in gens]


def row_key(fn, seed, dim):
    return ROW_KEY_FORMAT.format(function=fn, seed=seed, dim=dim)


def cell_key(fn, seed, dim, algorithm, setting):
    ops = ("-".join(setting[k] for k in OPERATOR_KEYS)
           if OPERATOR_KEYS[0] in setting else "-")
    return CELL_KEY_FORMAT.format(function=fn, seed=seed, dim=dim, algorithm=algorithm,
                                  population=setting["population_size"],
                                  generations=setting["max_generations"], operators=ops)


def env_versions():
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("scipy", "mealpy"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:                                   # noqa: BLE001
            out[mod] = "unavailable"
    return out


# ---------------------------------------------------------------------------
# One cell (runs in a worker process)
# ---------------------------------------------------------------------------

def run_cell(task):
    """All runs of one (problem, algorithm, setting) cell. Raw best values are
    never sanitised: a failed or non-finite run is stored as null."""
    fn, seed, dim, algorithm, setting, run_seeds = task
    t0 = time.perf_counter()
    raw, evals, errors = [], [], []
    try:
        problem = getattr(aad, fn)(dim=dim, seed=seed)
    except Exception as exc:                                # noqa: BLE001
        problem = None
        errors.append(f"problem: {type(exc).__name__}: {exc}")

    for s in run_seeds:
        if problem is None:
            raw.append(None)
            evals.append(None)
            continue
        count = [0]
        original = problem.evaluate

        def counted(x, _evaluate=original):
            count[0] += 1
            return _evaluate(x)

        problem.evaluate = counted                  # counts every objective call
        try:
            config = {"algorithm": algorithm, **setting}
            if algorithm == "ea":
                np.random.seed(s)                   # "ea" draws from NumPy's global RNG
            else:
                config["seed"] = s
            _, best = build_algorithm(config).run(problem)
            raw.append(float(best))
        except Exception as exc:                            # noqa: BLE001
            raw.append(None)
            errors.append(f"seed={s}: {type(exc).__name__}: {exc}")
        finally:
            del problem.evaluate                    # back to the class method
        evals.append(count[0])

    finite = [v for v in raw if v is not None and np.isfinite(v)]
    arr = np.asarray(finite, dtype=float)
    stats = ({"mean": float(np.mean(arr)), "median": float(np.median(arr)),
              "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
              "min": float(np.min(arr)), "max": float(np.max(arr))}
             if len(arr) else {k: None for k in ("mean", "median", "std", "min", "max")})
    return cell_key(fn, seed, dim, algorithm, setting), {
        "algorithm": algorithm, "function": fn, "pseudo_instance_seed": seed, "dim": dim,
        **setting,
        "fitnesses": [v if (v is not None and np.isfinite(v)) else None for v in raw],
        "evaluations": evals,
        "run_seeds": list(run_seeds),
        **stats,
        "n_runs_requested": len(run_seeds),
        "n_runs_finite": len(finite),
        "n_nonfinite": len(raw) - len(finite),
        "errors": errors,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def read_journal(path):
    """(header, features, cells) from a journal; a torn last line is skipped."""
    header, features, cells = None, {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                            # torn final line after a hard kill
            kind = rec.get("_kind")
            if kind == "header":
                header = rec["value"]
            elif kind == "feature":
                features.setdefault(rec["key"], rec["value"])
            elif kind == "cell":
                cells.setdefault(rec["key"], rec["value"])
    return header, features, cells


def comparable(header):
    """The settings that must match to resume or merge (not env, not shard)."""
    return {k: v for k, v in header.items() if k not in ("env", "shard", "started")}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def parse_shard(text):
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        sys.exit(f"--shard must look like K/N, got {text!r}")
    if not 1 <= k <= n:
        sys.exit(f"--shard {text}: need 1 <= K <= N")
    return k, n


def build(args):
    k, n = parse_shard(args.shard)
    functions, seeds, dims = FUNCTIONS, PSEUDO_INSTANCE_SEEDS, DIMENSIONS
    algorithms, run_seeds, out_dir = ALGORITHMS, RUN_SEEDS, OUT_DIR
    settings = {a: settings_for(a) for a in algorithms}
    if args.dry_run:
        functions, seeds, dims = FUNCTIONS[:2], PSEUDO_INSTANCE_SEEDS[:1], [2, 5]
        algorithms, run_seeds = ["PSO", "SA", "ea"], RUN_SEEDS[:2]
        settings = {a: settings_for(a) for a in algorithms}
        settings["ea"] = settings["ea"][::18]       # 4 of the 72, spread over the grid
        out_dir = os.path.join(OUT_DIR, "dryrun")
    os.makedirs(out_dir, exist_ok=True)

    rows = [(fn, s, d) for fn in functions for s in seeds for d in dims]
    mine = [r for i, r in enumerate(rows) if i % n == k - 1]
    per_row = sum(len(settings[a]) for a in algorithms)
    header = {"functions": functions, "pseudo_instance_seeds": seeds, "dimensions": dims,
              "algorithms": algorithms, "grid": EA_CONFIG_SPACE,
              "settings_per_algorithm": {a: len(settings[a]) for a in algorithms},
              "n_runs": len(run_seeds), "run_seeds": run_seeds,
              "budget": "not equal: population x max_generations, original working "
                        "model (mealpy epoch=max_generations, no max_fe); real "
                        "evaluation counts in 'evaluations'",
              "feature_samples": FEATURE_SAMPLES, "feature_seed": FEATURE_SEED,
              "row_key_format": ROW_KEY_FORMAT, "cell_key_format": CELL_KEY_FORMAT,
              "dry_run": bool(args.dry_run), "shard": f"{k}/{n}", "env": env_versions(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    journal = os.path.join(out_dir, f"shard_{k}of{n}.jsonl")
    features, done = {}, {}
    if os.path.exists(journal):
        old, features, done = read_journal(journal)
        if old is None or comparable(old) != comparable(header) or old["shard"] != header["shard"]:
            sys.exit(f"REFUSING TO RESUME: {journal} was started with other settings. "
                     "Move it away to start afresh; nothing is mixed.")
        print(f"  resuming: {len(done)} cells and {len(features)} feature rows already done")
    else:
        with open(journal, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_kind": "header", "value": header}) + "\n")

    total_cells = len(mine) * per_row
    print(BAR)
    print("  CONFIGURATION-GRID DATASET" + ("  [DRY RUN]" if args.dry_run else ""))
    print(BAR)
    print(f"  problems (rows)    : {len(rows)} in all; this shard {k}/{n} builds {len(mine)}")
    print(f"  settings per row   : {per_row}  "
          f"({', '.join(f'{a}={len(settings[a])}' for a in algorithms[-2:])}, ...)")
    print(f"  runs per cell      : {len(run_seeds)}  (seeds {run_seeds[0]}..{run_seeds[-1]})")
    print(f"  cells / runs       : {total_cells} / {total_cells * len(run_seeds)} in this shard")
    print("  budget             : NOT equal; population x generations (see docstring)")
    print(f"  workers            : {args.workers}")
    print(f"  journal            : {journal}")
    print(BAR)

    extractor = LandscapeFeatureExtractor(n_samples=FEATURE_SAMPLES)
    t0, finished, cpu_seconds, runs_done = time.perf_counter(), 0, 0.0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (fn, seed, dim) in enumerate(mine, 1):
            rk = row_key(fn, seed, dim)
            new_lines = []
            if rk not in features:
                try:
                    np.random.seed(FEATURE_SEED)
                    feats = {key: float(v) for key, v in extractor.extract_features(
                        getattr(aad, fn)(dim=dim, seed=seed)).items()}
                except Exception as exc:                    # noqa: BLE001
                    feats = {"error": f"{type(exc).__name__}: {exc}"}
                features[rk] = feats
                new_lines.append({"_kind": "feature", "key": rk, "value": feats})

            tasks = [(fn, seed, dim, a, s, run_seeds) for a in algorithms for s in settings[a]
                     if cell_key(fn, seed, dim, a, s) not in done]
            with open(journal, "a", encoding="utf-8") as fh:
                for line in new_lines:
                    fh.write(json.dumps(line) + "\n")
                for key, cell in ex.map(run_cell, tasks, chunksize=1):
                    done[key] = cell
                    fh.write(json.dumps({"_kind": "cell", "key": key, "value": cell}) + "\n")
                    fh.flush()
                    finished += 1
                    cpu_seconds += cell["wall_seconds"]
                    runs_done += cell["n_runs_requested"]
                os.fsync(fh.fileno())

            elapsed = time.perf_counter() - t0
            remaining = total_cells - len(done)
            eta = elapsed / finished * remaining if finished else float("nan")
            print(f"  row {i:>4}/{len(mine)}  {rk:<38} cells done {len(done):>7}/{total_cells}"
                  f"  elapsed {elapsed / 3600:6.2f} h  eta {eta / 3600:7.1f} h", flush=True)

    print(f"\n  shard {k}/{n} complete: {len(done)} cells in {journal}")
    if runs_done:
        per_run = cpu_seconds / runs_done
        full_runs = (len(FUNCTIONS) * len(PSEUDO_INSTANCE_SEEDS) * len(DIMENSIONS)
                     * sum(len(settings_for(a)) for a in ALGORITHMS) * N_RUNS)
        print(f"  measured: {per_run:.3f} s of worker time per run")
        if args.dry_run:
            print(f"  rough projection for the full build ({full_runs} runs): "
                  f"{full_runs * per_run / args.workers / 86400:.1f} days with "
                  f"{args.workers} workers (the dry-run settings are not a fair sample "
                  "of the whole grid)")


# ---------------------------------------------------------------------------
# Merge and verify
# ---------------------------------------------------------------------------

def merge(args):
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "shard_*of*.jsonl")))
    if not paths:
        sys.exit(f"No shard journals in {OUT_DIR}.")
    base, shards, envs, features, cells = None, [], {}, {}, {}
    for p in paths:
        header, f, c = read_journal(p)
        if header is None:
            sys.exit(f"{p} has no header line.")
        if base is None:
            base = header
        elif comparable(header) != comparable(base):
            sys.exit(f"{p} was built with other settings than {paths[0]} -- not merging.")
        shards.append(header["shard"])
        envs[header["shard"]] = header["env"]
        features.update(f)
        for key, cell in c.items():
            if key in cells and cells[key] != cell:
                sys.exit(f"Cell {key} appears in two shards with different values.")
            cells[key] = cell
    n_values = {s.split("/")[1] for s in shards}
    if len(n_values) != 1 or len(set(shards)) != int(n_values.pop()):
        print(f"  WARNING: shards found {sorted(shards)} -- not a complete K/N set.")

    expected = [cell_key(fn, s, d, a, st)
                for fn in base["functions"] for s in base["pseudo_instance_seeds"]
                for d in base["dimensions"] for a in base["algorithms"]
                for st in settings_for(a)]
    missing = [key for key in expected if key not in cells]
    meta = {**comparable(base), "shards": sorted(shards), "env_per_shard": envs,
            "merged": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cells_expected": len(expected), "cells_present": len(cells),
            "cells_missing": len(missing)}
    tmp = MERGED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "features": features, "cells": cells}, fh)
    os.replace(tmp, MERGED)
    if len({json.dumps(e, sort_keys=True) for e in envs.values()}) > 1:
        print("  NOTE: shards were built with different library versions; see "
              "meta.env_per_shard.")
    print(f"  merged {len(paths)} shard(s): {len(cells)} of {len(expected)} cells, "
          f"{len(missing)} missing -> {MERGED}")


def verify(args):
    if not os.path.exists(MERGED):
        sys.exit(f"{MERGED} not found; run --merge first.")
    with open(MERGED, encoding="utf-8") as fh:
        data = json.load(fh)
    meta, cells, feats = data["meta"], data["cells"], data["features"]
    print(BAR)
    print("  VERIFY " + MERGED)
    print(BAR)
    print(f"  cells              : {meta['cells_present']} of {meta['cells_expected']} "
          f"({meta['cells_missing']} missing)")
    bad = [k for k, c in cells.items() if c["n_nonfinite"]]
    errs = [k for k, c in cells.items() if c["errors"]]
    print(f"  cells with a failed or non-finite run : {len(bad)}")
    print(f"  cells with errors  : {len(errs)}" + (f"  e.g. {errs[0]}: "
                                                    f"{cells[errs[0]]['errors'][0][:100]}"
                                                    if errs else ""))
    feat_err = [r for r, f in feats.items() if "error" in f]
    print(f"  feature rows       : {len(feats)} ({len(feat_err)} with errors)")
    good = {r: f for r, f in feats.items() if "error" not in f}
    if good:
        keys = sorted(next(iter(good.values())))
        X = np.array([[good[r][k] for k in keys] for r in sorted(good)], float)
        dead = [k for j, k in enumerate(keys) if np.std(X[:, j]) <= 1e-12]
        print(f"  constant feature columns: {dead or 'none'}")
    print("\n  real evaluations per run (min / median / max) by algorithm:")
    by_alg = {}
    for c in cells.values():
        by_alg.setdefault(c["algorithm"], []).extend(e for e in c["evaluations"] if e is not None)
    for a in sorted(by_alg):
        e = np.array(by_alg[a])
        if len(e):
            print(f"    {a:<8} {int(e.min()):>7} / {int(np.median(e)):>7} / {int(e.max()):>7}")
    print(BAR)


def main():
    ap = argparse.ArgumentParser(description="Build the configuration-grid dataset.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="2 functions x 1 seed x dims 2, 5 x PSO, SA, ea x 2 runs")
    mode.add_argument("--merge", action="store_true", help="combine all shard journals")
    mode.add_argument("--verify", action="store_true", help="check the merged file")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--shard", default="1/1", help="build part K of N, e.g. 2/3")
    args = ap.parse_args()
    if args.merge:
        merge(args)
    elif args.verify:
        verify(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
