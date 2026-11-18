"""
phase1b_algorithm_screening.py
------------------------------
Cheap health screen of all registered mealpy algorithms before Phase 2.

Purpose is NOT to rank algorithms. It is to catch, early:
  - anything that crashes outright
  - anything that lands wildly off on the easiest possible problem

Setup (deliberately identical to phase1_baseline_comparison.py so numbers
are comparable):
  problem : BBOB_F1_Sphere(dim=10, seed=42)   -- trivial, unimodal
  config  : population_size=50, max_fe=5000  (EQUAL evaluation budget)
  runs    : 3 seeded runs per algorithm

Budget note: algorithms are capped by function evaluations, not generations.
Population methods spend pop_size evaluations per generation while
single-solution methods (SA) spend about one, so equal generations is not an
equal budget -- SA previously received 150 evaluations against everyone
else's 5050.

The built-in "ea" algorithm is NOT screened here: EvolutionaryAlgorithm does
not read a "seed" config key, so its runs cannot be made reproducible through
this path. Only the registered mealpy algorithms are covered.

CMA_ES, AOA and DE are no longer registered: they run without error but do
not optimise. See the exclusion note in aad/mealpy_algorithms.py.

Writes results to results/phase1b/phase1b_screening.{txt,json}.

Run:
    .venv\\Scripts\\python.exe phase1b_algorithm_screening.py
"""

import json
import os
import sys
import time
import traceback

import numpy as np

from aad.evaluation import EvaluationFramework
from aad.mealpy_algorithms import MEALPY_ALGORITHM_KEYS
from aad.problems import BBOB_F1_Sphere

PROBLEM_DIM = 10
PROBLEM_SEED = 42
POPULATION_SIZE = 50
MAX_FE = 5000
SEEDS = [1000, 1001, 1002]

OUT_DIR = os.path.join("results", "phase1b")
OUT_STEM = os.path.join(OUT_DIR, "phase1b_screening")

BAR = "=" * 78

# Sphere is trivial; these thresholds flag "did it refine at all", not quality.
T_EXCELLENT = 1e-6
T_MODERATE = 1e-2


def classify(median_fitness: float) -> str:
    if median_fitness < T_EXCELLENT:
        return "excellent"
    if median_fitness < T_MODERATE:
        return "moderate"
    return "POOR"


def screen_one(algorithm: str, problem) -> dict:
    """Run *algorithm* once per seed. Never raises -- records failures."""
    evaluator = EvaluationFramework(n_runs=1)
    fitnesses, runtimes = [], []

    for seed in SEEDS:
        config = {
            "algorithm": algorithm,
            "population_size": POPULATION_SIZE,
            "max_fe": MAX_FE,
            "seed": seed,
        }
        try:
            t0 = time.perf_counter()
            metrics = evaluator.evaluate(config, problem, verbose=False)
            runtimes.append(time.perf_counter() - t0)
            fitnesses.append(float(metrics["best_fitness"]))
        except Exception as exc:                      # noqa: BLE001 - screening
            return {
                "algorithm": algorithm,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().strip().splitlines()[-1],
                "median": None, "min": None, "max": None, "mean_runtime": None,
                "fitnesses": fitnesses,
                "verdict": "FAILED",
            }

    arr = np.asarray(fitnesses, dtype=float)
    finite = bool(np.all(np.isfinite(arr)))
    median = float(np.median(arr))

    return {
        "algorithm": algorithm,
        "status": "ok" if finite else "NON-FINITE",
        "error": None if finite else "produced NaN or Inf",
        "traceback": None,
        "median": median,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_runtime": float(np.mean(runtimes)),
        "fitnesses": [float(f) for f in arr],
        "verdict": classify(median) if finite else "FAILED",
    }


def build_report(rows: list) -> str:
    lines = []
    a = lines.append

    a(BAR)
    a("  PHASE 1b - ALGORITHM HEALTH SCREEN")
    a(BAR)
    a(f"  problem          : BBOB_F1_Sphere(dim={PROBLEM_DIM}, seed={PROBLEM_SEED})")
    a(f"  population_size  : {POPULATION_SIZE}")
    a(f"  max_fe           : {MAX_FE}   (equal evaluation budget)")
    a(f"  seeds per algo   : {SEEDS}")
    a(f"  algorithms       : {len(rows)} (mealpy only; 'ea' excluded - not seedable)")
    a("")
    a(f"  thresholds on median fitness (Sphere optimum = 0.0):")
    a(f"    excellent < {T_EXCELLENT:g}   moderate < {T_MODERATE:g}   POOR >= {T_MODERATE:g}")
    a("")

    header = (f"  {'algorithm':<10}{'verdict':<12}{'median':>14}"
              f"{'min':>14}{'max':>14}{'sec/run':>10}")
    a(header)
    a("  " + "-" * (len(header) - 2))

    for r in sorted(rows, key=lambda x: (x["median"] is None, x["median"])):
        if r["median"] is None:
            a(f"  {r['algorithm']:<10}{r['verdict']:<12}"
              f"{'-':>14}{'-':>14}{'-':>14}{'-':>10}")
        else:
            a(f"  {r['algorithm']:<10}{r['verdict']:<12}"
              f"{r['median']:>14.4e}{r['min']:>14.4e}{r['max']:>14.4e}"
              f"{r['mean_runtime']:>10.2f}")

    failed = [r for r in rows if r["verdict"] == "FAILED"]
    poor = [r for r in rows if r["verdict"] == "POOR"]
    moderate = [r for r in rows if r["verdict"] == "moderate"]
    excellent = [r for r in rows if r["verdict"] == "excellent"]

    a("")
    a(BAR)
    a("  SUMMARY")
    a(BAR)
    a(f"  excellent (< {T_EXCELLENT:g}) : {len(excellent):>2}  {[r['algorithm'] for r in excellent]}")
    a(f"  moderate  (< {T_MODERATE:g}) : {len(moderate):>2}  {[r['algorithm'] for r in moderate]}")
    a(f"  POOR      (>= {T_MODERATE:g}): {len(poor):>2}  {[r['algorithm'] for r in poor]}")
    a(f"  FAILED              : {len(failed):>2}  {[r['algorithm'] for r in failed]}")

    if failed:
        a("")
        a("  Failure details:")
        for r in failed:
            a(f"    {r['algorithm']}: {r['error']}")

    a("")
    a("  NOTE: 'POOR' does not mean broken. On a trivial unimodal problem it")
    a("  means the algorithm's default settings do not refine finely. That is")
    a("  a real property of the algorithm as configured, not necessarily a bug.")
    a(BAR)

    return "\n".join(lines)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    problem = BBOB_F1_Sphere(dim=PROBLEM_DIM, seed=PROBLEM_SEED)

    print(f"Screening {len(MEALPY_ALGORITHM_KEYS)} algorithms "
          f"x {len(SEEDS)} seeds ...\n")

    rows = []
    for i, algorithm in enumerate(MEALPY_ALGORITHM_KEYS, 1):
        print(f"  [{i:>2}/{len(MEALPY_ALGORITHM_KEYS)}] {algorithm:<10} ", end="", flush=True)
        row = screen_one(algorithm, problem)
        rows.append(row)
        if row["median"] is None:
            print(f"{row['verdict']}  ({row['error']})")
        else:
            print(f"{row['verdict']:<10} median={row['median']:.4e}")

    report = build_report(rows)
    print("\n" + report)

    with open(f"{OUT_STEM}.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    payload = {
        "problem": f"BBOB_F1_Sphere(dim={PROBLEM_DIM}, seed={PROBLEM_SEED})",
        "population_size": POPULATION_SIZE,
        "max_fe": MAX_FE,
        "seeds": SEEDS,
        "thresholds": {"excellent": T_EXCELLENT, "moderate": T_MODERATE},
        "results": rows,
    }
    with open(f"{OUT_STEM}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSaved:\n  {OUT_STEM}.txt\n  {OUT_STEM}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
