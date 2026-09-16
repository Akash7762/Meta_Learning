"""
phase1_baseline_comparison.py
-----------------------------
Phase 1 — minimal controlled baseline: 2 algorithms, 1 problem, real statistics.

Fixed experiment, deliberately the smallest defensible one:
  problem    : BBOB_F1_Sphere(dim=10)   (unimodal, separable, trivial)
  algorithms : "PSO" and "GA"           (both in MEALPY_REGISTRY)
  config     : population_size=50, max_generations=100
  runs       : 30 independent seeded runs per algorithm
  test       : scipy.stats.mannwhitneyu (independent samples, two-sided)

No meta-learning, no surrogate, no LLM designer.

Reproducibility
---------------
Every run is explicitly seeded and every seed is printed. mealpy seeds its own
np.random.Generator from the seed passed to solve(), so the global numpy RNG
does NOT control it -- the seed must travel through the config dict.

PSO and GA use DISJOINT seed sets so the two samples are genuinely independent,
which is what Mann-Whitney U assumes (it is not a paired test).

Run:
    .venv\\Scripts\\python.exe phase1_baseline_comparison.py
"""

import sys

import numpy as np
import scipy
from scipy.stats import mannwhitneyu

from aad.evaluation import EvaluationFramework
from aad.problems import BBOB_F1_Sphere

# ---------------------------------------------------------------------------
# Fixed experiment parameters
# ---------------------------------------------------------------------------

PROBLEM_DIM = 10
PROBLEM_SEED = 42          # _BBOBBase default; stated explicitly for the record
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
N_RUNS = 30

ALGORITHMS = ("PSO", "GA")
SEED_BASE = {"PSO": 1000, "GA": 2000}   # disjoint -> independent samples

ALPHA = 0.05

BAR = "=" * 78


def seeds_for(algorithm: str) -> list:
    """The explicit, reproducible seed list for *algorithm*."""
    base = SEED_BASE[algorithm]
    return [base + i for i in range(N_RUNS)]


# ---------------------------------------------------------------------------
# Step 3 — sanity check
# ---------------------------------------------------------------------------

def sanity_check(problem) -> None:
    """
    Sphere's global optimum is 0.0 at xopt. If this does not hold, the
    benchmark itself is broken and no algorithm comparison built on it means
    anything -- so this aborts rather than reporting downstream numbers.
    """
    print(f"\n{BAR}\n  STEP 3 - SANITY CHECK\n{BAR}")
    value = problem.evaluate(problem.xopt)
    print(f"  problem                  : {type(problem).__name__}(dim={PROBLEM_DIM}, seed={PROBLEM_SEED})")
    print(f"  problem.evaluate(xopt)   : {value!r}")
    print(f"  required                 : < 1e-6")
    if not value < 1e-6:
        print("\n  SANITY CHECK FAILED - aborting before any comparison.")
        sys.exit(1)
    print("  result                   : PASS")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_algorithm(algorithm: str, problem) -> list:
    """
    Run *algorithm* N_RUNS times, one explicit seed per run, and return the
    list of best-of-run fitnesses.

    EvaluationFramework.evaluate() runs n_runs iterations from ONE config, so
    a single seeded config would produce n_runs identical runs. To seed each
    run individually we call it with n_runs=1, N_RUNS times.
    """
    evaluator = EvaluationFramework(n_runs=1)
    results = []

    print(f"\n  [{algorithm}] {N_RUNS} runs "
          f"(pop={POPULATION_SIZE}, gens={MAX_GENERATIONS}) ...")

    for seed in seeds_for(algorithm):
        config = {
            "algorithm": algorithm,
            "population_size": POPULATION_SIZE,
            "max_generations": MAX_GENERATIONS,
            "seed": seed,
        }
        metrics = evaluator.evaluate(config, problem, verbose=False)
        best = float(metrics["best_fitness"])
        results.append(best)
        print(f"    seed={seed:<6} best_fitness={best:.6e}")

    return results


def describe(algorithm: str, values: list) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "algorithm": algorithm,
        "n_runs": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_table(rows: list) -> None:
    print(f"\n{BAR}\n  STEP 5 - RESULTS TABLE  (lower is better; minimisation)\n{BAR}")
    header = f"  {'algorithm':<10}{'n_runs':>8}{'mean':>15}{'std':>15}{'median':>15}{'min':>15}{'max':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['algorithm']:<10}{r['n_runs']:>8}"
              f"{r['mean']:>15.6e}{r['std']:>15.6e}{r['median']:>15.6e}"
              f"{r['min']:>15.6e}{r['max']:>15.6e}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compare(pso_vals: list, ga_vals: list) -> None:
    print(f"\n{BAR}\n  STEP 4 - STATISTICAL COMPARISON\n{BAR}")

    n1, n2 = len(pso_vals), len(ga_vals)
    u_stat, p_value = mannwhitneyu(pso_vals, ga_vals, alternative="two-sided")

    # Rank-biserial correlation from U:
    #     r = 1 - (2 * U) / (n1 * n2)
    # with U the statistic for the FIRST sample (PSO). Because this is a
    # minimisation problem, r > 0 means PSO tended to reach LOWER (better)
    # fitness than GA; r < 0 means GA did.
    r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)

    print(f"  test                 : scipy.stats.mannwhitneyu "
          f"(two-sided, independent samples)")
    print(f"  scipy version        : {scipy.__version__}")
    print(f"  n1 (PSO), n2 (GA)    : {n1}, {n2}")
    print(f"  U statistic          : {u_stat!r}")
    print(f"  p-value              : {p_value!r}")
    print(f"  effect size formula  : r = 1 - (2 * U) / (n1 * n2)")
    print(f"  rank-biserial r      : {r_rb!r}")
    print(f"  alpha                : {ALPHA}")

    significant = p_value < ALPHA
    print(f"\n  significant at alpha={ALPHA}? {'YES' if significant else 'NO'} "
          f"(p = {p_value:.6g} {'<' if significant else '>='} {ALPHA})")

    if significant:
        better = "PSO" if r_rb > 0 else "GA"
        print(f"  direction            : {better} reached lower fitness more often")
    else:
        print("  direction            : no statistically detectable difference")


def print_seed_lists() -> None:
    print(f"\n{BAR}\n  REPRODUCIBILITY - ALL SEEDS USED\n{BAR}")
    print(f"  problem seed (xopt/rotations): {PROBLEM_SEED}")
    for algorithm in ALGORITHMS:
        print(f"\n  {algorithm} seeds ({N_RUNS}):")
        print(f"    {seeds_for(algorithm)}")


# ---------------------------------------------------------------------------

def main() -> int:
    print(BAR)
    print("  PHASE 1 - BASELINE COMPARISON: PSO vs GA on BBOB F1 (Sphere), dim=10")
    print(BAR)
    print(f"  python           : {sys.version.split()[0]}")
    print(f"  numpy            : {np.__version__}")
    print(f"  scipy            : {scipy.__version__}")
    print(f"  population_size  : {POPULATION_SIZE}")
    print(f"  max_generations  : {MAX_GENERATIONS}")
    print(f"  runs per algo    : {N_RUNS}")

    problem = BBOB_F1_Sphere(dim=PROBLEM_DIM, seed=PROBLEM_SEED)

    sanity_check(problem)

    print(f"\n{BAR}\n  STEP 2 - RUNS\n{BAR}")
    values = {a: run_algorithm(a, problem) for a in ALGORITHMS}

    print_table([describe(a, values[a]) for a in ALGORITHMS])
    compare(values["PSO"], values["GA"])
    print_seed_lists()

    print(f"\n{BAR}\n  DONE\n{BAR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
