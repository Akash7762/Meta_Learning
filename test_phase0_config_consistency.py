"""
test_phase0_config_consistency.py
---------------------------------
Phase 0 consistency harness for the AAD framework's config-space decision.

Checks that the "algorithm" key contract is honoured at every top-level entry
point (build_algorithm, EA_Designer, LLMDesigner), and that the "ea" branch and
the mealpy branch dispatch to the classes they are supposed to.

This script is diagnostic: it records and prints a PASS/FAIL line for every
check and keeps going after a failure, so one run shows the full picture.
It modifies nothing and writes nothing.

Run:
    .venv\\Scripts\\python.exe test_phase0_config_consistency.py

Exit code is 0 if every check passed, 1 otherwise.
"""

import random
import sys
import traceback

from aad.config_space import (
    EA_CONFIG_SPACE,
    EXTENDED_CONFIG_SPACE,
    random_config,
)
from aad.evolutionary_algorithm import EvolutionaryAlgorithm, build_algorithm
from aad.mealpy_algorithms import MealpyAlgorithm
from aad.meta_learner import EA_Designer
from aad.llm_designer import LLMDesigner
from aad.problems import SphereProblem


# ---------------------------------------------------------------------------
# Tiny check harness (collect-all, not fail-fast)
# ---------------------------------------------------------------------------

RESULTS: list = []


def check(name: str, fn) -> None:
    """
    Run *fn* and record a PASS/FAIL line.

    *fn* returns either ``True``, or ``(ok: bool, detail: str)``.  Any
    uncaught exception inside *fn* is itself a FAIL, never a crash.
    """
    try:
        outcome = fn()
        if isinstance(outcome, tuple):
            ok, detail = outcome
        else:
            ok, detail = bool(outcome), ""
    except Exception as exc:                     # noqa: BLE001 - diagnostic harness
        ok = False
        detail = f"unexpected {type(exc).__name__}: {exc}"
        RESULTS.append((ok, name, detail, traceback.format_exc()))
        print(f"[FAIL] {name}\n         {detail}")
        return

    RESULTS.append((ok, name, detail, None))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"\n         {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


# ---------------------------------------------------------------------------
# Group 1 — the config spaces themselves
# ---------------------------------------------------------------------------

def g1_ea_space_has_no_algorithm_key():
    missing = "algorithm" not in EA_CONFIG_SPACE
    return missing, f"EA_CONFIG_SPACE keys = {sorted(EA_CONFIG_SPACE)}"


def g1_extended_space_has_algorithm_key():
    if "algorithm" not in EXTENDED_CONFIG_SPACE:
        return False, "EXTENDED_CONFIG_SPACE has no 'algorithm' key"
    opts = EXTENDED_CONFIG_SPACE["algorithm"]
    ok = opts[0] == "ea" and len(opts) == 25
    return ok, f"{len(opts)} options, first={opts[0]!r} (expected 25, 'ea')"


def g1_random_config_ea_space_omits_algorithm():
    cfg = random_config(EA_CONFIG_SPACE)
    return "algorithm" not in cfg, f"sampled keys = {sorted(cfg)}"


def g1_random_config_extended_includes_algorithm():
    cfg = random_config(EXTENDED_CONFIG_SPACE)
    return "algorithm" in cfg, f"sampled algorithm = {cfg.get('algorithm')!r}"


# ---------------------------------------------------------------------------
# Group 2 — build_algorithm dispatch contract
# ---------------------------------------------------------------------------

def g2_missing_algorithm_raises_clear_error():
    try:
        build_algorithm({"population_size": 20, "max_generations": 10})
    except ValueError as exc:
        clear = "algorithm" in str(exc)
        return clear, f"ValueError: {exc}"
    except Exception as exc:                     # noqa: BLE001
        return False, f"raised {type(exc).__name__}, expected ValueError: {exc}"
    return False, "no exception raised - config without 'algorithm' was accepted"


def g2_unknown_algorithm_raises():
    cfg = {"algorithm": "NOT_A_REAL_ALGORITHM", "population_size": 20,
           "max_generations": 10}
    try:
        build_algorithm(cfg)
    except ValueError as exc:
        return True, f"ValueError: {str(exc)[:70]}..."
    except Exception as exc:                     # noqa: BLE001
        return False, f"raised {type(exc).__name__}, expected ValueError"
    return False, "no exception raised for an unregistered algorithm key"


def g2_explicit_ea_builds_evolutionary_algorithm():
    alg = build_algorithm({"algorithm": "ea", "population_size": 20,
                           "max_generations": 10})
    ok = isinstance(alg, EvolutionaryAlgorithm)
    return ok, f"got {type(alg).__name__}"


def g2_explicit_pso_builds_mealpy_algorithm():
    alg = build_algorithm({"algorithm": "PSO", "population_size": 20,
                           "max_generations": 10})
    ok = isinstance(alg, MealpyAlgorithm)
    return ok, f"got {type(alg).__name__}"


# ---------------------------------------------------------------------------
# Group 3 — sampled EXTENDED configs dispatch correctly
# ---------------------------------------------------------------------------

def g3_sampled_ea_config_uses_operators():
    """
    A config sampled from EXTENDED_CONFIG_SPACE with algorithm="ea" must build
    an EvolutionaryAlgorithm whose operators actually reflect the config.
    """
    cfg = random_config(EXTENDED_CONFIG_SPACE)
    cfg["algorithm"] = "ea"
    alg = build_algorithm(cfg)

    if not isinstance(alg, EvolutionaryAlgorithm):
        return False, f"got {type(alg).__name__}, expected EvolutionaryAlgorithm"

    mismatches = [
        f"{key}: config={cfg[key]!r} instance={getattr(alg, key + '_name')!r}"
        for key in ("selection", "crossover", "mutation")
        if cfg[key] != getattr(alg, key + "_name")
    ]
    if mismatches:
        return False, "operator not carried through -> " + "; ".join(mismatches)

    instantiated = all(
        getattr(alg, key, None) is not None
        for key in ("selection", "crossover", "mutation")
    )
    if not instantiated:
        return False, "operator objects were not instantiated on the instance"

    return True, (
        f"sel={alg.selection_name} cx={alg.crossover_name} "
        f"mut={alg.mutation_name}, operator objects live"
    )


def g3_every_non_ea_algorithm_builds_mealpy():
    """Every non-'ea' value in EXTENDED_CONFIG_SPACE must yield a MealpyAlgorithm."""
    bad: list = []
    keys = [a for a in EXTENDED_CONFIG_SPACE["algorithm"] if a != "ea"]
    for alg_key in keys:
        cfg = random_config(EXTENDED_CONFIG_SPACE)
        cfg["algorithm"] = alg_key
        try:
            alg = build_algorithm(cfg)
        except Exception as exc:                 # noqa: BLE001
            bad.append(f"{alg_key}: raised {type(exc).__name__}")
            continue
        if not isinstance(alg, MealpyAlgorithm):
            bad.append(f"{alg_key}: got {type(alg).__name__}")
    if bad:
        return False, f"{len(bad)}/{len(keys)} wrong -> {'; '.join(bad[:5])}"
    return True, f"all {len(keys)} mealpy keys dispatched to MealpyAlgorithm"


def g3_mealpy_ignores_operator_keys():
    """
    Documents the asymmetry: operator keys are silently dropped for non-'ea'
    algorithms. Two configs differing only in operators must be identical to
    MealpyAlgorithm.
    """
    base = {"algorithm": "PSO", "population_size": 20, "max_generations": 10}
    a = build_algorithm({**base, "selection": "tournament",
                         "crossover": "sbx", "mutation": "gaussian"})
    b = build_algorithm({**base, "selection": "roulette",
                         "crossover": "uniform", "mutation": "polynomial"})
    same = (
        a.algorithm_key == b.algorithm_key
        and a.population_size == b.population_size
        and a.max_generations == b.max_generations
        and not hasattr(a, "selection_name")
    )
    return same, "operator keys are ignored by MealpyAlgorithm (expected)"


# ---------------------------------------------------------------------------
# Group 4 — EA_Designer entry point
# ---------------------------------------------------------------------------

def g4_designer_default_space_is_predictable():
    """
    EA_Designer() with no config_space must behave predictably: either it
    defaults to a space carrying 'algorithm', or it fails with a clear error.
    Silently producing algorithm-less configs is the failure mode.
    """
    d = EA_Designer(population_size=2, generations=1, runs_per_eval=1,
                    use_surrogate=False)
    cand = d.generate_candidate()

    if "algorithm" in cand:
        return True, "default space carries 'algorithm' (compliant)"

    try:
        d.evolve(SphereProblem(dim=3), verbose=False)
    except ValueError as exc:
        return True, (
            "default space omits 'algorithm'; evolve() fails loudly with "
            f"ValueError: {str(exc)[:60]}... (predictable, but non-compliant "
            "with the Phase 0 decision - see docs/planning/phase0/phase0_decision.md item 1, git tag v1-final)"
        )
    except Exception as exc:                     # noqa: BLE001
        return False, f"failed with {type(exc).__name__}, not a clear ValueError"

    return False, "evolve() silently succeeded on an algorithm-less config"


def g4_designer_candidates_carry_algorithm_with_extended():
    d = EA_Designer(config_space=EXTENDED_CONFIG_SPACE, population_size=2,
                    generations=1, runs_per_eval=1, use_surrogate=False)
    cands = [d.generate_candidate() for _ in range(20)]
    missing = [c for c in cands if "algorithm" not in c]
    return not missing, f"{len(cands) - len(missing)}/{len(cands)} carry 'algorithm'"


def g4_designer_variation_preserves_algorithm():
    """Mutation and crossover must never drop the 'algorithm' key."""
    d = EA_Designer(config_space=EXTENDED_CONFIG_SPACE, population_size=2,
                    generations=1, runs_per_eval=1, use_surrogate=False)
    bad = 0
    for _ in range(50):
        p1, p2 = d.generate_candidate(), d.generate_candidate()
        child = d._mutate_config(d._crossover_configs(p1, p2))
        if "algorithm" not in child:
            bad += 1
    return bad == 0, f"{50 - bad}/50 offspring retained 'algorithm'"


def g4_designer_evolve_smoke_on_extended():
    """
    End-to-end smoke on a deliberately narrowed copy of EXTENDED_CONFIG_SPACE
    (small budgets, ea + PSO only) so the harness stays fast.
    """
    space = dict(EXTENDED_CONFIG_SPACE)
    space["algorithm"] = ["ea", "PSO"]
    space["population_size"] = [10]
    space["max_generations"] = [10]

    d = EA_Designer(config_space=space, population_size=2, generations=1,
                    runs_per_eval=1, use_surrogate=False)
    cfg, perf = d.evolve(SphereProblem(dim=3), verbose=False)
    ok = isinstance(cfg, dict) and "algorithm" in cfg and isinstance(perf, float)
    return ok, f"best algorithm={cfg.get('algorithm')!r} perf={perf:.6g}"


# ---------------------------------------------------------------------------
# Group 5 — LLMDesigner entry point
# ---------------------------------------------------------------------------
# Only the rule-based path is exercised: this is an offline consistency check
# and must not depend on a network call or an API key.

def g5_rule_based_emits_algorithm_with_extended():
    d = LLMDesigner(config_space=EXTENDED_CONFIG_SPACE)
    cfg = d._rule_based_proposal(
        "A high-dimensional noisy multimodal benchmark with 50 variables."
    )
    if "algorithm" not in cfg:
        return False, f"proposal missing 'algorithm': keys={sorted(cfg)}"
    valid = cfg["algorithm"] in EXTENDED_CONFIG_SPACE["algorithm"]
    return valid, f"proposed algorithm={cfg['algorithm']!r}"


def g5_rule_based_output_is_buildable():
    d = LLMDesigner(config_space=EXTENDED_CONFIG_SPACE)
    cfg = d._rule_based_proposal("A smooth convex unimodal 10-variable problem.")
    alg = build_algorithm(cfg)
    return True, f"proposal built -> {type(alg).__name__}"


def g5_rule_based_with_ea_space_is_predictable():
    """
    LLMDesigner() defaulting to EA_CONFIG_SPACE must still emit an 'algorithm'
    key. It does, but it can only ever be "ea" - which is the (a)/(b)
    conflation noted in the decision doc, not a crash.
    """
    d = LLMDesigner(config_space=EA_CONFIG_SPACE)
    cfgs = [
        d._rule_based_proposal("A high-dimensional noisy multimodal problem.")
        for _ in range(10)
    ]
    missing = [c for c in cfgs if "algorithm" not in c]
    if missing:
        return False, f"{len(missing)}/10 proposals had no 'algorithm' key"
    proposed = {c["algorithm"] for c in cfgs}
    return True, (
        f"always emits 'algorithm'; reachable values = {sorted(proposed)} "
        "(collapses to 'ea' - conflates question (a) with (b))"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    random.seed(0)

    print("=" * 70)
    print("  PHASE 0 - CONFIG-SPACE CONSISTENCY CHECKS")
    print("=" * 70)

    section("1. config space definitions")
    check("EA_CONFIG_SPACE has no 'algorithm' key",
          g1_ea_space_has_no_algorithm_key)
    check("EXTENDED_CONFIG_SPACE has 'algorithm' with 25 options",
          g1_extended_space_has_algorithm_key)
    check("random_config(EA_CONFIG_SPACE) omits 'algorithm'",
          g1_random_config_ea_space_omits_algorithm)
    check("random_config(EXTENDED_CONFIG_SPACE) includes 'algorithm'",
          g1_random_config_extended_includes_algorithm)

    section("2. build_algorithm dispatch contract")
    check("build_algorithm raises a clear error on missing 'algorithm'",
          g2_missing_algorithm_raises_clear_error)
    check("build_algorithm raises on an unregistered algorithm key",
          g2_unknown_algorithm_raises)
    check("build_algorithm({'algorithm':'ea'}) -> EvolutionaryAlgorithm",
          g2_explicit_ea_builds_evolutionary_algorithm)
    check("build_algorithm({'algorithm':'PSO'}) -> MealpyAlgorithm",
          g2_explicit_pso_builds_mealpy_algorithm)

    section("3. sampled EXTENDED configs dispatch correctly")
    check("sampled EXTENDED + algorithm='ea' -> EA with operators applied",
          g3_sampled_ea_config_uses_operators)
    check("sampled EXTENDED + any non-'ea' -> MealpyAlgorithm (all 21)",
          g3_every_non_ea_algorithm_builds_mealpy)
    check("MealpyAlgorithm ignores selection/crossover/mutation",
          g3_mealpy_ignores_operator_keys)

    section("4. EA_Designer entry point")
    check("EA_Designer() default config_space behaves predictably",
          g4_designer_default_space_is_predictable)
    check("EA_Designer(EXTENDED) candidates carry 'algorithm'",
          g4_designer_candidates_carry_algorithm_with_extended)
    check("EA_Designer variation operators preserve 'algorithm'",
          g4_designer_variation_preserves_algorithm)
    check("EA_Designer.evolve smoke run on narrowed EXTENDED",
          g4_designer_evolve_smoke_on_extended)

    section("5. LLMDesigner entry point (rule-based path only, no network)")
    check("rule-based proposal emits a valid 'algorithm' (EXTENDED)",
          g5_rule_based_emits_algorithm_with_extended)
    check("rule-based proposal is accepted by build_algorithm",
          g5_rule_based_output_is_buildable)
    check("LLMDesigner(EA_CONFIG_SPACE) still emits 'algorithm'",
          g5_rule_based_with_ea_space_is_predictable)

    # ---- summary --------------------------------------------------------
    passed = sum(1 for ok, *_ in RESULTS if ok)
    failed = len(RESULTS) - passed

    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {len(RESULTS)} total")
    print("=" * 70)

    if failed:
        print("\n  Failed checks:")
        for ok, name, detail, _tb in RESULTS:
            if not ok:
                print(f"    - {name}")
                if detail:
                    print(f"      {detail}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
