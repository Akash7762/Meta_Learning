"""
user_model_runner.py
--------------------
Interactive end-to-end runner for the AAD framework that follows the exact
pipeline documented in README.md:

    Problem  ->  Landscape Features  ->  Meta-Learner  ->  Best Config  ->  Solution
                                              ^
                                       Surrogate Model
                                       (cheap screening)

Execution phases
----------------
Phase 0 : user selects one problem and at least three algorithms (of 24)
Phase 1 : landscape feature extraction        (LandscapeFeatureExtractor)
Phase 2 : per-algorithm meta-learning         (EA_Designer + SurrogateModel)
          followed by multi-run evaluation    (EvaluationFramework)
Phase 3 : empirical comparison + ranking of the user-selected algorithms
Phase 4 : global surrogate fitted on Phase-2 evidence, used to score all 24
Phase 5 : LLM selects ONE final algorithm of 24, informed by Phases 1/3/4,
          then that pick is meta-learned and fully evaluated
Phase 6 : plots + JSON export

Run:
    python user_model_runner.py
"""

import json
import os

import numpy as np

from aad import (
    EXTENDED_CONFIG_SPACE,
    MEALPY_ALGORITHM_KEYS,
    EA_Designer,
    EvaluationFramework,
    LLMDesigner,
    LandscapeFeatureExtractor,
    SurrogateModel,
    Visualizer,
    config_to_vector,
    # problems
    SphereProblem,
    RastriginProblem,
    RosenbrockProblem,
    AckleyProblem,
    BBOB_F1_Sphere,
    BBOB_F3_RastriginSeparable,
    BBOB_F8_Rosenbrock,
    BBOB_F15_RastriginRotated,
    BBOB_F21_Gallagher101,
    BBOB_F24_LunacekBiRastrigin,
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

MPOP = 8       # meta-population size  (EA_Designer)
MGEN = 4       # meta-generations      (EA_Designer)
MRUN = 2       # EA runs per meta-evaluation
FRUN = 10      # runs in the final EvaluationFramework evaluation
DIM = 10       # problem dimension
ODIR = "results"

# Problems offered at the prompt
PROBS = {
    "1": ("BBOB_F1_Sphere", BBOB_F1_Sphere),
    "2": ("BBOB_F3_RastriginSeparable", BBOB_F3_RastriginSeparable),
    "3": ("BBOB_F8_Rosenbrock", BBOB_F8_Rosenbrock),
    "4": ("BBOB_F15_RastriginRotated", BBOB_F15_RastriginRotated),
    "5": ("BBOB_F21_Gallagher101", BBOB_F21_Gallagher101),
    "6": ("BBOB_F24_LunacekBiRastrigin", BBOB_F24_LunacekBiRastrigin),
    "7": ("SphereProblem", SphereProblem),
    "8": ("RastriginProblem", RastriginProblem),
    "9": ("RosenbrockProblem", RosenbrockProblem),
    "10": ("AckleyProblem", AckleyProblem),
}

SEP = "=" * 70
BAR = "#" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sp(a: str) -> dict:
    """
    Return a copy of EXTENDED_CONFIG_SPACE restricted to a single algorithm.

    This is what makes the meta-learner tune *within* one algorithm
    (population size, generations, and — for "ea" only — the operators)
    rather than searching across algorithms, so each of the user's
    algorithms gets a fair, individually-tuned comparison.
    """
    s = dict(EXTENDED_CONFIG_SPACE)
    s["algorithm"] = [a]
    return s


def _fin(v: float) -> float:
    """Replace NaN/Inf with a large finite penalty so JSON/plots stay valid."""
    return float(v) if np.isfinite(v) else 1e12


def _slog(v: float) -> float:
    """Signed log1p — used only for display of wide-range fitness values."""
    return float(np.sign(v) * np.log1p(abs(v)))


# ---------------------------------------------------------------------------
# Phase 0 : user selection
# ---------------------------------------------------------------------------

def pick_problem():
    """Prompt the user for one benchmark problem."""
    print(f"\n{SEP}\n  PHASE 0a - SELECT PROBLEM\n{SEP}")
    for k, (n, _) in PROBS.items():
        print(f"   {k:>2s}) {n}")

    while True:
        c = input("\n  Problem number: ").strip()
        if c in PROBS:
            n, cls = PROBS[c]
            print(f"  -> selected: {n} (dim={DIM})")
            return cls(dim=DIM)
        print("  Invalid choice, try again.")


def pick_algorithms():
    """Prompt the user for at least three algorithms out of the 24."""
    print(f"\n{SEP}\n  PHASE 0b - SELECT AT LEAST 3 ALGORITHMS (of 24)\n{SEP}")
    for i, a in enumerate(MEALPY_ALGORITHM_KEYS, start=1):
        end = "\n" if i % 6 == 0 else "  "
        print(f"  {i:>2d}) {a:<8s}", end=end)
    print()

    while True:
        raw = input("\n  Numbers (comma-separated, min 3): ").strip()
        try:
            ids = [int(x) for x in raw.replace(" ", "").split(",") if x]
        except ValueError:
            print("  Numbers only, e.g. 2,7,12")
            continue

        if any(i < 1 or i > len(MEALPY_ALGORITHM_KEYS) for i in ids):
            print(f"  Out of range (1-{len(MEALPY_ALGORITHM_KEYS)}).")
            continue

        algs = sorted({MEALPY_ALGORITHM_KEYS[i - 1] for i in ids})
        if len(algs) < 3:
            print("  At least 3 distinct algorithms are required.")
            continue

        print(f"  -> selected: {', '.join(algs)}")
        return algs


# ---------------------------------------------------------------------------
# Phase 1 : landscape features
# ---------------------------------------------------------------------------

def phase1(prob):
    """Extract ELA-inspired landscape features for the chosen problem."""
    print(f"\n{BAR}\n#  PHASE 1 - LANDSCAPE FEATURE EXTRACTION\n{BAR}")
    fx = LandscapeFeatureExtractor()
    ft = fx.extract_features(prob)
    fv = fx.features_to_vector(ft)

    for k, v in ft.items():
        print(f"    {k:<20s}: {v:.4f}")
    return ft, fv


# ---------------------------------------------------------------------------
# Phase 2 : per-algorithm meta-learning + evaluation
# ---------------------------------------------------------------------------

def phase2(prob, algs, fv):
    """
    For each user-selected algorithm:
      (a) run EA_Designer (surrogate-assisted) restricted to that algorithm,
      (b) fully evaluate the tuned configuration over FRUN independent runs.

    Returns
    -------
    res : dict  algorithm -> {config, metrics, meta_convergence}
    mx  : list  surrogate training rows  (landscape features + config encoding)
    my  : list  corresponding mean performance values
    """
    print(f"\n{BAR}\n#  PHASE 2 - META-LEARNING PER ALGORITHM\n{BAR}")
    ev = EvaluationFramework(n_runs=FRUN)
    res, mx, my = {}, [], []

    for a in algs:
        print(f"\n{SEP}\n  [{a}] meta-learning ({MPOP} configs x {MGEN} gens)\n{SEP}")

        dz = EA_Designer(
            config_space=_sp(a),
            population_size=MPOP,
            generations=MGEN,
            runs_per_eval=MRUN,
            use_surrogate=True,
            surrogate_type="rf",
        )
        cfg, perf = dz.evolve(prob, verbose=True)

        print(f"\n  [{a}] full evaluation of tuned config ({FRUN} runs) ...")
        mt = ev.evaluate(cfg, prob, verbose=False)
        EvaluationFramework.summarize(mt)

        res[a] = {
            "config": cfg,
            "metrics": {k: v for k, v in mt.items() if k != "histories"},
            "histories": mt["histories"],
            "meta_convergence": [_fin(x) for x in dz.meta_convergence],
        }

        # Re-encode every config the meta-learner visited against the FULL
        # config space so all rows share one dimensionality, letting Phase 4
        # fit a single surrogate across algorithms.
        for c, p in zip(dz.best_config_history, dz.meta_convergence):
            mx.append(list(fv) + config_to_vector(c, EXTENDED_CONFIG_SPACE))
            my.append(_fin(p))

    return res, mx, my


# ---------------------------------------------------------------------------
# Phase 3 : comparison
# ---------------------------------------------------------------------------

def phase3(res):
    """Rank the user-selected algorithms and print a comparison table."""
    print(f"\n{BAR}\n#  PHASE 3 - COMPARISON OF SELECTED ALGORITHMS\n{BAR}")

    rk = sorted(res.items(), key=lambda kv: kv[1]["metrics"]["best_fitness"])

    hdr = (
        f"  {'rank':<5s}{'algorithm':<10s}{'best':>15s}{'mean':>15s}"
        f"{'std':>14s}{'stab':>10s}{'sec/run':>9s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i, (a, r) in enumerate(rk, start=1):
        m = r["metrics"]
        print(
            f"  {i:<5d}{a:<10s}{m['best_fitness']:>15.4e}"
            f"{m['mean_fitness']:>15.4e}{m['std_fitness']:>14.4e}"
            f"{m['stability']:>10.2f}{m['mean_runtime']:>9.2f}"
        )

    win = rk[0][0]
    wm = rk[0][1]["metrics"]
    print(f"\n  Best by best_fitness   : {win}")
    print(f"  Best by mean_fitness   : "
          f"{min(res, key=lambda a: res[a]['metrics']['mean_fitness'])}")
    print(f"  Most stable (low std)  : "
          f"{min(res, key=lambda a: res[a]['metrics']['std_fitness'])}")
    print(f"  Fastest per run        : "
          f"{min(res, key=lambda a: res[a]['metrics']['mean_runtime'])}")
    print(f"\n  [Phase-3 winner] {win}  best={wm['best_fitness']:.6e}  "
          f"mean={wm['mean_fitness']:.6e}")
    print("\n  NOTE: ranking by a single best_fitness value is a point estimate."
          f"\n  With n_runs={FRUN} per algorithm, report a Wilcoxon signed-rank"
          "\n  (pairwise) or Friedman (multi-algorithm) test in the thesis before"
          "\n  claiming any of these differences is statistically significant.")
    return win, rk


# ---------------------------------------------------------------------------
# Phase 4 : global surrogate over all 24 algorithms
# ---------------------------------------------------------------------------

def phase4(fv, mx, my, algs):
    """
    Fit one surrogate on all Phase-2 evidence and score every one of the 24
    algorithms, producing an advisory shortlist for the LLM.

    Caveat (important, and stated in the printed output as well): the
    surrogate has seen training data for the user's selected algorithms
    only. Because `config_to_vector` one-hot encodes the algorithm field,
    the 21 unseen algorithms carry no training signal, so their predictions
    are extrapolation, not evidence. The shortlist is therefore advisory
    input to the LLM, never a substitute for real evaluation.
    """
    print(f"\n{BAR}\n#  PHASE 4 - GLOBAL SURROGATE SCREENING (all 24)\n{BAR}")

    if len(mx) < 5:
        print("  Not enough Phase-2 samples to fit a surrogate; skipping.")
        return {}

    sg = SurrogateModel(model_type="rf")
    sg.fit(mx, my)
    print(f"  Surrogate fitted on {len(mx)} (features + config) samples.")

    pr = {}
    for a in MEALPY_ALGORITHM_KEYS:
        c = {
            "algorithm": a,
            "population_size": 50,
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "max_generations": 100,
        }
        x = np.array(list(fv) + config_to_vector(c, EXTENDED_CONFIG_SPACE))
        pr[a] = _fin(float(sg.predict(x)[0]))

    sh = sorted(pr, key=lambda a: pr[a])[:8]
    print("\n  Surrogate-predicted ranking (top 8, lower = better):")
    for i, a in enumerate(sh, start=1):
        tag = "  <- evaluated in Phase 2" if a in algs else ""
        print(f"    {i}. {a:<8s} pred={pr[a]:>14.4e}{tag}")

    print("\n  CAVEAT: only the Phase-2 algorithms have training support;"
          "\n  the remaining predictions are extrapolation and are used purely"
          "\n  as advisory context for the LLM in Phase 5.")
    return {"predictions": pr, "shortlist": sh}


# ---------------------------------------------------------------------------
# Phase 5 : LLM final selection
# ---------------------------------------------------------------------------

def phase5(prob, ft, res, sur, rk):
    """
    Ask the LLM designer to name ONE final algorithm out of the 24, given
    the landscape features, the Phase-3 empirical results, and the Phase-4
    surrogate shortlist. The pick is then meta-learned and fully evaluated
    so that its numbers are directly comparable with Phase 3.
    """
    print(f"\n{BAR}\n#  PHASE 5 - LLM FINAL ALGORITHM SELECTION\n{BAR}")

    ld = LLMDesigner(
        api_key=os.getenv("GEMINI_API_KEY"),
        config_space=EXTENDED_CONFIG_SPACE,
    )
    mode = "Gemini API" if ld._use_llm else "rule-based fallback"
    print(f"  Designer mode: {mode}")

    # Build an evidence-rich description rather than the bare default one.
    ev_txt = "; ".join(
        f"{a}: best={r['metrics']['best_fitness']:.4e}" for a, r in rk
    )
    desc = (
        ld.generate_description(prob)
        + f" Landscape: variance={ft['fitness_variance']:.3f},"
        + f" modality={ft['modality']:.3f},"
        + f" ruggedness={ft['ruggedness']:.3f}."
        + f" Measured results so far -> {ev_txt}."
    )
    if sur.get("shortlist"):
        desc += f" Surrogate shortlist: {', '.join(sur['shortlist'][:5])}."

    print(f"\n  Description given to designer:\n    {desc}")

    cfg = ld.propose_algorithm(desc)
    a = cfg.get("algorithm")
    print(f"\n  [LLM] proposed config : {cfg}")

    if a not in MEALPY_ALGORITHM_KEYS:
        print(f"  [LLM] '{a}' is not one of the 24; falling back to the"
              " surrogate's top pick.")
        a = (sur.get("shortlist") or [rk[0][0]])[0]
        cfg["algorithm"] = a
        print(f"  [LLM] corrected to     : {a}")

    # Tune the LLM's pick with the same budget as Phase 2, so the final
    # comparison is like-for-like rather than favouring either side.
    print(f"\n{SEP}\n  [{a}] meta-learning the LLM pick\n{SEP}")
    dz = EA_Designer(
        config_space=_sp(a),
        population_size=MPOP,
        generations=MGEN,
        runs_per_eval=MRUN,
        use_surrogate=True,
        surrogate_type="rf",
    )
    bc, _ = dz.evolve(prob, verbose=True)

    print(f"\n  [{a}] full evaluation ({FRUN} runs) ...")
    ev = EvaluationFramework(n_runs=FRUN)
    mt = ev.evaluate(bc, prob, verbose=False)
    EvaluationFramework.summarize(mt)

    out = {
        "mode": mode,
        "description": desc,
        "proposed_config": cfg,
        "tuned_config": bc,
        "metrics": {k: v for k, v in mt.items() if k != "histories"},
        "histories": mt["histories"],
        "meta_convergence": [_fin(x) for x in dz.meta_convergence],
    }

    # Final verdict against the Phase-3 winner.
    w, wr = rk[0]
    lb = mt["best_fitness"]
    wb = wr["metrics"]["best_fitness"]
    print(f"\n{SEP}\n  FINAL VERDICT\n{SEP}")
    print(f"  Phase-3 winner (user set) : {w:<8s} best={wb:.6e}")
    print(f"  Phase-5 LLM selection     : {a:<8s} best={lb:.6e}")

    if a == w:
        print("\n  -> The LLM converged on the same algorithm the empirical"
              "\n     comparison already identified. This is agreement, not"
              "\n     independent confirmation: the LLM was shown those results.")
    elif lb < wb:
        print(f"\n  -> The LLM's pick ({a}) outperformed the best of the user's"
              f"\n     selected set ({w}) by {abs(wb - lb):.4e}.")
    else:
        print(f"\n  -> The LLM's pick ({a}) did NOT beat {w}. Report this;"
              "\n     a negative result for LLM-guided selection is a finding.")

    return out, a


# ---------------------------------------------------------------------------
# Phase 6 : plots + export
# ---------------------------------------------------------------------------

def phase6(prob, res, llm, la):
    """Save convergence / comparison plots and a JSON record of the run."""
    print(f"\n{BAR}\n#  PHASE 6 - VISUALISATION + EXPORT\n{BAR}")
    vz = Visualizer(output_dir=ODIR)
    pn = prob.name
    pt = []

    for a, r in res.items():
        pt.append(vz.plot_fitness_curves(
            r["histories"], f"{pn} - {a}", f"fit_{pn}_{a}.png"))
        pt.append(vz.plot_meta_convergence(
            r["meta_convergence"], f"{pn} - {a} meta", f"meta_{pn}_{a}.png"))

    pt.append(vz.plot_fitness_curves(
        llm["histories"], f"{pn} - LLM pick {la}", f"fit_{pn}_LLM_{la}.png"))

    cmp_in = [(r["config"], r["metrics"]) for r in res.values()]
    cmp_in.append((llm["tuned_config"], llm["metrics"]))
    pt.append(vz.plot_algorithm_comparison(
        cmp_in, f"{pn} - algorithm comparison", f"cmp_{pn}.png"))

    rec = {
        "problem": pn,
        "dimension": DIM,
        "budget": {"meta_pop": MPOP, "meta_gen": MGEN,
                   "runs_per_eval": MRUN, "final_runs": FRUN},
        "phase2": {a: {"config": r["config"], "metrics": r["metrics"],
                       "meta_convergence": r["meta_convergence"]}
                   for a, r in res.items()},
        "phase5_llm": {k: v for k, v in llm.items() if k != "histories"},
    }

    os.makedirs(ODIR, exist_ok=True)
    fp = os.path.join(ODIR, f"user_model_{pn}.json")
    with open(fp, "w") as f:
        json.dump(rec, f, indent=2, default=str)

    for p in pt:
        print(f"    saved: {p}")
    print(f"    saved: {fp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{BAR}\n#  AUTONOMOUS ALGORITHM DESIGN - USER MODEL RUNNER\n{BAR}")
    print("  Pipeline: Problem -> Features -> Meta-Learner (+Surrogate)")
    print("            -> Comparison -> LLM Selection -> Final Evaluation")

    prob = pick_problem()
    algs = pick_algorithms()

    ft, fv = phase1(prob)
    res, mx, my = phase2(prob, algs, fv)
    win, rk = phase3(res)
    sur = phase4(fv, mx, my, algs)
    llm, la = phase5(prob, ft, res, sur, rk)
    phase6(prob, res, llm, la)

    print(f"\n{BAR}\n#  RUN COMPLETE\n{BAR}")
    print(f"  Problem            : {prob.name}")
    print(f"  Algorithms compared: {', '.join(algs)}")
    print(f"  Phase-3 winner     : {win}")
    print(f"  Phase-5 LLM pick   : {la}")


if __name__ == "__main__":
    main()
