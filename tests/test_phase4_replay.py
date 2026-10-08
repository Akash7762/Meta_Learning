"""
tests/test_phase4_replay.py
---------------------------
Checks for the Phase 4 changes: EA_Designer's run budget, evaluator hook,
feature switch and skipped-candidate penalty, plus phase4_replay's exact
best-of-k formula and stored-run replay.
Run with:  python -m pytest tests/ -q
"""

import itertools
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from aad.config_space import EXTENDED_CONFIG_SPACE
from aad.meta_learner import EA_Designer
from aad.problems import SphereProblem
from phase4_replay import ReplayEvaluator, expected_best_of_k, run_replay

ALGOS = sorted(EXTENDED_CONFIG_SPACE["algorithm"])
SPACE = {"algorithm": list(EXTENDED_CONFIG_SPACE["algorithm"])}


def fake_runs(n_runs, seed=0, offset=0.0):
    """Stored runs for one row: algorithms x runs, mixed signs if offset < 0."""
    return np.random.default_rng(seed).normal(size=(len(ALGOS), n_runs)) * 10 + offset


# --- exact best-of-k -------------------------------------------------------

@pytest.mark.parametrize("values,k", [
    (np.random.default_rng(1).normal(size=5), 1),
    (np.random.default_rng(2).normal(size=5), 3),
    (np.random.default_rng(3).normal(size=5), 5),
    (np.random.default_rng(4).normal(size=7), 4),
    (np.array([1.0, 1.0, 2.0, 3.0]), 2),          # ties
])
def test_expected_best_of_k_matches_brute_force(values, k):
    brute = np.mean([min(c) for c in itertools.combinations(values, k)])
    assert expected_best_of_k(values, k) == pytest.approx(brute, rel=1e-12)


def test_expected_best_of_k_with_all_runs_is_the_minimum():
    v = np.random.default_rng(5).normal(size=15)
    assert expected_best_of_k(v, 15) == v.min()


# --- replay ----------------------------------------------------------------

def test_evaluator_hands_out_each_stored_run_once():
    ev = ReplayEvaluator({"A": np.array([3.0, 1.0, 2.0])}, np.random.default_rng(0))
    got = sorted(ev({"algorithm": "A"}) for _ in range(3))
    assert got == [1.0, 2.0, 3.0] and ev.reused == 0
    ev({"algorithm": "A"})                         # cell exhausted
    assert ev.reused == 1


@pytest.mark.parametrize("pop,surrogate", [(5, True), (5, False), (10, True), (10, False)])
def test_replay_spends_the_exact_budget_without_reuse(pop, surrogate):
    out = run_replay(fake_runs(15, offset=-50.0), ALGOS, SPACE,
                     {"population_size": pop, "use_surrogate": surrogate}, seed=123)
    assert sum(out["runs_per_algo"].values()) == 15
    assert out["reused"] == 0


def test_replay_is_repeatable():
    runs = fake_runs(15, seed=1, offset=-50.0)
    variant = {"population_size": 5, "use_surrogate": True}
    assert run_replay(runs, ALGOS, SPACE, variant, seed=7) == \
        run_replay(runs, ALGOS, SPACE, variant, seed=7)


# --- EA_Designer changes ---------------------------------------------------

def test_no_budget_keeps_the_fixed_schedule():
    d = EA_Designer(population_size=3, generations=2, runs_per_eval=1,
                    use_surrogate=False, evaluator=lambda c, p: 1.0)
    d.evolve(SphereProblem(dim=3), verbose=False)
    assert d.runs_used == 3 + 3 * 2
    assert len(d.meta_convergence) == 3 and d.leftover_runs == 0


def test_leftover_runs_go_to_the_best_config():
    d = EA_Designer(config_space=SPACE, population_size=5, generations=0,
                    runs_per_eval=1, use_surrogate=False,
                    evaluator=lambda c, p: float(ALGOS.index(c["algorithm"])),
                    max_runs=15, use_features=False)
    best, _ = d.evolve(SphereProblem(dim=3), verbose=False)
    assert d.runs_used == 15 and d.leftover_runs == 10
    assert all(r["config"] == best for r in d.run_log[5:])


def test_budget_smaller_than_one_candidate_is_refused():
    with pytest.raises(ValueError):
        EA_Designer(runs_per_eval=3, max_runs=2)


def test_use_features_false_never_extracts(monkeypatch):
    d = EA_Designer(config_space=SPACE, population_size=3, generations=1,
                    runs_per_eval=1, use_surrogate=True,
                    evaluator=lambda c, p: 1.0, max_runs=6, use_features=False)

    def boom(*args, **kwargs):
        raise AssertionError("features were extracted")

    monkeypatch.setattr(d._feature_extractor, "extract_features", boom)
    d.evolve(SphereProblem(dim=3), verbose=False)
    assert d.runs_used == 6


def test_skipped_candidate_never_becomes_elite_on_negative_values():
    """With all values negative, the old 'prediction * 1.05' made skipped
    children look better than every real run. The elite must be a real run."""
    random.seed(0)
    d = EA_Designer(config_space=SPACE, population_size=5, generations=5,
                    runs_per_eval=1, use_surrogate=True,
                    evaluator=lambda c, p: -1000.0 + 0.5 * ALGOS.index(c["algorithm"]),
                    use_features=False, surrogate_n_jobs=1)
    _, best_perf = d.evolve(SphereProblem(dim=3), verbose=False)
    skipped = 5 * 5 - (d.runs_used - 5)
    assert skipped > 0, "no child was skipped, so the test would prove nothing"
    assert best_perf in {r["fitness"] for r in d.run_log}
