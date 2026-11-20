"""
mealpy_algorithms.py
--------------------
Adapter layer that connects the `mealpy` meta-heuristic library
(https://github.com/thieu1995/mealpy) to the AAD framework's
`OptimizationProblem` interface.

Classes
-------
MealpyAlgorithm  : Drop-in replacement for `EvolutionaryAlgorithm` that
                   runs any algorithm registered in `MEALPY_REGISTRY`.

Constants
---------
MEALPY_REGISTRY  : dict mapping short algorithm keys to
                   (module_path, class_name, extra_kwargs).
MEALPY_ALGORITHM_KEYS : sorted list of all registered keys.

Usage
-----
    from aad.mealpy_algorithms import MealpyAlgorithm

    cfg = {"algorithm": "PSO", "population_size": 50, "max_generations": 100}
    alg = MealpyAlgorithm(cfg)
    best_sol, best_fit = alg.run(problem)
    print(alg.history)        # convergence curve (one value per epoch)
"""

import importlib

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry: key -> (module_path, class_name, extra_kwargs)
#
# extra_kwargs are passed to the constructor *in addition* to
# epoch=max_generations and pop_size=population_size.
#
# Algorithms suspected of plagiarism (as flagged on the mealpy README) are
# intentionally excluded.
#
# CMA_ES, AOA and DE were excluded in v1 (Phase 1b diagnostic, 2026-09-15)
# and restored on 2026-11-19 at the user's request, so the registry again
# holds the project's original 24 algorithms. Keep in mind what v1 measured:
# all three construct and run without error and return finite values (so a
# crash-based smoke test does not catch them), but their mealpy
# implementations did not optimise -- a 10x evaluation budget did not improve
# the result on BBOB F1 Sphere, dim=10:
#
#   CMA_ES  Covariance matrix collapses (||C|| 2.97 -> 0.093) while the step
#           size never adapts (||sigma|| ~3.0 throughout), freezing the search
#           3.34 away from the optimum. Identical result at 5k/20k/50k
#           evaluations. mealpy exposes no parameters for it beyond epoch and
#           pop_size, so there is no adapter-level fix.
#   AOA     Returns 2.4756e+01 at every budget and never improves from its
#           first epoch -- more completely frozen than CMA_ES.
#   DE      Stagnates at ~7.3e-01 after early progress; 2000 vs 20000
#           evaluations differ only in the fifth decimal.
#
# The v2 algorithm health screen (Phase 2) re-tests them before any data is
# built. Evidence: results/phase1b/phase1b_screening.txt in git tag v1-final.

MEALPY_REGISTRY: dict = {
    # ------------------------------------------------------------------ #
    # Evolutionary-based                                                   #
    # ------------------------------------------------------------------ #
    "GA": (
        "mealpy.evolutionary_based.GA",
        "BaseGA",
        {},
    ),
    "DE": (
        "mealpy.evolutionary_based.DE",
        "OriginalDE",
        {},
    ),
    "JADE": (
        "mealpy.evolutionary_based.DE",
        "JADE",
        {},
    ),
    "CMA_ES": (
        "mealpy.evolutionary_based.ES",
        "CMA_ES",
        {},
    ),
    "SHADE": (
        "mealpy.evolutionary_based.SHADE",
        "OriginalSHADE",
        {},
    ),
    "L_SHADE": (
        "mealpy.evolutionary_based.SHADE",
        "L_SHADE",
        {},
    ),
    # ------------------------------------------------------------------ #
    # Swarm-based                                                          #
    # ------------------------------------------------------------------ #
    "PSO": (
        "mealpy.swarm_based.PSO",
        "OriginalPSO",
        {},
    ),
    "GWO": (
        "mealpy.swarm_based.GWO",
        "OriginalGWO",
        {},
    ),
    "WOA": (
        "mealpy.swarm_based.WOA",
        "OriginalWOA",
        {},
    ),
    "HHO": (
        "mealpy.swarm_based.HHO",
        "OriginalHHO",
        {},
    ),
    "ABC": (
        "mealpy.swarm_based.ABC",
        "OriginalABC",
        {},
    ),
    "GJO": (
        "mealpy.swarm_based.GJO",
        "OriginalGJO",
        {},
    ),
    "ARO": (
        "mealpy.swarm_based.ARO",
        "OriginalARO",
        {},
    ),
    "AO": (
        "mealpy.swarm_based.AO",
        "OriginalAO",
        {},
    ),
    "MPA": (
        "mealpy.swarm_based.MPA",
        "OriginalMPA",
        {},
    ),
    "HBA": (
        "mealpy.swarm_based.HBA",
        "OriginalHBA",
        {},
    ),
    # ------------------------------------------------------------------ #
    # Physics-based                                                        #
    # ------------------------------------------------------------------ #
    "SA": (
        "mealpy.physics_based.SA",
        "OriginalSA",
        {},
    ),
    "EO": (
        "mealpy.physics_based.EO",
        "OriginalEO",
        {},
    ),
    "MVO": (
        "mealpy.physics_based.MVO",
        "OriginalMVO",
        {},
    ),
    "RIME": (
        "mealpy.physics_based.RIME",
        "OriginalRIME",
        {},
    ),
    # ------------------------------------------------------------------ #
    # Bio-inspired                                                         #
    # ------------------------------------------------------------------ #
    "SMA": (
        "mealpy.bio_based.SMA",
        "OriginalSMA",
        {},
    ),
    "BBO": (
        "mealpy.bio_based.BBO",
        "OriginalBBO",
        {},
    ),
    # ------------------------------------------------------------------ #
    # Math-based                                                           #
    # ------------------------------------------------------------------ #
    "SCA": (
        "mealpy.math_based.SCA",
        "OriginalSCA",
        {},
    ),
    "AOA": (
        "mealpy.math_based.AOA",
        "OriginalAOA",
        {},
    ),
}

MEALPY_ALGORITHM_KEYS: list = sorted(MEALPY_REGISTRY.keys())

# (algorithm_key, pop_size) -> (init_cost, per_epoch_cost) in evaluations.
# Populated lazily by MealpyAlgorithm._measure_cost; see its docstring.
_COST_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class MealpyAlgorithm:
    """
    Wraps any mealpy optimizer with the same interface as
    `EvolutionaryAlgorithm`, so it can be used transparently by the
    evaluation and meta-learning layers.

    Parameters
    ----------
    config : dict
        Required key:
          algorithm        : str  – key into MEALPY_REGISTRY (e.g. "PSO")
        Optional keys (shared with EvolutionaryAlgorithm):
          population_size  : int  – passed as ``pop_size`` to mealpy (default 50)
          max_generations  : int  – passed as ``epoch``    to mealpy (default 100)
          seed             : int  – passed to ``solve(seed=...)`` for a
                                    reproducible run (default None = random).
                                    mealpy seeds its own Generator from this,
                                    so the global numpy RNG has no effect on it.
          max_fe           : int  – hard cap on objective-function evaluations,
                                    applied via ``mealpy.Termination(max_fe=…)``.
                                    Use this, not ``max_generations``, when
                                    comparing algorithms: population methods
                                    spend pop_size evaluations per generation
                                    while single-solution methods (e.g. SA)
                                    spend roughly one, so equal generations is
                                    NOT an equal budget. Default None = use
                                    ``max_generations`` instead.

    Attributes
    ----------
    history : list[float]
        Global-best fitness at each epoch (populated after `run`).
    """

    def __init__(self, config: dict):
        self.config = config
        self.algorithm_key = config.get("algorithm")
        self.population_size = int(config.get("population_size", 50))
        self.max_generations = int(config.get("max_generations", 100))
        seed = config.get("seed")
        self.seed = None if seed is None else int(seed)
        max_fe = config.get("max_fe")
        self.max_fe = None if max_fe is None else int(max_fe)

        if self.algorithm_key not in MEALPY_REGISTRY:
            raise ValueError(
                f"Unknown mealpy algorithm: '{self.algorithm_key}'. "
                f"Available: {MEALPY_ALGORITHM_KEYS}"
            )

        self.history: list = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_model(self, epoch: int = None):
        """Instantiate the mealpy optimizer from the registry."""
        module_path, class_name, extra_kwargs = MEALPY_REGISTRY[self.algorithm_key]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(
            epoch=self.max_generations if epoch is None else epoch,
            pop_size=self.population_size,
            **extra_kwargs,
        )

    @staticmethod
    def _scalar(fitness):
        """Collapse multi-objective fitness to a scalar (sum)."""
        if isinstance(fitness, (list, tuple)):
            return sum(fitness)
        return float(fitness)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _measure_cost(self, mealpy_problem) -> tuple:
        """
        Return ``(init_cost, per_epoch_cost)`` in objective evaluations.

        Evaluations used by an algorithm are well modelled by
        ``total(e) = init + e * per_epoch``: a one-off initialisation (usually
        the starting population) plus a fixed cost per epoch. Both terms are
        recovered from two short probe runs, which cancels the initialisation
        term -- a single probe would wrongly fold ``init`` into the per-epoch
        rate, badly mis-sizing single-solution methods such as SA (measured 26
        evaluations/epoch instead of the true 1).

        Results are cached per (algorithm, pop_size) so the probe cost is paid
        once per process, not once per run.
        """
        cache_key = (self.algorithm_key, self.population_size)
        if cache_key in _COST_CACHE:
            return _COST_CACHE[cache_key]

        e1, e2 = 2, 6
        counts = []
        base_obj = mealpy_problem["obj_func"]

        for epochs in (e1, e2):
            count = {"n": 0}

            def counting_obj(solution, _c=count):
                _c["n"] += 1
                return base_obj(solution)

            probe = dict(mealpy_problem, obj_func=counting_obj)
            self._make_model(epoch=epochs).solve(probe, seed=self.seed)
            counts.append(count["n"])

        per_epoch = max(1.0, (counts[1] - counts[0]) / (e2 - e1))
        init = max(0.0, counts[0] - e1 * per_epoch)
        _COST_CACHE[cache_key] = (init, per_epoch)
        return init, per_epoch

    def _epochs_for_budget(self, mealpy_problem) -> int:
        """
        Return the epoch count whose natural length matches ``max_fe``.

        Simply raising the epoch ceiling is NOT a valid way to equalise
        budgets: many algorithms (GWO, WOA, SCA, MPA, ...) use the epoch count
        as a *schedule length*, decaying an exploration parameter across it.
        Told they have far more epochs than they will actually run, they never
        transition to exploitation and perform dramatically worse.

        The probe runs are independently seeded, so they do not affect the
        reproducibility of the real run that follows. Probe evaluations are
        measurement overhead and are not counted against ``max_fe``, which is
        the search budget.
        """
        init, per_epoch = self._measure_cost(mealpy_problem)
        epochs = (self.max_fe - init) / per_epoch
        return max(1, min(int(round(epochs)), 100000))

    def run(self, problem):
        """
        Run the wrapped mealpy algorithm on *problem*.

        Parameters
        ----------
        problem : OptimizationProblem

        Returns
        -------
        (best_solution, best_fitness) : (np.ndarray, float)
        """
        from mealpy import FloatVar  # local import keeps mealpy optional at module load

        lb, ub = problem.bounds()

        def _obj(solution):
            return self._scalar(problem.evaluate(solution))

        mealpy_problem = {
            "obj_func": _obj,
            "bounds": FloatVar(lb=tuple(lb), ub=tuple(ub)),
            "minmax": "min",
            "log_to": None,   # suppress mealpy console output
        }

        termination = None
        epoch = None
        if self.max_fe is not None:
            from mealpy.utils.termination import Termination
            termination = Termination(max_fe=self.max_fe)
            epoch = self._epochs_for_budget(mealpy_problem)

        model = self._make_model(epoch=epoch)

        # mealpy seeds its own Generator from solve(seed=...), but three of its
        # algorithms leak: JADE (DE.py:248), SHADE (SHADE.py:100) and L_SHADE
        # (SHADE.py:257) draw their F parameter with scipy's
        # ``cauchy.rvs(...)``, passing no random_state. scipy falls back to
        # NumPy's *global* RNG, which mealpy's seed never touches, so those
        # three produce different results on every run despite being seeded.
        # Seeding the global RNG here closes that leak. It must happen after
        # _epochs_for_budget, whose probe runs consume from the same stream.
        if self.seed is not None:
            np.random.seed(self.seed)

        g_best = model.solve(
            mealpy_problem, termination=termination, seed=self.seed
        )

        # Convergence history: one float per epoch
        self.history = [float(v) for v in model.history.list_global_best_fit]

        return np.asarray(g_best.solution, dtype=float), float(g_best.target.fitness)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"MealpyAlgorithm(alg={self.algorithm_key}, "
            f"pop={self.population_size}, "
            f"epochs={self.max_generations})"
        )
