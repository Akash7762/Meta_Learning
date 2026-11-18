"""
meta_learner.py
---------------
EA_Designer – a meta-optimiser that searches over the space of
Evolutionary Algorithm configurations to find the best EA for a
given optimisation problem.

Strategy: Evolution of EAs  (a (μ + λ)-style search in config space)
  1. Initialise a population of EA configurations.
  2. Evaluate each by running the EA on the target problem N times.
  3. Evolve the meta-population through crossover + mutation.
  4. Optionally use a surrogate model to pre-screen candidates.

Meta-level convergence and configuration history are tracked for analysis.
"""

import random

import numpy as np

from .config_space import EXTENDED_CONFIG_SPACE, random_config, config_to_vector
from .evolutionary_algorithm import build_algorithm
from .landscape_features import LandscapeFeatureExtractor
from .surrogate_model import SurrogateModel

# sentinel assigned to a run that returned NaN/Inf, so a single
# numerically unstable trial (e.g. an adaptive-DE variant collapsing)
# is heavily penalised rather than corrupting the surrogate's training
# data or the meta-population's fitness ranking.
_FITNESS_PENALTY = 1e12

def _sanitize_fitness(value: float) -> float:
    """Return *value* unchanged, or a large finite penalty if it is NaN/Inf."""
    return float(value) if np.isfinite(value) else _FITNESS_PENALTY


class EA_Designer:
    """
    Meta-optimiser that searches over EA design configurations.

    Parameters
    ----------
    config_space : dict, optional
        Configuration space to search over.  Defaults to
        :data:`~aad.config_space.EXTENDED_CONFIG_SPACE`, i.e. selection
        across every registered algorithm.  Pass a space pinned to
        ``algorithm=["ea"]`` for the secondary operator-tuning experiment.
    population_size : int, default 10
        Number of EA configurations in the meta-population.
    generations : int, default 5
        Number of meta-evolution generations.
    runs_per_eval : int, default 3
        Independent EA runs per configuration (for stable performance
        estimates).
    use_surrogate : bool, default True
        If True, a Random Forest surrogate is used to pre-screen
        candidate configurations, reducing expensive evaluations.
    surrogate_type : str, default ``"rf"``
        Surrogate backend passed to :class:`~aad.surrogate_model.SurrogateModel`.
    evaluator : callable, optional
        ``evaluator(config, problem) -> float``, the raw best fitness of ONE
        run. Default ``None`` runs ``build_algorithm(config).run(problem)``.
        v1 Phase 4 passes a replay evaluator that hands out stored matrix runs.
    max_runs : int, optional
        Run budget for each :meth:`evolve` call. ``None`` (default) keeps the
        fixed schedule: ``population_size`` initial candidates, then
        ``generations`` generations. With a budget, evolution stops as soon
        as the next candidate cannot be paid for, keeps going while runs
        remain (``generations`` becomes a cap), and runs still left after
        the cap go to the current best configuration. The whole budget is
        spent, up to a remainder smaller than ``runs_per_eval``. A candidate
        the surrogate skips costs nothing.
    use_features : bool, default True
        Extract landscape features and feed them to the surrogate. Inside
        one problem every candidate gets the same feature values, so they
        say nothing about which candidate is better; ``False`` skips the
        extraction and the function evaluations it spends.
    surrogate_n_jobs : int, default -1
        Threads for the surrogate's Random Forest. ``1`` makes it
        bit-for-bit repeatable (see :class:`~aad.surrogate_model.SurrogateModel`).

    Attributes
    ----------
    meta_convergence : list[float]
        Best EA performance at each meta-generation (lower is better).
    best_config_history : list[dict]
        Best configuration at each meta-generation.
    run_log : list[dict]
        Every run of the last :meth:`evolve` call, in order:
        ``{"config": dict, "fitness": float}``, raw (not sanitised).
    runs_used : int
        Runs performed in the last :meth:`evolve` call.
    leftover_runs : int
        Of those, runs spent on the best configuration after the
        generation cap (budgeted calls only).
    """

    def __init__(
        self,
        config_space: dict = None,
        population_size: int = 10,
        generations: int = 5,
        runs_per_eval: int = 3,
        use_surrogate: bool = True,
        surrogate_type: str = "rf",
        evaluator=None,
        max_runs: int = None,
        use_features: bool = True,
        surrogate_n_jobs: int = -1,
    ):
        if max_runs is not None and max_runs < runs_per_eval:
            raise ValueError(
                f"max_runs={max_runs} cannot pay for a single candidate "
                f"(runs_per_eval={runs_per_eval})."
            )
        self.config_space = config_space or EXTENDED_CONFIG_SPACE
        self.population_size = population_size
        self.generations = generations
        self.runs_per_eval = runs_per_eval
        self.use_surrogate = use_surrogate
        self.evaluator = evaluator
        self.max_runs = max_runs
        self.use_features = use_features

        self._surrogate = (
            SurrogateModel(model_type=surrogate_type, n_jobs=surrogate_n_jobs)
            if use_surrogate else None
        )
        self._feature_extractor = LandscapeFeatureExtractor()

        # Memory for surrogate fitting: (combined_feature_vector, performance)
        self._mem_X: list = []
        self._mem_y: list = []
        self._last_prediction: float = None

        # Meta-level tracking
        self.meta_convergence: list = []
        self.best_config_history: list = []

        # Run accounting, reset at the start of every evolve() call
        self.run_log: list = []
        self.runs_used: int = 0
        self.leftover_runs: int = 0

    # ------------------------------------------------------------------
    # Candidate generation / variation
    # ------------------------------------------------------------------

    def generate_candidate(self) -> dict:
        """Sample a random EA configuration from the config space."""
        return random_config(self.config_space)

    def _mutate_config(self, config: dict) -> dict:
        """Mutate one randomly selected parameter."""
        new_config = config.copy()
        key = random.choice(list(self.config_space.keys()))
        new_config[key] = random.choice(self.config_space[key])
        return new_config

    def _crossover_configs(self, config1: dict, config2: dict) -> dict:
        """Uniform crossover: each parameter inherited from parent1 or parent2."""
        return {
            key: (config1[key] if random.random() < 0.5 else config2[key])
            for key in self.config_space
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_candidate(self, ea_config: dict, problem) -> float:
        """
        Run the EA described by *ea_config* on *problem* for
        *runs_per_eval* independent trials and return the mean best fitness.
        Every trial is counted in :attr:`runs_used` and logged in :attr:`run_log`.
        """
        scores = []
        for _ in range(self.runs_per_eval):
            scores.append(_sanitize_fitness(self._run_once(ea_config, problem)))
        return float(np.mean(scores))

    def _run_once(self, ea_config: dict, problem):
        """One run: the evaluator hook if set, otherwise a real optimiser run."""
        if self.evaluator is None:
            ea = build_algorithm(ea_config)
            _, best_fit = ea.run(problem)
        else:
            best_fit = self.evaluator(ea_config, problem)
        self.runs_used += 1
        self.run_log.append({"config": dict(ea_config), "fitness": float(best_fit)})
        return best_fit

    def _can_afford(self) -> bool:
        """True if one more candidate fits the run budget (always, without one)."""
        return (self.max_runs is None
                or self.runs_used + self.runs_per_eval <= self.max_runs)

    def _should_evaluate(
        self,
        config: dict,
        feature_vec: np.ndarray,
        meta_fitnesses: list,
    ) -> bool:
        """
        Return True if the candidate should be fully evaluated.

        If the surrogate is fitted, only configurations whose predicted
        performance is below the median of the current meta-population are
        evaluated in full; others are rejected cheaply.
        """
        if self._surrogate is None or not self._surrogate.is_fitted:
            return True

        config_vec = config_to_vector(config, self.config_space)
        combined = np.concatenate([feature_vec, config_vec])
        predicted = float(self._surrogate.predict(combined)[0])
        self._last_prediction = predicted       # reused if the candidate is skipped
        threshold = float(np.median(meta_fitnesses))
        return predicted <= threshold

    def _update_surrogate(self) -> None:
        """Re-fit the surrogate on all accumulated memory."""
        if (
            self._surrogate is not None
            and len(self._mem_X) >= 5
        ):
            self._surrogate.fit(self._mem_X, self._mem_y)

    # ------------------------------------------------------------------
    # Main meta-evolution loop
    # ------------------------------------------------------------------

    def evolve(self, problem, verbose: bool = True) -> tuple:
        """
        Run the meta-evolution loop to find the best EA for *problem*.

        Parameters
        ----------
        problem : OptimizationProblem
        verbose : bool, default True

        Returns
        -------
        (best_config, best_performance) : (dict, float)
            ``best_performance`` is the elite's score, a mean over its runs.
            For a budgeted comparison, score from :attr:`run_log`, which
            holds every run that really happened.
        """
        if verbose:
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  AAD Meta-Learning  |  Problem: {problem.name}")
            print(sep)

        # Run accounting for this call
        self.run_log = []
        self.runs_used = 0
        self.leftover_runs = 0

        # --- Extract problem features ---
        if self.use_features:
            if verbose:
                print("  Extracting landscape features …")
            features = self._feature_extractor.extract_features(problem)
            feature_vec = self._feature_extractor.features_to_vector(features)

            if verbose:
                print(
                    f"  dim={features['dimension']:.0f}  "
                    f"variance={features['fitness_variance']:.3f}  "
                    f"modality={features['modality']:.3f}  "
                    f"ruggedness={features['ruggedness']:.3f}"
                )
        else:
            # Every candidate of this problem would get the same values.
            feature_vec = np.empty(0)
            if verbose:
                print("  Landscape features skipped (use_features=False)")

        # --- Initialise meta-population ---
        meta_pop = [self.generate_candidate() for _ in range(self.population_size)]

        if verbose:
            print(
                f"\n  Evaluating initial meta-population "
                f"({self.population_size} configs) …"
            )

        meta_fits: list = []
        for cfg in meta_pop:
            if not self._can_afford():
                break
            perf = self.evaluate_candidate(cfg, problem)
            meta_fits.append(perf)
            cfg_vec = config_to_vector(cfg, self.config_space)
            self._mem_X.append(list(feature_vec) + cfg_vec)
            self._mem_y.append(perf)
        # A run budget smaller than the population: keep only the evaluated.
        meta_pop = meta_pop[: len(meta_fits)]

        self._update_surrogate()

        best_idx = int(np.argmin(meta_fits))
        best_cfg, best_perf = meta_pop[best_idx], meta_fits[best_idx]
        self.meta_convergence.append(best_perf)
        self.best_config_history.append(best_cfg.copy())

        if verbose:
            print(f"  Gen 0  best={best_perf:.6f}  cfg={best_cfg}")

        # --- Meta-evolution ---
        # Without a run budget: exactly `generations` generations. With one,
        # `generations` is a cap and the loop also stops as soon as the next
        # candidate cannot be paid for.
        gen = 0
        while gen < self.generations and self._can_afford():
            gen += 1
            new_pop: list = []
            new_fits: list = []

            for _ in range(self.population_size):
                if not self._can_afford():
                    break
                # Variation
                p1, p2 = random.choices(meta_pop, k=2)
                child = self._crossover_configs(p1, p2)
                child = self._mutate_config(child)

                # Surrogate pre-screening
                if self._should_evaluate(child, feature_vec, meta_fits):
                    perf = self.evaluate_candidate(child, problem)
                else:
                    # The prediction _should_evaluate just made (the code used
                    # to ask the unchanged surrogate again), made 5% worse
                    # (lower is better). A plain "* 1.05" made a NEGATIVE
                    # prediction look better, so a never-run candidate could
                    # become the elite with a made-up score. BBOB values do
                    # go negative.
                    pred = self._last_prediction
                    perf = pred * 1.05 if pred >= 0 else pred * 0.95

                new_pop.append(child)
                new_fits.append(perf)

                # Store in memory for surrogate
                cfg_vec = config_to_vector(child, self.config_space)
                self._mem_X.append(list(feature_vec) + cfg_vec)
                self._mem_y.append(perf)

            # Elitist (μ + λ) selection
            combined_pop = meta_pop + new_pop
            combined_fits = meta_fits + new_fits
            top_idxs = np.argsort(combined_fits)[: self.population_size]
            meta_pop = [combined_pop[i] for i in top_idxs]
            meta_fits = [combined_fits[i] for i in top_idxs]

            self._update_surrogate()

            best_cfg, best_perf = meta_pop[0], meta_fits[0]
            self.meta_convergence.append(best_perf)
            self.best_config_history.append(best_cfg.copy())

            if verbose:
                print(f"  Gen {gen}  best={best_perf:.6f}  cfg={best_cfg}")

        # --- Leftover budget ---
        # The generation cap was reached with runs unspent: spend them on the
        # current best configuration, so every budgeted call uses its budget.
        if self.max_runs is not None:
            while self._can_afford():
                self.evaluate_candidate(best_cfg, problem)
                self.leftover_runs += self.runs_per_eval

        if verbose:
            print(f"\n  [OK] Best config : {best_cfg}")
            print(f"  [OK] Performance : {best_perf:.6f}")
            print("=" * 60)

        return best_cfg, best_perf
