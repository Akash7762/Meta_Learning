"""
llm_designer.py
---------------
LLM-based (or rule-based fallback) algorithm proposer.

:class:`LLMDesigner` generates an EA configuration given a plain-text
description of the optimisation problem.

Two modes
---------
LLM mode   : uses the Google Gemini API, only when an *api_key* is passed
             explicitly (or the instance is built with
             :meth:`LLMDesigner.from_env`).
Rule-based : a deterministic heuristic mapping of problem keywords to
             configuration choices (always available, no API key needed).

The rule-based fallback is useful for testing, demos, and offline use.

``load_dotenv()`` still runs at import, so callers that pass
``api_key=os.getenv("GEMINI_API_KEY")`` themselves keep working. The class
never reads the environment on its own: ``api_key=None`` means offline,
whatever ``.env`` holds.

Phase 5 uses :meth:`LLMDesigner.select_algorithm`: one call that must name one
of the configured algorithms, with fixed generation settings, and that returns
an unusable answer as unusable instead of replacing it.
"""

import json
import random
import os
from dotenv import load_dotenv

from .config_space import EXTENDED_CONFIG_SPACE

load_dotenv()  # Load environment variables from .env file

DEFAULT_MODEL = "gemini-3.6-flash"

# Full names, from the docstrings of the installed mealpy classes
# (MEALPY_REGISTRY in aad/mealpy_algorithms.py) and, for "ea", from the
# operators the matrices were built with (phase2_build_matrix.EA_OPERATORS).
ALGORITHM_NAMES = {
    "ea": "this project's own evolutionary algorithm (tournament selection, "
          "SBX crossover, Gaussian mutation)",
    "GA": "Genetic Algorithm",
    "JADE": "JADE, an adaptive Differential Evolution",
    "SHADE": "Success-History Adaptation Differential Evolution",
    "L_SHADE": "Linear Population Size Reduction Success-History Adaptation "
               "Differential Evolution",
    "PSO": "Particle Swarm Optimization",
    "GWO": "Grey Wolf Optimizer",
    "WOA": "Whale Optimization Algorithm",
    "HHO": "Harris Hawks Optimization",
    "ABC": "Artificial Bee Colony",
    "GJO": "Golden Jackal Optimization",
    "ARO": "Artificial Rabbits Optimization",
    "AO": "Aquila Optimization",
    "MPA": "Marine Predators Algorithm",
    "HBA": "Honey Badger Algorithm",
    "SA": "Simulated Annealing",
    "EO": "Equilibrium Optimizer",
    "MVO": "Multi-Verse Optimizer",
    "RIME": "RIME, an optimizer inspired by the physics of rime ice",
    "SMA": "Slime Mould Algorithm",
    "BBO": "Biogeography-Based Optimization",
    "SCA": "Sine Cosine Algorithm",
}

# What each stored landscape feature really is, from
# aad/landscape_features.py. Some names mislead on their own: "modality" is a
# skewness and "fitness_variance" a coefficient of variation.
FEATURE_DEFINITIONS = {
    "fitness_mean": "mean of the sampled objective values",
    "fitness_std": "standard deviation of the sampled values",
    "fitness_variance": "coefficient of variation: standard deviation / |mean|",
    "fitness_min": "lowest sampled value",
    "fitness_max": "highest sampled value",
    "gradient_magnitude": "mean norm of a forward finite-difference gradient "
                          "(step 1e-5 of the box width) at the first 50 "
                          "sampled points",
    "modality": "skewness of the sampled values",
    "ruggedness": "mean absolute difference between consecutive sampled "
                  "values, in sampling order, divided by their standard "
                  "deviation",
    "dimension": "number of variables",
}


class LLMDesigner:
    """
    Propose EA configurations from natural-language problem descriptions.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key. If ``None`` (default) the rule-based fallback
        is used, whatever the environment or ``.env`` holds. To use the key
        from the environment, call :meth:`from_env`.
    model : str, default :data:`DEFAULT_MODEL`
        Gemini model to use (only relevant when *api_key* is provided).
    config_space : dict, optional
        The config space to constrain the proposals.  Defaults to
        :data:`~aad.config_space.EXTENDED_CONFIG_SPACE`, so a bare
        ``LLMDesigner()`` proposes across every registered algorithm.
    temperature : float, optional
        Sampling temperature sent to the API. ``None`` keeps the model's
        default.
    seed : int, optional
        Seed sent to the API. Gemini treats it as "a best effort to provide
        the same response for repeated requests", not a guarantee.
    timeout_s : float, default 120
        Time limit for one HTTP request. Without one, a stalled connection
        can block forever (seen in the first Phase 5 test call, 2026-10-08).
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = DEFAULT_MODEL,
        config_space: dict = None,
        temperature: float = None,
        seed: int = None,
        timeout_s: float = 120.0,
    ):
        self.api_key = api_key
        self.model = model
        self.config_space = config_space or EXTENDED_CONFIG_SPACE
        self.temperature = temperature
        self.seed = seed
        self.timeout_s = timeout_s
        self._use_llm = bool(self.api_key)
        self._client = None

    @classmethod
    def from_env(cls, var: str = "GEMINI_API_KEY", **kwargs) -> "LLMDesigner":
        """LLM mode with the key from the environment (``.env`` is loaded at
        import). The explicit way to switch the LLM on; never prints the key."""
        key = os.getenv(var)
        if not key:
            raise RuntimeError(f"{var} is not set in the environment or .env.")
        return cls(api_key=key, **kwargs)

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        return self._client

    def _generation_config(self, **extra):
        from google.genai import types
        return types.GenerateContentConfig(
            temperature=self.temperature, seed=self.seed, **extra)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_proposal(self, description: str) -> dict:
        """
        Derive an EA configuration from keyword heuristics in *description*.

        Heuristics
        ----------
        - High-dimensional / complex → larger population
        - Multimodal / noisy → tournament selection
        - Combinatorial / discrete → uniform crossover
        - Smooth / unimodal → polynomial mutation
        """
        desc = description.lower()
        config: dict = {}

        # --- Algorithm ---
        algorithm_options = self.config_space.get("algorithm", ["ea"])
        if any(kw in desc for kw in ("multimodal", "noisy", "rugged")):
            preferred = [a for a in algorithm_options if a in ("GWO", "WOA", "HHO", "SA", "GA")]
        elif any(kw in desc for kw in ("smooth", "unimodal", "convex")):
            preferred = [a for a in algorithm_options if a in ("DE", "L_SHADE", "CMA_ES", "ea")]
        else:
            preferred = algorithm_options
        config["algorithm"] = random.choice(preferred or algorithm_options)

        # --- Population size ---
        if any(kw in desc for kw in ("high-dimensional", "complex", "large")):
            config["population_size"] = 100
        elif any(kw in desc for kw in ("simple", "low-dimensional", "small")):
            config["population_size"] = 20
        else:
            config["population_size"] = 50

        # --- Selection ---
        if any(kw in desc for kw in ("multimodal", "noisy", "rugged")):
            config["selection"] = "tournament"
        elif any(kw in desc for kw in ("smooth", "unimodal", "convex")):
            config["selection"] = "roulette"
        else:
            config["selection"] = random.choice(["tournament", "roulette"])

        # --- Crossover ---
        if any(kw in desc for kw in ("combinatorial", "discrete", "binary")):
            config["crossover"] = "uniform"
        else:
            config["crossover"] = "sbx"

        # --- Mutation ---
        if any(kw in desc for kw in ("smooth", "unimodal", "polynomial")):
            config["mutation"] = "polynomial"
        else:
            config["mutation"] = "gaussian"

        # --- Generations ---
        if any(kw in desc for kw in ("high-dimensional", "complex", "hard")):
            config["max_generations"] = 200
        elif any(kw in desc for kw in ("simple", "easy", "fast")):
            config["max_generations"] = 50
        else:
            config["max_generations"] = 100

        return config

    # ------------------------------------------------------------------
    # LLM mode
    # ------------------------------------------------------------------

    def _llm_proposal(self, description: str) -> dict:
        """
        Query the Google Gemini API to propose an EA configuration.

        Falls back to rule-based if the API call or JSON parsing fails.
        """
        try:
            client = self._get_client()

            # Enumerate valid values for each parameter
            space_str = json.dumps(
                {k: v for k, v in self.config_space.items()}, indent=2
            )

            prompt = (
                "You are an expert in evolutionary computation.\n"
                "Given the following optimisation problem description, "
                "suggest the best algorithm configuration.\n\n"
                f"Problem: {description}\n\n"
                f"Valid configuration space:\n{space_str}\n\n"
                "Return a JSON object with exactly these keys:\n"
                f"  {', '.join(self.config_space)}\n"
                "Each value must be one of the options listed in the configuration space above.\n"
                "Return ONLY the JSON, no explanation."
            )

            response = client.models.generate_content(
                model=self.model, contents=prompt,
                config=self._generation_config())
            content = response.text.strip()
            # Strip any markdown code fences
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            config = json.loads(content)

            # Validate that all required keys are present
            for key in self.config_space:
                if key not in config:
                    raise ValueError(f"LLM response missing key: '{key}'")

            return config

        except Exception as exc:  # noqa: BLE001
            print(
                f"[LLMDesigner] API call failed ({exc}). "
                "Falling back to rule-based proposal."
            )
            return self._rule_based_proposal(description)

    # ------------------------------------------------------------------
    # Algorithm selection (Phase 5)
    # ------------------------------------------------------------------

    @staticmethod
    def describe_for_selection(dim: int, lower: float, upper: float,
                               budget_fe: int, population: int,
                               name: str = None, features: dict = None,
                               n_samples: int = None) -> str:
        """Problem text for :meth:`select_algorithm`. ``name`` and
        ``features`` are optional, so the same wording serves every
        Phase 5 arm; a feature block needs ``n_samples``."""
        lines = []
        if name is not None:
            lines.append(f"Problem name: {name}.")
        lines.append("Single-objective, continuous, minimisation.")
        lines.append(f"Dimension: {dim}. Search box: [{lower:.2f}, {upper:.2f}] "
                     "in every coordinate.")
        lines.append(f"Budget: one run of {budget_fe} function evaluations, "
                     f"population size {population}.")
        if features is not None:
            lines.append(f"Landscape features, computed from {n_samples} points "
                         "sampled uniformly at random in the search box:")
            for key, meaning in FEATURE_DEFINITIONS.items():
                lines.append(f"- {key} = {features[key]:.4g}: {meaning}")
        return "\n".join(lines)

    def selection_prompt(self, problem_text: str) -> str:
        """The full prompt sent by :meth:`select_algorithm`."""
        listing = "\n".join(f"- {a}: {ALGORITHM_NAMES.get(a, a)}"
                            for a in self.config_space["algorithm"])
        return (
            "You are an expert in evolutionary computation and black-box "
            "optimisation.\n"
            "Choose the one algorithm from the list below that is most likely "
            "to reach the lowest objective value on the problem described.\n\n"
            f"Problem:\n{problem_text}\n\n"
            f"Algorithms:\n{listing}\n\n"
            'Answer with JSON only, in the form {"algorithm": "<one name from '
            'the list>"}.'
        )

    @staticmethod
    def parse_selection(text: str, algorithms) -> tuple:
        """``(name, None)`` for a usable answer, ``(None, reason)`` otherwise.
        Only an exact name from *algorithms* is usable."""
        s = (text or "").strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:]
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return None, "not JSON"
        if not isinstance(obj, dict) or "algorithm" not in obj:
            return None, "no 'algorithm' key"
        name = obj["algorithm"]
        if not isinstance(name, str) or name not in algorithms:
            return None, f"not one of the {len(algorithms)} algorithms: {name!r}"
        return name, None

    def select_algorithm(self, problem_text: str) -> dict:
        """
        One API call that must name one configured algorithm.

        Sent with this instance's temperature and seed, and a JSON schema that
        only allows the configured names. Network and API errors are raised to
        the caller. An unusable answer comes back with ``valid=False`` and is
        never replaced by a rule-based pick.
        """
        if not self._use_llm:
            raise RuntimeError("select_algorithm needs LLM mode: pass api_key "
                               "or use LLMDesigner.from_env().")
        from google.genai import types
        algorithms = list(self.config_space["algorithm"])
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={"algorithm": types.Schema(type=types.Type.STRING,
                                                  enum=algorithms)},
            required=["algorithm"])
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=self.selection_prompt(problem_text),
            config=self._generation_config(response_mime_type="application/json",
                                           response_schema=schema))
        try:
            text = response.text or ""
            text_error = None
        except Exception as exc:  # noqa: BLE001  (blocked or empty answer)
            text, text_error = "", f"no text: {exc}"
        name, error = self.parse_selection(text, algorithms)
        usage = getattr(response, "usage_metadata", None)
        return {
            "raw": text,
            "algorithm": name,
            "valid": name is not None,
            "error": text_error or error,
            "model_version": getattr(response, "model_version", None),
            "tokens": None if usage is None else {
                k: getattr(usage, k, None)
                for k in ("prompt_token_count", "candidates_token_count",
                          "thoughts_token_count", "total_token_count")},
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def propose_algorithm(self, problem_description: str) -> dict:
        """
        Return an EA configuration dict for the given *problem_description*.

        Uses the LLM if an API key was supplied, otherwise uses the
        rule-based heuristic.

        Parameters
        ----------
        problem_description : str

        Returns
        -------
        dict – a valid EA configuration
        """
        if self._use_llm:
            return self._llm_proposal(problem_description)
        return self._rule_based_proposal(problem_description)

    def generate_description(self, problem) -> str:
        """
        Auto-generate a natural-language description of *problem* that
        can be fed directly to :meth:`propose_algorithm`.

        Parameters
        ----------
        problem : OptimizationProblem

        Returns
        -------
        str
        """
        lb, ub = problem.bounds()
        dim = len(lb)
        is_mo = problem.is_multiobjective()

        parts = [f"{problem.name} optimisation problem."]
        parts.append(f"Dimension: {dim}.")
        parts.append(f"Search bounds: [{lb[0]:.2f}, {ub[0]:.2f}].")

        if is_mo:
            parts.append("Multi-objective continuous optimisation problem.")
        else:
            parts.append("Single-objective continuous optimisation problem.")

        if dim > 20:
            parts.append(
                "High-dimensional problem that likely requires a larger "
                "population and more generations."
            )
        elif dim <= 5:
            parts.append("Low-dimensional, potentially simple structure.")

        return " ".join(parts)
