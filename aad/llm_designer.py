"""
llm_designer.py
---------------
LLM-based (or rule-based fallback) algorithm proposer.

:class:`LLMDesigner` generates an EA configuration given a plain-text
description of the optimisation problem.

Two modes
---------
LLM mode   : uses the Google Gemini API when an *api_key* is supplied.
Rule-based : a deterministic heuristic mapping of problem keywords to
             configuration choices (always available, no API key needed).

The rule-based fallback is useful for testing, demos, and offline use.
"""

import json
import random
import os
from dotenv import load_dotenv

from .config_space import EXTENDED_CONFIG_SPACE

load_dotenv()  # Load environment variables from .env file

class LLMDesigner:
    """
    Propose EA configurations from natural-language problem descriptions.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key.  If ``None`` (default) the rule-based fallback
        is used automatically.
    model : str, default ``"gemini-3.6-flash"``
        Gemini model to use (only relevant when *api_key* is provided).
    config_space : dict, optional
        The config space to constrain the proposals.  Defaults to
        :data:`~aad.config_space.EXTENDED_CONFIG_SPACE`, so a bare
        ``LLMDesigner()`` proposes across every registered algorithm.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-3.6-flash",
        config_space: dict = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.config_space = config_space or EXTENDED_CONFIG_SPACE
        self._use_llm = bool(self.api_key)

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
            from google import genai

            client = genai.Client(api_key=self.api_key)

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
                "  algorithm, population_size, selection, crossover, mutation, max_generations\n"
                "Each value must be one of the options listed in the configuration space above.\n"
                "Return ONLY the JSON, no explanation."
            )

            response = client.models.generate_content(model=self.model, contents=prompt)
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
