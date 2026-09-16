import os
import sys
import warnings
import random
from dotenv import load_dotenv


# Terminal printing UTF-8 support
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress library deprecation / future warnings
warnings.filterwarnings("ignore")

# Load API keys from .env
load_dotenv()

from aad import (
    EA_Designer,
    LLMDesigner,
    EvaluationFramework,
    EXTENDED_CONFIG_SPACE,
    BBOB_F1_Sphere,
    BBOB_F5_LinearSlope,
    BBOB_F8_Rosenbrock,
    BBOB_F15_RastriginRotated,
    BBOB_F21_Gallagher101
)

def run_full_bbob_random_test():
    """
    Runs the full AAD Meta-Learning + LLM Evaluation pipeline 
    on 3 randomly selected BBOB functions for a complete test output.
    """
    # 5 different problems landscapes list
    available_problems = [
        BBOB_F1_Sphere(dim=10),
        BBOB_F5_LinearSlope(dim=10),
        BBOB_F8_Rosenbrock(dim=10),
        BBOB_F15_RastriginRotated(dim=10),
        BBOB_F21_Gallagher101(dim=10)
    ]
    
    # 3 random functions selection for testing
    selected_problems = random.sample(available_problems, 3)
    
    # Tools setup
    evaluator = EvaluationFramework(n_runs=2)
    llm_designer = LLMDesigner(config_space=EXTENDED_CONFIG_SPACE)
    
    for problem in selected_problems:
        pname = problem.name
        dim = problem.dimension
        is_mo = problem.is_multiobjective()

        # Print Top Header
        print(f"\n{'#' * 70}")
        print(f"#  Problem : {pname}  (dim={dim}, multi-obj={is_mo})")
        print(f"{'#' * 70}")

        # 1. Meta-learning search (EA_Designer)
        designer = EA_Designer(
            config_space=EXTENDED_CONFIG_SPACE,
            population_size=20,
            generations=5,
            runs_per_eval=2,
            use_surrogate=True,
        )

        
        best_config, best_perf = designer.evolve(problem, verbose=True)

        # 2. LLM Designer (Description & Proposal)
        desc = llm_designer.generate_description(problem)
        llm_config = llm_designer.propose_algorithm(desc)
        print(f"\n  [LLM Designer] description  : {desc}")
        print(f"  [LLM Designer] proposed cfg : {llm_config}")
        
        
        llm_metrics = evaluator.evaluate(llm_config, problem, verbose=False)
        print(f"  [LLM Designer] best fitness : {llm_metrics['best_fitness']:.6e}")

        # 3. Full evaluation of best config (EA Designer wala winner)
        print(f"\n  [Evaluation] Full evaluation of best config …")
        
        # verbose=True se "run 1/2: best=..." walo lines print hongi
        full_metrics = evaluator.evaluate(best_config, problem, verbose=True)

        # Print Final Results summary
        print(f"\n  [Results] {pname}")
        EvaluationFramework.summarize(full_metrics)

def test_algorithm_key_present_and_valid():
    """
    Regression test: LLMDesigner._rule_based_proposal must always emit
    an 'algorithm' key that is a valid member of EXTENDED_CONFIG_SPACE,
    and build_algorithm must reject configs missing that key.
    """
    from aad import EXTENDED_CONFIG_SPACE
    from aad.evolutionary_algorithm import build_algorithm

    llm_designer = LLMDesigner(config_space=EXTENDED_CONFIG_SPACE)
    desc = llm_designer.generate_description(BBOB_F1_Sphere(dim=10))
    llm_config = llm_designer.propose_algorithm(desc)

    assert "algorithm" in llm_config, "LLM designer output missing 'algorithm' key"
    assert llm_config["algorithm"] in EXTENDED_CONFIG_SPACE["algorithm"], (
        f"'{llm_config['algorithm']}' not in EXTENDED_CONFIG_SPACE['algorithm']"
    )

    try:
        build_algorithm({"population_size": 20, "max_generations": 10})
        raise AssertionError("build_algorithm should have raised on missing 'algorithm' key")
    except ValueError:
        pass

    print("  [OK] test_algorithm_key_present_and_valid passed")


if __name__ == "__main__":
    test_algorithm_key_present_and_valid()
    run_full_bbob_random_test()
