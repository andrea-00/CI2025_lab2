# scripts/03_run_showcase.py

"""
Prerequisites:
    - '01_generate_baseline.py' must be run.
    - '02_run_tuning.py' must be run.
    - '01_tuning_analysis.ipynb' must be analyzed to find the
      'WINNING_STRATEGY_NAME' (which you must set below).
"""

import numpy as np
import glob
import json
import time
import os
import sys
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Path Setup
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Framework Imports
from evolvepy import EvolutionaryAlgorithm, Logger, LogLevel
from evolvepy.strategies import (
    TournamentSelection, PlusSelection, UniformSelection,
    StandardReproduction, MutationOnlyReproduction,
    OrderedCrossover, CycleCrossover,
    SwapMutation, InversionMutation
)
from problems.tsp import TSPProblem
from utils.strategy_builder import StrategyGridBuilder

# Experiment Configuration

# !!! ATTENTION !!!
# You must set this variable manually after analyzing the
# results from '01_tuning_analysis.ipynb'.
WINNING_STRATEGY_NAME = "Shape_Deep_Tourn_k7_StdRep_OX_0.5_Invert_0.8_Elitism"

# Other Parameters
N_STATISTICAL_RUNS = 10  # Use a higher number for the final showcase
TOTAL_EVALUATION_BUDGET = 20000
POPULATION_CONFIGS = [
    {"name": "Shape_Broad", "pop_size": 100},
    {"name": "Shape_Medium", "pop_size": 50},
    {"name": "Shape_Deep", "pop_size": 25},
]

# I/O Configuration
BASELINE_FILE = os.path.join(project_root, "data/baseline_hc.json")
PROBLEM_DATA_DIR = os.path.join(project_root, "data/tsp_problems")
SHOWCASE_OUTPUT_FILE = os.path.join(project_root, "experiments/showcase_results.json")


# Strategy Component Definitions
# This MUST be identical to the definitions in '02_run_tuning.py'
# to ensure the StrategyGridBuilder reconstructs the exact same recipes.
PARENT_SELECTORS = [
    {"name": "Tourn_k7", "config": TournamentSelection(k=7)},
    {"name": "Tourn_k5", "config": TournamentSelection(k=5)},
    #{"name": "Uniform", "config": UniformSelection()}
]

CROSSOVER_OPS = [
    {"name": "OX_0.2", "config": OrderedCrossover(crossover_rate=0.2)},
    {"name": "CX_0.2", "config": CycleCrossover(crossover_rate=0.2)},
    {"name": "OX_0.5", "config": OrderedCrossover(crossover_rate=0.5)},
    {"name": "CX_0.5", "config": CycleCrossover(crossover_rate=0.5)}
]

MUTATION_OPS = [
    {"name": "Swap_0.8", "config": SwapMutation(individual_mutation_prob=0.8)},
    {"name": "Invert_0.8", "config": InversionMutation(individual_mutation_prob=0.8)},
    {"name": "Swap_0.5", "config": SwapMutation(individual_mutation_prob=0.5)},
    {"name": "Invert_0.5", "config": InversionMutation(individual_mutation_prob=0.5)}
]

# The survivor selectors will be initialized later with the
# correct population size inside the StrategyGrid.
SURVIVOR_SELECTORS = [
    {"name": "Elitism", "class": PlusSelection},
]


def find_winner_config(builder: StrategyGridBuilder) -> Optional[Dict[str, Any]]:
    """
    Re-builds the strategy grid to find the config for the winner.
    
    This function iterates through all population shapes and
    their corresponding strategy grids to find the one matching
    the 'WINNING_STRATEGY_NAME'.

    Returns:
        A dictionary containing the winner's 'setup', 'pop_size',
        and 'generations', or None if not found.
    """
    print("Searching for winning strategy configuration...")
    for pop_config in POPULATION_CONFIGS:
        pop_name = pop_config["name"]
        pop_size = pop_config["pop_size"]
        
        # Calculate generations based on the budget
        generations = (TOTAL_EVALUATION_BUDGET - pop_size) // pop_size
        
        # Build the strategy grid for this specific pop_size
        strategy_grid = builder.build_grid(pop_size=pop_size) 
        
        # Search this grid for the winner
        for setup in strategy_grid:
            # Construct the full, unique name
            full_name = f"{pop_name}_{setup['name']}"
            
            if full_name == WINNING_STRATEGY_NAME:
                # We found it!
                print(f"Found winner: {WINNING_STRATEGY_NAME}")
                return {
                    "setup": setup,
                    "pop_size": pop_size,
                    "generations": generations
                }
    
    return None # Winner not found

# Main Pipeline Function
def run_showcase():
    """
    Orchestrates the final benchmark run.
    """
    
    # Find the Winning Strategy Config
    print("Building strategy grid to find winner...")
    builder = StrategyGridBuilder(
        parent_selectors=PARENT_SELECTORS,
        survivor_selectors=SURVIVOR_SELECTORS,
        crossover_ops=CROSSOVER_OPS,
        mutation_ops=MUTATION_OPS
    )
    
    winner = find_winner_config(builder)
    
    if winner is None:
        print(f"Error: Winning strategy name '{WINNING_STRATEGY_NAME}' not found in grid.")
        print("Did you copy the name correctly from the tuning analysis notebook?")
        print("Do the POPULATION_CONFIGS and component lists match '02_run_tuning.py'?")
        return

    # Extract the winning parameters
    winner_setup = winner["setup"]
    winner_pop_size = winner["pop_size"]
    winner_generations = winner["generations"]

    print(f"--- Starting Showcase for Winner ---")
    print(f"  Name: {WINNING_STRATEGY_NAME}")
    print(f"  Config: Pop={winner_pop_size}, Gen={winner_generations}")

    # Load Baseline & All Valid Problems
    print(f"Loading baseline file: {BASELINE_FILE}")
    try:
        with open(BASELINE_FILE, 'r') as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Baseline file not found at {BASELINE_FILE}.")
        return
    
    # Get *all* valid problems for the final showcase
    all_valid_problems = [name for name, fitness in baseline_results.items() if fitness is not None]
    if not all_valid_problems:
        print("Error: No valid problems found in baseline file.")
        return
        
    print(f"Found {len(all_valid_problems)} valid problems for the showcase.")
    
    # Execute Showcase Loops
    silent_logger = Logger(level=LogLevel.NONE)
    start_time = time.time()
    
    # This dict will store the *final, aggregated* stats
    # { "problem_name": {"mean_fitness": -2950, "mean_improvement_pct": 3.4, ...}}
    showcase_results: Dict[str, Dict[str, Any]] = {}

    # Outer Loop: By Problem (All valid problems)
    for problem_name in tqdm(all_valid_problems, desc="Running Showcase"):
        problem_file = os.path.join(PROBLEM_DATA_DIR, problem_name)
        baseline_fitness = baseline_results[problem_name]
        
        try:
            matrix = np.load(problem_file)
            problem = TSPProblem(matrix)
            
            # This list will store the *final* fitness of each N run
            run_final_fitnesses: List[float] = []
            
            # Inner Loop: N Statistical Runs
            for i in range(N_STATISTICAL_RUNS):
                ea = EvolutionaryAlgorithm(
                    fitness_function=problem.get_fitness_function(),
                    initialization_function=problem.get_initializer_function(),
                    parent_selection=winner_setup["parent_selection"],
                    reproduction_strategy=winner_setup["reproduction_strategy"],
                    survivor_selection=winner_setup["survivor_selection"],
                    population_size=winner_pop_size,
                    logger=silent_logger
                )
                
                best_ind, _ = ea.run(winner_generations)
                run_final_fitnesses.append(best_ind.fitness)

            # Aggregate & Normalize Results (for this problem)
            
            # Calculate final stats
            mean_fitness = np.mean(run_final_fitnesses)
            std_fitness = np.std(run_final_fitnesses)
            best_fitness = np.max(run_final_fitnesses)
            worst_fitness = np.min(run_final_fitnesses)
            median_fitness = np.median(run_final_fitnesses)
            
            # Calculate % improvement over baseline
            improvement_pct = 0.0
            if baseline_fitness != 0: # Avoid division by zero
                improvement_pct = (mean_fitness - baseline_fitness) / abs(baseline_fitness) * 100
            
            # Save the aggregated dictionary
            showcase_results[problem_name] = {
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "best_fitness": best_fitness,
                "worst_fitness": worst_fitness,
                "median_fitness": median_fitness,
                "baseline_fitness_hc": baseline_fitness,
                "mean_improvement_pct": improvement_pct
            }
            
        except Exception as e:
            print(f"\nERROR on {problem_name}: {e}")
            showcase_results[problem_name] = None # Mark as failed

    # Save Final Report Data
    try:
        with open(SHOWCASE_OUTPUT_FILE, 'w') as f:
            json.dump(showcase_results, f, indent=4)
        
        end_time = time.time()
        print(f"\n--- Showcase Phase Complete ---")
        print(f"Final aggregated results saved to: {SHOWCASE_OUTPUT_FILE}")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to save JSON results: {e}")


if __name__ == "__main__":
    run_showcase()