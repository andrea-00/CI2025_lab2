# scripts/02_run_tuning.py

"""
Prerequisite:
    - 'scripts/01_generate_baseline.py' must be run first to
      create the 'data/baseline_hc.json' file.
"""

import numpy as np
import glob
import json
import time
import os
import sys
from tqdm import tqdm
from typing import List, Dict, Any

# Path Setup
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
print(project_root)
sys.path.append(project_root)

# Framework Imports
from evolvepy import EvolutionaryAlgorithm, Logger, LogLevel
from evolvepy.strategies import (
    TournamentSelection, UniformSelection, 
    PlusSelection, CommaSelection, PlusAgeBasedSelection,
    StandardReproduction, MutationOnlyReproduction,
    OrderedCrossover, CycleCrossover,
    SwapMutation, InversionMutation
)
from problems.tsp import TSPProblem
from utils.strategy_builder import StrategyGridBuilder

# Experiment Configuration
# Number of times to run each (strategy, problem) pair to
# average out stochastic randomness.
N_STATISTICAL_RUNS = 3

# Product of population size and generations
TOTAL_EVALUATION_BUDGET = 20000
POPULATION_CONFIGS = [
    #{"name": "Shape_Broad", "pop_size": 100},
    #{"name": "Shape_Medium", "pop_size": 50},
    {"name": "Shape_Deep", "pop_size": 25},
]

# Number of *valid* problems to use for the tuning phase.
# We use a subset for speed. The final showcase will use all.
TUNING_SET_SIZE = 5

# I/O Configuration
BASELINE_FILE = os.path.join(project_root, "data/baseline_hc.json")
RESULTS_DIR = os.path.join(project_root, "experiments/tuning_results")
PROBLEM_DATA_DIR = os.path.join(project_root, "data/tsp_problems")


# Strategy Component Definition
# This is the "control panel" for the experiment.
# We define the "bricks" that the StrategyGridBuilder will use.

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
    {"name": "Comma", "class": CommaSelection},
]


# Main Pipeline Function
def run_tuning():
    """
    Orchestrates the entire tuning process:
    1. Builds the strategy grid.
    2. Loads and filters problems.
    3. Runs the experiment loops.
    4. Saves all raw results.
    """
    
    # Build Strategy Grid
    print("Building strategy grid...")
    builder = StrategyGridBuilder(
        parent_selectors=PARENT_SELECTORS,
        survivor_selectors=SURVIVOR_SELECTORS,
        crossover_ops=CROSSOVER_OPS,
        mutation_ops=MUTATION_OPS
    )

    # Load Baseline & Filter Problems
    print(f"Loading baseline file: {BASELINE_FILE}")
    try:
        with open(BASELINE_FILE, 'r') as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Baseline file not found at {BASELINE_FILE}.")
        print("Please run 'scripts/01_generate_baseline.py' first.")
        return
    
    # Filter for *only* the problems that were successfully
    # validated and solved by the baseline (i.e., not 'null').
    valid_problems = [name for name, fitness in baseline_results.items() if fitness is not None]
    
    if not valid_problems:
        print("Error: No valid problems found in baseline file. Stopping.")
        return
        
    # Create Tuning Set
    # Select a subset of the valid problems for tuning
    tuning_set_files = valid_problems[:TUNING_SET_SIZE]
    print(f"Tuning set created: {len(tuning_set_files)} valid problems.")
    
    # Prepare Output Directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Execute Tuning Loops
    # We use a silent logger to prevent console flooding
    # during the thousands of EA runs.
    silent_logger = Logger(level=LogLevel.NONE)
    
    start_time = time.time()

    # Initial build to count strategies
    strategy_grid = builder.build_grid(
        pop_size=POPULATION_CONFIGS[0]["pop_size"],
        include_mutate_only=False
    )
    
    print(f"\n--- Starting Tuning Phase ---")
    total_runs = TOTAL_EVALUATION_BUDGET * N_STATISTICAL_RUNS
    print(f"  Strategies to test: {len(strategy_grid)}")
    print(f"  Problems in Tuning Set: {len(tuning_set_files)}")
    print(f"  Statistical Runs per Problem: {N_STATISTICAL_RUNS}")
    print(f"  == Total EA runs to perform: {total_runs} ==")

    # Outer Loop 1: By Population Shape
    for pop_config in POPULATION_CONFIGS:
        
        pop_name = pop_config["name"]
        pop_size = pop_config["pop_size"]
        
        # Calculate generations to keep total evals constant
        generations = (TOTAL_EVALUATION_BUDGET - pop_size) // pop_size
        
        print(f"\n--- Testing Population Shape: {pop_name} (Pop={pop_size}, Gen={generations}) ---")
        
        # Build strategy grid for this population size
        strategy_grid = builder.build_grid(pop_size=pop_size, include_mutate_only=False)

        # Outer Loop 2: By Strategy
        for setup in tqdm(strategy_grid, desc=f"Strategies for {pop_name}"):
            strategy_name = f"{pop_name}_{setup['name']}"
            
            # This dict will store { "problem_name": [list_of_full_histories] }
            strategy_results: Dict[str, List[Dict[str, List]]] = {} 

            # Middle Loop: By Problem
            for problem_name in tuning_set_files:
                problem_file = os.path.join(PROBLEM_DATA_DIR, problem_name)
                
                try:
                    # Load the problem data
                    matrix = np.load(problem_file)
                    problem = TSPProblem(matrix)
                    
                    # This list will store the history of each of the N runs
                    run_histories: List[Dict[str, List]] = []
                    
                    # Inner Loop: N Statistical Runs
                    for i in range(N_STATISTICAL_RUNS):
                        # Assemble the EA instance
                        ea = EvolutionaryAlgorithm(
                            fitness_function=problem.get_fitness_function(),
                            initialization_function=problem.get_initializer_function(),
                            parent_selection=setup["parent_selection"],
                            reproduction_strategy=setup["reproduction_strategy"],
                            survivor_selection=setup["survivor_selection"],
                            population_size=pop_size,
                            logger=silent_logger # Use the silent logger
                        )
                        
                        # Run the algorithm and get the history
                        _, history = ea.run(generations=generations)
                        
                        # Store the *entire history* for later analysis
                        # (e.g., convergence speed plotting)
                        run_histories.append(history)

                    # Store all N histories for this problem
                    strategy_results[problem_name] = run_histories
                    
                except Exception as e:
                    # Log errors but continue the experiment
                    print(f"\nERROR on {problem_name} with {strategy_name}: {e}")
            
            # Save Results for This Strategy
            # We save one JSON file *per strategy* to disk.
            # This is robust: if the script crashes, we don't lose all progress.
            results_path = os.path.join(RESULTS_DIR, f"{strategy_name}.json")
            try:
                with open(results_path, 'w') as f:
                    json.dump(strategy_results, f)
            except Exception as e:
                print(f"\nCRITICAL: Failed to save results for {strategy_name}: {e}")

    end_time = time.time()
    print(f"\n--- Tuning Phase Complete ---")
    print(f"Raw results saved to: {RESULTS_DIR}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # This standard Python pattern ensures that 'run_tuning()'
    # is only called when the script is executed directly.
    run_tuning()