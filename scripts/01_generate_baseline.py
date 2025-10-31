# scripts/01_generate_baseline.py

"""
Generates the baseline performance file for all TSP problems.

This script iterates through all problem files in 'data/tsp_problems/',
runs a simple Stochastic Hill Climber (from 'baselines/') for N
statistical runs, and records the *best fitness* found for each
problem.

The output is 'data/baseline_hc.json', which is used by all
other analysis notebooks to normalize results (i.e., calculate
percent improvement over this baseline).
"""

import numpy as np
import glob
import json
import time
import os
import sys
from typing import List, Dict

# Path Setup
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Import the problem definition and the baseline solver
from problems.tsp import TSPProblem
from baselines.hill_climber import HillClimber

# Experiment Configuration
# Total number of iterations (tweak attempts) for *each* HC run
HC_ITERATIONS = 50000

# Number of times to run the HC on *each* problem.
# We do this to mitigate the effect of a "lucky" or "unlucky"
# random start. We will record the *best* score from these runs.
N_STATISTICAL_RUNS = 20

# Path to the problem data (using glob)
PROBLEM_PATH_GLOB = "data/tsp_problems/*.npy"

# The final output file
BASELINE_OUTPUT_FILE = "data/baseline_hc.json"



def run_benchmark():
    """
    Main pipeline function. Loads problems, runs the
    Hill Climber benchmark, and saves the results.
    """
    print("--- Starting Baseline Generation (Hill Climber) ---")
    print(f"Iterations per run: {HC_ITERATIONS}")
    print(f"Statistical runs per problem: {N_STATISTICAL_RUNS}")
    
    # Find all problem files
    problem_files = glob.glob(PROBLEM_PATH_GLOB)
    if not problem_files:
        print(f"Error: No problem files found at '{PROBLEM_PATH_GLOB}'")
        return

    # This dictionary will store the final results:
    # { "problem_name.npy": best_fitness_found }
    baseline_results: Dict[str, float] = {}
    
    start_time = time.time()

    # Outer Loop: Iterate over all 50 problems
    for problem_file in problem_files:
        problem_name = os.path.basename(problem_file)
        
        try:
            # Load the problem matrix and instantiate the problem class
            matrix = np.load(problem_file)
            problem_instance = TSPProblem(matrix)
            
            # This list will store the best fitness from each of the N runs
            run_best_fitnesses: List[float] = []
            
            # Middle Loop: N statistical runs for this problem
            for i in range(N_STATISTICAL_RUNS):
                
                # Create a new solver instance *with a different seed*
                # This ensures each of the N runs is unique.
                solver = HillClimber(problem_instance, seed=i)
                
                # Run the solver. It will maximize fitness (negative distance).
                _, best_fitness = solver.solve(HC_ITERATIONS)
                
                run_best_fitnesses.append(best_fitness)

            # Aggregation
            # From the N runs, find the *single best* fitness achieved.
            # Since fitness is -distance, we want the *maximum* value.
            best_overall_fitness = np.max(run_best_fitnesses)
            baseline_results[problem_name] = best_overall_fitness
            
        except Exception as e:
            # Log any errors (e.g., bad matrix) but don't stop the whole benchmark
            print(f"\nError processing {problem_name}: {e}")
            baseline_results[problem_name] = None  # Use 'None' for failed problems

    # Save Results
    try:
        with open(BASELINE_OUTPUT_FILE, 'w') as f:
            # Save the dictionary to a JSON file with nice formatting
            json.dump(baseline_results, f, indent=4)
        
        end_time = time.time()

        total_problems = len(baseline_results)
        failed_problems = sum(1 for v in baseline_results.values() if v is None)
        success_problems = total_problems - failed_problems
        
        percent_success = 0.0
        percent_failed = 0.0
        if total_problems > 0:
            percent_success = (success_problems / total_problems) * 100
            percent_failed = (failed_problems / total_problems) * 100
        
        print("\n--- Baseline Generation Complete ---")
        print(f"Results for {len(baseline_results)} problems saved to: {BASELINE_OUTPUT_FILE}")
        print("\n--- Validation Summary ---")
        print(f"  Total Problems Found:     {total_problems}")
        print(f"  Succeeded (Valid TSP): {success_problems} ({percent_success:.1f}%)")
        print(f"  Failed (Invalid/Null):  {failed_problems} ({percent_failed:.1f}%)")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to save JSON results: {e}")

# Script Execution
if __name__ == "__main__":
    # This ensures the benchmark only runs when the script
    # is executed directly (not when imported)
    run_benchmark()