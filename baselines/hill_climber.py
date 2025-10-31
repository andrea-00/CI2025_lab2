# baselines/hill_climber.py

"""
Implements a simple Stochastic Hill Climber algorithm.

This module provides a `HillClimber` class that serves as an
independent baseline solver. It is not part of the 'evolvepy'
framework and is used purely for benchmarking purposes.
"""

import random
from typing import List, Optional, Tuple, Callable

# Add the project root to the path to allow importing from 'problems'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.tsp import TSPProblem


class HillClimber:
    """
    Implements a (1+1) Stochastic Hill Climbing algorithm.
    
    This solver is designed to work with a 'TSPProblem' instance.
    It iteratively applies a simple mutation (tweak) and accepts
    the change only if it results in a better (greater) or equal 
    fitness.
    """

    def __init__(self, problem: TSPProblem, seed: Optional[int] = None):
        """
        Initializes the HillClimber solver.
        
        The problem instance is "injected" (using composition)
        to de-couple the solver logic from the problem logic.

        Args:
            problem (TSPProblem): The problem instance to solve.
            seed (Optional[int], optional): A seed for the random
                number generator to ensure reproducibility. Defaults to None.
        """
        self.problem = problem
        
        # Use a local Random instance to avoid affecting the global state
        self.rng = random.Random(seed)
        
        # Get the cost function from the problem.
        # The problem's fitness function returns -distance (for maximization).
        self._cost_function: Callable[[List[int]], float] = \
            lambda genotype: self.problem.get_fitness_function()(genotype)

    def _get_initial_solution(self) -> List[int]:
        """Fetches a new, random initial solution from the problem."""
        return self.problem.get_initializer_function()()

    def _tweak(self, solution: List[int]) -> List[int]:
        """
        Creates a "neighbor" solution by applying one swap mutation.

        Args:
            solution (List[int]): The current solution.

        Returns:
            List[int]: A new, slightly modified solution.
        """
        # Always work on a copy to avoid mutating the original
        neighbor = solution.copy()
        neighbor_len = len(neighbor)
        
        # Guard clause for neighbor too small to swap
        if neighbor_len < 2:
            return neighbor
            
        # Select two *distinct* random indices
        idx1, idx2 = self.rng.sample(range(neighbor_len), 2)
        
        # Perform the swap
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def solve(self, max_iterations: int) -> Tuple[List[int], float]:
        """
        Executes the Stochastic Hill Climbing algorithm.
        
        This loop will run for a fixed number of iterations,
        attempting to *maximize* the fitness (negative total distance).

        Args:
            max_iterations (int): The total number of iterations (tweak attempts)
                                  to perform.

        Returns:
            Tuple[List[int], float]: A tuple containing:
                (best_solution_found, best_cost_found)
        """
        
        # Initialization
        current_solution = self._get_initial_solution()
        current_cost = self._cost_function(current_solution)
        
        # We need a separate 'best' tracker because the hill
        # climber can get stuck on a plateau, but the *initial*
        # solution might have been the best one.
        best_solution = current_solution
        best_cost = current_cost

        # Main Loop
        for _ in range(max_iterations):
            # Generate a neighbor
            neighbor = self._tweak(current_solution)
            neighbor_cost = self._cost_function(neighbor)
            
            # Selection (Maximization)
            if neighbor_cost >= current_cost:
                # Move to the better neighbor
                current_solution = neighbor
                current_cost = neighbor_cost
                
                # Check if this is also a new all-time best
                if current_cost > best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
            
        # Return the best solution found *during the entire run*
        return best_solution, best_cost