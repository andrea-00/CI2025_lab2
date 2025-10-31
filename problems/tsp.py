# problems/tsp.py

import numpy as np
import random
from typing import List, Callable

from evolvepy import BaseProblem 

# Metadata
CITIES_METADATA = [
    "Rome", "Milan", "Naples", "Turin", "Palermo", "Genoa", 
    "Bologna", "Florence", "Bari", "Catania", "Venice", "Verona", 
    "Messina", "Padua", "Trieste", "Taranto", "Brescia", "Prato", 
    "Parma", "Modena",
]

class TSPProblem(BaseProblem[List[int]]):
    """
    Encapsulates all logic for a specific Travelling Salesman Problem instance.

    This class is instantiated with a specific distance matrix and provides
    the problem-specific callables (fitness function, genotype initializer)
    required by the general-purpose EA engine.
    """

    def __init__(self, distance_matrix: np.ndarray):
        """
        Initializes the TSP problem instance.

        Args:
            distance_matrix (np.ndarray): A 2D square matrix where
                matrix[i, j] is the distance from city 'i' to city 'j'.
        Raises:
            ValueError: If the distance matrix fails validation
                        (e.g., non-square, negative values, asymmetric, etc.).
        """
        self.matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Distance matrix must be a 2D square matrix.")
        
        self._validate_matrix()

    def get_fitness_function(self) -> Callable[[List[int]], float]:
        """
        Returns the problem-specific fitness function as a callable.
        
        The returned function "closes over" this object's state
        (self.matrix and self.num_cities).
        """
        # Return the method itself, which is a valid callable
        return self._calculate_fitness

    def get_initializer_function(self) -> Callable[[], List[int]]:
        """
        Returns the problem-specific genotype initializer as a callable.
        """
        # Return the method itself
        return self._create_genotype
    
    def _validate_matrix(self):
        """
        Performs rigorous validation on the loaded distance matrix.
        
        Checks for:
        1. Negative distances.
        2. Non-zero diagonal (dist(A, A) must be 0).
        3. Asymmetry (dist(A, B) must equal dist(B, A)).
        4. Triangular Inequality (dist(A, C) <= dist(A, B) + dist(B, C)).
        """
        
        # Check for negative values
        if np.any(self.matrix < 0):
            raise ValueError("Distance matrix contains negative values.")

        # Check diagonal
        # Distance from a city to itself must be 0.
        diagonal = np.diag(self.matrix)
        if not np.allclose(diagonal, 0):
            raise ValueError("Diagonal elements are not all zero. "
                             "Distance from a city to itself must be 0.")

        # Check for symmetry
        # dist(A, B) == dist(B, A)
        if not np.allclose(self.matrix, self.matrix.T):
            raise ValueError("Distance matrix is not symmetric. "
                             "Dist(A, B) must equal Dist(B, A).")

        # Check Triangular Inequality
        for j in range(self.num_cities):
            # col_j[i] = dist(i, j)
            col_j = self.matrix[:, j:j+1] # Shape (N, 1)
            # row_j[k] = dist(j, k)
            row_j = self.matrix[j:j+1, :] # Shape (1, N)
            
            # path_matrix[i, k] = dist(i, j) + dist(j, k)
            path_matrix_via_j = col_j + row_j
            
            # Check if any direct path dist(i, k) is *strictly* greater
            # than the indirect path dist(i, j) + dist(j, k).
            # We add a small tolerance for floating-point errors.
            tolerance = 1e-9
            if np.any(self.matrix > path_matrix_via_j + tolerance):
                # Find the specific violation for a clear error message
                violations = np.where(self.matrix > path_matrix_via_j + tolerance)
                i, k = violations[0][0], violations[1][0]
                raise ValueError(
                    f"Triangular inequality violated (via intermediate node {j}). "
                    f"dist({i}, {k}) = {self.matrix[i, k]:.2f} > "
                    f"dist({i}, {j}) + dist({j}, {k}) = "
                    f"({self.matrix[i, j]:.2f} + {self.matrix[j, k]:.2f}) = "
                    f"{path_matrix_via_j[i, k]:.2f}"
                )

    def _calculate_fitness(self, genotype: List[int]) -> float:
        """
        Calculates the fitness of a TSP solution (a permutation).
        
        The genotype is expected to be a list of city indices,
        e.g., [3, 1, 0, 2] for 4 cities.
        
        Fitness is defined as the negative of the total distance,
        as the EA engine is designed to *maximize* fitness.

        Args:
            genotype: A list of integers representing a permutation
                      of city indices (from 0 to N-1).

        Returns:
            The fitness (1 / total_distance).
        """
        total_distance = 0.0
        
        for i in range(self.num_cities):
            # Get the current city index and the next city index
            city_a_idx = genotype[i]
            
            # Use modulo to wrap around from the last city to the first
            city_b_idx = genotype[(i + 1) % self.num_cities]
            
            # Add the distance from the pre-loaded matrix
            total_distance += self.matrix[city_a_idx, city_b_idx]
            
        # Negative of the total distance for maximization problem
        return -total_distance

    def _create_genotype(self) -> List[int]:
        """
        Creates a new, random genotype for the TSP.
        
        A TSP genotype is a random permutation of all city indices.

        Returns:
            A new list, e.g., [2, 0, 3, 1]
        """
        # Create a list [0, 1, 2, ..., N-1]
        genotype = list(range(self.num_cities))
        
        # Shuffle it in-place
        random.shuffle(genotype)
        
        return genotype