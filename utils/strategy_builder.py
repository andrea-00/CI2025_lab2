# utils/experiment_builder.py

"""
Provides a builder class for programmatically generating EA experiment grids.

This module decouples the *definition* of an experiment (the components,
which are passed in) from the *generation* of the full experiment grid.
It handles the complex logic of combining simple and composite strategies.
"""

import itertools
from typing import List, Dict, Any, Optional

# Import all necessary "bricks" from the evolvepy framework.
# This module assumes 'evolvepy' is installed as a dependency
# (e.g., via requirements.txt).
from evolvepy.strategies import (
    StandardReproduction, 
    MutationOnlyReproduction
)


class StrategyGridBuilder:
    """
    Builds a grid of EA experiment configurations (recipes).

    This class takes lists of individual components (e.g., mutation operators,
    crossover operators) and correctly assembles them into complex,
    valid "recipe" dictionaries ready to be consumed by the tuning script.
    
    It correctly handles composite strategies, such as building a list
    of `StandardReproduction` strategies from separate crossover and
    mutation lists.
    """

    def __init__(self,
                 parent_selectors: List[Dict[str, Any]],
                 survivor_selectors: List[Dict[str, Any]],
                 crossover_ops: Optional[List[Dict[str, Any]]] = None,
                 mutation_ops: Optional[List[Dict[str, Any]]] = None
                ):
        """
        Initializes the builder with lists of strategy components.

        Each component in the lists is expected to be a dictionary
        with two keys:
            - 'name': A short, unique string for logging (e.g., "Tourn_k3").
            - 'config': The actual, initialized strategy object 
                        (e.g., TournamentSelection(k=3)).

        Args:
            parent_selectors: List of parent selection strategy components.
            survivor_selectors: List of survivor selection strategy components.
            crossover_ops (Optional): List of crossover operator components.
            mutation_ops (Optional): List of mutation operator components.
        """
        self.parent_selectors = parent_selectors
        self.survivor_selectors = survivor_selectors
        self.crossover_ops = crossover_ops or []
        self.mutation_ops = mutation_ops or []

    def _build_standard_recipes(self) -> List[Dict[str, Any]]:
        """
        Generates composite recipes for StandardReproduction (Crossover + Mutation).
        
        This performs a Cartesian product of all provided Crossover
        and Mutation operators.

        Returns:
            A list of reproduction strategy components (dict format).
        """
        repro_recipes = []
        
        # Guard clause: Cannot build this type without both operators
        if not self.crossover_ops or not self.mutation_ops:
            print("Warning: 'StandardReproduction' recipes skipped: "
                  "missing crossover_ops or mutation_ops.")
            return []

        # Cartesian product: (Crossover x Mutation)
        for cross, mut in itertools.product(self.crossover_ops, self.mutation_ops):
            
            repro_name = f"StdRep_{cross['name']}_{mut['name']}"
            repro_config = StandardReproduction(
                recombination_strategy=cross['config'],
                mutation_strategy=mut['config']
            )
            repro_recipes.append({"name": repro_name, "config": repro_config})
            
        return repro_recipes

    def _build_mutate_only_recipes(self) -> List[Dict[str, Any]]:
        """
        Generates composite recipes for MutationOnlyReproduction.

        This loops through all provided Mutation operators.

        Returns:
            A list of reproduction strategy components (dict format).
        """
        repro_recipes = []
        
        # Guard clause: Cannot build without mutation operators
        if not self.mutation_ops:
            print("Warning: 'MutationOnlyReproduction' recipes skipped: "
                  "missing mutation_ops.")
            return []

        for mut in self.mutation_ops:
            repro_name = f"MutOnly_{mut['name']}"
            repro_config = MutationOnlyReproduction(
                mutation_strategy=mut['config']
            )
            repro_recipes.append({"name": repro_name, "config": repro_config})
            
        return repro_recipes

    def build_grid(self, 
                   pop_size: int,
                   include_standard: bool = True, 
                   include_mutate_only: bool = True) -> List[Dict[str, Any]]:
        """
        Builds the final, complete grid of EA experiment "recipes".

        This method first generates all valid reproduction strategies
        (based on the flags) and then performs the final Cartesian
        product to combine them with parent and survivor selectors.

        Args:
            pop_size (int): The population size for the experiments.
            include_standard (bool): If True, generates (Crossover + Mutation) recipes.
            include_mutate_only (bool): If True, generates (Mutation-Only) recipes.

        Returns:
            List[Dict[str, Any]]: The complete grid of EA recipes. Each
            dict contains a 'name' and all necessary strategy objects.
        
        Raises:
            ValueError: If no reproduction strategies could be generated
                        (e.g., missing components or all flags are False).
        """
        final_grid = []
        all_repro_recipes = []
        
        # 1. Build the list of all possible reproduction strategies
        if include_standard:
            all_repro_recipes.extend(self._build_standard_recipes())
        
        if include_mutate_only:
            all_repro_recipes.extend(self._build_mutate_only_recipes())
        
        initialized_survivors = []
        for s in self.survivor_selectors:
            try:
                if s["class"].__name__ == "PlusAgeBasedSelection":
                    # initialize age-based survivor selector with pop size and max age
                    initialized_survivors.append({
                        "name": s["name"],
                        "config": s["class"](population_size=pop_size, max_age=5)
                    })
                else:
                    # initialize survivor selectors with the correct pop size
                    initialized_survivors.append({
                        "name": s["name"],
                        "config": s["class"](population_size=pop_size)
                    })
            except Exception as e:
                print(f"Errore: {s['name']} non ha 'population_size'? {e}")
        
        # Check if we actually built any recipes
        if not all_repro_recipes:
            raise ValueError(
                "No reproduction strategies were generated. "
                "Check if component lists (e.g., mutation_ops) are empty "
                "or if all 'include_' flags are False."
            )

        # 2. Build the final grid
        # This is the main Cartesian product:
        # (Parent x Reproduction x Survivor)
        components_to_combine = [
            self.parent_selectors, 
            all_repro_recipes, 
            initialized_survivors
        ]
        
        # Check for empty component lists
        for component_list in components_to_combine:
            if not component_list:
                raise ValueError("Cannot build grid: parent_selectors, survivor_selectors, "
                                 "or the generated repro_recipes list is empty.")

        for parent, repro, survivor in itertools.product(*components_to_combine):
            # Combine the components into a single "recipe" dictionary
            # that the '02_run_tuning.py' script can directly use.
            final_grid.append({
                "name": f"{parent['name']}_{repro['name']}_{survivor['name']}",
                "parent_selection": parent['config'],
                "reproduction_strategy": repro['config'],
                "survivor_selection": survivor['config']
            })
                
        print(f"Grid built. Total strategies to test: {len(final_grid)}")
        return final_grid