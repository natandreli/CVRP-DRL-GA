import random
import time
from typing import Callable, Optional

from app.core.ga.individual import Individual
from app.core.ga.operators import (
    apply_crossover,
    apply_mutation,
    apply_selection,
)
from app.core.ga.population_generator import generate_initial_population
from app.schemas import CVRPInstance, GAConfig, Solution


class GeneticAlgorithm:
    """Genetic Algorithm for CVRP."""

    def __init__(
        self,
        instance: CVRPInstance,
        config: GAConfig,
        callback: Optional[Callable[[int, Individual], None]] = None,
    ) -> None:
        """
        Initialize Genetic Algorithm.

        Args:
            instance (CVRPInstance): CVRP instance to solve
            config (GAConfig): GA configuration
            callback (Optional[Callable[[int, Individual], None]]): Optional callback(generation, best_individual)
        """
        self.instance = instance
        self.config = config
        self.callback = callback

        # Set random seed
        if config.seed is not None:
            random.seed(config.seed)

        self.population: list[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.convergence_history: list[float] = []

    def initialize_population(
        self,
        custom_population: Optional[list[Individual]] = None,
        use_heuristics: bool = False,
    ) -> None:
        """
        Create initial population.

        Args:
            custom_population (Optional[list[Individual]]): Pre-generated population (e.g., from DRL).
            use_heuristics (Optional[bool]): If True, use heuristics (NN + savings + random).
                                         If False, use pure random initialization.
        """
        if custom_population is not None:
            self.population = [ind.copy() for ind in custom_population]
        else:
            self.population = generate_initial_population(
                instance=self.instance,
                population_size=self.config.population_size,
                use_heuristics=use_heuristics,
                seed=self.config.seed,
            )

    def run_generation(self) -> None:
        """Evolve population for one generation."""
        new_population = []

        # Elitism: preserve best individuals
        if self.config.elitism_count > 0:
            elite = sorted(self.population, key=lambda c: c.fitness)[
                : self.config.elitism_count
            ]
            new_population.extend([e.copy() for e in elite])

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = apply_selection(
                self.population,
                method=self.config.selection_method,
                tournament_size=3,
            )
            parent2 = apply_selection(
                self.population,
                method=self.config.selection_method,
                tournament_size=3,
            )

            # Crossover
            if random.random() < self.config.crossover_rate:
                offspring = apply_crossover(
                    parent1,
                    parent2,
                    method=self.config.crossover_method,
                )
            else:
                offspring = parent1.copy()

            # Mutation
            if random.random() < self.config.mutation_rate:
                offspring = apply_mutation(
                    offspring,
                    method=self.config.mutation_method,
                )

            new_population.append(offspring)

        self.population = new_population[: self.config.population_size]

    def get_best_individual(self) -> Individual:
        """
        Get best Individual from current population.

        Returns:
            Individual: Best individual
        """
        return min(self.population, key=lambda c: c.fitness)

    def run(
        self,
        custom_population: Optional[list[Individual]] = None,
        use_heuristics: Optional[bool] = False,
    ) -> Solution:
        """
        Run the genetic algorithm.

        Args:
            custom_population (Optional[list[Individual]]): Pre-generated population (e.g., from DRL)
            use_heuristics (Optional[bool]): If True, use heuristics (NN + savings + random).
                                        If False, use pure random initialization.

        Returns:
            Solution: Best solution found
        """
        start_time = time.time()

        # Initialize population
        self.initialize_population(custom_population, use_heuristics)

        # Track best
        self.best_individual = self.get_best_individual()
        self.convergence_history.append(self.best_individual.fitness)

        # Evolution loop
        for generation in range(self.config.generations):
            # Evolve population for one generation
            self.run_generation()

            # Update best
            current_best = self.get_best_individual()
            if current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best

            # Track convergence
            self.convergence_history.append(self.best_individual.fitness)

            # Callback
            if self.callback:
                self.callback(generation + 1, self.best_individual)

        # Convert best to solution
        computation_time = time.time() - start_time
        solution = self.best_individual.to_solution(
            algorithm="ga",
            generation=self.config.generations,
        )
        solution.computation_time = computation_time
        solution.convergence_history = self.convergence_history

        # Validate solution
        from app.core.utils.solution_evaluator import validate_solution

        is_valid, message = validate_solution(solution, self.instance)
        solution.is_valid = is_valid

        return solution


def solve_cvrp_with_ga(
    instance: CVRPInstance,
    config: Optional[GAConfig] = None,
    verbose: bool = False,
) -> Solution:
    """
    Solve CVRP using Genetic Algorithm.

    Args:
        instance (CVRPInstance): CVRP instance
        config (Optional[GAConfig]): GA configuration (uses defaults if None)
        verbose (bool): Print progress

    Returns:
        Solution: Best solution found
    """
    if config is None:
        from app.config import settings

        config = GAConfig(
            population_size=settings.GA_DEFAULT_POPULATION,
            generations=settings.GA_DEFAULT_GENERATIONS,
        )

    def callback(generation: int, best: Individual) -> None:
        if verbose and generation % 10 == 0:
            print(
                f"Generation {generation}/{config.generations}: "
                f"Best fitness = {best.fitness:.2f}"
            )

    ga = GeneticAlgorithm(
        instance=instance,
        config=config,
        callback=callback if verbose else None,
    )

    return ga.run()
