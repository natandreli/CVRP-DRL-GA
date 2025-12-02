import uuid
from datetime import datetime
from typing import Optional

from app.config.logging import logger
from app.core.drl.actor_critic_agent import ActorCriticAgent
from app.core.drl.population_generator import generate_population_with_drl
from app.core.ga.genetic_algorithm import GeneticAlgorithm
from app.core.utils import evaluate_solution
from app.schemas import DRLConfig, GAConfig, Solution
from app.services.instances import instance_manager


class SolverService:
    """
    Service for solving CVRP instances with different algorithms.

    Orchestrates: instance retrieval, algorithm execution, solution evaluation.
    """

    def solve_with_ga(
        self,
        instance_id: str,
        config: Optional[GAConfig] = None,
        use_heuristics: bool = True,
    ) -> Solution:
        """
        Solve CVRP instance using Genetic Algorithm.

        Args:
            instance_id: ID of the CVRP instance
            config: GA configuration (uses defaults if None)
            use_heuristics: Whether to use heuristic initialization (NN, savings)

        Returns:
            Solution object with routes and metrics

        Raises:
            ValueError: If instance not found or GA fails
        """
        logger.info(f"Starting GA solver for instance '{instance_id}'")

        # 1. Get instance
        instance = instance_manager.get_instance(instance_id)

        # 2. Use default config if not provided
        if config is None:
            config = GAConfig()

        # 3. Initialize and run GA
        ga = GeneticAlgorithm(instance=instance, config=config)
        ga.initialize_population(use_heuristics=use_heuristics)
        ga.run()

        # 4. Get best solution
        best_individual = ga.best_individual
        if best_individual is None:
            raise ValueError("GA failed to produce a solution")

        # 5. Convert to Solution schema
        solution = Solution(
            id=f"sol_ga_{uuid.uuid4().hex[:8]}",
            instance_id=instance_id,
            algorithm="ga",
            routes=best_individual.routes,
            total_cost=best_individual.fitness,
            created_at=datetime.now(),
            metadata={
                "population_size": config.population_size,
                "generations": config.generations,
                "crossover_rate": config.crossover_rate,
                "mutation_rate": config.mutation_rate,
                "use_heuristics": use_heuristics,
            },
        )

        # 6. Evaluate and validate
        is_valid, error_msg = evaluate_solution(solution, instance)
        solution.is_valid = is_valid
        if not is_valid:
            logger.warning(f"GA produced invalid solution: {error_msg}")
            solution.metadata["validation_error"] = error_msg

        logger.info(
            f"GA completed: {solution.num_vehicles_used} vehicles, "
            f"cost={solution.total_cost:.2f}, valid={is_valid}"
        )

        return solution

    def solve_with_drl(
        self,
        instance_id: str,
        config: Optional[DRLConfig] = None,
        model_path: Optional[str] = None,
    ) -> Solution:
        """
        Solve CVRP instance using Deep Reinforcement Learning (Actor-Critic).

        Args:
            instance_id: ID of the CVRP instance
            config: DRL configuration (uses defaults if None)
            model_path: Path to pre-trained model (if None, uses greedy policy without training)

        Returns:
            Solution object with routes and metrics

        Raises:
            ValueError: If instance not found or DRL fails
        """
        logger.info(f"Starting DRL solver for instance '{instance_id}'")

        # 1. Get instance
        instance = instance_manager.get_instance(instance_id)

        # 2. Use default config if not provided
        if config is None:
            config = DRLConfig()

        # 3. Initialize agent
        agent = ActorCriticAgent(instance=instance, config=config)

        # 4. Load pre-trained model if provided
        if model_path is not None:
            agent.load_model(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")

        # 5. Generate solution using greedy policy
        solution_obj = agent.generate_solution(greedy=True)

        # 6. Convert to Solution schema
        solution = Solution(
            id=f"sol_drl_{uuid.uuid4().hex[:8]}",
            instance_id=instance_id,
            algorithm="drl",
            routes=solution_obj.routes,
            total_cost=solution_obj.total_cost,
            created_at=datetime.now(),
            metadata={
                "model_path": model_path,
                "embedding_dim": config.embedding_dim,
                "device": config.device,
            },
        )

        # 7. Evaluate and validate
        is_valid, error_msg = evaluate_solution(solution, instance)
        solution.is_valid = is_valid
        if not is_valid:
            logger.warning(f"DRL produced invalid solution: {error_msg}")
            solution.metadata["validation_error"] = error_msg

        logger.info(
            f"DRL completed: {solution.num_vehicles_used} vehicles, "
            f"cost={solution.total_cost:.2f}, valid={is_valid}"
        )

        return solution

    def solve_with_neurogen(
        self,
        instance_id: str,
        ga_config: Optional[GAConfig] = None,
        drl_config: Optional[DRLConfig] = None,
        model_path: Optional[str] = None,
        drl_population_size: int = 50,
    ) -> Solution:
        """
        Solve CVRP using NeuroGen hybrid approach:
        DRL generates intelligent initial population → GA evolves it.

        Args:
            instance_id: ID of the CVRP instance
            ga_config: GA configuration
            drl_config: DRL configuration
            model_path: Path to pre-trained DRL model
            drl_population_size: Number of solutions to generate with DRL

        Returns:
            Solution object with routes and metrics

        Raises:
            ValueError: If instance not found or NeuroGen fails
        """
        logger.info(f"Starting NeuroGen solver for instance '{instance_id}'")

        # 1. Get instance
        instance = instance_manager.get_instance(instance_id)

        # 2. Use default configs if not provided
        if ga_config is None:
            ga_config = GAConfig()
        if drl_config is None:
            drl_config = DRLConfig()

        # 3. Initialize DRL agent
        agent = ActorCriticAgent(instance=instance, config=drl_config)

        # 4. Load pre-trained model if provided
        if model_path is not None:
            agent.load_model(model_path)
            logger.info(f"Loaded pre-trained DRL model from {model_path}")

        # 5. Generate initial population with DRL
        logger.info(f"Generating {drl_population_size} solutions with DRL...")
        initial_population = generate_population_with_drl(
            instance=instance,
            agent=agent,
            population_size=drl_population_size,
        )

        logger.info(
            f"DRL generated population with avg fitness: "
            f"{sum(ind.fitness for ind in initial_population) / len(initial_population):.2f}"
        )

        # 6. Initialize and run GA with DRL-generated population
        ga = GeneticAlgorithm(instance=instance, config=ga_config)
        ga.initialize_population(custom_population=initial_population)
        ga.run()

        # 7. Get best solution
        best_individual = ga.best_individual
        if best_individual is None:
            raise ValueError("NeuroGen failed to produce a solution")

        # 8. Convert to Solution schema
        solution = Solution(
            id=f"sol_neurogen_{uuid.uuid4().hex[:8]}",
            instance_id=instance_id,
            algorithm="neurogen",
            routes=best_individual.routes,
            total_cost=best_individual.fitness,
            created_at=datetime.now(),
            metadata={
                "ga_population_size": ga_config.population_size,
                "ga_generations": ga_config.generations,
                "drl_population_size": drl_population_size,
                "drl_model_path": model_path,
            },
        )

        # 9. Evaluate and validate
        is_valid, error_msg = evaluate_solution(solution, instance)
        solution.is_valid = is_valid
        if not is_valid:
            logger.warning(f"NeuroGen produced invalid solution: {error_msg}")
            solution.metadata["validation_error"] = error_msg

        logger.info(
            f"NeuroGen completed: {solution.num_vehicles_used} vehicles, "
            f"cost={solution.total_cost:.2f}, valid={is_valid}"
        )

        return solution


# Singleton instance
solver_service = SolverService()
