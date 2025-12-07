import time
from pathlib import Path
from typing import Optional

import torch

from app.core.drl.actor_critic_agent import ActorCriticAgent
from app.core.drl.population_generator import generate_population_with_drl
from app.core.experiments.result_collector import ExperimentResult
from app.core.ga.genetic_algorithm import GeneticAlgorithm
from app.schemas import CVRPInstance, DRLConfig, GAConfig


class ExperimentRunner:
    """
    Orchestrates experimental runs for NeuroGen validation.

    Manages:
    - DRL agent loading
    - Population initialization (random vs. DRL-based)
    - GA execution with metric tracking
    - Result collection
    """

    def __init__(
        self,
        ga_config: GAConfig,
        drl_diversity: float = 0.3,
    ) -> None:
        """
        Initialize experiment runner.

        Args:
            ga_config: Configuration for genetic algorithm
            drl_diversity: Exploration parameter for DRL agents (0-1)
        """
        self.ga_config = ga_config
        self.drl_diversity = drl_diversity

        # Agent cache
        self.agents: dict[str, Optional[ActorCriticAgent]] = {
            "junior": None,
            "mid": None,
            "expert": None,
        }

    def load_agent(self, agent_name: str, instance: CVRPInstance) -> ActorCriticAgent:
        """
        Load pre-trained DRL agent.

        Args:
            agent_name (str): Agent identifier (junior, mid, expert)
            instance (CVRPInstance): CVRP instance (for environment setup)

        Returns:
            Loaded ActorCriticAgent
        """
        cache_key = f"{agent_name}_{instance.num_customers}"
        
        if cache_key in self.agents and self.agents[cache_key] is not None:
            self.agents[cache_key].instance = instance
            self.agents[cache_key].env.instance = instance
            self.agents[cache_key].env.reset()
            return self.agents[cache_key]

        # Load checkpoint
        checkpoint_path = Path(f"app/core/drl/checkpoints/{agent_name}.pth")

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Agent checkpoint not found: {checkpoint_path}\n"
                f"Please train the {agent_name} agent first using:\n"
                f"  python -m app.scripts.drl.train_{agent_name}"
            )

        # Create agent with default config
        config = DRLConfig(
            episodes=0,  # No training
            learning_rate_actor=3e-4,
            learning_rate_critic=1e-4,
        )

        agent = ActorCriticAgent(instance, config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.network.load_state_dict(checkpoint["network"])
        agent.epsilon = 0.0  # Disable exploration by default

        print(f"  ✓ Loaded {agent_name} agent (size={instance.num_customers}) from {checkpoint_path}")

        # Cache agent with size-specific key
        self.agents[cache_key] = agent

        return agent

    def run_ga_pure(
        self,
        instance: CVRPInstance,
        replicate: int,
        seed: int,
    ) -> ExperimentResult:
        """
        Run GA with pure random initialization (baseline).

        Args:
            instance (CVRPInstance): CVRP instance to solve
            replicate (int): Replicate number
            seed (int): Random seed

        Returns:
            ExperimentResult with comprehensive metrics
        """
        # Update seed in config
        config = self.ga_config.model_copy()
        config.seed = seed

        # Track initialization time
        init_start = time.time()

        ga = GeneticAlgorithm(instance=instance, config=config)
        ga.initialize_population(custom_population=None, use_heuristics=False)

        initial_cost = ga.get_best_individual().fitness
        init_time = time.time() - init_start

        # Track evolution time
        evolution_start = time.time()

        # Run evolution (already initialized)
        ga.best_individual = ga.get_best_individual()
        ga.convergence_history.append(ga.best_individual.fitness)

        for generation in range(config.generations):
            ga.run_generation()

            current_best = ga.get_best_individual()
            if current_best.fitness < ga.best_individual.fitness:
                ga.best_individual = current_best

            ga.convergence_history.append(ga.best_individual.fitness)

        evolution_time = time.time() - evolution_start

        # Convert to solution
        solution = ga.best_individual.to_solution(algorithm="ga_pure")
        final_cost = solution.total_cost

        # Calculate convergence point (20 generations without improvement)
        convergence_gen = self._find_convergence_point(ga.convergence_history)

        # Build result
        result = ExperimentResult(
            instance_id=instance.id,
            instance_name=instance.name,
            instance_size=instance.num_customers,
            configuration="ga_pure",
            agent_used=None,
            replicate=replicate,
            seed=seed,
            initial_cost=initial_cost,
            final_cost=final_cost,
            improvement_gap=((initial_cost - final_cost) / initial_cost * 100),
            total_time=init_time + evolution_time,
            initialization_time=init_time,
            evolution_time=evolution_time,
            generations_run=config.generations,
            generations_to_convergence=convergence_gen,
            convergence_history=ga.convergence_history,
            best_solution=solution,
        )

        return result

    def run_drl_ga(
        self,
        instance: CVRPInstance,
        agent_name: str,
        replicate: int,
        seed: int,
    ) -> ExperimentResult:
        """
        Run GA with DRL-based initialization (hybrid approach).

        Args:
            instance (CVRPInstance): CVRP instance to solve
            agent_name (str): DRL agent to use (junior, mid, expert)
            replicate (int): Replicate number
            seed (int): Random seed

        Returns:
            ExperimentResult with comprehensive metrics
        """
        # Update seed in config
        config = self.ga_config.model_copy()
        config.seed = seed

        # Load agent
        agent = self.load_agent(agent_name, instance)

        # Track initialization time
        init_start = time.time()

        # Generate population with DRL
        population = generate_population_with_drl(
            instance=instance,
            agent=agent,
            population_size=config.population_size,
            diversity=self.drl_diversity,
        )

        initial_cost = min(ind.fitness for ind in population)
        init_time = time.time() - init_start

        # Track evolution time
        evolution_start = time.time()

        # Run GA evolution
        ga = GeneticAlgorithm(instance=instance, config=config)
        ga.initialize_population(custom_population=population, use_heuristics=False)

        ga.best_individual = ga.get_best_individual()
        ga.convergence_history.append(ga.best_individual.fitness)

        for generation in range(config.generations):
            ga.run_generation()

            current_best = ga.get_best_individual()
            if current_best.fitness < ga.best_individual.fitness:
                ga.best_individual = current_best

            ga.convergence_history.append(ga.best_individual.fitness)

        evolution_time = time.time() - evolution_start

        # Convert to solution
        configuration = f"drl_{agent_name}_ga"
        solution = ga.best_individual.to_solution(algorithm=configuration)
        final_cost = solution.total_cost

        # Calculate convergence point
        convergence_gen = self._find_convergence_point(ga.convergence_history)

        # Build result
        result = ExperimentResult(
            instance_id=instance.id,
            instance_name=instance.name,
            instance_size=instance.num_customers,
            configuration=configuration,
            agent_used=agent_name,
            replicate=replicate,
            seed=seed,
            initial_cost=initial_cost,
            final_cost=final_cost,
            improvement_gap=((initial_cost - final_cost) / initial_cost * 100),
            total_time=init_time + evolution_time,
            initialization_time=init_time,
            evolution_time=evolution_time,
            generations_run=config.generations,
            generations_to_convergence=convergence_gen,
            convergence_history=ga.convergence_history,
            best_solution=solution,
        )

        return result

    def _find_convergence_point(
        self,
        history: list[float],
        stagnation_window: int = 20,
    ) -> Optional[int]:
        """
        Find generation where algorithm converged (stagnated).

        Args:
            history (list[float]): List of best fitness per generation
            stagnation_window (int): Number of generations without improvement

        Returns:
            Generation number where convergence occurred, or None
        """
        if len(history) < stagnation_window:
            return None

        for i in range(len(history) - stagnation_window):
            window = history[i : i + stagnation_window]

            # Check if all values are the same (no improvement)
            if len(set(window)) == 1:
                return i

        return None
