from typing import Optional

from app.core.drl.actor_critic_agent import ActorCriticAgent
from app.core.ga.individual import Individual
from app.schemas import CVRPInstance, DRLConfig


def generate_population_with_drl(
    instance: CVRPInstance,
    agent: ActorCriticAgent,
    population_size: int = 50,
    diversity: float = 0.3,
) -> list[Individual]:
    """
    Generate initial population using trained DRL agent.

    This is the core of the NeuroGen approach: using DRL to avoid
    the "cold start" problem in genetic algorithms.

    Args:
        instance (CVRPInstance): CVRP instance
        agent (ActorCriticAgent): Trained DRL agent
        population_size (int): Number of individuals to generate
        diversity (float): Exploration parameter (0-1, higher = more diverse)
    Returns:
        list[Individual]: List of Individual objects
    """
    population = []

    for _ in range(population_size):
        # Use exploration to ensure diversity
        explore = agent.epsilon > 0 or diversity > 0

        # Temporarily adjust epsilon for diversity
        original_epsilon = agent.epsilon
        if diversity > 0:
            agent.epsilon = diversity

        solution = agent.generate_solution(explore=explore)

        # Restore epsilon
        agent.epsilon = original_epsilon

        # Extract customer sequence from solution
        customer_sequence = []
        for route in solution.routes:
            customer_sequence.extend(route.customer_sequence)

        # Create Individual
        individual = Individual.from_giant_tour(customer_sequence, instance)
        population.append(individual)

    return population


def get_trained_agent(
    instance: CVRPInstance,
    config: Optional[DRLConfig] = None,
    callback=None,
) -> ActorCriticAgent:
    """
    Train an Actor-Critic DRL agent for CVRP instance.

    Args:
        instance (CVRPInstance): CVRP instance
        config (Optional[DRLConfig]): DRL configuration (uses defaults if None)
        callback (Optional[Callable]): Optional callback(episode, cost)

    Returns:
        ActorCriticAgent: Trained DRL agent
    """
    if config is None:
        config = DRLConfig()

    agent = ActorCriticAgent(instance, config, callback)
    agent.train()

    return agent
