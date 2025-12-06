"""
Training script for Mid Level DRL Agent (Regional Distribution).

Scenario: Regional/national courier services.
Real-world equivalent: Regional distributors, medium-scale logistics companies,
                        cross-city delivery networks.

Characteristics:
- Medium scale: 40-100 customers per route
- Mix of urban, suburban, and some rural areas
- Medium trucks with good capacity
- Balanced exploration and exploitation

run with: `python -m app.scripts.drl.train_mid`
"""

import random

import numpy as np

from app.api.routers.instances.payload_schemas import (
    GenerateClusteredInstanceRequest,
    GenerateRandomInstanceRequest,
)
from app.config import settings
from app.core.drl.actor_critic_agent import ActorCriticAgent
from app.core.drl.cvrp_environment import CVRPEnvironment
from app.core.operations.instances import (
    generate_clustered_instance,
    generate_random_instance,
)
from app.schemas.drl_config import DRLConfig


def train_mid_agent():
    """
    Train Mid-level agent for regional distribution scenarios.

    Training strategy:
    - 50% random instances: Handles diverse geographical spread
    - 50% clustered instances: Learns regional patterns
    - Moderate learning rate: Stable convergence
    """
    print("=" * 80)
    print("MID-LEVEL AGENT TRAINING - Regional Distribution")
    print("=" * 80)
    print("Scenario: Regional courier services, cross-city logistics")
    print("Customer range: 40-100 customers")
    print("Vehicle capacity: 120-180 units")
    print("Episodes: 6000")
    print("Instance distribution: 50% random, 50% clustered")
    print("=" * 80)
    print()

    checkpoint_path = settings.CHECKPOINTS_DIR / "mid.pth"

    # Mid-level config: Balanced learning
    config = DRLConfig(
        episodes=1,  # Handle episodes manually for mixed instances
        learning_rate_actor=5e-4,  # Moderate learning rate
        learning_rate_critic=2.5e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9998,  # Slower decay for better exploration
        gamma=0.99,
        device=settings.DRL_DEVICE,
    )

    total_episodes = 6000  # Balanced for learning vs time
    random_ratio = 0.50  # 50-50 split

    # Instance generation parameters
    gen_params = {
        "min_cust": 40,
        "max_cust": 100,
        "min_cap": 120,
        "max_cap": 180,
        "min_dem": 5,
        "max_dem": 25,
        "min_grid": 100,
        "max_grid": 200,
    }

    print("Generating fixed validation set (Benchmark)...")
    print()

    validation_set = []
    validation_sizes = [40, 48, 55, 62, 68, 74, 80, 86, 93, 100]
    for i, num_cust in enumerate(validation_sizes):
        params = GenerateRandomInstanceRequest(
            num_customers=num_cust,
            grid_size=150,
            vehicle_capacity=150,
            min_customer_demand=gen_params["min_dem"],
            max_customer_demand=gen_params["max_dem"],
            seed=1000 + i,
        )
        validation_set.append(generate_random_instance(params))

    initial_instance = validation_set[0]
    agent = ActorCriticAgent(instance=initial_instance, config=config)

    best_val_avg = float("inf")
    episode_costs = []

    print("Starting training...")
    print()

    for episode in range(1, total_episodes + 1):
        num_customers = random.randint(gen_params["min_cust"], gen_params["max_cust"])
        vehicle_capacity = random.randint(gen_params["min_cap"], gen_params["max_cap"])
        grid_size = random.randint(gen_params["min_grid"], gen_params["max_grid"])

        use_random = random.random() < random_ratio

        if use_random:
            params = GenerateRandomInstanceRequest(
                num_customers=num_customers,
                grid_size=grid_size,
                vehicle_capacity=vehicle_capacity,
                min_customer_demand=gen_params["min_dem"],
                max_customer_demand=gen_params["max_dem"],
                seed=episode,
            )
            instance = generate_random_instance(params)
            instance_type = "Random"
        else:
            num_clusters = random.randint(3, 6)
            params = GenerateClusteredInstanceRequest(
                num_customers=num_customers,
                grid_size=grid_size,
                vehicle_capacity=vehicle_capacity,
                min_customer_demand=gen_params["min_dem"],
                max_customer_demand=gen_params["max_dem"],
                num_clusters=num_clusters,
                seed=episode,
            )
            instance = generate_clustered_instance(params)
            instance_type = f"Clustered({num_clusters})"

        agent.instance = instance
        agent.env = CVRPEnvironment(instance)

        # Train 1 episode per instance for better generalization
        agent.train(episodes=1)

        current_train_cost = (
            agent.episode_costs[-1] if agent.episode_costs else float("inf")
        )
        episode_costs.append(current_train_cost)

        # Validation every 100 episodes
        if episode % 100 == 0:
            val_costs = []

            for val_inst in validation_set:
                cost, _ = agent.solve(val_inst)
                val_costs.append(cost)

            current_val_avg = np.mean(val_costs)

            if current_val_avg < best_val_avg:
                best_val_avg = current_val_avg
                agent.save(str(checkpoint_path))
                print()
                print(f">>> NEW BEST MODEL SAVED! Val Avg: {best_val_avg:.2f}")
                print()

            avg_train_last_100 = np.mean(episode_costs[-100:])
            print(f"Episode {episode}/{total_episodes}")
            print(f" Train Inst: {num_customers} cust, {instance_type}")
            print(
                f" Train Cost: {current_train_cost:.2f} | Avg (100): {avg_train_last_100:.2f}"
            )
            print(f" Validation Avg: {current_val_avg:.2f} (Best: {best_val_avg:.2f})")
            print(f" Epsilon: {agent.epsilon:.4f}")
            print("-" * 40)

    print()
    print("Training completed!")
    print(f"Best Validation Average: {best_val_avg:.2f}")

    agent.save(str(checkpoint_path))
    print(f"Final model saved to: {checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    train_mid_agent()
