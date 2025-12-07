"""
Training script for Junior Level DRL Agent (Last-Mile Delivery).

Scenario: Local courier services - food delivery, pharmacy, local parcels.
Real-world equivalent: Small local delivery companies, restaurant delivery, pharmacy runs.

Characteristics:
- Small scale: 15-50 customers per route
- Urban/suburban environment with moderate clustering
- Small vehicles (vans, small trucks)
- Quick learning required for dynamic environments

run with: `python -m app.scripts.drl.train_junior`
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


def train_junior_agent():
    """
    Train Junior agent for last-mile delivery scenarios.

    Training strategy:
    - 60% random instances: Handles unpredictable delivery locations
    - 40% clustered instances: Learns neighborhood patterns
    - Fast learning rate: Adapts quickly to dynamic demand
    """
    print("=" * 80)
    print("JUNIOR AGENT TRAINING - Last-Mile Delivery")
    print("=" * 80)
    print("Scenario: Local courier (food delivery, pharmacy, local parcels)")
    print("Customer range: 15-50 customers")
    print("Vehicle capacity: 80-100 units")
    print("Episodes: 6000")
    print("Instance distribution: 60% random, 40% clustered")
    print("=" * 80)
    print()

    checkpoint_path = settings.CHECKPOINTS_DIR / "junior.pth"

    # Junior config: Fast learning, moderate exploration
    config = DRLConfig(
        episodes=1,  # Handle episodes manually for mixed instances
        learning_rate_actor=1e-3,  # Fast learning
        learning_rate_critic=5e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,  # Moderate decay
        gamma=0.99,
        device=settings.DRL_DEVICE,
    )

    total_episodes = 6000  # Balanced for learning vs time
    random_ratio = 0.60  # 60% random instances

    # Instance generation parameters
    gen_params = {
        "min_cust": 15,
        "max_cust": 50,
        "min_cap": 80,
        "max_cap": 100,
        "min_dem": 3,
        "max_dem": 15,
        "min_grid": 50,
        "max_grid": 100,
    }

    print("Generating fixed validation set (Benchmark)...")
    print()

    validation_set = []
    validation_sizes = [15, 20, 25, 30, 32, 35, 38, 42, 46, 50]

    # Add random instances
    for i, num_cust in enumerate(validation_sizes):
        params = GenerateRandomInstanceRequest(
            num_customers=num_cust,
            grid_size=100,
            vehicle_capacity=90,
            min_customer_demand=gen_params["min_dem"],
            max_customer_demand=gen_params["max_dem"],
            seed=1000 + i,
        )
        validation_set.append(generate_random_instance(params, save=False))

    # Add clustered instances
    for i, num_cust in enumerate(validation_sizes):
        params = GenerateClusteredInstanceRequest(
            num_customers=num_cust,
            grid_size=100,
            vehicle_capacity=90,
            min_customer_demand=gen_params["min_dem"],
            max_customer_demand=gen_params["max_dem"],
            num_clusters=3,
            seed=2000 + i,
        )
        validation_set.append(generate_clustered_instance(params, save=False))

    initial_instance = validation_set[0]
    agent = ActorCriticAgent(instance=initial_instance, config=config)

    best_val_avg = float("inf")
    episode_costs = []

    print("Starting training...")

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
            instance = generate_random_instance(params, save=False)
            instance_type = "Random"
        else:
            num_clusters = random.randint(2, 4)
            params = GenerateClusteredInstanceRequest(
                num_customers=num_customers,
                grid_size=grid_size,
                vehicle_capacity=vehicle_capacity,
                min_customer_demand=gen_params["min_dem"],
                max_customer_demand=gen_params["max_dem"],
                num_clusters=num_clusters,
                seed=episode,
            )
            instance = generate_clustered_instance(params, save=False)
            instance_type = f"Clustered({num_clusters})"

        agent.instance = instance
        agent.env = CVRPEnvironment(instance)

        # Train 1 episode per instance for better generalization
        agent.train(episodes=1)

        current_train_cost = (
            agent.episode_costs[-1] if agent.episode_costs else float("inf")
        )
        episode_costs.append(current_train_cost)

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

        # Checkpoint every 1000 episodes
        if episode % 1000 == 0:
            intermediate_checkpoint = (
                settings.CHECKPOINTS_DIR / f"junior_checkpoint_{episode}.pth"
            )
            agent.save(str(intermediate_checkpoint))
            print(f" Checkpoint saved: {intermediate_checkpoint.name}")
            print()

    print()
    print("Training completed!")
    print(f"Best Validation Average: {best_val_avg:.2f}")

    agent.save(str(checkpoint_path))
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    train_junior_agent()
