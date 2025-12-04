"""
Training script for Expert Level DRL Agent (Industrial-Scale Operations).

Scenario: Large-scale industrial logistics operations.
Real-world equivalent: Amazon Prime, FedEx, UPS, DHL - highly optimized urban delivery.

Characteristics:
- Large scale: 80-150 customers per route
- Highly clustered urban environments (neighborhoods, business districts)
- Large trucks with high capacity
- Conservative learning for stability and fine optimization
- Long training for superior performance
"""

import random

import numpy as np

from app.config import settings
from app.core.drl.actor_critic_agent import ActorCriticAgent
from app.core.drl.cvrp_environment import CVRPEnvironment
from app.core.operations.instances import (
    generate_clustered_instance,
    generate_random_instance,
)
from app.schemas.drl_config import DRLConfig


def train_expert_agent():
    """
    Train Expert agent for industrial-scale logistics operations.

    Training strategy:
    - 20% random instances: Handles edge cases and irregular patterns
    - 80% clustered instances: Reflects real-world urban delivery (neighborhoods)
    - Low learning rate: Fine-tuned optimization
    - Extended training: Achieves superior performance
    """
    print("=" * 80)
    print("EXPERT AGENT TRAINING - Industrial-Scale Logistics")
    print("=" * 80)
    print("Scenario: Amazon Prime, FedEx, UPS - optimized urban delivery")
    print("Customer range: 80-150 customers")
    print("Vehicle capacity: 180-250 units")
    print("Episodes: 8000")
    print("Instance distribution: 20% random, 80% clustered")
    print("=" * 80)
    print()

    checkpoint_path = settings.CHECKPOINTS_DIR / "expert.pth"

    # Expert config: Conservative, stable learning
    config = DRLConfig(
        episodes=1,  # Handle episodes manually for mixed instances
        learning_rate_actor=1e-4,  # Conservative learning
        learning_rate_critic=5e-5,
        epsilon_start=1.0,
        epsilon_end=0.01,  # Very low final exploration
        epsilon_decay=0.9999,  # Very slow decay for thorough exploration
        gamma=0.99,
        device=settings.DRL_DEVICE,
    )

    # Training configuration
    total_episodes = 8000  # Balanced for expert-level learning
    random_ratio = 0.20  # 20% random, 80% clustered (realistic)

    # Instance generation parameters
    gen_params = {
        "min_cust": 80,
        "max_cust": 150,
        "min_cap": 180,
        "max_cap": 250,
        "min_dem": 5,
        "max_dem": 30,
        "min_grid": 150,
        "max_grid": 300,
    }

    print("Starting training...")
    print("WARNING: This will take significant time due to large problem sizes.")
    print()

    print("Generating fixed validation set (Benchmark)...")
    print()

    validation_set = []
    validation_sizes = [80, 90, 100, 110, 118, 126, 133, 140, 145, 150]
    for i, num_cust in enumerate(validation_sizes):
        validation_set.append(
            generate_clustered_instance(  # Validate mainly on clustered (primary scenario)
                num_customers=num_cust,
                grid_size=225,
                vehicle_capacity=215,
                min_customer_demand=gen_params["min_dem"],
                max_customer_demand=gen_params["max_dem"],
                num_clusters=6,
                seed=1000 + i,
            )
        )

    initial_instance = validation_set[0]
    agent = ActorCriticAgent(instance=initial_instance, config=config)

    best_val_avg = float("inf")
    episode_costs = []

    for episode in range(1, total_episodes + 1):
        num_customers = random.randint(gen_params["min_cust"], gen_params["max_cust"])
        vehicle_capacity = random.randint(gen_params["min_cap"], gen_params["max_cap"])
        grid_size = random.randint(gen_params["min_grid"], gen_params["max_grid"])

        use_random = random.random() < random_ratio

        if use_random:
            instance = generate_random_instance(
                num_customers=num_customers,
                grid_size=grid_size,
                vehicle_capacity=vehicle_capacity,
                min_customer_demand=gen_params["min_dem"],
                max_customer_demand=gen_params["max_dem"],
                seed=episode,
            )
            instance_type = "Random"
        else:
            num_clusters = random.randint(5, 10)
            instance = generate_clustered_instance(
                num_customers=num_customers,
                grid_size=grid_size,
                vehicle_capacity=vehicle_capacity,
                min_customer_demand=gen_params["min_dem"],
                max_customer_demand=gen_params["max_dem"],
                num_clusters=num_clusters,
                seed=episode,
            )
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

        # Checkpoint every 500 episodes (important for long training)
        if episode % 500 == 0:
            checkpoint_path = (
                settings.CHECKPOINTS_DIR / f"expert_checkpoint_{episode}.pth"
            )
            agent.save(str(checkpoint_path))
            print(f" Checkpoint saved: {checkpoint_path.name}")
            print()

    print()
    print("Training completed!")
    print(f"Best Validation Average: {best_val_avg:.2f}")

    agent.save(str(checkpoint_path))
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    train_expert_agent()
