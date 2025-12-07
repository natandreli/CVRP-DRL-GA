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

run with: `python -m app.scripts.drl.train_expert`
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
    print("Strategy: Curriculum Learning (Easy -> Hard -> Expert)")
    print("Episodes: 25,000 (Required for convergence on 150 nodes)")
    print("Validation: 1 Random : 2 Clustered (Weighted for reality)")
    print("Instance distribution: 20% random, 80% clustered")
    print("=" * 80)
    print()

    checkpoint_path = settings.CHECKPOINTS_DIR / "expert.pth"

    # Expert config: Conservative, stable learning
    config = DRLConfig(
        episodes=1,  # Handle episodes manually for mixed instances
        learning_rate_actor=3e-4,  # Conservative learning
        learning_rate_critic=1e-5,
        epsilon_start=1.0,
        epsilon_end=0.02,  # Very low final exploration
        epsilon_decay=0.99985,  # Very slow decay for thorough exploration
        gamma=0.99,
        device=settings.DRL_DEVICE,
    )

    # Training configuration
    total_episodes = 25000  # Balanced for expert-level learning
    random_ratio = 0.20  # 20% random, 80% clustered (realistic)

    curriculum = [
        # Phase 1: Warm-up (50-90 customers)
        {
            "end_episode": 7000,
            "min_customers": 50,
            "max_customers": 90,
            "min_capacity": 150,
            "max_capacity": 200,
        },
        # Phase 2: Scaling (80-120 customers)
        {
            "end_episode": 15000,
            "min_customers": 80,
            "max_customers": 120,
            "min_capacity": 180,
            "max_capacity": 230,
        },
        # Phase 3: Full Expert (100-150 customers)
        {
            "end_episode": 25000,
            "min_customers": 100,
            "max_customers": 150,
            "min_capacity": 200,
            "max_capacity": 250,
        },
    ]

    # Instance generation parameters (for validation and training)
    gen_params = {
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
    validation_sizes = [100, 110, 118, 126, 133, 140, 145, 150]

    # 1. Add random instances (1 per size)
    for i, num_cust in enumerate(validation_sizes):
        params = GenerateRandomInstanceRequest(
            num_customers=num_cust,
            grid_size=225,
            vehicle_capacity=215,
            min_customer_demand=gen_params["min_dem"],
            max_customer_demand=gen_params["max_dem"],
            seed=1000 + i,
        )
        validation_set.append(generate_random_instance(params, save=False))

    # 2. Add clustered instances - Type A (1 per size)
    for i, num_cust in enumerate(validation_sizes):
        params = GenerateClusteredInstanceRequest(
            num_customers=num_cust,
            grid_size=225,
            vehicle_capacity=215,
            min_customer_demand=gen_params["min_dem"],
            max_customer_demand=gen_params["max_dem"],
            num_clusters=6,
            seed=2000 + i,
        )
        validation_set.append(generate_clustered_instance(params, save=False))

    # 3. Add clustered instances - Type B (1 per size) - To maintain 1:2 ratio
    for i, num_cust in enumerate(validation_sizes):
        params = GenerateClusteredInstanceRequest(
            num_customers=num_cust,
            grid_size=225,
            vehicle_capacity=215,
            min_customer_demand=gen_params["min_dem"],
            max_customer_demand=gen_params["max_dem"],
            num_clusters=8,
            seed=3000 + i,
        )
        validation_set.append(generate_clustered_instance(params, save=False))

    initial_instance = validation_set[0]
    agent = ActorCriticAgent(instance=initial_instance, config=config)

    best_val_avg = float("inf")
    episode_costs = []

    for episode in range(1, total_episodes + 1):
        current_phase = curriculum[0]
        for phase in curriculum:
            if episode <= phase["end_episode"]:
                current_phase = phase
                break

        num_customers = random.randint(
            current_phase["min_customers"], current_phase["max_customers"]
        )

        vehicle_capacity = random.randint(
            current_phase["min_capacity"], current_phase["max_capacity"]
        )
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
            num_clusters = random.randint(5, 10)
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

        # Checkpoint every 5000 episodes (important for long training)
        if episode % 5000 == 0:
            intermediate_checkpoint = (
                settings.CHECKPOINTS_DIR / f"expert_checkpoint_{episode}.pth"
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
    train_expert_agent()
