MODELS = {
    "junior": {
        "name": "Junior Level Agent",
        "subname": "Last-Mile Delivery",
        "description": (
            "Specialized in local courier services such as food delivery, pharmacy runs, "
            "and local parcels. Designed for small-scale operations (15-50 customers) "
            "in urban/suburban environments using small vehicles (80-100 capacity). "
            "Ideal for dynamic environments requiring quick adaptation."
        ),
        "training_summary": (
            "Trained over 6,000 episodes using an Actor-Critic architecture with Pointer Networks. "
            "The training strategy utilized a 60/40 split between random and clustered instances "
            "to ensure robustness against unpredictable delivery locations while learning neighborhood patterns. "
            "Validation was performed every 100 episodes against a graduated benchmark set of 20 instances (10 random + 10 clustered) "
            "spanning sizes 15-50 customers to ensure generalization capability across the full difficulty range."
        ),
        "training_specs": {
            "algorithm": "Actor-Critic with Pointer Network",
            "total_episodes": 6000,
            "instance_distribution": "60% Random / 40% Clustered",
            "problem_size": "15-50 Customers",
            "vehicle_capacity": "80-100 units",
            "learning_rate": "Actor: 1e-3, Critic: 5e-4",
            "epsilon_decay": "0.9995 (moderate exploration decay)",
            "validation_set": "20 instances (10 random + 10 clustered)",
            "validation_frequency": "Every 100 episodes",
            "checkpoints": "Every 1,000 episodes",
        },
    },
    "mid": {
        "name": "Mid-Level Agent",
        "subname": "Regional Distribution",
        "description": (
            "Optimized for regional courier services and cross-city logistics. "
            "Handles medium-scale operations (40-100 customers) with vehicles "
            "of 120-180 capacity. Suitable for diverse geographical spreads and "
            "regional delivery patterns."
        ),
        "training_summary": (
            "Trained over 10,000 episodes using an Actor-Critic architecture with Pointer Networks. "
            "The training utilized a balanced 50/50 split between random and clustered instances to capture "
            "diverse regional layouts and delivery patterns. Validation was performed every 100 episodes against "
            "a comprehensive benchmark set of 20 instances (10 random + 10 clustered) spanning the entire problem "
            "size range (40-100 customers) to ensure consistent performance across all difficulty levels."
        ),
        "training_specs": {
            "algorithm": "Actor-Critic with Pointer Network",
            "total_episodes": 10000,
            "instance_distribution": "50% Random / 50% Clustered",
            "problem_size": "40-100 Customers",
            "vehicle_capacity": "120-180 units",
            "learning_rate": "Actor: 4e-4, Critic: 2e-4",
            "epsilon_decay": "0.9997 (slower exploration decay for better learning)",
            "validation_set": "20 instances (10 random + 10 clustered)",
            "validation_frequency": "Every 100 episodes",
            "checkpoints": "Every 1,500 episodes",
        },
    },
    "expert": {
        "name": "Expert Level Agent",
        "subname": "Industrial-Scale Logistics",
        "description": (
            "Tailored for large-scale logistics operations such as those of Amazon Prime, "
            "handling extensive delivery networks with 100-150 customers and vehicle capacities of 200-250. "
            "Designed for complex, multi-regional distribution requiring advanced route optimization."
        ),
        "training_summary": (
            "Trained over 25,000 episodes using an Actor-Critic architecture with Pointer Networks and curriculum learning. "
            "The training incorporated a 20/80 split between random and clustered instances to reflect real-world urban delivery patterns. "
            "Used a 3-phase curriculum: Phase 1 (50-90 customers), Phase 2 (80-120 customers), and Phase 3 (100-150 customers). "
            "Validation was performed every 100 episodes against a weighted benchmark set of 24 instances (8 random + 16 clustered) "
            "maintaining a 1:2 ratio to reflect realistic deployment scenarios, spanning 100-150 customers."
        ),
        "training_specs": {
            "algorithm": "Actor-Critic with Pointer Network + Curriculum Learning",
            "total_episodes": 25000,
            "instance_distribution": "20% Random / 80% Clustered (realistic ratio)",
            "problem_size": "100-150 Customers",
            "vehicle_capacity": "200-250 units",
            "learning_rate": "Actor: 3e-4, Critic: 1e-5 (conservative for stability)",
            "epsilon_decay": "0.99985 (very slow decay for thorough exploration)",
            "curriculum_phases": "3 phases: 50-90, 80-120, 100-150 customers",
            "validation_set": "24 instances (8 random + 16 clustered, 1:2 ratio)",
            "validation_frequency": "Every 100 episodes",
            "checkpoints": "Every 5,000 episodes",
        },
    },
}
