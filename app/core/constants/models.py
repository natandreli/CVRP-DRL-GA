MODELS = {
    "junior": {
        "name": "Junior Level Agent (Last-Mile Delivery)",
        "description": (
            "Specialized in local courier services such as food delivery, pharmacy runs, "
            "and local parcels. Designed for small-scale operations (15-50 customers) "
            "in urban/suburban environments using small vehicles (80-100 capacity). "
            "Ideal for dynamic environments requiring quick adaptation."
        ),
        "training_summary": (
            "Trained over 4,000 episodes using an Actor-Critic architecture with Pointer Networks. "
            "The training strategy utilized a 60/40 split between random and clustered instances "
            "to ensure robustness against unpredictable delivery locations while learning neighborhood patterns. "
            "Validation was performed against a fixed benchmark set to ensure generalization capability."
        ),
        "training_specs": {
            "algorithm": "Actor-Critic with Pointer Network",
            "total_episodes": 4000,
            "instance_distribution": "60% Random / 40% Clustered",
            "problem_size": "15-50 Customers",
            "learning_strategy": "Fast learning rate with moderate exploration decay (0.9996)",
            "validation_method": "Greedy evaluation on graduated benchmark set (15-50 nodes)",
        },
    },
    "mid": {
        "name": "Mid-Level Agent (Regional Distribution)",
        "description": (
            "Optimized for regional courier services and cross-city logistics. "
            "Handles medium-scale operations (40-100 customers) with vehicles "
            "of 120-180 capacity. Suitable for diverse geographical spreads and "
            "regional delivery patterns."
        ),
        "training_summary": (
            "Trained over 6,000 episodes using an Actor-Critic architecture with Pointer Networks. "
            "The training strategy employed a balanced 50/50 split between random and clustered instances "
            "to effectively manage diverse geographical distributions while learning regional patterns. "
            "Validation was conducted on a fixed benchmark set to ensure the agent's adaptability."
        ),
        "training_specs": {
            "algorithm": "Actor-Critic with Pointer Network",
            "total_episodes": 6000,
            "instance_distribution": "50% Random / 50% Clustered",
            "problem_size": "40-100 Customers",
            "learning_strategy": "Moderate learning rate with slower exploration decay (0.9998)",
            "validation_method": "Greedy evaluation on graduated benchmark set (40-100 nodes)",
        },
    },
    "expert": {
        "name": "Expert Level Agent (Industrial-Scale Logistics)",
        "description": (
            "Tailored for large-scale logistics operations such as those of Amazon Prime, "
            "handling extensive delivery networks with 80-150 customers and vehicle capacities of 150-200. "
            "Designed for complex, multi-regional distribution requiring advanced route optimization."
        ),
        "training_summary": (
            "Trained over 8,000 episodes using an Actor-Critic architecture with Pointer Networks. "
            "The training incorporated a 20/80 split between random and clustered instances to capture both broad and localized delivery patterns. "
            "Validation was performed on a comprehensive benchmark set to ensure high scalability and efficiency."
        ),
        "training_specs": {
            "algorithm": "Actor-Critic with Pointer Network",
            "total_episodes": 8000,
            "instance_distribution": "20% Random / 80% Clustered",
            "problem_size": "80-150 Customers",
            "learning_strategy": "Slow learning rate with minimal exploration decay (0.9999)",
            "validation_method": "Greedy evaluation on comprehensive benchmark set (80-150 nodes)",
        },
    },
}
