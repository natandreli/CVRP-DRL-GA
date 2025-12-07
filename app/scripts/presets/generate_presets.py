"""
Script to generate preset CVRP instances for the application.

These presets will be available to all users and persist across server restarts.

Run with: uv run python -m app.scripts.presets.generate_presets
"""

import json
import random
from pathlib import Path

import numpy as np

from app.schemas.cvrp_instance import Customer, CVRPInstance, Location

PRESETS = [
    # Junior Level (Last-Mile Delivery)
    {
        "id": "preset_junior_random_25",
        "name": "Junior Random - 25 customers",
        "description": "Small random instance for local delivery scenarios (25 customers)",
        "num_customers": 25,
        "vehicle_capacity": 100,
        "grid_size": 100,
        "min_demand": 5,
        "max_demand": 15,
        "clustered": False,
        "num_clusters": None,
        "seed": 42,
    },
    {
        "id": "preset_junior_clustered_25",
        "name": "Junior Clustered - 25 customers (3 zones)",
        "description": "Clustered instance for neighborhood delivery (25 customers, 3 zones)",
        "num_customers": 25,
        "vehicle_capacity": 100,
        "grid_size": 100,
        "min_demand": 5,
        "max_demand": 15,
        "clustered": True,
        "num_clusters": 3,
        "seed": 4201,
    },
    {
        "id": "preset_junior_clustered_30",
        "name": "Junior Clustered - 30 customers (3 zones)",
        "description": "Clustered instance representing neighborhood delivery (30 customers, 3 zones)",
        "num_customers": 30,
        "vehicle_capacity": 100,
        "grid_size": 100,
        "min_demand": 5,
        "max_demand": 15,
        "clustered": True,
        "num_clusters": 3,
        "seed": 123,
    },
    # Scalability - Small instances
    {
        "id": "preset_scalability_random_20",
        "name": "Scalability Random - 20 customers",
        "description": "Very small instance for scalability testing (20 customers)",
        "num_customers": 20,
        "vehicle_capacity": 100,
        "grid_size": 100,
        "min_demand": 5,
        "max_demand": 15,
        "clustered": False,
        "num_clusters": None,
        "seed": 2001,
    },
    # Mid Level (Regional Distribution)
    {
        "id": "preset_mid_random_50",
        "name": "Mid Random - 50 customers",
        "description": "Medium random instance for regional distribution (50 customers)",
        "num_customers": 50,
        "vehicle_capacity": 150,
        "grid_size": 150,
        "min_demand": 5,
        "max_demand": 20,
        "clustered": False,
        "num_clusters": None,
        "seed": 456,
    },
    {
        "id": "preset_mid_clustered_50",
        "name": "Mid Clustered - 50 customers (4 districts)",
        "description": "Clustered instance for multi-district delivery (50 customers, 4 districts)",
        "num_customers": 50,
        "vehicle_capacity": 150,
        "grid_size": 150,
        "min_demand": 5,
        "max_demand": 20,
        "clustered": True,
        "num_clusters": 4,
        "seed": 4501,
    },
    {
        "id": "preset_mid_clustered_60",
        "name": "Mid Clustered - 60 customers (4 districts)",
        "description": "Clustered instance for multi-district delivery (60 customers, 4 districts)",
        "num_customers": 60,
        "vehicle_capacity": 150,
        "grid_size": 150,
        "min_demand": 5,
        "max_demand": 20,
        "clustered": True,
        "num_clusters": 4,
        "seed": 789,
    },
    # Scalability - Medium-Large instances
    {
        "id": "preset_scalability_random_75",
        "name": "Scalability Random - 75 customers",
        "description": "Medium-large instance for scalability testing (75 customers)",
        "num_customers": 75,
        "vehicle_capacity": 175,
        "grid_size": 175,
        "min_demand": 8,
        "max_demand": 25,
        "clustered": False,
        "num_clusters": None,
        "seed": 7501,
    },
    # Expert Level (Industrial Logistics)
    {
        "id": "preset_expert_random_100",
        "name": "Expert Random - 100 customers",
        "description": "Large random instance for industrial logistics (100 customers)",
        "num_customers": 100,
        "vehicle_capacity": 200,
        "grid_size": 200,
        "min_demand": 10,
        "max_demand": 30,
        "clustered": False,
        "num_clusters": None,
        "seed": 1001,
    },
    {
        "id": "preset_expert_clustered_100",
        "name": "Expert Clustered - 100 customers (5 regions)",
        "description": "Clustered instance for multi-region supply chain (100 customers, 5 regions)",
        "num_customers": 100,
        "vehicle_capacity": 200,
        "grid_size": 200,
        "min_demand": 10,
        "max_demand": 30,
        "clustered": True,
        "num_clusters": 5,
        "seed": 10001,
    },
    {
        "id": "preset_expert_clustered_120",
        "name": "Expert Clustered - 120 customers (5 regions)",
        "description": "Clustered instance for multi-region supply chain (120 customers, 5 regions)",
        "num_customers": 120,
        "vehicle_capacity": 200,
        "grid_size": 200,
        "min_demand": 10,
        "max_demand": 30,
        "clustered": True,
        "num_clusters": 5,
        "seed": 2002,
    },
    # Scalability - Very Large instance
    {
        "id": "preset_scalability_random_150",
        "name": "Scalability Random - 150 customers",
        "description": "Very large instance for scalability testing (150 customers)",
        "num_customers": 150,
        "vehicle_capacity": 250,
        "grid_size": 250,
        "min_demand": 10,
        "max_demand": 35,
        "clustered": False,
        "num_clusters": None,
        "seed": 15001,
    },
]


def _generate_instance(preset: dict) -> CVRPInstance:
    """Generate instance based on preset configuration."""
    random.seed(preset["seed"])
    np.random.seed(preset["seed"])

    depot = Location(x=preset["grid_size"] / 2, y=preset["grid_size"] / 2)
    customers = []

    if preset["clustered"]:
        cluster_centers = []
        for _ in range(preset["num_clusters"]):
            center = Location(
                x=random.uniform(preset["grid_size"] * 0.2, preset["grid_size"] * 0.8),
                y=random.uniform(preset["grid_size"] * 0.2, preset["grid_size"] * 0.8),
            )
            cluster_centers.append(center)

        customers_per_cluster = preset["num_customers"] // preset["num_clusters"]
        customer_id = 1

        for cluster_center in cluster_centers:
            for _ in range(customers_per_cluster):
                offset_x = random.gauss(0, preset["grid_size"] * 0.1)
                offset_y = random.gauss(0, preset["grid_size"] * 0.1)
                location = Location(
                    x=max(0, min(preset["grid_size"], cluster_center.x + offset_x)),
                    y=max(0, min(preset["grid_size"], cluster_center.y + offset_y)),
                )
                demand = random.randint(preset["min_demand"], preset["max_demand"])
                customers.append(
                    Customer(id=customer_id, location=location, demand=demand)
                )
                customer_id += 1

        remaining = preset["num_customers"] - len(customers)
        for _ in range(remaining):
            location = Location(
                x=random.uniform(0, preset["grid_size"]),
                y=random.uniform(0, preset["grid_size"]),
            )
            demand = random.randint(preset["min_demand"], preset["max_demand"])
            customers.append(Customer(id=customer_id, location=location, demand=demand))
            customer_id += 1
    else:
        for i in range(1, preset["num_customers"] + 1):
            location = Location(
                x=random.uniform(0, preset["grid_size"]),
                y=random.uniform(0, preset["grid_size"]),
            )
            demand = random.randint(preset["min_demand"], preset["max_demand"])
            customers.append(Customer(id=i, location=location, demand=demand))

    return CVRPInstance(
        id=preset["id"],
        type="preset",
        name=preset["name"],
        description=preset["description"],
        depot=depot,
        customers=customers,
        vehicle_capacity=preset["vehicle_capacity"],
        max_vehicles=None,
        distance_matrix=None,
        seed=preset["seed"],
        num_clusters=preset["num_clusters"],
    )


def generate_presets():
    """Generate all preset instances and save them to the presets directory."""
    project_root = Path(__file__).parent.parent.parent.parent
    presets_dir = project_root / "app" / "data" / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Presets directory: {presets_dir}")
    print(f"Generating {len(PRESETS)} preset instances...")
    print("=" * 70)

    for preset in PRESETS:
        print(f"\nGenerating: {preset['name']}")
        print(f"  ID: {preset['id']}")
        print(f"  Customers: {preset['num_customers']}")
        print(f"  Type: {'Clustered' if preset['clustered'] else 'Random'}")

        try:
            instance = _generate_instance(preset)

            filepath = presets_dir / f"{instance.id}.json"
            with open(filepath, "w") as f:
                json.dump(instance.model_dump(), f, indent=2)

            print(f"  ✓ Saved to: {filepath}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"✓ Generated {len(PRESETS)} presets successfully!")
    print(f"Presets are stored in: {presets_dir}")


if __name__ == "__main__":
    generate_presets()
