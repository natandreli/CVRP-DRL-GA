import json
from pathlib import Path

from app.api.routers.instances.payload_schemas import (
    GenerateClusteredInstanceRequest,
    GenerateRandomInstanceRequest,
)
from app.config import settings
from app.core.operations.instances import (
    generate_clustered_instance,
    generate_random_instance,
)
from app.schemas import CVRPInstance


def generate_experiment_instances():
    """
    Generate complete experimental dataset.

    Specification:
    - 3 ranges: Junior (20-50), Mid (60-100), Expert (110-150)
    - 4 sizes per range
    - 10 instances per size (5 random + 5 clustered)
    - Total: 120 instances

    Naming convention: <range>_<type>_<size>_<replicate>
    Example: junior_random_30_001, expert_clustered_125_003
    """

    print("=" * 80)
    print("EXPERIMENTAL INSTANCE GENERATION - NeuroGen Validation Dataset")
    print("=" * 80)
    print("Generating stratified synthetic instances for systematic evaluation")
    print()

    # Define experimental design
    experimental_design = {
        "junior": {
            "sizes": [20, 30, 40, 50],
            "demand_range": (3, 15),
            "capacity_multiplier": 1.8,  # Q proportional to sqrt(n) * demand_avg
            "num_clusters": 4,
        },
        "mid": {
            "sizes": [60, 75, 90, 100],
            "demand_range": (5, 25),
            "capacity_multiplier": 2.0,
            "num_clusters": 5,
        },
        "expert": {
            "sizes": [110, 125, 140, 150],
            "demand_range": (5, 30),
            "capacity_multiplier": 2.2,
            "num_clusters": 6,
        },
    }

    base_output_dir = settings.EXPERIMENT_INSTANCES_DIR
    total_instances = 0

    # Generate instances for each range
    for range_name, config in experimental_design.items():
        print(f"\n{'─' * 80}")
        print(f"Range: {range_name.upper()}")
        print(f"{'─' * 80}")

        range_dir = base_output_dir / range_name
        range_dir.mkdir(parents=True, exist_ok=True)

        for size in config["sizes"]:
            print(f"\n  Size: {size} customers")

            # Calculate capacity (proportional to sqrt(n) * avg_demand)
            avg_demand = sum(config["demand_range"]) / 2
            base_capacity = int(
                avg_demand * (size**0.5) * config["capacity_multiplier"]
            )

            # Ensure at least min_vehicles constraint is reasonable
            total_demand_estimate = size * avg_demand
            min_vehicles = max(3, int(total_demand_estimate / base_capacity))
            vehicle_capacity = int(total_demand_estimate / min_vehicles)

            print(f"    Estimated capacity: {vehicle_capacity}")
            print(f"    Demand range: {config['demand_range']}")

            # Generate 5 random instances
            for i in range(1, 6):
                seed = hash(f"{range_name}_random_{size}_{i}") % (2**31)
                instance_name = f"{range_name}_random_{size:03d}_{i:02d}"

                request = GenerateRandomInstanceRequest(
                    name=instance_name,
                    description=f"Experimental instance - {range_name} range, random distribution, {size} customers",
                    num_customers=size,
                    vehicle_capacity=vehicle_capacity,
                    demand_min=config["demand_range"][0],
                    demand_max=config["demand_range"][1],
                    seed=seed,
                )

                instance = generate_random_instance(request, save=False)

                # Save instance
                save_instance(instance, range_dir, instance_name)
                total_instances += 1

                if i == 1:
                    print(f"    ✓ Generated {instance_name} (random, seed={seed})")

            # Generate 5 clustered instances
            for i in range(1, 6):
                seed = hash(f"{range_name}_clustered_{size}_{i}") % (2**31)
                instance_name = f"{range_name}_clustered_{size:03d}_{i:02d}"

                request = GenerateClusteredInstanceRequest(
                    name=instance_name,
                    description=f"Experimental instance - {range_name} range, clustered distribution, {size} customers",
                    num_customers=size,
                    vehicle_capacity=vehicle_capacity,
                    demand_min=config["demand_range"][0],
                    demand_max=config["demand_range"][1],
                    num_clusters=config["num_clusters"],
                    seed=seed,
                )

                instance = generate_clustered_instance(request, save=False)

                # Save instance
                save_instance(instance, range_dir, instance_name)
                total_instances += 1

                if i == 1:
                    print(f"    ✓ Generated {instance_name} (clustered, seed={seed})")

    # Generate summary
    summary = {
        "total_instances": total_instances,
        "ranges": {},
    }

    for range_name, config in experimental_design.items():
        summary["ranges"][range_name] = {
            "sizes": config["sizes"],
            "instances_per_size": 10,
            "total_instances": len(config["sizes"]) * 10,
        }

    summary_path = base_output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total instances generated: {total_instances}")
    print(f"Dataset summary saved to: {summary_path}")
    print(f"Instances stored in: {base_output_dir}")
    print()


def save_instance(instance: CVRPInstance, directory: Path, name: str) -> None:
    """
    Save CVRP instance to JSON file.

    Args:
        instance: CVRPInstance object
        directory: Target directory
        name: Instance name (without extension)
    """
    filepath = directory / f"{name}.json"

    with open(filepath, "w") as f:
        json.dump(instance.model_dump(), f, indent=2)


def load_instance(filepath: Path) -> CVRPInstance:
    """
    Load CVRP instance from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        CVRPInstance object
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return CVRPInstance(**data)


def get_all_instances(range_name: str = None) -> list[tuple[str, CVRPInstance]]:
    """
    Load all experimental instances, optionally filtered by range.

    Args:
        range_name: Optional filter for range (junior, mid, expert)

    Returns:
        List of tuples (instance_name, instance_object)
    """
    base_dir = settings.EXPERIMENT_INSTANCES_DIR
    instances = []

    ranges = [range_name] if range_name else ["junior", "mid", "expert"]

    for range_dir in ranges:
        range_path = base_dir / range_dir

        if not range_path.exists():
            continue

        for json_file in sorted(range_path.glob("*.json")):
            if json_file.name == "dataset_summary.json":
                continue

            instance_name = json_file.stem
            instance = load_instance(json_file)
            instances.append((instance_name, instance))

    return instances


if __name__ == "__main__":
    generate_experiment_instances()
