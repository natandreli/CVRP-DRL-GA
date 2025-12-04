import json
import random
from datetime import datetime

import numpy as np
from fastapi import UploadFile

from app.api.routers.instances.payload_schemas import (
    GenerateClusteredInstanceRequest,
    GenerateRandomInstanceRequest,
)
from app.config import settings
from app.core.utils.vrplib_loader import load_vrplib_instance
from app.exceptions import InstanceParseException, UnsupportedFileFormatException
from app.schemas.cvrp_instance import Customer, CVRPInstance, Location


def save_instance(instance: CVRPInstance) -> str:
    """
    Save a CVRP instance to disk as JSON.

    Args:
        instance: CVRP instance to save

    Returns:
        str: Path where the instance was saved
    """
    settings.INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{instance.id}.json"
    filepath = settings.INSTANCES_DIR / filename

    with open(filepath, "w") as f:
        json.dump(instance.model_dump(), f, indent=2)

    return str(filepath)


def load_instance_by_id(instance_id: str) -> CVRPInstance:
    """
    Load a CVRP instance by its ID/filename.

    Args:
        instance_id: Instance identifier (filename without extension)

    Returns:
        CVRPInstance: Loaded instance

    Raises:
        InstanceParseException: If instance cannot be found or loaded
    """
    instances_dir = settings.INSTANCES_DIR

    json_path = instances_dir / f"{instance_id}.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                return CVRPInstance(**data)
        except Exception as e:
            raise InstanceParseException(instance_id, str(e))

    vrp_path = instances_dir / f"{instance_id}.vrp"
    if vrp_path.exists():
        try:
            return load_vrplib_instance(str(vrp_path))
        except Exception as e:
            raise InstanceParseException(instance_id, str(e))

    raise InstanceParseException(instance_id, f"Instance not found in {instances_dir}")


def get_instances() -> list[CVRPInstance]:
    """
    List all available CVRP instance presets.

    Returns:
        list[CVRPInstance]: List of preset CVRP instances
    """
    presets = []
    instances_dir = settings.INSTANCES_DIR

    if not instances_dir.exists():
        return presets

    for file_path in instances_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                instance = CVRPInstance(**data)
                presets.append(instance)
        except Exception:
            continue

    for file_path in instances_dir.glob("*.vrp"):
        try:
            instance = load_vrplib_instance(str(file_path))
            presets.append(instance)
        except Exception:
            continue

    return presets


def generate_random_instance(params: GenerateRandomInstanceRequest) -> CVRPInstance:
    """
    Generate a random CVRP instance based on the provided parameters.

    Args:
        params: Parameters for generating the random instance

    Returns:
        CVRPInstance: Generated CVRP instance
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    # Depot in the center
    depot = Location(x=params.grid_size / 2, y=params.grid_size / 2)

    # Generate random customers
    customers = []
    for i in range(1, params.num_customers + 1):
        location = Location(
            x=random.uniform(0, params.grid_size),
            y=random.uniform(0, params.grid_size),
        )

        demand = random.randint(params.min_customer_demand, params.max_customer_demand)

        customer = Customer(
            id=i,
            location=location,
            demand=demand,
        )
        customers.append(customer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"s{params.seed}" if params.seed else f"ts{timestamp}"
    instance_id = f"synthetic_{params.num_customers}c_{params.vehicle_capacity}q_random_{seed_str}"

    instance = CVRPInstance(
        id=instance_id,
        name=f"Synthetic Random: {params.num_customers} customers, Q={params.vehicle_capacity}",
        description=f"Synthetic random CVRP instance with {params.num_customers} customers and a "
        f"vehicle capacity of {params.vehicle_capacity}. Generated using seed {params.seed}, "
        f"with customer demands ranging from {params.min_customer_demand} to {params.max_customer_demand}.",
        depot=depot,
        customers=customers,
        vehicle_capacity=params.vehicle_capacity,
        max_vehicles=None,  # Unlimited fleet
    )

    save_instance(instance)

    return instance


def generate_clustered_instance(
    params: GenerateClusteredInstanceRequest,
) -> CVRPInstance:
    """
    Generate a clustered CVRP instance based on the provided parameters.

    Args:
        params: Parameters for generating the clustered instance

    Returns:
        CVRPInstance: Generated CVRP instance
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    # Depot in the center
    depot = Location(x=params.grid_size / 2, y=params.grid_size / 2)

    # Generate cluster centers
    cluster_centers = []
    for _ in range(params.num_clusters):
        center = Location(
            x=random.uniform(params.grid_size * 0.2, params.grid_size * 0.8),
            y=random.uniform(params.grid_size * 0.2, params.grid_size * 0.8),
        )
        cluster_centers.append(center)

    # Distribute customers in clusters
    customers = []
    customers_per_cluster = params.num_customers // params.num_clusters
    customer_id = 1

    for cluster_center in cluster_centers:
        for _ in range(customers_per_cluster):
            # Customer near the cluster center
            offset_x = random.gauss(0, params.grid_size * 0.1)
            offset_y = random.gauss(0, params.grid_size * 0.1)

            location = Location(
                x=max(0, min(params.grid_size, cluster_center.x + offset_x)),
                y=max(0, min(params.grid_size, cluster_center.y + offset_y)),
            )

            demand = random.randint(
                params.min_customer_demand, params.max_customer_demand
            )

            customer = Customer(
                id=customer_id,
                location=location,
                demand=demand,
            )
            customers.append(customer)
            customer_id += 1

    # Add remaining customers if num_customers is not divisible
    remaining = params.num_customers - len(customers)
    for _ in range(remaining):
        location = Location(
            x=random.uniform(0, params.grid_size),
            y=random.uniform(0, params.grid_size),
        )

        demand = random.randint(params.min_customer_demand, params.max_customer_demand)
        customer = Customer(
            id=customer_id,
            location=location,
            demand=demand,
        )
        customers.append(customer)
        customer_id += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"s{params.seed}" if params.seed else f"ts{timestamp}"
    instance_id = f"synthetic_{params.num_customers}c_{params.vehicle_capacity}q_clustered{params.num_clusters}_{seed_str}"

    instance = CVRPInstance(
        id=instance_id,
        name=f"Synthetic Clustered: {params.num_customers} customers, {params.num_clusters} clusters, Q={params.vehicle_capacity}",
        description=f"Synthetic clustered CVRP instance with {params.num_customers} customers "
        f"distributed across {params.num_clusters} clusters and a vehicle capacity of "
        f"{params.vehicle_capacity}. Generated using seed {params.seed}, with customer demands "
        f"ranging from {params.min_customer_demand} to {params.max_customer_demand}.",
        depot=depot,
        customers=customers,
        vehicle_capacity=params.vehicle_capacity,
        max_vehicles=None,  # Unlimited fleet
    )

    save_instance(instance)

    return instance


async def upload_instance(file: UploadFile) -> CVRPInstance:
    """
    Upload a CVRP instance from a given file path.

    Args:
        file: Uploaded file containing CVRP instance

    Returns:
        CVRPInstance: Loaded CVRP instance

    Raises:
        UnsupportedFileFormatException: If file format is not supported
        InstanceParseException: If file cannot be parsed
    """
    filename = file.filename or "unknown"

    if not (filename.endswith(".json") or filename.endswith(".vrp")):
        raise UnsupportedFileFormatException(filename)

    try:
        content = await file.read()

        if filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            instance = CVRPInstance(**data)
            return instance

        else:
            temp_path = settings.INSTANCES_DIR / filename
            with open(temp_path, "wb") as f:
                f.write(content)

            instance = load_vrplib_instance(str(temp_path))
            return instance

    except (UnsupportedFileFormatException, InstanceParseException):
        raise
    except Exception as e:
        raise InstanceParseException(filename, str(e))
