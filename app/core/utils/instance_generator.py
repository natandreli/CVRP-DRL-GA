import random
from datetime import datetime
from typing import Optional

import numpy as np

from app.schemas import Customer, CVRPInstance, Location


def generate_random_instance(
    num_customers: int = 20,
    grid_size: int = 100,
    vehicle_capacity: int = 100,
    min_demand: int = 5,
    max_demand: int = 30,
    seed: Optional[int] = None,
) -> CVRPInstance:
    """
    Generate synthetic CVRP instance with random customer locations and demands.

    Args:
        num_customers (int): Number of customers
        grid_size (int): Size of the grid (0,0) to (grid_size, grid_size)
        vehicle_capacity (int): Vehicle capacity
        min_demand (int): Minimum customer demand
        max_demand (int): Maximum customer demand
        seed (Optional[int]): Seed for reproducibility

    Returns:
        CVRPInstance: Generated CVRP instance
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Depot in the center
    depot = Location(x=grid_size / 2, y=grid_size / 2)

    # Generate random customers
    customers = []
    for i in range(1, num_customers + 1):
        location = Location(
            x=random.uniform(0, grid_size),
            y=random.uniform(0, grid_size),
        )
        demand = random.randint(min_demand, max_demand)

        customer = Customer(
            id=i,
            location=location,
            demand=demand,
        )
        customers.append(customer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"s{seed}" if seed else f"ts{timestamp}"
    instance_id = f"synthetic_{num_customers}c_{vehicle_capacity}q_random_{seed_str}"

    return CVRPInstance(
        id=instance_id,
        name=f"Synthetic Random: {num_customers} customers, Q={vehicle_capacity}",
        depot=depot,
        customers=customers,
        vehicle_capacity=vehicle_capacity,
        max_vehicles=None,  # Unlimited fleet
    )


def generate_clustered_instance(
    num_customers: int = 20,
    num_clusters: int = 3,
    grid_size: int = 100,
    vehicle_capacity: int = 100,
    seed: Optional[int] = None,
) -> CVRPInstance:
    """
    Generate CVRP instance with customers grouped in clusters.

    Args:
        num_customers (int): Number of customers
        num_clusters (int): Number of clusters
        grid_size (int): Size of the grid (0,0) to (grid_size, grid_size)
        vehicle_capacity (int): Vehicle capacity
        seed (Optional[int]): Seed for reproducibility

    Returns:
        CVRPInstance: Generated clustered CVRP instance
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Depot in the center
    depot = Location(x=grid_size / 2, y=grid_size / 2)

    # Generate cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        center = Location(
            x=random.uniform(grid_size * 0.2, grid_size * 0.8),
            y=random.uniform(grid_size * 0.2, grid_size * 0.8),
        )
        cluster_centers.append(center)

    # Distribute customers in clusters
    customers = []
    customers_per_cluster = num_customers // num_clusters
    customer_id = 1

    for cluster_center in cluster_centers:
        for _ in range(customers_per_cluster):
            # Customer near the cluster center
            offset_x = random.gauss(0, grid_size * 0.1)
            offset_y = random.gauss(0, grid_size * 0.1)

            location = Location(
                x=max(0, min(grid_size, cluster_center.x + offset_x)),
                y=max(0, min(grid_size, cluster_center.y + offset_y)),
            )

            demand = random.randint(5, 30)

            customer = Customer(
                id=customer_id,
                location=location,
                demand=demand,
            )
            customers.append(customer)
            customer_id += 1

    # Add remaining customers if num_customers is not divisible
    remaining = num_customers - len(customers)
    for _ in range(remaining):
        location = Location(
            x=random.uniform(0, grid_size),
            y=random.uniform(0, grid_size),
        )
        demand = random.randint(5, 30)

        customer = Customer(
            id=customer_id,
            location=location,
            demand=demand,
        )
        customers.append(customer)
        customer_id += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"s{seed}" if seed else f"ts{timestamp}"
    instance_id = f"synthetic_{num_customers}c_{vehicle_capacity}q_clustered{num_clusters}_{seed_str}"

    return CVRPInstance(
        id=instance_id,
        name=f"Synthetic Clustered: {num_customers} customers, {num_clusters} clusters, Q={vehicle_capacity}",
        depot=depot,
        customers=customers,
        vehicle_capacity=vehicle_capacity,
        max_vehicles=None,  # Unlimited fleet
    )
