import random
from typing import Optional

from app.core.ga.individual import Individual
from app.schemas import CVRPInstance


def create_random_population(
    instance: CVRPInstance,
    population_size: int,
    seed: Optional[int] = None,
) -> list[Individual]:
    """
    Create random initial population.

    Each Individual is a random permutation of customers split into routes.

    Args:
        instance (CVRPInstance): CVRP instance
        population_size (int): Number of individuals
        seed (Optional[int]): Random seed

    Returns:
        list[Individual]: List of individuals
    """
    if seed is not None:
        random.seed(seed)

    population = []
    customer_ids = [c.id for c in instance.customers]

    for _ in range(population_size):
        # Random permutation
        shuffled = customer_ids.copy()
        random.shuffle(shuffled)

        # Convert to individual (will split into routes)
        individual = Individual.from_giant_tour(shuffled, instance)
        population.append(individual)

    return population


def create_nearest_neighbor_individual(
    instance: CVRPInstance,
    seed: Optional[int] = None,
) -> Individual:
    """
    Nearest neighbor heuristic: greedy construction.

    Start at depot, always add nearest unvisited customer to current route.
    Start new route when capacity exceeded.

    Args:
        instance (CVRPInstance): CVRP instance
        seed (Optional[int]): Random seed for tie-breaking

    Returns:
        Individual: Created Individual
    """
    if seed is not None:
        random.seed(seed)

    from app.core.utils.distance_calculator import calculate_euclidean_distance

    routes = []
    unvisited = set(c.id for c in instance.customers)
    customer_dict = {c.id: c for c in instance.customers}

    while unvisited:
        current_route = []
        current_capacity = 0
        current_location = instance.depot

        while unvisited:
            # Find nearest unvisited customer that fits
            best_customer = None
            best_distance = float("inf")

            for customer_id in unvisited:
                customer = customer_dict[customer_id]

                # Check capacity
                if current_capacity + customer.demand > instance.vehicle_capacity:
                    continue

                # Calculate distance
                distance = calculate_euclidean_distance(
                    current_location, customer.location
                )

                if distance < best_distance:
                    best_distance = distance
                    best_customer = customer

            if best_customer is None:
                # No customer fits, start new route
                break

            # Add customer to route
            current_route.append(best_customer.id)
            current_capacity += best_customer.demand
            current_location = best_customer.location
            unvisited.remove(best_customer.id)

        if current_route:
            routes.append(current_route)

    return Individual(routes=routes, instance=instance)


def create_savings_individual(instance: CVRPInstance) -> Individual:
    """
    Clarke-Wright Savings Algorithm for initial solution.

    1. Start with each customer in separate route
    2. Calculate savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
    3. Merge routes with highest savings while feasible

    Args:
        instance (CVRPInstance): CVRP instance

    Returns:
        Individual: Created Individual
    """
    from app.core.utils.distance_calculator import calculate_distance_matrix

    distance_matrix = calculate_distance_matrix(instance)
    customer_dict = {c.id: c for c in instance.customers}

    # Start with each customer in separate route
    routes = [[c.id] for c in instance.customers]

    # Calculate all savings
    savings = []
    customer_ids = [c.id for c in instance.customers]

    for i in range(len(customer_ids)):
        for j in range(i + 1, len(customer_ids)):
            id_i = customer_ids[i]
            id_j = customer_ids[j]

            # Savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
            saving = (
                distance_matrix[0][id_i]
                + distance_matrix[0][id_j]
                - distance_matrix[id_i][id_j]
            )
            savings.append((saving, id_i, id_j))

    # Sort by savings (descending)
    savings.sort(reverse=True, key=lambda x: x[0])

    # Merge routes based on savings
    for saving_value, id_i, id_j in savings:
        # Find routes containing i and j
        route_i = None
        route_j = None

        for route in routes:
            if id_i in route:
                route_i = route
            if id_j in route:
                route_j = route

        if route_i is None or route_j is None or route_i == route_j:
            continue

        # Check if i and j are at route ends
        i_at_end = route_i[0] == id_i or route_i[-1] == id_i
        j_at_end = route_j[0] == id_j or route_j[-1] == id_j

        if not (i_at_end and j_at_end):
            continue

        # Check capacity
        demand_i = sum(customer_dict[cid].demand for cid in route_i)
        demand_j = sum(customer_dict[cid].demand for cid in route_j)

        if demand_i + demand_j > instance.vehicle_capacity:
            continue

        # Merge routes
        if route_i[-1] == id_i and route_j[0] == id_j:
            merged = route_i + route_j
        elif route_j[-1] == id_j and route_i[0] == id_i:
            merged = route_j + route_i
        elif route_i[0] == id_i and route_j[0] == id_j:
            merged = route_i[::-1] + route_j
        elif route_i[-1] == id_i and route_j[-1] == id_j:
            merged = route_i + route_j[::-1]
        else:
            continue

        # Replace routes
        routes.remove(route_i)
        routes.remove(route_j)
        routes.append(merged)

    return Individual(routes=routes, instance=instance)


def generate_initial_population(
    instance: CVRPInstance,
    population_size: int,
    seed: Optional[int] = None,
    use_heuristics: Optional[bool] = False,
) -> list[Individual]:
    """
    Create diverse initial population using multiple strategies.

    - 20% nearest neighbor variants
    - 10% savings algorithm
    - 70% random

    Args:
        instance (CVRPInstance): CVRP instance
        population_size (int): Population size
        seed (Optional[int]): Random seed
        use_heuristics (Optional[bool]): If True, use heuristics (NN + savings + random).
                                         If False, use pure random initialization.

    Returns:
        list[Individual]: List of individuals
    """
    if seed is not None:
        random.seed(seed)

    if not use_heuristics:
        return create_random_population(instance, population_size, seed)

    population = []

    # Nearest neighbor variants (20%)
    nn_count = max(1, int(population_size * 0.2))
    for i in range(nn_count):
        nn_individual = create_nearest_neighbor_individual(
            instance, seed=seed + i if seed else None
        )
        population.append(nn_individual)

    # Savings algorithm (10%)
    savings_count = max(1, int(population_size * 0.1))
    for _ in range(savings_count):
        savings_individual = create_savings_individual(instance)
        population.append(savings_individual)

    # Random fill remaining
    remaining = population_size - len(population)
    if remaining > 0:
        random_pop = create_random_population(instance, remaining, seed)
        population.extend(random_pop)

    return population[:population_size]
