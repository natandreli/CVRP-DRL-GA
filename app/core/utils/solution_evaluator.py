import numpy as np

from app.config.logging import logger
from app.core.utils import (
    calculate_route_distance,
    calculate_solution_cost,
    get_customers_by_ids,
)
from app.schemas import CVRPInstance, Route, Solution


def validate_solution(solution: Solution, instance: CVRPInstance) -> tuple[bool, str]:
    """
    Validate if a solution is valid for a CVRP instance.

    Validation checks:
    1. All customers are visited exactly once.
    2. No route exceeds vehicle capacity.
    3. Routes start and end at the depot.

    Args:
        solution (Solution): Solution to validate
        instance (CVRPInstance): CVRP instance

    Returns:
        tuple[bool, str]: Tuple (is_valid, error_message)
    """
    # 1. Check all customers are visited exactly once
    all_customer_ids = set()
    for route in solution.routes:
        all_customer_ids.update(route.customer_sequence)

    expected_customer_ids = {c.id for c in instance.customers}

    if all_customer_ids != expected_customer_ids:
        missing = expected_customer_ids - all_customer_ids
        duplicated = [
            cid
            for cid in all_customer_ids
            if sum(cid in route.customer_sequence for route in solution.routes) > 1
        ]

        if missing:
            return False, f"Missing customers: {missing}"
        if duplicated:
            return False, f"Customers visited multiple times: {duplicated}"
        return False, "Customer set does not match"

    # 2. Check no route exceeds vehicle capacity
    for i, route in enumerate(solution.routes):
        customers = get_customers_by_ids(instance, route.customer_sequence)
        total_demand = sum(c.demand for c in customers)

        if total_demand > instance.vehicle_capacity:
            return (
                False,
                f"Route {i} exceeds capacity: {total_demand} > {instance.vehicle_capacity}",
            )

    return True, "Valid solution"


def update_route_metrics(
    route: Route, instance: CVRPInstance, distance_matrix: np.ndarray
) -> Route:
    """
    Calculate metrics for a route (demand, distance).

    Args:
        route (Route): Route to evaluate
        instance (CVRPInstance): CVRP instance
        distance_matrix (np.ndarray): Distance matrix

    Returns:
        Route: Route updated with calculated metrics
    """
    # Calculate demand
    customers = get_customers_by_ids(instance, route.customer_sequence)
    total_demand = sum(c.demand for c in customers)

    # Calculate distance
    route_sequence = [0] + route.customer_sequence + [0]
    total_distance = calculate_route_distance(route_sequence, distance_matrix)

    route.total_demand = total_demand
    route.total_distance = total_distance

    return route


def evaluate_solution(
    solution: Solution, instance: CVRPInstance, distance_matrix: np.ndarray
) -> Solution:
    """
    Evaluate a solution: calculate metrics and validate.

    Args:
        solution (Solution): Solution to evaluate
        instance (CVRPInstance): CVRP instance
        distance_matrix (np.ndarray): Distance matrix

    Returns:
        Solution: Solution updated with metrics and validation results
    """
    # Calculate metrics for each route
    for route in solution.routes:
        update_route_metrics(route, instance, distance_matrix)

    # Calculate total cost
    routes_with_depot = [[0] + r.customer_sequence + [0] for r in solution.routes]
    solution.total_cost = calculate_solution_cost(routes_with_depot, distance_matrix)

    # Validate
    is_valid, message = validate_solution(solution, instance)
    solution.is_valid = is_valid

    if not is_valid:
        logger.info(f"Invalid solution: {message}")

    return solution
