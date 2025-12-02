from typing import Optional

from app.schemas import Customer, CVRPInstance, Route, Solution


def get_customer_by_id(instance: CVRPInstance, customer_id: int) -> Optional[Customer]:
    """
    Get a customer by ID from the CVRP instance.

    Args:
        instance (CVRPInstance): CVRP instance
        customer_id (int): ID of the customer to find

    Returns:
        Optional[Customer]: Customer if found, None otherwise
    """
    for customer in instance.customers:
        if customer.id == customer_id:
            return customer


def get_customers_by_ids(
    instance: CVRPInstance, customer_ids: list[int]
) -> list[Customer]:
    """
    Get customer objects by their IDs.

    Args:
        instance: CVRP instance
        customer_ids: List of customer IDs

    Returns:
        List of Customer objects

    Raises:
        ValueError: If customer ID not found
    """
    customers = []
    for customer_id in customer_ids:
        customer = get_customer_by_id(instance, customer_id)
        if customer is None:
            raise ValueError(f"Customer {customer_id} not found")
        customers.append(customer)
    return customers


def calculate_route_distance(
    sequence: list[int], distance_matrix: list[list[float]]
) -> float:
    """
    Calculate total distance for a route sequence.

    Args:
        sequence: Sequence of locations (including depot)
        distance_matrix: Distance matrix

    Returns:
        Total distance
    """
    total_distance = 0.0
    for i in range(len(sequence) - 1):
        total_distance += distance_matrix[sequence[i]][sequence[i + 1]]
    return total_distance


def convert_route_to_sequence_with_depot(route: Route) -> list[int]:
    """
    Convert route to sequence with depot at start and end.

    Args:
        route (Route): Vehicle route

    Returns:
        List[int]: List with [0, customer1, customer2, ..., 0]
    """
    return [0] + route.customer_sequence + [0]


def convert_solution_to_route_sequences(solution: Solution) -> list[list[int]]:
    """
    Convert solution to list of sequences (each with depot).

    Args:
        solution (Solution): Complete solution

    Returns:
        List[List[int]]: List of lists, each representing a route with depot
    """
    return [convert_route_to_sequence_with_depot(route) for route in solution.routes]


def get_all_customer_ids_from_solution(solution: Solution) -> list[int]:
    """
    Get a flat list of all customer IDs in the solution.

    Args:
        solution (Solution): Complete solution

    Returns:
        List[int]: List of customer IDs
    """
    customer_ids = []
    for route in solution.routes:
        customer_ids.extend(route.customer_sequence)
    return customer_ids
