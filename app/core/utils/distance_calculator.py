import math

import numpy as np

from app.schemas import CVRPInstance, Location


def calculate_euclidean_distance(loc1: Location, loc2: Location) -> float:
    """
    Calculate Euclidean distance between two locations.

    Args:
        loc1(Location): First location
        loc2(Location): Second location

    Returns:
        float: Euclidean distance between loc1 and loc2
    """
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


def calculate_distance_matrix(instance: CVRPInstance) -> np.ndarray:
    """
    Calculate distance matrix for a CVRP instance.

    Index 0 represents the depot.

    Args:
        instance (CVRPInstance): CVRP instance

    Returns:
        numpy.ndarray: Distance matrix (n+1 x n+1) where n is the number of customers
    """
    num_customers = instance.num_customers + 1  # +1 for the depot
    distance_matrix = np.zeros((num_customers, num_customers))

    # List of all locations (depot + customers)
    all_locations = [instance.depot] + [c.location for c in instance.customers]

    # Calculate distances
    for i in range(num_customers):
        for j in range(i + 1, num_customers):
            dist = calculate_euclidean_distance(all_locations[i], all_locations[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist  # Symmetric

    return distance_matrix


def calculate_route_distance(
    route_sequence: list[int], distance_matrix: np.ndarray
) -> float:
    """
    Calculate total distance of a route given its sequence and distance matrix.

    Args:
        route_sequence (list[int]): Sequence of nodes (should include depot: [0, 1, 2, 3, 0])
        distance_matrix (numpy.ndarray): Distance matrix

    Returns:
        float: Total distance of the route
    """
    total_distance = 0.0
    for i in range(len(route_sequence) - 1):
        from_node = route_sequence[i]
        to_node = route_sequence[i + 1]
        total_distance += distance_matrix[from_node][to_node]
    return total_distance


def calculate_solution_cost(
    routes: list[list[int]], distance_matrix: np.ndarray
) -> float:
    """
    Calculate total cost of a solution (sum of distances of all routes).

    Args:
        routes (list[list[int]]): List of routes (each route includes depot)
        distance_matrix (numpy.ndarray): Distance matrix

    Returns:
        float: Total cost (total distance) of the solution
    """
    total_cost = 0.0
    for route in routes:
        total_cost += calculate_route_distance(route, distance_matrix)
    return total_cost
