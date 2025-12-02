import random
from typing import Optional

from app.core.utils import (
    calculate_distance_matrix,
    calculate_route_distance,
    calculate_solution_cost,
    get_customers_by_ids,
)
from app.schemas import CVRPInstance, Route, Solution


class Individual:
    """
    Individual representation for CVRP.

    Represents a solution as a list of routes (sequences of customer IDs).
    Each route starts and ends at depot (implicit, not stored).
    """

    def __init__(
        self,
        routes: list[list[int]],
        instance: CVRPInstance,
        fitness: Optional[float] = None,
    ) -> None:
        """
        Initialize individual.

        Args:
            routes (list[list[int]]): List of routes, each route is a list of customer IDs
            instance (CVRPInstance): CVRP instance
            fitness (Optional[float]): Fitness value (distance), if already computed
        """
        self.routes = routes
        self.instance = instance
        self._fitness = fitness

    @property
    def fitness(self) -> float:
        """
        Get fitness (total distance). Lower is better.

        Returns:
            float: Total distance of the solution
        """
        if self._fitness is None:
            distance_matrix = calculate_distance_matrix(self.instance)
            self._fitness = calculate_solution_cost(self.routes, distance_matrix)

        return self._fitness

    def to_solution(self, algorithm: str = "ga", generation: int = 0) -> Solution:
        """
        Convert individual to Solution schema.

        Args:
            algorithm (str): Algorithm identifier
            generation (int): Generation number

        Returns:
            Solution: CVRP solution
        """
        distance_matrix = calculate_distance_matrix(self.instance)

        solution_routes = []
        for vehicle_id, customer_ids in enumerate(self.routes):
            if not customer_ids:  # Skip empty routes
                continue

            # Calculate route metrics
            customers = get_customers_by_ids(self.instance, customer_ids)
            total_demand = sum(c.demand for c in customers)

            # Add depot at start and end for distance calculation
            sequence_with_depot = [0] + customer_ids + [0]
            total_distance = calculate_route_distance(
                sequence_with_depot, distance_matrix
            )

            route = Route(
                vehicle_id=vehicle_id,
                customer_sequence=customer_ids,
                total_demand=total_demand,
                total_distance=total_distance,
            )
            solution_routes.append(route)

        return Solution(
            id=f"ga_gen{generation}_{random.randint(1000, 9999)}",
            instance_id=self.instance.id,
            algorithm=algorithm,
            routes=solution_routes,
            total_cost=self.fitness,
            computation_time=0,
            is_valid=True,  # Will be validated later if needed
        )

    @staticmethod
    def from_giant_tour(
        customer_sequence: list[int],
        instance: CVRPInstance,
    ) -> "Individual":
        """
        Create Individual from giant tour by splitting into feasible routes.

        Args:
            customer_sequence (list[int]): Flat list of customer IDs
            instance (CVRPInstance): CVRP instance

        Returns:
            Individual: Created individual
        """
        routes = []
        current_route = []
        current_capacity = 0

        for customer_id in customer_sequence:
            customers = get_customers_by_ids(instance, [customer_id])
            if not customers:
                continue

            customer = customers[0]
            demand = customer.demand

            # Check if adding this customer exceeds capacity
            if current_capacity + demand > instance.vehicle_capacity:
                # Start new route
                if current_route:
                    routes.append(current_route)
                current_route = [customer_id]
                current_capacity = demand
            else:
                # Add to current route
                current_route.append(customer_id)
                current_capacity += demand

        # Add last route
        if current_route:
            routes.append(current_route)

        return Individual(routes=routes, instance=instance)

    def get_all_customers(self) -> list[int]:
        """
        Get flat list of all customers in individual.

        Returns:
            list[int]: List of customer IDs
        """
        all_customers = []
        for route in self.routes:
            all_customers.extend(route)
        return all_customers

    def copy(self) -> "Individual":
        """
        Create deep copy of individual.

        Returns:
            Individual: Copied individual
        """
        routes_copy = [route.copy() for route in self.routes]
        return Individual(
            routes=routes_copy,
            instance=self.instance,
            fitness=self._fitness,
        )

    def __repr__(self) -> str:
        """
        String representation.

        Returns:
            str: String representation
        """
        return f"Individual(routes={len(self.routes)}, fitness={self.fitness:.2f})"
