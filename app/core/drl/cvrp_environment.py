from typing import Tuple

import numpy as np

from app.core.utils.distance_calculator import calculate_distance_matrix
from app.schemas import CVRPInstance


class CVRPEnvironment:
    """
    Gym-like environment for CVRP.

    Agent constructs a solution by sequentially choosing customers to visit.
    When vehicle capacity is reached, automatically returns to depot and starts new route.
    """

    def __init__(self, instance: CVRPInstance) -> None:
        """
        Initialize environment.

        Args:
            instance (CVRPInstance): CVRP instance to solve
        """
        self.instance = instance
        self.num_customers = instance.num_customers

        # State tracking
        self.current_location_id: int = 0  # Start at depot
        self.current_capacity: int = 0
        self.visited: set[int] = set()  # Customer IDs visited
        self.current_route: list[int] = []  # Current route being built
        self.routes: list[list[int]] = []  # Completed routes
        self.total_distance: float = 0.0

        # Distance matrix (precomputed)
        self.distance_matrix = calculate_distance_matrix(instance)

        self.done = False

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            np.ndarray: Initial state representation
        """
        self.current_location_id = 0
        self.current_capacity = 0
        self.visited = set()
        self.current_route = []
        self.routes = []
        self.total_distance = 0.0
        self.done = False

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.

        State vector includes:
        - Current location (normalized coordinates)
        - Current capacity used (normalized)
        - For each customer:
          - Visited flag (0/1)
          - Demand (normalized)
          - Distance to current location (normalized)
          - X, Y coordinates (normalized)

        Returns:
            np.ndarray: State vector
        """
        state_components = []

        # Current location coordinates (normalized)
        if self.current_location_id == 0:
            current_loc = self.instance.depot
        else:
            current_loc = self.instance.customers[self.current_location_id - 1].location

        state_components.extend(
            [
                current_loc.x / 100.0,  # Normalize assuming grid 0-100
                current_loc.y / 100.0,
            ]
        )

        # Current capacity (normalized)
        state_components.append(self.current_capacity / self.instance.vehicle_capacity)

        # For each customer
        for customer in self.instance.customers:
            customer_id = customer.id

            # Visited flag
            state_components.append(1.0 if customer_id in self.visited else 0.0)

            # Demand (normalized)
            state_components.append(customer.demand / self.instance.vehicle_capacity)

            # Distance to current location (normalized, max assumed 150)
            distance = self.distance_matrix[self.current_location_id][customer_id]
            state_components.append(distance / 150.0)

            # Coordinates (normalized)
            state_components.extend(
                [
                    customer.location.x / 100.0,
                    customer.location.y / 100.0,
                ]
            )

        return np.array(state_components, dtype=np.float32)

    def get_valid_actions(self) -> list[int]:
        """
        Get list of valid actions (customer IDs that can be visited).

        A customer is valid if:
        - Not yet visited
        - Demand fits in current vehicle capacity

        Returns:
            list[int]: List of valid customer IDs
        """
        valid_actions = []

        for customer in self.instance.customers:
            customer_id = customer.id

            # Skip if already visited
            if customer_id in self.visited:
                continue

            # Check capacity constraint
            if (
                self.current_capacity + customer.demand
                <= self.instance.vehicle_capacity
            ):
                valid_actions.append(customer_id)

        return valid_actions

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action (visit customer).

        Args:
            action (int): Customer ID to visit

        Returns:
            Tuple[np.ndarray, float, bool, dict]: (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode already finished. Call reset().")

        # Validate action
        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            # Invalid action: large penalty
            return self._get_state(), -100.0, True, {"error": "Invalid action"}

        # Get customer
        customer = self.instance.customers[action - 1]  # Customer IDs start at 1

        # Calculate distance from current location to customer
        distance = self.distance_matrix[self.current_location_id][action]

        # Update state
        self.visited.add(action)
        self.current_route.append(action)
        self.current_capacity += customer.demand
        self.total_distance += distance
        self.current_location_id = action

        # Calculate reward (negative distance)
        reward = -distance

        # Check if all customers visited
        if len(self.visited) == self.num_customers:
            # Return to depot
            distance_to_depot = self.distance_matrix[self.current_location_id][0]
            self.total_distance += distance_to_depot
            reward -= distance_to_depot

            # Save current route
            if self.current_route:
                self.routes.append(self.current_route)

            self.done = True

            return (
                self._get_state(),
                reward,
                True,
                {
                    "routes": self.routes,
                    "total_distance": self.total_distance,
                },
            )

        # Check if need to start new route (no valid actions)
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            # Return to depot and start new route
            distance_to_depot = self.distance_matrix[self.current_location_id][0]
            self.total_distance += distance_to_depot
            reward -= distance_to_depot

            # Save current route
            self.routes.append(self.current_route.copy())

            # Reset for new route
            self.current_route = []
            self.current_capacity = 0
            self.current_location_id = 0

        return self._get_state(), reward, False, {}

    def get_state_size(self) -> int:
        """
        Get size of state vector.

        Returns:
            int: Size of state vector
        """
        # 2 (current loc) + 1 (capacity) + num_customers * 5 (visited, demand, distance, x, y)
        return 3 + self.num_customers * 5

    def get_action_size(self) -> int:
        """
        Get size of action space (number of customers).

        Returns:
            int: Number of customers
        """
        return self.num_customers
