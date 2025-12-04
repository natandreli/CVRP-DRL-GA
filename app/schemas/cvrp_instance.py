from typing import Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Coordinates of a point in the 2D plane."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class Customer(BaseModel):
    """Customer in the CVRP problem."""

    id: int = Field(..., description="Customer ID")
    location: Location = Field(..., description="Customer location")
    demand: int = Field(..., ge=0, description="Customer demand")
    ready_time: Optional[int] = Field(None, description="Start of time window")
    due_time: Optional[int] = Field(None, description="End of time window")
    service_time: Optional[int] = Field(0, description="Service time")


class CVRPInstance(BaseModel):
    """Capacitated Vehicle Routing Problem Instance."""

    id: str = Field(..., description="Unique identifier of the instance")
    name: str = Field(..., description="Name of the instance")
    description: Optional[str] = Field(None, description="Description of the instance")
    depot: Location = Field(..., description="Location of the depot")
    customers: list[Customer] = Field(..., description="List of customers")
    vehicle_capacity: int = Field(..., gt=0, description="Vehicle capacity")
    max_vehicles: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of vehicles available (None = unlimited fleet)",
    )
    distance_matrix: Optional[list[list[float]]] = Field(
        None, description="Precomputed distance matrix"
    )

    @property
    def num_customers(self) -> int:
        """
        Number of customers (excluding the depot).

        Returns:
            int: Number of customers
        """
        return len(self.customers)

    @property
    def total_demand(self) -> int:
        """
        Total demand of all customers.

        Returns:
            int: Total demand
        """
        return sum(customer.demand for customer in self.customers)

    @property
    def min_vehicles_needed(self) -> int:
        """
        Minimum theoretical vehicles needed based on total demand.

        Returns:
            int: Minimum number of vehicles needed
        """
        if self.vehicle_capacity == 0:
            return len(self.customers)  # One vehicle per customer
        return max(
            1, (self.total_demand + self.vehicle_capacity - 1) // self.vehicle_capacity
        )

    @staticmethod
    def from_vrplib(instance_data: dict, instance_id: str) -> "CVRPInstance":
        """
        Create CVRP instance from vrplib data.

        Args:
            instance_data: Dict with instance data from vrplib
            instance_id: Unique ID for the instance

        Returns:
            Configured CVRPInstance
        """
        # Get coordinates (depot is the first node, index 0)
        node_coords = instance_data.get("node_coord", [])
        demands = instance_data.get("demand", [])
        capacity = instance_data.get("capacity", 100)

        # The depot is the first node
        depot_coords = node_coords[0] if node_coords else [0, 0]
        depot = Location(x=depot_coords[0], y=depot_coords[1])

        # Create customers (from node 1 onwards)
        customers = []
        for i in range(1, len(node_coords)):
            coord = node_coords[i]
            demand = demands[i] if i < len(demands) else 0

            customer = Customer(
                id=i,
                location=Location(x=coord[0], y=coord[1]),
                demand=demand,
            )
            customers.append(customer)

        # Max vehicles from vrplib (if specified), otherwise None (unlimited)
        max_vehicles = instance_data.get("vehicles")

        return CVRPInstance(
            id=instance_id,
            name=instance_id,
            depot=depot,
            customers=customers,
            vehicle_capacity=capacity,
            max_vehicles=max_vehicles,
        )
