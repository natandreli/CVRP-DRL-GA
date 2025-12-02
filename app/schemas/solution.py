from typing import Optional

from pydantic import BaseModel, Field


class Route(BaseModel):
    """Route of a vehicle in the CVRP solution."""

    vehicle_id: int = Field(..., description="Vehicle ID")
    customer_sequence: list[int] = Field(
        ..., description="Sequence of customer IDs (excluding depot)"
    )
    total_demand: int = Field(0, ge=0, description="Total demand of the route")
    total_distance: float = Field(0, ge=0, description="Total distance of the route")

    @property
    def num_customers(self) -> int:
        """
        Number of customers in the route.

        Returns:
            int: Number of customers in the route
        """
        return len(self.customer_sequence)


class Solution(BaseModel):
    """Complete CVRP solution."""

    id: str = Field(..., description="Unique ID of the solution")
    instance_id: str = Field(..., description="ID of the CVRP instance")
    algorithm: str = Field(..., description="Algorithm used (ga, drl)")
    routes: list[Route] = Field(..., description="List of routes")
    total_cost: float = Field(..., ge=0, description="Total cost (distance)")
    computation_time: float = Field(..., ge=0, description="Computation time (s)")
    is_valid: bool = Field(..., description="Whether the solution is valid")
    convergence_history: Optional[list[float]] = Field(
        None, description="Convergence history"
    )

    @property
    def num_vehicles_used(self) -> int:
        """
        Number of vehicles used.

        Returns:
            int: Number of vehicles used
        """
        return len(self.routes)

    @property
    def num_customers_served(self) -> int:
        """
        Total number of customers served.

        Returns:
            int: Number of customers served
        """
        return sum(route.num_customers for route in self.routes)
