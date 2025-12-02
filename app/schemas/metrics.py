from typing import Optional

from pydantic import BaseModel, Field


class SolutionMetrics(BaseModel):
    """Metrics of a CVRP solution."""

    solution_id: str = Field(..., description="Solution ID")
    total_distance: float = Field(..., ge=0, description="Total distance")
    num_routes: int = Field(..., ge=0, description="Number of routes")
    vehicles_used: int = Field(..., ge=0, description="Vehicles used")
    average_route_length: float = Field(..., ge=0, description="Average route length")
    max_route_length: float = Field(..., ge=0, description="Maximum route length")
    min_route_length: float = Field(..., ge=0, description="Minimum route length")
    capacity_utilization: float = Field(
        ..., ge=0, le=1, description="Capacity utilization (0-1 scale)"
    )
    route_balance: float = Field(
        ...,
        ge=0,
        le=1,
        description="Route balance (0=unbalanced, 1=perfectly balanced)",
    )
    gap_from_optimal: Optional[float] = Field(
        None, description="Gap vs known optimal (%)"
    )


class ComparisonMetrics(BaseModel):
    """Comparison metrics between algorithms."""

    instance_id: str = Field(..., description="Instance ID")

    # GA metrics
    ga_best_cost: float = Field(..., ge=0, description="Best GA cost")
    ga_average_cost: float = Field(..., ge=0, description="Average GA cost")
    ga_std_dev: float = Field(..., ge=0, description="GA standard deviation")
    ga_average_time: float = Field(..., ge=0, description="Average GA time (s)")
    ga_success_rate: float = Field(..., ge=0, le=100, description="GA success rate (%)")

    # DRL metrics
    drl_best_cost: float = Field(..., ge=0, description="Best DRL cost")
    drl_average_cost: float = Field(..., ge=0, description="Average DRL cost")
    drl_std_dev: float = Field(..., ge=0, description="DRL standard deviation")
    drl_average_time: float = Field(..., ge=0, description="Average DRL time (s)")
    drl_success_rate: float = Field(
        ..., ge=0, le=100, description="DRL success rate (%)"
    )

    # Comparison
    winner: str = Field(..., description="Winning algorithm")
    winner_reason: str = Field(..., description="Reason for the winner")
    cost_difference: float = Field(..., description="Cost difference (GA - DRL)")
    time_difference: float = Field(..., description="Time difference (GA - DRL)")
