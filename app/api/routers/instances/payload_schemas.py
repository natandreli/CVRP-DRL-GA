from typing import Optional

from pydantic import BaseModel, Field


class GenerateRandomInstanceRequest(BaseModel):
    num_customers: int = Field(
        20,
        description="Number of customers.",
        example=20,
        gt=1,
    )
    grid_size: int = Field(
        100,
        description="Size of the grid (0,0) to (grid_size, grid_size).",
        example=100,
        gt=1,
    )
    vehicle_capacity: int = Field(
        100,
        description="Vehicle capacity.",
        example=100,
        gt=1,
    )
    min_demand: int = Field(
        5,
        description="Minimum customer demand.",
        example=5,
        gt=1,
    )
    max_demand: int = Field(
        30,
        description="Maximum customer demand.",
        example=30,
        gt=1,
    )
    seed: Optional[int] = Field(
        None,
        description="Seed for reproducibility.",
        example=42,
    )


class GenerateClusteredInstanceRequest(GenerateRandomInstanceRequest):
    num_clusters: int = Field(
        3,
        description="Number of clusters.",
        example=3,
        gt=1,
    )
