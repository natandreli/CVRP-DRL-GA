from pydantic import BaseModel, Field

from app.schemas import CVRPInstance, GAConfig


class ComparisonRequest(BaseModel):
    instance: CVRPInstance = Field(
        ...,
        description="CVRP instance to solve.",
        example="instance_001",
    )
    ga_config: GAConfig = Field(
        ...,
        description="Genetic Algorithm configuration.",
        example={
            "population_size": 100,
            "generations": 500,
            "crossover_rate": 0.8,
            "selection_method": "tournament",
            "tournament_size": 5,
            "crossover_method": "ox",
            "mutation_method": "swap",
            "mutation_rate": 0.1,
            "seed": 42,
        },
    )
    drl_model_id: str = Field(
        ...,
        description="ID of the DRL model to use for generating initial population.",
        example="drl_model_001",
    )
