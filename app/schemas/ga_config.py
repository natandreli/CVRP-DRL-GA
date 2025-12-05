from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.config import settings


class GAConfig(BaseModel):
    """Genetic Algorithm Configuration."""

    population_size: int = Field(
        default=settings.GA_DEFAULT_POPULATION,
        gt=0,
        description="Population size",
    )
    generations: int = Field(
        default=settings.GA_DEFAULT_GENERATIONS,
        gt=0,
        description="Number of generations",
    )
    crossover_rate: float = Field(
        default=settings.GA_DEFAULT_CROSSOVER_RATE,
        ge=0,
        le=1,
        description="Crossover rate",
    )
    mutation_rate: float = Field(
        default=settings.GA_DEFAULT_MUTATION_RATE,
        ge=0,
        le=1,
        description="Mutation rate",
    )
    selection_method: Literal["tournament", "roulette"] = Field(
        default="tournament", description="Selection method"
    )
    tournament_size: int = Field(
        default=settings.GA_TOURNAMENT_SIZE,
        gt=1,
        description="Tournament size (if selection_method='tournament')",
    )
    elitism_count: int = Field(
        default=settings.GA_ELITISM_COUNT,
        ge=1,
        description="Number of top individuals to preserve (elitism)",
    )
    crossover_method: Literal["ox", "pmx", "edge"] = Field(
        default="ox", description="Crossover method (Order Crossover, PMX, Edge)"
    )
    mutation_method: Literal["swap", "insert", "inversion", "2opt"] = Field(
        default="swap", description="Mutation method"
    )
    seed: Optional[int] = Field(None, description="Seed for reproducibility")
