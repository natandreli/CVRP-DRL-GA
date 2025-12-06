from typing import Optional

from pydantic import BaseModel

from app.schemas import Solution


class ConvergencePoint(BaseModel):
    generation: int
    fitness: float


class AlgorithmResult(BaseModel):
    algorithm_name: str
    initial_fitness: float
    final_solution: Solution
    computation_time: float
    population_generation_time: float
    ga_convergence_time: float
    convergence_history: Optional[list[ConvergencePoint]] = None


class ComparisonMetrics(BaseModel):
    improvement_absolute: float
    improvement_percentage: float
    time_difference: float
    initial_gap_percentage: float
    vehicles_difference: int
    winner: str


class ComparisonResponse(BaseModel):
    neurogen: AlgorithmResult
    ga_pure: AlgorithmResult
    metrics: ComparisonMetrics
