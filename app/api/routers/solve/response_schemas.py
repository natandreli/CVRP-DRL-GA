from pydantic import BaseModel

from app.schemas import Solution


class AlgorithmResult(BaseModel):
    algorithm_name: str
    initial_fitness: float
    final_solution: Solution
    computation_time: float


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
