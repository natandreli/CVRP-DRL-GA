from typing import List, Optional

from pydantic import BaseModel


class TrainingSpecs(BaseModel):
    algorithm: str
    total_episodes: int
    instance_distribution: str
    problem_size: str
    learning_strategy: str
    validation_method: str


class ModelInfo(BaseModel):
    id: str
    name: str
    subname: Optional[str] = None
    description: Optional[str] = None
    training_summary: Optional[str] = None
    training_specs: Optional[TrainingSpecs] = None


class GetModelsResponse(BaseModel):
    models: List[ModelInfo]
