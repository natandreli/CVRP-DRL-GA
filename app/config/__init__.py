from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App configuration settings."""

    # API Settings
    API_TITLE: str = "NeuroGen CVRP Backend"
    API_DESCRIPTION: str = "Backend API for CVRP solving using Deep Reinforcement Learning and Genetic Algorithms"
    API_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    SOLOMON_DIR: Path = DATA_DIR / "solomon"
    INSTANCES_DIR: Path = DATA_DIR / "instances"
    SOLUTIONS_DIR: Path = DATA_DIR / "solutions"
    MODELS_DIR: Path = BASE_DIR / "models" / "drl"
    RESULTS_DIR: Path = BASE_DIR / "results"
    PLOTS_DIR: Path = RESULTS_DIR / "plots"
    METRICS_DIR: Path = RESULTS_DIR / "metrics"

    # DRL Settings
    DRL_DEVICE: str = "cpu"  # "cuda" if gpu is available else "cpu"
    DRL_DEFAULT_EPISODES: int = 200
    DRL_DEFAULT_LR: float = 0.001
    DRL_EPSILON_START: float = 1.0
    DRL_EPSILON_END: float = 0.1
    DRL_EPSILON_DECAY: float = 0.995
    DRL_BATCH_SIZE: int = 16
    DRL_REPLAY_BUFFER_SIZE: int = 5000
    DRL_HIDDEN_DIMS: list[int] = [64, 64]

    # GA Settings
    GA_DEFAULT_POPULATION: int = 50
    GA_DEFAULT_GENERATIONS: int = 150
    GA_DEFAULT_CROSSOVER_RATE: float = 0.8
    GA_DEFAULT_MUTATION_RATE: float = 0.2
    GA_TOURNAMENT_SIZE: int = 3
    GA_ELITISM_COUNT: int = 2

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost", "*"]

    # Logging
    LOGGING_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()


# Global instance
settings = get_settings()

# Create directories if they don't exist
for directory in [
    settings.DATA_DIR,
    settings.SOLOMON_DIR,
    settings.INSTANCES_DIR,
    settings.SOLUTIONS_DIR,
    settings.MODELS_DIR,
    settings.RESULTS_DIR,
    settings.PLOTS_DIR,
    settings.METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
