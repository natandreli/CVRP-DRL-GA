from typing import Optional

from pydantic import BaseModel, Field

from app.config import settings


class DRLConfig(BaseModel):
    """Deep Reinforcement Learning Configuration."""

    # Training settings
    episodes: int = Field(
        default=settings.DRL_DEFAULT_EPISODES, gt=0, description="Number of episodes"
    )
    learning_rate_actor: float = Field(
        default=settings.DRL_DEFAULT_LR, gt=0, description="Learning rate (Actor)"
    )
    learning_rate_critic: float = Field(
        default=settings.DRL_DEFAULT_LR * 0.5,
        gt=0,
        description="Learning rate (Critic)",
    )
    epsilon_start: float = Field(
        default=settings.DRL_EPSILON_START,
        ge=0,
        le=1,
        description="Initial epsilon (exploration)",
    )
    epsilon_end: float = Field(
        default=settings.DRL_EPSILON_END,
        ge=0,
        le=1,
        description="Final epsilon (exploration)",
    )
    epsilon_decay: float = Field(
        default=settings.DRL_EPSILON_DECAY,
        gt=0,
        le=1,
        description="Decay factor for epsilon",
    )

    # Network architecture
    hidden_dims: list[int] = Field(
        default=settings.DRL_HIDDEN_DIMS, description="Hidden layer dimensions"
    )

    # Training hyperparameters
    batch_size: int = Field(
        default=settings.DRL_BATCH_SIZE, gt=0, description="Batch size"
    )
    replay_buffer_size: int = Field(
        default=settings.DRL_REPLAY_BUFFER_SIZE,
        gt=0,
        description="Replay buffer size",
    )
    target_update_frequency: int = Field(
        default=10, gt=0, description="Target network update frequency"
    )
    gamma: float = Field(default=0.99, ge=0, le=1, description="Discount factor")

    # Device
    device: str = Field(default=settings.DRL_DEVICE, description="Device (cpu or cuda)")

    # Checkpointing
    save_checkpoints: bool = Field(
        default=True, description="Whether to save checkpoints during training"
    )
    checkpoint_interval: int = Field(
        default=50, gt=0, description="Save checkpoint every N episodes"
    )

    # Reproducibility
    seed: Optional[int] = Field(None, description="Seed for reproducibility")
