from app.api.routers.drl.response_schemas import GetModelsResponse, ModelInfo
from app.config import settings
from app.core.constants.models import MODELS


def get_models() -> GetModelsResponse:
    """
    List all trained DRL models available in checkpoints directory.

    Returns:
        GetModelsResponse: List of available DRL models
    """
    models = []
    checkpoints_dir = settings.CHECKPOINTS_DIR

    for model_id, config in MODELS.items():
        model_path = checkpoints_dir / f"{model_id}.pth"

        if not model_path.exists():
            continue

        models.append(
            ModelInfo(
                id=model_id,
                name=config["name"],
                description=config["description"],
                training_summary=config["training_summary"],
                training_specs=config["training_specs"],
            )
        )

    return GetModelsResponse(models=models)
