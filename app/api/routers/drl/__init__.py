from fastapi import APIRouter, HTTPException

from app.api.routers.drl.response_schemas import GetModelsResponse
from app.core.operations.drl import get_models

router = APIRouter(
    prefix="/drl",
    tags=["DRL"],
)


@router.get(
    "/models",
    description="List all available DRL models.",
    response_model=GetModelsResponse,
)
def handle_get_models():
    try:
        return get_models()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {str(e)}",
        )
