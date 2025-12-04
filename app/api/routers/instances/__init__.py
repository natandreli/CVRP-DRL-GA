from fastapi import APIRouter, HTTPException, UploadFile

from app.api.routers.instances.payload_schemas import (
    GenerateClusteredInstanceRequest,
    GenerateRandomInstanceRequest,
)
from app.core.operations.instances import (
    generate_clustered_instance,
    generate_random_instance,
    get_instances,
    upload_instance,
)
from app.exceptions import InstanceParseException, UnsupportedFileFormatException
from app.schemas.cvrp_instance import CVRPInstance

router = APIRouter(
    prefix="/instances",
    tags=["Instances"],
)


@router.get(
    "/instances",
    description="List all available instances.",
    response_model=list[CVRPInstance],
)
def handle_get_instances():
    try:
        return get_instances()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load instances: {str(e)}",
        )


@router.post(
    "/generate/random",
    description="Generate a CVRP instance with randomly distributed customers.",
    response_model=CVRPInstance,
)
def handle_generate_random_instance(request: GenerateRandomInstanceRequest):
    try:
        return generate_random_instance(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/generate/clustered",
    description="Generate a CVRP instance with clustered customer distributions.",
    response_model=CVRPInstance,
)
def handle_generate_clustered_instance(request: GenerateClusteredInstanceRequest):
    """Generate a clustered CVRP instance."""
    try:
        return generate_clustered_instance(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/upload",
    description="Upload a CVRP instance file.",
    response_model=CVRPInstance,
)
async def handle_upload_instance(file: UploadFile):
    try:
        return await upload_instance(file)
    except UnsupportedFileFormatException as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except InstanceParseException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse '{e.filename}': {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error uploading file: {str(e)}",
        )
