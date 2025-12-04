from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.api.routers.instances.payload_schemas import (
    GenerateClusteredInstanceRequest,
    GenerateRandomInstanceRequest,
)
from app.core.operations.instances import (
    generate_clustered_instance,
    generate_random_instance,
    get_all_instances,
    upload_instance,
)
from app.exceptions import InstanceParseException, UnsupportedFileFormatException
from app.schemas.cvrp_instance import CVRPInstance

router = APIRouter(
    prefix="/instances",
    tags=["Instances"],
)


@router.get(
    "/all",
    description="List all available instances.",
    response_model=list[CVRPInstance],
)
def handle_get_all_instances(request: Request):
    try:
        session_id = request.state.session_id
        return get_all_instances(session_id=session_id)
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
def handle_generate_random_instance(
    params: GenerateRandomInstanceRequest, request: Request
):
    try:
        session_id = request.state.session_id
        return generate_random_instance(params, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/generate/clustered",
    description="Generate a CVRP instance with clustered customer distributions.",
    response_model=CVRPInstance,
)
def handle_generate_clustered_instance(
    params: GenerateClusteredInstanceRequest, request: Request
):
    """Generate a clustered CVRP instance."""
    try:
        session_id = request.state.session_id
        return generate_clustered_instance(params, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/upload",
    description="Upload a CVRP instance file.",
    response_model=CVRPInstance,
)
async def handle_upload_instance(file: UploadFile, request: Request):
    try:
        session_id = request.state.session_id
        return await upload_instance(file, session_id=session_id)
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
