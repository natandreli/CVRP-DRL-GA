from fastapi import APIRouter, HTTPException, UploadFile

from app.api.routers.instances.payload_schemas import (
    GenerateClusteredInstanceRequest,
    GenerateRandomInstanceRequest,
)
from app.core.operations.instances import get_presets, upload_instance
from app.core.utils.instance_generator import (
    generate_clustered_instance as generate_clustered,
)
from app.core.utils.instance_generator import (
    generate_random_instance as generate_random,
)
from app.exceptions import InstanceParseException, UnsupportedFileFormatException
from app.schemas.cvrp_instance import CVRPInstance

router = APIRouter(
    prefix="/instances",
    tags=["Instances"],
)


@router.get(
    "/presets",
    description="List all available instance presets.",
    response_model=list[CVRPInstance],
)
def handle_get_presets():
    try:
        return get_presets()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load presets: {str(e)}",
        )


@router.post(
    "/generate/random",
    description="Generate a CVRP instance with randomly distributed customers.",
    response_model=CVRPInstance,
)
def handle_enerate_random_instance(request: GenerateRandomInstanceRequest):
    try:
        instance = generate_random(
            num_customers=request.num_customers,
            grid_size=request.grid_size,
            vehicle_capacity=request.vehicle_capacity,
            min_customer_demand=request.min_demand,
            max_customer_demand=request.max_demand,
            seed=request.seed,
        )
        return instance
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
        instance = generate_clustered(
            num_customers=request.num_customers,
            grid_size=request.grid_size,
            num_clusters=request.num_clusters,
            vehicle_capacity=request.vehicle_capacity,
            min_customer_demand=request.min_demand,
            max_customer_demand=request.max_demand,
            seed=request.seed,
        )
        return instance
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
