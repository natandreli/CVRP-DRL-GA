import json

from fastapi import UploadFile

from app.config import settings
from app.core.utils.vrplib_loader import load_vrplib_instance
from app.exceptions import InstanceParseException, UnsupportedFileFormatException
from app.schemas.cvrp_instance import CVRPInstance


def get_presets() -> list[CVRPInstance]:
    """
    Load preset instances from the instances directory.

    Returns:
        list[CVRPInstance]: List of preset CVRP instances
    """
    presets = []
    instances_dir = settings.INSTANCES_DIR

    if not instances_dir.exists():
        return presets

    for file_path in instances_dir.glob("*.vrp"):
        try:
            instance = load_vrplib_instance(str(file_path))
            presets.append(instance)
        except Exception:
            continue

    for file_path in instances_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                instance = CVRPInstance(**data)
                presets.append(instance)
        except Exception:
            continue

    return presets


async def upload_instance(file: UploadFile) -> CVRPInstance:
    """
    Upload a CVRP instance from a given file path.

    Args:
        file: Uploaded file containing CVRP instance

    Returns:
        CVRPInstance: Loaded CVRP instance

    Raises:
        UnsupportedFileFormatException: If file format is not supported
        InstanceParseException: If file cannot be parsed
    """
    filename = file.filename or "unknown"

    # Check file format
    if not (filename.endswith(".json") or filename.endswith(".vrp")):
        raise UnsupportedFileFormatException(filename)

    try:
        content = await file.read()

        # Try JSON first
        if filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            instance = CVRPInstance(**data)
            return instance

        # Try VRPLib format
        else:  # .vrp
            temp_path = settings.INSTANCES_DIR / filename
            with open(temp_path, "wb") as f:
                f.write(content)

            instance = load_vrplib_instance(str(temp_path))
            return instance

    except (UnsupportedFileFormatException, InstanceParseException):
        raise
    except Exception as e:
        raise InstanceParseException(filename, str(e))
