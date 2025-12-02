import vrplib

from app.config.logging import logger
from app.schemas import CVRPInstance


def load_vrplib_instance(instance_name: str) -> CVRPInstance:
    """
    Load a CVRP instance from vrplib by its name.

    Args:
        instance_name (str): Name of the instance in vrplib

    Returns:
        CVRPInstance: Loaded CVRP instance

    Raises:
        ValueError: If the instance cannot be loaded
    """
    try:
        instance_data = vrplib.read_instance(instance_name)
        cvrp_instance = CVRPInstance.from_vrplib(instance_data, instance_name)

        return cvrp_instance

    except Exception as e:
        logger.info(f"Error loading the instance '{instance_name}': {e}")
        raise ValueError(f"Instance '{instance_name}' could not be loaded from vrplib.")


def list_available_vrplib_instances(limit: int = 50) -> list[str]:
    """
    List available instances in vrplib.

    Args:
        limit (int): Maximum number of instances to list

    Returns:
        list[str]: List of instance names
    """
    try:
        all_names = vrplib.list_names()
        return all_names[:limit]
    except Exception:
        # If vrplib is not configured, return an empty list
        return []


def load_vrplib_solution(instance_name: str) -> dict:
    """
    Load known optimal solution from vrplib (if it exists).

    Args:
        instance_name: Name of the instance

    Returns:
        Dictionary with solution (routes, cost)

    Raises:
        ValueError: If no solution exists for that instance
    """
    try:
        solution_data = vrplib.read_solution(instance_name)
        return solution_data
    except Exception as e:
        logger.info(f"No known solution exists for '{instance_name}': {e}")
        raise ValueError(f"No known solution exists for '{instance_name}': {e}")
