import json
from pathlib import Path
from typing import Optional

from app.config import settings
from app.core.utils import (
    calculate_distance_matrix,
    generate_clustered_instance,
    generate_random_instance,
    load_vrplib_instance,
)
from app.schemas import CVRPInstance


class InstanceManager:
    """Singleton to manage CVRP instances in memory and disk."""

    def __init__(self):
        """Initialize manager with empty cache."""
        self._instances: dict[str, CVRPInstance] = {}
        self._distance_matrices: dict[str, any] = {}  # Cache distance matrices
        self._instances_dir = settings.INSTANCES_DIR

    def get_instance(self, instance_id: str) -> CVRPInstance:
        """
        Get instance by ID (from cache, disk, or vrplib).

        Args:
            instance_id (str): Instance ID

        Returns:
            CVRPInstance: Requested instance

        Raises:
            ValueError: If the instance does not exist
        """
        # 1. Search in cache
        if instance_id in self._instances:
            return self._instances[instance_id]

        # 2. Search on disk (data/instances/)
        file_path = self._instances_dir / f"{instance_id}.json"
        if file_path.exists():
            instance = self._load_from_file(file_path)
            self._instances[instance_id] = instance
            return instance

        # 3. Try to load from vrplib
        try:
            instance = load_vrplib_instance(instance_id)
            self._instances[instance_id] = instance
            # Save to disk for future use
            self.save_instance(instance_id, instance)
            return instance
        except ValueError:
            pass

        raise ValueError(f"Instance '{instance_id}' not found")

    def save_instance(self, instance_id: str, instance: CVRPInstance) -> None:
        """
        Save instance to cache and disk.

        Args:
            instance_id (str): Instance ID
            instance (CVRPInstance): CVRP instance
        """
        # Save to cache
        self._instances[instance_id] = instance

        # Save to disk
        file_path = self._instances_dir / f"{instance_id}.json"
        with open(file_path, "w") as f:
            json.dump(instance.model_dump(), f, indent=2)

    def list_instances(self) -> list[str]:
        """
        List IDs of all available instances (disk + cache).

        Returns:
            list[str]: List of instance IDs
        """
        # IDs from disk
        disk_ids = {f.stem for f in self._instances_dir.glob("*.json")}

        # IDs from cache
        cache_ids = set(self._instances.keys())

        # Combine both
        all_ids = disk_ids.union(cache_ids)

        return sorted(list(all_ids))

    def delete_instance(self, instance_id: str) -> None:
        """
        Delete instance from cache and disk.

        Args:
            instance_id (str): Instance ID
        """
        # Delete from cache
        if instance_id in self._instances:
            del self._instances[instance_id]

        if instance_id in self._distance_matrices:
            del self._distance_matrices[instance_id]

        # Delete from disk
        file_path = self._instances_dir / f"{instance_id}.json"
        if file_path.exists():
            file_path.unlink()

    def generate_and_save_instance(
        self,
        num_customers: int,
        vehicle_capacity: int,
        instance_type: str = "random",
        seed: Optional[int] = None,
    ) -> CVRPInstance:
        """
        Generate synthetic instance and save it.

        Args:
            num_customers (int): Number of customers
            vehicle_capacity (int): Vehicle capacity
            instance_type (str): Type of instance ('random' or 'clustered')
            seed (Optional[int]): Seed for reproducibility

        Returns:
            CVRPInstance: Generated instance
        """
        if instance_type == "random":
            instance = generate_random_instance(
                num_customers=num_customers,
                vehicle_capacity=vehicle_capacity,
                seed=seed,
            )
        elif instance_type == "clustered":
            instance = generate_clustered_instance(
                num_customers=num_customers,
                vehicle_capacity=vehicle_capacity,
                seed=seed,
            )
        else:
            raise ValueError(f"Invalid instance_type: {instance_type}")

        # Save instance
        self.save_instance(instance.id, instance)

        return instance

    def get_distance_matrix(self, instance_id: str):
        """
        Get or compute distance matrix for instance.

        Args:
            instance_id (str): Instance ID

        Returns:
            numpy.ndarray: Distance matrix
        """
        if instance_id in self._distance_matrices:
            return self._distance_matrices[instance_id]

        instance = self.get_instance(instance_id)
        distance_matrix = calculate_distance_matrix(instance)

        # Cache it
        self._distance_matrices[instance_id] = distance_matrix

        return distance_matrix

    def _load_from_file(self, file_path: Path) -> CVRPInstance:
        """Load instance from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        return CVRPInstance(**data)


# Singleton global
instance_manager = InstanceManager()
