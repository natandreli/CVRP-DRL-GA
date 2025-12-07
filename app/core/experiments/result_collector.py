import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.schemas import Solution


@dataclass
class ExperimentResult:
    """
    Comprehensive result data for a single experimental run.

    Attributes:
        instance_id: Unique identifier of the CVRP instance
        instance_name: Human-readable name of the instance
        instance_size: Number of customers
        configuration: Experimental configuration (ga_pure, drl_junior_ga, etc.)
        agent_used: DRL agent used for initialization (None for ga_pure)
        replicate: Replicate number (1-5)
        seed: Random seed used

        initial_cost: Best cost in initial population P0
        final_cost: Best cost after evolution
        improvement_gap: Percentage improvement (initial - final) / initial * 100

        total_time: Total execution time (seconds)
        initialization_time: Time to generate P0 (seconds)
        evolution_time: Time for GA evolution (seconds)

        generations_run: Total generations executed
        generations_to_convergence: Generations until stagnation
        convergence_history: List of best fitness per generation

        best_solution: Final best solution object
    """

    # Instance metadata
    instance_id: str
    instance_name: str
    instance_size: int

    # Experimental setup
    configuration: str  # ga_pure, drl_junior_ga, drl_mid_ga, drl_expert_ga
    agent_used: Optional[str] = None  # junior, mid, expert, or None
    replicate: int = 1
    seed: int = 42

    # Quality metrics
    initial_cost: float = 0.0
    final_cost: float = 0.0
    improvement_gap: float = 0.0

    # Time metrics
    total_time: float = 0.0
    initialization_time: float = 0.0
    evolution_time: float = 0.0

    # Convergence metrics
    generations_run: int = 0
    generations_to_convergence: Optional[int] = None
    convergence_history: list[float] = field(default_factory=list)

    # Solution data
    best_solution: Optional[Solution] = None

    def to_dict(self, include_solution: bool = False) -> dict:
        """
        Convert result to dictionary.

        Args:
            include_solution (bool): If True, include full solution object

        Returns:
            Dictionary representation
        """
        data = asdict(self)

        if not include_solution:
            data.pop("best_solution", None)
        elif self.best_solution:
            data["best_solution"] = self.best_solution.model_dump()

        return data

    def to_json(self, filepath: Path, include_solution: bool = False) -> None:
        """
        Save result to JSON file.

        Args:
            filepath (Path): Path to save the JSON file
            include_solution (bool): If True, include full solution object
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(include_solution=include_solution), f, indent=2)


class ResultCollector:
    """
    Centralized collector for experimental results.

    Manages result storage, aggregation, and persistence across all experimental runs.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize result collector.

        Args:
            output_dir (Path): Base directory for storing results
        """
        self.output_dir = output_dir
        self.results: list[ExperimentResult] = []

        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw_data").mkdir(exist_ok=True)

        for config in ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]:
            (self.output_dir / "raw_data" / config).mkdir(exist_ok=True)

    def add_result(self, result: ExperimentResult) -> None:
        """
        Add a result to the collector.

        Args:
            result: ExperimentResult object to add
        """
        self.results.append(result)

        # Save individual result
        filename = (
            f"{result.instance_name}_"
            f"{result.configuration}_"
            f"rep{result.replicate}_"
            f"seed{result.seed}.json"
        )

        filepath = self.output_dir / "raw_data" / result.configuration / filename
        result.to_json(filepath, include_solution=True)

    def save_summary(self) -> None:
        """
        Save summary statistics for all collected results.

        Creates a comprehensive JSON file with aggregated metrics.
        """
        summary = {
            "total_runs": len(self.results),
            "configurations": {},
        }

        # Group by configuration
        for config in ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]:
            config_results = [r for r in self.results if r.configuration == config]

            if not config_results:
                continue

            summary["configurations"][config] = {
                "total_runs": len(config_results),
                "avg_final_cost": sum(r.final_cost for r in config_results)
                / len(config_results),
                "avg_improvement_gap": sum(r.improvement_gap for r in config_results)
                / len(config_results),
                "avg_total_time": sum(r.total_time for r in config_results)
                / len(config_results),
                "avg_generations_to_convergence": sum(
                    r.generations_to_convergence
                    for r in config_results
                    if r.generations_to_convergence
                )
                / len([r for r in config_results if r.generations_to_convergence]),
            }

        # Save summary
        summary_path = self.output_dir / "summary_statistics.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"Summary statistics saved to: {summary_path}")
        print(f"{'=' * 80}")

    def get_results_by_configuration(
        self, configuration: str
    ) -> list[ExperimentResult]:
        """
        Retrieve all results for a specific configuration.

        Args:
            configuration (str): Configuration name (e.g., 'ga_pure')

        Returns:
            List of results for the configuration
        """
        return [r for r in self.results if r.configuration == configuration]

    def get_results_by_instance_size(self, size: int) -> list[ExperimentResult]:
        """
        Retrieve all results for instances of a specific size.

        Args:
            size: Number of customers

        Returns:
            List of results for instances of that size
        """
        return [r for r in self.results if r.instance_size == size]
