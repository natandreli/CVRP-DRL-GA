"""
Main experimental execution script for NeuroGen validation.

Systematically evaluates the hybrid DRL-GA architecture against pure GA baseline
across 120 stratified synthetic instances with controlled replication.

Experimental Design:
- 4 configurations: GA Pure, DRL-Junior+GA, DRL-Mid+GA, DRL-Expert+GA
- 3 ranges: Junior (20-50), Mid (60-100), Expert (110-150)
- 10 instances per size (5 random + 5 clustered)
- 5 replicates per instance-configuration pair
- Total: 2,400 experimental runs

Run with: `python -m app.scripts.experiments.run_experiments`

Options:
    --quick: Run quick test (1 size per range, 2 replicates)
    --range: Run only specific range (junior, mid, expert)
    --config: Run only specific configuration (ga_pure, drl_junior_ga, drl_mid_ga, drl_expert_ga)
"""

import argparse
from datetime import datetime

from app.config import settings
from app.core.experiments.experiment_runner import ExperimentRunner
from app.core.experiments.result_collector import ResultCollector
from app.schemas import GAConfig
from app.scripts.experiments.generate_experiment_instances import get_all_instances


def run_experiments(
    quick_mode: bool = False,
    range_filter: str = None,
    config_filter: str = None,
) -> None:
    """
    Execute complete experimental protocol.

    Args:
        quick_mode (bool): If True, run abbreviated test (1 size/range, 2 reps)
        range_filter (str): Optional filter for instance range (junior, mid, expert)
        config_filter (str): Optional filter for configuration
    """
    print("=" * 80)
    print("NEURGEN EXPERIMENTAL VALIDATION")
    print("Hybrid DRL-GA Architecture for CVRP")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if quick_mode:
        print("\nQUICK MODE: Running abbreviated test")
        print("  - 1 size per range (20, 60, 110 customers)")
        print("  - 2 replicates per configuration")
        print("  - Total: ~80 runs\n")
    else:
        print("\nFULL MODE: Complete experimental protocol")
        print("  - 12 sizes across 3 ranges")
        print("  - 10 instances per size (5 random + 5 clustered)")
        print("  - 5 replicates per instance-configuration")
        print("  - Total: 2,400 runs\n")

    if range_filter:
        print(f"Range filter: {range_filter}")
    if config_filter:
        print(f"Configuration filter: {config_filter}")

    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = settings.EXPERIMENT_RESULTS_DIR / f"experiments_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Initialize components
    ga_config = GAConfig(
        population_size=50,
        generations=100,
        crossover_rate=0.85,
        mutation_rate=0.15,
        selection_method="tournament",
        tournament_size=3,
        elitism_count=2,
        crossover_method="ox",
        mutation_method="2opt",
    )

    runner = ExperimentRunner(ga_config=ga_config, drl_diversity=0.3)
    collector = ResultCollector(output_dir=output_dir)

    # Load instances
    print("Loading experimental instances...")
    all_instances = get_all_instances(range_name=range_filter)

    if not all_instances:
        print("No instances found!")
        print("Please generate instances first with:")
        print("  python -m app.scripts.experiments.generate_experiment_instances")
        return

    # Filter instances for quick mode
    if quick_mode:
        quick_sizes = {20, 60, 110}
        all_instances = [
            (name, inst)
            for name, inst in all_instances
            if inst.num_customers in quick_sizes
        ]

    print(f"✓ Loaded {len(all_instances)} instances\n")

    # Define configurations
    configurations = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]

    if config_filter:
        configurations = [c for c in configurations if c == config_filter]

    # Define replicates
    num_replicates = 2 if quick_mode else 5

    # Calculate total runs
    total_runs = len(all_instances) * len(configurations) * num_replicates
    current_run = 0

    print(f"{'=' * 80}")
    print("STARTING EXPERIMENTAL RUNS")
    print(f"{'=' * 80}")
    print(f"Total runs to execute: {total_runs}\n")

    # Execute experiments
    for instance_name, instance in all_instances:
        print(f"\n{'─' * 80}")
        print(f"Instance: {instance_name} ({instance.num_customers} customers)")
        print(f"{'─' * 80}")

        for config_name in configurations:
            print(f"\n  Configuration: {config_name}")

            for replicate in range(1, num_replicates + 1):
                current_run += 1

                # Generate unique seed
                seed = hash(f"{instance_name}_{config_name}_{replicate}") % (2**31)

                print(
                    f"    Replicate {replicate}/{num_replicates} (seed={seed})...",
                    end=" ",
                    flush=True,
                )

                try:
                    # Run experiment
                    if config_name == "ga_pure":
                        result = runner.run_ga_pure(
                            instance=instance,
                            replicate=replicate,
                            seed=seed,
                        )
                    else:
                        # Extract agent name from config (e.g., drl_junior_ga -> junior)
                        agent_name = config_name.split("_")[1]

                        result = runner.run_drl_ga(
                            instance=instance,
                            agent_name=agent_name,
                            replicate=replicate,
                            seed=seed,
                        )

                    # Collect result
                    collector.add_result(result)

                    # Print metrics
                    print(
                        f"Cost: {result.final_cost:.2f}, "
                        f"Gap: {result.improvement_gap:.2f}%, "
                        f"Time: {result.total_time:.2f}s"
                    )

                except Exception as e:
                    print(f"FAILED: {str(e)}")
                    continue

                # Progress update
                if current_run % 10 == 0:
                    progress = (current_run / total_runs) * 100
                    print(f"\n  Progress: {current_run}/{total_runs} ({progress:.1f}%)")

    # Save summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENTAL EXECUTION COMPLETE")
    print(f"{'=' * 80}")

    collector.save_summary()

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {output_dir}")


def main():
    """Parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description="Run NeuroGen experimental validation")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test mode (1 size/range, 2 replicates)",
    )

    parser.add_argument(
        "--range",
        type=str,
        choices=["junior", "mid", "expert"],
        help="Run only specific range",
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"],
        help="Run only specific configuration",
    )

    args = parser.parse_args()

    run_experiments(
        quick_mode=args.quick,
        range_filter=args.range,
        config_filter=args.config,
    )


if __name__ == "__main__":
    main()
