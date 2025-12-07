import argparse
import sys
from pathlib import Path

from app.core.experiments.statistical_analysis import StatisticalAnalyzer


def analyze_results(results_dir: str) -> None:
    """
    Perform statistical analysis on experimental results.

    Args:
        results_dir (str): Path to directory containing experimental results
    """
    results_path = Path(results_dir)

    # Validate directory
    if not results_path.exists():
        print(f"Error: Directory not found: {results_path}")
        print("\nAvailable experiment directories:")

        results_base = Path("results")
        if results_base.exists():
            exp_dirs = sorted(results_base.glob("experiments_*"))
            for exp_dir in exp_dirs:
                print(f"  - {exp_dir}")

        return

    raw_data_dir = results_path / "raw_data"
    if not raw_data_dir.exists():
        print(f"Error: raw_data directory not found in {results_path}")
        print("This does not appear to be a valid experimental results directory")
        return

    # Check for data
    config_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]
    if not config_dirs:
        print(f"Error: No configuration data found in {raw_data_dir}")
        return

    print("\n" + "=" * 80)
    print("NEURGEN EXPERIMENTAL ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {results_path}")
    print("\nConfigurations found:")

    for config_dir in sorted(config_dirs):
        num_files = len(list(config_dir.glob("*.json")))
        print(f"  - {config_dir.name}: {num_files} results")

    print()

    # Run analysis
    try:
        analyzer = StatisticalAnalyzer(results_path)
        analyzer.run_full_analysis()

        print("\n" + "=" * 80)
        print("ANALYSIS FILES GENERATED")
        print("=" * 80)
        print(f"\nThe following files have been created in {results_path}:")
        print("  - summary_statistics.json: Aggregated metrics by configuration")
        print("  - friedman_test.json: Non-parametric test for overall differences")

        if (results_path / "nemenyi_test.json").exists():
            print("  - nemenyi_test.json: Pairwise comparison results")

        print("  - specialization_analysis.json: Agent specialization evaluation")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\nYou can now:")
        print("  1. Review the JSON files for detailed statistical results")
        print("  2. Import data into visualization tools (Python, R, Excel)")
        print("  3. Generate publication-ready plots for your paper")
        print("\nRecommended visualizations:")
        print("  - Boxplots: Distribution of final costs by configuration")
        print("  - Convergence curves: Fitness evolution over generations")
        print("  - Heatmap: Agent performance matrix (agent × instance size)")
        print("  - Bar charts: Average improvement gap by configuration")
        print()

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return


def main():
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze NeuroGen experimental results"
    )

    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        help="Path to experimental results directory",
    )

    args = parser.parse_args()

    # If no directory provided, try to find most recent
    if not args.results_dir:
        results_base = Path("results")

        if not results_base.exists():
            print("Error: No results directory found")
            print(
                "\nUsage: python -m app.scripts.experiments.analyze_results <results_directory>"
            )
            sys.exit(1)

        exp_dirs = sorted(results_base.glob("experiments_*"), reverse=True)

        if not exp_dirs:
            print("Error: No experiment directories found in results/")
            print("\nPlease run experiments first with:")
            print("  python -m app.scripts.experiments.run_experiments")
            sys.exit(1)

        # Use most recent
        args.results_dir = str(exp_dirs[0])
        print(f"No directory specified. Using most recent: {args.results_dir}\n")

    analyze_results(args.results_dir)


if __name__ == "__main__":
    main()
