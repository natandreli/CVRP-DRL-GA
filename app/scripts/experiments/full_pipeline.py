import argparse
from pathlib import Path

from app.scripts.experiments.generate_experiment_instances import (
    generate_experiment_instances,
    get_all_instances,
)


def check_prerequisites() -> dict:
    """
    Check availability of experimental prerequisites.

    Returns:
        dict: Dictionary with status of each component
    """
    status = {
        "instances": False,
        "agents": {
            "junior": False,
            "mid": False,
            "expert": False,
        },
    }

    # Check instances
    instances_dir = Path("app/data/experiment_instances")
    if instances_dir.exists():
        instances = get_all_instances()
        status["instances"] = len(instances) >= 120

    # Check agent checkpoints
    checkpoints_dir = Path("app/core/drl/checkpoints")
    if checkpoints_dir.exists():
        status["agents"]["junior"] = (checkpoints_dir / "junior.pth").exists()
        status["agents"]["mid"] = (checkpoints_dir / "mid.pth").exists()
        status["agents"]["expert"] = (checkpoints_dir / "expert.pth").exists()

    return status


def print_status(status: dict) -> None:
    """Print the status of prerequisites."""
    print("\n" + "=" * 80)
    print("PREREQUISITE STATUS CHECK")
    print("=" * 80)

    print("\nExperimental Instances:")
    if status["instances"]:
        print("  ✓ Complete dataset available (120 instances)")
    else:
        print("  ✗ Dataset not found or incomplete")
        print("    → Instances will be generated automatically")

    print("\nDRL Agent Checkpoints:")
    for agent, exists in status["agents"].items():
        if exists:
            print(f"  ✓ {agent.capitalize()} agent checkpoint available")
        else:
            print(f"  ✗ {agent.capitalize()} agent checkpoint not found")
            print("    → You must train the agent with:")
            print(f"       python -m app.scripts.drl.train_{agent}")

    print()


def run_full_pipeline(
    quick_mode: bool = False,
    skip_generation: bool = False,
    skip_training: bool = False,
    skip_experiments: bool = False,
    results_dir: str = None,
) -> None:
    """
    Execute the complete experimental pipeline.

    Args:
        quick_mode (bool): Run in quick mode (testing)
        skip_generation (bool): Skip instance generation
        skip_training (bool): Skip training verification
        skip_experiments (bool): Skip experiments (analysis only, requires results_dir)
        results_dir (str): Directory with results for analysis
    """
    print("\n" + "=" * 80)
    print("NEURGEN COMPLETE EXPERIMENTAL PIPELINE")
    print("=" * 80)

    if quick_mode:
        print("\nRunning in QUICK MODE (for testing)")
    else:
        print("\nRunning in FULL MODE (for publication)")

    # Check prerequisites
    status = check_prerequisites()
    print_status(status)

    # Step 0: Validate prerequisites
    if not skip_experiments:
        # Need agents trained
        agents_ready = all(status["agents"].values())

        if not agents_ready and not skip_training:
            print("ERROR: DRL agents not trained")
            print("\nYou must train all agents before running experiments:")
            print("  python -m app.scripts.drl.train_junior")
            print("  python -m app.scripts.drl.train_mid")
            print("  python -m app.scripts.drl.train_expert")
            print("\nOr skip this check with --skip-training")
            return

    # Step 1: Generate instances
    if not skip_generation and not skip_experiments:
        if not status["instances"]:
            print("\n" + "─" * 80)
            print("STEP 1: GENERATING EXPERIMENTAL INSTANCES")
            print("─" * 80)

            try:
                generate_experiment_instances()
            except Exception as e:
                print(f"\nERROR during instance generation: {e}")
                return
        else:
            print("\n✓ STEP 1: Using existing experimental instances")

    # Step 2: Run experiments
    if not skip_experiments:
        print("\n" + "─" * 80)
        print("STEP 2: RUNNING EXPERIMENTAL PROTOCOL")
        print("─" * 80)

        from app.scripts.experiments.run_experiments import run_experiments

        try:
            run_experiments(quick_mode=quick_mode)

            # Find most recent results directory
            results_base = Path("results")
            exp_dirs = sorted(results_base.glob("experiments_*"), reverse=True)

            if exp_dirs:
                results_dir = str(exp_dirs[0])
            else:
                print("ERROR: No results directory found after experiments")
                return
        except Exception as e:
            print(f"\nERROR during experimental runs: {e}")
            import traceback

            traceback.print_exc()
            return
    else:
        if not results_dir:
            # Find most recent
            results_base = Path("results")
            exp_dirs = sorted(results_base.glob("experiments_*"), reverse=True)

            if not exp_dirs:
                print("ERROR: No results directory found")
                print("Please provide --results-dir or run experiments first")
                return

            results_dir = str(exp_dirs[0])
            print(f"\n✓ STEP 2: Using existing results from {results_dir}")

    # Step 3: Statistical analysis
    print("\n" + "─" * 80)
    print("STEP 3: STATISTICAL ANALYSIS")
    print("─" * 80)

    from app.scripts.experiments.analyze_results import analyze_results

    try:
        analyze_results(results_dir)
    except Exception as e:
        print(f"\nERROR during statistical analysis: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 4: Generate visualizations
    print("\n" + "─" * 80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("─" * 80)

    try:
        from app.scripts.experiments.visualizations import generate_all_visualizations

        output_dir = Path(results_dir) / "visualizations"
        generate_all_visualizations(results_dir, str(output_dir))
    except ImportError:
        print("\nWARNING: Visualization libraries not installed")
        print("To generate plots, install dependencies:")
        print("  pip install matplotlib seaborn pandas")
        print("\nYou can generate visualizations later with:")
        print(f"  python -m app.scripts.experiments.visualizations {results_dir}")
    except Exception as e:
        print(f"\nERROR during visualization: {e}")
        import traceback

        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print("\nGenerated artifacts:")
    print("  raw_data/: Individual experimental runs")
    print("  summary_statistics.json: Aggregated metrics")
    print("  friedman_test.json: Statistical test results")
    print("  nemenyi_test.json: Pairwise comparisons")
    print("  specialization_analysis.json: Agent specialization")

    viz_dir = Path(results_dir) / "visualizations"
    if viz_dir.exists():
        print("  visualizations/: Publication-ready plots")

    print("\n" + "=" * 80)
    print("NEXT STEPS FOR YOUR PAPER")
    print("=" * 80)
    print("\n1. Review statistical test results:")
    print("   - Check friedman_test.json for H1 validation")
    print("   - Check specialization_analysis.json for H2 validation")

    print("\n2. Import visualizations into your paper:")
    if viz_dir.exists():
        print(f"   - Figures available in: {viz_dir}")
    print("   - cost_boxplots.png → Section 5.1")
    print("   - convergence_*.png → Section 5.1")
    print("   - specialization_heatmap.png → Section 5.2")
    print("   - improvement_gap.png → Section 5.1")

    print("\n3. Create results tables:")
    print("   - Use summary_statistics.json for Table 4 (results comparison)")
    print("   - Use friedman_test.json for ranking table")

    print("\n4. Write discussion section:")
    print("   - Interpret p-values from statistical tests")
    print("   - Discuss specialization findings")
    print("   - Compare with related work")

    print()


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Run complete NeuroGen experimental pipeline"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode (1 size/range, 2 replicates)",
    )

    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip instance generation (use existing)",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip agent training check (use at your own risk)",
    )

    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experimental runs (only analysis and visualization)",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        help="Results directory for analysis (required with --skip-experiments)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.skip_experiments and not args.results_dir:
        # Try to find most recent
        results_base = Path("results")
        if results_base.exists():
            exp_dirs = sorted(results_base.glob("experiments_*"), reverse=True)
            if exp_dirs:
                print(f"No --results-dir specified, using most recent: {exp_dirs[0]}")
                args.results_dir = str(exp_dirs[0])

    run_full_pipeline(
        quick_mode=args.quick,
        skip_generation=args.skip_generation,
        skip_training=args.skip_training,
        skip_experiments=args.skip_experiments,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
