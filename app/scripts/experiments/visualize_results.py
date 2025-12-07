import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.figsize"] = (8, 6)


def load_results(results_dir: Path) -> Dict[str, List[dict]]:
    """
    Load all experimental results from directory.

    Args:
        results_dir (Path): Path to experimental results directory

    Returns:
        Dictionary mapping configuration names to list of result dicts
    """
    raw_data_dir = results_dir / "raw_data"

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"raw_data directory not found in {results_dir}")

    data_by_config = {}

    for config_dir in raw_data_dir.iterdir():
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        results = []

        for result_file in config_dir.glob("*.json"):
            with open(result_file, "r") as f:
                results.append(json.load(f))

        data_by_config[config_name] = results

    return data_by_config


def plot_convergence_curves(data_by_config: Dict[str, List[dict]], output_dir: Path):
    """
    Plot convergence curves for all configurations (Figure for Section 5.1).

    Shows mean convergence ± standard deviation across all replicates.
    Highlights the quality difference in initial population P0.

    Args:
        data_by_config (Dict[str, List[dict]]): Loaded experimental results
        output_dir (Path): Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "ga_pure": "#e74c3c",
        "drl_junior_ga": "#3498db",
        "drl_mid_ga": "#2ecc71",
        "drl_expert_ga": "#9b59b6",
    }

    labels = {
        "ga_pure": "GA Pure (Random Init)",
        "drl_junior_ga": "DRL-Junior + GA",
        "drl_mid_ga": "DRL-Mid + GA",
        "drl_expert_ga": "DRL-Expert + GA",
    }

    for config_name in ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]:
        if config_name not in data_by_config:
            continue

        results = data_by_config[config_name]

        histories = [
            r["convergence_history"] for r in results if r["convergence_history"]
        ]

        if not histories:
            continue

        max_len = max(len(h) for h in histories)
        padded_histories = np.array(
            [h + [h[-1]] * (max_len - len(h)) for h in histories]
        )

        mean_history = padded_histories.mean(axis=0)
        std_history = padded_histories.std(axis=0)

        generations = np.arange(len(mean_history))

        ax.plot(
            generations,
            mean_history,
            label=labels[config_name],
            color=colors[config_name],
            linewidth=2,
        )

        ax.fill_between(
            generations,
            mean_history - std_history,
            mean_history + std_history,
            color=colors[config_name],
            alpha=0.2,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Cost")
    ax.set_title("Convergence Comparison: DRL-Initialized vs. Random Initialization")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(
        2, ax.get_ylim()[1] * 0.95, "P₀ (Initial Population)", fontsize=9, color="gray"
    )

    output_file = output_dir / "convergence_curves.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Convergence curves saved to: {output_file}")
    plt.close()


def plot_cost_boxplots(data_by_config: Dict[str, List[dict]], output_dir: Path):
    """
    Plot boxplots of final cost distribution (Figure for Section 5.1).

    Shows median, quartiles, and outliers for each configuration.

    Args:
        data_by_config (Dict[str, List[dict]]): Loaded experimental results
        output_dir (Path): Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]
    labels = ["GA Pure", "DRL-Junior+GA", "DRL-Mid+GA", "DRL-Expert+GA"]

    data_to_plot = []
    valid_labels = []

    for config in configs:
        if config in data_by_config:
            costs = [r["final_cost"] for r in data_by_config[config]]
            data_to_plot.append(costs)
            valid_labels.append(labels[configs.index(config)])

    bp = ax.boxplot(
        data_to_plot,
        labels=valid_labels,
        patch_artist=True,
        notch=True,
        showmeans=True,
    )

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[: len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Final Cost")
    ax.set_title("Cost Distribution Comparison Across Configurations")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    output_file = output_dir / "cost_boxplots.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Cost boxplots saved to: {output_file}")
    plt.close()


def plot_specialization_heatmap(
    data_by_config: Dict[str, List[dict]], output_dir: Path
):
    """
    Plot heatmap of agent performance by instance range (Figure for Section 5.2).

    Validates hypothesis H2: agents are specialized for their training ranges.

    Args:
        data_by_config (Dict[str, List[dict]]): Loaded experimental results
        output_dir (Path): Directory to save the figure
    """
    ranges = {
        "Junior (20-50)": (20, 50),
        "Mid (60-100)": (60, 100),
        "Expert (110-150)": (110, 150),
    }

    agents = ["junior", "mid", "expert"]

    heatmap_data = np.zeros((len(agents), len(ranges)))

    for i, agent in enumerate(agents):
        config_name = f"drl_{agent}_ga"

        if config_name not in data_by_config:
            continue

        for j, (range_name, (min_size, max_size)) in enumerate(ranges.items()):
            range_costs = [
                r["final_cost"]
                for r in data_by_config[config_name]
                if min_size <= r["instance_size"] <= max_size
            ]

            if range_costs:
                heatmap_data[i, j] = np.mean(range_costs)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(np.arange(len(ranges)))
    ax.set_yticks(np.arange(len(agents)))
    ax.set_xticklabels(list(ranges.keys()))
    ax.set_yticklabels([f"DRL-{a.capitalize()}" for a in agents])

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Final Cost", rotation=270, labelpad=20)

    ax.set_title("Agent Specialization: Performance by Instance Range")
    ax.set_xlabel("Instance Range")
    ax.set_ylabel("Agent")

    output_file = output_dir / "specialization_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Specialization heatmap saved to: {output_file}")
    plt.close()


def plot_initial_quality(data_by_config: Dict[str, List[dict]], output_dir: Path):
    """
    Plot initial population quality (P0) comparison (Figure for Section 5.2).

    Shows that DRL agents generate better starting points.

    Args:
        data_by_config (Dict[str, List[dict]]): Loaded experimental results
        output_dir (Path): Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ranges = {
        "Junior\n(20-50)": (20, 50),
        "Mid\n(60-100)": (60, 100),
        "Expert\n(110-150)": (110, 150),
    }

    agents = ["junior", "mid", "expert"]
    colors = ["#3498db", "#2ecc71", "#9b59b6"]

    x = np.arange(len(ranges))
    width = 0.2

    for i, agent in enumerate(agents):
        config_name = f"drl_{agent}_ga"

        if config_name not in data_by_config:
            continue

        initial_costs = []

        for range_name, (min_size, max_size) in ranges.items():
            range_initial = [
                r["initial_cost"]
                for r in data_by_config[config_name]
                if min_size <= r["instance_size"] <= max_size
            ]

            initial_costs.append(np.mean(range_initial) if range_initial else 0)

        ax.bar(
            x + i * width,
            initial_costs,
            width,
            label=f"DRL-{agent.capitalize()}",
            color=colors[i],
            alpha=0.8,
        )

    if "ga_pure" in data_by_config:
        baseline_costs = []

        for range_name, (min_size, max_size) in ranges.items():
            range_initial = [
                r["initial_cost"]
                for r in data_by_config["ga_pure"]
                if min_size <= r["instance_size"] <= max_size
            ]

            baseline_costs.append(np.mean(range_initial) if range_initial else 0)

        ax.bar(
            x + 3 * width,
            baseline_costs,
            width,
            label="GA Pure (Random)",
            color="#e74c3c",
            alpha=0.8,
        )

    ax.set_ylabel("Initial Cost (P₀)")
    ax.set_title("Initial Population Quality: DRL vs. Random Initialization")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(list(ranges.keys()))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    output_file = output_dir / "initial_quality.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Initial quality plot saved to: {output_file}")
    plt.close()


def visualize_results(results_dir: str) -> None:
    """
    Generate all visualizations for experimental results.

    Args:
        results_dir (str): Path to directory containing experimental results
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory not found: {results_path}")
        return

    print("\n" + "=" * 80)
    print("NEURGEN EXPERIMENTAL VISUALIZATIONS")
    print("=" * 80)
    print(f"Results directory: {results_path}\n")

    print("Loading experimental data...")
    try:
        data_by_config = load_results(results_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    print(
        f"Loaded {sum(len(v) for v in data_by_config.values())} results across {len(data_by_config)} configurations\n"
    )

    figures_dir = results_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Generating visualizations...\n")

    try:
        plot_convergence_curves(data_by_config, figures_dir)
        plot_cost_boxplots(data_by_config, figures_dir)
        plot_specialization_heatmap(data_by_config, figures_dir)
        plot_initial_quality(data_by_config, figures_dir)
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\n{'=' * 80}")
    print(f"All figures saved to: {figures_dir}")
    print(f"{'=' * 80}\n")
    print("Figures generated:")
    print("  1. convergence_curves.png    - Section 5.1: Convergence comparison")
    print("  2. cost_boxplots.png          - Section 5.1: Cost distribution")
    print("  3. specialization_heatmap.png - Section 5.2: Agent specialization")
    print("  4. initial_quality.png        - Section 5.2: P0 quality comparison")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for NeuroGen experimental results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize specific experiment
  python -m app.scripts.experiments.visualize_results results/experiments_20251207_120000
  
  # Find and visualize most recent experiment
  python -m app.scripts.experiments.visualize_results
        """,
    )

    parser.add_argument(
        "results_dir",
        nargs="?",
        help="Path to experimental results directory (default: most recent)",
    )

    args = parser.parse_args()

    if args.results_dir is None:
        results_base = Path("results")

        if not results_base.exists():
            print("Error: No results directory found")
            sys.exit(1)

        exp_dirs = sorted(results_base.glob("experiments_*"), reverse=True)

        if not exp_dirs:
            print("Error: No experiment directories found in results/")
            sys.exit(1)

        args.results_dir = str(exp_dirs[0])
        print(f"Using most recent experiment: {args.results_dir}\n")

    visualize_results(args.results_dir)


if __name__ == "__main__":
    main()
