import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_experimental_data(results_dir: Path) -> dict:
    """
    Load experimental results from JSON files.

    Args:
        results_dir (Path): Directory containing experimental results

    Returns:
        dict: Loaded results organized by configuration
    """
    data = {
        "ga_pure": [],
        "drl_junior_ga": [],
        "drl_mid_ga": [],
        "drl_expert_ga": [],
    }

    raw_data_dir = results_dir / "raw_data"

    for config in data.keys():
        config_dir = raw_data_dir / config

        if not config_dir.exists():
            continue

        for json_file in config_dir.glob("*.json"):
            with open(json_file, "r") as f:
                result = json.load(f)
                data[config].append(result)

    return data


def plot_convergence_comparison(
    results_dir: Path,
    instance_name: str,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate convergence plot comparing all configurations.

    Demonstrates the difference in initial population quality (P₀) between
    pure GA and hybrid DRL-GA configurations.

    Args:
        results_dir (Path): Directory containing experimental results
        instance_name (str): Name of the instance to plot
        output_path (Optional[Path]): Path to save the figure (optional)
    """
    data = load_experimental_data(results_dir)

    plt.style.use("seaborn-v0_8-paper")
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

    for config, results in data.items():
        instance_results = [r for r in results if r["instance_name"] == instance_name]

        if not instance_results:
            continue

        histories = [r["convergence_history"] for r in instance_results]
        max_len = max(len(h) for h in histories)

        padded = []
        for h in histories:
            padded_h = h + [h[-1]] * (max_len - len(h))
            padded.append(padded_h)

        avg_history = np.mean(padded, axis=0)
        std_history = np.std(padded, axis=0)

        generations = range(len(avg_history))

        ax.plot(
            generations,
            avg_history,
            label=labels[config],
            color=colors[config],
            linewidth=2,
        )

        ax.fill_between(
            generations,
            avg_history - std_history,
            avg_history + std_history,
            color=colors[config],
            alpha=0.2,
        )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (Total Distance)", fontsize=12)
    ax.set_title(
        f"Convergence Comparison - {instance_name}", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Convergence plot saved to {output_path}")
    else:
        plt.show()


def plot_cost_boxplots(
    results_dir: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate boxplots comparing the distribution of final solution costs.

    Visualizes the quality and consistency of final solutions
    for each configuration.

    Args:
        results_dir (Path): Directory containing experimental results
        output_path (Optional[Path]): Path to save the figure (optional)
    """
    data = load_experimental_data(results_dir)

    plot_data = []

    for config, results in data.items():
        for result in results:
            plot_data.append(
                {
                    "Configuration": config.replace("_", " ").title(),
                    "Final Cost": result["final_cost"],
                }
            )

    df = pd.DataFrame(plot_data)

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="Configuration",
        y="Final Cost",
        palette="Set2",
        ax=ax,
    )

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Final Cost (Total Distance)", fontsize=12)
    ax.set_title("Distribution of Final Solution Costs", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Boxplot saved to {output_path}")
    else:
        plt.show()


def plot_specialization_heatmap(
    results_dir: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate agent specialization heatmap.

    Shows the performance (average cost) of each DRL agent across
    different ranges of instance sizes.

    Args:
        results_dir (Path): Directory containing experimental results
        output_path (Optional[Path]): Path to save the figure (optional)
    """
    data = load_experimental_data(results_dir)

    ranges = {
        "Junior (20-50)": list(range(20, 51)),
        "Mid (60-100)": list(range(60, 101)),
        "Expert (110-150)": list(range(110, 151)),
    }

    agents = ["Junior", "Mid", "Expert"]

    matrix = []

    for agent in agents:
        config_name = f"drl_{agent.lower()}_ga"
        agent_results = data[config_name]

        row = []

        for range_name, size_range in ranges.items():
            range_results = [
                r for r in agent_results if r["instance_size"] in size_range
            ]

            if range_results:
                avg_cost = np.mean([r["final_cost"] for r in range_results])
            else:
                avg_cost = np.nan

            row.append(avg_cost)

        matrix.append(row)

    matrix = np.array(matrix)

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 6))

    matrix_normalized = matrix / matrix.min(axis=0)

    im = ax.imshow(matrix_normalized, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(ranges)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels(ranges.keys(), fontsize=11)
    ax.set_yticklabels([f"DRL-{a}" for a in agents], fontsize=11)

    for i in range(min(len(agents), len(ranges))):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="blue",
            linewidth=3,
        )
        ax.add_patch(rect)

    ax.set_title(
        "Agent Specialization Heatmap\n(Blue boxes = Training range)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Instance Range", fontsize=12)
    ax.set_ylabel("DRL Agent", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Performance\n(Lower is better)", fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Heatmap saved to {output_path}")
    else:
        plt.show()


def plot_improvement_gap_bars(
    results_dir: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Bar plot: Average improvement gap by configuration.

    Shows the difference in initial population quality (P₀).
    A lower gap indicates better initialization (less improvement needed).

    Args:
        results_dir (Path): Directory containing experimental results
        output_path (Optional[Path]): Path to save the figure (optional)
    """
    data = load_experimental_data(results_dir)

    gaps = {}

    for config, results in data.items():
        if results:
            gaps[config] = np.mean([r["improvement_gap"] for r in results])

    configs = list(gaps.keys())
    config_labels = [c.replace("_", " ").title() for c in configs]
    gap_values = [gaps[c] for c in configs]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(config_labels, gap_values, color=colors, alpha=0.8, edgecolor="black")

    for bar, value in zip(bars, gap_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Average Improvement Gap (%)", fontsize=12)
    ax.set_title(
        "Improvement Gap by Configuration\n(Lower = Better Initial Population)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Improvement gap plot saved to {output_path}")
    else:
        plt.show()


def generate_all_visualizations(
    results_dir: str,
    output_dir: str = "visualizations",
) -> None:
    """
    Generates all recommended visualizations for the paper.

    Args:
        results_dir (str): Directory containing experimental results
        output_dir (str): Directory to save the figures
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS FOR PAPER")
    print("=" * 80)
    print(f"Results directory: {results_path}")
    print(f"Output directory: {output_path}\n")

    # 1. Final cost boxplots
    print("Generating cost distribution boxplots...")
    plot_cost_boxplots(results_path, output_path / "cost_boxplots.png")

    # 2. Specialization heatmap
    print("\nGenerating specialization heatmap...")
    plot_specialization_heatmap(
        results_path, output_path / "specialization_heatmap.png"
    )

    # 3. Improvement gap bars
    print("\nGenerating improvement gap comparison...")
    plot_improvement_gap_bars(results_path, output_path / "improvement_gap.png")

    # 4. Convergence for representative instances
    print("\nGenerating convergence plots for sample instances...")

    data = load_experimental_data(results_path)

    # Select one instance from each range
    sample_instances = []
    for config, results in data.items():
        if results:
            # One instance from each range
            for size in [30, 75, 125]:  # Junior, Mid, Expert
                instance_results = [r for r in results if r["instance_size"] == size]
                if instance_results:
                    sample_instances.append(instance_results[0]["instance_name"])
                    break

    sample_instances = list(set(sample_instances))[:3]

    for instance_name in sample_instances:
        safe_name = instance_name.replace("/", "_").replace("\\", "_")
        output_file = output_path / f"convergence_{safe_name}.png"

        print(f"  - {instance_name}")
        plot_convergence_comparison(results_path, instance_name, output_file)

    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nAll figures saved to: {output_path}")
    print("\nGenerated files:")
    print("  - cost_boxplots.png: Distribution of final costs")
    print("  - specialization_heatmap.png: Agent performance by range")
    print("  - improvement_gap.png: Quality of initial population")
    print("  - convergence_*.png: Evolution of fitness over generations")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizations.py <results_directory>")
        print("\nExample:")
        print("  python visualizations.py results/experiments_20251207_120000")
        sys.exit(1)

    results_dir = sys.argv[1]
    generate_all_visualizations(results_dir)
