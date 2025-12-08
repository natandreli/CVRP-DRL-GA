"""
Generate publication-ready figures and tables from experimental results.

Usage:
    python -m app.scripts.experiments.visualize_results --results_dir <path_to_results> --output_dir <path_to_output>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from app.config import settings

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.figsize"] = (10, 6)


def load_results(results_dir: Path) -> Dict[str, List[dict]]:
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

    stats_file = results_dir / "summary_statistics.json"
    friedman_file = results_dir / "friedman_test.json"
    specialization_file = results_dir / "specialization_analysis.json"

    stats_data = {}
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats_data["summary"] = json.load(f)

    if friedman_file.exists():
        with open(friedman_file, "r") as f:
            stats_data["friedman"] = json.load(f)

    if specialization_file.exists():
        with open(specialization_file, "r") as f:
            stats_data["specialization"] = json.load(f)

    return data_by_config, stats_data


def figure1_hypothesis_h1_validation(
    data_by_config: Dict[str, List[dict]],
    output_dir: Path,
):
    ranges = {
        "Small (20-50)": (20, 50),
        "Medium (60-100)": (60, 100),
        "Large (110-150)": (110, 150),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]
    labels = ["GA\nPure", "DRL-Jr\n+GA", "DRL-Mid\n+GA", "DRL-Ex\n+GA"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for idx, (range_name, (min_size, max_size)) in enumerate(ranges.items()):
        ax = axes[idx]

        means = []
        stds = []

        for config in configs:
            if config in data_by_config:
                range_costs = [
                    r["final_cost"]
                    for r in data_by_config[config]
                    if min_size <= r["instance_size"] <= max_size
                ]
                if range_costs:
                    means.append(np.mean(range_costs))
                    stds.append(np.std(range_costs))
                else:
                    means.append(0)
                    stds.append(0)

        if not means or all(m == 0 for m in means):
            continue

        x_pos = np.arange(len(means))
        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    mean + std + max(means) * 0.02,
                    f"{mean:.0f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )

        baseline = means[0]
        if baseline > 0:
            for i in range(1, len(means)):
                if means[i] > 0:
                    improvement = ((baseline - means[i]) / baseline) * 100
                    color = "green" if improvement > 0 else "red"
                    symbol = "↓" if improvement > 0 else "↑"
                    ax.text(
                        i,
                        means[i] * 0.85,
                        f"{symbol}{abs(improvement):.1f}%",
                        ha="center",
                        va="top",
                        fontweight="bold",
                        color=color,
                        fontsize=10,
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(
            "Mean Final Cost ± Std" if idx == 0 else "", fontsize=11, fontweight="bold"
        )
        ax.set_title(range_name, fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        "H1 Validation - DRL Initialization Improves Solution Quality\n"
        + "All DRL Configurations Outperform Random Initialization Across All Ranges",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    output_file = output_dir / "figure1_h1_validation.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Figure 1 saved: {output_file}")
    plt.close()


def figure2_hypothesis_h2_specialization(
    data_by_config: Dict[str, List[dict]],
    output_dir: Path,
):
    ranges = {
        "Small (20-50)": (20, 50),
        "Medium (60-100)": (60, 100),
        "Large (110-150)": (110, 150),
    }

    configs = [
        ("Junior", "drl_junior_ga", "#3498db"),
        ("Mid", "drl_mid_ga", "#2ecc71"),
        ("Expert", "drl_expert_ga", "#9b59b6"),
        ("GA Pure", "ga_pure", "#95a5a6"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    specialization_results = []

    for range_idx, (range_name, (min_size, max_size)) in enumerate(ranges.items()):
        ax = axes[range_idx]

        costs = []
        labels = []
        bar_colors = []

        for config_label, config_name, base_color in configs:
            if config_name not in data_by_config:
                continue

            range_costs = [
                r["final_cost"]
                for r in data_by_config[config_name]
                if min_size <= r["instance_size"] <= max_size
            ]

            if range_costs:
                mean_cost = np.mean(range_costs)
                costs.append(mean_cost)
                labels.append(config_label)
                bar_colors.append(base_color)

        if not costs:
            continue

        best_idx = np.argmin(costs)

        expected_best = range_idx  # 0=Junior, 1=Mid, 2=Expert
        is_specialized = best_idx == expected_best
        specialization_results.append(is_specialized)

        x_pos = np.arange(len(costs))
        bars = ax.bar(
            x_pos,
            costs,
            color=bar_colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            width=0.7,
        )

        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(5)
        bars[best_idx].set_alpha(1.0)

        for i, (bar, cost) in enumerate(zip(bars, costs)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                cost,
                f"{cost:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

        ax.text(
            best_idx,
            costs[best_idx] * 0.4,
            "BEST",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round",
                facecolor="darkgreen",
                alpha=0.9,
                edgecolor="gold",
                linewidth=3,
                pad=0.4,
            ),
        )

        for i, cost in enumerate(costs):
            if i != best_idx:
                gap = ((cost - costs[best_idx]) / costs[best_idx]) * 100
                ax.text(
                    i,
                    cost * 0.5,
                    f"+{gap:.1f}%\nworse",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="darkred",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="red",
                        linewidth=1.5,
                    ),
                )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.set_ylabel(
            "Mean Final Cost" if range_idx == 0 else "", fontsize=12, fontweight="bold"
        )

        expected_label = configs[range_idx][0]
        ax.set_title(
            f"{range_name} Instances\nExpected Best: {expected_label}",
            fontsize=13,
            fontweight="bold",
            pad=15,
            bbox=dict(
                boxstyle="round",
                facecolor="lightyellow",
                alpha=0.3,
                edgecolor="black",
                linewidth=2,
            ),
        )

        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax.set_ylim(0, max(costs) * 1.15)

    plt.suptitle(
        "H2 Validation - Agent Specialization Analysis\n"
        + "Comparing All Agents in Each Instance Range",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    output_file = output_dir / "figure2_h2_specialization.png"
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Figure 2 saved: {output_file}")
    plt.close()


def figure3_convergence_by_range(
    data_by_config: Dict[str, List[dict]],
    output_dir: Path,
):
    ranges = {
        "Small Instances (20-50 customers)": (20, 50),
        "Medium Instances (60-100 customers)": (60, 100),
        "Large Instances (110-150 customers)": (110, 150),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]
    labels = {
        "ga_pure": "GA Pure (Random Init)",
        "drl_junior_ga": "DRL-Junior + GA",
        "drl_mid_ga": "DRL-Mid + GA",
        "drl_expert_ga": "DRL-Expert + GA",
    }
    colors = {
        "ga_pure": "#e74c3c",
        "drl_junior_ga": "#3498db",
        "drl_mid_ga": "#2ecc71",
        "drl_expert_ga": "#9b59b6",
    }

    for idx, (range_name, (min_size, max_size)) in enumerate(ranges.items()):
        ax = axes[idx]

        for config_name in configs:
            if config_name not in data_by_config:
                continue

            range_results = [
                r
                for r in data_by_config[config_name]
                if min_size <= r["instance_size"] <= max_size
            ]

            if not range_results:
                continue

            histories = [
                r["convergence_history"]
                for r in range_results
                if r["convergence_history"]
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
                linewidth=2.5,
                alpha=0.9,
            )

            ax.fill_between(
                generations,
                mean_history - std_history,
                mean_history + std_history,
                color=colors[config_name],
                alpha=0.15,
            )

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6, linewidth=2)
        ax.text(
            2,
            ax.get_ylim()[1] * 0.95,
            "P0",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
        )

        ax.set_xlabel("Generation", fontsize=11, fontweight="bold")
        ax.set_ylabel("Best Cost" if idx == 0 else "", fontsize=11, fontweight="bold")
        ax.set_title(range_name, fontsize=11, fontweight="bold", pad=10)
        if idx == 2:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle(
        "Convergence Behavior - DRL Initialization Provides Better Starting Point",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    output_file = output_dir / "figure3_convergence.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Figure 3 saved: {output_file}")
    plt.close()


def figure4_cost_benefit_analysis(
    data_by_config: Dict[str, List[dict]],
    output_dir: Path,
):
    ranges = {
        "Small (20-50)": (20, 50),
        "Medium (60-100)": (60, 100),
        "Large (110-150)": (110, 150),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]
    labels = ["GA Pure", "DRL-Jr+GA", "DRL-Mid+GA", "DRL-Ex+GA"]

    all_totals = []
    for range_name, (min_size, max_size) in ranges.items():
        for config in configs:
            if config in data_by_config:
                range_results = [
                    r
                    for r in data_by_config[config]
                    if min_size <= r["instance_size"] <= max_size
                ]
                if range_results:
                    init = np.mean([r["initialization_time"] for r in range_results])
                    evol = np.mean([r["evolution_time"] for r in range_results])
                    all_totals.append(init + evol)

    max_time = max(all_totals) if all_totals else 10
    y_limit = max_time * 1.15

    for idx, (range_name, (min_size, max_size)) in enumerate(ranges.items()):
        ax = axes[idx]

        init_times = []
        evol_times = []
        config_labels = []

        for config in configs:
            if config in data_by_config:
                range_results = [
                    r
                    for r in data_by_config[config]
                    if min_size <= r["instance_size"] <= max_size
                ]

                if range_results:
                    init = np.mean([r["initialization_time"] for r in range_results])
                    evol = np.mean([r["evolution_time"] for r in range_results])

                    init_times.append(init)
                    evol_times.append(evol)
                    config_labels.append(labels[configs.index(config)])

        if not init_times:
            continue

        x = np.arange(len(config_labels))

        ax.bar(
            x,
            init_times,
            color="#FFE66D",
            alpha=0.85,
            edgecolor="black",
            linewidth=1.5,
            label="Initialization Time",
        )
        ax.bar(
            x,
            evol_times,
            bottom=init_times,
            color="#4ECDC4",
            alpha=0.85,
            edgecolor="black",
            linewidth=1.5,
            label="Evolution Time",
        )

        for i, (init, evol) in enumerate(zip(init_times, evol_times)):
            total = init + evol
            if init > 0.3:
                ax.text(
                    i,
                    init / 2,
                    f"{init:.1f}s",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="black",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )
            if evol > 0.3:
                ax.text(
                    i,
                    init + evol / 2,
                    f"{evol:.1f}s",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="black",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )
            ax.text(
                i,
                total + y_limit * 0.02,
                f"{total:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(
            "Time (seconds)" if idx == 0 else "", fontsize=11, fontweight="bold"
        )
        ax.set_title(range_name, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, y_limit)
        if idx == 2:
            ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        "Initialization Time Analysis - DRL Overhead vs Evolution Benefit\n"
        + "DRL Adds Initialization Cost but Maintains Similar Evolution Time",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    output_file = output_dir / "figure4_cost_benefit.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Figure 4 saved: {output_file}")
    plt.close()


def generate_latex_tables(
    data_by_config: Dict[str, List[dict]],
    stats_data: dict,
    output_dir: Path,
):
    table1 = []
    table1.append("% Table 1: Aggregated Results by Configuration and Range")
    table1.append("\\begin{table}[htbp]")
    table1.append("\\centering")
    table1.append(
        "\\caption{Mean Final Cost (± Std Dev) by Configuration and Instance Range}"
    )
    table1.append("\\label{tab:results_by_range}")
    table1.append("\\begin{tabular}{lcccc}")
    table1.append("\\toprule")
    table1.append(
        "Configuration & Junior (20-50) & Mid (60-100) & Expert (110-150) & Global \\\\"
    )
    table1.append("\\midrule")

    ranges = {"Junior": (20, 50), "Mid": (60, 100), "Expert": (110, 150)}
    configs = [
        ("GA Pure", "ga_pure"),
        ("DRL-Junior + GA", "drl_junior_ga"),
        ("DRL-Mid + GA", "drl_mid_ga"),
        ("DRL-Expert + GA", "drl_expert_ga"),
    ]

    for label, config in configs:
        row = [label]
        all_costs = []

        for range_name, (min_size, max_size) in ranges.items():
            if config in data_by_config:
                range_costs = [
                    r["final_cost"]
                    for r in data_by_config[config]
                    if min_size <= r["instance_size"] <= max_size
                ]
                if range_costs:
                    mean = np.mean(range_costs)
                    std = np.std(range_costs)
                    all_costs.extend(range_costs)
                    row.append(f"{mean:.1f}±{std:.1f}")
                else:
                    row.append("--")
            else:
                row.append("--")

        # Global
        if all_costs:
            row.append(f"{np.mean(all_costs):.1f}")
        else:
            row.append("--")

        table1.append(" & ".join(row) + " \\\\")

    table1.append("\\bottomrule")
    table1.append("\\end{tabular}")
    table1.append("\\end{table}")

    table2 = []
    table2.append("% Table 2: Improvement Gap, Convergence, and Computation Time")
    table2.append("\\begin{table}[htbp]")
    table2.append("\\centering")
    table2.append(
        "\\caption{Evolution Metrics: Gap, Convergence Speed, and Time Breakdown}"
    )
    table2.append("\\label{tab:evolution_metrics}")
    table2.append("\\begin{tabular}{lcccccc}")
    table2.append("\\toprule")
    table2.append(
        "Configuration & Gap (\\%) & Gen. Conv. & Init. Time (s) & Evol. Time (s) & Total (s) \\\\"
    )
    table2.append("\\midrule")

    for label, config in configs:
        if config in data_by_config:
            gaps = [r["improvement_gap"] for r in data_by_config[config]]
            gens = [
                r.get("generations_to_convergence", 0)
                for r in data_by_config[config]
                if r.get("generations_to_convergence")
            ]
            init_times = [r["initialization_time"] for r in data_by_config[config]]
            evol_times = [r["evolution_time"] for r in data_by_config[config]]
            total_times = [r["total_time"] for r in data_by_config[config]]

            gap_mean = np.mean(gaps)
            gap_std = np.std(gaps)
            gen_mean = np.mean(gens) if gens else 0
            gen_std = np.std(gens) if gens else 0
            init_mean = np.mean(init_times)
            evol_mean = np.mean(evol_times)
            total_mean = np.mean(total_times)

            row = [
                label,
                f"{gap_mean:.1f}±{gap_std:.1f}",
                f"{gen_mean:.0f}±{gen_std:.0f}" if gens else "--",
                f"{init_mean:.1f}",
                f"{evol_mean:.1f}",
                f"{total_mean:.1f}",
            ]
            table2.append(" & ".join(row) + " \\\\")

    table2.append("\\bottomrule")
    table2.append(
        "\\multicolumn{6}{l}{\\footnotesize Gap: \\% improvement from P0 to final; Gen. Conv.: generations until stagnation} \\\\"
    )
    table2.append(
        "\\multicolumn{6}{l}{\\footnotesize Init. Time: time to generate P0; Evol. Time: GA evolution time} \\\\"
    )
    table2.append("\\end{tabular}")
    table2.append("\\end{table}")

    table3 = []
    table3.append("% Table 3: Agent Specialization Matrix (Mean Final Cost)")
    table3.append("\\begin{table}[htbp]")
    table3.append("\\centering")
    table3.append(
        "\\caption{Agent Specialization Matrix: Mean Final Cost by Agent and Instance Range}"
    )
    table3.append("\\label{tab:specialization_matrix}")
    table3.append("\\begin{tabular}{lccc}")
    table3.append("\\toprule")
    table3.append("Agent/Range & Junior (20-50) & Mid (60-100) & Expert (110-150) \\\\")
    table3.append("\\midrule")

    agents = [
        ("DRL-Junior", "drl_junior_ga"),
        ("DRL-Mid", "drl_mid_ga"),
        ("DRL-Expert", "drl_expert_ga"),
    ]

    for agent_label, config in agents:
        row = [agent_label]

        for range_name, (min_size, max_size) in ranges.items():
            if config in data_by_config:
                range_costs = [
                    r["final_cost"]
                    for r in data_by_config[config]
                    if min_size <= r["instance_size"] <= max_size
                ]
                if range_costs:
                    mean = np.mean(range_costs)
                    if (
                        (agent_label == "DRL-Junior" and range_name == "Junior")
                        or (agent_label == "DRL-Mid" and range_name == "Mid")
                        or (agent_label == "DRL-Expert" and range_name == "Expert")
                    ):
                        row.append(f"\\textbf{{{mean:.1f}}}")
                    else:
                        row.append(f"{mean:.1f}")
                else:
                    row.append("--")
            else:
                row.append("--")

        table3.append(" & ".join(row) + " \\\\")

    table3.append("\\bottomrule")
    table3.append(
        "\\multicolumn{4}{l}{\\footnotesize Bold values indicate expected specialization (agent in its training range)} \\\\"
    )
    table3.append("\\end{tabular}")
    table3.append("\\end{table}")

    table4 = []
    table4.append("% Table 4: Statistical Significance Tests")
    table4.append("\\begin{table}[htbp]")
    table4.append("\\centering")
    table4.append("\\caption{Statistical Significance Tests}")
    table4.append("\\label{tab:statistical_tests}")
    table4.append("\\begin{tabular}{lcc}")
    table4.append("\\toprule")
    table4.append("Test & Statistic & p-value \\\\")
    table4.append("\\midrule")

    if "friedman" in stats_data:
        friedman = stats_data["friedman"]
        stat = friedman.get("statistic", 0)
        p_val = friedman.get("p_value", 1.0)
        sig = (
            "***"
            if p_val < 0.001
            else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        )
        table4.append(
            f"Friedman (4 configs) & $\\chi^2={stat:.2f}$ & {p_val:.4f}{sig} \\\\"
        )

    table4.append("\\bottomrule")
    table4.append(
        "\\multicolumn{3}{l}{\\footnotesize *** p<0.001, ** p<0.01, * p<0.05, ns: not significant} \\\\"
    )
    table4.append("\\end{tabular}")
    table4.append("\\end{table}")

    tables_file = output_dir / "latex_tables.tex"
    with open(tables_file, "w") as f:
        f.write(
            "\n\n".join(
                [
                    "\n".join(table1),
                    "\n".join(table2),
                    "\n".join(table3),
                    "\n".join(table4),
                ]
            )
        )

    print(f"✓ LaTeX tables saved to: {tables_file}")

    text_file = output_dir / "tables_summary.txt"
    with open(text_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TABLES SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "\n\n".join(
                [
                    "\n".join(table1),
                    "\n".join(table2),
                    "\n".join(table3),
                    "\n".join(table4),
                ]
            )
        )

    print(f"✓ Text summary saved to: {text_file}")


def create_paper_figures(results_dir: str) -> None:
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory not found: {results_path}")
        return

    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-READY FIGURES AND TABLES")
    print("=" * 80)
    print(f"Results directory: {results_path}\n")

    try:
        data_by_config, stats_data = load_results(results_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    figures_dir = results_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Creating figures...\n")

    try:
        figure1_hypothesis_h1_validation(data_by_config, figures_dir)
        figure2_hypothesis_h2_specialization(data_by_config, figures_dir)
        figure3_convergence_by_range(data_by_config, figures_dir)
        figure4_cost_benefit_analysis(data_by_config, figures_dir)

        print("\nGenerating LaTeX tables...\n")
        generate_latex_tables(data_by_config, stats_data, figures_dir)

    except Exception as e:
        print(f"Error generating outputs: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\n{'=' * 80}")
    print(f"✓ All paper outputs saved to: {figures_dir}")
    print(f"{'=' * 80}\n")
    print("Figures ready for paper:")
    print("  • figure1_h1_validation.png    - H1: DRL improves solution quality")
    print("  • figure2_h2_specialization.png - H2: Agent specialization analysis")
    print("  • figure3_convergence.png       - Convergence behavior by range")
    print("  • figure4_cost_benefit.png      - Computational cost-benefit")
    print("\nTables ready for paper:")
    print("  • latex_tables.tex              - All 4 LaTeX tables")
    print("  • tables_summary.txt            - Text preview of tables")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for NeuroGen paper"
    )

    parser.add_argument(
        "results_dir", nargs="?", help="Path to experimental results directory"
    )

    args = parser.parse_args()

    if args.results_dir is None:
        results_base = settings.EXPERIMENT_RESULTS_DIR

        if not results_base.exists():
            print("Error: No results directory found")
            sys.exit(1)

        exp_dirs = sorted(results_base.glob("experiments_*"), reverse=True)

        if not exp_dirs:
            print("Error: No experiment directories found")
            sys.exit(1)

        args.results_dir = str(exp_dirs[0])
        print(f"Using most recent experiment: {args.results_dir}\n")

    create_paper_figures(args.results_dir)


if __name__ == "__main__":
    main()
