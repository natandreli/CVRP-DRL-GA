import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """
    Performs comprehensive statistical analysis on experimental results.

    Key analyses:
    1. Friedman test: Overall differences between configurations
    2. Nemenyi post-hoc: Pairwise comparisons
    3. Specialization analysis: Agent performance within/outside training range
    """

    def __init__(self, results_dir: Path) -> None:
        """
        Initialize statistical analyzer.

        Args:
            results_dir (Path): Directory containing experimental results
        """
        self.results_dir = results_dir
        self.raw_data_dir = results_dir / "raw_data"

        # Data structures
        self.data_by_config: dict[str, list[dict]] = defaultdict(list)
        self.data_by_instance: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self._load_data()

    def _load_data(self) -> None:
        """Load all experimental results from JSON files."""
        for config_dir in self.raw_data_dir.iterdir():
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name

            for json_file in config_dir.glob("*.json"):
                with open(json_file, "r") as f:
                    result = json.load(f)

                self.data_by_config[config_name].append(result)

                # Index by instance for Friedman test
                instance_name = result["instance_name"]
                self.data_by_instance[instance_name][config_name].append(
                    result["final_cost"]
                )

    def friedman_test(self) -> dict:
        """
        Perform Friedman test across all configurations.

        Tests null hypothesis: No significant difference between configurations
        across all instances.

        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 80)
        print("FRIEDMAN TEST - Overall Configuration Comparison")
        print("=" * 80)

        # Prepare data: matrix of [instances × configurations]
        configurations = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]

        data_matrix = []
        instance_names = []

        for instance_name in sorted(self.data_by_instance.keys()):
            # Get average cost for each configuration on this instance
            row = []

            for config in configurations:
                costs = self.data_by_instance[instance_name].get(config, [])

                if not costs:
                    print(f"Warning: Missing data for {instance_name}, {config}")
                    continue

                row.append(np.mean(costs))

            if len(row) == len(configurations):
                data_matrix.append(row)
                instance_names.append(instance_name)

        data_matrix = np.array(data_matrix)

        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*data_matrix.T)

        # Calculate effect size (Kendall's W)
        n_instances = data_matrix.shape[0]
        k_configs = data_matrix.shape[1]

        # Rank configurations for each instance
        ranks = np.apply_along_axis(stats.rankdata, 1, data_matrix)
        rank_sums = ranks.sum(axis=0)

        # Kendall's W
        mean_rank_sum = rank_sums.mean()
        ss_ranks = ((rank_sums - mean_rank_sum) ** 2).sum()
        kendalls_w = (12 * ss_ranks) / (k_configs**2 * (n_instances**3 - n_instances))

        # Average ranks
        avg_ranks = {
            config: rank_sums[i] / n_instances
            for i, config in enumerate(configurations)
        }

        result = {
            "test": "Friedman",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significance_level": 0.05,
            "is_significant": p_value < 0.05,
            "kendalls_w": float(kendalls_w),
            "n_instances": n_instances,
            "configurations": configurations,
            "average_ranks": avg_ranks,
            "interpretation": (
                "Significant differences detected between configurations"
                if p_value < 0.05
                else "No significant differences detected"
            ),
        }

        # Print results
        print(f"\nInstances analyzed: {n_instances}")
        print(f"Configurations: {len(configurations)}")
        print(f"\nFriedman statistic: {statistic:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Kendall's W (effect size): {kendalls_w:.4f}")
        print("\nAverage ranks (lower is better):")
        for config, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
            print(f"  {config:20s}: {rank:.2f}")

        print(f"\n{result['interpretation']}")

        return result

    def nemenyi_test(self, alpha: float = 0.05) -> dict:
        """
        Perform Nemenyi post-hoc test for pairwise comparisons.

        Args:
            alpha (float): Significance level

        Returns:
            Dictionary with pairwise comparison results
        """
        print("\n" + "=" * 80)
        print("NEMENYI POST-HOC TEST - Pairwise Comparisons")
        print("=" * 80)

        configurations = ["ga_pure", "drl_junior_ga", "drl_mid_ga", "drl_expert_ga"]

        # Build data matrix
        data_matrix = []

        for instance_name in sorted(self.data_by_instance.keys()):
            row = []

            for config in configurations:
                costs = self.data_by_instance[instance_name].get(config, [])
                if costs:
                    row.append(np.mean(costs))

            if len(row) == len(configurations):
                data_matrix.append(row)

        data_matrix = np.array(data_matrix)
        n_instances = data_matrix.shape[0]
        k_configs = data_matrix.shape[1]

        # Compute ranks
        ranks = np.apply_along_axis(stats.rankdata, 1, data_matrix)
        avg_ranks = ranks.mean(axis=0)

        # Critical distance for Nemenyi test
        q_alpha = 2.569  # Critical value for k=4, alpha=0.05 (from table)
        cd = q_alpha * np.sqrt((k_configs * (k_configs + 1)) / (6 * n_instances))

        # Pairwise comparisons
        comparisons = []

        for i in range(k_configs):
            for j in range(i + 1, k_configs):
                rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                is_significant = rank_diff > cd

                comparisons.append(
                    {
                        "config_1": configurations[i],
                        "config_2": configurations[j],
                        "rank_diff": float(rank_diff),
                        "critical_distance": float(cd),
                        "is_significant": is_significant,
                        "better_config": (
                            configurations[i]
                            if avg_ranks[i] < avg_ranks[j]
                            else configurations[j]
                        ),
                    }
                )

        result = {
            "test": "Nemenyi",
            "alpha": alpha,
            "n_instances": n_instances,
            "k_configurations": k_configs,
            "critical_distance": float(cd),
            "average_ranks": {
                config: float(avg_ranks[i]) for i, config in enumerate(configurations)
            },
            "pairwise_comparisons": comparisons,
        }

        # Print results
        print(f"\nCritical distance (CD): {cd:.4f}")
        print("Rank differences > CD are statistically significant\n")

        print("Pairwise comparisons:")
        for comp in comparisons:
            sig_marker = "***" if comp["is_significant"] else "   "
            print(
                f"  {comp['config_1']:20s} vs {comp['config_2']:20s}: "
                f"Δrank = {comp['rank_diff']:.4f} {sig_marker}"
            )
            if comp["is_significant"]:
                print(f"    → {comp['better_config']} is significantly better")

        return result

    def specialization_analysis(self) -> dict:
        """
        Analyze agent specialization (Hypothesis H2).

        Compares performance of each DRL agent within vs. outside their
        training range.

        Returns:
            Dictionary with specialization metrics
        """
        print("\n" + "=" * 80)
        print("SPECIALIZATION ANALYSIS - Agent Performance by Range")
        print("=" * 80)

        # Define ranges
        ranges = {
            "junior": list(range(20, 51)),  # 20-50
            "mid": list(range(60, 101)),  # 60-100
            "expert": list(range(110, 151)),  # 110-150
        }

        # Collect data by agent and range
        analysis = {}

        for agent_name in ["junior", "mid", "expert"]:
            config_name = f"drl_{agent_name}_ga"
            agent_data = self.data_by_config[config_name]

            # Group by range
            in_range_costs = []
            out_range_costs = []

            for result in agent_data:
                instance_size = result["instance_size"]
                cost = result["final_cost"]

                if instance_size in ranges[agent_name]:
                    in_range_costs.append(cost)
                else:
                    out_range_costs.append(cost)

            # Statistics
            in_range_mean = np.mean(in_range_costs) if in_range_costs else 0
            out_range_mean = np.mean(out_range_costs) if out_range_costs else 0

            # Two-sample t-test
            if in_range_costs and out_range_costs:
                t_stat, p_value = stats.ttest_ind(in_range_costs, out_range_costs)
            else:
                t_stat, p_value = 0, 1.0

            analysis[agent_name] = {
                "training_range": f"{min(ranges[agent_name])}-{max(ranges[agent_name])}",
                "in_range_performance": {
                    "mean_cost": float(in_range_mean),
                    "std_cost": float(np.std(in_range_costs)) if in_range_costs else 0,
                    "n_instances": len(in_range_costs),
                },
                "out_range_performance": {
                    "mean_cost": float(out_range_mean),
                    "std_cost": float(np.std(out_range_costs))
                    if out_range_costs
                    else 0,
                    "n_instances": len(out_range_costs),
                },
                "relative_performance": float(
                    (out_range_mean - in_range_mean) / in_range_mean * 100
                    if in_range_mean > 0
                    else 0
                ),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_specialized": p_value < 0.05 and in_range_mean < out_range_mean,
            }

        # Print results
        for agent_name, data in analysis.items():
            print(f"\n{agent_name.upper()} Agent:")
            print(f"  Training range: {data['training_range']} customers")
            print(
                f"  In-range mean cost: {data['in_range_performance']['mean_cost']:.2f}"
            )
            print(
                f"  Out-range mean cost: {data['out_range_performance']['mean_cost']:.2f}"
            )
            print(f"  Performance degradation: {data['relative_performance']:.2f}%")
            print(f"  p-value: {data['p_value']:.6f}")
            print(f"  Specialized: {'Yes' if data['is_specialized'] else 'No'}")

        return {
            "agents": analysis,
            "hypothesis_h2": all(a["is_specialized"] for a in analysis.values()),
        }

    def generate_summary_statistics(self) -> dict:
        """
        Generate comprehensive summary statistics for all configurations.

        Returns:
            Dictionary with aggregated metrics
        """
        summary = {}

        for config_name, results in self.data_by_config.items():
            if not results:
                continue

            costs = [r["final_cost"] for r in results]
            gaps = [r["improvement_gap"] for r in results]
            times = [r["total_time"] for r in results]

            summary[config_name] = {
                "n_runs": len(results),
                "cost": {
                    "mean": float(np.mean(costs)),
                    "std": float(np.std(costs)),
                    "min": float(np.min(costs)),
                    "max": float(np.max(costs)),
                    "median": float(np.median(costs)),
                },
                "improvement_gap": {
                    "mean": float(np.mean(gaps)),
                    "std": float(np.std(gaps)),
                },
                "computation_time": {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                },
            }

        return summary

    def run_full_analysis(self) -> None:
        """
        Run complete statistical analysis pipeline and save results.
        """
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS - NeuroGen Experimental Validation")
        print("=" * 80)

        # Generate summary statistics
        summary = self.generate_summary_statistics()
        summary_path = self.results_dir / "summary_statistics.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Summary statistics saved to: {summary_path}")

        # Friedman test
        friedman_result = self.friedman_test()
        friedman_path = self.results_dir / "friedman_test.json"

        with open(friedman_path, "w") as f:
            json.dump(friedman_result, f, indent=2)

        print(f"✓ Friedman test results saved to: {friedman_path}")

        # Nemenyi test (only if Friedman is significant)
        if friedman_result["is_significant"]:
            nemenyi_result = self.nemenyi_test()
            nemenyi_path = self.results_dir / "nemenyi_test.json"

            with open(nemenyi_path, "w") as f:
                json.dump(nemenyi_result, f, indent=2)

            print(f"✓ Nemenyi test results saved to: {nemenyi_path}")
        else:
            print("\nSkipping Nemenyi test (Friedman not significant)")

        # Specialization analysis
        specialization_result = self.specialization_analysis()
        specialization_path = self.results_dir / "specialization_analysis.json"

        with open(specialization_path, "w") as f:
            json.dump(specialization_result, f, indent=2)

        print(f"✓ Specialization analysis saved to: {specialization_path}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
