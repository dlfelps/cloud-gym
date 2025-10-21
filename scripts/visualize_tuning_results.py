#!/usr/bin/env python3
"""Visualize hyperparameter tuning results."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_comparison_bars(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing all variants."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Separate A2C and DQN
    a2c_df = df[df['algorithm'] == 'a2c'].copy()
    dqn_df = df[df['algorithm'] == 'dqn'].copy()

    # Extract variant names (remove algorithm prefix)
    a2c_df['variant'] = a2c_df['variant_name'].str.replace('a2c_', '')
    dqn_df['variant'] = dqn_df['variant_name'].str.replace('dqn_', '')

    # Sort by completion rate
    a2c_df = a2c_df.sort_values('mean_completion_rate', ascending=False)
    dqn_df = dqn_df.sort_values('mean_completion_rate', ascending=False)

    # Plot 1: A2C Completion Rate
    ax = axes[0, 0]
    bars = ax.barh(a2c_df['variant'], a2c_df['mean_completion_rate'] * 100)
    # Color baseline differently
    for i, variant in enumerate(a2c_df['variant']):
        if 'baseline' in variant:
            bars[i].set_color('#FF6B6B')
        else:
            bars[i].set_color('#4ECDC4')
    ax.set_xlabel('Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('A2C: Completion Rate by Variant', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (variant, val) in enumerate(zip(a2c_df['variant'], a2c_df['mean_completion_rate'])):
        ax.text(val * 100 + 1, i, f'{val*100:.1f}%', va='center', fontsize=9)

    # Plot 2: DQN Completion Rate
    ax = axes[0, 1]
    bars = ax.barh(dqn_df['variant'], dqn_df['mean_completion_rate'] * 100)
    for i, variant in enumerate(dqn_df['variant']):
        if 'baseline' in variant:
            bars[i].set_color('#FF6B6B')
        else:
            bars[i].set_color('#95E1D3')
    ax.set_xlabel('Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('DQN: Completion Rate by Variant', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (variant, val) in enumerate(zip(dqn_df['variant'], dqn_df['mean_completion_rate'])):
        ax.text(val * 100 + 1, i, f'{val*100:.1f}%', va='center', fontsize=9)

    # Plot 3: A2C Mean Reward
    ax = axes[1, 0]
    a2c_sorted_reward = a2c_df.sort_values('mean_reward', ascending=False)
    bars = ax.barh(a2c_sorted_reward['variant'], a2c_sorted_reward['mean_reward'])
    for i, variant in enumerate(a2c_sorted_reward['variant']):
        if 'baseline' in variant:
            bars[i].set_color('#FF6B6B')
        else:
            bars[i].set_color('#4ECDC4')
    ax.set_xlabel('Mean Reward', fontsize=11, fontweight='bold')
    ax.set_title('A2C: Mean Reward by Variant', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Plot 4: DQN Mean Reward
    ax = axes[1, 1]
    dqn_sorted_reward = dqn_df.sort_values('mean_reward', ascending=False)
    bars = ax.barh(dqn_sorted_reward['variant'], dqn_sorted_reward['mean_reward'])
    for i, variant in enumerate(dqn_sorted_reward['variant']):
        if 'baseline' in variant:
            bars[i].set_color('#FF6B6B')
        else:
            bars[i].set_color('#95E1D3')
    ax.set_xlabel('Mean Reward', fontsize=11, fontweight='bold')
    ax.set_title('DQN: Mean Reward by Variant', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.suptitle('Hyperparameter Tuning Results Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'tuning_comparison_bars.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'tuning_comparison_bars.png'}")
    plt.close()


def plot_metrics_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of normalized metrics for all variants."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (algo, algo_name) in zip(axes, [('a2c', 'A2C'), ('dqn', 'DQN')]):
        algo_df = df[df['algorithm'] == algo].copy()
        algo_df['variant'] = algo_df['variant_name'].str.replace(f'{algo}_', '')

        # Select metrics
        metrics = ['mean_completion_rate', 'mean_overall_sla_success', 'mean_reward', 'mean_total_cost']
        metric_labels = ['Completion\nRate', 'Overall\nSLA', 'Mean\nReward', 'Total\nCost']

        # Create matrix
        data = algo_df[metrics].values

        # Normalize each column to [0, 1] (higher is better)
        normalized_data = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            col = data[:, i]
            if i == 3:  # Cost - lower is better
                normalized_data[:, i] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-8)
            else:  # Others - higher is better
                normalized_data[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)

        # Plot heatmap
        im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_yticks(np.arange(len(algo_df)))
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_yticklabels(algo_df['variant'], fontsize=9)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Score', rotation=270, labelpad=20, fontsize=10)

        # Add values in cells
        for i in range(len(algo_df)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{normalized_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title(f'{algo_name} Variants - Normalized Metrics', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'tuning_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'tuning_heatmap.png'}")
    plt.close()


def plot_scatter_matrix(df: pd.DataFrame, output_dir: Path):
    """Scatter plot matrix showing relationships between metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Completion rate vs Reward
    ax = axes[0, 0]
    for algo, color, marker in [('a2c', '#4ECDC4', 'o'), ('dqn', '#95E1D3', 's')]:
        algo_df = df[df['algorithm'] == algo]
        ax.scatter(algo_df['mean_completion_rate'] * 100,
                  algo_df['mean_reward'],
                  c=color, marker=marker, s=100, alpha=0.7,
                  label=algo.upper(), edgecolors='black', linewidth=0.5)
        # Annotate baseline
        baseline = algo_df[algo_df['variant_name'].str.contains('baseline')]
        if not baseline.empty:
            ax.scatter(baseline['mean_completion_rate'] * 100,
                      baseline['mean_reward'],
                      marker='*', s=300, c='red', edgecolors='black', linewidth=1.5,
                      zorder=10)
    ax.set_xlabel('Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
    ax.set_title('Completion Rate vs Reward', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Completion rate vs SLA
    ax = axes[0, 1]
    for algo, color, marker in [('a2c', '#4ECDC4', 'o'), ('dqn', '#95E1D3', 's')]:
        algo_df = df[df['algorithm'] == algo]
        ax.scatter(algo_df['mean_completion_rate'] * 100,
                  algo_df['mean_overall_sla_success'] * 100,
                  c=color, marker=marker, s=100, alpha=0.7,
                  label=algo.upper(), edgecolors='black', linewidth=0.5)
        baseline = algo_df[algo_df['variant_name'].str.contains('baseline')]
        if not baseline.empty:
            ax.scatter(baseline['mean_completion_rate'] * 100,
                      baseline['mean_overall_sla_success'] * 100,
                      marker='*', s=300, c='red', edgecolors='black', linewidth=1.5,
                      zorder=10)
    ax.set_xlabel('Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Overall SLA Success (%)', fontsize=11, fontweight='bold')
    ax.set_title('Completion Rate vs SLA Success', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Completion rate vs Cost
    ax = axes[1, 0]
    for algo, color, marker in [('a2c', '#4ECDC4', 'o'), ('dqn', '#95E1D3', 's')]:
        algo_df = df[df['algorithm'] == algo]
        ax.scatter(algo_df['mean_completion_rate'] * 100,
                  algo_df['mean_total_cost'],
                  c=color, marker=marker, s=100, alpha=0.7,
                  label=algo.upper(), edgecolors='black', linewidth=0.5)
        baseline = algo_df[algo_df['variant_name'].str.contains('baseline')]
        if not baseline.empty:
            ax.scatter(baseline['mean_completion_rate'] * 100,
                      baseline['mean_total_cost'],
                      marker='*', s=300, c='red', edgecolors='black', linewidth=1.5,
                      zorder=10)
    ax.set_xlabel('Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax.set_title('Completion Rate vs Cost', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Reward distribution
    ax = axes[1, 1]
    a2c_rewards = df[df['algorithm'] == 'a2c']['mean_reward']
    dqn_rewards = df[df['algorithm'] == 'dqn']['mean_reward']

    parts = ax.violinplot([a2c_rewards, dqn_rewards], positions=[1, 2],
                          widths=0.6, showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['A2C', 'DQN'], fontsize=11)
    ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
    ax.set_title('Reward Distribution by Algorithm', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Hyperparameter Tuning - Metric Relationships', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'tuning_scatter_matrix.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'tuning_scatter_matrix.png'}")
    plt.close()


def plot_improvement_over_baseline(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing improvement over baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (algo, algo_name) in zip(axes, [('a2c', 'A2C'), ('dqn', 'DQN')]):
        algo_df = df[df['algorithm'] == algo].copy()
        algo_df['variant'] = algo_df['variant_name'].str.replace(f'{algo}_', '')

        # Get baseline performance
        baseline = algo_df[algo_df['variant'] == 'baseline']
        if baseline.empty:
            print(f"Warning: No baseline found for {algo_name}")
            continue

        baseline_completion = baseline['mean_completion_rate'].iloc[0]

        # Calculate improvement
        algo_df['improvement'] = ((algo_df['mean_completion_rate'] - baseline_completion)
                                 / baseline_completion * 100)

        # Remove baseline from plot
        algo_df = algo_df[algo_df['variant'] != 'baseline']

        # Sort by improvement
        algo_df = algo_df.sort_values('improvement', ascending=True)

        # Plot
        colors = ['#4ECDC4' if x >= 0 else '#FF6B6B' for x in algo_df['improvement']]
        bars = ax.barh(algo_df['variant'], algo_df['improvement'], color=colors)

        ax.set_xlabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{algo_name}: Completion Rate Improvement', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (variant, val) in enumerate(zip(algo_df['variant'], algo_df['improvement'])):
            ax.text(val + (2 if val >= 0 else -2), i,
                   f'{val:+.1f}%', va='center',
                   ha='left' if val >= 0 else 'right',
                   fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'tuning_improvement.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'tuning_improvement.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter tuning results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results/hyperparameter_tuning_results.csv",
        help="Path to tuning results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} variants from {results_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_comparison_bars(df, output_dir)
    plot_metrics_heatmap(df, output_dir)
    plot_scatter_matrix(df, output_dir)
    plot_improvement_over_baseline(df, output_dir)

    print("\n[SUCCESS] All visualizations generated!")
    print(f"   Location: {output_dir.resolve()}")
    print("\nGenerated plots:")
    print("  - tuning_comparison_bars.png - Completion rate and reward by variant")
    print("  - tuning_heatmap.png - Normalized metrics heatmap")
    print("  - tuning_scatter_matrix.png - Metric relationships")
    print("  - tuning_improvement.png - Improvement over baseline")


if __name__ == "__main__":
    main()
