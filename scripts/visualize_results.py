"""Visualization script for baseline comparison results."""

import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from CSV or JSON file."""
    path = Path(results_path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create comprehensive comparison plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Mean Reward Comparison (Bar Plot with Error Bars)
    fig, ax = plt.subplots(figsize=(12, 6))
    policies = df['policy_name']
    mean_rewards = df['mean_reward']
    std_rewards = df['std_reward']

    x_pos = np.arange(len(policies))
    colors = sns.color_palette("husl", len(policies))

    bars = ax.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, color=colors, alpha=0.8)
    ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Mean Episode Reward by Policy', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, mean_rewards, std_rewards)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'reward_comparison.png'}")

    # 2. Multi-Metric Comparison (Grouped Bar Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ('mean_completion_rate', 'Task Completion Rate', axes[0, 0]),
        ('mean_sla_satisfaction_rate', 'SLA Satisfaction Rate', axes[0, 1]),
        ('mean_rejected_tasks', 'Mean Rejected Tasks', axes[1, 0]),
        ('mean_total_cost', 'Mean Total Cost', axes[1, 1]),
    ]

    for metric_name, title, ax in metrics:
        x_pos = np.arange(len(policies))
        values = df[metric_name]

        bars = ax.bar(x_pos, values, color=colors, alpha=0.8)
        ax.set_xlabel('Policy', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(policies, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'metrics_comparison.png'}")

    # 3. Radar Chart for Multi-Objective Performance
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    # Normalize metrics to [0, 1] range
    normalized_df = df.copy()
    metrics_to_plot = [
        'mean_completion_rate',
        'mean_sla_satisfaction_rate',
        'mean_reward',
    ]

    # Normalize mean_reward
    reward_min = df['mean_reward'].min()
    reward_max = df['mean_reward'].max()
    normalized_df['mean_reward'] = (df['mean_reward'] - reward_min) / (reward_max - reward_min + 1e-8)

    # Set up angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Plot each policy
    for i, (_, row) in enumerate(normalized_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=row['policy_name'], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Completion\nRate', 'SLA\nSatisfaction', 'Reward'], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title('Multi-Objective Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'radar_comparison.png'}")

    # 4. Box Plot for Reward Distribution (if we have episode-level data)
    # Note: This would require storing individual episode rewards, not just means
    # For now, we'll create a simplified version showing mean ± std

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create pseudo box plot data from mean and std
    box_data = []
    for _, row in df.iterrows():
        # Generate synthetic distribution from mean and std (Gaussian approximation)
        synthetic_samples = np.random.normal(
            row['mean_reward'],
            row['std_reward'],
            size=100
        )
        box_data.append(synthetic_samples)

    bp = ax.boxplot(box_data, labels=policies, patch_artist=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward Distribution', fontsize=12, fontweight='bold')
    ax.set_title('Reward Distribution Comparison (Synthetic)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'reward_distribution.png'}")

    # 5. Heatmap of Normalized Performance
    fig, ax = plt.subplots(figsize=(12, 8))

    # Select key metrics for heatmap
    heatmap_metrics = [
        'mean_reward',
        'mean_completion_rate',
        'mean_sla_satisfaction_rate',
        'mean_rejected_tasks',
        'mean_sla_violations',
        'mean_total_cost',
    ]

    heatmap_data = df[heatmap_metrics].copy()

    # Normalize each metric to [0, 1]
    for col in heatmap_data.columns:
        col_min = heatmap_data[col].min()
        col_max = heatmap_data[col].max()
        if col_max - col_min > 0:
            heatmap_data[col] = (heatmap_data[col] - col_min) / (col_max - col_min)

    # For "bad" metrics (rejections, violations, cost), invert normalization
    bad_metrics = ['mean_rejected_tasks', 'mean_sla_violations', 'mean_total_cost']
    for col in bad_metrics:
        if col in heatmap_data.columns:
            heatmap_data[col] = 1 - heatmap_data[col]

    # Set policy names as index
    heatmap_data.index = df['policy_name']

    # Rename columns for better display
    heatmap_data.columns = [
        'Reward',
        'Completion Rate',
        'SLA Satisfaction',
        'Task Acceptance',
        'SLA Compliance',
        'Cost Efficiency',
    ]

    # Create heatmap
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized Performance (0-1)'},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title('Normalized Performance Heatmap\n(Higher is Better for All Metrics)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'performance_heatmap.png'}")

    # 6. Summary Table (saved as image)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Select key metrics for table
    table_df = df[[
        'policy_name',
        'mean_reward',
        'mean_completion_rate',
        'mean_sla_satisfaction_rate',
        'mean_rejected_tasks',
        'mean_total_cost'
    ]].copy()

    # Round values
    table_df['mean_reward'] = table_df['mean_reward'].round(2)
    table_df['mean_completion_rate'] = table_df['mean_completion_rate'].round(3)
    table_df['mean_sla_satisfaction_rate'] = table_df['mean_sla_satisfaction_rate'].round(3)
    table_df['mean_rejected_tasks'] = table_df['mean_rejected_tasks'].round(1)
    table_df['mean_total_cost'] = table_df['mean_total_cost'].round(2)

    # Rename columns
    table_df.columns = ['Policy', 'Reward', 'Completion Rate', 'SLA Satisfaction', 'Rejected Tasks', 'Total Cost']

    # Create table
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(table_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Color rows alternately
    for i in range(1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')

    ax.set_title('Baseline Comparison Summary', fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_path / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize baseline comparison results")
    parser.add_argument(
        "--results",
        type=str,
        default="results/baseline_comparison.csv",
        help="Path to results CSV or JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}...")
    df = load_results(args.results)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_comparison_plots(df, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
