#!/usr/bin/env python3
"""Evaluate hyperparameter tuning results for A2C and DQN variants."""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv


def make_eval_env(seed: int = 42):
    """Create evaluation environment."""
    def _init():
        env = CloudResourceEnv(
            n_vms=10,
            n_availability_zones=3,
            max_episode_steps=200,
            arrival_rate=3.0,
            vm_failure_rate=0.001,
            seed=seed,
        )
        return env
    return _init


def evaluate_model(model, env, n_episodes: int = 100):
    """Evaluate a model and return metrics."""
    episode_rewards = []
    episode_metrics = {
        'completed_tasks': [],
        'rejected_tasks': [],
        'completion_rate': [],
        'overall_sla_success': [],
        'total_cost': [],
    }

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            if done:
                metrics = info[0]['metrics'] if isinstance(info, (list, tuple)) else info['metrics']
                episode_metrics['completed_tasks'].append(metrics['completed_tasks'])
                episode_metrics['rejected_tasks'].append(metrics['rejected_tasks'])

                total_tasks = metrics['total_tasks']
                completion_rate = metrics['completed_tasks'] / total_tasks if total_tasks > 0 else 0
                episode_metrics['completion_rate'].append(completion_rate)

                # Overall SLA success (deadline_met / total_tasks)
                overall_sla = metrics['deadline_met'] / total_tasks if total_tasks > 0 else 0
                episode_metrics['overall_sla_success'].append(overall_sla)

                total_cost = metrics['total_energy_cost'] + metrics['total_vm_cost']
                episode_metrics['total_cost'].append(total_cost)

        episode_rewards.append(episode_reward)

    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_completed_tasks': np.mean(episode_metrics['completed_tasks']),
        'mean_rejected_tasks': np.mean(episode_metrics['rejected_tasks']),
        'mean_completion_rate': np.mean(episode_metrics['completion_rate']),
        'std_completion_rate': np.std(episode_metrics['completion_rate']),
        'mean_overall_sla_success': np.mean(episode_metrics['overall_sla_success']),
        'std_overall_sla_success': np.std(episode_metrics['overall_sla_success']),
        'mean_total_cost': np.mean(episode_metrics['total_cost']),
    }

    return results


def load_and_evaluate_variant(variant_path: Path, algorithm: str, n_episodes: int):
    """Load a variant model and evaluate it."""
    variant_name = variant_path.name

    # Find the final model
    final_model_files = list(variant_path.glob(f"{algorithm}_{variant_name.split('_', 1)[1]}_final.zip"))
    if not final_model_files:
        print(f"  ⚠ No final model found for {variant_name}")
        return None

    model_path = final_model_files[0]

    # Create environment
    env = DummyVecEnv([make_eval_env(seed=42)])

    # Load VecNormalize if exists
    vec_normalize_path = variant_path / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False

    # Load model
    try:
        if algorithm == "a2c":
            model = A2C.load(str(model_path), env=env)
        else:  # dqn
            model = DQN.load(str(model_path), env=env)
    except Exception as e:
        print(f"  ⚠ Error loading {variant_name}: {e}")
        return None

    # Evaluate
    print(f"  Evaluating {variant_name}...")
    results = evaluate_model(model, env, n_episodes)
    results['variant_name'] = variant_name
    results['algorithm'] = algorithm

    env.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hyperparameter tuning results"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/hyperparameter_tuning",
        help="Directory containing tuned models",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per variant",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hyperparameter_tuning_results.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return

    print("=" * 80)
    print("EVALUATING HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Episodes per variant: {args.n_episodes}")
    print("=" * 80)

    all_results = []

    # Evaluate A2C variants
    a2c_variants = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("a2c_")])
    if a2c_variants:
        print(f"\n[A2C] Evaluating {len(a2c_variants)} variants...")
        for variant_path in a2c_variants:
            results = load_and_evaluate_variant(variant_path, "a2c", args.n_episodes)
            if results:
                all_results.append(results)

    # Evaluate DQN variants
    dqn_variants = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("dqn_")])
    if dqn_variants:
        print(f"\n[DQN] Evaluating {len(dqn_variants)} variants...")
        for variant_path in dqn_variants:
            results = load_and_evaluate_variant(variant_path, "dqn", args.n_episodes)
            if results:
                all_results.append(results)

    if not all_results:
        print("\n⚠ No results to save!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by mean reward (descending)
    df = df.sort_values('mean_reward', ascending=False)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Best A2C variant
    a2c_results = df[df['algorithm'] == 'a2c']
    if not a2c_results.empty:
        best_a2c = a2c_results.iloc[0]
        print("\n[WINNER] BEST A2C VARIANT:")
        print(f"  Name: {best_a2c['variant_name']}")
        print(f"  Mean Reward: {best_a2c['mean_reward']:.2f} ± {best_a2c['std_reward']:.2f}")
        print(f"  Completion Rate: {best_a2c['mean_completion_rate']:.1%} ± {best_a2c['std_completion_rate']:.1%}")
        print(f"  Overall SLA: {best_a2c['mean_overall_sla_success']:.1%} ± {best_a2c['std_overall_sla_success']:.1%}")
        print(f"  Total Cost: ${best_a2c['mean_total_cost']:.2f}")

        # Compare to baseline
        baseline_a2c = a2c_results[a2c_results['variant_name'].str.contains('baseline')]
        if not baseline_a2c.empty and best_a2c['variant_name'] != baseline_a2c.iloc[0]['variant_name']:
            baseline = baseline_a2c.iloc[0]
            improvement = ((best_a2c['mean_completion_rate'] - baseline['mean_completion_rate'])
                          / baseline['mean_completion_rate'] * 100)
            print(f"\n  Improvement over baseline:")
            print(f"    Completion rate: {improvement:+.1f}%")
            print(f"    Reward: {best_a2c['mean_reward'] - baseline['mean_reward']:+.2f}")

    # Best DQN variant
    dqn_results = df[df['algorithm'] == 'dqn']
    if not dqn_results.empty:
        best_dqn = dqn_results.iloc[0]
        print("\n[WINNER] BEST DQN VARIANT:")
        print(f"  Name: {best_dqn['variant_name']}")
        print(f"  Mean Reward: {best_dqn['mean_reward']:.2f} ± {best_dqn['std_reward']:.2f}")
        print(f"  Completion Rate: {best_dqn['mean_completion_rate']:.1%} ± {best_dqn['std_completion_rate']:.1%}")
        print(f"  Overall SLA: {best_dqn['mean_overall_sla_success']:.1%} ± {best_dqn['std_overall_sla_success']:.1%}")
        print(f"  Total Cost: ${best_dqn['mean_total_cost']:.2f}")

        # Compare to baseline
        baseline_dqn = dqn_results[dqn_results['variant_name'].str.contains('baseline')]
        if not baseline_dqn.empty and best_dqn['variant_name'] != baseline_dqn.iloc[0]['variant_name']:
            baseline = baseline_dqn.iloc[0]
            improvement = ((best_dqn['mean_completion_rate'] - baseline['mean_completion_rate'])
                          / baseline['mean_completion_rate'] * 100)
            print(f"\n  Improvement over baseline:")
            print(f"    Completion rate: {improvement:+.1f}%")
            print(f"    Reward: {best_dqn['mean_reward'] - baseline['mean_reward']:+.2f}")

    # Top 3 overall
    print("\n[TOP 3] OVERALL (by completion rate):")
    top3 = df.nlargest(3, 'mean_completion_rate')
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"\n  {i}. {row['variant_name']} ({row['algorithm'].upper()})")
        print(f"     Completion: {row['mean_completion_rate']:.1%}, Reward: {row['mean_reward']:.2f}")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Results saved to: {output_path}")
    print("\nTo visualize results:")
    print(f"  uv run scripts/visualize_tuning_results.py --results {args.output}")
    print("\nTo view training curves:")
    print(f"  tensorboard --logdir {args.model_dir}")


if __name__ == "__main__":
    main()
