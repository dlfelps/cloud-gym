#!/usr/bin/env python3
"""Comprehensive evaluation of all policies: PPO, A2C, DQN, and heuristics."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv
from cloud_resource_gym.policies import FirstFitPolicy, BestFitPolicy, WorstFitPolicy, RandomPolicy


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
        'energy_cost': [],
        'vm_cost': [],
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

                overall_sla = metrics['deadline_met'] / total_tasks if total_tasks > 0 else 0
                episode_metrics['overall_sla_success'].append(overall_sla)

                episode_metrics['energy_cost'].append(metrics['total_energy_cost'])
                episode_metrics['vm_cost'].append(metrics['total_vm_cost'])
                total_cost = metrics['total_energy_cost'] + metrics['total_vm_cost']
                episode_metrics['total_cost'].append(total_cost)

        episode_rewards.append(episode_reward)

    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_completion_rate': np.mean(episode_metrics['completion_rate']),
        'std_completion_rate': np.std(episode_metrics['completion_rate']),
        'mean_overall_sla_success': np.mean(episode_metrics['overall_sla_success']),
        'std_overall_sla_success': np.std(episode_metrics['overall_sla_success']),
        'mean_energy_cost': np.mean(episode_metrics['energy_cost']),
        'mean_vm_cost': np.mean(episode_metrics['vm_cost']),
        'mean_total_cost': np.mean(episode_metrics['total_cost']),
    }

    return results


def evaluate_heuristic_policy(policy_class, policy_name: str, n_episodes: int = 100):
    """Evaluate a heuristic policy."""
    env = CloudResourceEnv(
        n_vms=10,
        n_availability_zones=3,
        max_episode_steps=200,
        arrival_rate=3.0,
        vm_failure_rate=0.001,
        seed=42,
    )

    episode_rewards = []
    episode_metrics = {
        'completed_tasks': [],
        'rejected_tasks': [],
        'completion_rate': [],
        'overall_sla_success': [],
        'energy_cost': [],
        'vm_cost': [],
        'total_cost': [],
    }

    policy = policy_class(env)

    for episode in tqdm(range(n_episodes), desc=f"  Evaluating {policy_name}"):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if done:
                metrics = info['metrics']
                episode_metrics['completed_tasks'].append(metrics['completed_tasks'])
                episode_metrics['rejected_tasks'].append(metrics['rejected_tasks'])

                total_tasks = metrics['total_tasks']
                completion_rate = metrics['completed_tasks'] / total_tasks if total_tasks > 0 else 0
                episode_metrics['completion_rate'].append(completion_rate)

                overall_sla = metrics['deadline_met'] / total_tasks if total_tasks > 0 else 0
                episode_metrics['overall_sla_success'].append(overall_sla)

                episode_metrics['energy_cost'].append(metrics['total_energy_cost'])
                episode_metrics['vm_cost'].append(metrics['total_vm_cost'])
                total_cost = metrics['total_energy_cost'] + metrics['total_vm_cost']
                episode_metrics['total_cost'].append(total_cost)

        episode_rewards.append(episode_reward)

    results = {
        'policy_name': policy_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_completion_rate': np.mean(episode_metrics['completion_rate']),
        'std_completion_rate': np.std(episode_metrics['completion_rate']),
        'mean_overall_sla_success': np.mean(episode_metrics['overall_sla_success']),
        'std_overall_sla_success': np.std(episode_metrics['overall_sla_success']),
        'mean_energy_cost': np.mean(episode_metrics['energy_cost']),
        'mean_vm_cost': np.mean(episode_metrics['vm_cost']),
        'mean_total_cost': np.mean(episode_metrics['total_cost']),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all policies comprehensively"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_comparison.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE POLICY EVALUATION")
    print("=" * 80)
    print(f"Episodes per policy: {args.n_episodes}")
    print("=" * 80)

    all_results = []

    # Evaluate heuristics
    print("\n[HEURISTICS]")
    heuristic_policies = [
        (FirstFitPolicy, "FirstFit"),
        (BestFitPolicy, "BestFit"),
        (WorstFitPolicy, "WorstFit"),
        (RandomPolicy, "Random"),
    ]

    for policy_class, policy_name in heuristic_policies:
        results = evaluate_heuristic_policy(policy_class, policy_name, args.n_episodes)
        all_results.append(results)

    # Evaluate RL models
    print("\n[RL MODELS]")

    # PPO
    print("  Evaluating PPO...")
    ppo_path = Path("models/ppo_cloud_500k.zip")
    if ppo_path.exists():
        env = DummyVecEnv([make_eval_env(seed=42)])
        vec_normalize_path = Path("models/ppo_vecnormalize.pkl")
        if vec_normalize_path.exists():
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
        model = PPO.load(str(ppo_path), env=env)
        results = evaluate_model(model, env, args.n_episodes)
        results['policy_name'] = 'PPO'
        all_results.append(results)
        env.close()
    else:
        print(f"    [WARNING] PPO model not found at {ppo_path}")

    # A2C - High Entropy
    print("  Evaluating A2C...")
    a2c_path = Path("models/winners/a2c_high_entropy_500k/a2c_high_entropy_final.zip")
    if a2c_path.exists():
        env = DummyVecEnv([make_eval_env(seed=42)])
        vec_normalize_path = Path("models/winners/a2c_high_entropy_500k/vec_normalize.pkl")
        if vec_normalize_path.exists():
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
        model = A2C.load(str(a2c_path), env=env)
        results = evaluate_model(model, env, args.n_episodes)
        results['policy_name'] = 'A2C'
        all_results.append(results)
        env.close()
    else:
        print(f"    [WARNING] A2C model not found at {a2c_path}")

    # DQN - High LR
    print("  Evaluating DQN (high_lr)...")
    dqn_lr_path = Path("models/winners/dqn_high_lr_500k/dqn_high_lr_final.zip")
    if dqn_lr_path.exists():
        env = DummyVecEnv([make_eval_env(seed=42)])
        vec_normalize_path = Path("models/winners/dqn_high_lr_500k/vec_normalize.pkl")
        if vec_normalize_path.exists():
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
        model = DQN.load(str(dqn_lr_path), env=env)
        results = evaluate_model(model, env, args.n_episodes)
        results['policy_name'] = 'DQN_HighLR'
        all_results.append(results)
        env.close()
    else:
        print(f"    [WARNING] DQN (high_lr) model not found at {dqn_lr_path}")

    # DQN - High Exploration
    print("  Evaluating DQN (high_exploration)...")
    dqn_exp_path = Path("models/winners/dqn_high_exploration_500k/dqn_high_exploration_final.zip")
    if dqn_exp_path.exists():
        env = DummyVecEnv([make_eval_env(seed=42)])
        vec_normalize_path = Path("models/winners/dqn_high_exploration_500k/vec_normalize.pkl")
        if vec_normalize_path.exists():
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
        model = DQN.load(str(dqn_exp_path), env=env)
        results = evaluate_model(model, env, args.n_episodes)
        results['policy_name'] = 'DQN_HighExploration'
        all_results.append(results)
        env.close()
    else:
        print(f"    [WARNING] DQN (high_exploration) model not found at {dqn_exp_path}")

    # Create DataFrame and save
    df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        'policy_name',
        'mean_reward',
        'std_reward',
        'mean_completion_rate',
        'std_completion_rate',
        'mean_overall_sla_success',
        'std_overall_sla_success',
        'mean_energy_cost',
        'mean_vm_cost',
        'mean_total_cost',
    ]
    df = df[column_order]

    # Sort by mean reward (descending)
    df = df.sort_values('mean_reward', ascending=False)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nTop 5 by Mean Reward:")
    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"  {i}. {row['policy_name']:20} | Reward: {row['mean_reward']:8.2f} | Completion: {row['mean_completion_rate']:5.1%}")

    print("\nTop 5 by Completion Rate:")
    df_by_completion = df.sort_values('mean_completion_rate', ascending=False)
    for i, (_, row) in enumerate(df_by_completion.head(5).iterrows(), 1):
        print(f"  {i}. {row['policy_name']:20} | Completion: {row['mean_completion_rate']:5.1%} | Reward: {row['mean_reward']:8.2f}")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Results saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Regenerate plots: uv run scripts/visualize_results.py")
    print("  2. View results in results/baseline_comparison.csv")

if __name__ == "__main__":
    main()
