"""Evaluation script for comparing baseline policies."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

from stable_baselines3 import PPO, A2C, DQN

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv
from cloud_resource_gym.policies.heuristic import (
    RandomPolicy,
    RoundRobinPolicy,
    FirstFitPolicy,
    BestFitPolicy,
    WorstFitPolicy,
    PriorityBestFitPolicy,
    EarliestDeadlineFirstPolicy,
)


class PolicyEvaluator:
    """Evaluate policies on cloud resource allocation environment."""

    def __init__(
        self,
        env: CloudResourceEnv,
        n_episodes: int = 100,
        seed: int = 0,
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def evaluate_heuristic_policy(
        self,
        policy_class,
        policy_name: str,
    ) -> dict:
        """Evaluate a heuristic policy."""
        print(f"Evaluating {policy_name}...")

        policy = policy_class(n_vms=self.env.n_vms, seed=self.seed)
        episode_rewards = []
        episode_metrics = []

        for episode in tqdm(range(self.n_episodes), desc=policy_name):
            # Reset environment and policy
            obs, info = self.env.reset(seed=self.seed + episode)
            policy.reset()

            episode_reward = 0.0
            done = False

            while not done:
                # Get current task if any
                if self.env.current_task_index < len(self.env.pending_tasks):
                    task = self.env.pending_tasks[self.env.current_task_index]
                    action_mask = obs['action_mask']

                    # Get policy action
                    action = policy.select_action(
                        task=task,
                        vms=self.env.vms,
                        action_mask=action_mask,
                        current_time=self.env.current_time,
                    )
                else:
                    # No task to process, should not happen but handle gracefully
                    action = self.env.n_vms  # Reject action

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_metrics.append(info['metrics'])

        # Aggregate results
        return self._aggregate_results(policy_name, episode_rewards, episode_metrics)

    def evaluate_rl_policy(
        self,
        model_path: str,
        algorithm: str,
        policy_name: str,
    ) -> dict:
        """Evaluate a trained RL policy."""
        print(f"Evaluating {policy_name}...")

        # Load model
        if algorithm == "ppo":
            model = PPO.load(model_path)
        elif algorithm == "a2c":
            model = A2C.load(model_path)
        elif algorithm == "dqn":
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        episode_rewards = []
        episode_metrics = []

        for episode in tqdm(range(self.n_episodes), desc=policy_name):
            obs, info = self.env.reset(seed=self.seed + episode)

            episode_reward = 0.0
            done = False

            while not done:
                # Get model action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_metrics.append(info['metrics'])

        return self._aggregate_results(policy_name, episode_rewards, episode_metrics)

    def _aggregate_results(
        self,
        policy_name: str,
        episode_rewards: list,
        episode_metrics: list,
    ) -> dict:
        """Aggregate evaluation results."""
        # Compute statistics
        rewards_array = np.array(episode_rewards)

        # Extract metrics
        total_tasks = [m['total_tasks'] for m in episode_metrics]
        completed_tasks = [m['completed_tasks'] for m in episode_metrics]
        rejected_tasks = [m['rejected_tasks'] for m in episode_metrics]
        sla_violations = [m['sla_violations'] for m in episode_metrics]
        tasks_with_deadlines = [m.get('tasks_with_deadlines', 0) for m in episode_metrics]
        deadline_met = [m.get('deadline_met', 0) for m in episode_metrics]
        energy_costs = [m['total_energy_cost'] for m in episode_metrics]
        vm_costs = [m['total_vm_cost'] for m in episode_metrics]

        # Calculate derived metrics
        completion_rates = [
            c / t if t > 0 else 0 for c, t in zip(completed_tasks, total_tasks)
        ]

        # FIXED: SLA success rate only counts tasks with deadlines
        sla_satisfaction_rates = [
            (dm / twd) if twd > 0 else 1.0
            for dm, twd in zip(deadline_met, tasks_with_deadlines)
        ]

        sla_violation_rates = [
            (v / twd) if twd > 0 else 0.0
            for v, twd in zip(sla_violations, tasks_with_deadlines)
        ]

        # BETTER METRIC: Overall task success rate (completed with SLA / total with deadlines)
        # This accounts for rejections - rejected tasks with deadlines count as failures
        overall_sla_success_rates = [
            (dm / tt) if tt > 0 else 0.0  # deadline_met / total_tasks
            for dm, tt in zip(deadline_met, total_tasks)
        ]

        results = {
            'policy_name': policy_name,
            'n_episodes': self.n_episodes,
            # Reward statistics
            'mean_reward': float(np.mean(rewards_array)),
            'std_reward': float(np.std(rewards_array)),
            'min_reward': float(np.min(rewards_array)),
            'max_reward': float(np.max(rewards_array)),
            # Task statistics
            'mean_total_tasks': float(np.mean(total_tasks)),
            'mean_completed_tasks': float(np.mean(completed_tasks)),
            'mean_rejected_tasks': float(np.mean(rejected_tasks)),
            'mean_completion_rate': float(np.mean(completion_rates)),
            'std_completion_rate': float(np.std(completion_rates)),
            # SLA statistics (FIXED)
            'mean_tasks_with_deadlines': float(np.mean(tasks_with_deadlines)),
            'mean_deadline_met': float(np.mean(deadline_met)),
            'mean_sla_violations': float(np.mean(sla_violations)),
            'mean_sla_satisfaction_rate': float(np.mean(sla_satisfaction_rates)),
            'std_sla_satisfaction_rate': float(np.std(sla_satisfaction_rates)),
            'mean_sla_violation_rate': float(np.mean(sla_violation_rates)),
            'std_sla_violation_rate': float(np.std(sla_violation_rates)),
            # BETTER: Overall SLA success (includes rejected tasks as failures)
            'mean_overall_sla_success': float(np.mean(overall_sla_success_rates)),
            'std_overall_sla_success': float(np.std(overall_sla_success_rates)),
            # Cost statistics
            'mean_energy_cost': float(np.mean(energy_costs)),
            'mean_vm_cost': float(np.mean(vm_costs)),
            'mean_total_cost': float(np.mean([e + v for e, v in zip(energy_costs, vm_costs)])),
        }

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained RL models",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create environment with balanced rewards for fair comparison
    # Higher arrival rate (3.0) creates more resource contention and queueing
    # This makes deadline-aware policies significantly better than naive ones
    env = CloudResourceEnv(
        n_vms=10,
        n_availability_zones=3,
        max_episode_steps=200,
        arrival_rate=3.0,  # Increased from 2.0 to create more pressure
        vm_failure_rate=0.001,
        reward_weights={
            'utilization': 2.0,
            'sla_violation': -3.0,
            'energy_cost': -0.005,
            'queue_length': -0.02,
            'completion': 2.0,
        },
        seed=args.seed,
    )

    # Create evaluator
    evaluator = PolicyEvaluator(
        env=env,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )

    # Evaluate all heuristic policies
    heuristic_policies = [
        (RandomPolicy, "Random"),
        (RoundRobinPolicy, "RoundRobin"),
        (FirstFitPolicy, "FirstFit"),
        (BestFitPolicy, "BestFit"),
        (WorstFitPolicy, "WorstFit"),
        (PriorityBestFitPolicy, "PriorityBestFit"),
        (EarliestDeadlineFirstPolicy, "EarliestDeadlineFirst"),
    ]

    all_results = []

    for policy_class, policy_name in heuristic_policies:
        results = evaluator.evaluate_heuristic_policy(policy_class, policy_name)
        all_results.append(results)

    # Evaluate RL policies if models exist
    rl_policies = [
        ("ppo", "PPO", Path(args.model_dir) / "ppo_improved/ppo_final.zip"),
        ("a2c", "A2C", Path(args.model_dir) / "winners/a2c_high_entropy_500k/a2c_high_entropy_final.zip"),
        ("dqn", "DQN", Path(args.model_dir) / "winners/dqn_high_lr_500k/dqn_high_lr_final.zip"),
    ]

    for algorithm, policy_name, model_path in rl_policies:
        if model_path.exists():
            results = evaluator.evaluate_rl_policy(
                model_path=str(model_path),
                algorithm=algorithm,
                policy_name=policy_name,
            )
            all_results.append(results)
        else:
            print(f"Warning: Model not found at {model_path}, skipping {policy_name}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_csv_path = Path(args.output_dir) / "baseline_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")

    # Save detailed JSON
    results_json_path = Path(args.output_dir) / "baseline_comparison.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to {results_json_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))

    # Print rankings
    print("\n" + "="*80)
    print("RANKINGS")
    print("="*80)
    print("\nBy Mean Reward:")
    ranked_by_reward = results_df.sort_values('mean_reward', ascending=False)
    print(ranked_by_reward[['policy_name', 'mean_reward', 'std_reward']].to_string(index=False))

    print("\nBy Completion Rate:")
    ranked_by_completion = results_df.sort_values('mean_completion_rate', ascending=False)
    print(ranked_by_completion[['policy_name', 'mean_completion_rate', 'std_completion_rate']].to_string(index=False))

    print("\nBy SLA Satisfaction Rate (Completed Tasks Only):")
    ranked_by_sla = results_df.sort_values('mean_sla_satisfaction_rate', ascending=False)
    print(ranked_by_sla[['policy_name', 'mean_sla_satisfaction_rate', 'std_sla_satisfaction_rate']].to_string(index=False))

    print("\nBy Overall SLA Success (Includes Rejections as Failures - BETTER METRIC):")
    ranked_by_overall_sla = results_df.sort_values('mean_overall_sla_success', ascending=False)
    print(ranked_by_overall_sla[['policy_name', 'mean_overall_sla_success', 'std_overall_sla_success', 'mean_completion_rate']].to_string(index=False))


if __name__ == "__main__":
    main()
