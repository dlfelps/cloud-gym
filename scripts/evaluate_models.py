"""Evaluate and compare different trained models on the same environment."""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import json

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv


def make_eval_env(seed: int = 0, reward_config: str = "balanced"):
    """Create evaluation environment with specified reward configuration."""
    def _init():
        reward_configs = {
            "balanced": {
                'utilization': 2.0,
                'sla_violation': -3.0,
                'energy_cost': -0.005,
                'queue_length': -0.02,
                'completion': 2.0,
            },
            "original": None,  # Use default from CloudResourceEnv
        }

        env = CloudResourceEnv(
            n_vms=10,
            n_availability_zones=3,
            max_episode_steps=200,
            arrival_rate=2.0,
            vm_failure_rate=0.001,
            reward_weights=reward_configs.get(reward_config),
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def load_model(model_path: str, algorithm: str, env):
    """Load a trained model."""
    if algorithm.lower() == "ppo":
        return PPO.load(model_path, env=env)
    elif algorithm.lower() == "a2c":
        return A2C.load(model_path, env=env)
    elif algorithm.lower() == "dqn":
        return DQN.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_model(
    model_path: str,
    algorithm: str,
    n_eval_episodes: int = 20,
    seed: int = 42,
    reward_config: str = "balanced",
    vec_normalize_path: str = None,
):
    """
    Evaluate a single model.

    Args:
        model_path: Path to the saved model
        algorithm: Algorithm type (ppo, a2c, dqn)
        n_eval_episodes: Number of evaluation episodes
        seed: Random seed
        reward_config: Reward configuration to use
        vec_normalize_path: Path to VecNormalize stats (if applicable)

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {model_path}...")

    # Create evaluation environment
    env = DummyVecEnv([make_eval_env(seed, reward_config)])

    # Load VecNormalize stats if provided
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading normalization stats from {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during eval
        env.norm_reward = False  # Use raw rewards for evaluation

    # Load model
    model = load_model(model_path, algorithm, env)

    # Evaluate
    print(f"Running {n_eval_episodes} episodes...")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )

    # Collect detailed metrics from Monitor
    detailed_metrics = {
        'completed_tasks': [],
        'rejected_tasks': [],
        'sla_violations': [],
        'total_tasks': [],
        'tasks_with_deadlines': [],
        'deadline_met': [],
    }

    # Run episodes to collect detailed info
    for _ in range(min(10, n_eval_episodes)):  # Sample 10 episodes for detailed stats
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                if 'metrics' in info[0]:
                    metrics = info[0]['metrics']
                    detailed_metrics['completed_tasks'].append(metrics.get('completed_tasks', 0))
                    detailed_metrics['rejected_tasks'].append(metrics.get('rejected_tasks', 0))
                    detailed_metrics['sla_violations'].append(metrics.get('sla_violations', 0))
                    detailed_metrics['total_tasks'].append(metrics.get('total_tasks', 0))
                    detailed_metrics['tasks_with_deadlines'].append(metrics.get('tasks_with_deadlines', 0))
                    detailed_metrics['deadline_met'].append(metrics.get('deadline_met', 0))

    # Calculate statistics
    results = {
        'model_path': model_path,
        'algorithm': algorithm,
        'reward_config': reward_config,
        'n_episodes': n_eval_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
    }

    # Add detailed metrics if available
    if detailed_metrics['completed_tasks']:
        results['mean_completed_tasks'] = float(np.mean(detailed_metrics['completed_tasks']))
        results['mean_rejected_tasks'] = float(np.mean(detailed_metrics['rejected_tasks']))
        results['mean_sla_violations'] = float(np.mean(detailed_metrics['sla_violations']))
        results['mean_total_tasks'] = float(np.mean(detailed_metrics['total_tasks']))
        results['mean_tasks_with_deadlines'] = float(np.mean(detailed_metrics['tasks_with_deadlines']))
        results['mean_deadline_met'] = float(np.mean(detailed_metrics['deadline_met']))

        # Completion rate: completed / total tasks
        results['completion_rate'] = results['mean_completed_tasks'] / max(results['mean_total_tasks'], 1)

        # Rejection rate: rejected / total tasks
        results['rejection_rate'] = results['mean_rejected_tasks'] / max(results['mean_total_tasks'], 1)

        # SLA success rate: CORRECT - only count tasks with deadlines
        if results['mean_tasks_with_deadlines'] > 0:
            results['sla_success_rate'] = results['mean_deadline_met'] / results['mean_tasks_with_deadlines']
        else:
            results['sla_success_rate'] = 1.0  # No tasks with deadlines = 100% success

        # Alternative: SLA violation rate (violations / tasks with deadlines)
        if results['mean_tasks_with_deadlines'] > 0:
            results['sla_violation_rate'] = results['mean_sla_violations'] / results['mean_tasks_with_deadlines']
        else:
            results['sla_violation_rate'] = 0.0

    env.close()
    return results


def compare_models(model_configs: list, output_file: str = None):
    """
    Compare multiple models.

    Args:
        model_configs: List of dicts with keys: model_path, algorithm, vec_normalize_path (optional)
        output_file: Optional path to save comparison results
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    all_results = []

    for config in model_configs:
        try:
            results = evaluate_model(
                model_path=config['model_path'],
                algorithm=config['algorithm'],
                n_eval_episodes=config.get('n_eval_episodes', 20),
                seed=config.get('seed', 42),
                reward_config=config.get('reward_config', 'balanced'),
                vec_normalize_path=config.get('vec_normalize_path'),
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {config['model_path']}: {e}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Mean Reward':>12} {'Completion':>12} {'SLA Success':>12} {'Rejection':>10}")
    print("-" * 80)

    for result in sorted(all_results, key=lambda x: x['mean_reward'], reverse=True):
        model_name = Path(result['model_path']).stem
        completion = result.get('completion_rate', 0.0)
        sla_success = result.get('sla_success_rate', 0.0)
        rejection = result.get('rejection_rate', 0.0)
        print(f"{model_name:<40} {result['mean_reward']:>12.2f} {completion:>11.1%} {sla_success:>11.1%} {rejection:>9.1%}")

    # Print detailed metrics table
    print("\n" + "=" * 80)
    print("DETAILED SLA METRICS")
    print("=" * 80)
    print(f"{'Model':<40} {'Tasks w/ DL':>12} {'DL Met':>10} {'Violations':>12} {'Violation %':>13}")
    print("-" * 80)

    for result in sorted(all_results, key=lambda x: x.get('sla_success_rate', 0.0), reverse=True):
        model_name = Path(result['model_path']).stem
        tasks_dl = result.get('mean_tasks_with_deadlines', 0.0)
        dl_met = result.get('mean_deadline_met', 0.0)
        violations = result.get('mean_sla_violations', 0.0)
        viol_rate = result.get('sla_violation_rate', 0.0)
        print(f"{model_name:<40} {tasks_dl:>12.1f} {dl_met:>10.1f} {violations:>12.1f} {viol_rate:>12.1%}")

    # Save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained models")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Paths to model files (e.g., models/ppo/ppo_final.zip models/a2c/a2c_final.zip)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Algorithm for each model (ppo, a2c, dqn)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default="balanced",
        choices=["balanced", "original"],
        help="Reward configuration to use for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.json",
        help="Output file for detailed results",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Automatically find and compare all models in models/ directory",
    )

    args = parser.parse_args()

    if args.compare_all:
        # Find all models
        model_configs = []
        models_dir = Path("models")

        # Look for common model files
        for algo in ["ppo", "a2c", "dqn", "ppo_improved"]:
            for model_file in models_dir.rglob(f"{algo}*final.zip"):
                config = {
                    'model_path': str(model_file),
                    'algorithm': algo.replace('_improved', ''),
                    'n_eval_episodes': args.episodes,
                    'seed': args.seed,
                    'reward_config': args.reward_config,
                }

                # Check for VecNormalize stats
                vec_norm_path = model_file.parent / "vec_normalize.pkl"
                if vec_norm_path.exists():
                    config['vec_normalize_path'] = str(vec_norm_path)

                model_configs.append(config)

        if not model_configs:
            print("No models found in models/ directory")
            return

        print(f"Found {len(model_configs)} models to compare")
        compare_models(model_configs, args.output)

    elif args.models and args.algorithms:
        if len(args.models) != len(args.algorithms):
            print("Error: Number of models and algorithms must match")
            return

        model_configs = [
            {
                'model_path': model,
                'algorithm': algo,
                'n_eval_episodes': args.episodes,
                'seed': args.seed,
                'reward_config': args.reward_config,
            }
            for model, algo in zip(args.models, args.algorithms)
        ]

        compare_models(model_configs, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
