"""Improved PPO training script with better reward shaping and hyperparameters."""

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import multiprocessing

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv


def make_env(seed: int = 0, reward_config: str = "balanced"):
    """
    Create and wrap the environment with improved reward configuration.

    Args:
        seed: Random seed
        reward_config: One of "balanced", "aggressive", "conservative", "exploration"
    """
    def _init():
        # Define reward weight configurations
        reward_configs = {
            "balanced": {
                'utilization': 2.0,
                'sla_violation': -3.0,      # Less harsh
                'energy_cost': -0.005,      # Less harsh
                'queue_length': -0.02,      # Less harsh
                'completion': 2.0,          # Higher reward
            },
            "aggressive": {
                'utilization': 5.0,          # Focus on utilization
                'sla_violation': -2.0,       # Very lenient
                'energy_cost': -0.001,       # Almost ignore
                'queue_length': -0.01,       # Almost ignore
                'completion': 5.0,           # High completion reward
            },
            "conservative": {
                'utilization': 1.0,
                'sla_violation': -8.0,       # Strict SLA
                'energy_cost': -0.01,
                'queue_length': -0.1,
                'completion': 1.0,
            },
            "exploration": {
                'utilization': 1.5,
                'sla_violation': -1.0,       # Very lenient for exploration
                'energy_cost': -0.001,
                'queue_length': -0.01,
                'completion': 3.0,           # Encourage trying things
            },
        }

        env = CloudResourceEnv(
            n_vms=10,
            n_availability_zones=3,
            max_episode_steps=200,
            arrival_rate=3.0,  # Higher rate = more contention = harder scheduling
            vm_failure_rate=0.001,
            reward_weights=reward_configs.get(reward_config, reward_configs["balanced"]),
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def train_ppo_improved(
    total_timesteps: int = 200_000,
    seed: int = 0,
    output_dir: str = "models/ppo_improved",
    reward_config: str = "balanced",
    use_parallel_envs: bool = True,
    normalize_rewards: bool = True,
    hyperparameter_preset: str = "stable",
):
    """
    Train PPO agent with improved settings.

    Args:
        total_timesteps: Total training timesteps
        seed: Random seed
        output_dir: Output directory for models
        reward_config: Reward configuration ("balanced", "aggressive", "conservative", "exploration")
        use_parallel_envs: Whether to use parallel environments (recommended)
        normalize_rewards: Whether to normalize rewards (recommended)
        hyperparameter_preset: One of "stable", "fast", "exploration"
    """
    print(f"Training Improved PPO for {total_timesteps} timesteps...")
    print(f"Reward config: {reward_config}")
    print(f"Parallel envs: {use_parallel_envs}")
    print(f"Reward normalization: {normalize_rewards}")
    print(f"Hyperparameter preset: {hyperparameter_preset}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create environments
    if use_parallel_envs:
        n_envs = min(8, multiprocessing.cpu_count())
        print(f"Using {n_envs} parallel environments")
        train_env = SubprocVecEnv([
            make_env(seed + i, reward_config) for i in range(n_envs)
        ])
        eval_env = SubprocVecEnv([
            make_env(seed + 1000 + i, reward_config) for i in range(4)
        ])
    else:
        train_env = DummyVecEnv([make_env(seed, reward_config)])
        eval_env = DummyVecEnv([make_env(seed + 1000, reward_config)])

    # Normalize rewards (helps with learning stability)
    if normalize_rewards:
        print("Applying reward normalization")
        train_env = VecNormalize(
            train_env,
            norm_obs=False,  # Don't normalize observations (already scaled)
            norm_reward=True,  # Normalize rewards
            clip_reward=10.0,  # Clip to [-10, 10]
            gamma=0.99,
        )
        # Wrap eval env too, but don't normalize rewards (use raw for evaluation)
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=False,  # Don't normalize rewards in eval
            training=False,  # Don't update stats during eval
            clip_reward=10.0,
            gamma=0.99,
        )

    # Hyperparameter presets
    hyperparameters = {
        "stable": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "clip_range": 0.2,
        },
        "fast": {
            "learning_rate": 5e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "ent_coef": 0.02,
            "clip_range": 0.2,
        },
        "exploration": {
            "learning_rate": 1e-4,
            "n_steps": 4096,
            "batch_size": 128,
            "n_epochs": 20,
            "gamma": 0.995,
            "ent_coef": 0.05,  # High exploration
            "clip_range": 0.3,
        },
    }

    hparams = hyperparameters.get(hyperparameter_preset, hyperparameters["stable"])
    print(f"\nHyperparameters:")
    for k, v in hparams.items():
        print(f"  {k}: {v}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(5000 // (n_envs if use_parallel_envs else 1), 1000),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // (n_envs if use_parallel_envs else 1), 1000),
        save_path=output_dir,
        name_prefix="ppo_improved_checkpoint",
        save_vecnormalize=normalize_rewards,
    )

    # Create PPO model with improved settings
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=hparams["learning_rate"],
        n_steps=hparams["n_steps"],
        batch_size=hparams["batch_size"],
        n_epochs=hparams["n_epochs"],
        gamma=hparams["gamma"],
        gae_lambda=0.95,
        clip_range=hparams["clip_range"],
        ent_coef=hparams["ent_coef"],
        vf_coef=0.5,              # Value function coefficient
        max_grad_norm=0.5,        # Gradient clipping for stability
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],    # Actor network
                vf=[256, 256]     # Critic network
            )
        ),
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    print("\nStarting training...")
    print("=" * 60)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(f"{output_dir}/ppo_improved_final")
    if normalize_rewards:
        train_env.save(f"{output_dir}/vec_normalize.pkl")

    print("=" * 60)
    print(f"Training complete!")
    print(f"Model saved to {output_dir}/ppo_improved_final")
    if normalize_rewards:
        print(f"Normalization stats saved to {output_dir}/vec_normalize.pkl")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with improved reward shaping and hyperparameters"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ppo_improved",
        help="Output directory for models",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        choices=["balanced", "aggressive", "conservative", "exploration"],
        default="balanced",
        help="Reward configuration preset",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel environments (slower)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable reward normalization",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        choices=["stable", "fast", "exploration"],
        default="stable",
        help="Hyperparameter preset",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("IMPROVED PPO TRAINING")
    print("=" * 60)
    print()

    train_ppo_improved(
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
        reward_config=args.reward_config,
        use_parallel_envs=not args.no_parallel,
        normalize_rewards=not args.no_normalize,
        hyperparameter_preset=args.hyperparameters,
    )


if __name__ == "__main__":
    main()
