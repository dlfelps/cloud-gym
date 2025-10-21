#!/usr/bin/env python3
"""Train the winning hyperparameter configurations with full 500k timesteps."""

import argparse
from pathlib import Path
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv


def make_env(seed: int = 0):
    """Create environment."""
    def _init():
        env = CloudResourceEnv(
            n_vms=10,
            n_availability_zones=3,
            max_episode_steps=200,
            arrival_rate=3.0,
            vm_failure_rate=0.001,
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def create_envs(seed: int, n_train_envs: int = 8):
    """Create training and evaluation environments."""
    train_env = SubprocVecEnv([
        make_env(seed + i) for i in range(n_train_envs)
    ])
    eval_env = SubprocVecEnv([
        make_env(seed + 1000 + i) for i in range(4)
    ])

    # Apply normalization
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        training=False,
        clip_reward=10.0,
        gamma=0.99,
    )

    return train_env, eval_env


def train_a2c_high_entropy(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_envs: int = 8,
):
    """Train A2C with high entropy (WINNER configuration)."""
    print("\n" + "=" * 80)
    print("TRAINING A2C - HIGH ENTROPY (WINNER)")
    print("=" * 80)
    print("Hyperparameters:")
    print("  Learning rate: 7e-4")
    print("  N steps: 5")
    print("  Entropy coef: 0.05 (5x higher than baseline)")
    print("  Network arch: [256, 256]")
    print("  Total timesteps: 500,000")
    print("=" * 80)

    output_path = Path(output_dir) / "a2c_high_entropy_500k"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env, eval_env = create_envs(seed, n_envs)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path),
        log_path=str(output_path),
        eval_freq=max(5000 // n_envs, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path=str(output_path),
        name_prefix="a2c_high_entropy_checkpoint",
        save_vecnormalize=True,
    )

    # Create model with WINNING hyperparameters
    model = A2C(
        "MultiInputPolicy",
        train_env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.05,  # <<<< KEY: 5x higher than baseline
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log=f"{output_path}/tensorboard",
        seed=seed,
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_path}/a2c_high_entropy_final")
    train_env.save(f"{output_path}/vec_normalize.pkl")

    print(f"\n[SUCCESS] A2C high_entropy training complete!")
    print(f"Model saved to: {output_path}")

    train_env.close()
    eval_env.close()
    return model


def train_dqn_high_lr(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_envs: int = 8,
):
    """Train DQN with high learning rate (WINNER configuration)."""
    print("\n" + "=" * 80)
    print("TRAINING DQN - HIGH LEARNING RATE (WINNER)")
    print("=" * 80)
    print("Hyperparameters:")
    print("  Learning rate: 3e-4 (3x higher than baseline)")
    print("  Buffer size: 50,000")
    print("  Batch size: 128")
    print("  Exploration fraction: 0.1")
    print("  Exploration final eps: 0.05")
    print("  Network arch: [256, 256]")
    print("  Total timesteps: 500,000")
    print("=" * 80)

    output_path = Path(output_dir) / "dqn_high_lr_500k"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env, eval_env = create_envs(seed, n_envs)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path),
        log_path=str(output_path),
        eval_freq=max(5000 // n_envs, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path=str(output_path),
        name_prefix="dqn_high_lr_checkpoint",
        save_vecnormalize=True,
    )

    # Create model with WINNING hyperparameters
    model = DQN(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,  # <<<< KEY: 3x higher than baseline
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[256, 256]
        ),
        verbose=1,
        tensorboard_log=f"{output_path}/tensorboard",
        seed=seed,
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_path}/dqn_high_lr_final")
    train_env.save(f"{output_path}/vec_normalize.pkl")

    print(f"\n[SUCCESS] DQN high_lr training complete!")
    print(f"Model saved to: {output_path}")

    train_env.close()
    eval_env.close()
    return model


def train_dqn_high_exploration(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_envs: int = 8,
):
    """Train DQN with high exploration (RUNNER-UP configuration)."""
    print("\n" + "=" * 80)
    print("TRAINING DQN - HIGH EXPLORATION (RUNNER-UP)")
    print("=" * 80)
    print("Hyperparameters:")
    print("  Learning rate: 1e-4")
    print("  Buffer size: 50,000")
    print("  Batch size: 128")
    print("  Exploration fraction: 0.3 (3x higher than baseline)")
    print("  Exploration final eps: 0.1 (2x higher than baseline)")
    print("  Network arch: [256, 256]")
    print("  Total timesteps: 500,000")
    print("=" * 80)

    output_path = Path(output_dir) / "dqn_high_exploration_500k"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env, eval_env = create_envs(seed, n_envs)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path),
        log_path=str(output_path),
        eval_freq=max(5000 // n_envs, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path=str(output_path),
        name_prefix="dqn_high_exploration_checkpoint",
        save_vecnormalize=True,
    )

    # Create model with WINNING hyperparameters
    model = DQN(
        "MultiInputPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,  # <<<< KEY: 3x higher
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # <<<< KEY: 2x higher
        policy_kwargs=dict(
            net_arch=[256, 256]
        ),
        verbose=1,
        tensorboard_log=f"{output_path}/tensorboard",
        seed=seed,
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_path}/dqn_high_exploration_final")
    train_env.save(f"{output_path}/vec_normalize.pkl")

    print(f"\n[SUCCESS] DQN high_exploration training complete!")
    print(f"Model saved to: {output_path}")

    train_env.close()
    eval_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train winning hyperparameter configurations"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["a2c_high_entropy", "dqn_high_lr", "dqn_high_exploration", "all"],
        default=["all"],
        help="Which winner(s) to train (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/winners",
        help="Output directory for models",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )

    args = parser.parse_args()

    # Determine variants to train
    variants = args.variants
    if "all" in variants:
        variants = ["a2c_high_entropy", "dqn_high_lr", "dqn_high_exploration"]

    print("=" * 80)
    print("TRAINING WINNER CONFIGURATIONS")
    print("=" * 80)
    print(f"Variants: {', '.join(variants)}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    print("\nEstimated time:")
    print("  A2C high_entropy: ~10 minutes")
    print("  DQN high_lr: ~42 minutes")
    print("  DQN high_exploration: ~42 minutes")
    print("  Total (all 3): ~95 minutes (~1.6 hours)")
    print("=" * 80)

    # Train each variant
    if "a2c_high_entropy" in variants:
        train_a2c_high_entropy(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=args.output_dir,
            n_envs=args.n_envs,
        )

    if "dqn_high_lr" in variants:
        train_dqn_high_lr(
            total_timesteps=args.timesteps,
            seed=args.seed + 100,
            output_dir=args.output_dir,
            n_envs=args.n_envs,
        )

    if "dqn_high_exploration" in variants:
        train_dqn_high_exploration(
            total_timesteps=args.timesteps,
            seed=args.seed + 200,
            output_dir=args.output_dir,
            n_envs=args.n_envs,
        )

    print("\n" + "=" * 80)
    print("ALL WINNER TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Evaluate: uv run scripts/evaluate_baselines.py --n-episodes 100")
    print("  2. Visualize: uv run scripts/visualize_results.py")
    print("  3. Compare to original baseline_comparison.csv")
    print(f"  4. TensorBoard: tensorboard --logdir {args.output_dir}")


if __name__ == "__main__":
    main()
