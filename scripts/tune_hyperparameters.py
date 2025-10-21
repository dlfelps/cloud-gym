#!/usr/bin/env python3
"""Hyperparameter tuning for A2C and DQN to improve performance.

This script trains multiple variants of A2C and DQN with different:
- Learning rates
- Network architectures
- Exploration parameters (DQN)
- Entropy coefficients (A2C)
"""

import argparse
from pathlib import Path
import multiprocessing
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv


def make_env(seed: int = 0):
    """Create environment with current best configuration."""
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


def train_a2c_variant(
    variant_name: str,
    learning_rate: float,
    n_steps: int,
    ent_coef: float,
    net_arch: list,
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_envs: int = 8,
):
    """Train A2C with specific hyperparameters."""
    print("\n" + "=" * 80)
    print(f"TRAINING A2C - {variant_name}")
    print("=" * 80)
    print(f"Learning rate: {learning_rate}")
    print(f"N steps: {n_steps}")
    print(f"Entropy coef: {ent_coef}")
    print(f"Network arch: {net_arch}")
    print("=" * 80)

    output_path = Path(output_dir) / f"a2c_{variant_name}"
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
        name_prefix=f"a2c_{variant_name}_checkpoint",
        save_vecnormalize=True,
    )

    # Create model
    model = A2C(
        "MultiInputPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=net_arch, vf=net_arch)
        ),
        verbose=1,
        tensorboard_log=f"{output_path}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_path}/a2c_{variant_name}_final")
    train_env.save(f"{output_path}/vec_normalize.pkl")

    print(f"\nA2C {variant_name} training complete!")

    train_env.close()
    eval_env.close()
    return model


def train_dqn_variant(
    variant_name: str,
    learning_rate: float,
    buffer_size: int,
    batch_size: int,
    exploration_fraction: float,
    exploration_final_eps: float,
    net_arch: list,
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_envs: int = 8,
):
    """Train DQN with specific hyperparameters."""
    print("\n" + "=" * 80)
    print(f"TRAINING DQN - {variant_name}")
    print("=" * 80)
    print(f"Learning rate: {learning_rate}")
    print(f"Buffer size: {buffer_size}")
    print(f"Batch size: {batch_size}")
    print(f"Exploration fraction: {exploration_fraction}")
    print(f"Exploration final eps: {exploration_final_eps}")
    print(f"Network arch: {net_arch}")
    print("=" * 80)

    output_path = Path(output_dir) / f"dqn_{variant_name}"
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
        name_prefix=f"dqn_{variant_name}_checkpoint",
        save_vecnormalize=True,
    )

    # Create model
    model = DQN(
        "MultiInputPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=dict(
            net_arch=net_arch
        ),
        verbose=1,
        tensorboard_log=f"{output_path}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_path}/dqn_{variant_name}_final")
    train_env.save(f"{output_path}/vec_normalize.pkl")

    print(f"\nDQN {variant_name} training complete!")

    train_env.close()
    eval_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for A2C and DQN"
    )
    parser.add_argument(
        "--algorithm",
        choices=["a2c", "dqn", "both"],
        default="both",
        help="Which algorithm to tune",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps per variant (default: 500,000)",
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
        default="models/hyperparameter_tuning",
        help="Output directory for models",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (50k timesteps)",
    )

    args = parser.parse_args()

    timesteps = 50_000 if args.quick_test else args.timesteps

    print("=" * 80)
    print("HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Timesteps per variant: {timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # A2C variants to test
    a2c_variants = [
        # Original (baseline)
        {
            "name": "baseline",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.01,
            "net_arch": [256, 256],
        },
        # Higher learning rate
        {
            "name": "high_lr",
            "learning_rate": 1e-3,
            "n_steps": 5,
            "ent_coef": 0.01,
            "net_arch": [256, 256],
        },
        # Lower learning rate
        {
            "name": "low_lr",
            "learning_rate": 3e-4,
            "n_steps": 5,
            "ent_coef": 0.01,
            "net_arch": [256, 256],
        },
        # More steps (longer rollouts)
        {
            "name": "long_rollout",
            "learning_rate": 7e-4,
            "n_steps": 20,
            "ent_coef": 0.01,
            "net_arch": [256, 256],
        },
        # Higher entropy (more exploration)
        {
            "name": "high_entropy",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.05,
            "net_arch": [256, 256],
        },
        # Deeper network
        {
            "name": "deep_network",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.01,
            "net_arch": [512, 512, 256],
        },
        # Wider network
        {
            "name": "wide_network",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.01,
            "net_arch": [512, 512],
        },
        # Combined best guesses
        {
            "name": "tuned_v1",
            "learning_rate": 5e-4,
            "n_steps": 10,
            "ent_coef": 0.03,
            "net_arch": [512, 256],
        },
    ]

    # DQN variants to test
    dqn_variants = [
        # Original (baseline)
        {
            "name": "baseline",
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "batch_size": 128,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [256, 256],
        },
        # Higher learning rate
        {
            "name": "high_lr",
            "learning_rate": 3e-4,
            "buffer_size": 50_000,
            "batch_size": 128,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [256, 256],
        },
        # Lower learning rate
        {
            "name": "low_lr",
            "learning_rate": 5e-5,
            "buffer_size": 50_000,
            "batch_size": 128,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [256, 256],
        },
        # Larger buffer
        {
            "name": "large_buffer",
            "learning_rate": 1e-4,
            "buffer_size": 100_000,
            "batch_size": 128,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [256, 256],
        },
        # Larger batch size
        {
            "name": "large_batch",
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "batch_size": 256,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [256, 256],
        },
        # More exploration
        {
            "name": "high_exploration",
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "batch_size": 128,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.1,
            "net_arch": [256, 256],
        },
        # Deeper network
        {
            "name": "deep_network",
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "batch_size": 128,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [512, 512, 256],
        },
        # Wider network
        {
            "name": "wide_network",
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "batch_size": 128,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "net_arch": [512, 512],
        },
        # Combined best guesses
        {
            "name": "tuned_v1",
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "batch_size": 256,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.02,
            "net_arch": [512, 256],
        },
    ]

    # Train A2C variants
    if args.algorithm in ["a2c", "both"]:
        print("\n" + "=" * 80)
        print(f"TRAINING {len(a2c_variants)} A2C VARIANTS")
        print("=" * 80)

        for i, variant in enumerate(a2c_variants, 1):
            print(f"\n[{i}/{len(a2c_variants)}] Training A2C variant: {variant['name']}")
            train_a2c_variant(
                variant_name=variant["name"],
                learning_rate=variant["learning_rate"],
                n_steps=variant["n_steps"],
                ent_coef=variant["ent_coef"],
                net_arch=variant["net_arch"],
                total_timesteps=timesteps,
                seed=args.seed + i,
                output_dir=args.output_dir,
                n_envs=args.n_envs,
            )

    # Train DQN variants
    if args.algorithm in ["dqn", "both"]:
        print("\n" + "=" * 80)
        print(f"TRAINING {len(dqn_variants)} DQN VARIANTS")
        print("=" * 80)

        for i, variant in enumerate(dqn_variants, 1):
            print(f"\n[{i}/{len(dqn_variants)}] Training DQN variant: {variant['name']}")
            train_dqn_variant(
                variant_name=variant["name"],
                learning_rate=variant["learning_rate"],
                buffer_size=variant["buffer_size"],
                batch_size=variant["batch_size"],
                exploration_fraction=variant["exploration_fraction"],
                exploration_final_eps=variant["exploration_final_eps"],
                net_arch=variant["net_arch"],
                total_timesteps=timesteps,
                seed=args.seed + 100 + i,
                output_dir=args.output_dir,
                n_envs=args.n_envs,
            )

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {args.output_dir}")
    print("\nTo evaluate all variants:")
    print(f"  uv run scripts/evaluate_tuning_results.py --model-dir {args.output_dir}")
    print("\nTo view training curves:")
    print(f"  tensorboard --logdir {args.output_dir}")


if __name__ == "__main__":
    main()
