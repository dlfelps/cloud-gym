"""Train all RL algorithms (PPO, A2C, DQN) with improved process."""

import argparse
from pathlib import Path
import multiprocessing
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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
                'sla_violation': -3.0,
                'energy_cost': -0.005,
                'queue_length': -0.02,
                'completion': 2.0,
            },
            "aggressive": {
                'utilization': 5.0,
                'sla_violation': -2.0,
                'energy_cost': -0.001,
                'queue_length': -0.01,
                'completion': 5.0,
            },
            "conservative": {
                'utilization': 1.0,
                'sla_violation': -8.0,
                'energy_cost': -0.01,
                'queue_length': -0.1,
                'completion': 1.0,
            },
            "exploration": {
                'utilization': 1.5,
                'sla_violation': -1.0,
                'energy_cost': -0.001,
                'queue_length': -0.01,
                'completion': 3.0,
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


def create_envs(seed: int, reward_config: str, use_parallel: bool, n_train_envs: int = 8):
    """Create training and evaluation environments."""
    if use_parallel:
        train_env = SubprocVecEnv([
            make_env(seed + i, reward_config) for i in range(n_train_envs)
        ])
        eval_env = SubprocVecEnv([
            make_env(seed + 1000 + i, reward_config) for i in range(4)
        ])
    else:
        train_env = DummyVecEnv([make_env(seed, reward_config)])
        eval_env = DummyVecEnv([make_env(seed + 1000, reward_config)])

    return train_env, eval_env


def train_ppo_improved(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    reward_config: str,
    use_parallel: bool,
    normalize_rewards: bool,
    n_envs: int,
):
    """Train PPO with improved settings."""
    print("\n" + "=" * 80)
    print("TRAINING PPO")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env, eval_env = create_envs(seed, reward_config, use_parallel, n_envs)

    # Apply normalization
    if normalize_rewards:
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

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(5000 // (n_envs if use_parallel else 1), 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // (n_envs if use_parallel else 1), 1000),
        save_path=output_dir,
        name_prefix="ppo_checkpoint",
        save_vecnormalize=normalize_rewards,
    )

    # Create model
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_dir}/ppo_final")
    if normalize_rewards:
        train_env.save(f"{output_dir}/vec_normalize.pkl")

    print(f"\nPPO training complete! Model saved to {output_dir}/ppo_final")

    train_env.close()
    eval_env.close()
    return model


def train_a2c_improved(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    reward_config: str,
    use_parallel: bool,
    normalize_rewards: bool,
    n_envs: int,
):
    """Train A2C with improved settings."""
    print("\n" + "=" * 80)
    print("TRAINING A2C")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env, eval_env = create_envs(seed, reward_config, use_parallel, n_envs)

    # Apply normalization
    if normalize_rewards:
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

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(5000 // (n_envs if use_parallel else 1), 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // (n_envs if use_parallel else 1), 1000),
        save_path=output_dir,
        name_prefix="a2c_checkpoint",
        save_vecnormalize=normalize_rewards,
    )

    # Create model
    model = A2C(
        "MultiInputPolicy",
        train_env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_dir}/a2c_final")
    if normalize_rewards:
        train_env.save(f"{output_dir}/vec_normalize.pkl")

    print(f"\nA2C training complete! Model saved to {output_dir}/a2c_final")

    train_env.close()
    eval_env.close()
    return model


def train_dqn_improved(
    total_timesteps: int,
    seed: int,
    output_dir: str,
    reward_config: str,
    use_parallel: bool,
    normalize_rewards: bool,
    n_envs: int,
):
    """Train DQN with improved settings."""
    print("\n" + "=" * 80)
    print("TRAINING DQN")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env, eval_env = create_envs(seed, reward_config, use_parallel, n_envs)

    # Apply normalization
    if normalize_rewards:
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

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(5000 // (n_envs if use_parallel else 1), 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // (n_envs if use_parallel else 1), 1000),
        save_path=output_dir,
        name_prefix="dqn_checkpoint",
        save_vecnormalize=normalize_rewards,
    )

    # Create model
    model = DQN(
        "MultiInputPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,  # Increased from 32
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
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save
    model.save(f"{output_dir}/dqn_final")
    if normalize_rewards:
        train_env.save(f"{output_dir}/vec_normalize.pkl")

    print(f"\nDQN training complete! Model saved to {output_dir}/dqn_final")

    train_env.close()
    eval_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train all RL algorithms with improved process"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["ppo", "a2c", "dqn", "all"],
        default=["all"],
        help="Which algorithms to train (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps per algorithm (default: 200,000)",
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
        default="models",
        help="Base output directory for models",
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
        help="Disable parallel environments",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable reward normalization",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: auto-detect CPU count)",
    )

    args = parser.parse_args()

    # Determine algorithms to train
    algorithms = args.algorithms
    if "all" in algorithms:
        algorithms = ["ppo", "a2c", "dqn"]

    # Determine number of parallel environments
    use_parallel = not args.no_parallel
    if args.n_envs:
        n_envs = args.n_envs
    else:
        n_envs = min(8, multiprocessing.cpu_count()) if use_parallel else 1

    normalize_rewards = not args.no_normalize

    print("=" * 80)
    print("TRAINING ALL ALGORITHMS (IMPROVED PROCESS)")
    print("=" * 80)
    print(f"Algorithms: {', '.join(algorithms).upper()}")
    print(f"Timesteps per algorithm: {args.timesteps:,}")
    print(f"Reward config: {args.reward_config}")
    print(f"Parallel environments: {use_parallel} ({n_envs} envs)" if use_parallel else "Single environment")
    print(f"Reward normalization: {normalize_rewards}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # Train each algorithm
    if "ppo" in algorithms:
        train_ppo_improved(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=f"{args.output_dir}/ppo_improved",
            reward_config=args.reward_config,
            use_parallel=use_parallel,
            normalize_rewards=normalize_rewards,
            n_envs=n_envs,
        )

    if "a2c" in algorithms:
        train_a2c_improved(
            total_timesteps=args.timesteps,
            seed=args.seed + 100,  # Different seed for diversity
            output_dir=f"{args.output_dir}/a2c_improved",
            reward_config=args.reward_config,
            use_parallel=use_parallel,
            normalize_rewards=normalize_rewards,
            n_envs=n_envs,
        )

    if "dqn" in algorithms:
        train_dqn_improved(
            total_timesteps=args.timesteps,
            seed=args.seed + 200,  # Different seed for diversity
            output_dir=f"{args.output_dir}/dqn_improved",
            reward_config=args.reward_config,
            use_parallel=use_parallel,
            normalize_rewards=normalize_rewards,
            n_envs=n_envs,
        )

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to:")
    for algo in algorithms:
        print(f"  - {args.output_dir}/{algo}_improved/{algo}_final.zip")
    print("\nTo compare models, run:")
    print("  python scripts/evaluate_models.py --compare-all")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {args.output_dir}")


if __name__ == "__main__":
    main()
