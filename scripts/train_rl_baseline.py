"""Training script for RL baseline algorithms."""

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from cloud_resource_gym.envs.cloud_env import CloudResourceEnv


def make_env(seed: int = 0, use_balanced_rewards: bool = True):
    """Create and wrap the environment."""
    def _init():
        # Use same balanced reward weights as improved script for fair comparison
        reward_weights = None
        if use_balanced_rewards:
            reward_weights = {
                'utilization': 2.0,
                'sla_violation': -3.0,
                'energy_cost': -0.005,
                'queue_length': -0.02,
                'completion': 2.0,
            }

        env = CloudResourceEnv(
            n_vms=10,
            n_availability_zones=3,
            max_episode_steps=200,
            arrival_rate=2.0,
            vm_failure_rate=0.001,
            reward_weights=reward_weights,
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def train_ppo(
    total_timesteps: int = 100_000,
    seed: int = 0,
    output_dir: str = "models/ppo"
):
    """Train PPO agent."""
    print(f"Training PPO for {total_timesteps} timesteps...")

    # Create environment
    env = DummyVecEnv([make_env(seed)])
    eval_env = DummyVecEnv([make_env(seed + 1000)])

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix="ppo_checkpoint",
    )

    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    # Save final model
    model.save(f"{output_dir}/ppo_final")
    print(f"PPO training complete. Model saved to {output_dir}/ppo_final")

    return model


def train_a2c(
    total_timesteps: int = 100_000,
    seed: int = 0,
    output_dir: str = "models/a2c"
):
    """Train A2C agent."""
    print(f"Training A2C for {total_timesteps} timesteps...")

    # Create environment
    env = DummyVecEnv([make_env(seed)])
    eval_env = DummyVecEnv([make_env(seed + 1000)])

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix="a2c_checkpoint",
    )

    # Create A2C model
    model = A2C(
        "MultiInputPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    # Save final model
    model.save(f"{output_dir}/a2c_final")
    print(f"A2C training complete. Model saved to {output_dir}/a2c_final")

    return model


def train_dqn(
    total_timesteps: int = 100_000,
    seed: int = 0,
    output_dir: str = "models/dqn"
):
    """Train DQN agent."""
    print(f"Training DQN for {total_timesteps} timesteps...")

    # Create environment
    env = DummyVecEnv([make_env(seed)])
    eval_env = DummyVecEnv([make_env(seed + 1000)])

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix="dqn_checkpoint",
    )

    # Create DQN model
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        seed=seed,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    # Save final model
    model.save(f"{output_dir}/dqn_final")
    print(f"DQN training complete. Model saved to {output_dir}/dqn_final")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL baseline agents")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "a2c", "dqn", "all"],
        default="ppo",
        help="Algorithm to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps",
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
        help="Output directory for models",
    )

    args = parser.parse_args()

    if args.algorithm == "ppo" or args.algorithm == "all":
        train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=f"{args.output_dir}/ppo",
        )

    if args.algorithm == "a2c" or args.algorithm == "all":
        train_a2c(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=f"{args.output_dir}/a2c",
        )

    if args.algorithm == "dqn" or args.algorithm == "all":
        train_dqn(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=f"{args.output_dir}/dqn",
        )


if __name__ == "__main__":
    main()
