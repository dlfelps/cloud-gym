#!/usr/bin/env python3
"""Monitor hyperparameter tuning progress."""

import argparse
from pathlib import Path
import time
import subprocess


def count_completed_variants(model_dir: Path):
    """Count how many variants have completed training."""
    a2c_completed = len(list(model_dir.glob("a2c_*/a2c_*_final.zip")))
    dqn_completed = len(list(model_dir.glob("dqn_*/dqn_*_final.zip")))
    return a2c_completed, dqn_completed


def get_latest_tensorboard_data(model_dir: Path):
    """Get latest training metrics from tensorboard logs."""
    # This is a placeholder - in practice you'd parse tensorboard event files
    # For now, just return directory info
    a2c_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("a2c_")])
    dqn_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("dqn_")])
    return a2c_dirs, dqn_dirs


def main():
    parser = argparse.ArgumentParser(description="Monitor hyperparameter tuning progress")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/hyperparameter_tuning",
        help="Model directory to monitor",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval in seconds",
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    print("=" * 80)
    print("MONITORING HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Update interval: {args.interval}s")
    print("=" * 80)
    print("\nPress Ctrl+C to stop monitoring\n")

    try:
        while True:
            a2c_completed, dqn_completed = count_completed_variants(model_dir)
            total_completed = a2c_completed + dqn_completed

            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"A2C: {a2c_completed}/8 | "
                  f"DQN: {dqn_completed}/9 | "
                  f"Total: {total_completed}/17", end='', flush=True)

            if total_completed == 17:
                print("\n\n[SUCCESS] All variants completed!")
                print("\nNext steps:")
                print("  1. Evaluate results: uv run scripts/evaluate_tuning_results.py")
                print("  2. Visualize: uv run scripts/visualize_tuning_results.py")
                print("  3. View TensorBoard: tensorboard --logdir models/hyperparameter_tuning")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"\nCurrent progress: A2C: {a2c_completed}/8, DQN: {dqn_completed}/9")


if __name__ == "__main__":
    main()
