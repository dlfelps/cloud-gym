"""
Cloud Resource Allocation Gym - Main Entry Point

Quick start script to demonstrate the environment and baseline policies.

Usage:
    uv run main.py                    # Using uv (recommended)
    python main.py                    # Using pip
    uv run cloud-gym-demo            # Using entry point
"""

from cloud_resource_gym import CloudResourceEnv
from cloud_resource_gym.policies import BestFitPolicy


def main():
    """Run a quick demonstration of the cloud resource allocation environment."""
    print("="*70)
    print("Cloud Resource Allocation Gym - Quick Demo")
    print("="*70)

    # Create environment
    print("\n📦 Creating environment...")
    env = CloudResourceEnv(
        n_vms=10,
        n_availability_zones=3,
        max_episode_steps=50,  # Short demo
        arrival_rate=2.0,
        seed=42,
    )
    print(f"✓ Environment created with {env.n_vms} VMs across {env.n_availability_zones} zones")

    # Create policy
    print("\n🤖 Using Best Fit policy...")
    policy = BestFitPolicy(n_vms=env.n_vms, seed=42)

    # Run one episode
    print("\n▶️  Running episode...")
    obs, info = env.reset()
    episode_reward = 0.0
    done = False
    step_count = 0

    while not done:
        # Get current task and select action
        if env.current_task_index < len(env.pending_tasks):
            task = env.pending_tasks[env.current_task_index]
            action = policy.select_action(
                task=task,
                vms=env.vms,
                action_mask=obs['action_mask'],
                current_time=env.current_time,
            )
        else:
            action = env.n_vms  # No task, reject action

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        step_count += 1

    # Print results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"\n🎯 Episode completed in {step_count} steps")
    print(f"💰 Total reward: {episode_reward:.2f}")
    print(f"\n📈 Metrics:")
    print(f"   • Total tasks: {info['metrics']['total_tasks']}")
    print(f"   • Completed: {info['metrics']['completed_tasks']}")
    print(f"   • Rejected: {info['metrics']['rejected_tasks']}")
    print(f"   • SLA violations: {info['metrics']['sla_violations']}")

    completion_rate = info['metrics']['completed_tasks'] / max(info['metrics']['total_tasks'], 1)
    print(f"   • Completion rate: {completion_rate:.1%}")

    if info['metrics']['completed_tasks'] > 0:
        sla_satisfaction = 1.0 - (info['metrics']['sla_violations'] / info['metrics']['completed_tasks'])
        print(f"   • SLA satisfaction: {sla_satisfaction:.1%}")

    print(f"\n💵 Costs:")
    print(f"   • Energy: ${info['metrics']['total_energy_cost']:.2f}")
    print(f"   • VM rental: ${info['metrics']['total_vm_cost']:.2f}")
    print(f"   • Total: ${info['metrics']['total_energy_cost'] + info['metrics']['total_vm_cost']:.2f}")

    print("\n" + "="*70)
    print("✨ Demo complete!")
    print("\n📚 Next steps:")
    print("   • See QUICKSTART.md for detailed guide")
    print("   • Run: python examples/simple_example.py")
    print("   • Run: python scripts/evaluate_baselines.py")
    print("="*70)


if __name__ == "__main__":
    main()
