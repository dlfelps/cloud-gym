"""Simple example demonstrating the cloud resource allocation environment."""

from cloud_resource_gym import CloudResourceEnv
from cloud_resource_gym.policies import BestFitPolicy, FirstFitPolicy, RandomPolicy


def run_episode(env, policy, policy_name: str):
    """Run one episode with the given policy."""
    obs, info = env.reset()
    policy.reset()

    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        # Get current task
        if env.current_task_index < len(env.pending_tasks):
            task = env.pending_tasks[env.current_task_index]
            action = policy.select_action(
                task=task,
                vms=env.vms,
                action_mask=obs['action_mask'],
                current_time=env.current_time,
            )
        else:
            # No task to process
            action = env.n_vms  # Reject action

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"Policy: {policy_name}")
    print(f"{'='*60}")
    print(f"Episode Steps: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"\nMetrics:")
    print(f"  Total Tasks: {info['metrics']['total_tasks']}")
    print(f"  Completed Tasks: {info['metrics']['completed_tasks']}")
    print(f"  Rejected Tasks: {info['metrics']['rejected_tasks']}")
    print(f"  SLA Violations: {info['metrics']['sla_violations']}")
    print(f"  Completion Rate: {info['metrics']['completed_tasks'] / max(info['metrics']['total_tasks'], 1):.2%}")

    if info['metrics']['completed_tasks'] > 0:
        sla_satisfaction = 1.0 - (info['metrics']['sla_violations'] / info['metrics']['completed_tasks'])
        print(f"  SLA Satisfaction Rate: {sla_satisfaction:.2%}")

    print(f"  Total Energy Cost: ${info['metrics']['total_energy_cost']:.2f}")
    print(f"  Total VM Cost: ${info['metrics']['total_vm_cost']:.2f}")
    print(f"  Total Cost: ${info['metrics']['total_energy_cost'] + info['metrics']['total_vm_cost']:.2f}")

    return total_reward, info['metrics']


def main():
    """Run simple comparison of baseline policies."""
    print("Cloud Resource Allocation Environment - Simple Example")
    print("="*60)

    # Create environment
    env = CloudResourceEnv(
        n_vms=10,
        n_availability_zones=3,
        max_episode_steps=100,  # Shorter for demo
        arrival_rate=2.0,
        vm_failure_rate=0.001,
        seed=42,
    )

    print(f"\nEnvironment Configuration:")
    print(f"  Number of VMs: {env.n_vms}")
    print(f"  Availability Zones: {env.n_availability_zones}")
    print(f"  Max Episode Steps: {env.max_episode_steps}")
    print(f"  Task Arrival Rate: {env.arrival_rate}")

    # Test different policies
    policies = [
        (RandomPolicy(n_vms=env.n_vms, seed=42), "Random"),
        (FirstFitPolicy(n_vms=env.n_vms, seed=42), "First Fit"),
        (BestFitPolicy(n_vms=env.n_vms, seed=42), "Best Fit"),
    ]

    results = {}
    for policy, name in policies:
        reward, metrics = run_episode(env, policy, name)
        results[name] = {'reward': reward, 'metrics': metrics}

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Policy':<20} {'Reward':>12} {'Completed':>12} {'SLA Viol':>12}")
    print("-"*60)

    for name, data in results.items():
        print(f"{name:<20} {data['reward']:>12.2f} "
              f"{data['metrics']['completed_tasks']:>12} "
              f"{data['metrics']['sla_violations']:>12}")

    print("\nBest Policy by Reward:")
    best_policy = max(results.items(), key=lambda x: x[1]['reward'])
    print(f"  {best_policy[0]} with reward {best_policy[1]['reward']:.2f}")


if __name__ == "__main__":
    main()
