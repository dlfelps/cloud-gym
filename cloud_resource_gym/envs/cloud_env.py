"""Cloud Resource Allocation Gymnasium Environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Any

from cloud_resource_gym.models import (
    VM, Task, TaskGenerator, VMType, Priority, VM_CONFIGS
)


class CloudResourceEnv(gym.Env):
    """
    Gymnasium environment for cloud resource allocation.

    The agent must allocate incoming tasks to heterogeneous VMs while optimizing
    resource utilization, SLA satisfaction, and operational costs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_vms: int = 10,
        n_availability_zones: int = 3,
        max_episode_steps: int = 200,
        arrival_rate: float = 2.0,
        vm_failure_rate: float = 0.001,
        reward_weights: Optional[dict] = None,
        priority_distribution: Optional[tuple[float, float, float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the cloud resource allocation environment.

        Args:
            n_vms: Number of VMs in the cluster
            n_availability_zones: Number of availability zones
            max_episode_steps: Maximum time steps per episode
            arrival_rate: Poisson arrival rate for tasks
            vm_failure_rate: Probability of VM failure per time step
            reward_weights: Weights for multi-objective reward
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.n_vms = n_vms
        self.n_availability_zones = n_availability_zones
        self.max_episode_steps = max_episode_steps
        self.arrival_rate = arrival_rate
        self.vm_failure_rate = vm_failure_rate

        # Reward weights
        self.reward_weights = reward_weights or {
            'utilization': 1.0,
            'sla_violation': -10.0,
            'energy_cost': -0.01,
            'queue_length': -0.1,
            'completion': 1.0,
        }

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Initialize VMs
        self.vms: list[VM] = []
        self._initialize_vms()

        # Task management
        # Default: (0.5, 0.3, 0.2) = 50% LOW, 30% MEDIUM, 20% HIGH
        # Challenging: (0.2, 0.3, 0.5) = 20% LOW, 30% MEDIUM, 50% HIGH (more tight deadlines)
        priority_probs = priority_distribution or (0.5, 0.3, 0.2)
        self.task_generator = TaskGenerator(
            rng=self.rng,
            arrival_rate=arrival_rate,
            priority_probs=priority_probs,
        )
        self.pending_tasks: list[Task] = []
        self.running_tasks: dict[int, Task] = {}  # task_id -> Task
        self.completed_tasks: list[Task] = []
        self.rejected_tasks: list[Task] = []

        # Episode state
        self.current_time = 0
        self.current_task_index = 0  # For sequential task processing

        # Define action space
        # Actions: [0, n_vms-1] = assign to VM, n_vms = reject, n_vms+1 = defer
        self.action_space = spaces.Discrete(n_vms + 2)

        # Define observation space
        self.observation_space = self._create_observation_space()

        # Metrics tracking
        self.episode_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'rejected_tasks': 0,
            'sla_violations': 0,
            'tasks_with_deadlines': 0,
            'deadline_met': 0,
            'total_energy_cost': 0.0,
            'total_vm_cost': 0.0,
        }

    def _initialize_vms(self):
        """Initialize heterogeneous VMs across availability zones."""
        self.vms = []
        vm_types = list(VMType)

        for i in range(self.n_vms):
            vm_type = vm_types[i % len(vm_types)]
            availability_zone = i % self.n_availability_zones
            vm = VM(vm_id=i, vm_type=vm_type, availability_zone=availability_zone)
            self.vms.append(vm)

    def _create_observation_space(self) -> spaces.Dict:
        """Create the observation space."""
        # Global state
        global_space = spaces.Box(
            low=0, high=np.inf,
            shape=(3,),  # [current_time, n_pending_tasks, n_running_tasks]
            dtype=np.float32
        )

        # Per-VM state: [avail_cpu, avail_mem, avail_disk, avail_bw, n_tasks, is_operational, zone]
        vm_space = spaces.Box(
            low=0, high=np.inf,
            shape=(self.n_vms, 7),
            dtype=np.float32
        )

        # Current task being processed (if any)
        # [cpu_req, mem_req, disk_req, bw_req, est_duration, priority, time_to_deadline]
        task_space = spaces.Box(
            low=0, high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        # Action mask (which actions are valid)
        action_mask_space = spaces.Box(
            low=0, high=1,
            shape=(self.action_space.n,),
            dtype=np.int8
        )

        return spaces.Dict({
            'global': global_space,
            'vms': vm_space,
            'current_task': task_space,
            'action_mask': action_mask_space,
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.task_generator.rng = self.rng

        # Reset VMs
        for vm in self.vms:
            vm.reset()

        # Reset task queues
        self.pending_tasks = []
        self.running_tasks = {}
        self.completed_tasks = []
        self.rejected_tasks = []

        # Reset episode state
        self.current_time = 0
        self.current_task_index = 0
        self.task_generator.next_task_id = 0

        # Reset metrics
        self.episode_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'rejected_tasks': 0,
            'sla_violations': 0,
            'tasks_with_deadlines': 0,
            'deadline_met': 0,
            'total_energy_cost': 0.0,
            'total_vm_cost': 0.0,
        }

        # Generate initial tasks
        initial_tasks = self.task_generator.generate_tasks(self.current_time)
        self.pending_tasks.extend(initial_tasks)
        self.episode_metrics['total_tasks'] += len(initial_tasks)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one time step in the environment.

        Args:
            action: Action to take (VM to assign, reject, or defer)

        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0

        # Process the current task (if any)
        if self.current_task_index < len(self.pending_tasks):
            task = self.pending_tasks[self.current_task_index]
            task_reward = self._process_action(action, task)
            reward += task_reward

            # Move to next task or advance time
            self.current_task_index += 1

        # If all pending tasks processed, advance time
        if self.current_task_index >= len(self.pending_tasks):
            time_step_reward = self._advance_time()
            reward += time_step_reward

        # Check termination
        terminated = self.current_time >= self.max_episode_steps
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _process_action(self, action: int, task: Task) -> float:
        """Process the action for the current task."""
        reward = 0.0

        if action < self.n_vms:
            # Assign to VM
            vm = self.vms[action]
            if vm.allocate(task):
                task.assigned_vm = vm.vm_id
                task.start_time = self.current_time
                self.running_tasks[task.task_id] = task
                reward += 0.5  # Small reward for successful allocation
            else:
                # Invalid action (VM doesn't have capacity)
                reward -= 1.0
                self.rejected_tasks.append(task)
                self.episode_metrics['rejected_tasks'] += 1

        elif action == self.n_vms:
            # Reject task
            self.rejected_tasks.append(task)
            self.episode_metrics['rejected_tasks'] += 1
            # VERY STRONG penalty based on priority (prevents 44% rejection equilibrium)
            # Rejection must be MUCH WORSE than attempting and risking SLA violation
            # Math: With these penalties, attempting (even with 50% failure rate) is better
            penalty = {
                Priority.LOW: -5.0,      # Was -2.0 (2.5x increase)
                Priority.MEDIUM: -15.0,  # Was -5.0 (3x increase)
                Priority.HIGH: -25.0     # Was -10.0 (2.5x increase)
            }
            reward += penalty[task.priority]

        elif action == self.n_vms + 1:
            # Defer task - keep in pending queue
            # Small penalty for delaying
            reward -= 0.01
            return reward  # Don't increment task index

        return reward

    def _advance_time(self) -> float:
        """Advance time by one step and update all entities."""
        self.current_time += 1
        reward = 0.0

        # Update running tasks
        completed_task_ids = []
        for task_id, task in self.running_tasks.items():
            task.remaining_duration -= 1
            if task.remaining_duration <= 0:
                # Task completed
                task.completion_time = self.current_time
                completed_task_ids.append(task_id)
                self.completed_tasks.append(task)
                self.episode_metrics['completed_tasks'] += 1

                # Deallocate resources
                vm = self.vms[task.assigned_vm]
                vm.deallocate(task)

                # Track deadline metrics
                if task.deadline is not None:
                    self.episode_metrics['tasks_with_deadlines'] += 1
                    if self.current_time <= task.deadline:
                        self.episode_metrics['deadline_met'] += 1

                # Reward based on priority and SLA
                if task.deadline and self.current_time > task.deadline:
                    # SLA violation
                    self.episode_metrics['sla_violations'] += 1
                    reward += self.reward_weights['sla_violation']
                else:
                    # Successful completion
                    # Increased rewards to make attempting more attractive than rejecting
                    priority_bonus = {
                        Priority.LOW: 1.0,      # Was 0.5 (2x increase)
                        Priority.MEDIUM: 3.0,   # Was 1.0 (3x increase)
                        Priority.HIGH: 8.0      # Was 2.0 (4x increase)
                    }
                    reward += self.reward_weights['completion'] * priority_bonus[task.priority]

        # Remove completed tasks
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]

        # Check for SLA violations in running tasks
        for task in self.running_tasks.values():
            if task.is_overdue(self.current_time):
                reward += self.reward_weights['sla_violation'] * 0.1  # Ongoing penalty

        # Calculate resource utilization reward
        total_utilization = 0.0
        for vm in self.vms:
            if vm.is_operational:
                util = vm.get_utilization()
                avg_util = np.mean([util['cpu'], util['memory']])
                total_utilization += avg_util
        avg_cluster_utilization = total_utilization / len(self.vms)
        reward += self.reward_weights['utilization'] * avg_cluster_utilization

        # Energy and cost penalties
        # FIXED: Charge based on VMs with running tasks (idle VMs use less power)
        # VMs with tasks: full power, idle VMs: 20% standby power
        energy_cost = 0.0
        vm_cost = 0.0
        for vm in self.vms:
            if vm.is_operational:
                if len(vm.running_tasks) > 0:
                    # VM actively running tasks: full cost
                    energy_cost += vm.config.power_watts / 1000.0
                    vm_cost += vm.config.cost_per_hour / 60.0
                else:
                    # VM idle: only standby power (20% of full)
                    energy_cost += (vm.config.power_watts * 0.2) / 1000.0
                    vm_cost += (vm.config.cost_per_hour * 0.2) / 60.0

        self.episode_metrics['total_energy_cost'] += energy_cost
        self.episode_metrics['total_vm_cost'] += vm_cost
        reward += self.reward_weights['energy_cost'] * (energy_cost + vm_cost)

        # Queue length penalty
        queue_penalty = len(self.pending_tasks) * self.reward_weights['queue_length']
        reward += queue_penalty

        # Handle VM failures
        self._handle_vm_failures()

        # Generate new tasks
        new_tasks = self.task_generator.generate_tasks(self.current_time)
        self.pending_tasks = new_tasks  # Replace with new batch
        self.current_task_index = 0
        self.episode_metrics['total_tasks'] += len(new_tasks)

        return reward

    def _handle_vm_failures(self):
        """Simulate VM failures based on failure rate."""
        for vm in self.vms:
            if vm.is_operational and self.rng.random() < self.vm_failure_rate:
                # VM failed
                vm.is_operational = False

                # Migrate running tasks (simplified: just fail them)
                failed_task_ids = vm.running_tasks.copy()
                for task_id in failed_task_ids:
                    if task_id in self.running_tasks:
                        task = self.running_tasks[task_id]
                        # Re-queue the task
                        task.assigned_vm = None
                        task.start_time = None
                        task.remaining_duration = task.actual_duration
                        self.pending_tasks.append(task)
                        del self.running_tasks[task_id]

                vm.running_tasks = []

            elif not vm.is_operational:
                # Recovery with some probability (simplified)
                if self.rng.random() < 0.1:  # 10% recovery chance per step
                    vm.is_operational = True
                    vm.reset()

    def _get_observation(self) -> dict:
        """Construct the current observation."""
        # Global state
        global_obs = np.array([
            self.current_time,
            len(self.pending_tasks),
            len(self.running_tasks),
        ], dtype=np.float32)

        # VM states
        vm_obs = np.zeros((self.n_vms, 7), dtype=np.float32)
        for i, vm in enumerate(self.vms):
            vm_obs[i] = [
                vm.available_cpu,
                vm.available_memory,
                vm.available_disk,
                vm.available_bandwidth,
                len(vm.running_tasks),
                1.0 if vm.is_operational else 0.0,
                vm.availability_zone,
            ]

        # Current task
        if self.current_task_index < len(self.pending_tasks):
            task = self.pending_tasks[self.current_task_index]
            time_to_deadline = task.time_until_deadline(self.current_time)
            if time_to_deadline is None:
                time_to_deadline = 999  # Large number for no deadline
            task_obs = np.array([
                task.cpu_required,
                task.memory_required,
                task.disk_required,
                task.bandwidth_required,
                task.estimated_duration,
                task.priority.value,
                time_to_deadline,
            ], dtype=np.float32)
        else:
            # No task to process
            task_obs = np.zeros(7, dtype=np.float32)

        # Action mask
        action_mask = self._get_action_mask()

        return {
            'global': global_obs,
            'vms': vm_obs,
            'current_task': task_obs,
            'action_mask': action_mask,
        }

    def _get_action_mask(self) -> np.ndarray:
        """Generate mask of valid actions."""
        mask = np.zeros(self.action_space.n, dtype=np.int8)

        if self.current_task_index < len(self.pending_tasks):
            task = self.pending_tasks[self.current_task_index]

            # Check which VMs can accept the task
            for i, vm in enumerate(self.vms):
                if vm.can_allocate(task):
                    mask[i] = 1

            # Reject and defer are always valid
            mask[self.n_vms] = 1      # Reject
            mask[self.n_vms + 1] = 1  # Defer
        else:
            # No task to process, all actions invalid (shouldn't happen in practice)
            pass

        return mask

    def _get_info(self) -> dict:
        """Get additional information."""
        return {
            'time': self.current_time,
            'metrics': self.episode_metrics.copy(),
            'n_pending': len(self.pending_tasks),
            'n_running': len(self.running_tasks),
            'n_completed': len(self.completed_tasks),
        }

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            print(f"\n=== Time Step: {self.current_time} ===")
            print(f"Pending Tasks: {len(self.pending_tasks)}")
            print(f"Running Tasks: {len(self.running_tasks)}")
            print(f"Completed: {self.episode_metrics['completed_tasks']}")
            print(f"SLA Violations: {self.episode_metrics['sla_violations']}")

            print("\nVM Status:")
            for vm in self.vms:
                util = vm.get_utilization()
                status = "UP" if vm.is_operational else "DOWN"
                print(f"  VM {vm.vm_id} ({vm.vm_type.value}): "
                      f"CPU {util['cpu']:.1%} | MEM {util['memory']:.1%} | {status}")
