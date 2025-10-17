"""Heuristic baseline policies for resource allocation."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from cloud_resource_gym.models import VM, Task, Priority


class BasePolicy(ABC):
    """Base class for allocation policies."""

    def __init__(self, n_vms: int, seed: Optional[int] = None):
        self.n_vms = n_vms
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        """
        Select an action for the given task.

        Args:
            task: The task to allocate
            vms: List of available VMs
            action_mask: Mask of valid actions (1=valid, 0=invalid)
            current_time: Current time step

        Returns:
            Action index (VM id, reject, or defer)
        """
        pass

    def reset(self):
        """Reset policy state (if any)."""
        pass


class RandomPolicy(BasePolicy):
    """Randomly select from valid actions."""

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return self.n_vms  # Reject if no valid actions
        return self.rng.choice(valid_actions)


class RoundRobinPolicy(BasePolicy):
    """Assign tasks to VMs in round-robin fashion."""

    def __init__(self, n_vms: int, seed: Optional[int] = None):
        super().__init__(n_vms, seed)
        self.current_vm = 0

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        # Try to assign to current VM in rotation
        attempts = 0
        while attempts < self.n_vms:
            if action_mask[self.current_vm] == 1:
                action = self.current_vm
                self.current_vm = (self.current_vm + 1) % self.n_vms
                return action
            self.current_vm = (self.current_vm + 1) % self.n_vms
            attempts += 1

        # No VM available, reject
        return self.n_vms

    def reset(self):
        self.current_vm = 0


class FirstFitPolicy(BasePolicy):
    """Assign to first VM that has sufficient resources."""

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        for i in range(self.n_vms):
            if action_mask[i] == 1:
                return i
        # No VM available, reject
        return self.n_vms


class BestFitPolicy(BasePolicy):
    """Assign to VM with least remaining capacity after allocation."""

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        best_vm = None
        min_remaining = float('inf')

        for i in range(self.n_vms):
            if action_mask[i] == 1:
                vm = vms[i]
                # Calculate remaining capacity after allocation (normalized)
                remaining_cpu = (vm.available_cpu - task.cpu_required) / vm.config.cpu_cores
                remaining_mem = (vm.available_memory - task.memory_required) / vm.config.memory_gb

                # Use minimum of CPU and memory as bottleneck resource
                remaining = min(remaining_cpu, remaining_mem)

                if remaining < min_remaining:
                    min_remaining = remaining
                    best_vm = i

        if best_vm is not None:
            return best_vm

        # No VM available, reject
        return self.n_vms


class WorstFitPolicy(BasePolicy):
    """Assign to VM with most remaining capacity after allocation."""

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        worst_vm = None
        max_remaining = -float('inf')

        for i in range(self.n_vms):
            if action_mask[i] == 1:
                vm = vms[i]
                # Calculate remaining capacity after allocation (normalized)
                remaining_cpu = (vm.available_cpu - task.cpu_required) / vm.config.cpu_cores
                remaining_mem = (vm.available_memory - task.memory_required) / vm.config.memory_gb

                # Use minimum of CPU and memory as bottleneck resource
                remaining = min(remaining_cpu, remaining_mem)

                if remaining > max_remaining:
                    max_remaining = remaining
                    worst_vm = i

        if worst_vm is not None:
            return worst_vm

        # No VM available, reject
        return self.n_vms


class PriorityBestFitPolicy(BasePolicy):
    """
    Best Fit policy that considers task priority.
    High priority tasks get preferential allocation to better VMs.
    """

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        best_vm = None
        best_score = -float('inf')

        for i in range(self.n_vms):
            if action_mask[i] == 1:
                vm = vms[i]

                # Calculate fit score
                remaining_cpu = (vm.available_cpu - task.cpu_required) / vm.config.cpu_cores
                remaining_mem = (vm.available_memory - task.memory_required) / vm.config.memory_gb
                remaining = min(remaining_cpu, remaining_mem)

                # Score: prefer tighter fit, but boost for high-priority tasks
                fit_score = -remaining  # Negative because we want tight fit

                # High priority tasks prefer VMs with more total capacity
                if task.priority == Priority.HIGH:
                    total_capacity = (vm.config.cpu_cores + vm.config.memory_gb) / 2
                    fit_score += total_capacity * 0.1

                if fit_score > best_score:
                    best_score = fit_score
                    best_vm = i

        if best_vm is not None:
            return best_vm

        # No VM available, reject
        return self.n_vms


class EarliestDeadlineFirstPolicy(BasePolicy):
    """
    Process tasks in deadline order, then use Best Fit for allocation.
    Note: This policy works best when combined with task sorting,
    but here we just prioritize based on deadline urgency.
    """

    def select_action(
        self,
        task: Task,
        vms: list[VM],
        action_mask: np.ndarray,
        current_time: int
    ) -> int:
        # If task has tight deadline, be more aggressive about allocation
        time_to_deadline = task.time_until_deadline(current_time)

        if time_to_deadline is not None and time_to_deadline < task.estimated_duration * 1.5:
            # Urgent task - use any available VM (First Fit)
            for i in range(self.n_vms):
                if action_mask[i] == 1:
                    return i
        else:
            # Not urgent - use Best Fit for efficiency
            best_vm = None
            min_remaining = float('inf')

            for i in range(self.n_vms):
                if action_mask[i] == 1:
                    vm = vms[i]
                    remaining_cpu = (vm.available_cpu - task.cpu_required) / vm.config.cpu_cores
                    remaining_mem = (vm.available_memory - task.memory_required) / vm.config.memory_gb
                    remaining = min(remaining_cpu, remaining_mem)

                    if remaining < min_remaining:
                        min_remaining = remaining
                        best_vm = i

            if best_vm is not None:
                return best_vm

        # No VM available, reject
        return self.n_vms


class PolicyWrapper:
    """
    Wrapper to use heuristic policies with Gymnasium environments.
    Converts observation dict to policy inputs.
    """

    def __init__(self, policy: BasePolicy):
        self.policy = policy

    def predict(self, observation: dict, deterministic: bool = True) -> tuple[int, None]:
        """
        Predict action (compatible with stable-baselines3 interface).

        Returns:
            (action, None) - None is for state (not used in non-recurrent policies)
        """
        # Extract relevant information from observation
        # Note: In practice, you'd need to reconstruct task and VM info from observation
        # This is a simplified version

        # For now, just use action mask and select first valid action
        action_mask = observation['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            # No valid actions, reject
            return observation['action_mask'].shape[0] - 2, None

        # Simple heuristic: select first valid action
        return int(valid_actions[0]), None

    def reset(self):
        """Reset policy state."""
        self.policy.reset()
