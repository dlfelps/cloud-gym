"""Data models for cloud resources and tasks."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class VMType(Enum):
    """VM instance types with different resource profiles."""
    COMPUTE_OPTIMIZED = "compute"  # High CPU, moderate memory
    MEMORY_OPTIMIZED = "memory"    # High memory, moderate CPU
    BALANCED = "balanced"          # Equal CPU/memory ratio
    BUDGET = "budget"              # Lower specs, lower cost


class Priority(Enum):
    """Task priority levels."""
    LOW = 0      # Best-effort, interruptible
    MEDIUM = 1   # Standard business workloads
    HIGH = 2     # SLA-critical, penalties for violations


@dataclass
class VMConfig:
    """Configuration for a VM type."""
    vm_type: VMType
    cpu_cores: int
    memory_gb: float
    disk_gb: float
    bandwidth_mbps: float
    cost_per_hour: float
    power_watts: float


# Predefined VM configurations
VM_CONFIGS = {
    VMType.COMPUTE_OPTIMIZED: VMConfig(
        vm_type=VMType.COMPUTE_OPTIMIZED,
        cpu_cores=16,
        memory_gb=32,
        disk_gb=200,
        bandwidth_mbps=1000,
        cost_per_hour=0.80,
        power_watts=300
    ),
    VMType.MEMORY_OPTIMIZED: VMConfig(
        vm_type=VMType.MEMORY_OPTIMIZED,
        cpu_cores=8,
        memory_gb=64,
        disk_gb=200,
        bandwidth_mbps=1000,
        cost_per_hour=0.90,
        power_watts=280
    ),
    VMType.BALANCED: VMConfig(
        vm_type=VMType.BALANCED,
        cpu_cores=8,
        memory_gb=32,
        disk_gb=200,
        bandwidth_mbps=1000,
        cost_per_hour=0.60,
        power_watts=250
    ),
    VMType.BUDGET: VMConfig(
        vm_type=VMType.BUDGET,
        cpu_cores=4,
        memory_gb=16,
        disk_gb=100,
        bandwidth_mbps=500,
        cost_per_hour=0.30,
        power_watts=150
    ),
}


class VM:
    """Represents a virtual machine with resources."""

    def __init__(self, vm_id: int, vm_type: VMType, availability_zone: int):
        self.vm_id = vm_id
        self.vm_type = vm_type
        self.availability_zone = availability_zone
        self.config = VM_CONFIGS[vm_type]

        # Current available resources
        self.available_cpu = self.config.cpu_cores
        self.available_memory = self.config.memory_gb
        self.available_disk = self.config.disk_gb
        self.available_bandwidth = self.config.bandwidth_mbps

        # Running tasks
        self.running_tasks: list[int] = []

        # State
        self.is_operational = True
        self.time_to_failure: Optional[int] = None

    def can_allocate(self, task: 'Task') -> bool:
        """Check if VM has enough resources for the task."""
        return (
            self.is_operational and
            self.available_cpu >= task.cpu_required and
            self.available_memory >= task.memory_required and
            self.available_disk >= task.disk_required and
            self.available_bandwidth >= task.bandwidth_required
        )

    def allocate(self, task: 'Task') -> bool:
        """Allocate resources for a task. Returns True if successful."""
        if not self.can_allocate(task):
            return False

        self.available_cpu -= task.cpu_required
        self.available_memory -= task.memory_required
        self.available_disk -= task.disk_required
        self.available_bandwidth -= task.bandwidth_required
        self.running_tasks.append(task.task_id)
        return True

    def deallocate(self, task: 'Task'):
        """Release resources used by a task."""
        if task.task_id in self.running_tasks:
            self.available_cpu += task.cpu_required
            self.available_memory += task.memory_required
            self.available_disk += task.disk_required
            self.available_bandwidth += task.bandwidth_required
            self.running_tasks.remove(task.task_id)

    def get_utilization(self) -> dict:
        """Get current utilization percentages."""
        return {
            'cpu': 1.0 - (self.available_cpu / self.config.cpu_cores),
            'memory': 1.0 - (self.available_memory / self.config.memory_gb),
            'disk': 1.0 - (self.available_disk / self.config.disk_gb),
            'bandwidth': 1.0 - (self.available_bandwidth / self.config.bandwidth_mbps),
        }

    def reset(self):
        """Reset VM to initial state."""
        self.available_cpu = self.config.cpu_cores
        self.available_memory = self.config.memory_gb
        self.available_disk = self.config.disk_gb
        self.available_bandwidth = self.config.bandwidth_mbps
        self.running_tasks = []
        self.is_operational = True
        self.time_to_failure = None


@dataclass
class Task:
    """Represents a computational task to be scheduled."""
    task_id: int

    # Resource requirements
    cpu_required: float
    memory_required: float
    disk_required: float
    bandwidth_required: float

    # Duration
    estimated_duration: int  # in time steps
    actual_duration: int     # actual duration (may differ due to uncertainty)
    remaining_duration: int  # time steps remaining

    # Priority and deadlines
    priority: Priority
    deadline: Optional[int]  # absolute time step by which task must complete

    # State tracking
    arrival_time: int
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    assigned_vm: Optional[int] = None

    def is_overdue(self, current_time: int) -> bool:
        """Check if task has missed its deadline."""
        if self.deadline is None:
            return False
        return current_time > self.deadline

    def time_until_deadline(self, current_time: int) -> Optional[int]:
        """Time remaining until deadline."""
        if self.deadline is None:
            return None
        return max(0, self.deadline - current_time)

    def waiting_time(self, current_time: int) -> int:
        """Time spent waiting in queue."""
        if self.start_time is None:
            return current_time - self.arrival_time
        return self.start_time - self.arrival_time

    def response_time(self, current_time: int) -> Optional[int]:
        """Total time from arrival to completion."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


class TaskGenerator:
    """Generates tasks with specified distributions."""

    def __init__(
        self,
        rng: np.random.Generator,
        arrival_rate: float = 2.0,
        mean_duration: int = 10,
        duration_std: float = 3.0,
        priority_probs: tuple[float, float, float] = (0.5, 0.3, 0.2)
    ):
        self.rng = rng
        self.arrival_rate = arrival_rate
        self.mean_duration = mean_duration
        self.duration_std = duration_std
        self.priority_probs = priority_probs
        self.next_task_id = 0

    def generate_tasks(self, current_time: int) -> list[Task]:
        """Generate tasks arriving at current time step (Poisson process)."""
        n_tasks = self.rng.poisson(self.arrival_rate)
        tasks = []

        for _ in range(n_tasks):
            task = self._create_task(current_time)
            tasks.append(task)

        return tasks

    def _create_task(self, arrival_time: int) -> Task:
        """Create a single task with random attributes."""
        task_id = self.next_task_id
        self.next_task_id += 1

        # Sample priority
        priority = Priority(self.rng.choice([0, 1, 2], p=self.priority_probs))

        # Sample duration with uncertainty
        estimated_duration = max(1, int(self.rng.normal(self.mean_duration, self.duration_std)))
        # Actual duration varies (±20% from estimate)
        actual_duration = max(1, int(estimated_duration * self.rng.uniform(0.8, 1.2)))

        # Sample resource requirements (DISCRETE INTEGERS for discrete state space)
        # This makes the entire observation space discrete!
        base_cpu = float(self.rng.integers(1, 9))     # 1-8 cores (discrete)
        base_memory = float(self.rng.integers(2, 33)) # 2-32 GB (discrete)

        # High priority tasks tend to need more resources
        if priority == Priority.HIGH:
            base_cpu = min(16.0, base_cpu * 1.5)  # Scale up but cap at max
            base_memory = min(64.0, base_memory * 1.5)

        # Round to integers to maintain discrete space
        base_cpu = float(int(base_cpu))
        base_memory = float(int(base_memory))

        # Calculate deadline based on priority (AGGRESSIVE - accounts for queueing)
        # Key insight: deadline must account for potential queueing delays
        # Real-world: SLA is from arrival, not from start of execution
        deadline = None
        if priority == Priority.HIGH:
            # VERY tight: only 1.05x estimated (5% slack)
            # With ±20% duration variance + queueing, this will cause violations
            deadline = arrival_time + max(2, int(estimated_duration * 1.05))
        elif priority == Priority.MEDIUM:
            # Tight: 1.2x estimated (20% slack)
            # Still challenging with queueing delays
            deadline = arrival_time + max(3, int(estimated_duration * 1.2))
        else:  # LOW priority
            # Moderate: 2.0x estimated duration
            # Relaxed but still trackable
            deadline = arrival_time + max(5, int(estimated_duration * 2.0))

        return Task(
            task_id=task_id,
            cpu_required=base_cpu,
            memory_required=base_memory,
            disk_required=float(self.rng.integers(10, 101)),  # 10-100 GB (discrete)
            bandwidth_required=float(self.rng.integers(1, 11) * 50),  # 50-500 Mbps in steps of 50 (discrete)
            estimated_duration=estimated_duration,
            actual_duration=actual_duration,
            remaining_duration=actual_duration,
            priority=priority,
            deadline=deadline,
            arrival_time=arrival_time,
        )
