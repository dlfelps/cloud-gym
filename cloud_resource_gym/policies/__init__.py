"""Baseline policies for cloud resource allocation."""

from cloud_resource_gym.policies.heuristic import (
    RandomPolicy,
    RoundRobinPolicy,
    FirstFitPolicy,
    BestFitPolicy,
    WorstFitPolicy,
    PriorityBestFitPolicy,
    EarliestDeadlineFirstPolicy,
)

__all__ = [
    "RandomPolicy",
    "RoundRobinPolicy",
    "FirstFitPolicy",
    "BestFitPolicy",
    "WorstFitPolicy",
    "PriorityBestFitPolicy",
    "EarliestDeadlineFirstPolicy",
]
