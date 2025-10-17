# Quick Start Guide - Cloud Resource Allocation Gym

## Overview

This project provides a complete reinforcement learning environment for cloud resource allocation with:
- ✅ Custom Gymnasium environment with realistic cloud dynamics
- ✅ 10 baseline policies (7 heuristic + 3 RL algorithms)
- ✅ Comprehensive evaluation framework
- ✅ Visualization tools
- ✅ Python 3.10+ compatible (works in Google Colab with Python 3.12)

## Installation (5 minutes)

### Option 1: Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project
cd gym

# uv automatically manages dependencies - no installation needed!
# Just use `uv run` commands below
```

### Option 2: Using pip

```bash
# Navigate to project
cd gym

# Install dependencies
pip install -e .
```

### Option 3: Google Colab

```python
# In a new Colab cell
!pip install gymnasium numpy stable-baselines3 matplotlib pandas seaborn torch tqdm
```

Then upload your project files or clone from git.

## Running the Baseline Comparison (15 minutes)

### Step 1: Test the Environment

```bash
# Using uv (recommended)
uv run main.py

# Or run the example
uv run examples/simple_example.py

# Using pip
python examples/simple_example.py
```

This will run a quick comparison of Random, First Fit, and Best Fit policies.

**Expected output:**
```
Policy: Best Fit
Completed Tasks: ~300-400
SLA Violations: ~5-15
Reward: ~150-250
```

### Step 2: Train RL Agents (Optional, 30-60 minutes)

```bash
# Using uv (recommended)
uv run cloud-gym-train --algorithm all --timesteps 50000 --seed 0

# Using pip
python scripts/train_rl_baseline.py --algorithm all --timesteps 50000 --seed 0
```

Or skip this step and just evaluate heuristic baselines.

### Step 3: Evaluate All Baselines

```bash
# Using uv (recommended) - evaluates heuristic baselines (fast, ~5 minutes)
uv run cloud-gym-eval --n-episodes 100 --seed 42

# Using pip
python scripts/evaluate_baselines.py --n-episodes 100 --seed 42

# This will create:
# - results/baseline_comparison.csv
# - results/baseline_comparison.json
```

### Step 4: Generate Visualizations

```bash
# Using uv (recommended)
uv run cloud-gym-viz --results results/baseline_comparison.csv

# Using pip
python scripts/visualize_results.py --results results/baseline_comparison.csv
```

This creates 6 plots in `results/plots/`:
1. Mean reward comparison (bar chart)
2. Multi-metric comparison (4 subplots)
3. Radar chart (multi-objective performance)
4. Reward distribution (box plots)
5. Performance heatmap
6. Summary table

## Understanding the Results

### Key Metrics to Look At

1. **Mean Reward**: Overall performance score
2. **Completion Rate**: How many tasks were successfully completed
3. **SLA Satisfaction Rate**: % of tasks meeting their deadlines
4. **Total Cost**: Energy + VM rental costs

### Expected Baseline Rankings (Heuristic Policies)

From our research, you should expect:

**Best Performers:**
- Priority Best Fit (best overall)
- Best Fit (good resource efficiency)
- Earliest Deadline First (good SLA satisfaction)

**Middle Performers:**
- First Fit (decent, fast)
- Round Robin (fair but not optimal)

**Worst Performers:**
- Worst Fit (poor utilization)
- Random (baseline for comparison)

### RL Algorithms (if trained)

PPO should outperform heuristics after sufficient training (~100k timesteps).

## Customizing the Environment

### Change Workload Characteristics

Edit `scripts/evaluate_baselines.py`:

```python
env = CloudResourceEnv(
    n_vms=20,              # More VMs
    arrival_rate=3.5,      # Busier workload
    vm_failure_rate=0.01,  # More failures
)
```

### Change Reward Weights

```python
env = CloudResourceEnv(
    reward_weights={
        'utilization': 2.0,      # Emphasize efficiency
        'sla_violation': -20.0,  # Heavy penalty for SLA misses
        'energy_cost': -0.05,
        'queue_length': -0.2,
        'completion': 1.5,
    }
)
```

## Common Issues & Solutions

### Issue 1: Import Errors

```
ModuleNotFoundError: No module named 'cloud_resource_gym'
```

**Solution:** Install the package in editable mode:
```bash
pip install -e .
```

### Issue 2: Torch/Stable-Baselines3 Issues

```
Error loading PyTorch model
```

**Solution:** Install CPU-only PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Matplotlib Display Issues in Colab

```
Plots not showing
```

**Solution:** Use inline backend:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

## Next Steps

### 1. Experiment with Different Scenarios

Try these variations:
- **High Load**: `arrival_rate=5.0`
- **Unreliable Infrastructure**: `vm_failure_rate=0.01`
- **Tight SLAs**: Modify task generator to create tighter deadlines
- **Heterogeneous Workloads**: Modify task distributions

### 2. Implement Your Own Policy

Create a new file `cloud_resource_gym/policies/my_policy.py`:

```python
from cloud_resource_gym.policies.heuristic import BasePolicy

class MySmartPolicy(BasePolicy):
    def select_action(self, task, vms, action_mask, current_time):
        # Your brilliant algorithm here!

        # Example: Prefer VMs in zone 0 for high-priority tasks
        if task.priority == Priority.HIGH:
            for i, vm in enumerate(vms):
                if action_mask[i] == 1 and vm.availability_zone == 0:
                    return i

        # Fallback to best fit
        best_vm = None
        min_remaining = float('inf')
        for i in range(self.n_vms):
            if action_mask[i] == 1:
                vm = vms[i]
                remaining = min(
                    vm.available_cpu / vm.config.cpu_cores,
                    vm.available_memory / vm.config.memory_gb
                )
                if remaining < min_remaining:
                    min_remaining = remaining
                    best_vm = i

        return best_vm if best_vm is not None else self.n_vms
```

Then add it to the evaluation script!

### 3. Train Better RL Agents

Hyperparameter tuning for PPO:

```python
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,    # Try: 1e-4, 3e-4, 1e-3
    n_steps=2048,          # Try: 512, 1024, 2048, 4096
    batch_size=64,         # Try: 32, 64, 128
    n_epochs=10,           # Try: 5, 10, 20
    gamma=0.99,            # Discount factor
    gae_lambda=0.95,       # GAE parameter
    clip_range=0.2,        # PPO clip range
    ent_coef=0.01,         # Entropy coefficient
)
```

### 4. Add New Features

Ideas for extensions:
- **Multi-agent RL**: Each VM has its own agent
- **Transfer learning**: Train on one workload, test on another
- **Adaptive policies**: Policy changes based on current load
- **Cost-aware scheduling**: Dynamic VM scaling
- **Network topology**: Add latency between zones
- **Container bin packing**: Multi-dimensional packing constraints

## File Structure Reference

```
gym/
├── cloud_resource_gym/          # Main package
│   ├── models.py                # VM, Task, TaskGenerator
│   ├── envs/
│   │   └── cloud_env.py         # Gymnasium environment
│   └── policies/
│       └── heuristic.py         # Baseline policies
├── scripts/
│   ├── train_rl_baseline.py     # Train RL agents
│   ├── evaluate_baselines.py    # Evaluate all baselines
│   └── visualize_results.py     # Generate plots
├── examples/
│   └── simple_example.py        # Quick demo
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
└── pyproject.toml              # Dependencies
```

## Getting Help

If you encounter issues:
1. **Using uv**: Check dependencies with `uv pip list | grep -E "gymnasium|stable-baselines3|torch"`
2. **Using pip**: Check dependencies with `pip list | grep -E "gymnasium|stable-baselines3|torch"`
3. Verify Python version: `python --version` (should be 3.10+)
4. Run the simple example first: `uv run main.py` or `python main.py`
5. Check file imports: Make sure you're in the `gym` directory

## Performance Expectations

On a typical laptop/Colab instance:
- Simple example: ~10 seconds
- Heuristic baseline evaluation (100 episodes): ~5 minutes
- RL training (50k timesteps): ~30-60 minutes
- Visualization generation: ~30 seconds

## Success Criteria

You've successfully established baselines if:
- ✅ All 7 heuristic policies run without errors
- ✅ Evaluation produces CSV with metrics
- ✅ Visualizations show clear differences between policies
- ✅ Priority Best Fit or Best Fit outperforms Random by >50%
- ✅ Completion rates are >70% for best policies

## What's Next?

Now that you have baselines established, you can:
1. **Analyze gaps**: Where do all policies struggle?
2. **Design improvements**: Use insights to create better policies
3. **Train custom RL agents**: With domain-specific reward shaping
4. **Publish results**: Compare your novel approach to these baselines!

Good luck with your research! 🚀
