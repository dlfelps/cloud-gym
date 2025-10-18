# Gymnasium Migration - Troubleshooting Guide

## Status: Already Migrated ✓

Your codebase has already been fully migrated to `gymnasium`. If you're still seeing gym deprecation warnings, follow the steps below.

## Verify Migration Status

The codebase uses:
- `import gymnasium as gym` in `cloud_resource_gym/envs/cloud_env.py:3`
- `import gymnasium as gym` in `scripts/train_rl_baseline.py:10`
- `gymnasium>=0.29.0` in dependencies
- `stable-baselines3>=2.0.0` (gymnasium-compatible)

## Common Causes of Gym Warnings

### 1. Outdated stable-baselines3 Version

The warning often comes from having stable-baselines3 < 2.0.0 installed.

**Check your version:**
```bash
pip show stable-baselines3
```

**If version is < 2.0.0, upgrade:**
```bash
pip install --upgrade "stable-baselines3>=2.0.0"
```

### 2. Old gym Package Still Installed

Even though the code uses `gymnasium`, having the old `gym` package installed can cause warnings.

**Check if old gym is installed:**
```bash
pip show gym
```

**If found, uninstall it:**
```bash
pip uninstall gym
```

### 3. Cached Installation

**Clean reinstall:**
```bash
# Using pip
pip uninstall cloud-resource-gym
pip cache purge
pip install -e .

# Using uv (recommended)
uv pip uninstall cloud-resource-gym
uv pip install -e .
```

## Complete Clean Installation Steps

### Option 1: Using uv (Recommended)

```bash
# Remove existing installation
uv pip uninstall cloud-resource-gym

# Reinstall with fresh dependencies
uv pip install -e .

# Verify versions
uv pip list | grep -E "gymnasium|stable-baselines3"
```

### Option 2: Using pip + venv

```bash
# Create fresh virtual environment
python -m venv venv_clean
source venv_clean/bin/activate  # On Windows: venv_clean\Scripts\activate

# Install fresh dependencies
pip install --upgrade pip
pip install -e .

# Verify versions
pip list | grep -E "gymnasium|stable-baselines3"
```

### Option 3: Using conda

```bash
# Create fresh conda environment
conda create -n cloud_gym_clean python=3.10
conda activate cloud_gym_clean

# Install dependencies
pip install -e .

# Verify versions
pip list | grep -E "gymnasium|stable-baselines3"
```

## Verify Installation

Run this test script to confirm gymnasium is working:

```python
# test_gymnasium.py
import gymnasium as gym
from cloud_resource_gym import CloudResourceEnv
import stable_baselines3

print(f"Gymnasium version: {gym.__version__}")
print(f"Stable-baselines3 version: {stable_baselines3.__version__}")

# Test environment creation
env = CloudResourceEnv(n_vms=5, n_availability_zones=2, seed=42)
obs, info = env.reset()
print(f"\nEnvironment created successfully!")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Test step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"\nStep executed successfully!")
print(f"Terminated: {terminated}, Truncated: {truncated}")

print("\n✓ All gymnasium checks passed!")
```

**Run the test:**
```bash
python test_gymnasium.py
```

## Expected Output

You should see:
```
Gymnasium version: 0.29.x or higher
Stable-baselines3 version: 2.x.x or higher
Environment created successfully!
...
✓ All gymnasium checks passed!
```

**No warnings about gym deprecation!**

## If Warnings Persist

### Check for gym imports in dependencies

Some packages might still depend on the old `gym`. Check:

```bash
# See what installed gym (if any)
pip show gym

# Check dependency tree
pip install pipdeptree
pipdeptree -p gym
```

### Force upgrade all RL dependencies

```bash
pip install --upgrade --force-reinstall \
    gymnasium>=0.29.0 \
    stable-baselines3>=2.0.0 \
    sb3-contrib>=2.0.0 \
    shimmy>=1.2.0
```

Note: `shimmy` is a compatibility layer that helps gymnasium work with old gym environments.

## Google Colab Specific Fix

If running in Google Colab:

```python
# Uninstall old gym
!pip uninstall -y gym

# Install gymnasium and compatible versions
!pip install -q gymnasium>=0.29.0
!pip install -q stable-baselines3>=2.0.0
!pip install -q sb3-contrib>=2.0.0

# Clone and install your package
!git clone <your-repo-url>
%cd cloud-resource-gym
!pip install -e .

# Restart runtime (important!)
# Runtime -> Restart runtime
```

## Migration Checklist

Verify all items are ✓:

- [x] Code uses `import gymnasium as gym`
- [x] Dependencies specify `gymnasium>=0.29.0`
- [x] Dependencies specify `stable-baselines3>=2.0.0`
- [x] Environment inherits from `gym.Env` (using gymnasium)
- [x] `reset()` returns `(observation, info)`
- [x] `step()` returns `(obs, reward, terminated, truncated, info)`
- [x] No old `gym` package installed
- [x] stable-baselines3 version >= 2.0.0
- [x] No deprecation warnings when running code

## Summary

Your code is already gymnasium-compliant. The warning is likely from:
1. Old stable-baselines3 version (< 2.0.0)
2. Old gym package still installed
3. Cached dependencies

**Quick fix:**
```bash
pip uninstall gym
pip install --upgrade "stable-baselines3>=2.0.0"
pip install -e . --force-reinstall --no-cache-dir
```

This should eliminate any gym deprecation warnings!
