# Google Colab Setup Guide

This guide walks you through setting up and training RL agents for the Cloud Resource Allocation Gym in Google Colab.

## Quick Start

### 1. Open Google Colab

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### 2. Clone the Repository

```python
# Clone the repository
!git clone https://github.com/yourusername/cloud-resource-gym.git
%cd cloud-resource-gym
```

### 3. Install Dependencies

```python
# Install required packages
!pip install -q gymnasium>=0.29.0
!pip install -q stable-baselines3>=2.0.0
!pip install -q sb3-contrib>=2.0.0
!pip install -q torch>=2.0.0
!pip install -q numpy>=1.24.0
!pip install -q pandas>=2.0.0
!pip install -q matplotlib>=3.7.0
!pip install -q seaborn>=0.12.0
!pip install -q tqdm>=4.65.0

# Install the gym environment
!pip install -e .
```

### 4. Enable GPU (Optional but Recommended)

For faster training:
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 GPU is free)
3. Click **Save**

Verify GPU is available:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Training Examples

### Example 1: Quick Demo

Run the basic demo to verify installation:

```python
from cloud_resource_gym import CloudResourceEnv
from cloud_resource_gym.policies import BestFitPolicy

# Create environment
env = CloudResourceEnv(
    n_vms=10,
    n_availability_zones=3,
    max_episode_steps=50,
    seed=42
)

# Run with baseline policy
policy = BestFitPolicy(n_vms=env.n_vms, seed=42)
obs, info = env.reset()
episode_reward = 0.0

for _ in range(50):
    if env.current_task_index < len(env.pending_tasks):
        task = env.pending_tasks[env.current_task_index]
        action = policy.select_action(
            task=task,
            vms=env.vms,
            action_mask=obs['action_mask'],
            current_time=env.current_time
        )
    else:
        action = env.n_vms

    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    if terminated or truncated:
        break

print(f"Episode reward: {episode_reward:.2f}")
print(f"Completed tasks: {info['metrics']['completed_tasks']}")
```

### Example 2: Train PPO Agent

Train a PPO agent with progress tracking:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from cloud_resource_gym import CloudResourceEnv
import os

# Create output directory
os.makedirs("colab_models", exist_ok=True)
os.makedirs("colab_logs", exist_ok=True)

# Create vectorized environment
env = make_vec_env(
    lambda: CloudResourceEnv(
        n_vms=10,
        n_availability_zones=3,
        max_episode_steps=100,
        seed=42
    ),
    n_envs=4
)

# Create evaluation environment
eval_env = CloudResourceEnv(
    n_vms=10,
    n_availability_zones=3,
    max_episode_steps=100,
    seed=123
)

# Setup evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./colab_models/",
    log_path="./colab_logs/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True
)

# Create PPO agent
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./colab_logs/tensorboard/"
)

# Train the agent
print("Starting training...")
model.learn(
    total_timesteps=100_000,
    callback=eval_callback,
    progress_bar=True
)

# Save final model
model.save("colab_models/ppo_final")
print("Training complete!")
```

### Example 3: Evaluate Trained Model

```python
from stable_baselines3 import PPO
from cloud_resource_gym import CloudResourceEnv
import numpy as np

# Load trained model
model = PPO.load("colab_models/best_model")

# Create environment
env = CloudResourceEnv(
    n_vms=10,
    n_availability_zones=3,
    max_episode_steps=100,
    seed=456
)

# Evaluate over multiple episodes
n_episodes = 10
episode_rewards = []
completion_rates = []
sla_rates = []

for episode in range(n_episodes):
    obs, info = env.reset()
    episode_reward = 0.0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    episode_rewards.append(episode_reward)

    # Calculate metrics
    total_tasks = info['metrics']['total_tasks']
    completed = info['metrics']['completed_tasks']
    violations = info['metrics']['sla_violations']

    completion_rates.append(completed / max(total_tasks, 1))
    sla_rates.append(1.0 - violations / max(completed, 1))

# Print results
print(f"\n{'='*60}")
print(f"Evaluation Results ({n_episodes} episodes)")
print(f"{'='*60}")
print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Completion rate: {np.mean(completion_rates):.1%} ± {np.std(completion_rates):.1%}")
print(f"SLA satisfaction: {np.mean(sla_rates):.1%} ± {np.std(sla_rates):.1%}")
```

### Example 4: Visualize Training Progress

```python
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Load training logs
log_dir = "./colab_logs/"
results = load_results(log_dir)

# Plot learning curve
plt.figure(figsize=(12, 5))

# Plot 1: Episode rewards
plt.subplot(1, 2, 1)
x, y = ts2xy(results, 'timesteps')
plt.plot(x, y)
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('Training Progress')
plt.grid(True)

# Plot 2: Rolling average
plt.subplot(1, 2, 2)
rolling_mean = pd.Series(y).rolling(window=10).mean()
plt.plot(x, rolling_mean)
plt.xlabel('Timesteps')
plt.ylabel('Average Reward (10 episodes)')
plt.title('Smoothed Training Progress')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Example 5: Compare Multiple Algorithms

```python
from stable_baselines3 import PPO, A2C, DQN
from cloud_resource_gym import CloudResourceEnv
import numpy as np
import time

def train_and_evaluate(algorithm_class, algorithm_name, timesteps=50_000):
    """Train and evaluate an algorithm."""
    print(f"\nTraining {algorithm_name}...")

    # Create environment
    env = CloudResourceEnv(n_vms=10, n_availability_zones=3, seed=42)

    # Create model
    model = algorithm_class(
        "MultiInputPolicy",
        env,
        verbose=0
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    train_time = time.time() - start_time

    # Evaluate
    eval_env = CloudResourceEnv(n_vms=10, n_availability_zones=3, seed=999)
    episode_rewards = []

    for _ in range(5):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    return {
        'algorithm': algorithm_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'train_time': train_time
    }

# Compare algorithms
results = []
results.append(train_and_evaluate(PPO, "PPO"))
results.append(train_and_evaluate(A2C, "A2C"))

# Print comparison
print(f"\n{'='*70}")
print(f"Algorithm Comparison")
print(f"{'='*70}")
for result in results:
    print(f"{result['algorithm']:10s} | "
          f"Reward: {result['mean_reward']:7.2f} ± {result['std_reward']:5.2f} | "
          f"Time: {result['train_time']:6.1f}s")
```

## Tips for Colab Training

### 1. Save Models to Google Drive

Mount Google Drive to save models persistently:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save models to Drive
model.save('/content/drive/MyDrive/cloud_gym_models/ppo_model')
```

### 2. Monitor Training with TensorBoard

```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir ./colab_logs/tensorboard/
```

### 3. Handle Runtime Disconnections

Use checkpoints to resume training:

```python
import os

checkpoint_path = "colab_models/checkpoint.zip"

# Check if checkpoint exists
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    model = PPO.load(checkpoint_path, env=env)
else:
    print("Creating new model...")
    model = PPO("MultiInputPolicy", env, verbose=1)

# Train with periodic saves
for i in range(5):
    model.learn(total_timesteps=20_000, reset_num_timesteps=False)
    model.save(checkpoint_path)
    print(f"Checkpoint {i+1} saved")
```

### 4. Optimize for Free GPU Time

Colab free tier has usage limits. Optimize training:

```python
# Use smaller networks for faster training
policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1
)
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or number of parallel environments

```python
model = PPO(
    "MultiInputPolicy",
    env,
    batch_size=32,  # Smaller batch size
    n_steps=1024,   # Fewer steps
    verbose=1
)
```

### Issue: Slow Training

**Solution**: Ensure GPU is enabled and reduce environment complexity

```python
# Verify GPU usage
import torch
print(f"Using device: {model.device}")

# Use fewer VMs for faster simulation
env = CloudResourceEnv(n_vms=5, n_availability_zones=2)
```

### Issue: Module Not Found

**Solution**: Reinstall the package

```python
!pip uninstall -y cloud-resource-gym
!pip install -e .
```

## Next Steps

- Experiment with different hyperparameters
- Try advanced algorithms (SAC, TD3, RecurrentPPO)
- Implement custom reward shaping
- Compare with baseline policies
- Scale to larger environments

## Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Google Colab Tips](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Example Colab Notebook

A complete ready-to-run notebook is available at:
`examples/cloud_gym_colab_training.ipynb`

Simply upload this notebook to Google Colab and run all cells!
