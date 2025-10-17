# Quick Command Reference

## Installation

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install -e .
```

## Running Scripts

### Quick Demo
```bash
uv run main.py                    # Using uv
python main.py                    # Using pip
uv run cloud-gym-demo            # Using entry point
```

### Simple Example (Compare 3 policies)
```bash
uv run examples/simple_example.py
python examples/simple_example.py
```

## Training RL Agents

### Train Single Algorithm
```bash
# PPO (recommended)
uv run cloud-gym-train --algorithm ppo --timesteps 100000

# A2C (faster)
uv run cloud-gym-train --algorithm a2c --timesteps 100000

# DQN (value-based)
uv run cloud-gym-train --algorithm dqn --timesteps 100000
```

### Train All Algorithms
```bash
uv run cloud-gym-train --algorithm all --timesteps 100000 --seed 0
```

### Custom Training
```bash
uv run scripts/train_rl_baseline.py \
    --algorithm ppo \
    --timesteps 50000 \
    --seed 42 \
    --output-dir my_models
```

## Evaluation

### Evaluate Heuristic Baselines Only
```bash
uv run cloud-gym-eval --n-episodes 100 --seed 42
```

### Evaluate All Baselines (Including RL)
```bash
uv run cloud-gym-eval \
    --n-episodes 100 \
    --seed 42 \
    --model-dir models \
    --output-dir results
```

### Full Command
```bash
uv run scripts/evaluate_baselines.py \
    --n-episodes 100 \
    --seed 42 \
    --output-dir results \
    --model-dir models
```

## Visualization

### Generate All Plots
```bash
uv run cloud-gym-viz --results results/baseline_comparison.csv
```

### Custom Output Directory
```bash
uv run cloud-gym-viz \
    --results results/baseline_comparison.csv \
    --output-dir my_plots
```

### Full Command
```bash
uv run scripts/visualize_results.py \
    --results results/baseline_comparison.csv \
    --output-dir results/plots
```

## Complete Workflow

```bash
# 1. Quick test
uv run main.py

# 2. Train RL agents (optional, takes time)
uv run cloud-gym-train --algorithm all --timesteps 100000

# 3. Evaluate all baselines
uv run cloud-gym-eval --n-episodes 100 --seed 42

# 4. Generate visualizations
uv run cloud-gym-viz --results results/baseline_comparison.csv

# 5. View results
ls results/
ls results/plots/
```

## Entry Points

These are defined in `pyproject.toml`:

| Entry Point | Script | Description |
|-------------|--------|-------------|
| `cloud-gym-demo` | `main.py` | Quick demo with Best Fit |
| `cloud-gym-train` | `scripts/train_rl_baseline.py` | Train RL agents |
| `cloud-gym-eval` | `scripts/evaluate_baselines.py` | Evaluate baselines |
| `cloud-gym-viz` | `scripts/visualize_results.py` | Generate plots |

## Common Options

### Training Options
- `--algorithm {ppo,a2c,dqn,all}` - Algorithm to train
- `--timesteps INT` - Total training timesteps (default: 100000)
- `--seed INT` - Random seed (default: 0)
- `--output-dir PATH` - Model save directory (default: "models")

### Evaluation Options
- `--n-episodes INT` - Number of evaluation episodes (default: 100)
- `--seed INT` - Random seed (default: 42)
- `--output-dir PATH` - Results directory (default: "results")
- `--model-dir PATH` - Trained models directory (default: "models")

### Visualization Options
- `--results PATH` - Path to results CSV/JSON (required)
- `--output-dir PATH` - Output directory for plots (default: "results/plots")

## Troubleshooting

### Check Dependencies
```bash
uv pip list | grep -E "gymnasium|stable-baselines3|torch"
```

### Reinstall Environment
```bash
rm -rf .venv
uv run main.py  # Will recreate
```

### Check Python Version
```bash
python --version  # Should be 3.10+
```

### Install CPU-Only PyTorch (Faster)
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## File Locations

- **Trained Models**: `models/{algorithm}/`
- **Evaluation Results**: `results/baseline_comparison.{csv,json}`
- **Visualizations**: `results/plots/*.png`
- **TensorBoard Logs**: `models/{algorithm}/tensorboard/`

## For Google Colab

```python
# Install dependencies
!pip install gymnasium numpy stable-baselines3 matplotlib pandas seaborn torch tqdm

# Run commands
!python main.py
!python scripts/evaluate_baselines.py --n-episodes 100
!python scripts/visualize_results.py --results results/baseline_comparison.csv

# View images
from IPython.display import Image, display
display(Image('results/plots/reward_comparison.png'))
```

## Quick Reference Card

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run demo
uv run main.py

# Train agents
uv run cloud-gym-train --algorithm all --timesteps 100000

# Evaluate
uv run cloud-gym-eval --n-episodes 100

# Visualize
uv run cloud-gym-viz --results results/baseline_comparison.csv
```

---

**Pro Tip**: Use `uv run` for automatic dependency management! 🚀

For detailed uv usage, see [UV_USAGE.md](UV_USAGE.md)
For full documentation, see [README.md](README.md)
For quick start guide, see [QUICKSTART.md](QUICKSTART.md)
