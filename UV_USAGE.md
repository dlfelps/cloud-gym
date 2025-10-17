# Using uv with Cloud Resource Gym

This project is optimized for [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver written in Rust.

## Why uv?

- ⚡ **10-100x faster** than pip
- 🔒 **Automatic virtual environment** management
- 📦 **Reproducible builds** with lock files
- 🎯 **Zero configuration** for most use cases
- 🔄 **Seamless compatibility** with pip and PyPI

## Installation

### Install uv

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Using pip (if you prefer)
pip install uv
```

### Verify Installation

```bash
uv --version
```

## Quick Start

```bash
# Navigate to project
cd gym

# Run the demo (uv automatically installs dependencies!)
uv run main.py
```

That's it! `uv` will:
1. Create a virtual environment (`.venv/`)
2. Install all dependencies from `pyproject.toml`
3. Run your script

## Common Commands

### Running Scripts

```bash
# Quick demo
uv run main.py

# Run example
uv run examples/simple_example.py

# Use entry points (defined in pyproject.toml)
uv run cloud-gym-demo
uv run cloud-gym-eval --n-episodes 100
uv run cloud-gym-train --algorithm ppo
uv run cloud-gym-viz --results results/baseline_comparison.csv
```

### Managing Dependencies

```bash
# Add a new dependency
uv pip install scipy

# Add to dev dependencies
uv pip install --dev ipython

# Sync environment with pyproject.toml
uv pip sync

# List installed packages
uv pip list

# Show dependency tree
uv pip tree
```

### Working with Virtual Environments

```bash
# uv creates .venv automatically, but you can manage it manually:

# Activate virtual environment (if needed)
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows

# Deactivate
deactivate

# Remove virtual environment
rm -rf .venv
```

## Project-Specific Commands

### 1. Quick Demo

```bash
uv run main.py
```

Expected output: Single episode with Best Fit policy (~10 seconds)

### 2. Compare Baselines

```bash
uv run examples/simple_example.py
```

Expected output: Comparison of Random, First Fit, and Best Fit (~30 seconds)

### 3. Train RL Agents

```bash
# Train single algorithm
uv run cloud-gym-train --algorithm ppo --timesteps 50000

# Train all algorithms
uv run cloud-gym-train --algorithm all --timesteps 100000

# With custom output directory
uv run cloud-gym-train --algorithm ppo --timesteps 50000 --output-dir my_models
```

### 4. Evaluate Baselines

```bash
# Evaluate all heuristic baselines
uv run cloud-gym-eval --n-episodes 100 --seed 42

# Include trained RL models
uv run cloud-gym-eval --n-episodes 100 --model-dir models

# Custom output location
uv run cloud-gym-eval --n-episodes 50 --output-dir my_results
```

### 5. Generate Visualizations

```bash
# From default location
uv run cloud-gym-viz --results results/baseline_comparison.csv

# Custom paths
uv run cloud-gym-viz \
    --results my_results/baseline_comparison.csv \
    --output-dir my_results/plots
```

## Advanced Usage

### Running Python Interactively

```bash
# Start Python REPL with dependencies loaded
uv run python

# In the REPL:
>>> from cloud_resource_gym import CloudResourceEnv
>>> env = CloudResourceEnv()
>>> obs, info = env.reset()
```

### Jupyter Notebook Support

```bash
# Add jupyter to dependencies
uv pip install jupyter

# Launch notebook
uv run jupyter notebook

# Or use JupyterLab
uv pip install jupyterlab
uv run jupyter lab
```

### Using with Different Python Versions

```bash
# Specify Python version
uv venv --python 3.11
uv run --python 3.11 main.py

# Or use system Python
uv run --python-platform system main.py
```

### Lock File for Reproducibility

```bash
# Generate lock file (similar to requirements.txt)
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip sync requirements.lock
```

## Troubleshooting

### Issue: "uv: command not found"

**Solution:**
```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc

# Or install uv again
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: Import errors when running scripts

**Solution:**
```bash
# Force reinstall dependencies
rm -rf .venv
uv run main.py  # Will recreate .venv
```

### Issue: Slow first run

**Solution:** This is normal! uv is downloading and caching dependencies. Subsequent runs will be much faster.

### Issue: Need to use specific PyTorch version (CPU-only)

**Solution:**
```bash
# Install CPU-only PyTorch (smaller, faster for training on CPU)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Comparison: uv vs pip

| Task | pip | uv |
|------|-----|-----|
| Install dependencies | `pip install -e .` | `uv run main.py` (automatic) |
| Run script | `python main.py` | `uv run main.py` |
| Add package | `pip install package` | `uv pip install package` |
| Virtual environment | Manual (`python -m venv .venv`) | Automatic |
| Typical install time | 30-60 seconds | 3-5 seconds |
| Lock file | Manual (`pip freeze`) | `uv pip compile` |

## Entry Points Reference

The following entry points are configured in `pyproject.toml`:

```bash
cloud-gym-demo    # Quick demo (main.py)
cloud-gym-train   # Train RL agents
cloud-gym-eval    # Evaluate baselines
cloud-gym-viz     # Generate visualizations
```

Usage:
```bash
uv run cloud-gym-demo
uv run cloud-gym-train --help
uv run cloud-gym-eval --help
uv run cloud-gym-viz --help
```

## Tips for Best Performance

1. **First run**: Let uv download everything. Subsequent runs are 10-100x faster.
2. **Cache**: uv caches packages globally, so multiple projects share dependencies efficiently.
3. **Lock files**: Use `uv pip compile` for reproducible research experiments.
4. **Entry points**: Use `uv run cloud-gym-*` commands instead of full script paths.

## Migrating from pip

Already using pip? No problem!

```bash
# Your old workflow
python -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py

# New workflow with uv
uv run main.py  # That's it!
```

Everything still works with pip if you prefer:
```bash
pip install -e .
python main.py
```

## Google Colab Note

Google Colab doesn't have uv pre-installed, so use pip there:

```python
!pip install gymnasium numpy stable-baselines3 matplotlib pandas seaborn torch tqdm
```

Then run scripts normally:
```python
!python main.py
!python scripts/evaluate_baselines.py --n-episodes 100
```

## Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [uv Installation Guide](https://github.com/astral-sh/uv#installation)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Summary**: Use `uv run` instead of `python` for faster, more reliable execution! 🚀
