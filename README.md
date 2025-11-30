# RL Hockey Project - Cheung Maenzer Abraham

This repository contains the reinforcement learning implementation for the Hockey environment.

## Conda Environment Setup

### Create and Activate Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create --name rl-hockey python=3.10

# Activate the environment
conda activate rl-hockey
```

### Complete Installation Command

If you prefer to install everything via pip after creating the conda environment:

```bash
# Create and activate environment
conda create --name rl-hockey python=3.10
conda activate rl-hockey

# Install swig (required for box2d-py to build)
conda install swig -c conda-forge

# Install all dependencies via pip
# PyTorch will automatically detect and use CUDA if available
pip install gymnasium numpy box2d-py jupyter matplotlib torch torchvision torchaudio pygame wandb tqdm tensorboardx ipympl

# Install laser-hockey environment
pip install git+https://github.com/martius-lab/laser-hockey-env.git
```

### Export Current Environment

To freeze the current environment versions into files:

```bash
# Activate the environment first
conda activate rl-hockey

# Export conda environment to environment.yml
conda env export > environment.yml

# Export pip packages to requirements.txt
pip freeze > requirements.txt
```

