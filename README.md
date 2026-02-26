# RL Hockey Project - Cheung Maenzer Abraham

This repository contains the reinforcement learning implementation for the Hockey environment.

1. [Conda Environment Setup](#conda-environment-setup)
2. [Developing](#developing)
3. [Running the Agents](#running-the-agents)
4. [Author Contributions](#author-contributions)

## Conda Environment Setup

### Create and Activate Conda Environment from YAML

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate rl-hockey
```

### Update Conda Environment

```bash
# Update the conda environment from environment.yml
conda env update -f environment.yml --prune
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

# Update the current conda env
conda env update -f environment.yml
```

## Developing

To set up the development environment, install the package in editable mode:

```bash
pip install -e .
```

Use absolute imports everywhere in the codebase to avoid import issues, e.g., `from rl_hockey.sac import SAC`.

## Running the Agents

### TD-MPC

### SAC
To run the SAC agent, execute the `run_sac.py` script located in `src/rl_hockey/sac/`:

```bash
python src/rl_hockey/sac/run_sac.py
```

The corresponding model checkpoint and the configuration file are located in the `models/sac/` directory.

### TD3
To run the TD3 agent, execute the `run_td3.py` script located in `src/rl_hockey/td3/`:

```bash
python src/rl_hockey/td3/run_td3.py
```

The corresponding model checkpoint and the configuration file are located in the `models/td3/` directory.

### REDQ
To run the REDQ-TD3 agent, execute the `run_redq.py` script located in `src/rl_hockey/REDQ/`:

```bash
python src/rl_hockey/td3/run_redq.py
```

The corresponding model checkpoint and the configuration file are located in the `models/redqtd3/` directory.

## Author Contributions

Niklas Abraham:
- TODO

Jannik Mänzer:
- SAC & Extensions
- Archive & Self-play Infrastructure

Ansel Cheung:
- TD3
- PER
- REDQ-TD3
