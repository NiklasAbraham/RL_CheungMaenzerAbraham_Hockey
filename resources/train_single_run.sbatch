#!/bin/bash

#SBATCH --job-name=hockey_single_run
# Job name

#SBATCH --partition=week
# Partition: day, week, or month

#SBATCH --gres=gpu:1
# Request 1 GPU

#SBATCH --mem=32G
# Request 32GB of memory (16 parallel environments + replay buffer need more memory)

#SBATCH --time=7-00:00:00
# Time limit: 7 days for week partition

#SBATCH --error=job.%J.err
# Error log file

#SBATCH --output=job.%J.out
# Output log file

#SBATCH --mail-type=ALL
# Email notifications: NONE, BEGIN, END, FAIL, REQUEUE, ALL

#SBATCH --mail-user=jannik.maenzer@student.uni-tuebingen.de
# Your email address

# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to project root
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Set number of parallel environments to match CPU cores
export NUM_ENVS=16

# Run the training script using singularity
# Since the container is read-only, we can't install packages into /venv
# Instead, we add the src directory to PYTHONPATH so Python can find rl_hockey
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && export NUM_ENVS=$NUM_ENVS && export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && python3 '$PROJECT_DIR/src/rl_hockey/common/training/train_single_run.py'"

# Alternative: Run with custom parameters by modifying train_single_run.py
# Or uncomment and modify the following line to pass parameters directly:
# singularity run \
#     --nv \
#     --bind "$PROJECT_DIR:$PROJECT_DIR" \
#     --pwd "$PROJECT_DIR" \
#     "$SINGULARITY_IMAGE" \
#     python3 -c "from rl_hockey.common.training.train_single_run import train_single_run; train_single_run('configs/curriculum_simple.json', base_output_dir='results/runs', run_name='cluster_run', num_envs=24, device='cuda:0')"
