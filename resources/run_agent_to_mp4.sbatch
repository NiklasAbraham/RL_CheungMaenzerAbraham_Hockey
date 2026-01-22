#!/bin/bash

#SBATCH --job-name=hockey_agent_mp4
# Job name

#SBATCH --partition=week
# Partition: day, week, or month

#SBATCH --gres=gpu:2080ti:1
# Request 1 A4000 GPU (CUDA capability 8.6, supports torch.compile)
# Alternative options: gpu:A4000:1 (CUDA 8.6) or gpu:1080ti:1 (CUDA 6.1, no torch.compile)

#SBATCH --mem=32G
# Request 32GB of memory

#SBATCH --time=12:00:00
# Time limit: 12 hours should be enough for 25 games + video encoding

#SBATCH --error=job.%J.err
# Error log file

#SBATCH --output=job.%J.out
# Output log file

#SBATCH --mail-type=ALL
# Email notifications: NONE, BEGIN, END, FAIL, REQUEUE, ALL

#SBATCH --mail-user=blabla@student.uni-tuebingen.de
# Your email address

# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to project root
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Set MODEL_PATH to the most recent model file
# Update this path to use a different model if needed
export MODEL_PATH="$PROJECT_DIR/results/tdmpc2_runs/2026-01-22_08-42-36/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_cfce4de1_20260121_161543_ep002100.pt"

# Run the agent video script using singularity
# Since the container is read-only, we can't install packages into /venv
# Instead, we add the src directory to PYTHONPATH so Python can find rl_hockey
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && export PROJECT_DIR='$PROJECT_DIR' && export MODEL_PATH='$MODEL_PATH' && export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && python3 '$PROJECT_DIR/src/rl_hockey/scripts/run_agent_to_mp4.py'"
