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

# Player 1 (recorded agent): SAC reference bot
export MODEL_PATH="$PROJECT_DIR/resources/reference_bots/SAC/run_lr1e03_bs256_h128_128_128_4c1f51eb_20260111_140638_vec24.pt"

# Player 2 (opponent): your TD-MPC2 agent. Path may be deeper on the cluster, e.g.:
#   results/tdmpc2_runs/DATE/run_name/models/MODEL.pt
# Update to your latest run and correct nesting (models/ can be one or two levels deep).
export OPPONENT_MODEL_PATH="$PROJECT_DIR/results/tdmpc2_runs/2026-01-25_11-40-04/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_cfce4de1_20260123_210009_ep009200.pt"

# Use loaded agent as opponent (SAC vs TD-MPC2)
export OPPONENT_TYPE="agent"

# Video output: under project root so it is found regardless of job cwd depth
export VIDEO_OUTPUT_DIR="$PROJECT_DIR/videos"

# Run the agent video script using singularity
# Since the container is read-only, we can't install packages into /venv
# Instead, we add the src directory to PYTHONPATH so Python can find rl_hockey
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && export PROJECT_DIR='$PROJECT_DIR' && export MODEL_PATH='$MODEL_PATH' && export OPPONENT_MODEL_PATH='$OPPONENT_MODEL_PATH' && export OPPONENT_TYPE='$OPPONENT_TYPE' && export VIDEO_OUTPUT_DIR='$VIDEO_OUTPUT_DIR' && export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && python3 '$PROJECT_DIR/src/rl_hockey/scripts/run_agent_to_mp4.py'"
