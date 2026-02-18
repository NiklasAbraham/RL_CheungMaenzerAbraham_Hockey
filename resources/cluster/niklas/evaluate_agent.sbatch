#!/bin/bash

#SBATCH --job-name=hockey_evaluate
# Job name

#SBATCH --partition=week
# Partition: day, week, or month

#SBATCH --gres=gpu:2080ti:1
# Request 1 2080ti GPU (CUDA capability 7.5, supports torch.compile)
# Alternative options: gpu:A4000:1 (CUDA 8.6) or gpu:1080ti:1 (CUDA 6.1, no torch.compile)

#SBATCH --mem=32G
# Request 32GB of memory (parallel evaluation processes need memory)

#SBATCH --time=7-00:00:00
# Time limit: 7 days for week partition

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

# Disable Python output buffering for immediate logging in batch jobs
export PYTHONUNBUFFERED=1

# Run the evaluation script using singularity
# Since the container is read-only, we can't install packages into /venv
# Install missing packages to a user-writable location if needed
USER_PACKAGES_DIR="$PROJECT_DIR/.user_packages"
mkdir -p "$USER_PACKAGES_DIR"

# Run the evaluation script using singularity
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && pip install --target '$USER_PACKAGES_DIR' psutil > /dev/null 2>&1 && export PYTHONUNBUFFERED=1 && export PYTHONPATH='$PROJECT_DIR/src:$USER_PACKAGES_DIR:\$PYTHONPATH' && python3 -u '$PROJECT_DIR/src/rl_hockey/common/evaluation/agent_evaluator.py'"

# Alternative: Run with custom parameters by modifying agent_evaluator.py
# Or uncomment and modify the following line to pass parameters directly:
# singularity exec \
#     --nv \
#     --bind "$PROJECT_DIR:$PROJECT_DIR" \
#     --pwd "$PROJECT_DIR" \
#     "$SINGULARITY_IMAGE" \
#     /bin/bash -c "source /venv/bin/activate && export PYTHONUNBUFFERED=1 && export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && python3 -u -c \"from rl_hockey.common.evaluation.agent_evaluator import evaluate_agent; print(evaluate_agent(agent_path='results/tdmpc2_runs/2026-01-25_11-40-04/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_cfce4de1_20260123_210009_ep007000.pt', config_path=None, num_games=500, weak_opponent=False, max_steps=500, num_parallel=8))\""
