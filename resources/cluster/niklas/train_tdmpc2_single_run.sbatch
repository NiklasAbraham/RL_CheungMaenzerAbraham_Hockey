#!/bin/bash

#SBATCH --job-name=hockey_tdmpc2
# Job name

#SBATCH --partition=week
# Partition: day, week, or month

#SBATCH --gres=gpu:2080ti:1
# Request 1 2080ti GPU (CUDA capability 7.5, supports torch.compile)
# Alternative options: gpu:A4000:1 (CUDA 8.6) or gpu:1080ti:1 (CUDA 6.1, no torch.compile)

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

#SBATCH --mail-user=blabla@student.uni-tuebingen.de
# Your email address

# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to project root (config paths are in train_tdmpc2.py)
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Disable Python output buffering for immediate logging in batch jobs
export PYTHONUNBUFFERED=1

USER_PACKAGES_DIR="$PROJECT_DIR/.user_packages"
mkdir -p "$USER_PACKAGES_DIR"

echo "=========================================="
echo "TDMPC2 Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && pip install --target '$USER_PACKAGES_DIR' psutil > /dev/null 2>&1 && export PYTHONUNBUFFERED=1 && export PYTHONPATH='$PROJECT_DIR/src:$USER_PACKAGES_DIR:\$PYTHONPATH' && python3 -u '$PROJECT_DIR/src/rl_hockey/common/training/train_tdmpc2.py'"

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
