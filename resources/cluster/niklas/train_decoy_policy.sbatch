#!/bin/bash

#SBATCH --job-name=decoy_policy
# Job name

#SBATCH --partition=week
# Partition: day, week, or month

#SBATCH --gres=gpu:2080ti:1
# Request 1 GPU (decoy training is lightweight; 2080ti sufficient)

#SBATCH --mem=16G
# Decoy training uses single env and replay buffer; 16G usually enough

#SBATCH --time=1-00:00:00
# Time limit: 1 day (decoy training typically shorter than full TDMPC2)

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

# Config is hardcoded in train_decoy_policy.py (CONFIG dict at top). Edit that file to change target/opponents.

# Disable Python output buffering for immediate logging in batch jobs
export PYTHONUNBUFFERED=1

USER_PACKAGES_DIR="$PROJECT_DIR/.user_packages"
mkdir -p "$USER_PACKAGES_DIR"

# Run decoy policy training (uses CONFIG in script)
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && pip install --target '$USER_PACKAGES_DIR' psutil tqdm > /dev/null 2>&1 && export PYTHONUNBUFFERED=1 && export PYTHONPATH='$PROJECT_DIR/src:$USER_PACKAGES_DIR:\$PYTHONPATH' && python3 -u '$PROJECT_DIR/src/rl_hockey/scripts/train_decoy_policy.py'"
