#!/bin/bash

#SBATCH --job-name=archive_tournament
# Job name for archive tournament

#SBATCH --partition=week
# Partition: day, week, or month

#SBATCH --gres=gpu:2080ti:1
# Request 1 2080ti GPU

#SBATCH --mem=32G
# Request 32GB of memory

#SBATCH --time=7-00:00:00
# Time limit: 7 days

#SBATCH --error=job.%J.err
# Error log file

#SBATCH --output=job.%J.out
# Output log file

#SBATCH --mail-type=ALL
# Email notifications

#SBATCH --mail-user=blabla@student.uni-tuebingen.de
# Your email address

# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to project root
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Disable Python output buffering
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Archive Tournament Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Run round-robin tournament (edit CONFIG in run_archive_tournament.py)
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "
        source /venv/bin/activate && \
        export PYTHONUNBUFFERED=1 && \
        export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && \
        python3 -u '$PROJECT_DIR/src/rl_hockey/common/archive/run_archive_tournament.py'
    "

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
