#!/bin/bash
# Script to sync project files to TCML cluster

# Configuration
SERVER="tcml-login1"
# SSH config should handle the full hostname mapping
REMOTE_DIR="~/RL_CheungMaenzerAbraham_Hockey"
# Get project root directory (one level up from resources folder where this script is located)
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing project to ${SERVER}:${REMOTE_DIR}..."
echo ""

# Use rsync to efficiently sync files, excluding unnecessary directories
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.pytest_cache' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.ipynb' \
    --exclude='results/' \
    --exclude='*.log' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='models/' \
    --exclude='runs/' \
    --exclude='results/' \
    --exclude='singularity_build/' \
    --exclude='old/' \
    --exclude='.conda' \
    --exclude='env/' \
    --exclude='venv/' \
    "${LOCAL_DIR}/" "${SERVER}:${REMOTE_DIR}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "Sync completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. SSH to the cluster: ssh ${SERVER}"
    echo "2. Navigate to: cd ${REMOTE_DIR}"
    echo "3. Run setup: bash resources/cluster_setup.sh"
    echo "4. Submit the job: sbatch resources/train_single_run.sbatch"
else
    echo ""
    echo "Sync failed. Please check the error messages above."
fi
