#!/bin/bash
# Script to test/validate the sbatch file configuration
# Run this on the cluster to check your configuration before submitting

echo "=== Testing sbatch Configuration ==="
echo ""

PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
echo "✓ Project directory exists: $PROJECT_DIR"

# Check if container exists
CONTAINER="$PROJECT_DIR/singularity_build/rl_hockey.simg"
if [ ! -f "$CONTAINER" ]; then
    echo "WARNING: Container not found: $CONTAINER"
    echo "  You need to build the container first: bash resources/cluster_setup.sh"
else
    echo "✓ Container exists: $CONTAINER"
fi

# Check if config file exists
CONFIG="$PROJECT_DIR/configs/curriculum_simple.json"
if [ ! -f "$CONFIG" ]; then
    echo "WARNING: Config file not found: $CONFIG"
else
    echo "✓ Config file exists: $CONFIG"
fi

# Check if train_single_run.py exists
SCRIPT="$PROJECT_DIR/src/rl_hockey/common/training/train_single_run.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Training script not found: $SCRIPT"
    exit 1
fi
echo "✓ Training script exists: $SCRIPT"

# Display resource requirements from sbatch file
echo ""
echo "=== Resource Requirements ==="
grep "^#SBATCH --" "$PROJECT_DIR/resources/train_single_run.sbatch" | grep -v "^#.*$" | while read line; do
    echo "  $line"
done

# Check if singularity is available
if command -v singularity &> /dev/null; then
    echo ""
    echo "✓ Singularity is available"
    singularity --version
else
    echo ""
    echo "WARNING: Singularity command not found"
fi

# Check if GPU is available (when on compute node)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo ""
    echo "NOTE: nvidia-smi not available (this is normal on login nodes)"
fi

echo ""
echo "=== Configuration Summary ==="
echo "  CPUs: 24 (maximum)"
echo "  Memory: 96GB (24 CPUs x 4GB)"
echo "  GPU: 1"
echo "  Parallel Environments: 24 (via NUM_ENVS)"
echo ""
echo "To submit the job, run:"
echo "  cd $PROJECT_DIR"
echo "  sbatch resources/train_single_run.sbatch"
