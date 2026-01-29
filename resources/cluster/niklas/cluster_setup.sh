#!/bin/bash
# Setup script to run on the TCML cluster after syncing files
# Run this script on the cluster: bash resources/cluster/niklas/cluster_setup.sh

set -e

PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

echo "Setting up RL Hockey project on TCML cluster..."
echo ""

# Create singularity build directory
echo "Creating singularity build directory..."
mkdir -p singularity_build
cd singularity_build

# Copy container definition
echo "Copying container definition..."
cp ../resources/container/container_abraham.def ./rl_hockey.def

# Copy requirements.txt to build directory
echo "Copying requirements.txt..."
cp ../requirements.txt ./requirements.txt

# Verify files exist before building
echo ""
echo "Verifying files are in place for build:"
if [ ! -f "./rl_hockey.def" ]; then
    echo "ERROR: rl_hockey.def not found!"
    exit 1
fi
echo "✓ rl_hockey.def found"

if [ ! -f "./requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    exit 1
fi
echo "✓ requirements.txt found in project root ($(wc -l < ../requirements.txt) lines)"

# Show files in build directory
echo "Files in build directory:"
ls -lh

# Build the container
echo ""
echo "Building Singularity container (this may take 10-30 minutes)..."
echo "This will install all packages from requirements.txt into the container."
singularity build --fakeroot rl_hockey.simg rl_hockey.def

echo ""
echo "Container built successfully!"
echo ""
echo "Container location: $PROJECT_DIR/singularity_build/rl_hockey.simg"
echo ""
echo "Next steps:"
echo "1. Update resources/train_single_run.sbatch if needed (check paths and email)"
echo "2. Submit the job: sbatch resources/train_single_run.sbatch"
echo "3. Monitor with: squeue -u \$USER"
