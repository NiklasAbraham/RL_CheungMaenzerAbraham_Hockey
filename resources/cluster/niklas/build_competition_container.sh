#!/bin/bash
# Build the competition container with comprl and all dependencies
# Run this on the cluster

set -e

PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
OUTPUT_DIR="$PROJECT_DIR/singularity_build"
CONTAINER_FILE="$OUTPUT_DIR/rl_hockey_competition.simg"
CONTAINER_DEF="$PROJECT_DIR/resources/container/container_competition.def"

echo "=== Building Competition Container ==="
echo "Container definition: $CONTAINER_DEF"
echo "Output container: $CONTAINER_FILE"
echo ""

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    echo "Sync the project to the cluster first:"
    echo "  bash resources/cluster/niklas/sync_to_cluster.sh"
    exit 1
fi

# Check if container definition exists
if [ ! -f "$CONTAINER_DEF" ]; then
    echo "ERROR: Container definition not found: $CONTAINER_DEF"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "ERROR: requirements.txt not found: $PROJECT_DIR/requirements.txt"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if container already exists
if [ -f "$CONTAINER_FILE" ]; then
    echo "WARNING: Container already exists: $CONTAINER_FILE"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
    rm -f "$CONTAINER_FILE"
fi

# Build the container
echo ""
echo "Building container (this will take 10-20 minutes)..."
echo ""

cd "$PROJECT_DIR"

singularity build --fakeroot "$CONTAINER_FILE" "$CONTAINER_DEF"

echo ""
echo "=== Build Complete ==="
echo "Container: $CONTAINER_FILE"
echo "Size: $(du -h $CONTAINER_FILE | cut -f1)"
echo ""
echo "To run the competition agent:"
echo "  sbatch resources/cluster/niklas/run_competition_agent.sbatch"
