#!/bin/bash
# Quick script to generate clean requirements.txt from conda environment
# Usage: bash resources/quick_fix_requirements.sh

set -e

echo "=========================================="
echo "Generating Clean Requirements.txt"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please activate your conda environment first."
    exit 1
fi

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: No conda environment detected."
    echo "Please activate your conda environment first:"
    echo "  conda activate rl-hockey"
    exit 1
fi

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Run the Python script
python resources/generate_clean_requirements.py --replace --test

echo ""
echo "=========================================="
echo "Done! Your requirements.txt has been updated."
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the generated requirements.txt"
echo "2. Rebuild your container: bash resources/cluster_setup.sh"
echo ""
