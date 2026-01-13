#!/bin/bash
# Setup script to run on the TCML cluster after syncing files
# Run this script on the cluster: bash resources/cluster_setup.sh

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
cp ../resources/container/container.def ./rl_hockey.def

# Embed requirements.txt directly into container.def by replacing the placeholder
echo "Embedding requirements.txt into container definition..."
python3 << 'PYEOF'
import sys
import re

# Read the container definition
with open('rl_hockey.def', 'r') as f:
    content = f.read()

# Read requirements
with open('../requirements.txt', 'r') as f:
    requirements = f.read()

# Find and replace the heredoc section
# Pattern: everything between '# Requirements will be embedded...' and 'REQUIREMENTS_EMBEDDED'
pattern = r'(# Requirements will be embedded here by cluster_setup.sh\n)(.*?)(REQUIREMENTS_EMBEDDED)'

def replace_func(match):
    return match.group(1) + requirements + '\n' + match.group(3)

new_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)

if new_content != content:
    # Write back
    with open('rl_hockey.def', 'w') as f:
        f.write(new_content)
    print(f"✓ Successfully embedded requirements.txt ({len(requirements.split(chr(10)))} lines)")
else:
    print("ERROR: Could not find placeholder in container.def")
    sys.exit(1)
PYEOF

# Verify files exist before building
echo ""
echo "Verifying files are in place for build:"
if [ ! -f "./rl_hockey.def" ]; then
    echo "ERROR: rl_hockey.def not found!"
    exit 1
fi
echo "✓ rl_hockey.def found"

if [ ! -f "./requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in build directory!"
    exit 1
fi
echo "✓ requirements.txt found ($(wc -l < ./requirements.txt) lines)"

# Show that requirements.txt is in the same directory as the .def file
echo "Files in build directory:"
ls -lh *.def *.txt 2>/dev/null || ls -lh

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
