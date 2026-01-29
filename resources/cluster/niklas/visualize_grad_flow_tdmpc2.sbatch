#!/bin/bash

#SBATCH --job-name=grad_flow_tdmpc2
# Job name

#SBATCH --partition=day
# Partition: day for short visualization job

#SBATCH --gres=gpu:2080ti:1
# Request 1 GPU (optional; script can run on CPU with small model)

#SBATCH --mem=16G
# 16GB is enough for one forward pass and torchviz graph

#SBATCH --time=01:00:00
# Time limit: 1 hour

#SBATCH --error=grad_flow.%J.err
# Error log file

#SBATCH --output=grad_flow.%J.out
# Output log file

#SBATCH --mail-type=END,FAIL
# Email when job ends or fails

#SBATCH --mail-user=blabla@student.uni-tuebingen.de
# Your email (change this)

# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to project root
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Output path for the autograd graph PDF (under project so it syncs)
OUTPUT_PATH="$PROJECT_DIR/report/figures/grad_flow_tdmpc2"
mkdir -p "$(dirname "$OUTPUT_PATH")"

export PYTHONUNBUFFERED=1

# Install torchviz to user-writable dir (container is read-only)
# If job fails with "dot: command not found", add graphviz to the container:
#   In container.def %post: apt-get install -y graphviz
#   Then rebuild the image.
USER_PACKAGES_DIR="$PROJECT_DIR/.user_packages"
mkdir -p "$USER_PACKAGES_DIR"

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && pip install --target '$USER_PACKAGES_DIR' torchviz > /dev/null 2>&1 && export PYTHONUNBUFFERED=1 && export GRAD_FLOW_OUT='$OUTPUT_PATH' && export PYTHONPATH='$PROJECT_DIR/src:$USER_PACKAGES_DIR:\$PYTHONPATH' && python3 -u '$PROJECT_DIR/src/rl_hockey/scripts/visualize_grad_flow_tdmpc2.py'"

echo ""
echo "Grad-flow visualization complete. Output: ${OUTPUT_PATH}.pdf"
