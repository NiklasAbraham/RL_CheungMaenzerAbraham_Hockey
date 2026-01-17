#!/bin/bash

#SBATCH --job-name=tdmpc2_profile
# Job name

#SBATCH --partition=day
# Partition: day for quick profiling jobs

#SBATCH --gres=gpu:2080ti:1
# Request 1 A4000 GPU (16GB memory, CUDA capability 8.6, supports torch.compile)
# A4000 has more memory than 2080ti (16GB vs 11GB) - better for profiling with memory tracking
# Alternative options: gpu:2080ti:1 (11GB) or gpu:1080ti:1 (11GB, CUDA 6.1, no torch.compile)

#SBATCH --mem=64G
# Request 64GB of memory (torch.profiler is very memory-intensive, especially with profile_memory=True)

#SBATCH --time=02:00:00
# Time limit: 2 hours should be plenty for profiling

#SBATCH --error=profile.%J.err
# Error log file

#SBATCH --output=profile.%J.out
# Output log file

#SBATCH --mail-type=END,FAIL
# Email notifications when job ends or fails

#SBATCH --mail-user=blabla@student.uni-tuebingen.de
# Your email address (change this to your email)

# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to project root
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Configuration options (modify these as needed)
CONFIG_PATH="$PROJECT_DIR/configs/curriculum_tdmpc2.json"
OUTPUT_DIR="$PROJECT_DIR/results/profiling"
NUM_ITERATIONS=50
# Reduced from 100 to save memory - torch.profiler uses significant memory per iteration
EXPORT_TRACES=""
# Set EXPORT_TRACES="--export-traces" if you want Chrome trace files

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the profiling script using singularity
singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && python3 '$PROJECT_DIR/src/rl_hockey/TD_MPC2/profile_tdmpc2.py' --config '$CONFIG_PATH' --output-dir '$OUTPUT_DIR' --num-iterations $NUM_ITERATIONS $EXPORT_TRACES"

# Print completion message
echo ""
echo "Profiling complete! Check the output above for performance analysis."
echo "Output directory: $OUTPUT_DIR"
if [ -n "$EXPORT_TRACES" ]; then
    echo "Trace files saved to: $OUTPUT_DIR/*.json"
    echo "Open in Chrome: chrome://tracing"
fi