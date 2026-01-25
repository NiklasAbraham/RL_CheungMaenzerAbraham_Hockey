#!/bin/bash

#SBATCH --job-name=hockey_hyperparam
# give it any name you want

#SBATCH --cpus-per-task=24
# max 24 per node, using maximum for best performance

#SBATCH --partition=week
# choose out of day, week, month depending on job duration
# This job tests 180 configurations, each with 3000 episodes, so week partition is recommended

#SBATCH --mem-per-cpu=2G
# max 251GB per node, using 48GB total (24 CPUs * 2GB)

# GPU not requested - focusing on maximum CPU cores

#SBATCH --time=7-00:00:00
# job length: 7 days (week partition limit)
# The job will run either until completion or until this timer runs out

#SBATCH --error=job.%J.err
# %J is the job ID, errors will be written to this file

#SBATCH --output=job.%J.out
# the output will be written in this file

#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL

#SBATCH --mail-user=niklas-sebastian.abraham@uni-tuebingen.de
# your email - CHANGE THIS!

# Activate the singularity container
# Path to the Singularity container image
SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"

# Set working directory to your project directory
# This should be the root of your RL_CheungMaenzerAbraham_Hockey project
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Install the rl_hockey package in editable mode (if not already installed in container)
# This allows code updates without rebuilding the container
singularity exec \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    pip install -e "$PROJECT_DIR" --quiet

# Run the hyperparameter training script using singularity
singularity run \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    python3 "$PROJECT_DIR/src/rl_hockey/common/training/shooting_training_simple_hyperparameters.py" \
    --num_parallel 8 \
    --output_dir results/hyperparameter_runs
