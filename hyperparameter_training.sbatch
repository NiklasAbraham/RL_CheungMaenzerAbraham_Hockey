#!/bin/bash

#SBATCH --job-name=hockey_hyperparam
# give it any name you want

#SBATCH --cpus-per-task=8
# max 24 per node, using 8 for multiprocessing

#SBATCH --partition=week
# choose out of day, week, month depending on job duration
# This job tests 180 configurations, each with 3000 episodes, so week partition is recommended

#SBATCH --mem-per-cpu=4G
# max 251GB per node, using 32GB total (8 CPUs * 4GB)

#SBATCH --gres=gpu:1
# how many gpus to use
# each node has 4 gpus (1080ti/A4000) or 8 gpus (2080ti/L40S)

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

#SBATCH --mail-user=YOUR_EMAIL@uni-tuebingen.de
# your email - CHANGE THIS!

# Activate the singularity container
# Adjust the path to your container image
SINGULARITY_IMAGE="/path/to/your/container.sif"

# Set working directory to your project directory
# This should be the root of your RL_CheungMaenzerAbraham_Hockey project
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

# Install the rl_hockey package in editable mode (if not already installed in container)
# This allows code updates without rebuilding the container
singularity exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    pip install -e "$PROJECT_DIR" --quiet

# Run the hyperparameter training script using singularity
singularity run --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    python3 "$PROJECT_DIR/src/rl_hockey/common/training/shooting_training_simple_hyperparameters.py" \
    --num_parallel 4 \
    --output_dir results/hyperparameter_runs
