#!/bin/bash

#SBATCH --job-name=hockey_run_tdmpc
# Job name

#SBATCH --partition=day
# Partition: day, week, or month

#SBATCH --gres=gpu:2080ti:1
# Request 1 GPU (2080ti 11GB; alternative gpu:A4000:1)

#SBATCH --mem=16G
# Memory for evaluation (no training)

#SBATCH --time=04:00:00
# Time limit: 4 hours for many episodes

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=blabla@student.uni-tuebingen.de

SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"
PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
cd "$PROJECT_DIR"

export CHECKPOINT_PATH="$PROJECT_DIR/models/tdmpc2/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep025646.pt"
export N=100
export WEAK_OPPONENT=0

export PYTHONUNBUFFERED=1

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -c "source /venv/bin/activate && export CHECKPOINT_PATH='$CHECKPOINT_PATH' && export N='$N' && export WEAK_OPPONENT='$WEAK_OPPONENT' && export PYTHONPATH='$PROJECT_DIR/src:\$PYTHONPATH' && export PYTHONUNBUFFERED=1 && python3 -u -c \"
import os
from rl_hockey.TD_MPC2.run_tdmpc import main
main(
    checkpoint_path=os.environ['CHECKPOINT_PATH'],
    config_path=None,
    N=int(os.environ.get('N', '100')),
    weak_opponent=os.environ.get('WEAK_OPPONENT', '0') == '1',
    render=False,
)
\""
