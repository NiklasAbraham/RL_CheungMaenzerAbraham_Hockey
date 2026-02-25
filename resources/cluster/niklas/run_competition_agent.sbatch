#!/bin/bash
#SBATCH --job-name=comprl_agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --partition=week
#SBATCH --time=7-00:00:00
#SBATCH --output=comprl_agent_%j.out
#SBATCH --error=comprl_agent_%j.err
# Game logs (start/end, score, W/D/L tally) go to the .out file. tail -f comprl_agent_<jobid>.out
# Usage: set MODEL_PATH below, then sbatch resources/cluster/niklas/run_competition_agent.sbatch

set -e

PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
CONTAINER="$PROJECT_DIR/singularity_build/rl_hockey_competition.simg"

# Path to .pt checkpoint (e.g. results/self_play/.../model.pt or archive/agents/.../checkpoint.pt)
# MODEL_PATH="results/self_play/2026-02-01_09-55-22/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep019488.pt"
MODEL_PATH="results/self_play/2026-02-01_09-55-22/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep025646.pt"


echo "=== Competition Agent Runner ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Agent: $MODEL_PATH"
echo "Start time: $(date)"
echo ""

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    echo "Build it: singularity build --fakeroot $CONTAINER resources/container/container_competition.def"
    exit 1
fi

# Resolve path relative to project if needed
if [ ! -f "$MODEL_PATH" ]; then
    FULL_PATH="$PROJECT_DIR/$MODEL_PATH"
    if [ -f "$FULL_PATH" ]; then
        MODEL_PATH="$FULL_PATH"
    fi
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi
echo "Using model: $MODEL_PATH"

# Load competition server credentials
ENV_FILE="$PROJECT_DIR/resources/competition_server/comprl-hockey-agent/niklas_env.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Env file not found: $ENV_FILE"
    echo "Create it with COMPRL_SERVER_URL, COMPRL_SERVER_PORT, COMPRL_ACCESS_TOKEN"
    exit 1
fi
source "$ENV_FILE"
echo "✓ Server: $COMPRL_SERVER_URL:$COMPRL_SERVER_PORT"
echo ""

export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

# Run client with cwd = PROJECT_DIR (project root) so paths resolve correctly.
CLIENT_SCRIPT="$PROJECT_DIR/resources/competition_server/comprl-hockey-agent/run_client.py"

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$CONTAINER" \
    /bin/bash -c "source /venv/bin/activate && \
        export PYTHONUNBUFFERED=1 && \
        export PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && \
        export COMPRL_SERVER_URL='$COMPRL_SERVER_URL' && \
        export COMPRL_SERVER_PORT='$COMPRL_SERVER_PORT' && \
        export COMPRL_ACCESS_TOKEN='$COMPRL_ACCESS_TOKEN' && \
        while true; do \
            python3 '$CLIENT_SCRIPT' --args --model-path='$MODEL_PATH' || true; \
            echo 'Restarting in 20s...'; sleep 20; \
        done"

echo ""
echo "=== Job ended at $(date) ==="
