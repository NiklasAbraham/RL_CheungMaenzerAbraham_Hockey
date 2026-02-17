#!/bin/bash
#SBATCH --job-name=comprl_agent_0010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --partition=week
#SBATCH --time=7-00:00:00
#SBATCH --output=comprl_agent_0010_%j.out
#SBATCH --error=comprl_agent_0010_%j.err
# Game logs (start/end, score, W/D/L tally) go to the .out file. tail -f comprl_agent_0010_<jobid>.out

# Competition server client: run archive agent on the cluster
# Usage: sbatch resources/cluster/niklas/run_competition_agent.sbatch
# Sync archive first: bash resources/cluster/niklas/sync_to_cluster_with_archive.sh

set -e

PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
CONTAINER="$PROJECT_DIR/singularity_build/rl_hockey_competition.simg"
ARCHIVE_AGENT_ID="0002_TDMPC2_2026-02-13_12-39-15"

echo "=== Competition Agent Runner ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Agent: $ARCHIVE_AGENT_ID"
echo "Start time: $(date)"
echo ""

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    echo "Build it: singularity build --fakeroot $CONTAINER resources/container/container_competition.def"
    exit 1
fi

# Check archive on cluster (sync_to_cluster.sh excludes archive/ by default)
ARCHIVE_REGISTRY="$PROJECT_DIR/archive/registry.json"
if [ ! -f "$ARCHIVE_REGISTRY" ]; then
    echo "ERROR: Archive not on cluster: $ARCHIVE_REGISTRY"
    echo "Sync with archive: bash resources/cluster/niklas/sync_to_cluster_with_archive.sh"
    exit 1
fi
if ! grep -q "\"$ARCHIVE_AGENT_ID\"" "$ARCHIVE_REGISTRY"; then
    echo "ERROR: Agent '$ARCHIVE_AGENT_ID' not in $ARCHIVE_REGISTRY"
    exit 1
fi

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

# Run client with cwd = PROJECT_DIR (project root) so archive paths
# (e.g. archive/agents/0010_...) resolve correctly. Restart loop like autorestart.sh.
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
            python3 '$CLIENT_SCRIPT' --args --agent=archive --archive-id=$ARCHIVE_AGENT_ID || true; \
            echo 'Restarting in 20s...'; sleep 20; \
        done"

echo ""
echo "=== Job ended at $(date) ==="
