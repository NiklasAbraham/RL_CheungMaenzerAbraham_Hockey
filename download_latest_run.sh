#!/bin/bash

# Script to download the latest hyperparameter run from the cluster
# Usage: ./download_latest_run.sh [server_address]

# Configuration
SERVER="${1:-tcml-login1}"
# SSH config should handle the full hostname mapping
REMOTE_PROJECT_DIR="~/RL_CheungMaenzerAbraham_Hockey"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_RUNS_DIR="${REMOTE_PROJECT_DIR}/results/hyperparameter_runs"
LOCAL_RUNS_DIR="${LOCAL_PROJECT_DIR}/results/hyperparameter_runs"

echo "Connecting to ${SERVER}..."
echo "Remote runs directory: ${REMOTE_RUNS_DIR}"
echo "Local runs directory: ${LOCAL_RUNS_DIR}"

# Find the latest timestamped directory on the remote server
echo ""
echo "Finding latest hyperparameter run directory..."
LATEST_DIR=$(ssh "${SERVER}" "cd ${REMOTE_RUNS_DIR} && ls -td */ 2>/dev/null | head -1 | sed 's|/$||'")

if [ -z "${LATEST_DIR}" ]; then
    echo "Error: No hyperparameter run directories found on remote server."
    exit 1
fi

echo "Latest directory found: ${LATEST_DIR}"

# Create local directory if it doesn't exist
mkdir -p "${LOCAL_RUNS_DIR}"

# Download the latest directory using rsync (more efficient than scp)
echo ""
echo "Downloading ${LATEST_DIR} from remote server..."
echo "This may take a while depending on the size of the run data..."

rsync -avz --progress "${SERVER}:${REMOTE_RUNS_DIR}/${LATEST_DIR}" "${LOCAL_RUNS_DIR}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully downloaded latest run: ${LATEST_DIR}"
    echo "Local path: ${LOCAL_RUNS_DIR}/${LATEST_DIR}"
else
    echo ""
    echo "Error: Failed to download the run directory."
    exit 1
fi
