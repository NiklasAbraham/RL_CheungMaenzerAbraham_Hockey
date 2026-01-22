#!/bin/bash

# Script to download the latest 5 tdmpc2 runs from the cluster
# Usage: ./resources/download_latest_run.sh [server_address]

# Configuration
SERVER="${1:-tcml-login1}"
# SSH config should handle the full hostname mapping
REMOTE_RUNS_DIR="/home/stud421/RL_CheungMaenzerAbraham_Hockey/results/tdmpc2_runs"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_RUNS_DIR="${LOCAL_PROJECT_DIR}/results/tdmpc2_runs"

echo "Connecting to ${SERVER}..."
echo "Remote runs directory: ${REMOTE_RUNS_DIR}"
echo "Local runs directory: ${LOCAL_RUNS_DIR}"

# Find the latest 5 timestamped directories on the remote server
echo ""
echo "Finding latest 30 tdmpc2 run directories..."
LATEST_DIRS=$(ssh "${SERVER}" "cd ${REMOTE_RUNS_DIR} && ls -td */ 2>/dev/null | head -30 | sed 's|/$||'")

if [ -z "${LATEST_DIRS}" ]; then
    echo "Error: No tdmpc2 run directories found on remote server."
    exit 1
fi

# Count how many directories were found
DIR_COUNT=$(echo "${LATEST_DIRS}" | wc -l)
echo "Found ${DIR_COUNT} directory/directories to download:"
echo "${LATEST_DIRS}" | nl

# Create local directory if it doesn't exist
mkdir -p "${LOCAL_RUNS_DIR}"

# Download each directory using rsync (more efficient than scp)
echo ""
echo "Starting download of ${DIR_COUNT} run(s)..."
echo "This may take a while depending on the size of the run data..."
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

while IFS= read -r DIR; do
    if [ -n "${DIR}" ]; then
        echo "=========================================="
        echo "Downloading: ${DIR}"
        echo "=========================================="
        
        rsync -avz --progress "${SERVER}:${REMOTE_RUNS_DIR}/${DIR}" "${LOCAL_RUNS_DIR}/"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "Successfully downloaded: ${DIR}"
            echo "Local path: ${LOCAL_RUNS_DIR}/${DIR}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo ""
            echo "Error: Failed to download: ${DIR}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        echo ""
    fi
done <<< "${LATEST_DIRS}"

echo "=========================================="
echo "Download summary:"
echo "  Successful: ${SUCCESS_COUNT}"
echo "  Failed: ${FAILED_COUNT}"
echo "=========================================="

if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
