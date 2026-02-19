#!/bin/bash

# Script to download the latest runs from tdmpc2_runs, decoy_policies, and sac_runs
# Usage: ./resources/download_latest_run.sh [server_address]

# Configuration
SERVER="${1:-tcml-login1}"
# SSH config should handle the full hostname mapping
REMOTE_PROJECT_BASE="/home/stud421/RL_CheungMaenzerAbraham_Hockey"
REMOTE_BASE="${REMOTE_PROJECT_BASE}/results"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_BASE="${LOCAL_PROJECT_DIR}/results"

# Folders to download: tdmpc2_runs, decoy_policies, sac_runs
FOLDERS=("tdmpc2_runs" "decoy_policies" "sac_runs" "tdmpc2_runs_test" "tdmpc2_runs_horizon", "self_play")
NUM_LATEST=30

echo "Connecting to ${SERVER}..."
echo "Remote base: ${REMOTE_BASE}"
echo "Local base: ${LOCAL_BASE}"

TOTAL_SUCCESS=0
TOTAL_FAILED=0

for FOLDER in "${FOLDERS[@]}"; do
    REMOTE_RUNS_DIR="${REMOTE_BASE}/${FOLDER}"
    LOCAL_RUNS_DIR="${LOCAL_BASE}/${FOLDER}"

    echo ""
    echo "========== Processing ${FOLDER} =========="
    echo "Remote: ${REMOTE_RUNS_DIR}"
    echo "Local: ${LOCAL_RUNS_DIR}"

    # Find the latest directories on the remote server
    LATEST_DIRS=$(ssh "${SERVER}" "cd ${REMOTE_RUNS_DIR} 2>/dev/null && ls -td */ 2>/dev/null | head -${NUM_LATEST} | sed 's|/$||'")

    if [ -z "${LATEST_DIRS}" ]; then
        echo "No directories found in ${FOLDER}, skipping..."
        continue
    fi

    DIR_COUNT=$(echo "${LATEST_DIRS}" | wc -l)
    echo "Found ${DIR_COUNT} directory/directories to download:"
    echo "${LATEST_DIRS}" | nl

    mkdir -p "${LOCAL_RUNS_DIR}"

    echo ""
    echo "Starting download of ${DIR_COUNT} run(s) from ${FOLDER}..."
    echo ""

    while IFS= read -r DIR; do
        if [ -n "${DIR}" ]; then
            echo "=========================================="
            echo "Downloading ${FOLDER}/${DIR}"
            echo "=========================================="

            rsync -avz --progress "${SERVER}:${REMOTE_RUNS_DIR}/${DIR}" "${LOCAL_RUNS_DIR}/"

            if [ $? -eq 0 ]; then
                echo ""
                echo "Successfully downloaded: ${FOLDER}/${DIR}"
                echo "Local path: ${LOCAL_RUNS_DIR}/${DIR}"
                TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
            else
                echo ""
                echo "Error: Failed to download: ${FOLDER}/${DIR}"
                TOTAL_FAILED=$((TOTAL_FAILED + 1))
            fi
            echo ""
        fi
    done <<< "${LATEST_DIRS}"
done

echo "=========================================="
echo "Download summary:"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAILED}"
echo "=========================================="

if [ ${TOTAL_FAILED} -gt 0 ]; then
    exit 1
fi
