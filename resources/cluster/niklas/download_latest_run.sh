#!/bin/bash

# Script to download the latest 5 tdmpc2 runs from the cluster
# Usage: ./resources/download_latest_run.sh [server_address]

# Configuration
SERVER="${1:-tcml-login1}"
# SSH config should handle the full hostname mapping
REMOTE_RUNS_DIR1="/home/stud421/RL_CheungMaenzerAbraham_Hockey/results/tdmpc2_runs"
REMOTE_RUNS_DIR2="/home/stud421/RL_CheungMaenzerAbraham_Hockey/results/sac_runs"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_RUNS_DIR1="${LOCAL_PROJECT_DIR}/../../results/tdmpc2_runs"
LOCAL_RUNS_DIR2="${LOCAL_PROJECT_DIR}/../../results/sac_runs"
echo "Connecting to ${SERVER}..."
echo "Remote runs directory: ${REMOTE_RUNS_DIR1}"
echo "Local runs directory: ${LOCAL_RUNS_DIR1}"
echo "Remote runs directory: ${REMOTE_RUNS_DIR2}"
echo "Local runs directory: ${LOCAL_RUNS_DIR2}"

# Find the latest 30 timestamped directories on the remote server
echo ""
echo "Finding latest 30 tdmpc2 run directories..."
LATEST_DIRS1=$(ssh "${SERVER}" "cd ${REMOTE_RUNS_DIR1} && ls -td */ 2>/dev/null | head -30 | sed 's|/$||'")
echo "Finding latest 30 sac run directories..."
LATEST_DIRS2=$(ssh "${SERVER}" "cd ${REMOTE_RUNS_DIR2} && ls -td */ 2>/dev/null | head -30 | sed 's|/$||'")

if [ -z "${LATEST_DIRS1}" ] && [ -z "${LATEST_DIRS2}" ]; then
    echo "Error: No run directories found on remote server."
    exit 1
fi

# Count how many directories were found
DIR_COUNT1=$(echo "${LATEST_DIRS1}" | grep -v '^$' | wc -l)
DIR_COUNT2=$(echo "${LATEST_DIRS2}" | grep -v '^$' | wc -l)
if [ ${DIR_COUNT1} -gt 0 ]; then
    echo "Found ${DIR_COUNT1} tdmpc2 directory/directories to download:"
    echo "${LATEST_DIRS1}" | nl
fi
if [ ${DIR_COUNT2} -gt 0 ]; then
    echo "Found ${DIR_COUNT2} sac directory/directories to download:"
    echo "${LATEST_DIRS2}" | nl
fi

# Create local directories if they don't exist
mkdir -p "${LOCAL_RUNS_DIR1}"
mkdir -p "${LOCAL_RUNS_DIR2}"

# Download each directory using rsync (more efficient than scp)
echo ""
SUCCESS_COUNT1=0
FAILED_COUNT1=0
SUCCESS_COUNT2=0
FAILED_COUNT2=0

# Download tdmpc2 runs
if [ ${DIR_COUNT1} -gt 0 ]; then
    echo "Starting download of ${DIR_COUNT1} tdmpc2 run(s)..."
    echo "This may take a while depending on the size of the run data..."
    echo ""
    
    while IFS= read -r DIR; do
        if [ -n "${DIR}" ]; then
            echo "=========================================="
            echo "Downloading tdmpc2: ${DIR}"
            echo "=========================================="
            
            rsync -avz --progress "${SERVER}:${REMOTE_RUNS_DIR1}/${DIR}" "${LOCAL_RUNS_DIR1}/"
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "Successfully downloaded: ${DIR}"
                echo "Local path: ${LOCAL_RUNS_DIR1}/${DIR}"
                SUCCESS_COUNT1=$((SUCCESS_COUNT1 + 1))
            else
                echo ""
                echo "Error: Failed to download: ${DIR}"
                FAILED_COUNT1=$((FAILED_COUNT1 + 1))
            fi
            echo ""
        fi
    done <<< "${LATEST_DIRS1}"
fi

# Download sac runs
if [ ${DIR_COUNT2} -gt 0 ]; then
    echo "Starting download of ${DIR_COUNT2} sac run(s)..."
    echo "This may take a while depending on the size of the run data..."
    echo ""
    
    while IFS= read -r DIR; do
        if [ -n "${DIR}" ]; then
            echo "=========================================="
            echo "Downloading sac: ${DIR}"
            echo "=========================================="
            
            rsync -avz --progress "${SERVER}:${REMOTE_RUNS_DIR2}/${DIR}" "${LOCAL_RUNS_DIR2}/"
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "Successfully downloaded: ${DIR}"
                echo "Local path: ${LOCAL_RUNS_DIR2}/${DIR}"
                SUCCESS_COUNT2=$((SUCCESS_COUNT2 + 1))
            else
                echo ""
                echo "Error: Failed to download: ${DIR}"
                FAILED_COUNT2=$((FAILED_COUNT2 + 1))
            fi
            echo ""
        fi
    done <<< "${LATEST_DIRS2}"
fi

echo "=========================================="
echo "Download summary:"
echo "  tdmpc2 - Successful: ${SUCCESS_COUNT1}, Failed: ${FAILED_COUNT1}"
echo "  sac - Successful: ${SUCCESS_COUNT2}, Failed: ${FAILED_COUNT2}"
echo "=========================================="

if [ ${FAILED_COUNT1} -gt 0 ] || [ ${FAILED_COUNT2} -gt 0 ]; then
    exit 1
fi
