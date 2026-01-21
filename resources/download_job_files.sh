#!/bin/bash

# Script to download job.* files from the cluster
# Usage: ./resources/download_job_files.sh [server_address]

# Configuration
SERVER="${1:-tcml-login1}"
# SSH config should handle the full hostname mapping
REMOTE_PROJECT_DIR="~/RL_CheungMaenzerAbraham_Hockey"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_JOB_FILES_DIR="${REMOTE_PROJECT_DIR}"
LOCAL_JOB_FILES_DIR="${LOCAL_PROJECT_DIR}"

echo "Connecting to ${SERVER}..."
echo "Remote project directory: ${REMOTE_JOB_FILES_DIR}"
echo "Local project directory: ${LOCAL_JOB_FILES_DIR}"

# Check if job files exist on the remote server
echo ""
echo "Checking for job.* files on remote server..."
JOB_FILES=$(ssh "${SERVER}" "cd ${REMOTE_JOB_FILES_DIR} && ls job.* 2>/dev/null || echo ''")

if [ -z "${JOB_FILES}" ]; then
    echo "Warning: No job.* files found on remote server."
    exit 0
fi

echo "Found job files:"
echo "${JOB_FILES}" | while read -r file; do
    if [ -n "${file}" ]; then
        echo "  ${file}"
    fi
done

# Download job files using rsync (more efficient than scp)
echo ""
echo "Downloading job.* files from remote server..."

rsync -avz --progress "${SERVER}:${REMOTE_JOB_FILES_DIR}/job."* "${LOCAL_JOB_FILES_DIR}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully downloaded job files"
    echo "Local path: ${LOCAL_JOB_FILES_DIR}/"
else
    echo ""
    echo "Error: Failed to download the job files."
    exit 1
fi
