#!/bin/bash

# Script to download the archive folder from the server
# Usage: ./resources/cluster/niklas/download_archive.sh [server_address]

# Configuration
SERVER="${1:-tcml-login1}"
REMOTE_BASE="/home/stud421/RL_CheungMaenzerAbraham_Hockey"
REMOTE_ARCHIVE="${REMOTE_BASE}/archive"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_ARCHIVE="${LOCAL_PROJECT_DIR}/archive"

echo "Connecting to ${SERVER}..."
echo "Remote archive: ${REMOTE_ARCHIVE}"
echo "Local archive:  ${LOCAL_ARCHIVE}"
echo ""

# Ensure local archive parent exists
mkdir -p "${LOCAL_PROJECT_DIR}"

echo "Downloading archive folder..."
rsync -avz --progress "${SERVER}:${REMOTE_ARCHIVE}/" "${LOCAL_ARCHIVE}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully downloaded archive folder"
    echo "Local path: ${LOCAL_ARCHIVE}"
    exit 0
else
    echo ""
    echo "Error: Failed to download archive folder"
    exit 1
fi
