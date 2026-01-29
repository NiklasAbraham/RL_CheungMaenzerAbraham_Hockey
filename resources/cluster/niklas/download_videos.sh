#!/bin/bash

# Script to download videos from the cluster
# Usage: ./resources/download_videos.sh [server_address] [num_videos]

# Configuration
SERVER="${1:-tcml-login1}"
NUM_VIDEOS="${2:-5}"
# SSH config should handle the full hostname mapping
REMOTE_VIDEOS_DIR="/home/stud421/RL_CheungMaenzerAbraham_Hockey/videos"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_VIDEOS_DIR="${LOCAL_PROJECT_DIR}/videos"

echo "Connecting to ${SERVER}..."
echo "Remote videos directory: ${REMOTE_VIDEOS_DIR}"
echo "Local videos directory: ${LOCAL_VIDEOS_DIR}"

# Find the latest video files on the remote server
echo ""
echo "Finding latest ${NUM_VIDEOS} video file(s)..."
LATEST_VIDEOS=$(ssh "${SERVER}" "cd ${REMOTE_VIDEOS_DIR} && ls -t *.mp4 2>/dev/null | head -${NUM_VIDEOS}")

if [ -z "${LATEST_VIDEOS}" ]; then
    echo "Error: No video files (.mp4) found on remote server."
    echo "Checking if videos directory exists..."
    ssh "${SERVER}" "ls -ld ${REMOTE_VIDEOS_DIR} 2>/dev/null || echo 'Directory does not exist'"
    exit 1
fi

# Count how many videos were found
VIDEO_COUNT=$(echo "${LATEST_VIDEOS}" | wc -l)
echo "Found ${VIDEO_COUNT} video file(s) to download:"
echo "${LATEST_VIDEOS}" | nl

# Create local directory if it doesn't exist
mkdir -p "${LOCAL_VIDEOS_DIR}"

# Download each video file using rsync (more efficient than scp)
echo ""
echo "Starting download of ${VIDEO_COUNT} video(s)..."
echo "This may take a while depending on the file size..."
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

while IFS= read -r VIDEO; do
    if [ -n "${VIDEO}" ]; then
        echo "=========================================="
        echo "Downloading: ${VIDEO}"
        echo "=========================================="
        
        rsync -avz --progress "${SERVER}:${REMOTE_VIDEOS_DIR}/${VIDEO}" "${LOCAL_VIDEOS_DIR}/"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "Successfully downloaded: ${VIDEO}"
            echo "Local path: ${LOCAL_VIDEOS_DIR}/${VIDEO}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo ""
            echo "Error: Failed to download: ${VIDEO}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        echo ""
    fi
done <<< "${LATEST_VIDEOS}"

echo "=========================================="
echo "Download summary:"
echo "  Successful: ${SUCCESS_COUNT}"
echo "  Failed: ${FAILED_COUNT}"
echo "=========================================="

if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo ""
    echo "Videos downloaded to: ${LOCAL_VIDEOS_DIR}"
fi

if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
