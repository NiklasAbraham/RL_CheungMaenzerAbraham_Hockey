#!/bin/bash

# Download test.txt (or another file) from the cluster.
# On the cluster, create test.txt from a job output first, e.g.:
#   cat job.1984903.out > test.txt
#
# Usage:
#   ./download_job_output.sh [server] [remote_file]
#   ./download_job_output.sh                    # uses default server, downloads test.txt
#   ./download_job_output.sh tcml-login1        # same, explicit server
#   ./download_job_output.sh tcml-login1 test.txt

SERVER="${1:-tcml-login1}"
REMOTE_FILE="${2:-test.txt}"
REMOTE_PROJECT_BASE="/home/stud421/RL_CheungMaenzerAbraham_Hockey"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_DEST="${LOCAL_PROJECT_DIR}/${REMOTE_FILE}"

echo "Server: ${SERVER}"
echo "Remote: ${SERVER}:${REMOTE_PROJECT_BASE}/${REMOTE_FILE}"
echo "Local:  ${LOCAL_DEST}"
echo ""

rsync -avz --progress "${SERVER}:${REMOTE_PROJECT_BASE}/${REMOTE_FILE}" "${LOCAL_DEST}"

if [ $? -eq 0 ]; then
    echo ""
    echo "Downloaded to: ${LOCAL_DEST}"
else
    echo ""
    echo "Download failed. On the cluster, create the file first, e.g.:"
    echo "  cat job.JOBID.out > test.txt"
    exit 1
fi
