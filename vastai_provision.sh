#!/bin/bash
set -eo pipefail

# Optional: Logging for Vast.ai UI
LOG_FILE="/var/log/provision.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "==== Starting Provisioning ===="

# Install any system dependencies you need
apt-get update
apt-get install -y --no-install-recommends git

# Clone your repository
mkdir -p /workspace
cd /workspace
git clone https://github.com/jekabsGritans/mt_fcgformer.git repo

# Use the pre-installed Python and pip from the provided venv
/venv/main/bin/pip install --upgrade pip

# Install requirements into the existing venv
/venv/main/bin/pip install --no-cache-dir -r /workspace/repo/requirements.txt

echo "==== Provisioning Completed ===="
