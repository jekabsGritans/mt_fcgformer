#!/bin/bash
set -e

apt-get update
apt-get install -y --no-install-recommends git

mkdir -p /workspace
cd /workspace

echo "Cloning repo..." | tee -a /workspace/provision.log
git clone https://github.com/jekabsGritans/mt_fcgformer.git /workspace/repo 2>&1 | tee -a /workspace/provision.log

if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
fi

echo "Provisioning complete." | tee -a /workspace/provision.log
