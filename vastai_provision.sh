#!/bin/bash
set -e

# Install dependencies
apt-get update
apt-get install -y --no-install-recommends build-essential git vim curl
rm -rf /var/lib/apt/lists/*

# Set up workspace and clone your code
mkdir -p /workspace
cd /workspace
git clone https://github.com/jekabsGritans/mt_fcgformer.git .

# Install Python dependencies if requirements.txt exists
if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
fi