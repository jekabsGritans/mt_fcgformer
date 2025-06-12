#!/bin/bash
set -eo pipefail

# Logging for Vast.ai UI
LOG_FILE="/var/log/provision.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "==== Starting Provisioning: $(date) ===="

# Check for required environment variable
if [ -z "$SSH_PASSWORD" ]; then
    echo "ERROR: SSH_PASSWORD environment variable must be provided!"
    exit 1
fi

# Install required tools
apt-get update
apt-get install -y --no-install-recommends git openssh-client autossh tmux sshpass ranger

# Clone repository
mkdir -p /workspace
cd /workspace
git clone https://github.com/jekabsGritans/mt_fcgformer.git repo

# Install requirements
/venv/main/bin/pip install --upgrade pip
/venv/main/bin/pip install --no-cache-dir -r /workspace/repo/requirements.txt
/venv/main/bin/pip install --no-cache-dir optuna pymysql

# Connection details
VPS_HOST=${VPS_HOST:-"138.199.214.167"}
VPS_USER=${VPS_USER:-"optuna-user"}
VPS_PORT=${VPS_PORT:-"22"}
DB_HOST=${DB_HOST:-"192.168.6.5"}
DB_PORT=${DB_PORT:-"3307"}
LOCAL_PORT=${LOCAL_PORT:-"13306"}
OPTUNA_DB_USER=${OPTUNA_DB_USER:-"user"}
OPTUNA_DB_PASSWORD=${OPTUNA_DB_PASSWORD:-"hu4sie2Aiwee"}

# Store password securely
mkdir -p /root/.ssh_tunnel
echo "$SSH_PASSWORD" > /root/.ssh_tunnel/password
chmod 600 /root/.ssh_tunnel/password

# Create tunnel script
cat > /root/tunnel.sh << EOF
#!/bin/bash
# Kill any existing tunnel
pkill -f "ssh.*$LOCAL_PORT:$DB_HOST:$DB_PORT" || true

# Start the tunnel with options that ignore home directory
sshpass -f /root/.ssh_tunnel/password ssh -o "StrictHostKeyChecking=no" \
    -o "UserKnownHostsFile=/dev/null" \
    -o "PreferredAuthentications=password" \
    -o "PubkeyAuthentication=no" \
    -o "BatchMode=no" \
    -N -L $LOCAL_PORT:$DB_HOST:$DB_PORT \
    $VPS_USER@$VPS_HOST -p $VPS_PORT
EOF
chmod +x /root/tunnel.sh

# Set up database URL
export OPTUNA_DB_URL="mysql+pymysql://$OPTUNA_DB_USER:$OPTUNA_DB_PASSWORD@127.0.0.1:$LOCAL_PORT/optuna"
echo "export OPTUNA_DB_URL=\"$OPTUNA_DB_URL\"" >> /root/.bashrc

# Start tunnel in background
echo "Starting SSH tunnel in background..."
tmux new-session -d -s tunnel "/root/tunnel.sh"

# Wait for tunnel to establish
sleep 5

# Start Optuna
echo "Starting Optuna optimization..."
tmux new-session -d -s optuna "cd /workspace/repo && OPTUNA_DB_URL=\"$OPTUNA_DB_URL\" /venv/main/bin/python optimize.py"

# Create helper script
cat > /root/status.sh << 'EOF'
#!/bin/bash
echo "=== TUNNEL STATUS ==="
netstat -tulpn | grep 13306
ps aux | grep "ssh.*13306" | grep -v grep

echo -e "\n=== OPTUNA STATUS ==="
tmux list-sessions | grep optuna

echo -e "\n=== COMMANDS ==="
echo "Restart tunnel: tmux kill-session -t tunnel && tmux new-session -d -s tunnel /root/tunnel.sh"
echo "Attach to Optuna: tmux attach -t optuna"
EOF
chmod +x /root/status.sh

echo "==== Provisioning Complete ===="
echo "Optuna is running in a tmux session"
echo "Check status with: /root/status.sh"