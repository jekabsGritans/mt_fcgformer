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

# Install SSH and other required tools
apt-get update
apt-get install -y --no-install-recommends git openssh-client autossh tmux netcat-openbsd sshpass

# Clone your repository
mkdir -p /workspace
cd /workspace
git clone https://github.com/jekabsGritans/mt_fcgformer.git repo

# Use the pre-installed Python and pip from the provided venv
/venv/main/bin/pip install --upgrade pip

# Install requirements into the existing venv
/venv/main/bin/pip install --no-cache-dir -r /workspace/repo/requirements.txt
/venv/main/bin/pip install --no-cache-dir optuna pymysql

# Set up SSH for tunnel with isolated configuration
echo "==== Setting up SSH tunnel ===="
mkdir -p ~/.ssh/optuna
chmod 700 ~/.ssh ~/.ssh/optuna

# SSH connection details (passed as environment variables)
VPS_HOST=${VPS_HOST:-"138.199.214.167"}
VPS_USER=${VPS_USER:-"optuna-user"}
VPS_PORT=${VPS_PORT:-"22"}
DB_HOST=${DB_HOST:-"192.168.6.5"} # Actual database server hostname
DB_PORT=${DB_PORT:-"3307"}
LOCAL_PORT=${LOCAL_PORT:-"13306"}

# Optuna DB credentials
OPTUNA_DB_USER=${OPTUNA_DB_USER:-"user"}
OPTUNA_DB_PASSWORD=${OPTUNA_DB_PASSWORD:-"hu4sie2Aiwee"}

# Create isolated SSH config for the tunnel
cat > ~/.ssh/optuna/config << EOF
Host optuna-tunnel
 HostName $VPS_HOST
 User $VPS_USER
 Port $VPS_PORT
 StrictHostKeyChecking no
 UserKnownHostsFile=/dev/null
EOF
chmod 600 ~/.ssh/optuna/config

# Store password in a secure file with restricted permissions
echo "$SSH_PASSWORD" > ~/.ssh/optuna/password
chmod 600 ~/.ssh/optuna/password

# Create a script to establish and maintain the SSH tunnel
cat > /root/start_tunnel.sh << EOF
#!/bin/bash
pkill -f "autossh.*optuna-tunnel-process" || true
# Connect to VPS and forward to the actual database server using password
export AUTOSSH_GATETIME=0
sshpass -f ~/.ssh/optuna/password autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -N -L $LOCAL_PORT:$DB_HOST:$DB_PORT -F ~/.ssh/optuna/config optuna-tunnel optuna-tunnel-process
EOF
chmod +x /root/start_tunnel.sh

# Create a diagnostic script
cat > /root/debug_ssh.sh << 'EOF'
#!/bin/bash
echo "=== SSH CONNECTION DIAGNOSTICS ==="
echo "Checking network connectivity..."
VPS_HOST=$(grep HostName ~/.ssh/optuna/config | awk '{print $2}')
VPS_PORT=$(grep Port ~/.ssh/optuna/config | awk '{print $2}')
echo "Testing connection to $VPS_HOST:$VPS_PORT..."

# Check if port is open
nc -zv $VPS_HOST $VPS_PORT

# Check SSH server response
echo "SSH server version detection:"
nc -v $VPS_HOST $VPS_PORT < /dev/null

# Try connection with max verbosity (using password)
echo "Attempting SSH with maximum verbosity..."
sshpass -f ~/.ssh/optuna/password ssh -vvv -F ~/.ssh/optuna/config -o "ConnectTimeout=10" optuna-tunnel exit

echo "SSH config file:"
cat ~/.ssh/optuna/config

echo "=== END DIAGNOSTICS ==="
EOF
chmod +x /root/debug_ssh.sh

# Export the database URL to use the local port
export OPTUNA_DB_URL="mysql://$OPTUNA_DB_USER:$OPTUNA_DB_PASSWORD@127.0.0.1:$LOCAL_PORT/optuna"
echo "export OPTUNA_DB_URL=\"$OPTUNA_DB_URL\"" >> /root/.bashrc

# Test the SSH connection with verbose output
echo "==== Testing SSH connection to VPS ===="
# First try a basic connectivity test
ping -c 1 $VPS_HOST || echo "WARNING: VPS host unreachable by ping (may be normal if ICMP blocked)"

# Test SSH with increasing verbosity
echo "Basic SSH test..."
sshpass -f ~/.ssh/optuna/password ssh -F ~/.ssh/optuna/config -o "BatchMode=no" -o "ConnectTimeout=10" optuna-tunnel exit
if [ $? -ne 0 ]; then
 echo "Basic SSH test failed, trying with verbose logging..."
 sshpass -f ~/.ssh/optuna/password ssh -v -F ~/.ssh/optuna/config -o "BatchMode=no" -o "ConnectTimeout=10" optuna-tunnel exit
 if [ $? -ne 0 ]; then
     echo "Verbose SSH test failed, trying with more debugging..."
     sshpass -f ~/.ssh/optuna/password ssh -vvv -F ~/.ssh/optuna/config -o "BatchMode=no" -o "ConnectTimeout=10" optuna-tunnel exit
     echo "ERROR: SSH connection failed! See debug output above."
     echo "Run /root/debug_ssh.sh for additional diagnostics"
     exit 1
 fi
fi

echo "SSH connection successful! Starting tunnel and Optuna..."

# Start the SSH tunnel in a tmux session
tmux new-session -d -s ssh_tunnel "/root/start_tunnel.sh"

# Wait for tunnel to establish
sleep 5

# Start Optuna in a tmux session
tmux new-session -d -s optuna "cd /workspace/repo && OPTUNA_DB_URL=\"$OPTUNA_DB_URL\" /venv/main/bin/python optimize.py"

# Create helper scripts
cat > /root/check_tunnel.sh << 'EOF'
#!/bin/bash
echo "Checking SSH tunnel status..."
netstat -tulpn | grep 13306
ps aux | grep autossh | grep -v grep
echo "To restart tunnel: tmux kill-session -t ssh_tunnel && tmux new-session -d -s ssh_tunnel '/root/start_tunnel.sh'"
EOF
chmod +x /root/check_tunnel.sh

cat > /root/attach_optuna.sh << 'EOF'
#!/bin/bash
tmux attach -t optuna
EOF
chmod +x /root/attach_optuna.sh

echo "==== Provisioning Completed: $(date) ===="
echo "SSH tunnel established to $VPS_HOST:$VPS_PORT forwarding localhost:$LOCAL_PORT to $DB_HOST:$DB_PORT"
echo "Database URL: $OPTUNA_DB_URL"
echo "To check tunnel status: /root/check_tunnel.sh"
echo "To attach to Optuna session: /root/attach_optuna.sh"
echo "If you have SSH issues, run: /root/debug_ssh.sh"