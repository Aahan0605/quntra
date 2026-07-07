#!/bin/bash
# Sync local QuNtra code + secrets to the EC2 box.
# Usage: bash infra/sync_to_ec2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source infra/.ec2_info 2>/dev/null || { echo "Run deploy_aws.sh first"; exit 1; }
KEY="$HOME/.ssh/quntra-key.pem"
HOST="ubuntu@$ELASTIC_IP"
REMOTE_DIR="/home/ubuntu/quntra"

echo "Syncing to $ELASTIC_IP…"

rsync -avz --progress \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude 'logs/*.log' \
    --exclude 'config/secrets.env' \
    --exclude '*.pid' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    . "$HOST:$REMOTE_DIR/"

# Secrets travel over SSH only — never through git
scp -i "$KEY" config/secrets.env "$HOST:$REMOTE_DIR/config/secrets.env"

# Data cache: needed for training/council; sync separately (large)
rsync -avz \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    data/cache/ "$HOST:$REMOTE_DIR/data/cache/"
rsync -avz \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    data/models/ "$HOST:$REMOTE_DIR/data/models/"

echo "Code synced. Setting up the remote environment…"

ssh -i "$KEY" "$HOST" << 'REMOTE'
    set -e
    cd ~/quntra
    python3 -m venv venv 2>/dev/null || true
    ./venv/bin/pip install -q --upgrade pip
    ./venv/bin/pip install -q -r requirements-pinned.txt
    ./venv/bin/pip install -q --no-deps jugaad-data==0.28
    sudo docker run -d --name quntra-db \
        -e POSTGRES_USER=quntra \
        -e POSTGRES_PASSWORD=quntra_dev \
        -e POSTGRES_DB=quntra \
        -p 5432:5432 \
        --restart unless-stopped \
        postgres:15 2>/dev/null || echo "PostgreSQL already running"
    sleep 5
    ./venv/bin/python -m alembic upgrade head
    echo "Remote environment ready."
REMOTE

echo ""
echo "=== SYNC COMPLETE ==="
echo "Start QuNtra on EC2:"
echo "  ssh -i $KEY $HOST"
echo "  sudo systemctl start quntra     # or: bash infra/start_quntra.sh"
