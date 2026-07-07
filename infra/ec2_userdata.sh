#!/bin/bash
# EC2 user-data — runs once on first boot as root.
# Supports --dry-run for local inspection: prints actions, changes nothing.

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

run() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        eval "$*"
    fi
}

echo "=== QuNtra EC2 first-boot setup ==="

run "apt-get update -y"
run "apt-get install -y python3 python3-pip python3-venv git \
    postgresql-client docker.io screen tmux"

# IST timezone — the scheduler's cron jobs are IST-anchored
run "timedatectl set-timezone Asia/Kolkata"
echo "Timezone: $($DRY_RUN && echo '[dry-run] Asia/Kolkata' || date +%Z)"

run "systemctl start docker"
run "systemctl enable docker"

run "mkdir -p /home/ubuntu/quntra"
run "chown ubuntu:ubuntu /home/ubuntu/quntra"

# systemd unit — QuNtra survives reboots
UNIT='[Unit]
Description=QuNtra Trading System
After=network.target docker.service
Requires=docker.service

[Service]
Type=forking
User=ubuntu
WorkingDirectory=/home/ubuntu/quntra
ExecStart=/home/ubuntu/quntra/infra/start_quntra.sh
ExecStop=/bin/bash -c "kill \$(cat /home/ubuntu/quntra/watchdog.pid) \$(cat /home/ubuntu/quntra/quntra.pid) \$(cat /home/ubuntu/quntra/telegram_bot.pid) 2>/dev/null; true"
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target'

if $DRY_RUN; then
    echo "[dry-run] write /etc/systemd/system/quntra.service:"
    echo "$UNIT" | sed 's/^/    /'
else
    echo "$UNIT" > /etc/systemd/system/quntra.service
fi

run "systemctl daemon-reload"
run "systemctl enable quntra"

echo "EC2 setup complete. Deploy code (sync_to_ec2.sh), then:"
echo "  sudo systemctl start quntra"
