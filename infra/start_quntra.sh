#!/bin/bash
# Start QuNtra (EC2 or any Linux/macOS box with the venv set up).
# The watchdog brings up and babysits the scheduler AND the Telegram
# bot runner — one entry point, three resident processes.
cd "$(dirname "$0")/.."

if [ "$(date +%Z)" != "IST" ]; then
    echo "WARNING: timezone is $(date +%Z), expected IST — scheduler jobs"
    echo "are IST-anchored. Fix: sudo timedatectl set-timezone Asia/Kolkata"
fi

mkdir -p logs
nohup ./venv/bin/python scripts/watchdog.py >> logs/watchdog.log 2>&1 &
echo $! > watchdog.pid
sleep 8   # watchdog launches scheduler + bot runner on its first cycle

echo "QuNtra started:"
for f in watchdog.pid quntra.pid telegram_bot.pid; do
    PID=$(cat $f 2>/dev/null || echo "?")
    echo "  $f: $PID"
done
./venv/bin/python scripts/paper_trading_status.py --telegram || true
