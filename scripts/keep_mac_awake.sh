#!/bin/bash
# Prevents the Mac from IDLE-sleeping while QuNtra paper trading runs.
# Display may still sleep; the machine itself stays up — while these two
# conditions hold, neither of which caffeinate can enforce:
#   1. The lid stays open. Closing it force-sleeps the machine regardless
#      of any caffeinate assertion (macOS ignores them on lid-close).
#      scripts/watchdog.py's handle_missed_job() alerts + same-day
#      catches up when this happens, but can't prevent the gap itself.
#   2. It stays plugged into AC power. caffeinate's -s flag ("prevent
#      sleep on AC power") does NOTHING on battery — on battery a MacBook
#      still sleeps/dies on its own schedule. watchdog.py now pages you
#      at <=40% battery on AC-power loss for exactly this reason.
# The only complete fix for both is running this off the laptop entirely
# (infra/deploy_aws.sh is already written for that — ask to run it).
# Usage: bash scripts/keep_mac_awake.sh    (Ctrl+C to stop)
echo "Keeping Mac awake for QuNtra paper trading…"
echo "Press Ctrl+C to stop (Mac can sleep again afterwards)"
echo "Reminder: keep the lid OPEN and stay on AC POWER — see comments above."
caffeinate -i -m -s &
CAFFEINATE_PID=$!
echo "caffeinate PID: $CAFFEINATE_PID"
trap "kill $CAFFEINATE_PID 2>/dev/null; echo 'Mac can now sleep'; exit 0" INT
wait $CAFFEINATE_PID
