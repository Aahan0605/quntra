#!/bin/bash
# Prevents the Mac from idle-sleeping while QuNtra paper trading runs.
# Display may still sleep; the machine itself stays up.
# Usage: bash scripts/keep_mac_awake.sh    (Ctrl+C to stop)
echo "Keeping Mac awake for QuNtra paper trading…"
echo "Press Ctrl+C to stop (Mac can sleep again afterwards)"
caffeinate -i -m -s &
CAFFEINATE_PID=$!
echo "caffeinate PID: $CAFFEINATE_PID"
trap "kill $CAFFEINATE_PID 2>/dev/null; echo 'Mac can now sleep'; exit 0" INT
wait $CAFFEINATE_PID
