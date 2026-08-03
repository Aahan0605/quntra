"""scripts/watchdog.py — battery_status() parsing (subprocess mocked).

caffeinate's -s flag only holds sleep off while on AC power; on battery it
does nothing, so a laptop left on battery overnight can fully power off
regardless of any sleep assertion. This pins the pmset-output parser that
backs the low-battery alert.
"""

from unittest.mock import MagicMock, patch

import scripts.watchdog as watchdog


def _run(stdout):
    return MagicMock(stdout=stdout)


def test_battery_status_on_ac_power():
    out = "Now drawing from 'AC Power'\n -InternalBattery-0\t85%; charged; 0:00 remaining present: true\n"
    with patch("subprocess.run", return_value=_run(out)):
        on_ac, pct = watchdog.battery_status()
    assert on_ac is True
    assert pct == 85


def test_battery_status_on_battery_low():
    out = "Now drawing from 'Battery Power'\n -InternalBattery-0\t59%; discharging; 2:55 remaining present: true\n"
    with patch("subprocess.run", return_value=_run(out)):
        on_ac, pct = watchdog.battery_status()
    assert on_ac is False
    assert pct == 59


def test_battery_status_fails_open_on_error():
    """No pmset (e.g. non-macOS) must not crash the watchdog loop, and
    must not fire a false low-battery alarm."""
    with patch("subprocess.run", side_effect=FileNotFoundError("no pmset")):
        on_ac, pct = watchdog.battery_status()
    assert on_ac is True
    assert pct is None


def test_low_battery_alert_threshold():
    assert watchdog.BATTERY_ALERT_PCT >= 30, (
        "must alert with real time left to act, not moments before shutdown")
