#!/usr/bin/env python3
"""
QuNtra process watchdog — keeps the scheduler AND the Telegram bot alive.

Checks every 60s; restarts a dead process, max 3 restarts per hour per
process (a persistent crash needs a human, not a crash loop). Run it
detached:

    nohup ./venv/bin/python scripts/watchdog.py >> logs/watchdog.log 2>&1 &
    echo $! > watchdog.pid

Pair with scripts/keep_mac_awake.sh so macOS idle-sleep can't kill
everything at once.
"""
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [WATCHDOG] %(message)s")
logger = logging.getLogger("quntra.watchdog")

CHECK_SECS = 60
MAX_RESTARTS_PER_HOUR = 3
PYTHON = str(ROOT / "venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

# name -> (pid file, launch argv, log file)
SERVICES = {
    "scheduler": (
        ROOT / "quntra.pid",
        [PYTHON, "scripts/scheduler.py", "--env", "config/secrets.env",
         "--log-file", "logs/quntra_paper.log"],
        ROOT / "logs" / "scheduler_stdout.log",
    ),
    "telegram_bot": (
        ROOT / "telegram_bot.pid",
        [PYTHON, "scripts/run_telegram_bot.py"],
        ROOT / "logs" / "telegram_bot.log",
    ),
}


def pid_alive(pid_file: Path) -> bool:
    """True when the PID in pid_file exists and is actually running.

    os.kill(pid, 0) is NOT enough: services the watchdog spawned become
    ZOMBIES of the watchdog when they die, and signal 0 still succeeds
    on a zombie. ps process-state is authoritative.
    """
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def reap_children() -> None:
    """Collect exit statuses of dead child services (clears zombies)."""
    try:
        while os.waitpid(-1, os.WNOHANG)[0] > 0:
            pass
    except ChildProcessError:
        pass


def start_service(name: str) -> bool:
    pid_file, argv, log_file = SERVICES[name]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        argv,
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid))
    logger.info("%s started (PID %d)", name, proc.pid)
    return True


DB_CONTAINER = "quntra-db"


def ensure_database() -> str:
    """Keep the Postgres container up whenever Docker is available.

    Returns a short status string. The scheduler uses Postgres by default;
    a stopped container (common after a Mac reboot if Docker auto-starts
    but the container doesn't) silently breaks every DB write, so the
    watchdog restarts it. If Docker itself is down, we can't fix it from
    here — surface it once so the operator knows.
    """
    if not shutil_which("docker"):
        return "docker-absent"
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "docker-daemon-down"
    running = subprocess.run(
        ["docker", "ps", "--filter", f"name={DB_CONTAINER}",
         "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    if DB_CONTAINER in running:
        return "ok"
    exists = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={DB_CONTAINER}",
         "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    if DB_CONTAINER in exists:
        subprocess.run(["docker", "start", DB_CONTAINER], capture_output=True)
        logger.warning("%s was stopped — restarted it", DB_CONTAINER)
        return "restarted"
    return "container-missing"


def shutil_which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)


def main() -> int:
    restarts: dict[str, list[datetime]] = {n: [] for n in SERVICES}
    alerted: set[str] = set()
    logger.info("Watchdog up — monitoring %s + %s every %ds",
                ", ".join(SERVICES), DB_CONTAINER, CHECK_SECS)
    while True:
        reap_children()
        db = ensure_database()
        if db in ("docker-daemon-down", "container-missing"):
            if "db" not in alerted:
                logger.error("Database unavailable (%s) — trading DB writes "
                             "will fail until fixed", db)
                _try_alert(f"🚨 WATCHDOG: database {db} — QuNtra can't "
                           f"persist trades. Start Docker / the quntra-db "
                           f"container.")
                alerted.add("db")
        else:
            alerted.discard("db")
        for name in SERVICES:
            pid_file, _, _ = SERVICES[name]
            if pid_alive(pid_file):
                continue
            now = datetime.now()
            restarts[name] = [t for t in restarts[name]
                              if now - t < timedelta(hours=1)]
            if len(restarts[name]) >= MAX_RESTARTS_PER_HOUR:
                if name not in alerted:
                    logger.error("%s hit %d restarts/hour — giving up until "
                                 "manual intervention", name,
                                 MAX_RESTARTS_PER_HOUR)
                    _try_alert(f"🚨 WATCHDOG: {name} crashed "
                               f"{MAX_RESTARTS_PER_HOUR}x in an hour — "
                               f"manual intervention needed")
                    alerted.add(name)
                continue
            logger.warning("%s not running — restarting", name)
            start_service(name)
            restarts[name].append(now)
            alerted.discard(name)
        time.sleep(CHECK_SECS)


def _try_alert(msg: str) -> None:
    try:
        from src.alerts.telegram_bot import TelegramAlerter
        TelegramAlerter.from_config().send(msg)
    except Exception:  # noqa: BLE001 — alerting is best-effort here
        pass


if __name__ == "__main__":
    sys.exit(main())
