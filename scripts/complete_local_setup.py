#!/usr/bin/env python3
"""
QuNtra completion loop — one command, runs Steps 1-8 on YOUR machine.

    python3 scripts/complete_local_setup.py            # full run
    python3 scripts/complete_local_setup.py --from 3   # resume at step 3
    python3 scripts/complete_local_setup.py --no-start # everything except
                                                       # launching the scheduler

Steps:
  1. PostgreSQL via Docker (falls back to SQLite if Docker missing)
  2. Fetch real market data (jugaad-data -> yfinance fallback, 23/25 gate)
  3. Train 25 ML models (54% OOS gate; 20+ must pass)
  4. Full validation (Sharpe > 1.0, DD > -15%, Calmar > 0.70 — hard gate)
  5. Telegram check (pauses with instructions if secrets missing)
  6. Secrets verification
  7. Scheduler dry-run (13/13 jobs)
  8. Start paper trading (nohup, PID file, log tail)

The script STOPS at any failed gate — per policy, never skip a phase gate.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

GREEN, RED, YELLOW, END = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def banner(step: int, title: str):
    print(f"\n{'=' * 60}\nSTEP {step} — {title}\n{'=' * 60}")


def ok(msg):
    print(f"{GREEN}✓ {msg}{END}")


def fail(msg):
    print(f"{RED}✗ {msg}{END}")


def warn(msg):
    print(f"{YELLOW}! {msg}{END}")


def run(cmd, timeout=None, capture=False):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, timeout=timeout,
                          capture_output=capture, text=True)


def sh_ok(cmd, timeout=None) -> bool:
    return run(cmd, timeout=timeout, capture=True).returncode == 0


# ----------------------------------------------------------------------- #

def step0_requirements() -> bool:
    banner(0, "Machine requirements")
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        fail(f"Python {v.major}.{v.minor} < 3.10")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    if not (ROOT / ".git").exists():
        fail("Not inside the quntra repo")
        return False
    ok("Inside quntra repo")
    secrets = ROOT / "config" / "secrets.env"
    if not secrets.exists():
        shutil.copy(ROOT / "config" / "secrets.env.example", secrets)
        warn("config/secrets.env created from example — fill in tokens later")
    ok("config/secrets.env exists")
    r = run("pip install -q -r requirements-pinned.txt", timeout=900)
    if r.returncode != 0:
        fail("pip install of pinned requirements failed")
        return False
    # jugaad-data pins beautifulsoup4==4.9.3 which conflicts with yfinance;
    # it runs fine on modern bs4, so install it without its dep resolution
    r = run("pip install -q --no-deps jugaad-data==0.28", timeout=300)
    if r.returncode != 0:
        fail("pip install of jugaad-data failed")
        return False
    ok("Pinned requirements installed (jugaad-data via --no-deps)")
    return True


def step1_postgres() -> bool:
    banner(1, "PostgreSQL")
    if shutil.which("docker") is None or not sh_ok("docker info", timeout=20):
        warn("Docker unavailable — falling back to SQLite (data/quntra.db). "
             "Fine for paper trading; install Docker later for production.")
        os.environ.pop("POSTGRES_URL", None)
    else:
        if not sh_ok("docker ps | grep -q quntra-db"):
            if sh_ok("docker ps -a | grep -q quntra-db"):
                run("docker start quntra-db")
            else:
                run("docker run -d --name quntra-db "
                    "-e POSTGRES_USER=quntra -e POSTGRES_PASSWORD=quntra_dev "
                    "-e POSTGRES_DB=quntra -p 5432:5432 "
                    "--restart unless-stopped postgres:15")
            time.sleep(8)
        os.environ["POSTGRES_URL"] = \
            "postgresql+psycopg2://quntra:quntra_dev@localhost:5432/quntra"
        ok("quntra-db container running")

    r = run("python3 -m alembic upgrade head", timeout=120)
    if r.returncode != 0:
        fail("Alembic migration failed")
        return False

    from sqlalchemy import inspect
    from src.db.session import get_engine
    tables = set(inspect(get_engine()).get_table_names())
    need = {"trades", "signals", "agent_credibility", "backtest_results",
            "price_data", "research_notes", "system_state"}
    missing = need - tables
    if missing:
        fail(f"Missing tables: {missing}")
        return False
    ok(f"All 7 tables confirmed ({'PostgreSQL' if os.environ.get('POSTGRES_URL') else 'SQLite'})")
    return True


def step2_data() -> bool:
    banner(2, "Fetch real market data (25 tickers, 5 years)")
    r = run("python3 scripts/fetch_data_cache.py --years 5 --sleep 2",
            timeout=3600)
    if r.returncode != 0:
        fail("Data fetch gate failed (< 23/25 tickers)")
        return False
    run("python3 scripts/check_data_quality.py", timeout=300)
    ok("Data cache populated")
    return True


def step3_train() -> bool:
    banner(3, "Train 25 ML models (honest OOS gate; see verify_models.py)")
    r = run("python3 -m src.ml.train_clean_models --verbose", timeout=7200)
    if r.returncode != 0:
        warn("Some tickers failed to train — checking the gate…")
    v = run("python3 scripts/verify_models.py", capture=True, timeout=120)
    print(v.stdout)
    if v.returncode == 0:
        ok("Training gate passed (>=20 trained, deployed models verified)")
        return True
    fail("Training gate failed — read verify_models.py output above. "
         "Run scripts/check_data_quality.py and retrain failing tickers.")
    return False


def step4_validate() -> bool:
    banner(4, "Full validation — the Phase 0 gate")
    r = run("python3 scripts/run_full_validation.py --verbose --save-tearsheet",
            timeout=1800)
    if r.returncode != 0:
        fail("VALIDATION FAILED — do not proceed. Diagnostics:\n"
             "  a) Sharpe low  -> check turnover in the verbose output\n"
             "  b) DD deep     -> check circuit-breaker thresholds\n"
             "  c) Calmar low  -> check cost model for double counting\n"
             "Fix, then rerun: python3 scripts/complete_local_setup.py --from 4")
        return False
    ok("All 3 targets PASS — Sharpe > 1.0, DD > -15%, Calmar > 0.70")
    return True


def _load_secrets_env():
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / "secrets.env")
    load_dotenv(ROOT / ".env")


def step5_telegram() -> bool:
    banner(5, "Telegram bot")
    _load_secrets_env()
    if not (os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")):
        warn("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing.\n"
             "  1. Telegram -> @BotFather -> /newbot -> copy token\n"
             "  2. Message your bot once, then open\n"
             "     https://api.telegram.org/bot<TOKEN>/getUpdates\n"
             "     and copy chat.id\n"
             "  3. Add both to config/secrets.env\n"
             "  4. Rerun: python3 scripts/complete_local_setup.py --from 5\n"
             "Continuing WITHOUT Telegram (alerts land in logs only).")
        return True  # soft gate — paper trading works without alerts
    from src.alerts.telegram_bot import TelegramBot
    bot = TelegramBot()
    if bot.send_message("QuNtra Telegram test — system online"):
        ok("Test message sent — check your Telegram")
        return True
    fail("Telegram send failed — verify token with "
         "https://api.telegram.org/bot<TOKEN>/getMe")
    return False


def step6_secrets() -> bool:
    banner(6, "Secrets verification")
    _load_secrets_env()
    required = []
    if shutil.which("docker"):
        required.append("POSTGRES_URL")
    missing = [k for k in required if not os.getenv(k)]
    soft = [k for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
            if not os.getenv(k)]
    if missing:
        fail(f"MISSING required: {missing}")
        return False
    if soft:
        warn(f"Optional (recommended) missing: {soft}")
    if os.getenv("PAPER_TRADE", "true").lower() == "false":
        fail("PAPER_TRADE=false — NEVER before the 40-day gate. Fix .env.")
        return False
    gitignore = (ROOT / ".gitignore").read_text()
    if "secrets.env" not in gitignore:
        fail("config/secrets.env not gitignored!")
        return False
    ok("All required secrets present; PAPER_TRADE=true; secrets gitignored")
    return True


def step7_dryrun() -> bool:
    banner(7, "End-to-end dry-run + smoke test")
    r = run("python3 scripts/scheduler.py --dry-run", timeout=300)
    if r.returncode != 0:
        fail("Dry-run failed — read the traceback above, fix, rerun --from 7")
        return False
    ok("13/13 jobs passed")
    r = run("python3 scripts/smoke_test.py", timeout=300)
    if r.returncode != 0:
        fail("Smoke test failed — fix the integration issue, rerun --from 7")
        return False
    ok("Smoke test passed — all 5 components integrate")
    return True


def step8_start() -> bool:
    banner(8, "Start paper trading")
    (ROOT / "logs").mkdir(exist_ok=True)
    if sh_ok("pgrep -f 'scripts/scheduler.py' | grep -v $$ > /dev/null"):
        warn("Scheduler already running — not starting a second instance")
        return True
    proc = subprocess.Popen(
        [sys.executable, "scripts/scheduler.py",
         "--env", "config/secrets.env",
         "--log-file", "logs/quntra_paper.log"],
        stdout=open(ROOT / "logs" / "scheduler_stdout.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (ROOT / "quntra.pid").write_text(str(proc.pid))
    time.sleep(5)
    if proc.poll() is not None:
        fail("Scheduler died on startup — read logs/scheduler_stdout.log")
        return False
    ok(f"Scheduler running, PID {proc.pid} (saved to quntra.pid)")
    print("\nFirst-day checkpoints (IST): 06:00 pre-market · 09:30 session\n"
          "· 15:30 post-market · 17:00 EOD Telegram · 22:00 overnight batch\n"
          "Weekly: python3 scripts/paper_performance_report.py\n"
          "Stop:   kill $(cat quntra.pid)")
    return True


STEPS = [
    (0, "requirements", step0_requirements),
    (1, "postgres", step1_postgres),
    (2, "data", step2_data),
    (3, "train", step3_train),
    (4, "validate", step4_validate),
    (5, "telegram", step5_telegram),
    (6, "secrets", step6_secrets),
    (7, "dryrun", step7_dryrun),
    (8, "start", step8_start),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start_from", type=int, default=0)
    ap.add_argument("--no-start", action="store_true",
                    help="run everything except step 8")
    args = ap.parse_args()

    for num, name, fn in STEPS:
        if num < args.start_from:
            continue
        if num == 8 and args.no_start:
            print("\n--no-start: skipping step 8. Launch later with:\n"
                  "  python3 scripts/complete_local_setup.py --from 8")
            break
        if not fn():
            fail(f"STOPPED at step {num} ({name}). Fix and rerun with "
                 f"--from {num}")
            return 1

    print(f"\n{GREEN}{'=' * 60}\nCOMPLETION LOOP FINISHED\n"
          f"Paper trading is live. The 40-day gate is now ticking.\n"
          f"{'=' * 60}{END}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
