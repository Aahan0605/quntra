# QuNtra Runbook — from this commit to a running paper-trading system

Everything below runs on **your Mac** (or a server). The build sandbox had
no market-data network access, so three steps are yours to run — each is
one command.

## 1. Environment (one-time)

```bash
cd ~/Claude/Projects/quntra
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-pinned.txt
python3 scripts/verify_imports.py     # expect: "All required imports OK"
python3 scripts/verify_universe.py    # expect: "25 unique tickers confirmed"
```

## 2. Data cache (unblocks P0-1 and P0-5)

```bash
python3 scripts/fetch_data_cache.py --years 4
# expect: "25/25 tickers cached"
```

## 3. Train the 25 models (P0-1)

```bash
python3 -m src.ml.train_clean_models
python3 scripts/verify_models.py      # expect: "25/25 models loaded successfully"
```

Models that fail the 54% OOS gate land in `data/models/rejected/` — that is
the gate working, not a bug. Retrain after more data or drop the ticker.

## 4. Validation (P0-5 — the Phase 0 gate)

```bash
python3 scripts/run_full_validation.py
# All 3 must PASS: Sharpe > 1.0, MaxDD > -15%, Calmar > 0.70
# Results -> data/backtest_results/validation_post_fix.json
```

Do NOT start paper trading until all three pass on real data.

## 5. PostgreSQL (production DB)

```bash
docker run -d --name quntra-db \
  -e POSTGRES_USER=quntra -e POSTGRES_PASSWORD=quntra_dev \
  -e POSTGRES_DB=quntra -p 5432:5432 postgres:15

cp config/secrets.env.example config/secrets.env   # fill in POSTGRES_URL etc.
python3 -m alembic upgrade head                    # creates all 7 tables
```

Without Docker/Postgres the system falls back to SQLite at `data/quntra.db`
automatically — fine for the first paper-trading days.

## 6. Telegram (alerts + remote control)

1. Message @BotFather -> `/newbot` -> copy the token into
   `config/secrets.env` as `TELEGRAM_BOT_TOKEN`.
2. Message your new bot once, then visit
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy your
   `chat.id` into `TELEGRAM_CHAT_ID`.
3. Commands available: /status /pause /resume /report /override /halt

## 7. Start the 40-day paper run (Phase 2 gate)

```bash
python3 scripts/scheduler.py --dry-run   # sanity check: 11 jobs, IST, holidays
nohup python3 scripts/scheduler.py > logs/scheduler.log 2>&1 &
```

`.env` ships with `PAPER_TRADE=true`, ₹25,000 simulated capital, max 4
trades/day, min signal score 9. The scheduler skips NSE holidays
automatically and survives reboots if you add it to launchd/systemd.

### Paper gate (all must hold before any real money)
- 40 trading days completed (~8 calendar weeks)
- Paper Sharpe > 1.0 · Max DD better than -15%
- Zero unrecovered crashes
- Every 3-loss kill-switch event handled correctly

## Known blockers (documented during the build)

- **BLOCKER: market data unreachable from build sandbox** (NSE + Yahoo
  blocked by proxy) — that's why steps 2-4 run on your machine.
- **BLOCKER: KITE_API_KEY not in secrets.env** — live OMS (`KiteOMS.connect()`)
  raises with a clear message until you add Zerodha credentials. Irrelevant
  until the paper gate passes.
- **pandas-ta is dead** (pulled from PyPI, GitHub repo private) — QuNtra
  uses the MIT `ta` library instead; equivalence enforced by
  `tests/test_indicator_equivalence.py`.
- **Bharat-SM-Data pinned to 3.0.0** — 4.x needs Python 3.12+.
