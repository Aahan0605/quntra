# QuNtra — Build Status

Last updated: 2026-07-06 (v4.0 hardening) · Test suite: **220 passing** (+3 live-network integration tests)

The system goal: Backtesting → **Paper trading (40-day gate)** → Live money.
Current position: **Parts A+B+v4.0 COMPLETE. Paper trading LIVE. Telegram control center LIVE on the operator's phone (chat_id captured 2026-07-06).**

## ✅ v4.0 hardening (2026-07-06)

- [x] First-contact chat_id capture + whitelist auth (verified live — operator authorized, /help delivered)
- [x] 30 Telegram commands incl. /help /trades /signals /regime /paper_progress /macro /chat
- [x] Auto-push: trade open/close, 08:45 morning briefing, 17:00 EOD, Friday 18:00 recap, risk levels 1/2/3, kill switch
- [x] Three resident processes: scheduler (`quntra.pid`), bot runner (`telegram_bot.pid`), watchdog (`watchdog.pid`, zombie-aware, max 3 restarts/hr)
- [x] `scripts/rotate_telegram_token.py` — **run this**: the token was exposed in a chat session
- [x] `scripts/keep_mac_awake.sh` (caffeinate) — run it or the Mac's idle-sleep can still pause everything

---

## ✅ PART A — local setup (completed 2026-07-06 on the Mac)

- [x] Python 3.10 venv + pinned deps (jugaad-data via `--no-deps`; psycopg2-binary added)
- [x] PostgreSQL 15 in Docker (`quntra-db`), Alembic migrated — 8 tables (+ `knowledge_items`)
- [x] Real data: **24/25 tickers × 5 years** (TATAMOTORS.NS delisted after 2025 demerger — effective universe is 24)
- [x] Models trained with an **honest OOS gate** = max(0.54, base rate + 1%): 3 deployed (ICICIBANK, BAJFINANCE, SUNPHARMA); other tickers get a NEUTRAL ML vote — coin-flip models don't trade
- [x] **VALIDATION PASSED on real data**: Sharpe **1.2276** (rf=0; 0.68 excess over 7%) · Max DD **−14.81%** · Calmar **1.0293** · 2.4% turnover — strategy simplified to pure inverse-vol after the momentum tilt worsened drawdown; NIFTY same-window comparison: 10.9%/yr, Sharpe 0.85, DD −15.8%
- [x] Scheduler running in paper mode (PID in `quntra.pid`), 13 IST jobs, NSE-holiday aware

## ✅ PART B — full vision build (completed 2026-07-06)

- [x] **B1 Hermes CEO orchestrator**: pre-market (research team → synthesis → watchlist → Telegram), market tick, post-market (lessons → knowledge), overnight batch (trainer + 9-task pipeline), weekly board report, monthly letter
- [x] **B2 Research team** (7): news / macro / sector / fundamental / geopolitical / company-events agents + research writer — live run: 52 s end-to-end, real earnings blackouts detected
- [x] **B3 Telegram command center**: 22 commands, dispatch never crashes, all logged to `system_state`
- [x] **B4 /note intelligence**: entity extraction → yfinance verification → relevance → macro-bias nudge → stored as USER_NOTE
- [x] **B5 KnowledgeManager**: 8 knowledge types, recall by keyword/regime/ticker/conditions, weekly digest
- [x] **B6 Overnight pipeline**: 9 prioritized tasks, per-task error isolation, pre-market draft by 6 AM
- [x] **B7 DailyTrainer**: rolling 90-day trade retrain, 10-day holdout, 54% gate + drift check, research env only, MLflow logged
- [x] **B8 Reporting**: daily 5 PM / weekly Sunday 8 PM / monthly 1st 9 AM — all metrics from the DB
- [x] **SignalCouncil** (was missing entirely): 5 auditable votes (technical/momentum/ML/macro/sector) → 0–12 score, ≥9 gate, 3-trade daily cap
- [x] **PaperTrader exit engine** (was a no-op): −2% stop / +4% target / 5-day time stop

## ⏳ PART C — the 40-day paper gate (day 1/40, started 2026-07-06)

Track daily: `python3 scripts/paper_trading_status.py`

- [ ] ≥ 40 trading days · rolling Sharpe > 1.0 · max DD > −15%
- [ ] Zero unrecovered crashes · kill switch fired + recovered at least once
- [ ] All 22 Telegram commands verified against the live bot
- [ ] Knowledge base ≥ 50 items (currently 9+ and growing nightly)

### Operator TODO (only human steps left)
1. **Message the Telegram bot** (@Sjebxhs_bot) once, then run:
   `./venv/bin/python scripts/complete_local_setup.py --from 5` — it will pick up the chat ID instructions (token already in `config/secrets.env`)
2. Keep the Mac (or move to a server) running so the scheduler stays up
3. Kite API keys stay **blank** until the paper gate passes

---

## Key files

| What | Where |
|---|---|
| Full setup instructions | `RUNBOOK.md` |
| Complete build history + decisions | `build_log.md` |
| Daily gate dashboard | `scripts/paper_trading_status.py` |
| Validation results (real data, all_pass) | `data/backtest_results/validation_post_fix.json` |
| Stop / start scheduler | `kill $(cat quntra.pid)` / `scripts/complete_local_setup.py --from 8` |

**Rule that overrides everything: capital preservation. No live money before every gate above passes.**
