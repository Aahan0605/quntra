# QuNtra — Build Status

Last updated: 2026-07-05 · Test suite: **128 passing** · Commits: `c9c5493` → `91ebfe9`

The system goal: Backtesting → **Paper trading (40-day gate)** → Live money.
Current position: **all code is built and tested; waiting on real-data steps
that must run on your Mac** (the build sandbox cannot reach NSE/Yahoo/Telegram
and has no Docker).

---

## ✅ DONE — built, tested, committed

### Phase 0 — Foundation
- [x] 25-ticker NSE universe, deduped (`src/utils/universe.py`) — `verify_universe.py` prints "25 unique tickers confirmed"
- [x] Weekly rebalancer: 3% drift threshold, 20% turnover cap (`src/portfolio/rebalancer.py`) — simulated turnover well under the 300% limit
- [x] ICICI cost model loaded from `config/costs.env` (`src/utils/costs.py`) — no hardcoded costs anywhere
- [x] Backtest engine rewritten: real friction on traded notional, drift-aware weights, Calmar added
- [x] Circuit breaker recalibrated: −3% / −4.5% / −7% levels, 30-min cooldown, manual `/resume` for Level 3 (`src/risk/drawdown_circuit.py`)
- [x] Consecutive-loss kill switch: 3 losses → halt + mistake report (`src/risk/consecutive_loss_guard.py`)
- [x] Pinned runtime: `requirements-pinned.txt` + `environment.yml` (numpy 2.2.6, pandas 2.3.3, sklearn 1.7.2, xgboost 3.2.0)
- [x] Training pipeline with 54% OOS gate, leakage-safe features (`src/ml/train_clean_models.py`)
- [x] Validation runner with hard gates + diagnostics + tearsheet (`scripts/run_full_validation.py`) — refuses to print PASS on synthetic data

### Phase 1 — Infrastructure
- [x] UnifiedDataFetcher: jugaad-data / Bharat-SM-Data / yfinance-global-only routing + data quality validator (`src/utils/data_fetcher.py`)
- [x] Indicator migration: pandas-ta is dead → MIT `ta` library; manual RSI/ADX fallbacks were mathematically wrong (up to 19 RSI pts off) and are now Wilder-correct, with equivalence tests
- [x] Database: 7 tables (trades, signals, agent_credibility, backtest_results, price_data, research_notes, system_state) via SQLAlchemy + Alembic; PostgreSQL in prod, SQLite fallback
- [x] Hermes coordinator: pre-market / market-session / post-market / overnight sequences, DB-backed state (`src/governor/hermes.py`)
- [x] QuNtra Brain: persistent memory + agent credibility (×1.05 / ×0.95, floor 0.1, ceiling 3.0) (`src/governor/brain.py`)
- [x] 24/7 scheduler: 11 jobs, all IST, skips NSE holidays, `--dry-run` prints 11/11 (`scripts/scheduler.py`)
- [x] Telegram command center: 6 alert types + /status /pause /resume /report /override /halt (`src/alerts/telegram_bot.py`)

### Phase 2 — Execution (code)
- [x] PaperTrader: live-price simulated fills, 0.05%/side slippage, ICICI fees, every trade in DB (`src/execution/paper_trader.py`)
- [x] KiteOMS: order state machine, signal-hash dedup, daily capital cap, 3-trades/day cap — interface-identical to PaperTrader (`src/execution/kite_oms.py`)
- [x] End-to-end smoke test passing (`scripts/smoke_test.py`)
- [x] Monitoring: `scripts/paper_trading_status.py` (daily gate dashboard) + `scripts/paper_performance_report.py` (weekly)
- [x] One-command completion runner: `scripts/complete_local_setup.py` (Steps 0–8, hard-stops at failed gates, resumable with `--from N`)
- [x] Also fixed along the way: pre-existing syntax error in `fundamental.py`; pyOpenSSL/kiteconnect conflict; Bharat-SM-Data pinned to 3.0.0 (4.x needs Python 3.12)

---

## ⏳ PENDING — requires your Mac, in this order

Run everything below with **one command** in Terminal:

```bash
cd ~/Claude/Projects/quntra && python3 scripts/complete_local_setup.py
```

- [ ] **1. PostgreSQL** — Docker container + Alembic migration (auto-falls back to SQLite if Docker missing)
- [ ] **2. Fetch real market data** — 5 years OHLCV for 25 tickers (gate: ≥23/25) — *nothing downstream can run without this*
- [ ] **3. Train the 25 models** — 54% OOS accuracy gate (gate: ≥20/25 pass)
- [ ] **4. Full validation on real data** — ALL THREE must pass: Sharpe > 1.00 after costs · Max DD > −15% · Calmar > 0.70. **Hard stop if any fail.**
- [ ] **5. Telegram secrets** — human step: @BotFather → token + chat ID into `config/secrets.env` (script prints instructions; optional but recommended)
- [ ] **6. Kite API keys** — intentionally BLANK until the paper gate passes
- [ ] **7. Start the scheduler** — paper mode, ₹25,000 simulated, max 3 trades/day, min score 9
- [ ] **8. The 40-trading-day paper gate** (~8 weeks, cannot be shortcut):
      ≥40 days · paper Sharpe > 1.0 · max DD better than −15% · zero unrecovered crashes · kill switch fired + recovered correctly
      Track daily: `python3 scripts/paper_trading_status.py`
- [ ] **9. Live capital** — only after gate 8 passes: ₹10,000–₹25,000, human approval per trade

---

## Key files

| What | Where |
|---|---|
| Full setup instructions | `RUNBOOK.md` |
| Complete build history + blockers | `build_log.md` |
| One-command completion | `scripts/complete_local_setup.py` |
| Daily gate dashboard | `scripts/paper_trading_status.py` |
| Validation results (after step 4) | `data/backtest_results/validation_post_fix.json` |

**Rule that overrides everything: capital preservation. No live money before every gate above passes.**
