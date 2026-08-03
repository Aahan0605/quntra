# QuNtra — review as if I were the CEO of the fund

Date: 2026-07-28. Written against measured numbers from this repo, not
impressions. Every figure below is reproducible from the commands cited.

---

## Verdict

You have built genuinely impressive **machinery** around an **unproven
edge**, and until last week the machinery was silently broken in ways that
made the edge unmeasurable anyway.

That is a better position than it sounds. Most people building this get the
edge wrong *and* have no infrastructure to discover it. You have the
infrastructure. What you do not yet have is evidence that the strategy makes
money, and the honest reading of the numbers is that it currently does not.

**No real capital should touch this system today.** Not because it is badly
built, but because nothing in it has yet demonstrated an edge that survives
statistical scrutiny.

---

## Finding 1 — the edge is not established (severity: existential)

194 tickers were trained; 25 cleared the accuracy gate. At 194 trials,
**~10 passes are expected from pure noise**. Applying Benjamini-Hochberg at
FDR q=0.10 across the full family:

```
./venv/bin/python -m src.ml.multiple_testing data/models_nifty200
  Trials run                        194
  Passed the naive per-model gate    25
  Survive BH at FDR q=0.1             0
```

**Zero.** The same audit on the models actually deployed
(`data/models`): 24 trials, 3 passed the gate, **0 survive**.

The per-ticker "honest OOS gate" was a good instinct, and it is not the
problem. The problem is that it answers a single-model question while the
pipeline asks it 194 times. Nothing in the system accounted for the number
of attempts.

This does not prove the models are worthless. It proves the evidence
gathered so far **cannot distinguish them from luck** — which, for capital
allocation, is the same thing.

*Now enforced:* `SignalCouncil._deployed_model()` loads only FDR survivors.
Since that set is currently empty, every ML vote is NEUTRAL. The system is
now honest about knowing nothing, which is the correct state.

## Finding 1b — the signal-council strategy backtests to flat-to-negative, and the index beats it by 53 points (severity: existential)

Finding 1 showed the ML component has no measurable edge. This finding
tests the whole strategy — not just the model — against 5 years of real
price history, by replaying the *actual* production scoring functions
(`SignalCouncil._technical_vote`, `_momentum_vote`, `_sector_votes`,
imported directly, not reimplemented) with the real exit rules
(−2%/+4%/5-day) and real cost model.

```
./venv/bin/python scripts/backtest_signal_council.py
```

24 tickers, 2021-07-23 → 2026-07-21, 725 trades:

| | Signal-council strategy | Buy-and-hold NIFTY, same window |
|---|---|---|
| Total return | **−0.51%** | **+52.55%** |
| CAGR | −0.11% | 9.01% |
| Sharpe | **−0.008** | 0.69 |
| Max drawdown | −8.9% | −17.2% |
| Win rate | 48% | — |

Five years of active, costed trading returned less than a savings account,
while simply holding the index made 52 points. The lower drawdown (−8.9% vs
−17.2%) is the one thing working — the exit discipline (Finding 3) limits
damage — but there is no return to protect.

`ml` and `macro` votes were pinned NEUTRAL (no historical data exists to
replay them; see the script's docstring) — this makes the gate *harder* to
clear than live trading, so if anything this is a conservative floor, not a
worst case. It should not be read as "the real system would do better" —
the FDR result in Finding 1 says the ml vote contributes nothing anyway,
and macro/sector history isn't available to check the other direction.

**One specific, interesting sub-finding:** the strategy selects for recent
strength (technical + momentum both need to be near-maximal to clear the
gate). Selected trades net roughly breakeven-to-negative, while *random*
entries through the same exit machinery net slightly positive (Finding 2).
That's consistent with well-documented short-term reversal in equities —
the council may be systematically buying into short-term overextension.
Worth a follow-up study; not a fix in itself.

## Finding 2 — costs consume ~90% of gross return (severity: high)

Simulating the actual exit rules over 9,848 random entries across 8 large
caps on real price paths:

| | gross / trade | net after costs |
|---|---|---|
| with −2% / +4% / 5-day exits | +0.372% | **+0.039%** |
| plain 5-day hold, no stops | +0.216% | −0.118% |

Round-trip friction is ~33 bps (STT 10, exchange 3, GST, slippage 5/side).
Gross edge on random entries is 37 bps. **Net: 4 bps per trade.**

Critically, these costs are *proportional*, not fixed — STT and slippage
scale with notional. Raising capital from ₹25,000 does **not** improve this
ratio. The only fix is a larger gross edge or fewer, higher-conviction
trades.

Note what this also says: on random entries in a bull market the system nets
approximately zero. Everything therefore rests on signal selection adding
real edge — which is exactly what Finding 1 says is unproven.

## Finding 3 — the exit rules are good, and I was wrong to doubt them

Worth recording because it cuts against intuition. The −2% stop sits inside
the noise band (large-cap daily vol is 1.3–1.6%; a −2% move occurs within 5
days ~30% of the time), and stops fire 2.35× more often than targets
(30.6% vs 13.0%). That looks fatal.

It is not. Measured against a plain 5-day hold, the exit machinery **adds
+0.156 pct-points per trade and cuts volatility from 3.19% to 2.18%**. The
stops are doing real work. Leave them alone.

## Finding 4 — position concentration is extreme (severity: high)

Every position taken has been 27–32% of capital:

```
RELIANCE.NS  ₹7,978  31.9%      TCS.NS  ₹6,778  27.1%
ICICIBANK.NS ₹7,257  29.0%
```

Three positions ≈ **88% of capital in three Nifty large caps**. In the 2008
and 2020 episodes documented in `docs/INDIAN_MARKET_CRASHES.md`, large-cap
correlations converged to ~1. This is not a three-position portfolio in a
crash; it is one position with three names.

`MAX_TRADES_PER_DAY=3` limits *turnover*, not *concentration*. There is no
per-position size cap and no exposure cap anywhere in the config.

**Recommend:** cap any single position at 10% of capital and gross exposure
at 60%, and require the crash-risk exposure multiplier to scale sizing.

## Finding 5 — the "real-time market data" claim is not true (severity: high)

Kite returns `TokenException` — it is not connected and contributes nothing.
All data comes from yfinance, which for Indian equities is **delayed**, not
real-time. The scheduler runs a 60-second `market_loop` against delayed
quotes.

A 60-second decision loop consuming delayed data is not a fast system; it is
a slow system that believes it is fast. Either connect Kite properly (fresh
token daily, ~07:30 IST expiry) or slow the loop to match the data. Do not
leave it in the current state, where the architecture implies a latency the
data cannot support.

## Finding 6 — single point of failure (severity: medium)

Everything runs on one Mac, dependent on `caffeinate` to prevent idle-sleep
from halting trading. macOS sleep already caused prior misfires (commit
`4c005e4`). AWS deployment scripts exist and are unused.

Acceptable for a paper gate. Unacceptable for real capital.

## Finding 7 — `price_data` is populated but stale and unused (severity: medium)

*Corrected from an earlier draft of this review, which claimed these tables
were empty. That was wrong: it read `pg_stat_user_tables.n_live_tup`, a
planner estimate that stays stale until autovacuum runs, and reported it as
a row count. `STATUS.md`'s claims were accurate; my audit was not.*

The true counts:

| table | rows | note |
|---|---|---|
| `price_data` | 29,688 | 24 tickers, 2021-07-06 → **2026-07-03** |
| `knowledge_items` | 9 | as documented |
| `research_notes` | 33 | growing |

The real issues are narrower but still worth fixing:

1. **`price_data` stops at 2026-07-03** — 25 days stale, having gone quiet
   around when paper trading began.
2. **It covers 24 tickers, not the 194-name Nifty-200 universe** the system
   now trades.
3. **The live path doesn't read it.** `cache_loader` reads CSVs from
   `data/cache/`. So Postgres holds a partial, stale copy of data the
   trading loop gets elsewhere — two sources of truth, one silently rotting.

Pick one source. If it is Postgres, backfill and keep it current; if CSVs,
stop writing `price_data` so nobody trusts it later.

**The process lesson stands regardless:** `STATUS.md` was hand-maintained,
and a status file nobody regenerates is how a wedged scheduler read as
healthy for five days. `scripts/generate_status.py` now derives it live —
and would have caught both the outage and my own bad row counts.

---

## What is genuinely good

Said plainly, because it is unusual:

- **The paper gate exists at all**, with pre-committed pass criteria and a
  `LIVE_TRADING=false` guard. Most people skip straight to real money.
- **Honest failure modes throughout**: models that fail the gate go to
  `rejected/` rather than being quietly reused; the drawdown circuit fails
  the gate rather than fudging it.
- **314 passing tests** on a solo quant project.
- **Auditable decisions**: the 5-vote council writes every vote to the DB,
  so any trade can be explained after the fact. Many funds cannot do this.
- **Costs modelled explicitly** in `config/costs.env`, including slippage —
  a very common omission that turns losing strategies into "winners".

---

## What I would do, in order

1. **Do not deploy capital.** The strategy backtests to −0.51% over 5 years
   against +52.55% for buy-and-hold. Nothing has demonstrated edge.
2. **Stop paper-trading the signal-council strategy as-is.** Finding 1b
   isn't a tuning problem — it's a full-strategy backtest with real exits
   and real costs that loses to holding cash. Retesting it with different
   thresholds risks p-hacking the same 194-ticker problem from Finding 1.
3. **Fix concentration** — done (10% cap) — though the strategy it was
   protecting has no return to protect either way.
4. **Rebuild around what actually worked.** See "the two paths" below —
   the passive allocator (Sharpe 1.14, real) and the crash overlay
   (measurably avoided −48.5% in 2008, −27.2% in 2020) are the only two
   things in this repo with real evidence behind them.
5. **Generate STATUS.md from the DB.** Never hand-maintain it again.
6. **Rotate the Telegram token and lock down Postgres** — see
   `docs/SECURITY_REVIEW.md`.

## The two paths forward, concretely

**A — Passive allocator + crash overlay (lowest risk, ships fastest).**
Trade the already-validated inverse-vol portfolio
(`scripts/run_full_validation.py`, Sharpe 1.14, real 4-year backtest) as the
core holding, and use `src/risk/crash_risk.py` purely to scale total
exposure down in a CRISIS regime. Retire the signal-council's stock-picking
entirely — Finding 1b says it isn't adding anything to defend. This is a
smart-beta index-tracker with a tail-risk brake, not a stock-picker. It is
the only strategy in this repo backed by a real backtest that beats cash.

**B — Repurpose the 5-agent apparatus as risk flags, not alpha signals.**
The news/macro/sector/fundamental agents already exist and already write to
the DB. Redirect them from "pick which stock to buy" (proven not to work)
to "which stocks to avoid/underweight this week" inside path A's allocator
— earnings blackouts and fundamental red flags are exactly this pattern
already (`_fundamental_vote` is a veto-only −1, never a bonus). Vetoes are
a much lower bar to clear statistically than 5-day direction prediction,
because you only need "is this noticeably bad" not "will this go up."

**What I would not do:** keep tuning the signal-council's thresholds or add
more agents to it. Finding 1b tested the strategy as actually built and it
lost to holding cash; the more promising path is the machinery you already
have for path A, not more machinery bolted onto path B's failed premise.

## The uncomfortable question

The system's central premise is that a 5-vote council can predict 5-day
direction on Nifty stocks. The multiple-testing audit says the ML component
of that premise is unsupported by the evidence gathered.

That is worth sitting with before writing more code. The most valuable
thing here may not be the predictor at all — it may be the risk machinery:
the crash overlay, the cost model, the circuit breakers, the audit trail.
A system that reliably *avoids* the −60% years and captures beta cheaply is
a real product. A system that predicts tomorrow is, on this evidence, not
yet one.
