# Indian market crashes — what happened, and what QuNtra does about it

Written to ground `src/risk/crash_risk.py`. The point of studying these is
not to predict the next one. It is to answer a narrower, answerable
question: **what did the tape look like on the way in, and can we detect
that in time to cut exposure?**

---

## The episodes

| Year | Event | Peak → trough | Depth | What actually broke |
|---|---|---|---|---|
| 1992 | Harshad Mehta securities scam | Apr → Aug 1992 | ~−43% | Bank receipts fraud funnelled interbank money into equities; unwound when discovered |
| 1996–98 | Political instability + Asian crisis | staggered | ~−40% | Coalition collapse, contagion from ASEAN currency devaluations |
| 2000–01 | Dot-com + Ketan Parekh | Feb 2000 → Sep 2001 | ~−57% | K-10 stocks financed by badla/pay-order leverage; broker default cascade |
| 2004 | Election shock (17 May) | single day | −11% intraday | Unexpected UPA win; policy uncertainty, forced margin unwind |
| 2006 | Global EM selloff (May–Jun) | ~1 month | ~−29% | Carry-trade unwind, FII flight |
| 2008 | Global financial crisis | Jan → Oct 2008 | **−60%** | Global credit seizure; FII outflows, leverage unwind |
| 2015–16 | China devaluation / global growth scare | Aug 2015 → Feb 2016 | ~−20% | Yuan devaluation, commodity collapse |
| 2018 | IL&FS / NBFC credit crisis | Aug → Oct 2018 | ~−15% | AAA issuer default froze NBFC funding; liquidity, not solvency |
| 2020 | COVID-19 | Jan → 23 Mar 2020 | **−38%** | Fastest ever; total-economy stop, correlations → 1 |
| 2024 | Election result (4 Jun) | single day | −5.9% | Exit-poll mispricing; recovered within weeks |

*Depths for 2008 and 2020 are measured on Nifty 50 from
`data/cache/NIFTY_LONG.csv`; earlier figures are Sensex, approximate.*

## What recurs

Reading across all ten, four things show up almost every time:

1. **Leverage, in whatever form the era allowed.** Badla in 1992 and 2000,
   FII carry in 2006, structured credit in 2008, NBFC short-term paper in
   2018. The instrument changes; the mechanism — forced selling into a
   falling market — does not.
2. **A volatility regime change before the worst of it.** Markets stop
   moving calmly before they collapse. This is the most reliably
   *measurable* precursor.
3. **Correlations converge to 1.** Diversification fails exactly when it is
   needed. A stock-picking system holding 5 "uncorrelated" names holds one
   position in a crash.
4. **The trigger is never the cause.** The trigger is unforecastable (a
   virus, an election, a broker default). The fragility that turns a
   trigger into a −60% drawdown builds visibly for months.

**Conclusion #4 is why QuNtra does not attempt to predict crashes.** It
monitors fragility, which is measurable, and ignores triggers, which are not.

## Why there is no ML model here

Ten crashes in thirty-three years. Any supervised classifier trained on ten
positive examples memorises ten dates. It would backtest beautifully and
fail on the eleventh crash, which will have a trigger absent from the
training set — as every crash in the table above did.

Additional reasons a model would mislead here:

- **The regime changes.** Badla financing no longer exists. Circuit
  breakers didn't exist in 1992. Features are not comparable across eras.
- **Survivorship and revision.** Index composition in 1992 barely overlaps
  today's.
- **Asymmetric costs.** A missed crash costs 40%+; a false alarm costs a
  few weeks of returns. That asymmetry argues for a blunt, transparent
  instrument that errs toward caution, not a finely-tuned one.

So: five transparent rules, coarse thresholds, auditable at any moment.

## Measured performance

From `tests/test_crash_risk.py`, on real Nifty data:

| Crash | De-risk signal | Drawdown already taken | Further downside avoided |
|---|---|---|---|
| 2008 GFC | 2008-01-22 | −22.1% | **−48.5%** |
| 2020 COVID | 2020-03-09 | −15.5% | **−27.2%** |

CRISIS fires on **1.5%** of all sessions across 2007–2026. Mean score in
calm years: 0.87 (2017), 1.82 (2021 recovery).

**Be clear about what this is.** It does not call the top. In both crashes
it gave up the first 15–22% of the fall. What it did was step aside before
the part that ends funds. In 2018 it signalled with only −2% of downside
left — effectively useless that time, which is exactly the honest record.

Milder corrections (2015 China, 2018 IL&FS) reach ELEVATED/HIGH but not
CRISIS. That is intended: they were corrections, not crashes.

## How it's wired in

`HermesCoordinator.crash_risk()` reads the benchmark series and scores it on
every 60-second market tick. The band sets an exposure multiplier:

| Score | Regime | Exposure | Effect |
|---|---|---|---|
| 0–34 | CALM | 1.00 | normal |
| 35–54 | ELEVATED | 0.60 | reduced sizing |
| 55–74 | HIGH | 0.25 | minimal new risk |
| 75–100 | CRISIS | 0.00 | **no new entries** |

At CRISIS, `run_market_session()` sets `can_trade = False` and logs the
rejection reason against every signal, so the block is auditable after the
fact.

Existing positions continue to be managed by the normal stop-loss engine —
the crash gate throttles *new risk*; it does not liquidate. Liquidating an
entire book on an indicator reading is a much stronger action than
declining to add, and is not something this indicator has earned.

If the benchmark file is missing, `crash_risk()` returns `None` and the gate
is inactive — logged as a warning. An unknown market must never be scored
as calm; the same principle makes `score_row()` return NaN rather than 0
when any input is missing.

## Known limitations

- **Index-only.** No breadth, no India VIX, no credit spreads, no FII flow.
  Those would likely improve lead time; all need data feeds QuNtra doesn't
  currently have. Breadth is the cheapest addition — the Nifty-200 cache
  already holds the constituents.
- **Single-day shocks are invisible.** 2024-06-04 (−5.9% in a day) never
  registered. A one-day gap cannot be de-risked against by a daily
  indicator; only position sizing helps there.
- **Thresholds are judgement, not optimisation.** Deliberately. Optimising
  them against ten events is the overfitting this design exists to avoid.
- **Validated on two crashes.** 2008 and 2020 are the only ones inside the
  available Nifty history. Two events is a weak basis for confidence, and
  no amount of clean code changes that.
