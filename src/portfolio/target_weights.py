"""Inverse-volatility target weights — the one strategy in this repo with
a real backtest behind it (Sharpe 1.14, 2022-2026, real NSE data).

Extracted from scripts/run_full_validation.py so the validated backtest and
the live allocator (src/portfolio/live_allocator.py) call the exact same
function. Two copies of "the strategy" is how a backtest quietly stops
describing what's actually trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 0.20


def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Redistribute any excess above `cap` (a fraction of the whole
    portfolio) proportionally among the not-yet-capped names, iterating
    until nothing exceeds it.

    A single clip-then-renormalize pass looks right but isn't: clipping
    the dominant name and dividing by the smaller post-clip sum can push
    it right back over the cap. In the 24-name universe this never fired
    (no real stock has near-zero volatility, so inverse-vol weights stay
    under ~5% and the 20% cap never binds — confirmed by rerunning
    scripts/run_full_validation.py, whose numbers are unchanged by this
    fix). It fires immediately with one very low-vol name, which is worth
    getting right before this code allocates real capital.
    """
    w = weights.copy()
    free = pd.Series(True, index=w.index)
    for _ in range(len(w)):
        over = free & (w > cap)
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free[over] = False
        pool = w[free].sum()
        if pool <= 0:
            break
        w[free] = w[free] + excess * (w[free] / pool)
    return w


def inverse_vol_weights(returns: pd.DataFrame,
                        weight_cap: float = WEIGHT_CAP) -> dict[str, float]:
    """Pure inverse-volatility, long-only, capped, normalized to sum to 1.

    `returns` should be trailing daily returns for the investable universe
    only — apply any vetoes (earnings blackout, fundamental flags, negative
    news) by column selection *before* calling this, so a vetoed ticker
    never receives a weight in the first place.
    """
    if returns.empty:
        return {}
    vol = returns.std()
    inv_vol = 1.0 / vol.replace(0, np.nan)
    w = inv_vol.fillna(0)
    if w.sum() == 0:
        return {}
    w = w / w.sum()
    return _apply_cap(w, weight_cap).to_dict()
