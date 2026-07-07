"""
Improved feature engineering — RESEARCH ENVIRONMENT ONLY.

Production (data/models/) is frozen for the 40-day paper gate; everything
here writes to data/models_research/ and nothing trades on it.

Hypotheses (2026-07 sweep):
  H1 volume features (OBV, volume spikes)
  H2 extended look-backs (RSI 7/21, SMA 10/200, 10/20d returns)
  H3 price position (Bollinger %B and width)
  H4 volatility regime (vol ratio, intraday range, overnight gap)
  H5 benchmark relative strength (kept — helped in production)
  H6 log returns

Anti-leakage: every feature is shift(1)'d; the 5-day forward label loses
its last 5 rows; the caller must purge the train/test boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

LOOK_FORWARD = 5


class ImprovedFeaturePipeline:
    """Research-phase feature builder. Strictly no look-ahead."""

    def build(self, prices: pd.DataFrame,
              bench: pd.Series | None = None,
              look_forward: int = LOOK_FORWARD
              ) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame(index=prices.index)
        close = prices["close"]
        high = prices["high"]
        low = prices["low"]
        vol = prices["volume"].astype(float)

        # Production baseline set
        df["rsi_14"] = ta.momentum.RSIIndicator(close, 14).rsi()
        df["macd"] = ta.trend.MACD(close).macd() / close
        df["atr_14"] = ta.volatility.AverageTrueRange(
            high, low, close, 14).average_true_range() / close
        df["vol_20"] = close.pct_change().rolling(20).std()
        df["ret_1d"] = close.pct_change()
        df["ret_5d"] = close.pct_change(5)

        # H1 volume
        obv = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        df["obv_ratio"] = obv / obv.rolling(20).mean().replace(0, np.nan)
        df["vol_spike"] = vol / vol.rolling(20).mean()

        # H2 extended windows
        df["rsi_7"] = ta.momentum.RSIIndicator(close, 7).rsi()
        df["rsi_21"] = ta.momentum.RSIIndicator(close, 21).rsi()
        df["ret_10d"] = close.pct_change(10)
        df["ret_20d"] = close.pct_change(20)

        # H3 price position
        bb = ta.volatility.BollingerBands(close, 20, 2)
        df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
        df["bb_pct"] = bb.bollinger_pband()

        # H4 volatility regime
        df["vol_ratio"] = df["vol_20"] / df["vol_20"].rolling(60).mean()
        df["hl_ratio"] = (high - low) / close
        df["gap"] = close / close.shift(1) - 1

        # H5 benchmark relative strength
        if bench is not None:
            b = bench.reindex(df.index).ffill()
            df["rel_nifty_5d"] = close.pct_change(5) - b.pct_change(5)
            df["rel_nifty_20d"] = close.pct_change(20) - b.pct_change(20)

        # H6 log returns
        df["logret_1d"] = np.log(close / close.shift(1))
        df["logret_5d"] = np.log(close / close.shift(5))

        # Trend
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        df["price_vs_sma20"] = close / sma20 - 1
        df["price_vs_sma50"] = close / sma50 - 1
        df["price_vs_sma200"] = close / sma200 - 1
        df["sma20_vs_sma50"] = sma20 / sma50 - 1
        df["dist_52w_high"] = close / close.rolling(
            252, min_periods=60).max() - 1

        # Strict anti-leakage: everything known only as of yesterday's close
        df = df.shift(1)

        fwd_ret = close.shift(-look_forward) / close - 1
        target = (fwd_ret > 0).astype(float)
        target.iloc[-look_forward:] = np.nan

        data = df.join(target.rename("target")).dropna()
        return data.drop(columns="target"), data["target"].astype(int)
