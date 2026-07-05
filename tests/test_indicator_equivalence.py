"""
Task 1-3 — indicator library migration tests.

NOTE: pandas-ta was withdrawn from PyPI and its GitHub repo made private
(commercial relaunch), so QuNtra standardizes on the MIT-licensed `ta`
library instead. These tests enforce:

  1. `ta` is installed and is the active indicator source (HAS_TA).
  2. Manual fallback implementations (used only if `ta` were missing)
     stay numerically equivalent to the library.
  3. The feature pipeline still produces 50+ features.
"""
import numpy as np
import pandas as pd
import pytest

import src.ml.features.technical as technical
from src.ml.features.technical import TechnicalFeatures


@pytest.fixture(scope="module")
def ohlcv():
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.bdate_range("2024-01-01", periods=n)
    ret = rng.normal(0.0004, 0.015, n)
    close = 2500 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_ta_library_is_active():
    """The pinned environment must use the ta library, not fallbacks."""
    assert technical.HAS_TA, "ta library missing — indicator source degraded"


def test_manual_rsi_matches_ta(ohlcv):
    from ta.momentum import RSIIndicator
    lib = RSIIndicator(close=ohlcv["close"], window=14).rsi()
    manual = TechnicalFeatures._manual_rsi(ohlcv["close"], 14)
    diff = (lib - manual).abs().dropna().tail(300)
    assert diff.max() < 1.0, f"manual RSI diverges from ta: max diff {diff.max():.4f}"


def test_manual_adx_close_to_ta(ohlcv):
    """Manual ADX uses SMA smoothing vs Wilder's — allow loose tolerance,
    but directionally they must correlate strongly."""
    from ta.trend import ADXIndicator
    lib = ADXIndicator(ohlcv["high"], ohlcv["low"], ohlcv["close"], window=14).adx()
    tf = TechnicalFeatures(ohlcv, timeframe="1d")
    tf._manual_adx(14)
    manual = tf.df["adx"]
    both = pd.concat([lib, manual], axis=1).dropna().tail(300)
    corr = both.corr().iloc[0, 1]
    assert corr > 0.85, f"manual ADX poorly correlated with ta: {corr:.3f}"


def test_macd_uses_library_and_matches_reference(ohlcv):
    """MACD path goes through ta; verify against the EMA definition."""
    tf = TechnicalFeatures(ohlcv, timeframe="1d")
    tf.add_macd()
    ema12 = ohlcv["close"].ewm(span=12, adjust=False).mean()
    ema26 = ohlcv["close"].ewm(span=26, adjust=False).mean()
    ref = ema12 - ema26
    diff = (tf.df["macd"] - ref).abs().dropna().tail(300)
    assert diff.max() < 0.01 * ohlcv["close"].mean() / 100, \
        f"MACD deviates from definition: {diff.max():.6f}"


def test_bollinger_uses_library(ohlcv):
    from ta.volatility import BollingerBands
    tf = TechnicalFeatures(ohlcv, timeframe="1d")
    tf.add_bollinger_bands()
    bb = BollingerBands(close=ohlcv["close"], window=20, window_dev=2)
    ref_upper = bb.bollinger_hband()
    col = next(c for c in tf.df.columns if "bb" in c and ("upper" in c or "hband" in c or "high" in c))
    diff = (tf.df[col] - ref_upper).abs().dropna().tail(300)
    assert diff.max() < 1e-6, f"Bollinger upper band mismatch: {diff.max()}"


def test_atr_uses_library(ohlcv):
    from ta.volatility import AverageTrueRange
    tf = TechnicalFeatures(ohlcv, timeframe="1d")
    tf.add_atr()
    ref = AverageTrueRange(ohlcv["high"], ohlcv["low"], ohlcv["close"], window=14).average_true_range()
    col = next(c for c in tf.df.columns if c == "atr" or c.startswith("atr_14") or c == "atr_14")
    diff = (tf.df[col] - ref).abs().dropna().tail(300)
    assert diff.max() < 1e-6, f"ATR mismatch: {diff.max()}"


def test_pipeline_produces_50plus_features(ohlcv):
    tf = TechnicalFeatures(ohlcv, timeframe="1d")
    out = tf.compute_all()
    feature_cols = [c for c in out.columns
                    if c not in ("open", "high", "low", "close", "volume", "_date")]
    assert len(feature_cols) >= 50, f"only {len(feature_cols)} features"
    # No column should be entirely NaN
    all_nan = [c for c in feature_cols if out[c].isna().all()]
    assert not all_nan, f"all-NaN features: {all_nan}"
