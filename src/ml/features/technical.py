"""
Technical Feature Engineering — 50+ indicators across multiple timeframes.
==========================================================================

Computes a comprehensive suite of technical indicators for BOTH intraday
(1m, 5m, 15m) and swing (1d, 1w) timeframes. Each category targets a
different aspect of market micro-structure:

  - Trend indicators: identify direction (MA, EMA, MACD, ADX, SuperTrend)
  - Momentum: measure speed of moves (RSI, Stochastic, CCI, ROC, Williams %R)
  - Volatility: measure risk/opportunity (ATR, Bollinger Bands, Keltner, Donchian)
  - Volume: confirm moves (OBV, VWAP, MFI, CMF, Volume Profile)
  - Support/Resistance: key price levels (Pivot Points, Fibonacci)
  - Pattern recognition: candlestick signals (20+ patterns, pure Python)
  - Market structure: HH/HL/LH/LL detection, BOS, CHoCH

All indicators are computed using the `ta` library where possible,
with custom implementations for advanced features (SuperTrend, VWAP bands,
Volume Profile, market structure).

Usage:
    tf = TechnicalFeatures(df, timeframe='5m')
    enriched = tf.compute_all()
    # enriched DataFrame has 80+ new feature columns
"""

import numpy as np
import pandas as pd
import warnings

try:
    import ta
    from ta.trend import (
        SMAIndicator, EMAIndicator, MACD, ADXIndicator,
        IchimokuIndicator
    )
    from ta.momentum import (
        RSIIndicator, StochasticOscillator, WilliamsRIndicator,
        ROCIndicator
    )
    from ta.volatility import (
        BollingerBands, AverageTrueRange, KeltnerChannel,
        DonchianChannel
    )
    from ta.volume import (
        OnBalanceVolumeIndicator, MFIIndicator,
        ChaikinMoneyFlowIndicator
    )
    HAS_TA = True
except ImportError:
    HAS_TA = False
    warnings.warn("'ta' library not installed. Install via: pip install ta")


class TechnicalFeatures:
    """
    Computes 50+ technical indicators across multiple timeframes.
    All indicators computed for BOTH intraday (1m, 5m, 15m)
    and swing (1d, 1w) timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        Column names are case-insensitive (auto-lowered).
    timeframe : str
        One of '1m', '5m', '15m', '1h', '1d', '1w'.
        Affects VWAP reset logic and pivot point calculation.
    """

    def __init__(self, df: pd.DataFrame, timeframe: str = '5m'):
        self.df = df.copy()
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        self.timeframe = timeframe

        # Validate required columns
        required = {'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Add date column if index is datetime
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df['_date'] = self.df.index.date
        elif 'date' in self.df.columns:
            self.df['_date'] = pd.to_datetime(self.df['date']).dt.date
        else:
            self.df['_date'] = 0  # single-day fallback

    # ─────────────────────────────────────────────
    # TREND INDICATORS
    # ─────────────────────────────────────────────

    def add_moving_averages(self):
        """
        Add SMA(9,20,50,200), EMA(9,21,55,200), WMA(20), HMA(20).
        Also add crossover signals:
          ma_golden_cross: 1 when EMA9 crosses above EMA21
          ma_death_cross: 1 when EMA9 crosses below EMA21
        """
        close = self.df['close']

        # Simple Moving Averages
        for p in [9, 20, 50, 200]:
            self.df[f'sma_{p}'] = close.rolling(window=p, min_periods=1).mean()

        # Exponential Moving Averages
        for p in [9, 21, 55, 200]:
            self.df[f'ema_{p}'] = close.ewm(span=p, adjust=False).mean()

        # Weighted Moving Average (20)
        weights = np.arange(1, 21)
        self.df['wma_20'] = close.rolling(window=20, min_periods=1).apply(
            lambda x: np.dot(x[-min(len(x), 20):],
                             weights[-min(len(x), 20):]) / weights[-min(len(x), 20):].sum(),
            raw=True
        )

        # Hull Moving Average (20) — less lag
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_wma = close.ewm(span=10, adjust=False).mean()
        full_wma = close.ewm(span=20, adjust=False).mean()
        hma_raw = 2 * half_wma - full_wma
        self.df['hma_20'] = hma_raw.ewm(span=int(np.sqrt(20)), adjust=False).mean()

        # VWMA (20) — Volume-Weighted Moving Average
        vwap_cum = (close * self.df['volume']).rolling(window=20, min_periods=1).sum()
        vol_cum = self.df['volume'].rolling(window=20, min_periods=1).sum()
        self.df['vwma_20'] = np.where(vol_cum > 0, vwap_cum / vol_cum, close)

        # Crossover signals
        ema9 = self.df['ema_9']
        ema21 = self.df['ema_21']
        self.df['ma_golden_cross'] = ((ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))).astype(int)
        self.df['ma_death_cross'] = ((ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))).astype(int)

    def add_macd(self):
        """
        Standard MACD(12,26,9) + histogram + signal line crossover.
        MACD divergence detection:
          macd_bullish_div: price makes lower low, MACD makes higher low
          macd_bearish_div: price makes higher high, MACD makes lower high
        """
        if HAS_TA:
            macd_ind = MACD(close=self.df['close'], window_slow=26,
                            window_fast=12, window_sign=9)
            self.df['macd'] = macd_ind.macd()
            self.df['macd_signal'] = macd_ind.macd_signal()
            self.df['macd_histogram'] = macd_ind.macd_diff()
        else:
            ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
            ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
            self.df['macd'] = ema12 - ema26
            self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
            self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']

        # MACD signal crossover
        macd = self.df['macd']
        sig = self.df['macd_signal']
        self.df['macd_cross_up'] = ((macd > sig) & (macd.shift(1) <= sig.shift(1))).astype(int)
        self.df['macd_cross_down'] = ((macd < sig) & (macd.shift(1) >= sig.shift(1))).astype(int)

        # Divergence detection (simplified: compare over 14-period windows)
        lookback = 14
        price_ll = self.df['close'].rolling(lookback).min()
        price_hh = self.df['close'].rolling(lookback).max()
        macd_ll = self.df['macd'].rolling(lookback).min()
        macd_hh = self.df['macd'].rolling(lookback).max()

        # Bullish div: price lower low, MACD higher low
        price_makes_ll = self.df['close'] <= price_ll * 1.001
        macd_makes_hl = self.df['macd'] > macd_ll * 1.05
        self.df['macd_bullish_div'] = (price_makes_ll & macd_makes_hl).astype(int)

        # Bearish div: price higher high, MACD lower high
        price_makes_hh = self.df['close'] >= price_hh * 0.999
        macd_makes_lh = self.df['macd'] < macd_hh * 0.95
        self.df['macd_bearish_div'] = (price_makes_hh & macd_makes_lh).astype(int)

    def add_adx(self):
        """ADX(14), +DI, -DI. adx_trending: 1 if ADX > 25."""
        if HAS_TA:
            adx_ind = ADXIndicator(high=self.df['high'], low=self.df['low'],
                                   close=self.df['close'], window=14)
            self.df['adx'] = adx_ind.adx()
            self.df['adx_pos_di'] = adx_ind.adx_pos()
            self.df['adx_neg_di'] = adx_ind.adx_neg()
        else:
            self._manual_adx(14)

        self.df['adx_trending'] = (self.df['adx'] > 25).astype(int)

    def _manual_adx(self, period=14):
        """Fallback ADX — Wilder's smoothing throughout, matching
        ta.trend.ADXIndicator (verified in tests/test_indicator_equivalence.py)."""
        high, low, close = self.df['high'], self.df['low'], self.df['close']
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
                         / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
                          / (atr + 1e-10))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        self.df['adx'] = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        self.df['adx_pos_di'] = plus_di
        self.df['adx_neg_di'] = minus_di

    def add_supertrend(self, period: int = 7, multiplier: float = 3.0):
        """
        SuperTrend indicator — critical for intraday trend following.
          supertrend_direction: 1=bullish, -1=bearish
          supertrend_level: the actual support/resistance level
          supertrend_signal: 1 on direction change to bullish,
                            -1 on direction change to bearish
        """
        high, low, close = self.df['high'].values, self.df['low'].values, self.df['close'].values
        n = len(close)

        # ATR
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        tr[0] = high[0] - low[0]

        atr = np.zeros(n)
        atr[:period] = np.mean(tr[:period]) if period <= n else np.mean(tr)
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Calculate bands
        hl2 = (high + low) / 2.0
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        direction = np.ones(n)  # 1 = bullish
        supertrend = np.zeros(n)

        for i in range(1, n):
            # Adjust bands
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                pass  # keep lower_band[i]
            else:
                lower_band[i] = lower_band[i - 1]

            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                pass
            else:
                upper_band[i] = upper_band[i - 1]

            # Direction
            if direction[i - 1] == 1:  # was bullish
                if close[i] < lower_band[i]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
            else:  # was bearish
                if close[i] > upper_band[i]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]

        self.df['supertrend_direction'] = direction
        self.df['supertrend_level'] = supertrend

        # Signal on direction change
        dir_series = pd.Series(direction, index=self.df.index)
        self.df['supertrend_signal'] = 0
        self.df.loc[
            (dir_series == 1) & (dir_series.shift(1) == -1), 'supertrend_signal'
        ] = 1
        self.df.loc[
            (dir_series == -1) & (dir_series.shift(1) == 1), 'supertrend_signal'
        ] = -1

    # ─────────────────────────────────────────────
    # MOMENTUM INDICATORS
    # ─────────────────────────────────────────────

    def add_rsi(self):
        """
        RSI(14) standard + RSI(7) for intraday sensitivity.
        rsi_oversold: 1 if RSI < 30, rsi_overbought: 1 if RSI > 70.
        rsi_divergence: detect bullish/bearish divergence vs price.
        """
        close = self.df['close']

        if HAS_TA:
            self.df['rsi_14'] = RSIIndicator(close=close, window=14).rsi()
            self.df['rsi_7'] = RSIIndicator(close=close, window=7).rsi()
        else:
            self.df['rsi_14'] = self._manual_rsi(close, 14)
            self.df['rsi_7'] = self._manual_rsi(close, 7)

        self.df['rsi_oversold'] = (self.df['rsi_14'] < 30).astype(int)
        self.df['rsi_overbought'] = (self.df['rsi_14'] > 70).astype(int)

        # RSI divergence (simplified over 14-bar window)
        lookback = 14
        price_ll = close.rolling(lookback).min()
        rsi_ll = self.df['rsi_14'].rolling(lookback).min()
        price_hh = close.rolling(lookback).max()
        rsi_hh = self.df['rsi_14'].rolling(lookback).max()

        self.df['rsi_bullish_div'] = (
            (close <= price_ll * 1.002) &
            (self.df['rsi_14'] > rsi_ll + 2)
        ).astype(int)

        self.df['rsi_bearish_div'] = (
            (close >= price_hh * 0.998) &
            (self.df['rsi_14'] < rsi_hh - 2)
        ).astype(int)

    @staticmethod
    def _manual_rsi(series, period):
        """Fallback RSI — Wilder's smoothing, numerically equivalent to
        ta.momentum.RSIIndicator (verified in tests/test_indicator_equivalence.py)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def add_stochastic(self):
        """Stochastic(14,3,3) — %K, %D, crossover signals."""
        if HAS_TA:
            stoch = StochasticOscillator(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], window=14, smooth_window=3
            )
            self.df['stoch_k'] = stoch.stoch()
            self.df['stoch_d'] = stoch.stoch_signal()
        else:
            low_14 = self.df['low'].rolling(14, min_periods=1).min()
            high_14 = self.df['high'].rolling(14, min_periods=1).max()
            self.df['stoch_k'] = 100 * (self.df['close'] - low_14) / (high_14 - low_14 + 1e-10)
            self.df['stoch_d'] = self.df['stoch_k'].rolling(3, min_periods=1).mean()

        self.df['stoch_cross_up'] = (
            (self.df['stoch_k'] > self.df['stoch_d']) &
            (self.df['stoch_k'].shift(1) <= self.df['stoch_d'].shift(1))
        ).astype(int)

    def add_cci(self):
        """CCI(20) — commodity channel index, overbought/oversold."""
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
        sma_tp = tp.rolling(20, min_periods=1).mean()
        mad = tp.rolling(20, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        self.df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        self.df['cci_overbought'] = (self.df['cci'] > 100).astype(int)
        self.df['cci_oversold'] = (self.df['cci'] < -100).astype(int)

    def add_williams_r(self):
        """Williams %R(14) — momentum oscillator."""
        if HAS_TA:
            self.df['williams_r'] = WilliamsRIndicator(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], lbp=14
            ).williams_r()
        else:
            high_14 = self.df['high'].rolling(14, min_periods=1).max()
            low_14 = self.df['low'].rolling(14, min_periods=1).min()
            self.df['williams_r'] = -100 * (high_14 - self.df['close']) / (high_14 - low_14 + 1e-10)

    def add_roc(self):
        """Rate of Change: ROC(9), ROC(21)."""
        close = self.df['close']
        for p in [9, 21]:
            shifted = close.shift(p)
            self.df[f'roc_{p}'] = ((close - shifted) / (shifted + 1e-10)) * 100

    def add_momentum(self):
        """Raw momentum: MOM(10), MOM(20)."""
        close = self.df['close']
        for p in [10, 20]:
            self.df[f'mom_{p}'] = close - close.shift(p)

    # ─────────────────────────────────────────────
    # VOLATILITY INDICATORS
    # ─────────────────────────────────────────────

    def add_bollinger_bands(self):
        """
        BB(20,2): upper, middle, lower bands.
        bb_width: (upper-lower)/middle — volatility expansion measure.
        bb_position: (close-lower)/(upper-lower) — where price sits.
        bb_squeeze: 1 if bb_width < 20-period min of bb_width.
        bb_breakout_up/down: 1 if close crosses above/below bands.
        """
        if HAS_TA:
            bb = BollingerBands(close=self.df['close'], window=20, window_dev=2)
            self.df['bb_upper'] = bb.bollinger_hband()
            self.df['bb_middle'] = bb.bollinger_mavg()
            self.df['bb_lower'] = bb.bollinger_lband()
        else:
            sma20 = self.df['close'].rolling(20, min_periods=1).mean()
            std20 = self.df['close'].rolling(20, min_periods=1).std()
            self.df['bb_middle'] = sma20
            self.df['bb_upper'] = sma20 + 2 * std20
            self.df['bb_lower'] = sma20 - 2 * std20

        bbu = self.df['bb_upper']
        bbl = self.df['bb_lower']
        bbm = self.df['bb_middle']

        self.df['bb_width'] = (bbu - bbl) / (bbm + 1e-10)
        self.df['bb_position'] = (self.df['close'] - bbl) / (bbu - bbl + 1e-10)

        # Squeeze detection
        bb_width_min = self.df['bb_width'].rolling(20, min_periods=1).min()
        self.df['bb_squeeze'] = (self.df['bb_width'] <= bb_width_min * 1.05).astype(int)

        # Breakout signals
        close = self.df['close']
        self.df['bb_breakout_up'] = (
            (close > bbu) & (close.shift(1) <= bbu.shift(1))
        ).astype(int)
        self.df['bb_breakout_down'] = (
            (close < bbl) & (close.shift(1) >= bbl.shift(1))
        ).astype(int)

    def add_atr(self):
        """
        ATR(14) — Average True Range. CRITICAL for stop loss calculation.
        atr_stop_loss_long: close - 2*ATR, atr_stop_loss_short: close + 2*ATR.
        atr_target_1r: close + 1.5*ATR, atr_target_2r: close + 3*ATR.
        atr_pct: ATR/close * 100 — normalized volatility.
        """
        if HAS_TA:
            atr_ind = AverageTrueRange(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], window=14
            )
            self.df['atr'] = atr_ind.average_true_range()
        else:
            tr = pd.concat([
                self.df['high'] - self.df['low'],
                (self.df['high'] - self.df['close'].shift(1)).abs(),
                (self.df['low'] - self.df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            self.df['atr'] = tr.rolling(14, min_periods=1).mean()

        atr = self.df['atr']
        close = self.df['close']

        self.df['atr_stop_loss_long'] = close - 2.0 * atr
        self.df['atr_stop_loss_short'] = close + 2.0 * atr
        self.df['atr_target_1r'] = close + 1.5 * atr
        self.df['atr_target_2r'] = close + 3.0 * atr
        self.df['atr_pct'] = (atr / (close + 1e-10)) * 100

    def add_keltner_channels(self):
        """Keltner(20, 2x ATR). Squeeze when BB inside Keltner."""
        if HAS_TA:
            kc = KeltnerChannel(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], window=20, window_atr=10
            )
            self.df['keltner_upper'] = kc.keltner_channel_hband()
            self.df['keltner_lower'] = kc.keltner_channel_lband()
        else:
            ema20 = self.df['close'].ewm(span=20, adjust=False).mean()
            atr10 = self.df.get('atr', self.df['close'].rolling(10).std())
            self.df['keltner_upper'] = ema20 + 2 * atr10
            self.df['keltner_lower'] = ema20 - 2 * atr10

        # TTM Squeeze: BB is INSIDE Keltner = consolidation
        if 'bb_upper' in self.df.columns:
            self.df['ttm_squeeze'] = (
                (self.df['bb_upper'] < self.df['keltner_upper']) &
                (self.df['bb_lower'] > self.df['keltner_lower'])
            ).astype(int)

    def add_donchian_channels(self):
        """Donchian(20): 20-period high/low channel. Breakout signals."""
        if HAS_TA:
            dc = DonchianChannel(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], window=20
            )
            self.df['donchian_upper'] = dc.donchian_channel_hband()
            self.df['donchian_lower'] = dc.donchian_channel_lband()
        else:
            self.df['donchian_upper'] = self.df['high'].rolling(20, min_periods=1).max()
            self.df['donchian_lower'] = self.df['low'].rolling(20, min_periods=1).min()

        close = self.df['close']
        du = self.df['donchian_upper']
        dl = self.df['donchian_lower']
        self.df['donchian_breakout_up'] = (
            (close >= du) & (close.shift(1) < du.shift(1))
        ).astype(int)
        self.df['donchian_breakout_down'] = (
            (close <= dl) & (close.shift(1) > dl.shift(1))
        ).astype(int)

    # ─────────────────────────────────────────────
    # VOLUME INDICATORS
    # ─────────────────────────────────────────────

    def add_vwap(self):
        """
        VWAP — most important intraday indicator.
        Resets at market open each day (9:15 AM IST for NSE, 9:30 AM EST).
        vwap_position: 1 if price above VWAP (bullish bias).
        vwap_distance_pct: (close - vwap) / vwap * 100.
        vwap_std_bands: VWAP ± 1σ, ± 2σ (institutional levels).
        """
        df = self.df
        is_intraday = self.timeframe in ('1m', '5m', '15m', '1h')

        if is_intraday and '_date' in df.columns:
            # Reset VWAP daily
            vwap_vals = np.zeros(len(df))
            vwap_upper1 = np.zeros(len(df))
            vwap_lower1 = np.zeros(len(df))
            vwap_upper2 = np.zeros(len(df))
            vwap_lower2 = np.zeros(len(df))

            for date, group in df.groupby('_date'):
                idx = group.index
                tp = (group['high'] + group['low'] + group['close']) / 3.0
                cum_tp_vol = (tp * group['volume']).cumsum()
                cum_vol = group['volume'].cumsum()
                vwap = cum_tp_vol / (cum_vol + 1e-10)

                # VWAP standard deviation bands
                tp_sq_vol = (tp ** 2 * group['volume']).cumsum()
                variance = (tp_sq_vol / (cum_vol + 1e-10)) - vwap ** 2
                std = np.sqrt(np.maximum(variance, 0))

                vwap_vals[df.index.get_indexer(idx)] = vwap.values
                vwap_upper1[df.index.get_indexer(idx)] = (vwap + std).values
                vwap_lower1[df.index.get_indexer(idx)] = (vwap - std).values
                vwap_upper2[df.index.get_indexer(idx)] = (vwap + 2 * std).values
                vwap_lower2[df.index.get_indexer(idx)] = (vwap - 2 * std).values

            df['vwap'] = vwap_vals
            df['vwap_upper_1std'] = vwap_upper1
            df['vwap_lower_1std'] = vwap_lower1
            df['vwap_upper_2std'] = vwap_upper2
            df['vwap_lower_2std'] = vwap_lower2
        else:
            # Non-intraday: rolling VWAP (no reset)
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            cum_tp_vol = (tp * df['volume']).cumsum()
            cum_vol = df['volume'].cumsum()
            df['vwap'] = cum_tp_vol / (cum_vol + 1e-10)
            df['vwap_upper_1std'] = df['vwap'] * 1.01
            df['vwap_lower_1std'] = df['vwap'] * 0.99
            df['vwap_upper_2std'] = df['vwap'] * 1.02
            df['vwap_lower_2std'] = df['vwap'] * 0.98

        df['vwap_position'] = (df['close'] > df['vwap']).astype(int)
        df['vwap_distance_pct'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10) * 100

    def add_obv(self):
        """On-Balance Volume + OBV EMA(20) + OBV divergence signal."""
        if HAS_TA:
            self.df['obv'] = OnBalanceVolumeIndicator(
                close=self.df['close'], volume=self.df['volume']
            ).on_balance_volume()
        else:
            sign = np.sign(self.df['close'].diff())
            sign.iloc[0] = 0
            self.df['obv'] = (sign * self.df['volume']).cumsum()

        self.df['obv_ema_20'] = self.df['obv'].ewm(span=20, adjust=False).mean()

        # OBV divergence: price up but OBV down (bearish), price down but OBV up (bullish)
        price_up = self.df['close'] > self.df['close'].shift(5)
        obv_down = self.df['obv'] < self.df['obv'].shift(5)
        self.df['obv_bearish_div'] = (price_up & obv_down).astype(int)

        price_down = self.df['close'] < self.df['close'].shift(5)
        obv_up = self.df['obv'] > self.df['obv'].shift(5)
        self.df['obv_bullish_div'] = (price_down & obv_up).astype(int)

    def add_mfi(self):
        """Money Flow Index(14) — volume-weighted RSI."""
        if HAS_TA:
            self.df['mfi'] = MFIIndicator(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], volume=self.df['volume'],
                window=14
            ).money_flow_index()
        else:
            tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
            mf = tp * self.df['volume']
            pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14, min_periods=1).sum()
            neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14, min_periods=1).sum()
            mfr = pos_mf / (neg_mf + 1e-10)
            self.df['mfi'] = 100 - (100 / (1 + mfr))

    def add_cmf(self):
        """Chaikin Money Flow(20) — institutional buying/selling."""
        if HAS_TA:
            self.df['cmf'] = ChaikinMoneyFlowIndicator(
                high=self.df['high'], low=self.df['low'],
                close=self.df['close'], volume=self.df['volume'],
                window=20
            ).chaikin_money_flow()
        else:
            hl_range = self.df['high'] - self.df['low']
            mf_mult = ((self.df['close'] - self.df['low']) -
                        (self.df['high'] - self.df['close'])) / (hl_range + 1e-10)
            mfv = mf_mult * self.df['volume']
            self.df['cmf'] = mfv.rolling(20, min_periods=1).sum() / \
                             (self.df['volume'].rolling(20, min_periods=1).sum() + 1e-10)

    def add_volume_profile(self):
        """
        Point of Control (POC), Value Area High/Low (VAH/VAL).
        volume_at_poc: volume at POC level.
        distance_from_poc_pct: how far price is from POC.
        """
        lookback = min(50, len(self.df))
        recent = self.df.tail(lookback)

        if len(recent) < 5:
            self.df['poc'] = self.df['close']
            self.df['vah'] = self.df['high']
            self.df['val'] = self.df['low']
            self.df['distance_from_poc_pct'] = 0
            return

        # Create price bins
        price_min, price_max = recent['low'].min(), recent['high'].max()
        n_bins = 20
        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Distribute volume across bins
        vol_profile = np.zeros(n_bins)
        for _, row in recent.iterrows():
            for j in range(n_bins):
                if row['low'] <= bin_centers[j] <= row['high']:
                    vol_profile[j] += row['volume'] / max(
                        1, sum(1 for b in bin_centers if row['low'] <= b <= row['high'])
                    )

        # POC: price level with highest volume
        poc_idx = np.argmax(vol_profile)
        poc_price = bin_centers[poc_idx]

        # Value Area: 70% of total volume centered on POC
        total_vol = vol_profile.sum()
        target_vol = 0.7 * total_vol
        cum_vol = vol_profile[poc_idx]
        lo_idx, hi_idx = poc_idx, poc_idx
        while cum_vol < target_vol and (lo_idx > 0 or hi_idx < n_bins - 1):
            lo_add = vol_profile[lo_idx - 1] if lo_idx > 0 else 0
            hi_add = vol_profile[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            if lo_add >= hi_add and lo_idx > 0:
                lo_idx -= 1
                cum_vol += lo_add
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                cum_vol += hi_add
            else:
                break

        self.df['poc'] = poc_price
        self.df['vah'] = bin_centers[hi_idx]
        self.df['val'] = bin_centers[lo_idx]
        self.df['distance_from_poc_pct'] = (
            (self.df['close'] - poc_price) / (poc_price + 1e-10) * 100
        )

    # ─────────────────────────────────────────────
    # SUPPORT & RESISTANCE
    # ─────────────────────────────────────────────

    def add_pivot_points(self):
        """
        Standard pivot points + Camarilla pivots.
        PP, R1-R3, S1-S3 for standard.
        nearest_resistance, nearest_support, distance percentages.
        """
        # Use previous period's HLC
        h = self.df['high'].shift(1)
        l = self.df['low'].shift(1)
        c = self.df['close'].shift(1)

        pp = (h + l + c) / 3.0
        self.df['pivot_pp'] = pp
        self.df['pivot_r1'] = 2 * pp - l
        self.df['pivot_r2'] = pp + (h - l)
        self.df['pivot_r3'] = h + 2 * (pp - l)
        self.df['pivot_s1'] = 2 * pp - h
        self.df['pivot_s2'] = pp - (h - l)
        self.df['pivot_s3'] = l - 2 * (h - pp)

        # Camarilla pivots (tighter, better for intraday NSE)
        hl_range = h - l
        self.df['cam_r1'] = c + hl_range * 1.1 / 12
        self.df['cam_r2'] = c + hl_range * 1.1 / 6
        self.df['cam_r3'] = c + hl_range * 1.1 / 4
        self.df['cam_s1'] = c - hl_range * 1.1 / 12
        self.df['cam_s2'] = c - hl_range * 1.1 / 6
        self.df['cam_s3'] = c - hl_range * 1.1 / 4

        # Nearest S/R
        close = self.df['close']
        resistances = self.df[['pivot_r1', 'pivot_r2', 'pivot_r3']].values
        supports = self.df[['pivot_s1', 'pivot_s2', 'pivot_s3']].values

        nearest_r = np.full(len(self.df), np.nan)
        nearest_s = np.full(len(self.df), np.nan)

        for i in range(len(self.df)):
            price = close.iloc[i]
            r_above = resistances[i][resistances[i] > price]
            s_below = supports[i][supports[i] < price]
            nearest_r[i] = np.nanmin(r_above) if len(r_above) > 0 else resistances[i].max()
            nearest_s[i] = np.nanmax(s_below) if len(s_below) > 0 else supports[i].min()

        self.df['nearest_resistance'] = nearest_r
        self.df['nearest_support'] = nearest_s
        self.df['distance_to_resistance_pct'] = (
            (nearest_r - close.values) / (close.values + 1e-10) * 100
        )
        self.df['distance_to_support_pct'] = (
            (close.values - nearest_s) / (close.values + 1e-10) * 100
        )

    def add_fibonacci_levels(self, lookback: int = 50):
        """
        Fibonacci retracement from recent swing high/low.
        fib_0 through fib_100 + nearest_fib_level, at_fib_level.
        """
        window = min(lookback, len(self.df))
        recent_high = self.df['high'].rolling(window, min_periods=1).max()
        recent_low = self.df['low'].rolling(window, min_periods=1).min()
        diff = recent_high - recent_low

        fib_levels = {
            'fib_0': 0.0, 'fib_236': 0.236, 'fib_382': 0.382,
            'fib_500': 0.500, 'fib_618': 0.618, 'fib_786': 0.786,
            'fib_100': 1.0
        }
        for name, ratio in fib_levels.items():
            self.df[name] = recent_high - diff * ratio

        # Nearest fib level & at-fib detection
        fib_cols = list(fib_levels.keys())
        close = self.df['close'].values
        nearest = np.zeros(len(self.df))
        at_fib = np.zeros(len(self.df), dtype=int)

        for i in range(len(self.df)):
            fib_vals = [self.df[c].iloc[i] for c in fib_cols]
            distances = [abs(close[i] - fv) for fv in fib_vals]
            min_idx = np.argmin(distances)
            nearest[i] = fib_vals[min_idx]
            # Within 0.2% of any fib level
            if distances[min_idx] / (close[i] + 1e-10) < 0.002:
                at_fib[i] = 1

        self.df['nearest_fib_level'] = nearest
        self.df['at_fib_level'] = at_fib

    # ─────────────────────────────────────────────
    # CANDLESTICK PATTERNS (pure Python, no TA-Lib)
    # ─────────────────────────────────────────────

    def add_candlestick_patterns(self):
        """
        Detect 20+ candlestick patterns using price relationships.
        Each pattern as binary column (1=present, 0=not).
        pattern_strength: sum(bullish) - sum(bearish).
        """
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        body = (c - o).abs()
        upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
        body_pct = body / (h - l + 1e-10)
        is_bullish = c > o
        is_bearish = c < o

        prev_o = o.shift(1)
        prev_c = c.shift(1)
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        prev_body = (prev_c - prev_o).abs()
        prev_bullish = prev_c > prev_o
        prev_bearish = prev_c < prev_o

        # --- BULLISH PATTERNS ---
        # Hammer: small body at top, long lower shadow (>= 2x body)
        self.df['pat_hammer'] = (
            is_bullish & (lower_shadow >= 2 * body) & (upper_shadow < body * 0.5) &
            (body_pct > 0.1)
        ).astype(int)

        # Inverted hammer
        self.df['pat_inverted_hammer'] = (
            is_bullish & (upper_shadow >= 2 * body) & (lower_shadow < body * 0.5) &
            prev_bearish
        ).astype(int)

        # Bullish engulfing
        self.df['pat_bullish_engulfing'] = (
            is_bullish & prev_bearish &
            (o <= prev_c) & (c >= prev_o) &
            (body > prev_body)
        ).astype(int)

        # Morning star (3-candle: bearish, small body, bullish)
        pp_bearish = o.shift(2) > c.shift(2)
        small_body = (c.shift(1) - o.shift(1)).abs() < body.shift(2) * 0.3
        self.df['pat_morning_star'] = (
            pp_bearish & small_body & is_bullish &
            (c > (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        # Three white soldiers
        prev2_bullish = c.shift(2) > o.shift(2)
        self.df['pat_three_white_soldiers'] = (
            is_bullish & prev_bullish & prev2_bullish &
            (c > prev_c) & (prev_c > c.shift(2)) &
            (body > body.mean() * 0.5) & (prev_body > body.mean() * 0.5)
        ).astype(int)

        # Bullish harami
        self.df['pat_bullish_harami'] = (
            is_bullish & prev_bearish &
            (o >= prev_c) & (c <= prev_o) &
            (body < prev_body * 0.5)
        ).astype(int)

        # Dragonfly doji (open/close near high, long lower shadow)
        self.df['pat_dragonfly_doji'] = (
            (body_pct < 0.05) & (lower_shadow >= 3 * body) &
            (upper_shadow < body * 0.5)
        ).astype(int)

        # Piercing line
        self.df['pat_piercing_line'] = (
            is_bullish & prev_bearish &
            (o < prev_l) & (c > (prev_o + prev_c) / 2) & (c < prev_o)
        ).astype(int)

        # --- BEARISH PATTERNS ---
        # Shooting star
        self.df['pat_shooting_star'] = (
            is_bearish & (upper_shadow >= 2 * body) & (lower_shadow < body * 0.5) &
            prev_bullish
        ).astype(int)

        # Hanging man
        self.df['pat_hanging_man'] = (
            is_bearish & (lower_shadow >= 2 * body) & (upper_shadow < body * 0.5) &
            prev_bullish
        ).astype(int)

        # Bearish engulfing
        self.df['pat_bearish_engulfing'] = (
            is_bearish & prev_bullish &
            (o >= prev_c) & (c <= prev_o) &
            (body > prev_body)
        ).astype(int)

        # Evening star (3-candle)
        pp_bullish = c.shift(2) > o.shift(2)
        self.df['pat_evening_star'] = (
            pp_bullish & small_body & is_bearish &
            (c < (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        # Three black crows
        prev2_bearish = c.shift(2) < o.shift(2)
        self.df['pat_three_black_crows'] = (
            is_bearish & prev_bearish & prev2_bearish &
            (c < prev_c) & (prev_c < c.shift(2)) &
            (body > body.mean() * 0.5) & (prev_body > body.mean() * 0.5)
        ).astype(int)

        # Bearish harami
        self.df['pat_bearish_harami'] = (
            is_bearish & prev_bullish &
            (o <= prev_c) & (c >= prev_o) &
            (body < prev_body * 0.5)
        ).astype(int)

        # Gravestone doji
        self.df['pat_gravestone_doji'] = (
            (body_pct < 0.05) & (upper_shadow >= 3 * body) &
            (lower_shadow < body * 0.5)
        ).astype(int)

        # Dark cloud cover
        self.df['pat_dark_cloud'] = (
            is_bearish & prev_bullish &
            (o > prev_h) & (c < (prev_o + prev_c) / 2) & (c > prev_o)
        ).astype(int)

        # --- NEUTRAL PATTERNS ---
        # Doji
        self.df['pat_doji'] = (body_pct < 0.03).astype(int)

        # Spinning top
        self.df['pat_spinning_top'] = (
            (body_pct < 0.2) & (body_pct >= 0.03) &
            (upper_shadow > body * 0.5) & (lower_shadow > body * 0.5)
        ).astype(int)

        # Inside bar
        self.df['pat_inside_bar'] = (
            (h < prev_h) & (l > prev_l)
        ).astype(int)

        # Aggregate pattern strength
        bullish_pats = [
            'pat_hammer', 'pat_inverted_hammer', 'pat_bullish_engulfing',
            'pat_morning_star', 'pat_three_white_soldiers', 'pat_bullish_harami',
            'pat_dragonfly_doji', 'pat_piercing_line'
        ]
        bearish_pats = [
            'pat_shooting_star', 'pat_hanging_man', 'pat_bearish_engulfing',
            'pat_evening_star', 'pat_three_black_crows', 'pat_bearish_harami',
            'pat_gravestone_doji', 'pat_dark_cloud'
        ]
        self.df['pattern_strength'] = (
            self.df[bullish_pats].sum(axis=1) - self.df[bearish_pats].sum(axis=1)
        )

    # ─────────────────────────────────────────────
    # MARKET STRUCTURE
    # ─────────────────────────────────────────────

    def add_market_structure(self):
        """
        Detect Higher Highs/Lows, Lower Highs/Lows for trend structure.
        Break of Structure (BOS) and Change of Character (CHoCH).
        """
        close = self.df['close'].values
        high = self.df['high'].values
        low = self.df['low'].values
        n = len(close)

        # Find swing points (local extrema over 5-bar window)
        swing_highs = np.zeros(n)
        swing_lows = np.zeros(n)
        lookback = 5

        for i in range(lookback, n - lookback):
            if high[i] == max(high[i - lookback:i + lookback + 1]):
                swing_highs[i] = high[i]
            if low[i] == min(low[i - lookback:i + lookback + 1]):
                swing_lows[i] = low[i]

        # Track swing high/low sequences for structure
        structure = np.zeros(n)  # 1=uptrend, -1=downtrend, 0=ranging
        last_sh = np.nan
        prev_sh = np.nan
        last_sl = np.nan
        prev_sl = np.nan

        bos_bullish = np.zeros(n, dtype=int)
        bos_bearish = np.zeros(n, dtype=int)
        choch_bullish = np.zeros(n, dtype=int)
        choch_bearish = np.zeros(n, dtype=int)

        for i in range(lookback, n):
            if swing_highs[i] > 0:
                prev_sh = last_sh
                last_sh = swing_highs[i]
            if swing_lows[i] > 0:
                prev_sl = last_sl
                last_sl = swing_lows[i]

            # Determine structure
            if not np.isnan(prev_sh) and not np.isnan(prev_sl):
                hh = last_sh > prev_sh if not np.isnan(last_sh) else False
                hl = last_sl > prev_sl if not np.isnan(last_sl) else False
                lh = last_sh < prev_sh if not np.isnan(last_sh) else False
                ll = last_sl < prev_sl if not np.isnan(last_sl) else False

                if hh and hl:
                    structure[i] = 1  # uptrend
                elif lh and ll:
                    structure[i] = -1  # downtrend
                else:
                    structure[i] = structure[i - 1] if i > 0 else 0

                # BOS: price breaks above recent swing high (bullish)
                if not np.isnan(last_sh) and close[i] > last_sh and structure[i - 1] != 1:
                    bos_bullish[i] = 1
                # BOS: price breaks below recent swing low (bearish)
                if not np.isnan(last_sl) and close[i] < last_sl and structure[i - 1] != -1:
                    bos_bearish[i] = 1

                # CHoCH: trend reversal
                if structure[i] == 1 and structure[i - 1] == -1:
                    choch_bullish[i] = 1
                elif structure[i] == -1 and structure[i - 1] == 1:
                    choch_bearish[i] = 1
            else:
                structure[i] = 0

        self.df['structure_trend'] = structure
        self.df['bos_bullish'] = bos_bullish
        self.df['bos_bearish'] = bos_bearish
        self.df['choch_bullish'] = choch_bullish
        self.df['choch_bearish'] = choch_bearish

    # ─────────────────────────────────────────────
    # MASTER COMPUTE
    # ─────────────────────────────────────────────

    def compute_all(self) -> pd.DataFrame:
        """
        Call every indicator method in correct order.
        Return enriched DataFrame with all 80+ feature columns.
        Drop NaN rows from indicator lookback periods.
        """
        # Trend
        self.add_moving_averages()
        self.add_macd()
        self.add_adx()
        self.add_supertrend()

        # Momentum
        self.add_rsi()
        self.add_stochastic()
        self.add_cci()
        self.add_williams_r()
        self.add_roc()
        self.add_momentum()

        # Volatility
        self.add_bollinger_bands()
        self.add_atr()
        self.add_keltner_channels()
        self.add_donchian_channels()

        # Volume
        self.add_vwap()
        self.add_obv()
        self.add_mfi()
        self.add_cmf()
        self.add_volume_profile()

        # Support & Resistance
        self.add_pivot_points()
        self.add_fibonacci_levels()

        # Patterns
        self.add_candlestick_patterns()

        # Market Structure
        self.add_market_structure()

        # Cleanup: drop internal columns and NaN-heavy initial rows
        if '_date' in self.df.columns:
            self.df.drop(columns=['_date'], inplace=True)

        # Drop rows where critical indicators haven't warmed up
        warmup = min(55, len(self.df) // 4)  # EMA55 needs 55 periods
        if warmup > 0 and len(self.df) > warmup * 2:
            self.df = self.df.iloc[warmup:].reset_index(drop=True)

        # Fill any remaining NaN with 0 for ML compatibility
        self.df = self.df.fillna(0)

        return self.df
