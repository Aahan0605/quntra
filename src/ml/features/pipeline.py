"""
Unified Feature Pipeline — combines technical + fundamental + sentiment.
========================================================================

Handles the complete data flow from raw ticker to ML-ready feature matrix:
  1. Download OHLCV data for appropriate timeframe
  2. Compute TechnicalFeatures (80+ columns)
  3. Fetch FundamentalFeatures (40+ scalars, broadcast to all rows)
  4. Compute SentimentFeatures (16+ scalars, broadcast)
  5. Create binary target variable (forward returns)
  6. Apply RobustScaler normalization
  7. Correlation-based feature selection (drop >0.95 corr)
  8. Time-ordered train/val/test split (70/15/15, no look-ahead)
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from .technical import TechnicalFeatures
from .fundamental import FundamentalFeatures
from .sentiment import SentimentFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Unified pipeline combining technical + fundamental + sentiment
    into a single feature matrix ready for ML models.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g., 'RELIANCE' for NSE, 'AAPL' for global).
    exchange : str
        'NSE', 'NYSE', or 'NASDAQ'.
    timeframe : str
        '1m', '5m', '15m', '1h', '1d', '1w'.
    mode : str
        'intraday' — uses 1m/5m/15m data, last 60 days.
        'swing' — uses 1d/1w data, last 3 years.
    """

    TIMEFRAME_MAP = {
        '1m': {'interval': '1m', 'period': '7d'},
        '5m': {'interval': '5m', 'period': '60d'},
        '15m': {'interval': '15m', 'period': '60d'},
        '1h': {'interval': '1h', 'period': '730d'},
        '1d': {'interval': '1d', 'period': '3y'},
        '1w': {'interval': '1wk', 'period': '5y'},
    }

    def __init__(self, ticker: str, exchange: str = 'NSE',
                 timeframe: str = '5m', mode: str = 'intraday'):
        self.ticker = ticker.upper().strip()
        self.exchange = exchange.upper().strip()
        self.timeframe = timeframe
        self.mode = mode

        # Format yfinance ticker
        if self.exchange == 'NSE' and not self.ticker.endswith('.NS'):
            self.yf_ticker = f"{self.ticker}.NS"
        else:
            self.yf_ticker = self.ticker

        self.scaler = RobustScaler() if HAS_SKLEARN else None
        self._feature_names = []
        self._raw_df = None
        self._target_col = 'target'

    def download_data(self) -> pd.DataFrame:
        """
        Download OHLCV data from yfinance for the configured timeframe.
        Returns DataFrame with columns: open, high, low, close, volume.
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required for data download.")

        tf_config = self.TIMEFRAME_MAP.get(self.timeframe, self.TIMEFRAME_MAP['5m'])
        logger.info(f"Downloading {self.yf_ticker} | interval={tf_config['interval']} | period={tf_config['period']}")

        df = yf.download(
            self.yf_ticker,
            interval=tf_config['interval'],
            period=tf_config['period'],
            progress=False,
            auto_adjust=True
        )

        if df.empty:
            raise ValueError(f"No data returned for {self.yf_ticker}")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Ensure required columns exist
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Missing columns. Got: {list(df.columns)}")

        # Drop any rows with NaN in OHLCV
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        self._raw_df = df.copy()
        logger.info(f"Downloaded {len(df)} rows for {self.yf_ticker}")
        return df

    def build_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build complete feature matrix with target variable.

        Returns (X, y) where:
          X: feature matrix (all technical + fundamental + sentiment)
          y: binary target —
             intraday: 1 if price > entry_price + 0.5% within 30 bars
             swing: 1 if price > entry_price + 2% within 5 bars

        Time-ordered split is NOT applied here — call split_data() separately.
        """
        # Step 1: Download data
        df = self.download_data()

        # Step 2: Technical features
        logger.info("Computing technical features...")
        tf = TechnicalFeatures(df.copy(), timeframe=self.timeframe)
        df = tf.compute_all()

        # Step 3: Fundamental features (scalar, broadcast)
        logger.info("Fetching fundamental features...")
        try:
            ff = FundamentalFeatures(self.ticker, exchange=self.exchange)
            fund_features = ff.compute_all_features()

            # Separate numeric from string features
            for k, v in fund_features.items():
                if k.startswith('_'):
                    continue  # Skip string metadata
                if isinstance(v, (int, float)):
                    df[k] = v  # Broadcast scalar to all rows
        except Exception as e:
            logger.warning(f"Fundamental features failed: {e}")

        # Step 4: Sentiment features (scalar, broadcast)
        logger.info("Computing sentiment features...")
        try:
            company_name = fund_features.get('_company_name', self.ticker) \
                if 'fund_features' in dir() else self.ticker
            sf = SentimentFeatures(self.ticker, company_name=company_name)
            sent_features = sf.compute_sentiment_features()

            for k, v in sent_features.items():
                if isinstance(v, (int, float)):
                    df[k] = v  # Broadcast
        except Exception as e:
            logger.warning(f"Sentiment features failed: {e}")

        # Step 5: Create target variable
        logger.info("Creating target variable...")
        df = self._create_target(df)

        # Step 6: Separate X and y
        feature_cols = [c for c in df.columns
                        if c != self._target_col
                        and c not in ('open', 'high', 'low', 'close', 'volume', 'date', 'datetime')]

        # Keep only numeric columns
        numeric_cols = []
        for col in feature_cols:
            if df[col].dtype in ('float64', 'float32', 'int64', 'int32', 'bool'):
                numeric_cols.append(col)

        X = df[numeric_cols].copy()
        y = df[self._target_col].copy()

        # Step 7: Remove high-correlation features (>0.95)
        X = self._remove_correlated_features(X, threshold=0.95)

        # Step 8: Scale features
        X = self._scale_features(X)

        self._feature_names = list(X.columns)
        logger.info(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

        return X, y

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable using forward returns.
        IMPORTANT: shift forward to avoid look-ahead bias.

        Intraday: 1 if max price in next 30 bars > close * 1.005 (0.5%)
        Swing: 1 if max price in next 5 bars > close * 1.02 (2%)
        """
        close = df['close'].values
        n = len(close)

        if self.mode == 'intraday':
            n_forward = 30
            threshold = 0.005  # 0.5%
        else:
            n_forward = 5
            threshold = 0.02  # 2%

        target = np.zeros(n, dtype=int)

        for i in range(n - n_forward):
            future_max = np.max(close[i + 1: i + 1 + n_forward])
            if future_max > close[i] * (1 + threshold):
                target[i] = 1

        df[self._target_col] = target

        # Drop last n_forward rows (no target available)
        df = df.iloc[:n - n_forward].copy()

        return df

    def _remove_correlated_features(self, X: pd.DataFrame,
                                     threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove features with correlation > threshold (keep one of each pair).
        This prevents multicollinearity issues in models.
        """
        if X.shape[1] < 2:
            return X

        try:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            return X.drop(columns=to_drop)
        except Exception as e:
            logger.warning(f"Correlation removal failed: {e}")
            return X

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply RobustScaler (handles outliers better than MinMaxScaler).
        """
        if self.scaler is None:
            return X

        try:
            cols = X.columns
            X_scaled = self.scaler.fit_transform(X.values)
            return pd.DataFrame(X_scaled, columns=cols, index=X.index)
        except Exception as e:
            logger.warning(f"Scaling failed: {e}")
            return X

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   train_pct: float = 0.70, val_pct: float = 0.15
                   ) -> dict:
        """
        Time-ordered train/val/test split.
        NO SHUFFLING — respects temporal order to prevent look-ahead leakage.

        Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
        """
        n = len(X)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        return {
            'X_train': X.iloc[:train_end],
            'y_train': y.iloc[:train_end],
            'X_val': X.iloc[train_end:val_end],
            'y_val': y.iloc[train_end:val_end],
            'X_test': X.iloc[val_end:],
            'y_test': y.iloc[val_end:],
        }

    def get_feature_names(self) -> list:
        """Return list of all feature names after selection."""
        return self._feature_names.copy()

    def inverse_transform(self, X_scaled: np.ndarray) -> pd.DataFrame:
        """Convert scaled features back to original scale."""
        if self.scaler is None:
            return pd.DataFrame(X_scaled, columns=self._feature_names)

        X_original = self.scaler.inverse_transform(X_scaled)
        return pd.DataFrame(X_original, columns=self._feature_names)

    def get_latest_features(self) -> Optional[pd.DataFrame]:
        """
        Get feature vector for the LATEST bar only (for live prediction).
        Downloads fresh data, computes features, returns last row.
        """
        try:
            df = self.download_data()
            tf = TechnicalFeatures(df.copy(), timeframe=self.timeframe)
            df = tf.compute_all()

            # Add fundamental + sentiment (broadcast)
            try:
                ff = FundamentalFeatures(self.ticker, exchange=self.exchange)
                fund = ff.compute_all_features()
                for k, v in fund.items():
                    if not k.startswith('_') and isinstance(v, (int, float)):
                        df[k] = v
            except Exception:
                pass

            try:
                sf = SentimentFeatures(self.ticker)
                sent = sf.compute_sentiment_features()
                for k, v in sent.items():
                    if isinstance(v, (int, float)):
                        df[k] = v
            except Exception:
                pass

            # Select only known feature columns
            if self._feature_names:
                available = [c for c in self._feature_names if c in df.columns]
                latest = df[available].iloc[[-1]]
            else:
                latest = df.iloc[[-1]]

            # Scale if scaler is fitted
            if self.scaler is not None:
                try:
                    latest_vals = self.scaler.transform(latest.values)
                    latest = pd.DataFrame(latest_vals, columns=latest.columns)
                except Exception:
                    pass

            return latest

        except Exception as e:
            logger.warning(f"Failed to get latest features: {e}")
            return None
