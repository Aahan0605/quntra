"""
Quantra Ensemble — Weighted combination of XGBoost + LSTM + TFT.
==================================================================

Weighting strategy: Dynamic weight allocation based on recent performance
(last 20 trades) of each model. The model that was more accurate recently
gets higher weight — a form of meta-learning.

Final signal logic:
  - All 3 agree: HIGH CONFIDENCE signal
  - 2/3 agree: MEDIUM CONFIDENCE signal
  - All disagree: NO TRADE (protect capital)
"""

import os
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .xgboost_model import XGBoostSignalModel
from .lstm_model import LSTMPriceModel
from .transformer_model import TemporalFusionTransformer

logger = logging.getLogger(__name__)


class QuantraEnsemble:
    """
    Weighted ensemble combining XGBoost + LSTM + TFT.

    Parameters
    ----------
    mode : str
        'intraday' or 'swing'.
    model_dir : str
        Root directory for all model weights.
    """

    def __init__(self, mode: str = 'intraday',
                 model_dir: str = 'models/'):
        self.mode = mode
        self.model_dir = model_dir

        self.xgb_model = XGBoostSignalModel(
            mode=mode, model_path=os.path.join(model_dir, 'xgboost/')
        )
        self.lstm_model = LSTMPriceModel(
            mode=mode, model_path=os.path.join(model_dir, 'lstm/')
        )
        self.tft_model = TemporalFusionTransformer(
            model_path=os.path.join(model_dir, 'transformer/')
        )

        self._models_loaded = False
        self._trade_journal_path = 'data/trade_journal/journal.csv'

    def load_all_models(self) -> bool:
        """Load all three models from disk. Return True if at least one loaded."""
        xgb_ok = self.xgb_model.load()
        lstm_ok = self.lstm_model.load()
        tft_ok = self.tft_model.load()
        self._models_loaded = xgb_ok or lstm_ok or tft_ok

        loaded = sum([xgb_ok, lstm_ok, tft_ok])
        logger.info(f"Ensemble: {loaded}/3 models loaded")
        return self._models_loaded

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        Read last 20 trade outcomes from trade journal.
        Compute accuracy per model and normalize to weights.
        """
        default_weights = {'xgboost': 0.40, 'lstm': 0.35, 'tft': 0.25}

        try:
            if not os.path.exists(self._trade_journal_path):
                return default_weights

            journal = pd.read_csv(self._trade_journal_path)
            if len(journal) < 5:
                return default_weights

            # Last 20 trades
            recent = journal.tail(20)

            accuracies = {}

            # XGBoost accuracy
            if 'xgb_signal' in recent.columns and 'was_correct' in recent.columns:
                xgb_mask = recent['xgb_signal'].notna()
                if xgb_mask.sum() > 0:
                    xgb_correct = recent.loc[xgb_mask, 'was_correct'].mean()
                    accuracies['xgboost'] = max(xgb_correct, 0.1)

            # LSTM accuracy
            if 'lstm_signal' in recent.columns and 'was_correct' in recent.columns:
                lstm_mask = recent['lstm_signal'].notna()
                if lstm_mask.sum() > 0:
                    lstm_correct = recent.loc[lstm_mask, 'was_correct'].mean()
                    accuracies['lstm'] = max(lstm_correct, 0.1)

            # TFT accuracy (check if expected move was in correct direction)
            if 'tft_expected_move' in recent.columns and 'was_correct' in recent.columns:
                tft_mask = recent['tft_expected_move'].notna()
                if tft_mask.sum() > 0:
                    tft_correct = recent.loc[tft_mask, 'was_correct'].mean()
                    accuracies['tft'] = max(tft_correct, 0.1)

            if not accuracies:
                return default_weights

            # Normalize to sum to 1
            total = sum(accuracies.values())
            weights = {k: v / total for k, v in accuracies.items()}

            # Fill missing models with minimum weight
            for model in ['xgboost', 'lstm', 'tft']:
                if model not in weights:
                    weights[model] = 0.1

            # Re-normalize
            total = sum(weights.values())
            weights = {k: round(v / total, 3) for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.warning(f"Dynamic weight calculation failed: {e}")
            return default_weights

    def predict(self, X_flat: np.ndarray,
                X_seq: Optional[np.ndarray] = None,
                X_static: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run all three models and combine via dynamic weights.

        Parameters
        ----------
        X_flat : array-like
            Flat feature vector for XGBoost (last row of features).
        X_seq : array-like, optional
            Sequence for LSTM [seq_len, n_features].
        X_static : array-like, optional
            Static features for TFT (fundamental features).

        Returns
        -------
        dict with final_signal, confidence, agreement, model_signals,
        trade_recommendation, risk_flags.
        """
        weights = self.get_dynamic_weights()
        model_signals = {}
        signals = []
        confidences = []

        # --- XGBoost Signal ---
        try:
            xgb_result = self.xgb_model.predict_signal(X_flat)
            model_signals['xgboost'] = xgb_result
            signals.append(xgb_result['signal'])
            confidences.append((xgb_result['confidence'], weights['xgboost']))
        except Exception as e:
            logger.debug(f"XGBoost prediction failed: {e}")
            model_signals['xgboost'] = {'signal': 'HOLD', 'confidence': 0}

        # --- LSTM Signal ---
        try:
            if X_seq is not None:
                lstm_result = self.lstm_model.predict_with_attention(X_seq)
            else:
                # Create sequence from flat features if available
                lstm_result = {'signal': 'HOLD', 'confidence': 0,
                               'attention_weights': [], 'probability': 0.5}
            model_signals['lstm'] = lstm_result
            signals.append(lstm_result['signal'])
            confidences.append((lstm_result.get('confidence', 0), weights['lstm']))
        except Exception as e:
            logger.debug(f"LSTM prediction failed: {e}")
            model_signals['lstm'] = {'signal': 'HOLD', 'confidence': 0}

        # --- TFT Signal ---
        try:
            if X_static is not None and X_seq is not None:
                tft_result = self.tft_model.predict(X_static, X_seq)
            else:
                tft_result = {
                    'expected_move_pct': 0, 'pessimistic_pct': -1,
                    'optimistic_pct': 1, 'variable_importance': {}
                }

            # Convert TFT quantiles to signal
            expected = tft_result['expected_move_pct']
            tft_signal = 'BUY' if expected > 0.3 else 'SELL' if expected < -0.3 else 'HOLD'
            tft_confidence = min(abs(expected) / 2.0, 1.0)
            tft_result['signal'] = tft_signal
            tft_result['confidence'] = tft_confidence

            model_signals['tft'] = tft_result
            signals.append(tft_signal)
            confidences.append((tft_confidence, weights['tft']))
        except Exception as e:
            logger.debug(f"TFT prediction failed: {e}")
            model_signals['tft'] = {'signal': 'HOLD', 'confidence': 0,
                                     'expected_move_pct': 0}

        # --- Combine Signals ---
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        hold_count = signals.count('HOLD')
        total_models = len(signals) if signals else 1

        # Agreement level
        if buy_count == total_models or sell_count == total_models:
            agreement = 'HIGH'
        elif buy_count >= 2 or sell_count >= 2:
            agreement = 'MEDIUM'
        else:
            agreement = 'LOW'

        # Final signal
        if agreement == 'LOW':
            final_signal = 'NO TRADE'
            final_confidence = 0.0
        elif buy_count >= sell_count and buy_count >= hold_count:
            final_signal = 'BUY'
            # Weighted confidence
            final_confidence = sum(c * w for c, w in confidences if c > 0.5) / max(
                sum(w for _, w in confidences), 1e-10
            )
        elif sell_count > buy_count:
            final_signal = 'SELL'
            final_confidence = sum(c * w for c, w in confidences if c > 0.5) / max(
                sum(w for _, w in confidences), 1e-10
            )
        else:
            final_signal = 'HOLD'
            final_confidence = 0.5

        final_confidence = min(max(final_confidence, 0), 1)

        # --- Trade Recommendation ---
        # Get price info from features if available
        entry_price = 0
        atr_val = 0
        if isinstance(X_flat, pd.DataFrame):
            if 'close' in X_flat.columns:
                entry_price = float(X_flat['close'].iloc[-1])
            if 'atr' in X_flat.columns:
                atr_val = float(X_flat['atr'].iloc[-1])
        elif isinstance(X_flat, np.ndarray) and X_flat.ndim == 1:
            # Assume close is first feature in flat array
            entry_price = float(X_flat[0]) if len(X_flat) > 0 else 0

        tft_pessimistic = model_signals.get('tft', {}).get('pessimistic_pct', -1.0)
        tft_optimistic = model_signals.get('tft', {}).get('optimistic_pct', 1.0)

        if entry_price > 0 and final_signal in ('BUY', 'SELL'):
            stop_loss = self.compute_stop_loss(
                entry_price, atr_val, tft_pessimistic,
                direction='long' if final_signal == 'BUY' else 'short'
            )
            sl_pct = abs(entry_price - stop_loss) / entry_price
            target_1 = entry_price * (1 + 1.5 * sl_pct) if final_signal == 'BUY' \
                else entry_price * (1 - 1.5 * sl_pct)
            target_2 = entry_price * (1 + 3.0 * sl_pct) if final_signal == 'BUY' \
                else entry_price * (1 - 3.0 * sl_pct)
            rr_ratio = 1.5  # minimum risk:reward

            pos_size = self.compute_position_size(
                100000, sl_pct, final_confidence
            )

            trade_rec = {
                'action': final_signal,
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target_1': round(target_1, 2),
                'target_2': round(target_2, 2),
                'hold_duration': '15-30 min' if self.mode == 'intraday' else '3-5 days',
                'position_size_pct': round(pos_size, 2),
                'risk_reward_ratio': round(rr_ratio, 2),
            }
        else:
            trade_rec = {
                'action': final_signal,
                'entry_price': round(entry_price, 2),
                'stop_loss': 0, 'target_1': 0, 'target_2': 0,
                'hold_duration': 'N/A',
                'position_size_pct': 0, 'risk_reward_ratio': 0,
            }

        return {
            'final_signal': final_signal,
            'confidence': round(final_confidence, 4),
            'agreement': agreement,
            'model_signals': model_signals,
            'dynamic_weights': weights,
            'trade_recommendation': trade_rec,
            'risk_flags': [],
        }

    def compute_stop_loss(self, price: float, atr: float,
                          tft_pessimistic: float,
                          direction: str = 'long') -> float:
        """
        Use the TIGHTER of two stop loss methods:
          Method 1: ATR-based: price - 2*ATR (for long)
          Method 2: TFT pessimistic quantile level
        """
        if direction == 'long':
            atr_sl = price - 2 * atr if atr > 0 else price * 0.98
            tft_sl = price * (1 + tft_pessimistic / 100) if tft_pessimistic < 0 \
                else price * 0.98
            return max(atr_sl, tft_sl)  # Tighter = higher for longs
        else:
            atr_sl = price + 2 * atr if atr > 0 else price * 1.02
            tft_sl = price * (1 - tft_pessimistic / 100) if tft_pessimistic < 0 \
                else price * 1.02
            return min(atr_sl, tft_sl)  # Tighter = lower for shorts

    @staticmethod
    def compute_position_size(capital: float, stop_loss_pct: float,
                              confidence: float,
                              max_risk_pct: float = 0.02) -> float:
        """
        Modified Kelly Criterion for position sizing.
        risk_per_trade = capital * max_risk_pct * confidence
        Cap position at 10% of capital regardless.
        """
        if stop_loss_pct <= 0:
            return 0.0

        risk_amount = capital * max_risk_pct * confidence
        position_value = risk_amount / stop_loss_pct
        position_pct = (position_value / capital) * 100

        # Cap at 10%
        return min(position_pct, 10.0)
