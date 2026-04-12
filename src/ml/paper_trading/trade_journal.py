"""
Trade Journal — Records every trade for analysis + RL self-learning.
=====================================================================
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TradeJournal:
    """
    Persistent trade journal for all paper (and future live) trades.
    CSV-based for easy analysis. Each row = one completed trade.
    """

    COLUMNS = [
        'trade_id', 'ticker', 'exchange', 'side', 'quantity',
        'entry_price', 'exit_price', 'entry_time', 'exit_time',
        'pnl', 'pnl_pct', 'charges', 'hold_duration_min',
        'xgb_signal', 'lstm_signal', 'tft_expected_move',
        'ensemble_signal', 'ensemble_confidence',
        'rl_action', 'rl_confidence',
        'entry_rsi', 'entry_macd', 'entry_vwap_position',
        'stop_loss', 'target_1', 'target_2',
        'hit_sl', 'hit_t1', 'hit_t2',
        'was_correct', 'notes',
    ]

    def __init__(self, journal_path: str = 'data/trade_journal/journal.csv'):
        self.journal_path = journal_path
        os.makedirs(os.path.dirname(journal_path), exist_ok=True)

        if not os.path.exists(journal_path):
            self._create_empty_journal()

        self._trade_counter = self._get_next_id()

    def _create_empty_journal(self):
        """Create CSV with headers."""
        with open(self.journal_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)

    def _get_next_id(self) -> int:
        """Determine next trade ID."""
        try:
            df = pd.read_csv(self.journal_path)
            if len(df) > 0:
                return len(df) + 1
        except Exception:
            pass
        return 1

    def log_trade(self, trade: Dict) -> str:
        """
        Log a completed trade to the journal.
        Returns trade_id.
        """
        trade_id = f"T{self._trade_counter:06d}"
        self._trade_counter += 1

        row = {col: trade.get(col, '') for col in self.COLUMNS}
        row['trade_id'] = trade_id

        # Compute derived fields
        entry = float(row.get('entry_price', 0) or 0)
        exit_ = float(row.get('exit_price', 0) or 0)
        qty = int(row.get('quantity', 0) or 0)

        if entry > 0 and exit_ > 0 and qty > 0:
            if row.get('side', '').upper() == 'BUY':
                row['pnl'] = round((exit_ - entry) * qty, 2)
                row['pnl_pct'] = round((exit_ / entry - 1) * 100, 2)
            else:
                row['pnl'] = round((entry - exit_) * qty, 2)
                row['pnl_pct'] = round((entry / exit_ - 1) * 100, 2)

        # was_correct: 1 if profitable, 0 if loss
        pnl = float(row.get('pnl', 0) or 0)
        row['was_correct'] = 1 if pnl > 0 else 0

        # Append to CSV
        try:
            with open(self.journal_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction='ignore')
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Failed to log trade: {e}")

        return trade_id

    def get_recent_trades(self, n: int = 20) -> List[Dict]:
        """Return last N trades as list of dicts."""
        try:
            df = pd.read_csv(self.journal_path)
            return df.tail(n).to_dict('records')
        except Exception:
            return []

    def get_performance_stats(self) -> Dict:
        """Compute aggregated performance statistics from journal."""
        try:
            df = pd.read_csv(self.journal_path)
        except Exception:
            return {'error': 'Failed to read journal'}

        if len(df) == 0:
            return {'total_trades': 0}

        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)

        total = len(df)
        winners = df[df['pnl'] > 0]
        losers = df[df['pnl'] < 0]

        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0

        # Win streak
        was_correct = df['was_correct'].astype(int).values
        max_streak = 0
        current_streak = 0
        for w in was_correct:
            if w == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Max drawdown
        cum_pnl = df['pnl'].cumsum()
        peak = cum_pnl.cummax()
        drawdown = cum_pnl - peak
        max_dd = drawdown.min()

        # Profit factor
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / max(gross_loss, 1)

        return {
            'total_trades': total,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': round(len(winners) / max(total, 1) * 100, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_pnl': round(df['pnl'].sum(), 2),
            'profit_factor': round(profit_factor, 2),
            'max_win_streak': max_streak,
            'max_drawdown': round(max_dd, 2),
            'expectancy': round((len(winners)/max(total,1) * avg_win +
                                len(losers)/max(total,1) * avg_loss), 2),
        }

    def get_model_accuracy(self) -> Dict:
        """Compute accuracy for each model's predictions."""
        try:
            df = pd.read_csv(self.journal_path)
        except Exception:
            return {}

        if len(df) < 5:
            return {'insufficient_data': True}

        df['was_correct'] = pd.to_numeric(df['was_correct'], errors='coerce')

        results = {}
        for model, col in [('xgboost', 'xgb_signal'), ('lstm', 'lstm_signal'),
                           ('tft', 'tft_expected_move'), ('ensemble', 'ensemble_signal'),
                           ('rl', 'rl_action')]:
            if col in df.columns:
                valid = df[df[col].notna() & (df[col] != '')]
                if len(valid) > 0:
                    accuracy = valid['was_correct'].mean()
                    results[model] = {
                        'accuracy': round(accuracy * 100, 1),
                        'n_trades': len(valid),
                    }

        return results
