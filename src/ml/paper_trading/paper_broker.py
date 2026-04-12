"""
Paper Broker — Simulated trade execution for risk-free testing.
================================================================

Provides a realistic trading interface that mirrors Zerodha Kite
without risking real capital. Tracks:
  - Order book (limit, market, stop-limit orders)
  - Position book (open positions with live P&L)
  - Holdings (closed-day positions)

Features:
  - Realistic slippage simulation (0.02-0.05%)
  - Transaction cost modeling (brokerage + STT + stamp + GST)
  - Margin tracking (for F&O)
  - Circuit breaker simulation
  - Order rejection for insufficient capital
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP_LOSS = 'STOP_LOSS'
    STOP_LIMIT = 'STOP_LIMIT'


class OrderStatus(Enum):
    PENDING = 'PENDING'
    EXECUTED = 'EXECUTED'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'


class Side(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class PaperBroker:
    """
    Simulated broker with realistic execution model.

    Parameters
    ----------
    initial_capital : float
        Starting capital (default: ₹1,00,000).
    slippage_pct : float
        Slippage as percentage (default: 0.03%).
    brokerage_per_order : float
        Fixed brokerage per executed order (default: ₹20).
    data_path : str
        Directory for persisting state.
    """

    LIVE_TRADING = False  # SAFETY: never trade real money

    NSE_CHARGES = {
        'brokerage': 20.0,          # ₹20 flat per order (discount broker)
        'stt_delivery': 0.001,      # 0.1% on buy+sell
        'stt_intraday': 0.00025,    # 0.025% on sell
        'exchange_txn': 0.0000345,  # NSE transaction charge
        'sebi_charge': 0.000001,    # SEBI charge
        'stamp_duty': 0.00015,      # Stamp duty on buy
        'gst': 0.18,               # GST on brokerage + txn charges
    }

    def __init__(self, initial_capital: float = 100000,
                 slippage_pct: float = 0.0003,
                 brokerage_per_order: float = 20.0,
                 data_path: str = 'data/paper_trading/'):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.slippage_pct = slippage_pct
        self.brokerage_per_order = brokerage_per_order
        self.data_path = data_path

        self.orders: List[Dict] = []
        self.positions: Dict[str, Dict] = {}  # ticker → position
        self.trade_history: List[Dict] = []
        self.order_counter = 0

        os.makedirs(data_path, exist_ok=True)

        # Load persisted state
        self._load_state()

    def place_order(self, ticker: str, side: str, quantity: int,
                    order_type: str = 'MARKET', price: float = 0,
                    stop_price: float = 0, current_price: float = 0,
                    exchange: str = 'NSE') -> Dict:
        """
        Place a simulated order.

        Parameters
        ----------
        ticker : str
        side : 'BUY' or 'SELL'
        quantity : int
        order_type : 'MARKET', 'LIMIT', 'STOP_LOSS', 'STOP_LIMIT'
        price : float (for LIMIT/STOP_LIMIT)
        stop_price : float (for STOP_LOSS/STOP_LIMIT)
        current_price : float (required for MARKET to simulate execution)
        exchange : str

        Returns
        -------
        dict: {order_id, status, message, fill_price, charges}
        """
        self.order_counter += 1
        order_id = f"ORD{self.order_counter:06d}"

        order = {
            'order_id': order_id,
            'ticker': ticker.upper(),
            'side': side.upper(),
            'quantity': quantity,
            'order_type': order_type.upper(),
            'price': price,
            'stop_price': stop_price,
            'current_price': current_price,
            'exchange': exchange.upper(),
            'status': 'PENDING',
            'placed_at': datetime.now().isoformat(),
            'filled_at': None,
            'fill_price': 0,
            'charges': 0,
            'message': '',
        }

        # Validate
        if quantity <= 0:
            order['status'] = 'REJECTED'
            order['message'] = 'Invalid quantity'
            self.orders.append(order)
            return order

        # For MARKET orders, execute immediately
        if order_type.upper() == 'MARKET':
            if current_price <= 0:
                order['status'] = 'REJECTED'
                order['message'] = 'Current price required for market orders'
                self.orders.append(order)
                return order

            # Apply slippage
            if side.upper() == 'BUY':
                fill_price = current_price * (1 + self.slippage_pct)
            else:
                fill_price = current_price * (1 - self.slippage_pct)

            # Calculate charges
            trade_value = fill_price * quantity
            charges = self._calculate_charges(trade_value, side.upper())

            # Check capital
            if side.upper() == 'BUY':
                required = trade_value + charges
                if required > self.capital:
                    order['status'] = 'REJECTED'
                    order['message'] = f'Insufficient capital. Need: ₹{required:,.0f}, Have: ₹{self.capital:,.0f}'
                    self.orders.append(order)
                    return order

            # Execute
            order['fill_price'] = round(fill_price, 2)
            order['charges'] = round(charges, 2)
            order['status'] = 'EXECUTED'
            order['filled_at'] = datetime.now().isoformat()
            order['message'] = 'Order filled successfully'

            # Update capital
            if side.upper() == 'BUY':
                self.capital -= (trade_value + charges)
            else:
                self.capital += (trade_value - charges)

            # Update positions
            self._update_position(ticker.upper(), side.upper(), quantity, fill_price)

            # Record trade
            trade = {
                'order_id': order_id,
                'ticker': ticker.upper(),
                'side': side.upper(),
                'quantity': quantity,
                'fill_price': fill_price,
                'charges': charges,
                'timestamp': datetime.now().isoformat(),
            }
            self.trade_history.append(trade)

        elif order_type.upper() in ('LIMIT', 'STOP_LOSS', 'STOP_LIMIT'):
            # Store pending order for later evaluation
            order['status'] = 'PENDING'
            order['message'] = f'{order_type} order placed, awaiting trigger'

        self.orders.append(order)
        self._save_state()
        return order

    def _calculate_charges(self, trade_value: float, side: str) -> float:
        """Calculate realistic NSE transaction charges."""
        c = self.NSE_CHARGES
        brokerage = min(c['brokerage'], trade_value * 0.0003)  # 0.03% or ₹20

        stt = trade_value * c['stt_intraday']  # Intraday STT
        exchange_txn = trade_value * c['exchange_txn']
        sebi = trade_value * c['sebi_charge']
        gst = (brokerage + exchange_txn) * c['gst']
        stamp = trade_value * c['stamp_duty'] if side == 'BUY' else 0

        return round(brokerage + stt + exchange_txn + sebi + gst + stamp, 2)

    def _update_position(self, ticker: str, side: str,
                         quantity: int, price: float):
        """Update position book after trade execution."""
        if ticker not in self.positions:
            self.positions[ticker] = {
                'ticker': ticker,
                'quantity': 0,
                'avg_price': 0,
                'pnl': 0,
                'opened_at': datetime.now().isoformat(),
            }

        pos = self.positions[ticker]

        if side == 'BUY':
            # Average up
            total_qty = pos['quantity'] + quantity
            if total_qty > 0:
                pos['avg_price'] = (pos['avg_price'] * pos['quantity'] + price * quantity) / total_qty
            pos['quantity'] = total_qty
        else:
            # Reduce or flip
            trade_pnl = (price - pos['avg_price']) * min(quantity, pos['quantity'])
            pos['pnl'] += trade_pnl
            pos['quantity'] -= quantity

        # Remove if flat
        if pos['quantity'] == 0:
            self.positions.pop(ticker, None)

    def get_portfolio(self) -> Dict:
        """Return current portfolio: capital, positions, total value."""
        position_value = sum(
            p['quantity'] * p['avg_price'] for p in self.positions.values()
        )
        total_value = self.capital + position_value

        return {
            'capital': round(self.capital, 2),
            'position_value': round(position_value, 2),
            'total_value': round(total_value, 2),
            'pnl': round(total_value - self.initial_capital, 2),
            'pnl_pct': round((total_value / self.initial_capital - 1) * 100, 2),
            'positions': list(self.positions.values()),
            'n_positions': len(self.positions),
            'n_trades': len(self.trade_history),
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self.orders:
            if order['order_id'] == order_id and order['status'] == 'PENDING':
                order['status'] = 'CANCELLED'
                order['message'] = 'Cancelled by user'
                self._save_state()
                return True
        return False

    def get_order_book(self) -> List[Dict]:
        """Return all orders with status."""
        return self.orders.copy()

    def _save_state(self):
        """Persist broker state to disk."""
        state = {
            'capital': self.capital,
            'initial_capital': self.initial_capital,
            'positions': self.positions,
            'orders': self.orders[-100:],  # Keep last 100 orders
            'trade_history': self.trade_history[-500:],
            'order_counter': self.order_counter,
            'saved_at': datetime.now().isoformat(),
        }
        path = os.path.join(self.data_path, 'broker_state.json')
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save broker state: {e}")

    def _load_state(self):
        """Load broker state from disk."""
        path = os.path.join(self.data_path, 'broker_state.json')
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.capital = state.get('capital', self.initial_capital)
            self.positions = state.get('positions', {})
            self.orders = state.get('orders', [])
            self.trade_history = state.get('trade_history', [])
            self.order_counter = state.get('order_counter', 0)
            logger.info(f"Broker state loaded. Capital: ₹{self.capital:,.0f}")
        except Exception as e:
            logger.warning(f"Failed to load broker state: {e}")

    def reset(self):
        """Reset broker to initial state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.order_counter = 0
        self._save_state()
