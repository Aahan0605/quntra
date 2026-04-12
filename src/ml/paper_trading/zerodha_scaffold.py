"""
Zerodha Kite API Scaffold — Integration hooks for live trading.
================================================================

⚠ SAFETY: LIVE_TRADING is ALWAYS False by default.
Requires explicit environment variable: QUANTRA_LIVE_TRADING=True

This module provides the interface layer between Quantra's signals
and Zerodha Kite Connect API. It scaffolds all the methods needed
for live order placement but guards every action behind the
LIVE_TRADING flag.

To use in production:
  1. Get API key from developer.kite.trade
  2. Set environment variables: KITE_API_KEY, KITE_API_SECRET
  3. Generate access token via login flow
  4. Set QUANTRA_LIVE_TRADING=True (ONLY after thorough paper testing)
"""

import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ZerodhaKiteScaffold:
    """
    Kite Connect integration scaffold.
    ALL methods check LIVE_TRADING flag before any real-money operation.
    """

    def __init__(self):
        self.LIVE_TRADING = os.environ.get('QUANTRA_LIVE_TRADING', 'False').lower() == 'true'
        self.api_key = os.environ.get('KITE_API_KEY', '')
        self.api_secret = os.environ.get('KITE_API_SECRET', '')
        self.access_token = os.environ.get('KITE_ACCESS_TOKEN', '')
        self._kite = None

        if self.LIVE_TRADING:
            logger.warning("⚠ LIVE TRADING ENABLED — Real money at risk!")
            self._init_kite()
        else:
            logger.info("Zerodha scaffold initialized in PAPER mode (safe).")

    def _init_kite(self):
        """Initialize Kite Connect client."""
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self.api_key)
            if self.access_token:
                self._kite.set_access_token(self.access_token)
        except ImportError:
            logger.warning("kiteconnect not installed. pip install kiteconnect")
        except Exception as e:
            logger.error(f"Kite init failed: {e}")

    def place_order(self, ticker: str, side: str, quantity: int,
                    order_type: str = 'MARKET', price: float = 0,
                    trigger_price: float = 0) -> Dict:
        """
        Place order via Kite Connect.
        Returns order_id or error.
        """
        if not self.LIVE_TRADING:
            return {
                'status': 'BLOCKED',
                'message': 'Live trading disabled. Use PaperBroker instead.',
                'order_id': None,
            }

        if self._kite is None:
            return {'status': 'ERROR', 'message': 'Kite not initialized'}

        try:
            exchange = 'NSE'
            variety = 'regular'
            product = 'MIS'  # Intraday by default

            order_id = self._kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=ticker.upper(),
                transaction_type=side.upper(),
                quantity=quantity,
                order_type=order_type.upper(),
                price=price if order_type != 'MARKET' else None,
                trigger_price=trigger_price if trigger_price > 0 else None,
                product=product,
            )
            return {'status': 'PLACED', 'order_id': order_id}
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def get_positions(self) -> Dict:
        """Fetch current positions from Kite."""
        if not self.LIVE_TRADING or self._kite is None:
            return {'status': 'PAPER_MODE', 'positions': []}

        try:
            return self._kite.positions()
        except Exception as e:
            return {'error': str(e)}

    def get_holdings(self) -> list:
        """Fetch holdings (delivery positions)."""
        if not self.LIVE_TRADING or self._kite is None:
            return []

        try:
            return self._kite.holdings()
        except Exception as e:
            logger.error(f"Holdings fetch failed: {e}")
            return []

    def get_margins(self) -> Dict:
        """Fetch available margins."""
        if not self.LIVE_TRADING or self._kite is None:
            return {'equity': {'available': {'live_balance': 0}}}

        try:
            return self._kite.margins()
        except Exception as e:
            return {'error': str(e)}

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel a pending order."""
        if not self.LIVE_TRADING or self._kite is None:
            return {'status': 'PAPER_MODE'}

        try:
            self._kite.cancel_order(variety='regular', order_id=order_id)
            return {'status': 'CANCELLED', 'order_id': order_id}
        except Exception as e:
            return {'error': str(e)}

    def get_quote(self, ticker: str, exchange: str = 'NSE') -> Dict:
        """Get real-time quote for a ticker."""
        if not self.LIVE_TRADING or self._kite is None:
            return {'status': 'PAPER_MODE', 'price': 0}

        try:
            instruments = [f"{exchange}:{ticker.upper()}"]
            quotes = self._kite.quote(instruments)
            if instruments[0] in quotes:
                q = quotes[instruments[0]]
                return {
                    'last_price': q.get('last_price', 0),
                    'change_pct': q.get('ohlc', {}).get('close', 0),
                    'volume': q.get('volume', 0),
                }
            return {'price': 0}
        except Exception as e:
            return {'error': str(e)}

    def is_market_open(self) -> bool:
        """Check if Indian market is open (9:15 AM - 3:30 PM IST)."""
        from datetime import datetime, time
        now = datetime.now()
        # Weekday check
        if now.weekday() >= 5:
            return False
        market_open = time(9, 15)
        market_close = time(15, 30)
        return market_open <= now.time() <= market_close
