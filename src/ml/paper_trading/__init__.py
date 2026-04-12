"""Paper Trading sub-package — Simulated trading system."""

try:
    from .paper_broker import PaperBroker
    from .trade_journal import TradeJournal
    from .zerodha_scaffold import ZerodhaKiteScaffold
    from .performance_tracker import PerformanceTracker
except ImportError:
    pass  # Dependencies not yet installed
