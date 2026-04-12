"""Analysis sub-package — Stock analysis engine."""

try:
    from .stock_analyzer import StockAnalyzer
    from .report_generator import ReportGenerator
except ImportError:
    pass  # Dependencies not yet installed
