"""
Fundamental Feature Engineering — Company financials for any NSE/NYSE/NASDAQ stock.
====================================================================================

Fetches and computes fundamental data using yfinance as the primary source.
Covers income statement, balance sheet, valuation ratios, event calendar,
and institutional activity.

Key insight for trading:
  - Intraday: fundamentals mostly irrelevant EXCEPT event risks
    (earnings dates, F&O ban, circuit breakers, promoter pledging)
  - Swing: P/E, P/B, ROE, debt/equity, revenue growth are critical
    for identifying value vs momentum opportunities

Data sources:
  - yfinance: primary (free, covers both NSE and global)
  - Fallback to cached/default values on data errors (never crash)
"""

import warnings
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinance not installed. Fundamental data unavailable.")

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """
    Fetches and computes fundamental data for any NSE/NYSE/NASDAQ stock.

    Parameters
    ----------
    ticker : str
        'RELIANCE' for NSE, 'AAPL' for NYSE/NASDAQ.
    exchange : str
        'NSE', 'NYSE', or 'NASDAQ'. Determines ticker formatting.
    """

    # NSE tickers that need .NS suffix for yfinance
    NSE_SUFFIX = '.NS'

    def __init__(self, ticker: str, exchange: str = 'NSE'):
        self.raw_ticker = ticker.upper().strip()
        self.exchange = exchange.upper().strip()

        # Format ticker for yfinance
        if self.exchange == 'NSE':
            self.yf_ticker = (
                self.raw_ticker if self.raw_ticker.endswith('.NS')
                else f"{self.raw_ticker}.NS"
            )
        elif self.exchange == 'BSE':
            self.yf_ticker = (
                self.raw_ticker if self.raw_ticker.endswith('.BO')
                else f"{self.raw_ticker}.BO"
            )
        else:
            self.yf_ticker = self.raw_ticker

        self._stock = None
        self._info = None
        self._info_fetched = False

    @property
    def stock(self):
        """Lazy-load yfinance Ticker object."""
        if self._stock is None and HAS_YFINANCE:
            self._stock = yf.Ticker(self.yf_ticker)
        return self._stock

    @property
    def info(self) -> dict:
        """Lazy-load and cache stock info (most expensive call)."""
        if not self._info_fetched:
            self._info_fetched = True
            try:
                self._info = self.stock.info if self.stock else {}
            except Exception as e:
                logger.warning(f"Failed to fetch info for {self.yf_ticker}: {e}")
                self._info = {}
        return self._info or {}

    def get_company_info(self) -> dict:
        """
        Return basic company information.
        {name, sector, industry, market_cap, employees,
         description, exchange, currency, country, website}
        """
        i = self.info
        return {
            'name': i.get('longName', i.get('shortName', self.raw_ticker)),
            'sector': i.get('sector', 'Unknown'),
            'industry': i.get('industry', 'Unknown'),
            'market_cap': i.get('marketCap', 0),
            'market_cap_cr': round(i.get('marketCap', 0) / 1e7, 2),  # In crores for NSE
            'employees': i.get('fullTimeEmployees', 0),
            'description': (i.get('longBusinessSummary', '') or '')[:500],
            'exchange': self.exchange,
            'currency': i.get('currency', 'INR' if self.exchange == 'NSE' else 'USD'),
            'country': i.get('country', 'India' if self.exchange == 'NSE' else 'Unknown'),
            'website': i.get('website', ''),
        }

    def get_income_statement(self, periods: int = 4) -> dict:
        """
        Quarterly income statement for last `periods` quarters.
        Key features extracted:
          revenue, revenue_growth_qoq, revenue_growth_yoy,
          gross_margin, operating_margin, net_margin, ebitda_margin,
          eps_actual, upcoming_earnings_days
        """
        result = {
            'revenue': 0, 'revenue_growth_qoq': 0, 'revenue_growth_yoy': 0,
            'gross_profit': 0, 'gross_margin': 0,
            'operating_income': 0, 'operating_margin': 0,
            'net_income': 0, 'net_margin': 0,
            'ebitda': 0, 'ebitda_margin': 0,
            'eps_actual': 0, 'upcoming_earnings_days': 999,
        }

        if not self.stock:
            return result

        try:
            q_fin = self.stock.quarterly_income_stmt
            if q_fin is None or q_fin.empty:
                return result

            # Get the most recent quarter
            cols = q_fin.columns[:periods]
            latest = q_fin[cols[0]] if len(cols) > 0 else pd.Series()

            def safe_get(series, keys):
                for key in keys:
                    if key in series.index:
                        val = series[key]
                        if pd.notna(val):
                            return float(val)
                return 0.0

            revenue = safe_get(latest, ['Total Revenue', 'Revenue', 'TotalRevenue'])
            cost = safe_get(latest, ['Cost Of Revenue', 'CostOfRevenue'])
            gross_profit = revenue - cost if cost > 0 else safe_get(latest, ['Gross Profit', 'GrossProfit'])
            operating = safe_get(latest, ['Operating Income', 'OperatingIncome', 'EBIT'])
            net = safe_get(latest, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
            ebitda = safe_get(latest, ['EBITDA', 'Ebitda'])

            result['revenue'] = revenue
            result['gross_profit'] = gross_profit
            result['gross_margin'] = (gross_profit / revenue * 100) if revenue > 0 else 0
            result['operating_income'] = operating
            result['operating_margin'] = (operating / revenue * 100) if revenue > 0 else 0
            result['net_income'] = net
            result['net_margin'] = (net / revenue * 100) if revenue > 0 else 0
            result['ebitda'] = ebitda
            result['ebitda_margin'] = (ebitda / revenue * 100) if revenue > 0 else 0

            # QoQ revenue growth
            if len(cols) >= 2:
                prev_rev = safe_get(q_fin[cols[1]], ['Total Revenue', 'Revenue', 'TotalRevenue'])
                if prev_rev > 0:
                    result['revenue_growth_qoq'] = (revenue - prev_rev) / prev_rev * 100

            # YoY revenue growth
            if len(cols) >= 4:
                yoy_rev = safe_get(q_fin[cols[3]], ['Total Revenue', 'Revenue', 'TotalRevenue'])
                if yoy_rev > 0:
                    result['revenue_growth_yoy'] = (revenue - yoy_rev) / yoy_rev * 100

            # EPS
            result['eps_actual'] = self.info.get('trailingEps', 0) or 0

            # Upcoming earnings
            try:
                cal = self.stock.calendar
                if cal is not None:
                    if isinstance(cal, dict) and 'Earnings Date' in cal:
                        earn_dates = cal['Earnings Date']
                        if earn_dates:
                            next_earn = pd.Timestamp(earn_dates[0])
                            days_to = (next_earn - pd.Timestamp.now()).days
                            result['upcoming_earnings_days'] = max(0, days_to)
                    elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
                        earn_date = cal.loc['Earnings Date'].iloc[0]
                        if pd.notna(earn_date):
                            days_to = (pd.Timestamp(earn_date) - pd.Timestamp.now()).days
                            result['upcoming_earnings_days'] = max(0, days_to)
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Error fetching income statement for {self.yf_ticker}: {e}")

        return result

    def get_balance_sheet(self) -> dict:
        """
        Latest balance sheet key ratios.
        Returns debt_to_equity, current_ratio, quick_ratio,
        book_value_per_share, price_to_book, cash, free_cash_flow,
        promoter_holding_pct, pledged_shares_pct.
        """
        result = {
            'total_assets': 0, 'total_liabilities': 0, 'total_equity': 0,
            'debt_to_equity': 0, 'current_ratio': 0, 'quick_ratio': 0,
            'book_value_per_share': 0, 'price_to_book': 0,
            'cash_and_equivalents': 0, 'free_cash_flow': 0,
            'promoter_holding_pct': 0, 'pledged_shares_pct': 0,
        }

        i = self.info
        result['debt_to_equity'] = i.get('debtToEquity', 0) or 0
        if result['debt_to_equity'] > 10:
            result['debt_to_equity'] = result['debt_to_equity'] / 100  # Normalize from %

        result['current_ratio'] = i.get('currentRatio', 0) or 0
        result['quick_ratio'] = i.get('quickRatio', 0) or 0
        result['book_value_per_share'] = i.get('bookValue', 0) or 0
        result['price_to_book'] = i.get('priceToBook', 0) or 0
        result['free_cash_flow'] = i.get('freeCashflow', 0) or 0

        if not self.stock:
            return result

        try:
            bs = self.stock.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]

                def safe_bs(keys):
                    for k in keys:
                        if k in latest.index and pd.notna(latest[k]):
                            return float(latest[k])
                    return 0.0

                result['total_assets'] = safe_bs(['Total Assets', 'TotalAssets'])
                result['total_liabilities'] = safe_bs([
                    'Total Liabilities Net Minority Interest',
                    'TotalLiabilitiesNetMinorityInterest',
                    'Total Liab'
                ])
                result['total_equity'] = safe_bs([
                    'Stockholders Equity', 'StockholdersEquity',
                    'Total Equity Gross Minority Interest'
                ])
                result['cash_and_equivalents'] = safe_bs([
                    'Cash And Cash Equivalents', 'CashAndCashEquivalents',
                    'Cash'
                ])

        except Exception as e:
            logger.warning(f"Error fetching balance sheet for {self.yf_ticker}: {e}")

        return result

    def get_valuation_ratios(self) -> dict:
        """
        Valuation ratios with sector comparison where available.
        pe_ratio, pb_ratio, ps_ratio, ev_ebitda, peg_ratio,
        dividend_yield, roe, roce, roa.
        """
        i = self.info
        return {
            'pe_ratio_ttm': i.get('trailingPE', 0) or 0,
            'pe_ratio_forward': i.get('forwardPE', 0) or 0,
            'pb_ratio': i.get('priceToBook', 0) or 0,
            'ps_ratio': i.get('priceToSalesTrailing12Months', 0) or 0,
            'ev_ebitda': i.get('enterpriseToEbitda', 0) or 0,
            'peg_ratio': i.get('pegRatio', 0) or 0,
            'dividend_yield': (i.get('dividendYield', 0) or 0) * 100,
            'payout_ratio': (i.get('payoutRatio', 0) or 0) * 100,
            'roe': (i.get('returnOnEquity', 0) or 0) * 100,
            'roa': (i.get('returnOnAssets', 0) or 0) * 100,
            'roce': 0,  # Not directly available from yfinance
            'ev': i.get('enterpriseValue', 0) or 0,
            'beta': i.get('beta', 1.0) or 1.0,
            'fifty_two_week_high': i.get('fiftyTwoWeekHigh', 0) or 0,
            'fifty_two_week_low': i.get('fiftyTwoWeekLow', 0) or 0,
            'fifty_day_avg': i.get('fiftyDayAverage', 0) or 0,
            'two_hundred_day_avg': i.get('twoHundredDayAverage', 0) or 0,
        }

    def get_event_calendar(self) -> dict:
        """
        CRITICAL for trade timing. Returns event-based risk flags.
        days_to_earnings, days_to_dividend_exdate, upcoming_splits,
        fno_ban, circuit_limit, is_illiquid.
        """
        i = self.info
        result = {
            'days_to_earnings': 999,
            'days_to_dividend_exdate': 999,
            'upcoming_splits': False,
            'fno_ban': False,
            'circuit_limit': '20%',
            'is_illiquid': False,
        }

        # Average volume check
        avg_vol = i.get('averageVolume', 0) or 0
        result['is_illiquid'] = avg_vol < 100000
        result['avg_volume'] = avg_vol

        # Dividend ex-date
        ex_div = i.get('exDividendDate', None)
        if ex_div:
            try:
                ex_date = datetime.fromtimestamp(ex_div)
                result['days_to_dividend_exdate'] = max(0, (ex_date - datetime.now()).days)
            except Exception:
                pass

        # Earnings date is already fetched in get_income_statement
        # We'll merge it in compute_all_features

        return result

    def get_institutional_activity(self) -> dict:
        """
        Institutional ownership data.
        For NSE: attempts to get FII/DII from info.
        For global: institutional_ownership_pct from yfinance.
        """
        i = self.info
        result = {
            'institutional_ownership_pct': (i.get('heldPercentInstitutions', 0) or 0) * 100,
            'insider_ownership_pct': (i.get('heldPercentInsiders', 0) or 0) * 100,
            'short_ratio': i.get('shortRatio', 0) or 0,
            'short_pct_float': (i.get('shortPercentOfFloat', 0) or 0) * 100,
        }

        # For NSE stocks, map institutional to FII and insider to promoter
        if self.exchange == 'NSE':
            result['fii_holding_pct'] = result['institutional_ownership_pct']
            result['promoter_holding_pct'] = result['insider_ownership_pct']
            result['dii_holding_pct'] = 0  # Not easily available from yfinance
            result['fii_change_qoq'] = 0
            result['promoter_change_qoq'] = 0
        else:
            result['fii_holding_pct'] = 0
            result['promoter_holding_pct'] = 0
            result['dii_holding_pct'] = 0
            result['fii_change_qoq'] = 0
            result['promoter_change_qoq'] = 0

        return result

    def compute_all_features(self) -> dict:
        """
        Aggregate all fundamental data into flat feature dict.
        Handle failures gracefully — if any source fails,
        fill with None/0 and log warning. Never crash on data error.
        Returns dict with 40+ fundamental features.
        """
        features = {}

        # Company info
        try:
            company = self.get_company_info()
            features.update({f"fund_{k}": v for k, v in company.items()
                             if isinstance(v, (int, float))})
            features['company_name'] = company.get('name', self.raw_ticker)
            features['sector'] = company.get('sector', 'Unknown')
            features['industry'] = company.get('industry', 'Unknown')
        except Exception as e:
            logger.warning(f"Company info failed: {e}")

        # Income statement
        try:
            income = self.get_income_statement()
            features.update({f"fund_{k}": v for k, v in income.items()
                             if isinstance(v, (int, float))})
        except Exception as e:
            logger.warning(f"Income statement failed: {e}")

        # Balance sheet
        try:
            bs = self.get_balance_sheet()
            features.update({f"fund_{k}": v for k, v in bs.items()
                             if isinstance(v, (int, float))})
        except Exception as e:
            logger.warning(f"Balance sheet failed: {e}")

        # Valuation ratios
        try:
            val = self.get_valuation_ratios()
            features.update({f"fund_{k}": v for k, v in val.items()
                             if isinstance(v, (int, float))})
        except Exception as e:
            logger.warning(f"Valuation ratios failed: {e}")

        # Event calendar
        try:
            events = self.get_event_calendar()
            features.update({f"fund_{k}": v for k, v in events.items()
                             if isinstance(v, (int, float, bool))})
            features['fund_fno_ban'] = int(events.get('fno_ban', False))
            features['fund_is_illiquid'] = int(events.get('is_illiquid', False))
        except Exception as e:
            logger.warning(f"Event calendar failed: {e}")

        # Institutional activity
        try:
            inst = self.get_institutional_activity()
            features.update({f"fund_{k}": v for k, v in inst.items()
                             if isinstance(v, (int, float))})
        except Exception as e:
            logger.warning(f"Institutional activity failed: {e}")

        # Ensure all values are numeric (convert bools, filter strings)
        numeric_features = {}
        for k, v in features.items():
            if isinstance(v, bool):
                numeric_features[k] = int(v)
            elif isinstance(v, (int, float)):
                if isinstance(v, float) and np.isnan(v):
                    continue  # drop NaN features
                numeric_features[k] = float(v)
            # Skip string features for ML (kept separately for display)

        # Re-add string features for display purposes
        numeric_features['_company_name'] = features.get('company_name', self.raw_ticker)
        numeric_features['_sector'] = features.get('sector', 'Unknown')
        numeric_features['_industry'] = features.get('industry', 'Unknown')

        return numeric_features
