"""
Stock Analyzer — Master orchestrator for comprehensive stock analysis.
=======================================================================

Type any NSE/NYSE/NASDAQ ticker and get a full analysis combining:
  - Technical analysis (trend, momentum, volatility, volume, patterns)
  - Fundamental analysis (valuation, growth, debt, institutional)
  - Sentiment analysis (news mood, event risks)
  - ML prediction (ensemble signals + confidence)
  - Trade plan (entry, stop, targets, timing, position sizing)
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from ..features.pipeline import FeaturePipeline
from ..features.technical import TechnicalFeatures
from ..features.fundamental import FundamentalFeatures
from ..features.sentiment import SentimentFeatures
from ..models.ensemble import QuantraEnsemble
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class StockAnalyzer:
    """
    Master analysis engine — the brain of Quantra's ANALYZE command.

    Usage:
        analyzer = StockAnalyzer()
        result = analyzer.analyze('RELIANCE', exchange='NSE')
        print(result['summary'])
    """

    def __init__(self, model_dir: str = 'models/'):
        self.model_dir = model_dir
        self.report_gen = ReportGenerator()

    def analyze(self, ticker: str, exchange: str = 'NSE',
                mode: str = 'swing') -> Dict[str, Any]:
        """
        Run full analysis pipeline.

        Returns dict with sections:
          overview, technical, fundamental, sentiment, ml_signals,
          trade_plan, risk_assessment, summary
        """
        logger.info(f"Analyzing {ticker} ({exchange}) in {mode} mode")
        result = {
            'ticker': ticker,
            'exchange': exchange,
            'mode': mode,
            'analyzed_at': datetime.now().isoformat(),
        }

        # 1. Company Overview
        try:
            ff = FundamentalFeatures(ticker, exchange=exchange)
            company = ff.get_company_info()
            result['overview'] = company
        except Exception as e:
            logger.warning(f"Overview failed: {e}")
            result['overview'] = {'name': ticker, 'sector': 'Unknown'}

        # 2. Technical Analysis
        try:
            result['technical'] = self._technical_analysis(ticker, exchange, mode)
        except Exception as e:
            logger.warning(f"Technical analysis failed: {e}")
            result['technical'] = {'error': str(e)}

        # 3. Fundamental Analysis
        try:
            result['fundamental'] = self._fundamental_analysis(ticker, exchange)
        except Exception as e:
            logger.warning(f"Fundamental analysis failed: {e}")
            result['fundamental'] = {'error': str(e)}

        # 4. Sentiment Analysis
        try:
            company_name = result['overview'].get('name', ticker)
            result['sentiment'] = self._sentiment_analysis(ticker, company_name)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            result['sentiment'] = {'error': str(e)}

        # 5. ML Signals
        try:
            result['ml_signals'] = self._ml_signals(ticker, exchange, mode)
        except Exception as e:
            logger.warning(f"ML signals failed: {e}")
            result['ml_signals'] = {'error': str(e)}

        # 6. Trade Plan
        result['trade_plan'] = self._generate_trade_plan(result)

        # 7. Risk Assessment
        result['risk_assessment'] = self._assess_risk(result)

        # 8. Summary
        result['summary'] = self.report_gen.generate_summary(result)

        return result

    def _technical_analysis(self, ticker: str, exchange: str,
                            mode: str) -> Dict:
        """Compute and interpret technical indicators."""
        import yfinance as yf

        yf_ticker = f"{ticker}.NS" if exchange == 'NSE' else ticker
        timeframe = '5m' if mode == 'intraday' else '1d'
        period = '60d' if mode == 'intraday' else '1y'
        interval = timeframe if mode == 'intraday' else '1d'

        df = yf.download(yf_ticker, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return {'error': 'No data available'}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]

        tf = TechnicalFeatures(df, timeframe=timeframe)
        enriched = tf.compute_all()

        last = enriched.iloc[-1]
        close = float(last.get('close', 0))

        # Interpret indicators
        trend_signals = []
        if last.get('ema_9', 0) > last.get('ema_21', 0):
            trend_signals.append('EMA9 > EMA21 (bullish)')
        else:
            trend_signals.append('EMA9 < EMA21 (bearish)')

        if last.get('supertrend_direction', 0) == 1:
            trend_signals.append('SuperTrend bullish')
        else:
            trend_signals.append('SuperTrend bearish')

        if last.get('adx_trending', 0):
            trend_signals.append(f'ADX={last.get("adx", 0):.0f} (trending)')
        else:
            trend_signals.append(f'ADX={last.get("adx", 0):.0f} (ranging)')

        if last.get('macd_cross_up', 0):
            trend_signals.append('MACD crossover UP')
        if last.get('macd_cross_down', 0):
            trend_signals.append('MACD crossover DOWN')

        momentum_signals = []
        rsi = last.get('rsi_14', 50)
        if rsi > 70:
            momentum_signals.append(f'RSI={rsi:.0f} OVERBOUGHT')
        elif rsi < 30:
            momentum_signals.append(f'RSI={rsi:.0f} OVERSOLD')
        else:
            momentum_signals.append(f'RSI={rsi:.0f} neutral')

        if last.get('stoch_cross_up', 0):
            momentum_signals.append('Stochastic crossover UP')

        volatility_info = {
            'atr': round(float(last.get('atr', 0)), 2),
            'atr_pct': round(float(last.get('atr_pct', 0)), 2),
            'bb_squeeze': bool(last.get('bb_squeeze', 0)),
            'ttm_squeeze': bool(last.get('ttm_squeeze', 0)),
            'bb_position': round(float(last.get('bb_position', 0.5)), 2),
        }

        volume_info = {
            'above_vwap': bool(last.get('vwap_position', 0)),
            'vwap_distance_pct': round(float(last.get('vwap_distance_pct', 0)), 2),
            'cmf': round(float(last.get('cmf', 0)), 4),
            'obv_bullish_div': bool(last.get('obv_bullish_div', 0)),
        }

        pattern_strength = int(last.get('pattern_strength', 0))
        structure = int(last.get('structure_trend', 0))

        # Bullish/bearish patterns present
        active_patterns = []
        for col in enriched.columns:
            if col.startswith('pat_') and last.get(col, 0) == 1:
                active_patterns.append(col.replace('pat_', '').replace('_', ' ').title())

        # Key levels
        levels = {
            'pivot_pp': round(float(last.get('pivot_pp', 0)), 2),
            'nearest_resistance': round(float(last.get('nearest_resistance', 0)), 2),
            'nearest_support': round(float(last.get('nearest_support', 0)), 2),
            'vwap': round(float(last.get('vwap', 0)), 2),
            'supertrend_level': round(float(last.get('supertrend_level', 0)), 2),
            'bb_upper': round(float(last.get('bb_upper', 0)), 2),
            'bb_lower': round(float(last.get('bb_lower', 0)), 2),
        }

        return {
            'close': close,
            'trend_signals': trend_signals,
            'momentum_signals': momentum_signals,
            'volatility': volatility_info,
            'volume': volume_info,
            'pattern_strength': pattern_strength,
            'active_patterns': active_patterns,
            'structure_trend': {1: 'UPTREND', -1: 'DOWNTREND', 0: 'RANGING'}
                              .get(structure, 'RANGING'),
            'key_levels': levels,
            'bos_bullish': bool(last.get('bos_bullish', 0)),
            'bos_bearish': bool(last.get('bos_bearish', 0)),
        }

    def _fundamental_analysis(self, ticker: str, exchange: str) -> Dict:
        """Fetch and interpret fundamentals."""
        ff = FundamentalFeatures(ticker, exchange=exchange)
        valuation = ff.get_valuation_ratios()
        income = ff.get_income_statement()
        balance = ff.get_balance_sheet()
        events = ff.get_event_calendar()
        institutional = ff.get_institutional_activity()

        # Score
        score = 0
        reasons = []

        pe = valuation.get('pe_ratio_ttm', 0)
        if 0 < pe < 15:
            score += 2; reasons.append(f'Low PE ({pe:.1f}) — VALUE')
        elif 15 <= pe < 30:
            score += 1; reasons.append(f'Fair PE ({pe:.1f})')
        elif pe >= 30:
            score -= 1; reasons.append(f'High PE ({pe:.1f}) — GROWTH')

        roe = valuation.get('roe', 0)
        if roe > 20:
            score += 2; reasons.append(f'Strong ROE ({roe:.1f}%)')
        elif roe > 10:
            score += 1; reasons.append(f'Decent ROE ({roe:.1f}%)')

        de = balance.get('debt_to_equity', 0)
        if de < 0.5:
            score += 1; reasons.append(f'Low debt (D/E: {de:.2f})')
        elif de > 2:
            score -= 2; reasons.append(f'HIGH debt (D/E: {de:.2f}) ⚠')

        rev_growth = income.get('revenue_growth_yoy', 0)
        if rev_growth > 20:
            score += 2; reasons.append(f'Strong revenue growth ({rev_growth:.0f}% YoY)')
        elif rev_growth > 5:
            score += 1; reasons.append(f'Moderate growth ({rev_growth:.0f}% YoY)')
        elif rev_growth < -5:
            score -= 1; reasons.append(f'Revenue declining ({rev_growth:.0f}% YoY)')

        return {
            'valuation': valuation,
            'income': income,
            'balance': balance,
            'events': events,
            'institutional': institutional,
            'score': score,
            'max_score': 7,
            'reasoning': reasons,
            'grade': 'A' if score >= 5 else 'B' if score >= 3 else 'C' \
                     if score >= 1 else 'D' if score >= -1 else 'F',
        }

    def _sentiment_analysis(self, ticker: str, company_name: str) -> Dict:
        """Run NLP sentiment analysis."""
        sf = SentimentFeatures(ticker, company_name=company_name)
        features = sf.compute_sentiment_features()
        headlines = sf.get_news_summary(n=5)

        mood = 'POSITIVE' if features.get('sent_news_sentiment_24h', 0) > 0.2 else \
               'NEGATIVE' if features.get('sent_news_sentiment_24h', 0) < -0.2 else \
               'NEUTRAL'

        return {
            'features': features,
            'top_headlines': headlines,
            'overall_mood': mood,
            'risk_flags': [
                'REGULATORY RISK' if features.get('sent_mentions_regulatory', 0) else None,
                'FRAUD RISK' if features.get('sent_mentions_fraud', 0) else None,
                'EARNINGS MISS' if features.get('sent_mentions_earnings_miss', 0) else None,
                'GEOPOLITICAL RISK' if features.get('sent_mentions_geopolitical', 0) else None,
            ],
        }

    def _ml_signals(self, ticker: str, exchange: str, mode: str) -> Dict:
        """Get ensemble ML signals."""
        ensemble = QuantraEnsemble(mode=mode, model_dir=self.model_dir)
        loaded = ensemble.load_all_models()

        if not loaded:
            return {
                'available': False,
                'note': 'No trained models. Run training first.',
            }

        try:
            pipeline = FeaturePipeline(
                ticker=ticker, exchange=exchange,
                timeframe='5m' if mode == 'intraday' else '1d',
                mode=mode
            )
            latest = pipeline.get_latest_features()
            if latest is not None:
                prediction = ensemble.predict(X_flat=latest)
                return {
                    'available': True,
                    'signal': prediction['final_signal'],
                    'confidence': prediction['confidence'],
                    'agreement': prediction['agreement'],
                    'model_signals': prediction['model_signals'],
                    'trade_recommendation': prediction['trade_recommendation'],
                }
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")

        return {
            'available': False,
            'note': 'Feature computation failed.',
        }

    def _generate_trade_plan(self, result: Dict) -> Dict:
        """Build actionable trade plan from all analysis."""
        tech = result.get('technical', {})
        fund = result.get('fundamental', {})
        sent = result.get('sentiment', {})
        ml = result.get('ml_signals', {})

        close = tech.get('close', 0)

        # Composite verdict
        scores = []  # (weight, score)

        # Technical score (-2 to +2)
        tech_score = 0
        if 'UPTREND' in tech.get('structure_trend', ''):
            tech_score += 1
        if any('bullish' in s.lower() for s in tech.get('trend_signals', [])):
            tech_score += 1
        if any('bearish' in s.lower() for s in tech.get('trend_signals', [])):
            tech_score -= 1
        tech_score += tech.get('pattern_strength', 0) * 0.5
        tech_score = max(-2, min(2, tech_score))
        scores.append((0.40, tech_score))

        # Fundamental score (-2 to +2)
        fund_score = fund.get('score', 0) / max(fund.get('max_score', 1), 1) * 2
        scores.append((0.25, fund_score))

        # Sentiment score (-2 to +2)
        mood = sent.get('overall_mood', 'NEUTRAL')
        sent_score = 1 if mood == 'POSITIVE' else -1 if mood == 'NEGATIVE' else 0
        scores.append((0.15, sent_score))

        # ML score (-2 to +2)
        if ml.get('available'):
            ml_signal = ml.get('signal', 'HOLD')
            ml_conf = ml.get('confidence', 0.5)
            ml_score = (2 if ml_signal == 'BUY' else -2 if ml_signal == 'SELL' else 0) * ml_conf
            scores.append((0.20, ml_score))
        else:
            scores.append((0.20, 0))

        # Weighted composite
        composite = sum(w * s for w, s in scores) / sum(w for w, _ in scores)

        if composite > 0.5:
            verdict = 'STRONG BUY'
        elif composite > 0.2:
            verdict = 'BUY'
        elif composite < -0.5:
            verdict = 'STRONG SELL'
        elif composite < -0.2:
            verdict = 'SELL'
        else:
            verdict = 'HOLD'

        # Levels
        levels = tech.get('key_levels', {})
        atr = tech.get('volatility', {}).get('atr', 0)

        return {
            'verdict': verdict,
            'composite_score': round(composite, 3),
            'entry_price': close,
            'stop_loss': round(close - 2 * atr, 2) if verdict in ('BUY', 'STRONG BUY') \
                         else round(close + 2 * atr, 2) if verdict in ('SELL', 'STRONG SELL') \
                         else 0,
            'target_1': round(close + 3 * atr, 2) if 'BUY' in verdict \
                        else round(close - 3 * atr, 2) if 'SELL' in verdict else 0,
            'target_2': round(close + 5 * atr, 2) if 'BUY' in verdict \
                        else round(close - 5 * atr, 2) if 'SELL' in verdict else 0,
            'key_resistance': levels.get('nearest_resistance', 0),
            'key_support': levels.get('nearest_support', 0),
            'risk_reward': round(3 * atr / (2 * atr), 2) if atr > 0 else 0,
        }

    def _assess_risk(self, result: Dict) -> Dict:
        """Assess overall risk level."""
        risks = []
        risk_score = 0

        # Volatility risk
        vol = result.get('technical', {}).get('volatility', {})
        if vol.get('atr_pct', 0) > 3:
            risks.append('HIGH VOLATILITY')
            risk_score += 2

        # Near earnings
        earnings = result.get('fundamental', {}).get('events', {})
        days_to_earn = earnings.get('upcoming_earnings_days', 999)
        if days_to_earn < 7:
            risks.append(f'EARNINGS IN {days_to_earn} DAYS')
            risk_score += 2

        # Negative sentiment
        mood = result.get('sentiment', {}).get('overall_mood', 'NEUTRAL')
        if mood == 'NEGATIVE':
            risks.append('NEGATIVE NEWS SENTIMENT')
            risk_score += 1

        # Sentiment risk flags
        for flag in result.get('sentiment', {}).get('risk_flags', []):
            if flag:
                risks.append(flag)
                risk_score += 1

        # Overvaluation
        pe = result.get('fundamental', {}).get('valuation', {}).get('pe_ratio_ttm', 0)
        if pe > 80:
            risks.append(f'EXTREME VALUATION (PE: {pe:.0f})')
            risk_score += 2

        level = 'LOW' if risk_score <= 1 else 'MEDIUM' if risk_score <= 3 \
                else 'HIGH' if risk_score <= 5 else 'EXTREME'

        return {
            'level': level,
            'score': risk_score,
            'flags': [r for r in risks if r],
        }
