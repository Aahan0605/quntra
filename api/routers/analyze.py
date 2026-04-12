"""
Analyze API Router — Stock analysis endpoints.
================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    ticker: str
    exchange: str = "NSE"
    mode: str = "swing"  # 'intraday' or 'swing'


@router.post("/stock")
async def analyze_stock(req: AnalyzeRequest):
    """
    Full stock analysis: technical + fundamental + sentiment + ML.
    Returns complete analysis with trade plan.
    """
    try:
        from src.ml.analysis.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        result = analyzer.analyze(
            ticker=req.ticker,
            exchange=req.exchange,
            mode=req.mode
        )

        return {
            'status': 'success',
            'ticker': req.ticker,
            'summary': result.get('summary', ''),
            'verdict': result.get('trade_plan', {}).get('verdict', 'HOLD'),
            'risk_level': result.get('risk_assessment', {}).get('level', 'MEDIUM'),
            'trade_plan': result.get('trade_plan', {}),
            'technical': result.get('technical', {}),
            'fundamental_grade': result.get('fundamental', {}).get('grade', 'N/A'),
            'sentiment_mood': result.get('sentiment', {}).get('overall_mood', 'NEUTRAL'),
            'ml_signal': result.get('ml_signals', {}).get('signal', None),
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technical/{ticker}")
async def get_technical(ticker: str, exchange: str = "NSE",
                        timeframe: str = "1d"):
    """Get technical analysis only."""
    try:
        from src.ml.analysis.stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        result = analyzer._technical_analysis(ticker, exchange, 'swing')
        return {'status': 'success', 'technical': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fundamental/{ticker}")
async def get_fundamental(ticker: str, exchange: str = "NSE"):
    """Get fundamental analysis only."""
    try:
        from src.ml.analysis.stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        result = analyzer._fundamental_analysis(ticker, exchange)
        return {'status': 'success', 'fundamental': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/{ticker}")
async def get_sentiment(ticker: str, company_name: str = ""):
    """Get sentiment analysis only."""
    try:
        from src.ml.features.sentiment import SentimentFeatures
        sf = SentimentFeatures(ticker, company_name=company_name or ticker)
        features = sf.compute_sentiment_features()
        headlines = sf.get_news_summary(n=5)
        return {
            'status': 'success',
            'features': features,
            'headlines': headlines,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals")
async def get_live_signals():
    """
    Get active signals for watchlist tickers.
    Used by the SIGNALS panel in the terminal.
    """
    try:
        from src.ml.models.ensemble import QuantraEnsemble
        from src.ml.features.pipeline import FeaturePipeline

        # Default watchlist (can be made configurable)
        watchlist = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

        signals = []
        ensemble = QuantraEnsemble(mode='swing')
        loaded = ensemble.load_all_models()

        for ticker in watchlist:
            try:
                if loaded:
                    pipeline = FeaturePipeline(
                        ticker=ticker, exchange='NSE',
                        timeframe='1d', mode='swing'
                    )
                    latest = pipeline.get_latest_features()
                    if latest is not None:
                        pred = ensemble.predict(X_flat=latest)
                        signals.append({
                            'ticker': ticker,
                            'signal': pred['final_signal'],
                            'confidence': pred['confidence'],
                            'agreement': pred['agreement'],
                        })
                    else:
                        signals.append({
                            'ticker': ticker,
                            'signal': 'NO DATA',
                            'confidence': 0,
                        })
                else:
                    signals.append({
                        'ticker': ticker,
                        'signal': 'UNTRAINED',
                        'confidence': 0,
                    })
            except Exception:
                signals.append({
                    'ticker': ticker,
                    'signal': 'ERROR',
                    'confidence': 0,
                })

        return {'signals': signals, 'count': len(signals)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
