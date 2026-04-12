"""
ML API Router — FastAPI endpoints for ML/DL intelligence layer.
================================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    ticker: str = "RELIANCE"
    exchange: str = "NSE"
    mode: str = "intraday"  # 'intraday' or 'swing'
    timesteps: int = 50000


class PredictRequest(BaseModel):
    ticker: str
    exchange: str = "NSE"
    mode: str = "intraday"


class PaperTradeRequest(BaseModel):
    ticker: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str = "MARKET"
    price: float = 0
    current_price: float = 0


@router.post("/train")
async def train_models(req: TrainRequest):
    """Train all ML models for a ticker."""
    try:
        from src.ml.features.pipeline import FeaturePipeline
        from src.ml.models.ensemble import QuantraEnsemble

        # Build features
        pipeline = FeaturePipeline(
            ticker=req.ticker, exchange=req.exchange,
            timeframe='5m' if req.mode == 'intraday' else '1d',
            mode=req.mode
        )
        X, y = pipeline.build_feature_matrix()
        split = pipeline.split_data(X, y)

        # Train XGBoost
        ensemble = QuantraEnsemble(mode=req.mode)
        xgb_result = ensemble.xgb_model.train(
            split['X_train'], split['y_train'],
            split['X_val'], split['y_val'],
            n_trials=20
        )
        ensemble.xgb_model.save()

        # Train LSTM
        lstm_result = ensemble.lstm_model.train(
            split['X_train'].values, split['y_train'].values,
            split['X_val'].values, split['y_val'].values,
            epochs=50
        )

        return {
            'status': 'success',
            'ticker': req.ticker,
            'mode': req.mode,
            'xgboost': {
                'auc': xgb_result.get('val_auc', 0),
                'accuracy': xgb_result.get('val_accuracy', 0),
            },
            'lstm': {
                'auc': lstm_result.get('val_auc', 0),
                'epochs': lstm_result.get('n_epochs_trained', 0),
            },
            'features': len(pipeline.get_feature_names()),
            'samples': len(X),
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_signal(req: PredictRequest):
    """Get ensemble prediction for a ticker."""
    try:
        from src.ml.models.ensemble import QuantraEnsemble
        from src.ml.features.pipeline import FeaturePipeline

        ensemble = QuantraEnsemble(mode=req.mode)
        loaded = ensemble.load_all_models()

        if not loaded:
            return {
                'status': 'no_models',
                'message': 'No trained models found. Run /ml/train first.',
            }

        pipeline = FeaturePipeline(
            ticker=req.ticker, exchange=req.exchange,
            timeframe='5m' if req.mode == 'intraday' else '1d',
            mode=req.mode
        )
        latest = pipeline.get_latest_features()

        if latest is None:
            raise HTTPException(status_code=400, detail="Could not compute features")

        prediction = ensemble.predict(X_flat=latest)

        return {
            'status': 'success',
            'signal': prediction['final_signal'],
            'confidence': prediction['confidence'],
            'agreement': prediction['agreement'],
            'trade_recommendation': prediction['trade_recommendation'],
            'model_signals': {
                k: {'signal': v.get('signal', 'N/A'), 'confidence': v.get('confidence', 0)}
                for k, v in prediction.get('model_signals', {}).items()
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper-trade")
async def paper_trade(req: PaperTradeRequest):
    """Execute a paper trade."""
    try:
        from src.ml.paper_trading.paper_broker import PaperBroker

        broker = PaperBroker()
        result = broker.place_order(
            ticker=req.ticker,
            side=req.side,
            quantity=req.quantity,
            order_type=req.order_type,
            price=req.price,
            current_price=req.current_price,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio():
    """Get paper trading portfolio."""
    try:
        from src.ml.paper_trading.paper_broker import PaperBroker
        broker = PaperBroker()
        return broker.get_portfolio()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/journal")
async def get_trade_journal():
    """Get trade journal stats."""
    try:
        from src.ml.paper_trading.trade_journal import TradeJournal
        journal = TradeJournal()
        return {
            'stats': journal.get_performance_stats(),
            'model_accuracy': journal.get_model_accuracy(),
            'recent_trades': journal.get_recent_trades(n=10),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl/train")
async def train_rl_agent(req: TrainRequest):
    """Train RL trading agent."""
    try:
        from src.ml.rl.trainer import RLTrainer
        trainer = RLTrainer()
        results = trainer.train_from_scratch(
            ticker=req.ticker, mode=req.mode,
            timesteps=req.timesteps
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl/improve")
async def rl_self_improve():
    """Trigger RL self-improvement loop."""
    try:
        from src.ml.rl.agent import QuantraPPOAgent
        from src.ml.rl.environment import QuantraTradingEnv
        from src.ml.paper_trading.trade_journal import TradeJournal

        journal = TradeJournal()
        recent = journal.get_recent_trades(n=20)

        agent = QuantraPPOAgent()
        if not agent.load():
            return {'status': 'no_model', 'message': 'Train RL agent first.'}

        result = agent.self_improve(recent)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
