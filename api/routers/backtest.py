from fastapi import APIRouter, HTTPException
import traceback

from api.schemas.models import BacktestRequest, BacktestResponse, TickerList
from src.backtest.engine import BacktestEngine
from api.services.cache import cache

router = APIRouter(prefix="/backtest", tags=["Strategy Backtesting"])

@router.post("/run", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest):
    """Run historical backtest for a specific static weights portfolio."""
    cache_key = f"backtest_{'-'.join(sorted(request.tickers))}_{request.start_date}_{request.end_date}_{request.rebalance_freq}_{hash(frozenset(request.weights.items()))}"
    
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    try:
        engine = BacktestEngine(
            tickers=request.tickers,
            weights_dict=request.weights,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )
        
        result = engine.run(rebalance_freq=request.rebalance_freq)
        cache.set(cache_key, result, ttl_seconds=3600)
        return result
        
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantum-vs-classical")
def compare_backtests(request: TickerList):
    """
    Utility endpoint that:
    1. Fetches Markowitz weights.
    2. Fetches QAOA weights.
    3. Runs a backtest for both on the identical timeframe.
    4. Returns dual curves and metrics.
    """
    from src.portfolio.markowitz import max_sharpe_portfolio, compute_stats
    from src.quantum.benchmarker import QuantumClassicalBenchmark
    from src.utils.data_loader import fetch_nifty50_prices, get_returns
    
    cache_key = f"qvc_backtest_{'-'.join(sorted(request.tickers))}_{request.start_date}_{request.end_date}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    try:
        # We need prices for the optimisation phase (let's say we optimise over the first part, 
        # or just use the whole period for simplicity here as we are just comparing engine mechanics)
        prices = fetch_nifty50_prices(request.tickers, request.start_date, request.end_date)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data found")
            
        returns_df = get_returns(prices)
        
        # 1. Classical Weights
        mu, Sigma, names = compute_stats(prices)
        c_res = max_sharpe_portfolio(mu, Sigma, names)
        c_weights = c_res["weights"]
        
        # 2. Quantum Weights
        n_assets = min(len(request.tickers) // 2, 5)
        n_assets = max(3, n_assets)
        benchmarker = QuantumClassicalBenchmark(returns_df, names)
        q_res = benchmarker.run_quantum(n_assets_to_select=n_assets, p_layers=1, max_iterations=20)
        q_weights = q_res["weights"]
        
        # 3. Backtest Both
        c_engine = BacktestEngine(request.tickers, c_weights, request.start_date, request.end_date)
        c_bt = c_engine.run(rebalance_freq='monthly')
        
        q_engine = BacktestEngine(request.tickers, q_weights, request.start_date, request.end_date)
        q_bt = q_engine.run(rebalance_freq='monthly')
        
        result = {
            "classical": c_bt,
            "quantum": q_bt
        }
        
        cache.set(cache_key, result, ttl_seconds=3600)
        return result
        
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
