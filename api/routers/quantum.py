import time
from fastapi import APIRouter, HTTPException
import yfinance as yf

from api.schemas.models import TickerList, QuantumResponse
from api.services.cache import cache
from src.utils.data_loader import fetch_nifty50_prices, get_returns
from src.quantum.benchmarker import QuantumClassicalBenchmark

router = APIRouter(prefix="/quantum", tags=["Quantum Optimization Layer"])

@router.post("/optimize", response_model=QuantumResponse)
def optimize_quantum(request: TickerList):
    """Run QAOA (Quantum Approximate Optimization Algorithm) portfolio optimization."""
    cache_key = f"qaoa_{'-'.join(sorted(request.tickers))}_{request.start_date}_{request.end_date}"
    
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    start_time = time.time()
    try:
        prices = fetch_nifty50_prices(request.tickers, request.start_date, request.end_date)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data found")
            
        returns_df = get_returns(prices)
        
        benchmarker = QuantumClassicalBenchmark(returns_df, list(prices.columns))
        
        # We select roughly half the assets, up to a modest number so QAOA doesn't blow up locally
        n_assets = min(len(request.tickers) // 2, 8)
        n_assets = max(3, n_assets)
        
        q_res = benchmarker.run_quantum(n_assets_to_select=n_assets, p_layers=1, max_iterations=20)
        
        # Prepare the exact schema payload
        result = {
            "weights": q_res["weights"],
            "expected_return": q_res["expected_return"],
            "volatility": q_res["expected_risk"],
            "sharpe_ratio": q_res["sharpe_ratio"],
            "computation_time": round(time.time() - start_time, 4),
            "circuit_depth": q_res.get("circuit_info", {}).get("depth", 0),
            "optimization_steps": 20, # configured max_iterations
            "quantum_backend": "qiskit.AerSimulator"
        }
        
        # Cache for 1 hour
        cache.set(cache_key, result, ttl_seconds=3600)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark")
def benchmark_pipelines(request: TickerList):
    """Compare Classical Markowitz vs Quantum QAOA pipelines."""
    cache_key = f"benchmark_{'-'.join(sorted(request.tickers))}_{request.start_date}_{request.end_date}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    try:
        prices = fetch_nifty50_prices(request.tickers, request.start_date, request.end_date)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data found")
            
        returns_df = get_returns(prices)
        benchmarker = QuantumClassicalBenchmark(returns_df, list(prices.columns))
        
        # Small params for API speed: p_layers=1, max_iter=20
        n_assets = min(len(request.tickers) // 2, 5)
        n_assets = max(3, n_assets)
        
        comparison = benchmarker.compare(n_assets_to_select=n_assets, p_layers=1, max_iterations=20)
        cache.set(cache_key, comparison, ttl_seconds=3600)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invalidate-cache")
def invalidate_quantum_cache():
    """Manually clear the quantum caches."""
    cache.clear()
    return {"status": "Cache cleared."}
