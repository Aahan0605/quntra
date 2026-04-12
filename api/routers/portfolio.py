import time
from fastapi import APIRouter, HTTPException
import yfinance as yf

from api.schemas.models import TickerList, PortfolioResponse
from api.services.cache import cache
from src.portfolio.markowitz import fetch_nifty50_prices, compute_stats, max_sharpe_portfolio, efficient_frontier

router = APIRouter(prefix="/portfolio", tags=["Portfolio Optimization"])

@router.post("/optimize", response_model=PortfolioResponse)
def optimize_portfolio(request: TickerList):
    """Run Markowitz mean-variance optimization on historical data."""
    cache_key = f"markowitz_{'-'.join(sorted(request.tickers))}_{request.start_date}_{request.end_date}"
    
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    start_time = time.time()
    try:
        prices = fetch_nifty50_prices(request.tickers, request.start_date, request.end_date)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data found for the given dates and tickers")
            
        mu, Sigma, names = compute_stats(prices)
        result = max_sharpe_portfolio(mu, Sigma, names)
        
        result["computation_time"] = round(time.time() - start_time, 4)
        
        cache.set(cache_key, result, ttl_seconds=3600)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/efficient-frontier")
def get_efficient_frontier(request: TickerList):
    """Compute the efficient frontier curve points."""
    cache_key = f"ef_{'-'.join(sorted(request.tickers))}_{request.start_date}_{request.end_date}"
    
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    try:
        prices = fetch_nifty50_prices(request.tickers, request.start_date, request.end_date)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data found")
            
        mu, Sigma, names = compute_stats(prices)
        frontier_df = efficient_frontier(mu, Sigma)
        
        result = {
            "target_return": frontier_df["target_return"].tolist(),
            "volatility": frontier_df["volatility"].tolist()
        }
        cache.set(cache_key, result, ttl_seconds=3600)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
