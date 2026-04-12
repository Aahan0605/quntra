from fastapi import APIRouter, HTTPException
import numpy as np
import yfinance as yf
from datetime import datetime

from api.schemas.models import OptionsRequest, OptionsChainResponse, OptionData
from src.options import black_scholes as bs

router = APIRouter(prefix="/options", tags=["Options Pricing"])

@router.post("/price")
def get_option_price(request: OptionsRequest):
    """Calculate the Black-Scholes price and Greeks for a specific option."""
    try:
        # Fetch current price
        ticker_data = yf.Ticker(request.ticker)
        current_price = ticker_data.history(period="1d")['Close'].iloc[-1]
        
        # Calculate time to expiry (T)
        start_date = datetime.now()
        end_date = datetime.strptime(request.expiry_date, "%Y-%m-%d")
        T = (end_date - start_date).days / 365.0
        
        if T <= 0:
            raise HTTPException(status_code=400, detail="Expiry date must be in the future")
            
        # Simplified implied volatility assumption (0.2 or 20%)
        # In a real system, we'd calculate this from historical prices or an IV surface
        sigma = 0.20
        
        if request.option_type.lower() == "call":
            price = bs.call_price(current_price, request.strike_price, T, request.risk_free_rate, sigma)
        else:
            price = bs.put_price(current_price, request.strike_price, T, request.risk_free_rate, sigma)
            
        greeks = bs.greeks(current_price, request.strike_price, T, request.risk_free_rate, sigma, request.option_type.lower())
        
        return {
            "underlying_price": round(current_price, 2),
            "price": round(price, 4),
            "greeks": {k: round(v, 4) for k, v in greeks.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chain", response_model=OptionsChainResponse)
def get_options_chain(request: OptionsRequest):
    """Generate a mock options chain with calculated Black-Scholes Greeks around current price."""
    try:
        ticker_data = yf.Ticker(request.ticker)
        current_price = ticker_data.history(period="1d")['Close'].iloc[-1]
        
        start_date = datetime.now()
        end_date = datetime.strptime(request.expiry_date, "%Y-%m-%d")
        T = (end_date - start_date).days / 365.0
        
        if T <= 0:
            raise HTTPException(status_code=400, detail="Expiry date must be in the future")
            
        sigma = 0.20
        
        # Generate strike prices from -10% to +10% of current price at round intervals
        base_strike = round(current_price / 50) * 50
        strikes = np.arange(base_strike * 0.9, base_strike * 1.1, 50)
        
        options_list = []
        for strike in strikes:
            for opt_type in ['call', 'put']:
                if opt_type == 'call':
                    price = bs.call_price(current_price, strike, T, request.risk_free_rate, sigma)
                else:
                    price = bs.put_price(current_price, strike, T, request.risk_free_rate, sigma)
                    
                g = bs.greeks(current_price, strike, T, request.risk_free_rate, sigma, opt_type)
                
                options_list.append(OptionData(
                    strike=round(strike, 2),
                    type=opt_type,
                    price=round(price, 2),
                    implied_volatility=sigma,
                    delta=round(g['delta'], 4),
                    gamma=round(g['gamma'], 4),
                    theta=round(g['theta'], 4),
                    vega=round(g['vega'], 4),
                    rho=round(g['rho'], 4)
                ))
                
        return OptionsChainResponse(
            ticker=request.ticker,
            expiry_date=request.expiry_date,
            current_price=round(current_price, 2),
            options=options_list
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
