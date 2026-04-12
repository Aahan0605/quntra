from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class TickerList(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers (e.g., RELIANCE.NS)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

class PortfolioResponse(BaseModel):
    weights: Dict[str, float] = Field(..., description="Optimized portfolio weights")
    expected_return: float = Field(..., description="Expected annualized return")
    volatility: float = Field(..., description="Expected annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio of the portfolio")
    computation_time: float = Field(..., description="Time taken to compute in seconds")

class QuantumResponse(PortfolioResponse):
    circuit_depth: int = Field(..., description="Depth of QAOA circuit")
    optimization_steps: int = Field(..., description="Number of steps QAOA took")
    quantum_backend: str = Field(..., description="Backend used (e.g., AerSimulator)")

class OptionsRequest(BaseModel):
    ticker: str = Field(..., description="Underlying ticker symbol")
    strike_price: float = Field(..., description="Strike price of the option")
    expiry_date: str = Field(..., description="Expiry date in YYYY-MM-DD format")
    option_type: str = Field("call", description="'call' or 'put'")
    risk_free_rate: float = Field(0.07, description="Risk-free rate (default 7% for India)")

class OptionData(BaseModel):
    strike: float
    type: str
    price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class OptionsChainResponse(BaseModel):
    ticker: str
    expiry_date: str
    current_price: float
    options: List[OptionData]

class BacktestRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers")
    weights: Dict[str, float] = Field(..., description="Weights of the portfolio")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    initial_capital: float = Field(1000000.0, description="Initial capital in INR")
    rebalance_freq: str = Field("monthly", description="'daily', 'weekly', 'monthly', 'quarterly'")

class BacktestMetricsSchema(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float

class BacktestResponse(BaseModel):
    metrics: BacktestMetricsSchema
    equity_curve: Dict[str, float] = Field(..., description="Date to portfolio value mapping")
    drawdown_curve: Dict[str, float] = Field(..., description="Date to drawdown mapping")
    attribution: Dict[str, float] = Field(..., description="Ticker to return contribution mapping")
