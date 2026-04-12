import pandas as pd

class PerformanceAttribution:
    """Calculates performance attribution for a generated backtest portfolio."""
    
    @staticmethod
    def calculate_attribution(weights: dict, asset_returns_df: pd.DataFrame) -> dict:
        """
        Calculate stock-level contribution to the overall portfolio return.
        Contribution = Weight * Total Return of Asset over period
        """
        if asset_returns_df.empty:
            return {}
            
        # Get total return for each asset over the period
        cumulative_asset_returns = (1 + asset_returns_df).cumprod() - 1
        period_returns = cumulative_asset_returns.iloc[-1]
        
        contributions = {}
        for ticker, weight in weights.items():
            if ticker in period_returns.index:
                # Stock contribution to total return
                contrib = float(weight * period_returns[ticker])
                contributions[ticker] = contrib
                
        return contributions
