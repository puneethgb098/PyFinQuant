"""
Basic usage examples for PyFinQuant.
"""

import numpy as np
import pyfinquant as pfq

def main():
    # Generate sample price data
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
    
    # Calculate returns
    returns = pfq.returns(prices)
    log_returns = pfq.log_returns(prices)
    
    print(f"Simple returns mean: {returns.mean():.4f}")
    print(f"Log returns mean: {log_returns.mean():.4f}")
    
    # Calculate risk metrics
    var = pfq.historical_var(returns)
    mdd = pfq.max_drawdown(prices)
    
    print(f"Historical VaR (95%): {var:.4f}")
    print(f"Maximum Drawdown: {mdd:.4f}")
    
    # Price an option
    option = pfq.Option(
        S=100,  # Spot price
        K=100,  # Strike price
        T=1.0,  # Time to maturity
        r=0.05,  # Risk-free rate
        sigma=0.2,  # Volatility
        option_type=pfq.OptionType.CALL
    )
    
    # Calculate Greeks
    greeks = pfq.AnalyticalGreeks()
    delta = greeks.delta(
        S=option.S,
        K=option.K,
        T=option.T,
        r=option.r,
        sigma=option.sigma,
        option_type=option.option_type
    )
    
    print(f"Option Delta: {delta:.4f}")

if __name__ == "__main__":
    main() 