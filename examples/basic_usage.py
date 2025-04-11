"""
Basic usage examples for PyFinQuant.
"""

import numpy as np
import pyfinquant as pfq

def main():
    """Run basic usage examples."""
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, 100)) + 100
    
    simple_returns = pfq.returns(prices)
    log_returns = pfq.log_returns(prices)
    
    var_95 = pfq.historical_var(prices, confidence_level=0.95)
    max_dd = pfq.max_drawdown(prices)
    
    print("Basic Statistics:")
    print(f"Simple Returns Mean: {np.mean(simple_returns):.4f}")
    print(f"Log Returns Mean: {np.mean(log_returns):.4f}")
    print(f"95% VaR: {var_95:.4f}")
    print(f"Maximum Drawdown: {max_dd:.4f}")
    
    S = 100  # Current price
    K = 100  # Strike price
    T = 1.0  # Time to maturity in years
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    q = 0.02  # Dividend yield
    
    model = pfq.BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type='call', q=q)
    
    price = model.price()
    print("\nOption Pricing:")
    print(f"Option Price: {price:.4f}")
    
    greeks = pfq.AnalyticalGreeks(model)
    print("\nOption Greeks:")
    print(f"Delta: {greeks.delta():.4f}")
    print(f"Gamma: {greeks.gamma():.4f}")
    print(f"Vega: {greeks.vega():.4f}")
    print(f"Rho: {greeks.rho():.4f}")
    print(f"Theta: {greeks.theta():.4f}")
    
    model_no_div = pfq.BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type='call')
    greeks_no_div = pfq.AnalyticalGreeks(model_no_div)
    
    print("\nComparison with Non-Dividend Case:")
    print(f"Price Difference: {model_no_div.price() - price:.4f}")
    print(f"Delta Difference: {greeks_no_div.delta() - greeks.delta():.4f}")

if __name__ == "__main__":
    main() 
