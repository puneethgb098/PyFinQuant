import numpy as np
import pyfinquant as pfq
import pandas as pd
from datetime import datetime, timedelta

def test_returns():
    print("\n=== Testing Returns Calculations ===")
    
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.random.normal(100, 5, 100).cumsum() + 1000
    
    # Calculate returns using both libraries
    simple_returns = pfq.returns(prices)
    log_returns = pfq.log_returns(prices)
    
    # Print results
    print("\nSimple Returns:")
    print(f"First 5 values: {simple_returns[:5]}")
    print(f"Mean: {np.mean(simple_returns):.6f}")
    print(f"Std: {np.std(simple_returns):.6f}")
    
    print("\nLog Returns:")
    print(f"First 5 values: {log_returns[:5]}")
    print(f"Mean: {np.mean(log_returns):.6f}")
    print(f"Std: {np.std(log_returns):.6f}")

def test_volatility():
    print("\n=== Testing Volatility Calculations ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    
    # Calculate volatility using both libraries
    daily_vol = pfq.volatility(returns)
    annual_vol = pfq.volatility(returns, annualize=True)
    
    # Print results
    print(f"Daily Volatility: {daily_vol:.6f}")
    print(f"Annualized Volatility: {annual_vol:.6f}")

def test_sharpe_ratio():
    print("\n=== Testing Sharpe Ratio Calculations ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    risk_free_rate = 0.02
    
    # Calculate Sharpe ratio using both libraries
    daily_sharpe = pfq.sharpe_ratio(returns, risk_free_rate)
    annual_sharpe = pfq.sharpe_ratio(returns, risk_free_rate, annualize=True)
    
    # Print results
    print(f"Daily Sharpe Ratio: {daily_sharpe:.6f}")
    print(f"Annualized Sharpe Ratio: {annual_sharpe:.6f}")

def main():
    print("Running manual tests for core module...")
    test_returns()
    test_volatility()
    test_sharpe_ratio()

if __name__ == "__main__":
    main() 