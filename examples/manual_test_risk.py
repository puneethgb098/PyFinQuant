import numpy as np
import pyfinquant as pfq
import pandas as pd
from datetime import datetime, timedelta

def test_var():
    print("\n=== Testing Value at Risk (VaR) Calculations ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Calculate different types of VaR
    historical_var = pfq.historical_var(returns, confidence_level=0.95)
    parametric_var = pfq.parametric_var(returns, confidence_level=0.95)
    monte_carlo_var = pfq.monte_carlo_var(returns, confidence_level=0.95, n_simulations=10000)
    
    print(f"Historical VaR (95%): {historical_var:.6f}")
    print(f"Parametric VaR (95%): {parametric_var:.6f}")
    print(f"Monte Carlo VaR (95%): {monte_carlo_var:.6f}")

def test_cvar():
    print("\n=== Testing Conditional Value at Risk (CVaR) Calculations ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Calculate CVaR
    cvar = pfq.conditional_var(returns, confidence_level=0.95)
    expected_shortfall = pfq.expected_shortfall(returns, confidence_level=0.95)
    
    print(f"CVaR (95%): {cvar:.6f}")
    print(f"Expected Shortfall (95%): {expected_shortfall:.6f}")

def test_drawdown():
    print("\n=== Testing Drawdown Calculations ===")
    
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.random.normal(100, 5, 100).cumsum() + 1000
    
    # Calculate drawdown metrics
    drawdown_series = pfq.drawdown(prices)
    max_dd = pfq.max_drawdown(prices)
    dd_duration = pfq.drawdown_duration(prices)
    recovery_time = pfq.recovery_time(prices)
    
    print(f"Maximum Drawdown: {max_dd:.6f}")
    print(f"Drawdown Duration: {dd_duration}")
    print(f"Recovery Time: {recovery_time} days")
    
    # Print first 5 drawdown values
    print("\nFirst 5 Drawdown Values:")
    print(drawdown_series[:5])

def test_risk_metrics():
    print("\n=== Testing Additional Risk Metrics ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Calculate various risk metrics
    volatility = pfq.volatility(returns)
    semi_deviation = pfq.semi_deviation(returns)
    downside_risk = pfq.downside_risk(returns, target_return=0.0)
    
    print(f"Volatility: {volatility:.6f}")
    print(f"Semi-Deviation: {semi_deviation:.6f}")
    print(f"Downside Risk: {downside_risk:.6f}")

def main():
    print("Running manual tests for risk module...")
    test_var()
    test_cvar()
    test_drawdown()
    test_risk_metrics()

if __name__ == "__main__":
    main() 