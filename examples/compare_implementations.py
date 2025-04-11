import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from pyfinquant.core.time_series import returns, log_returns
    from pyfinquant.risk.value_at_risk import historical_var, parametric_var, monte_carlo_var
    from pyfinquant.risk.drawdown import drawdown, max_drawdown, drawdown_duration, recovery_time
    print("Successfully imported pyfinquant modules")
except ImportError as e:
    print(f"Error importing pyfinquant modules: {e}")
    raise

def generate_price_series(n_points=100, start_price=1000, volatility=0.02, seed=42):
    """Generate a realistic price series with both up and down movements."""
    np.random.seed(seed)
    # Generate returns with a slight downward bias to ensure drawdowns
    returns = np.random.normal(-0.0001, volatility, n_points)
    # Convert returns to price movements
    price_factors = np.exp(returns)
    # Calculate prices
    prices = start_price * np.cumprod(price_factors)
    return prices

def compare_returns():
    print("\n=== Comparing Returns Calculations ===")
    
    try:
        # Generate sample price data
        prices = pd.Series(generate_price_series())
        
        # Calculate returns using both implementations
        simple_returns_result = returns(prices)
        log_returns_result = log_returns(prices)
        
        # Print first 5 values from each
        print("\nSimple Returns (first 5):")
        print(simple_returns_result[:5])
        print("\nLog Returns (first 5):")
        print(log_returns_result[:5])
        
        # Print summary statistics
        print("\nSimple Returns Statistics:")
        print(f"Mean: {simple_returns_result.mean():.6f}")
        print(f"Std Dev: {simple_returns_result.std():.6f}")
        
        print("\nLog Returns Statistics:")
        print(f"Mean: {log_returns_result.mean():.6f}")
        print(f"Std Dev: {log_returns_result.std():.6f}")
    except Exception as e:
        print(f"Error in compare_returns: {e}")
        raise

def compare_var():
    print("\n=== Comparing Value at Risk Calculations ===")
    
    try:
        # Generate sample returns
        np.random.seed(42)
        returns_data = pd.Series(np.random.normal(-0.0001, 0.02, 1000))
        
        # Calculate different types of VaR
        hist_var = historical_var(returns_data, confidence_level=0.95)
        param_var = parametric_var(returns_data, confidence_level=0.95)
        mc_var = monte_carlo_var(returns_data, confidence_level=0.95, n_simulations=10000)
        
        print("\nValue at Risk (95%):")
        print(f"Historical VaR: {hist_var:.6f}")
        print(f"Parametric VaR: {param_var:.6f}")
        print(f"Monte Carlo VaR: {mc_var:.6f}")
    except Exception as e:
        print(f"Error in compare_var: {e}")
        raise

def compare_drawdown():
    print("\n=== Comparing Drawdown Calculations ===")
    
    try:
        # Generate sample price data with drawdowns
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = pd.Series(generate_price_series(), index=dates)
        
        # Calculate drawdown metrics
        drawdown_series = drawdown(prices)
        max_dd = max_drawdown(prices)
        dd_duration = drawdown_duration(prices)
        recovery = recovery_time(prices)
        
        print("\nDrawdown Metrics:")
        print(f"Maximum Drawdown: {max_dd:.6f}")
        print(f"Drawdown Duration: {dd_duration}")
        print(f"Recovery Time: {recovery} days")
        
        # Print first 5 drawdown values
        print("\nFirst 5 Drawdown Values:")
        print(drawdown_series[:5])
        
        # Print some price statistics
        print("\nPrice Series Statistics:")
        print(f"Starting Price: {prices[0]:.2f}")
        print(f"Ending Price: {prices[-1]:.2f}")
        print(f"Min Price: {prices.min():.2f}")
        print(f"Max Price: {prices.max():.2f}")
    except Exception as e:
        print(f"Error in compare_drawdown: {e}")
        raise

def main():
    print("Running comparison tests between implementations...")
    try:
        compare_returns()
        compare_var()
        compare_drawdown()
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 