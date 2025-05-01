"""
Example demonstrating portfolio optimization and risk analysis using PyFinQuant.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyfinquant.portfolio.optimization import PortfolioOptimizer
from pyfinquant.risk.metrics import ValueAtRisk, ExpectedShortfall, MaxDrawdown

# Generate sample data
np.random.seed(42)
n_days = 252  # One trading year
n_assets = 4

# Simulate returns for 4 assets with different characteristics
returns_data = pd.DataFrame({
    'Tech Stock': np.random.normal(0.0012, 0.02, n_days),
    'Blue Chip': np.random.normal(0.0007, 0.01, n_days),
    'Government Bond': np.random.normal(0.0002, 0.005, n_days),
    'Small Cap': np.random.normal(0.0015, 0.025, n_days)
})

# Initialize portfolio optimizer
optimizer = PortfolioOptimizer(
    returns=returns_data.values,
    risk_free_rate=0.02/252  # Daily risk-free rate (2% annual)
)

# Find minimum volatility portfolio
min_vol_weights, min_vol_results = optimizer.minimum_volatility()
print("\nMinimum Volatility Portfolio:")
print("Weights:", dict(zip(returns_data.columns, min_vol_weights)))
print(f"Expected Return (annual): {min_vol_results['portfolio_return']*252:.2%}")
print(f"Volatility (annual): {min_vol_results['portfolio_volatility']*np.sqrt(252):.2%}")

# Find maximum Sharpe ratio portfolio
max_sharpe_weights, max_sharpe_results = optimizer.maximum_sharpe()
print("\nMaximum Sharpe Ratio Portfolio:")
print("Weights:", dict(zip(returns_data.columns, max_sharpe_weights)))
print(f"Expected Return (annual): {max_sharpe_results['portfolio_return']*252:.2%}")
print(f"Volatility (annual): {max_sharpe_results['portfolio_volatility']*np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {max_sharpe_results['sharpe_ratio']:.2f}")

# Calculate efficient frontier
returns, volatilities = optimizer.efficient_frontier(n_points=100)

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(volatilities*np.sqrt(252), returns*252, 'b-', label='Efficient Frontier')
plt.scatter(min_vol_results['portfolio_volatility']*np.sqrt(252),
           min_vol_results['portfolio_return']*252,
           color='g', marker='*', s=200, label='Minimum Volatility')
plt.scatter(max_sharpe_results['portfolio_volatility']*np.sqrt(252),
           max_sharpe_results['portfolio_return']*252,
           color='r', marker='*', s=200, label='Maximum Sharpe Ratio')
plt.xlabel('Annual Volatility')
plt.ylabel('Annual Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)

# Risk Analysis
# Calculate portfolio returns using maximum Sharpe ratio weights
portfolio_returns = np.sum(returns_data * max_sharpe_weights, axis=1)

# Value at Risk
var_calc = ValueAtRisk(portfolio_returns)
historical_var = var_calc.historical(confidence_level=0.95)
parametric_var = var_calc.parametric(confidence_level=0.95)

print("\nRisk Metrics for Maximum Sharpe Portfolio:")
print(f"95% Historical VaR (daily): {historical_var:.2%}")
print(f"95% Parametric VaR (daily): {parametric_var:.2%}")

# Expected Shortfall
es_calc = ExpectedShortfall(portfolio_returns)
historical_es = es_calc.historical(confidence_level=0.95)
print(f"95% Historical Expected Shortfall (daily): {historical_es:.2%}")

# Maximum Drawdown
portfolio_value = (1 + portfolio_returns).cumprod()
max_dd, peak_idx, trough_idx = MaxDrawdown.from_prices(portfolio_value)
print(f"Maximum Drawdown: {max_dd:.2%}")
print(f"Maximum Drawdown Period: {trough_idx - peak_idx} days")

# Plot portfolio value with drawdown
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='Portfolio Value')
plt.plot([peak_idx, trough_idx], [portfolio_value[peak_idx], portfolio_value[trough_idx]],
         'r--', label='Max Drawdown')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Performance and Maximum Drawdown')
plt.legend()
plt.grid(True)
plt.show() 