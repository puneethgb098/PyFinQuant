# PyFinQuant

A Python library for quantitative finance, providing tools for option pricing, risk analysis, and portfolio management.

## Features

- Option pricing using the Black-Scholes model
- Interest rate modeling using the Hull-White model
- Greeks calculations (Delta, Gamma, Vega, Theta, Rho)
- Risk metrics (VaR, Expected Shortfall, Maximum Drawdown)
- Portfolio analysis tools (Returns, Sharpe/Sortino Ratios, etc.)
- Support for dividend-paying assets

## Installation

```bash
# For users
pip install pyfinquant

# For developers (editable install with dev tools)
pip install -e ".[dev]"
```

## Quick Start

```python
import pyfinquant as pfq
import numpy as np

# Generate sample price data
prices = np.cumsum(np.random.normal(0, 1, 100)) + 100

# Calculate returns and risk metrics
simple_returns = pfq.returns(prices)
var_95 = pfq.historical_var(simple_returns, confidence_level=0.95) # Note: Pass returns to VaR
max_dd = pfq.max_drawdown(simple_returns)
sharpe = pfq.sharpe_ratio(simple_returns)

# Price an option
model_bs = pfq.BlackScholes(
    S=100,  # Current price
    K=100,  # Strike price
    T=1.0,  # Time to maturity in years
    r=0.05,  # Risk-free rate
    sigma=0.2,  # Volatility
    option_type='call',
    q=0.02  # Dividend yield
)
price_bs = model_bs.price()
greeks = pfq.AnalyticalGreeks(model_bs)
delta = greeks.delta()

# Price a zero-coupon bond using Hull-White
# (Requires defining initial term structure functions)
def flat_fwd(t): return 0.03
def flat_bond(t): return np.exp(-0.03 * t)

model_hw = pfq.HullWhite(
    a=0.1, 
    sigma=0.01, 
    initial_fwd_rate_func=flat_fwd, 
    initial_bond_price_func=flat_bond
)
price_zcb = model_hw.zero_coupon_bond_price(t=0.5, T=1.0, r_t=0.025) # Price at t=0.5 if r(0.5)=0.025

print(f"Option Price: {price_bs:.4f}, Delta: {delta:.4f}")
print(f"ZCB Price: {price_zcb:.4f}")
```

## Documentation

For detailed documentation and examples, please visit our [documentation](docs/).

## Contributing

Contributions are welcome! Please review the contribution guidelines [CONTRIBUTING.md](CONTRIBUTING.md) and feel free to submit a Pull Request.

## Development Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/puneethgb098/PyFinQuant.git # Replace with your repo URL if different
    cd PyFinQuant
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    # Example using venv
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```
3.  Install the package in editable mode with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
4.  Run tests:
    ```bash
    pytest tests/
    # Or using the python launcher:
    # py -m pytest tests/
    ```
5.  Check code style and formatting:
    ```bash
    black src/ tests/
    isort src/ tests/
    flake8 src/
    mypy src/
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
