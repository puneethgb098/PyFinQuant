# PyFinQuant

A Python library for financial quantitative analysis, focusing on options pricing and greeks.

## Features

- Options pricing using Black-Scholes model
- Greeks calculations (Delta, Gamma, Vega, Theta, Rho)
- Support for various option types (European, American)
- Utility functions for financial calculations
- Type hints and comprehensive documentation

## Installation

```bash
pip install pyfinquant
```

## Quick Start

```python
from pyfinquant import Option, OptionType, BlackScholes

# Create an option
option = Option(
    strike_price=100,
    time_to_expiry=1.0,
    option_type=OptionType.CALL
)

# Calculate price using Black-Scholes
model = BlackScholes(spot_price=100, volatility=0.2, risk_free_rate=0.05)
price = model.price(option)

# Calculate Greeks
greeks = model.greeks(option)
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/your-username/PyFinQuant.git
cd PyFinQuant

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
