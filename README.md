# PyFinQuant

A Python library for financial quantitative analysis, focusing on options pricing, risk management, and portfolio optimization.

## Features

- Option pricing and Greeks calculation
- Risk metrics (VaR, CVaR, drawdown analysis)
- Portfolio optimization
- Time series analysis
- Statistical tools

## Installation

1. Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)

2. Clone the repository:
```bash
git clone https://github.com/yourusername/pyfinquant.git
cd pyfinquant
```

3. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

## Development Setup

1. Install the package in development mode:
```bash
pip install -e .
```

2. Run tests:
```bash
pytest tests/ -v
```

3. Run linting:
```bash
flake8 src/
black src/
isort src/
mypy src/
```

## Usage

```python
import pyfinquant as pfq

# Calculate returns
returns = pfq.returns(prices)

# Calculate risk metrics
var = pfq.historical_var(returns)
mdd = pfq.max_drawdown(prices)

# Price an option
option = pfq.Option(
    S=100,  # Spot price
    K=100,  # Strike price
    T=1.0,  # Time to maturity
    r=0.05,  # Risk-free rate
    sigma=0.2,  # Volatility
    option_type=pfq.OptionType.CALL
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
