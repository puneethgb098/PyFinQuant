# PyFinQuant (In Progress)

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/UV-0.1.0-FFD43B?logo=python&logoColor=black)](https://github.com/astral-sh/uv)
[![pytest](https://img.shields.io/badge/pytest-7.4.0-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/en/stable/)
[![Ruff](https://img.shields.io/badge/Ruff-0.1.0-FF4B4B?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Sphinx](https://img.shields.io/badge/Sphinx-4.0.0-1A1A1A?logo=sphinx&logoColor=white)](https://www.sphinx-doc.org/)
[![Mypy](https://img.shields.io/badge/Mypy-0.910-1A1A1A?logo=python&logoColor=white)](https://mypy-lang.org/)
[![Codecov](https://img.shields.io/badge/Codecov-F01F7A?logo=codecov&logoColor=white)](https://codecov.io/)

A modern Python boilerplate for quantitative finance projects, providing a robust foundation for building financial analysis, backtesting, and portfolio optimization tools.
// ... existing code ...

A comprehensive Python library for quantitative finance, providing tools for financial analysis, backtesting, and portfolio optimization.

## Features

- **Backtesting Engine**: Test trading strategies with historical data
- **Portfolio Optimization**: Implement modern portfolio theory and risk management
- **Options Pricing**: Calculate option prices and Greeks using various models
- **Data Fetching**: Access financial data from various sources
- **Risk Management**: Implement risk metrics and management tools
- **Visualization**: Create insightful financial charts and graphs

## Installation

```bash
pip install pyfinquant
```

## Quick Start

```python
from pyfinquant import Backtest, Strategy
from pyfinquant.data_fetcher import YahooDataFetcher

# Fetch data
data = YahooDataFetcher.fetch_data(ticker="AAPL", period="1y")

# Create and run backtest
strategy = Strategy()  # Your custom strategy
backtest = Backtest(strategy=strategy, data=data)
results = backtest.run()
```

## Documentation

For detailed documentation, visit [Read the Docs](https://pyfinquant.readthedocs.io/).

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## Development Workflows

This project uses GitHub Actions for continuous integration and deployment. The workflows are organized as follows:

### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)

The main CI/CD pipeline handles:
- Testing across Python versions (3.8, 3.9, 3.10, 3.11)
- Coverage reporting
- Package building and deployment to PyPI

### 2. CodeQL Analysis (`.github/workflows/codeql.yml`)

Security analysis workflow that:
- Runs on push and pull requests
- Performs static code analysis
- Checks for security vulnerabilities
- Runs weekly scheduled scans

### 3. Pull Request Validation (`.github/workflows/pr.yml`)

Quality checks for pull requests:
- Linting with Ruff
- Type checking with mypy
- Security scanning with bandit
- Code formatting checks

### Workflow Triggers

- **Push to main**: Triggers full CI/CD pipeline
- **Pull Requests**: Triggers PR validation and CodeQL analysis
- **Weekly**: Scheduled security scans

### Required Secrets

The workflows require the following secrets to be set in the repository:
- `PYPI_API_TOKEN`: For publishing to PyPI
- `CODECOV_TOKEN`: For coverage reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

Puneeth G B - puneethgb30@gmail.com

[LinkedIn](https://www.linkedin.com/in/puneeth-g-b-463aa91a0/)
