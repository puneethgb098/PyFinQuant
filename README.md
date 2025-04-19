# PyFinQuant (In Progress)

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for their invaluable tools and libraries

## 📧 Contact

Puneeth G B - puneethgb30@gmail.com

[LinkedIn](https://www.linkedin.com/in/puneeth-g-b-463aa91a0/)
