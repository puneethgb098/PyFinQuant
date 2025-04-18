# PyFinQuant

A Python library for quantitative finance, providing tools for backtesting trading strategies, analyzing financial data, and managing portfolios.

## Features

- Backtesting engine for trading strategies
- Integration with Yahoo Finance for real-time data
- Moving Average Crossover strategy implementation
- Risk management tools (stop loss, take profit, trailing stop)
- Position sizing methods (fixed, risk-based, volatility-based)
- Performance metrics (returns, Sharpe ratio, drawdown, etc.)

## Installation

```bash
pip install pyfinquant
```

## Quick Start

Here's a simple example of how to use PyFinQuant to backtest a Moving Average Crossover strategy using Yahoo Finance data:

```python
from pyfinquant.strategies.moving_average import MovingAverageCrossover
from pyfinquant.backtest.backtest import Backtest

# Initialize the strategy with Yahoo Finance data
strategy = MovingAverageCrossover(
    ticker="AAPL",  # Apple Inc.
    short_window=20,
    long_window=50,
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Get ticker information
ticker_info = strategy.get_ticker_info()
print(f"Ticker: {ticker_info['name']}")
print(f"Sector: {ticker_info['sector']}")

# Initialize and run backtest
backtest = Backtest(
    strategy=strategy,
    initial_capital=100000,
    commission=0.001,
    slippage=0.0001,
    position_sizing='risk_based',
    risk_per_trade=0.02
)

results = backtest.run()

# Print results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Yahoo Finance Integration

PyFinQuant integrates with Yahoo Finance to fetch real-time market data. You can use any valid Yahoo Finance ticker symbol to backtest your strategies. The library handles:

- Fetching historical price data
- Getting ticker information (name, sector, industry, etc.)
- Data preprocessing and validation

### Available Tickers

You can use any valid Yahoo Finance ticker symbol. Some examples:
- Stocks: AAPL, MSFT, GOOGL, AMZN
- ETFs: SPY, QQQ, IWM
- Cryptocurrencies: BTC-USD, ETH-USD
- Forex: EURUSD=X, JPY=X

## Backtesting Features

### Position Sizing Methods

1. Fixed: Uses a constant fraction of capital for each trade
2. Risk-based: Adjusts position size based on risk per trade
3. Volatility-based: Adjusts position size based on asset volatility

### Risk Management

- Stop Loss: Automatically exits positions when price falls below a certain level
- Take Profit: Automatically exits positions when price rises above a certain level
- Trailing Stop: Adjusts stop loss level as price moves in favor of the position

### Performance Metrics

- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Number of Trades
- Win Rate

## Examples

Check out the `examples` directory for more detailed examples of how to use PyFinQuant:

- `backtest_example.py`: Basic backtesting example with Yahoo Finance data
- `multi_asset_backtest.py`: Example of backtesting multiple assets
- `custom_strategy.py`: Example of implementing a custom strategy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Puneeth G B - puneethgb30@gmail.com

[LinkedIn](https://www.linkedin.com/in/puneeth-g-b-463aa91a0/)
