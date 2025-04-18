"""
Example script demonstrating how to use the PyFinQuant backtesting framework with Yahoo Finance data.
"""

from pyfinquant.strategies.moving_average import MovingAverageCrossover
from pyfinquant.backtest.backtest import Backtest
import matplotlib.pyplot as plt


def main():
    # Initialize the moving average crossover strategy
    strategy = MovingAverageCrossover(
        ticker="AAPL",  # Apple Inc.
        short_window=20,
        long_window=50,
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    # Get ticker information
    ticker_info = strategy.get_ticker_info()
    print("\nTicker Information:")
    print(f"Name: {ticker_info['name']}")
    print(f"Sector: {ticker_info['sector']}")
    print(f"Industry: {ticker_info['industry']}")
    print(f"Market Cap: ${ticker_info['market_cap']:,.2f}")
    print(f"Currency: {ticker_info['currency']}\n")
    
    # Initialize the backtest
    backtest = Backtest(
        strategy=strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        position_sizing='risk_based',
        risk_per_trade=0.02,
        max_position_size=0.2,
        stop_loss=0.05,
        take_profit=0.1
    )
    
    # Run the backtest
    results = backtest.run()
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot equity curve
    results['equity_curve'].plot(ax=ax1, label='Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Equity Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Plot positions
    results['positions'].plot(ax=ax2, label='Positions')
    ax2.set_ylabel('Position Size')
    ax2.set_title('Trading Positions')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 