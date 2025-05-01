Trading Strategies
================

PyFinQuant provides various trading strategies and backtesting tools. Here are some examples:

Moving Average Crossover
-----------------------

.. code-block:: python

    from pyfinquant import Strategy
    import pandas as pd

    class MovingAverageCrossover(Strategy):
        def __init__(self, short_window=20, long_window=50):
            self.short_window = short_window
            self.long_window = long_window

        def generate_signals(self, data):
            # Calculate moving averages
            data['short_mavg'] = data['Close'].rolling(window=self.short_window).mean()
            data['long_mavg'] = data['Close'].rolling(window=self.long_window).mean()

            # Generate signals
            data['signal'] = 0.0
            data['signal'][self.long_window:] = np.where(
                data['short_mavg'][self.long_window:] > data['long_mavg'][self.long_window:], 1.0, 0.0
            )

            # Generate trading orders
            data['positions'] = data['signal'].diff()
            return data

    # Usage example
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    signals = strategy.generate_signals(stock_data)

Mean Reversion
-------------

.. code-block:: python

    from pyfinquant import Strategy
    import numpy as np

    class MeanReversion(Strategy):
        def __init__(self, window=20, threshold=2.0):
            self.window = window
            self.threshold = threshold

        def generate_signals(self, data):
            # Calculate rolling mean and standard deviation
            data['mean'] = data['Close'].rolling(window=self.window).mean()
            data['std'] = data['Close'].rolling(window=self.window).std()
            
            # Calculate z-score
            data['zscore'] = (data['Close'] - data['mean']) / data['std']
            
            # Generate signals
            data['signal'] = 0.0
            data['signal'] = np.where(data['zscore'] > self.threshold, -1.0,
                                    np.where(data['zscore'] < -self.threshold, 1.0, 0.0))
            
            # Generate trading orders
            data['positions'] = data['signal'].diff()
            return data

    # Usage example
    strategy = MeanReversion(window=20, threshold=2.0)
    signals = strategy.generate_signals(stock_data)

Momentum Strategy
---------------

.. code-block:: python

    from pyfinquant import Strategy
    import numpy as np

    class MomentumStrategy(Strategy):
        def __init__(self, window=12, threshold=0.0):
            self.window = window
            self.threshold = threshold

        def generate_signals(self, data):
            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            
            # Calculate momentum
            data['momentum'] = data['returns'].rolling(window=self.window).mean()
            
            # Generate signals
            data['signal'] = 0.0
            data['signal'] = np.where(data['momentum'] > self.threshold, 1.0,
                                    np.where(data['momentum'] < -self.threshold, -1.0, 0.0))
            
            # Generate trading orders
            data['positions'] = data['signal'].diff()
            return data

    # Usage example
    strategy = MomentumStrategy(window=12, threshold=0.0)
    signals = strategy.generate_signals(stock_data)

Backtesting
----------

.. code-block:: python

    from pyfinquant import Backtest
    from pyfinquant import MovingAverageCrossover

    # Create strategy instance
    strategy = MovingAverageCrossover(short_window=20, long_window=50)

    # Create backtest instance
    backtest = Backtest(
        strategy=strategy,
        data=stock_data,
        initial_capital=100000.0,
        commission=0.001
    )

    # Run backtest
    results = backtest.run()

    # Get performance metrics
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

    # Plot results
    backtest.plot_results() 