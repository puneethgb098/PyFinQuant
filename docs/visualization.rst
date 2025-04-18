Visualization
============

PyFinQuant provides comprehensive visualization tools for financial data analysis. Here are some examples:

Price Charts
-----------

.. code-block:: python

    from pyfinquant import Stock
    import matplotlib.pyplot as plt

    # Create a stock object
    stock = Stock("AAPL")

    # Get historical data
    data = stock.get_data()

    # Plot price chart
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'])
    plt.title('AAPL Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.show()

Technical Indicators
------------------

.. code-block:: python

    from pyfinquant import Stock
    import matplotlib.pyplot as plt

    # Create a stock object
    stock = Stock("AAPL")

    # Calculate and plot moving averages
    sma_20 = stock.calculate_sma(period=20)
    sma_50 = stock.calculate_sma(period=50)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Price')
    plt.plot(data.index, sma_20, label='20-day SMA')
    plt.plot(data.index, sma_50, label='50-day SMA')
    plt.title('AAPL Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

Returns Distribution
------------------

.. code-block:: python

    from pyfinquant import Stock
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a stock object
    stock = Stock("AAPL")

    # Calculate returns
    returns = stock.calculate_returns()

    # Plot returns distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, kde=True)
    plt.title('AAPL Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

Portfolio Performance
-------------------

.. code-block:: python

    from pyfinquant import Portfolio
    import matplotlib.pyplot as plt

    # Create a portfolio
    portfolio = Portfolio(
        symbols=["AAPL", "MSFT", "GOOGL"],
        weights=[0.4, 0.3, 0.3]
    )

    # Calculate portfolio returns
    returns = portfolio.calculate_returns()

    # Plot cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns)
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

Risk Metrics Visualization
------------------------

.. code-block:: python

    from pyfinquant import Portfolio
    import matplotlib.pyplot as plt

    # Create a portfolio
    portfolio = Portfolio(
        symbols=["AAPL", "MSFT", "GOOGL"],
        weights=[0.4, 0.3, 0.3]
    )

    # Calculate risk metrics
    metrics = portfolio.calculate_risk_metrics()

    # Plot risk metrics
    plt.figure(figsize=(10, 6))
    metrics.plot(kind='bar')
    plt.title('Portfolio Risk Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()