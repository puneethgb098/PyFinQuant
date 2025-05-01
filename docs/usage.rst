Usage
=====

Basic Usage
----------

Here's a quick example of how to use PyFinQuant:

.. code-block:: python

    from pyfinquant import PyFinQuant

    # Initialize with API key
    pfq = PyFinQuant(api_key='your_api_key')

    # Get stock data
    data = pfq.get_stock_data('AAPL', start_date='2020-01-01', end_date='2020-12-31')

    # Calculate returns
    returns = pfq.calculate_returns(data)

    # Calculate risk metrics
    risk_metrics = pfq.calculate_risk_metrics(returns)

    # Optimize portfolio
    portfolio = pfq.optimize_portfolio(returns)

Features
--------

PyFinQuant provides the following main features:

- Data Retrieval
  - Stock data
  - Market indices
  - Economic indicators
  - Cryptocurrency data

- Analysis Tools
  - Returns calculation
  - Risk metrics
  - Portfolio optimization
  - Technical indicators

- Visualization
  - Price charts
  - Returns distribution
  - Risk metrics plots
  - Portfolio performance

For more detailed examples and API documentation, see the :doc:`api` section.

Stock Analysis
-------------

PyFinQuant provides various tools for stock analysis:

.. code-block:: python

    from pyfinquant import Stock

    # Create a stock object
    stock = Stock("AAPL")

    # Get historical data
    data = stock.get_data()

    # Calculate technical indicators
    sma = stock.calculate_sma(period=20)
    rsi = stock.calculate_rsi(period=14)
    macd = stock.calculate_macd()

    # Calculate fundamental metrics
    pe_ratio = stock.get_pe_ratio()
    market_cap = stock.get_market_cap()
    dividend_yield = stock.get_dividend_yield()

Portfolio Management
------------------

PyFinQuant supports portfolio management features:

.. code-block:: python

    from pyfinquant import Portfolio

    # Create a portfolio
    portfolio = Portfolio(
        symbols=["AAPL", "MSFT", "GOOGL"],
        weights=[0.4, 0.3, 0.3]
    )

    # Calculate portfolio metrics
    returns = portfolio.calculate_returns()
    volatility = portfolio.calculate_volatility()
    sharpe_ratio = portfolio.calculate_sharpe_ratio()
    beta = portfolio.calculate_beta()

    # Optimize portfolio
    optimized_weights = portfolio.optimize_portfolio()

Market Data
----------

PyFinQuant provides tools for market data analysis:

.. code-block:: python

    from pyfinquant import MarketData

    # Create market data object
    market = MarketData()

    # Get market indices
    sp500 = market.get_index("^GSPC")
    nasdaq = market.get_index("^IXIC")

    # Get sector performance
    sectors = market.get_sector_performance()

    # Get market news
    news = market.get_market_news() 