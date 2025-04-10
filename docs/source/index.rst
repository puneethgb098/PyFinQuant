Welcome to PyFinQuant's documentation!
====================================

PyFinQuant is a comprehensive Python library for financial quantitative analysis, focusing on options pricing, risk management, and portfolio optimization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/index
   api/index

Features
--------

* Core Functionality
  * Statistical analysis (mean, std, skew, kurtosis)
  * Time series analysis (returns, log returns, cumulative returns)
  * Portfolio optimization

* Risk Management
  * Value at Risk (VaR) calculations
  * Drawdown analysis

* Options Pricing
  * Black-Scholes model
  * Greeks calculations

Installation
-----------

.. code-block:: bash

   pip install pyfinquant

Quick Start
----------

.. code-block:: python

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 