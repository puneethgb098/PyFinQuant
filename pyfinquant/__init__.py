"""
PyFinQuant - A Python library for quantitative finance

This package provides tools for option pricing, risk analysis, portfolio management,
and other quantitative finance applications.
"""

from importlib.metadata import version

__version__ = version("pyfinquant")

# Core functionality
from .models.black_scholes import BlackScholes
from .instruments.option import Option, OptionType
from .greeks.analytical import AnalyticalGreeks

# Risk metrics
from .risk.metrics import ValueAtRisk, ExpectedShortfall, MaxDrawdown
from .risk.volatility import HistoricalVolatility, ImpliedVolatility

# Portfolio management
from .portfolio.optimization import PortfolioOptimizer
from .portfolio.metrics import SharpeRatio, SortinoRatio, InformationRatio

# Visualization
from .visualization.charts import (
    plot_price,
    plot_returns_distribution,
    plot_risk_metrics,
    plot_portfolio_performance
)

# Strategies
from .strategies.base import Strategy as BaseStrategy
from .strategies.moving_average import MovingAverageCrossover
from .strategies.mean_reversion import MeanReversion
from .strategies.momentum import MomentumStrategy

# Backtesting
from .backtest.backtest import Backtest

# Utilities
from .utils.helpers import (
    calculate_returns,
    annualize_returns,
    annualize_volatility,
)

__all__ = [
    # Models
    "BlackScholes",
    
    # Instruments
    "Option",
    "OptionType",
    
    # Greeks
    "AnalyticalGreeks",
    
    # Risk metrics
    "ValueAtRisk",
    "ExpectedShortfall",
    "MaxDrawdown",
    "HistoricalVolatility",
    "ImpliedVolatility",
    
    # Portfolio management
    "PortfolioOptimizer",
    "SharpeRatio",
    "SortinoRatio",
    "InformationRatio",
    
    # Visualization
    "plot_price",
    "plot_returns_distribution",
    "plot_risk_metrics",
    "plot_portfolio_performance",
    
    # Strategies
    "BaseStrategy",
    "MovingAverageCrossover",
    "MeanReversion",
    "MomentumStrategy",
    
    # Backtesting
    "Backtest",
    
    # Utilities
    "calculate_returns",
    "annualize_returns",
    "annualize_volatility",
]
