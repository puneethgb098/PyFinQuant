"""
Time series analysis functions for quantitative finance.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from ..risk.drawdown import drawdown, max_drawdown


def returns(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Calculate the simple returns of a series."""
    return pd.Series(x).pct_change()


def log_returns(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Calculate the log returns of a series."""
    return np.log(pd.Series(x)).diff()


def cumulative_returns(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Calculate the cumulative returns of a series."""
    simple_returns = returns(x).fillna(0)  # Calculate simple returns and fill NaN with 0
    return (1 + simple_returns).cumprod() - 1


def sharpe_ratio(
    x: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sharpe ratio of a series.

    Args:
        x: The return series
        risk_free_rate: The risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        The annualized Sharpe ratio
    """
    # Convert input to pandas Series if it's a numpy array
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
        
    excess_returns = x - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(
    x: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sortino ratio of a series.

    Args:
        x: The return series
        risk_free_rate: The risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        The annualized Sortino ratio
    """
    # Convert input to pandas Series if it's a numpy array
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
        
    excess_returns = x - risk_free_rate / periods_per_year
    downside_std = excess_returns[excess_returns < 0].std()
    # Handle case where downside deviation is zero
    if downside_std == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def information_ratio(
    x: Union[np.ndarray, pd.Series],
    benchmark: Union[np.ndarray, pd.Series],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the information ratio of a series relative to a benchmark.

    Args:
        x: The return series
        benchmark: The benchmark return series
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        The annualized information ratio
    """
    # Convert inputs to pandas Series if they are numpy arrays
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    if isinstance(benchmark, np.ndarray):
        benchmark = pd.Series(benchmark)
        
    excess_returns = x - benchmark
    tracking_error = excess_returns.std()
    # Handle case where tracking error is zero
    if tracking_error == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / tracking_error
