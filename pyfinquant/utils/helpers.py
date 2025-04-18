"""
Internal utility functions for the PyFinQuant package.

These functions are generally intended for use by other modules within
this package and may not be part of the stable public API unless explicitly
exposed in pyfinquant.utils.__init__.py.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd

Numeric = Union[int, float]


def check_positive(value: Numeric, name: str = "Value") -> None:
    """
    Raises a ValueError if the provided numeric value is not strictly positive (> 0).

    Args:
        value: The numeric value to check (int or float).
        name: The name of the variable or parameter being checked, used in the error message.

    Raises:
        ValueError: If value is less than or equal to zero.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, but got {value}")


def check_non_negative(value: Numeric, name: str = "Value") -> None:
    """
    Raises a ValueError if the provided numeric value is negative (< 0).

    Args:
        value: The numeric value to check (int or float).
        name: The name of the variable or parameter being checked, used in the error message.

    Raises:
        ValueError: If value is less than zero.
    """
    if value < 0:
        raise ValueError(f"{name} cannot be negative, but got {value}")


def calculate_returns(
    prices: Union[pd.Series, np.ndarray],
    method: str = 'arithmetic'
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate returns from a series of prices.

    Args:
        prices: Series of prices.
        method: Method to calculate returns ('arithmetic' or 'log').

    Returns:
        Series of returns.
    """
    if method not in ['arithmetic', 'log']:
        raise ValueError("method must be either 'arithmetic' or 'log'")

    if isinstance(prices, pd.Series):
        if method == 'arithmetic':
            return prices.pct_change()
        else:
            return np.log(prices / prices.shift(1))
    else:
        if method == 'arithmetic':
            return np.diff(prices) / prices[:-1]
        else:
            return np.log(prices[1:] / prices[:-1])


def annualize_returns(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Annualize a series of returns.

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.).

    Returns:
        Annualized return.
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    else:
        returns = returns[~np.isnan(returns)]

    return np.mean(returns) * periods_per_year


def annualize_volatility(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Annualize the volatility of a series of returns.

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.).

    Returns:
        Annualized volatility.
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    else:
        returns = returns[~np.isnan(returns)]

    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)
