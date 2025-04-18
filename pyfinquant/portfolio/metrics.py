"""
Portfolio performance metrics module.

This module provides classes for calculating various portfolio performance metrics
including Sharpe ratio, Sortino ratio, and Information ratio.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from ..utils.helpers import calculate_returns


class SharpeRatio:
    """
    Calculate the Sharpe ratio for a portfolio.
    """

    def __init__(self, returns: Union[pd.Series, pd.DataFrame], risk_free_rate: float = 0.0):
        """
        Initialize the Sharpe ratio calculator.

        Args:
            returns: Portfolio returns as pandas Series or DataFrame
            risk_free_rate: Annual risk-free rate (default: 0.0)
        """
        self.returns = returns
        self.rf_rate = risk_free_rate / 252  # Convert annual rate to daily

    def calculate(self, annualize: bool = True) -> Union[float, pd.Series]:
        """
        Calculate the Sharpe ratio.

        Args:
            annualize: Whether to annualize the ratio (default: True)

        Returns:
            Sharpe ratio as float or pandas Series
        """
        excess_returns = self.returns - self.rf_rate
        if annualize:
            sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        else:
            sharpe = excess_returns.mean() / excess_returns.std()
        return sharpe


class SortinoRatio:
    """
    Calculate the Sortino ratio for a portfolio.
    """

    def __init__(self, returns: Union[pd.Series, pd.DataFrame], risk_free_rate: float = 0.0,
                 target_return: Optional[float] = None):
        """
        Initialize the Sortino ratio calculator.

        Args:
            returns: Portfolio returns as pandas Series or DataFrame
            risk_free_rate: Annual risk-free rate (default: 0.0)
            target_return: Minimum acceptable return (default: risk_free_rate)
        """
        self.returns = returns
        self.rf_rate = risk_free_rate / 252  # Convert annual rate to daily
        self.target_return = target_return if target_return is not None else self.rf_rate

    def calculate(self, annualize: bool = True) -> Union[float, pd.Series]:
        """
        Calculate the Sortino ratio.

        Args:
            annualize: Whether to annualize the ratio (default: True)

        Returns:
            Sortino ratio as float or pandas Series
        """
        excess_returns = self.returns - self.rf_rate
        downside_returns = excess_returns[excess_returns < self.target_return]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if annualize:
            sortino = np.sqrt(252) * (excess_returns.mean() / downside_std)
        else:
            sortino = excess_returns.mean() / downside_std
        return sortino


class InformationRatio:
    """
    Calculate the Information ratio for a portfolio.
    """

    def __init__(self, returns: Union[pd.Series, pd.DataFrame], benchmark_returns: Union[pd.Series, pd.DataFrame]):
        """
        Initialize the Information ratio calculator.

        Args:
            returns: Portfolio returns as pandas Series or DataFrame
            benchmark_returns: Benchmark returns as pandas Series or DataFrame
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns

    def calculate(self, annualize: bool = True) -> Union[float, pd.Series]:
        """
        Calculate the Information ratio.

        Args:
            annualize: Whether to annualize the ratio (default: True)

        Returns:
            Information ratio as float or pandas Series
        """
        active_returns = self.returns - self.benchmark_returns
        if annualize:
            info_ratio = np.sqrt(252) * (active_returns.mean() / active_returns.std())
        else:
            info_ratio = active_returns.mean() / active_returns.std()
        return info_ratio 