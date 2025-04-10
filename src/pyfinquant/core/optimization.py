"""
Portfolio optimization functions for quantitative finance.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimize_risk(
    returns: Union[np.ndarray, pd.DataFrame],
    target_return: Optional[float] = None,
    bounds: Optional[List[tuple]] = None,
) -> np.ndarray:
    """
    Minimize portfolio risk for a given target return.

    Args:
        returns: The return series for each asset
        target_return: The target portfolio return (if None, minimize risk without return constraint)
        bounds: The bounds for each weight (default: (0, 1) for each asset)

    Returns:
        The optimal portfolio weights
    """
    n_assets = returns.shape[1]
    cov_matrix = np.cov(returns, rowvar=False)

    if bounds is None:
        bounds = [(0, 1) for _ in range(n_assets)]

    def objective(weights):
        return weights.T @ cov_matrix @ weights

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda x: np.sum(x * returns.mean()) - target_return}
        )

    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def maximize_sharpe(
    returns: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.0,
    bounds: Optional[List[tuple]] = None,
) -> np.ndarray:
    """
    Maximize the Sharpe ratio of a portfolio.

    Args:
        returns: The return series for each asset
        risk_free_rate: The risk-free rate (default: 0.0)
        bounds: The bounds for each weight (default: (0, 1) for each asset)

    Returns:
        The optimal portfolio weights
    """
    n_assets = returns.shape[1]
    if bounds is None:
        bounds = [(0, 1) for _ in range(n_assets)]

    def objective(weights):
        portfolio_returns = np.sum(weights * returns, axis=1)
        excess_returns = portfolio_returns - risk_free_rate
        return -np.mean(excess_returns) / np.std(excess_returns)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def maximize_sortino(
    returns: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.0,
    bounds: Optional[List[tuple]] = None,
) -> np.ndarray:
    """
    Maximize the Sortino ratio of a portfolio.

    Args:
        returns: The return series for each asset
        risk_free_rate: The risk-free rate (default: 0.0)
        bounds: The bounds for each weight (default: (0, 1) for each asset)

    Returns:
        The optimal portfolio weights
    """
    n_assets = returns.shape[1]
    if bounds is None:
        bounds = [(0, 1) for _ in range(n_assets)]

    def objective(weights):
        portfolio_returns = np.sum(weights * returns, axis=1)
        excess_returns = portfolio_returns - risk_free_rate
        downside_std = np.std(excess_returns[excess_returns < 0])
        return -np.mean(excess_returns) / downside_std

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def minimize_tracking_error(
    returns: Union[np.ndarray, pd.DataFrame],
    benchmark_returns: Union[np.ndarray, pd.Series],
    bounds: Optional[List[tuple]] = None,
) -> np.ndarray:
    """
    Minimize the tracking error of a portfolio relative to a benchmark.

    Args:
        returns: The return series for each asset
        benchmark_returns: The benchmark return series
        bounds: The bounds for each weight (default: (0, 1) for each asset)

    Returns:
        The optimal portfolio weights
    """
    n_assets = returns.shape[1]
    if bounds is None:
        bounds = [(0, 1) for _ in range(n_assets)]

    def objective(weights):
        portfolio_returns = np.sum(weights * returns, axis=1)
        tracking_error = np.std(portfolio_returns - benchmark_returns)
        return tracking_error

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x
