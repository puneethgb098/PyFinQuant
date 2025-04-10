"""Risk metrics module for PyFinQuant."""

from typing import Union, Optional
import numpy as np
from ..utils.types import Numeric
from .value_at_risk import historical_var, parametric_var, monte_carlo_var, conditional_var
from .drawdown import drawdown, max_drawdown, drawdown_duration, recovery_time

def calculate_risk_metrics(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = "historical"
) -> dict:
    """
    Calculate various risk metrics for a given return series.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    confidence_level : float, optional
        Confidence level for VaR calculation, by default 0.95
    time_horizon : int, optional
        Time horizon for risk calculation, by default 1
    method : str, optional
        Method for VaR calculation ('historical', 'parametric', 'monte_carlo'), by default "historical"
    
    Returns
    -------
    dict
        Dictionary containing various risk metrics
    """
    metrics = {}
    
    # Calculate VaR based on selected method
    if method == "historical":
        metrics["var"] = historical_var(returns, confidence_level)
    elif method == "parametric":
        metrics["var"] = parametric_var(returns, confidence_level)
    elif method == "monte_carlo":
        metrics["var"] = monte_carlo_var(returns, confidence_level)
    else:
        raise ValueError("Invalid method. Choose from 'historical', 'parametric', or 'monte_carlo'")
    
    # Calculate CVaR
    metrics["cvar"] = conditional_var(returns, confidence_level)
    
    # Calculate drawdown metrics
    metrics["max_drawdown"] = max_drawdown(returns)
    start_idx, end_idx = drawdown_duration(returns)
    metrics["drawdown_duration"] = end_idx - start_idx if start_idx is not None and end_idx is not None else 0
    metrics["recovery_time"] = recovery_time(returns)
    
    # Calculate basic statistics
    metrics["volatility"] = np.std(returns) * np.sqrt(252)  # Annualized
    metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    return metrics

def calculate_portfolio_risk(
    returns: np.ndarray,
    weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95
) -> dict:
    """
    Calculate portfolio risk metrics.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (n_assets x n_periods)
    weights : Optional[np.ndarray], optional
        Portfolio weights, by default None (equal weights)
    confidence_level : float, optional
        Confidence level for VaR calculation, by default 0.95
    
    Returns
    -------
    dict
        Dictionary containing portfolio risk metrics
    """
    if weights is None:
        weights = np.ones(returns.shape[0]) / returns.shape[0]
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(weights, returns)
    
    # Calculate risk metrics
    metrics = calculate_risk_metrics(
        portfolio_returns,
        confidence_level=confidence_level
    )
    
    # Add portfolio-specific metrics
    metrics["portfolio_volatility"] = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns), weights))) * np.sqrt(252)
    
    return metrics 