"""
Value at Risk (VaR) calculations for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.stats import norm

def historical_var(returns: Union[np.ndarray, pd.Series],
                  confidence_level: float = 0.95) -> float:
    """
    Calculate the historical Value at Risk.
    
    Args:
        returns: The return series
        confidence_level: The confidence level (default: 0.95)
        
    Returns:
        The Value at Risk
    """
    return np.percentile(returns, (1 - confidence_level) * 100)

def parametric_var(returns: Union[np.ndarray, pd.Series],
                  confidence_level: float = 0.95,
                  mean: Optional[float] = None,
                  std: Optional[float] = None) -> float:
    """
    Calculate the parametric (Gaussian) Value at Risk.
    
    Args:
        returns: The return series
        confidence_level: The confidence level (default: 0.95)
        mean: The mean return (if None, calculated from returns)
        std: The standard deviation (if None, calculated from returns)
        
    Returns:
        The Value at Risk
    """
    if mean is None:
        mean = np.mean(returns)
    if std is None:
        std = np.std(returns, ddof=1)
    
    z_score = norm.ppf(1 - confidence_level)
    return mean + z_score * std

def monte_carlo_var(returns: Union[np.ndarray, pd.Series],
                   confidence_level: float = 0.95,
                   n_simulations: int = 10000,
                   seed: Optional[int] = None) -> float:
    """
    Calculate the Monte Carlo Value at Risk.
    
    Args:
        returns: The return series
        confidence_level: The confidence level (default: 0.95)
        n_simulations: Number of Monte Carlo simulations (default: 10000)
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        The Value at Risk
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    simulated_returns = np.random.normal(mean, std, n_simulations)
    return np.percentile(simulated_returns, (1 - confidence_level) * 100)

def conditional_var(returns: Union[np.ndarray, pd.Series],
                   confidence_level: float = 0.95) -> float:
    """
    Calculate the Conditional Value at Risk (Expected Shortfall).
    
    Args:
        returns: The return series
        confidence_level: The confidence level (default: 0.95)
        
    Returns:
        The Conditional Value at Risk
    """
    var = historical_var(returns, confidence_level)
    return np.mean(returns[returns <= var])

def expected_shortfall(returns: Union[np.ndarray, pd.Series],
                      confidence_level: float = 0.95) -> float:
    """
    Calculate the Expected Shortfall (same as Conditional VaR).
    
    Args:
        returns: The return series
        confidence_level: The confidence level (default: 0.95)
        
    Returns:
        The Expected Shortfall
    """
    return conditional_var(returns, confidence_level) 