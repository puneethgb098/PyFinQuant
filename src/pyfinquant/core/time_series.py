"""
Time series analysis functions for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from ..risk.drawdown import drawdown, max_drawdown

def returns(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Calculate the simple returns of a series."""
    return pd.Series(x).pct_change()

def log_returns(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Calculate the log returns of a series."""
    return np.log(pd.Series(x)).diff()

def cumulative_returns(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Calculate the cumulative returns of a series."""
    return (1 + returns(x)).cumprod() - 1

def sharpe_ratio(x: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio of a series.
    
    Args:
        x: The return series
        risk_free_rate: The risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)
        
    Returns:
        The annualized Sharpe ratio
    """
    excess_returns = returns(x) - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def sortino_ratio(x: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino ratio of a series.
    
    Args:
        x: The return series
        risk_free_rate: The risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)
        
    Returns:
        The annualized Sortino ratio
    """
    excess_returns = returns(x) - risk_free_rate / periods_per_year
    downside_std = excess_returns[excess_returns < 0].std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

def information_ratio(x: Union[np.ndarray, pd.Series], benchmark: Union[np.ndarray, pd.Series], periods_per_year: int = 252) -> float:
    """
    Calculate the information ratio of a series relative to a benchmark.
    
    Args:
        x: The return series
        benchmark: The benchmark return series
        periods_per_year: Number of periods per year (default: 252 for daily data)
        
    Returns:
        The annualized information ratio
    """
    excess_returns = returns(x) - returns(benchmark)
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std() 