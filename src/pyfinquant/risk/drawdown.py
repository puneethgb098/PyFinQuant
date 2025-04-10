"""
Drawdown calculations for quantitative finance.
"""

from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd


def drawdown(returns: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Calculate the drawdown series.

    Args:
        returns: The return series

    Returns:
        The drawdown series
    """
    # Convert input to pandas Series if it's a numpy array and fill initial NaN
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns).fillna(0)
    else: # Assuming pandas Series
        returns = returns.fillna(0)
        
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max() # Use expanding max
    drawdown_series = (cumulative_returns - running_max) / running_max
    # Handle potential division by zero if running_max is 0 (e.g., first element)
    drawdown_series = drawdown_series.fillna(0) 
    return drawdown_series


def max_drawdown(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the maximum drawdown.

    Args:
        returns: The return series

    Returns:
        The maximum drawdown
    """
    return drawdown(returns).min()


def drawdown_duration(returns: Union[np.ndarray, pd.Series]) -> Tuple[int, int]:
    """
    Calculate the duration of the maximum drawdown.

    Args:
        returns: The return series

    Returns:
        A tuple of (start_index, end_index) for the maximum drawdown period
    """
    dd = drawdown(returns)
    if dd.min() >= 0: # Handle case with no drawdown
        return 0, 0
        
    max_dd_idx = dd.idxmin() # Use idxmin for pandas Series

    # Find the peak (start) before the maximum drawdown index
    cumulative_returns = (1 + returns.fillna(0)).cumprod()
    start_idx = cumulative_returns[:max_dd_idx+1].idxmax()

    return start_idx, max_dd_idx


def recovery_time(returns: Union[np.ndarray, pd.Series]) -> Optional[int]:
    """
    Calculate the time taken to recover from the maximum drawdown.

    Args:
        returns: The return series

    Returns:
        The number of periods taken to recover, or None if never recovered.
    """
     # Convert input to pandas Series if it's a numpy array and fill initial NaN
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns).fillna(0)
    else: # Assuming pandas Series
        returns = returns.fillna(0)
        
    start_idx, max_dd_idx = drawdown_duration(returns)
    if start_idx == 0 and max_dd_idx == 0: # No drawdown
        return 0
        
    cumulative_returns = (1 + returns).cumprod()
    peak_value = cumulative_returns[start_idx]

    # Find the recovery index
    recovery_idx = -1
    for i in range(max_dd_idx + 1, len(cumulative_returns)):
        if cumulative_returns[i] >= peak_value:
            recovery_idx = i
            break

    if recovery_idx != -1:
        return recovery_idx - max_dd_idx
    else:
        return None # Never recovered
