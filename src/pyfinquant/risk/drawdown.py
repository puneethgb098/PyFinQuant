"""
Drawdown calculations for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple

def drawdown(returns: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Calculate the drawdown series.
    
    Args:
        returns: The return series
        
    Returns:
        The drawdown series
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    return (cumulative_returns - running_max) / running_max

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
    max_dd_idx = dd.argmin()
    
    # Find the start of the drawdown
    start_idx = 0
    for i in range(max_dd_idx, 0, -1):
        if dd[i] >= 0:
            start_idx = i + 1
            break
    
    return start_idx, max_dd_idx

def recovery_time(returns: Union[np.ndarray, pd.Series]) -> int:
    """
    Calculate the time taken to recover from the maximum drawdown.
    
    Args:
        returns: The return series
        
    Returns:
        The number of periods taken to recover
    """
    start_idx, max_dd_idx = drawdown_duration(returns)
    cumulative_returns = (1 + returns).cumprod()
    peak_value = cumulative_returns[start_idx]
    
    for i in range(max_dd_idx + 1, len(returns)):
        if cumulative_returns[i] >= peak_value:
            return i - max_dd_idx
    
    return len(returns) - max_dd_idx  # If never recovered 